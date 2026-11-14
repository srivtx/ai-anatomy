# %% [markdown]
# # 31 · Where the GPU memory goes
# Training memory = weights + gradients + optimizer state + activations. Predict each from a formula, then
# measure it. Then the three standard fixes: bf16, activation checkpointing and gradient accumulation, and why
# fp16 needs loss scaling. Chapters: **Precision**, **Distributed training**.
#
# Activations are measured with saved-tensor hooks, so this runs on a CPU too; on a GPU we also check the
# allocator's peak.
#
# **You're done when** measured memory matches your formula within ~10%, and gradient accumulation gives the
# same update as one big batch.

# %%
import os, math, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
QUICK = os.environ.get("QUICK") == "1"
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
d, L, T, B, V = (128, 2, 64, 4, 1000) if QUICK else (512, 6, 256, 8, 8000)

class Block(nn.Module):
    def __init__(self, d):
        super().__init__(); self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d); self.up, self.down = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
    def forward(self, x):
        Bb, Tt, D = x.shape; q, k, v = self.qkv(self.n1(x)).view(Bb, Tt, 3, 8, D // 8).permute(2, 0, 3, 1, 4)
        a = torch.softmax((q @ k.transpose(-1, -2)) / math.sqrt(D // 8) + torch.triu(torch.full((Tt, Tt), -1e9, device=x.device), 1), -1) @ v
        x = x + self.o(a.transpose(1, 2).reshape(Bb, Tt, D))
        return x + self.down(F.gelu(self.up(self.n2(x))))
class GPT(nn.Module):
    def __init__(self, ckpt=False):
        super().__init__(); self.ckpt = ckpt
        self.emb = nn.Embedding(V, d); self.blocks = nn.ModuleList(Block(d) for _ in range(L)); self.head = nn.Linear(d, V)
    def forward(self, x):
        h = self.emb(x)
        for b in self.blocks: h = checkpoint(b, h, use_reentrant=False) if self.ckpt else b(h)
        return self.head(h)

# %% [markdown]
# ## Predict, then measure
# Per parameter in fp32 with Adam: 4 bytes weight + 4 gradient + 8 optimizer state (two moments) = 16 bytes.
# Activations per layer, roughly: B·T·d·(34 bytes·…) for the linear layers plus B·heads·T²·(~10 bytes) for
# attention scores in fp32 (Korthikanti et al., 2022 give the exact accounting).

# %%
class SavedBytes:
    """Counts the bytes autograd saves for the backward pass (= activation memory), de-duplicating shared storage."""
    def __enter__(self):
        self.seen, self.bytes = set(), 0
        def pack(t):
            key = (t.untyped_storage().data_ptr(), t.dtype)
            if key not in self.seen: self.seen.add(key); self.bytes += t.untyped_storage().nbytes()
            return t
        self.ctx = torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t); self.ctx.__enter__(); return self
    def __exit__(self, *a): self.ctx.__exit__(*a)

model = GPT().to(dev); P = sum(p.numel() for p in model.parameters())
x = torch.randint(0, V, (B, T), device=dev)
with SavedBytes() as sb: loss = F.cross_entropy(model(x).flatten(0, 1), x.flatten())
act_measured = sb.bytes
heads = 8
act_formula = L * (B * T * d * 4 * (2 + 3 + 1 + 1 + 2 + 4 + 4 + 1) + B * heads * T * T * 4 * 2) + B * T * V * 4 * 2
print(f"parameters: {P / 1e6:.2f}M → weights {P * 4 / 2**20:.0f} MB, grads {P * 4 / 2**20:.0f} MB, Adam state {P * 8 / 2**20:.0f} MB")
print(f"activations: formula {act_formula / 2**20:.0f} MB, measured {act_measured / 2**20:.0f} MB ({act_measured / act_formula - 1:+.0%})")

if dev == "cuda":
    torch.cuda.reset_peak_memory_stats(); model = GPT().to(dev); opt = torch.optim.Adam(model.parameters())
    base = torch.cuda.memory_allocated()
    loss = F.cross_entropy(model(x).flatten(0, 1), x.flatten()); loss.backward(); opt.step()
    print(f"GPU peak after one step: {(torch.cuda.max_memory_allocated() - base + P * 4) / 2**20:.0f} MB "
          f"vs formula {(16 * P + act_formula) / 2**20:.0f} MB")

# %% [markdown]
# ## Fix 1: activation checkpointing
# Keep only each block's input; recompute the inside during backward. Activations drop to about one tensor per
# layer, at the cost of a second forward pass (~33% more compute).

# %%
ck = GPT(ckpt=True).to(dev); ck.load_state_dict(model.state_dict())
with SavedBytes() as sb2: F.cross_entropy(ck(x).flatten(0, 1), x.flatten())
print(f"activations with checkpointing: {sb2.bytes / 2**20:.0f} MB (was {act_measured / 2**20:.0f} MB)")

# %% [markdown]
# ## Fix 2: gradient accumulation gives the same update as a big batch

# %%
m1, m2 = GPT().to(dev), GPT().to(dev); m2.load_state_dict(m1.state_dict())
big = torch.randint(0, V, (4 * B, T), device=dev)
F.cross_entropy(m1(big).flatten(0, 1), big.flatten()).backward()
for chunk in big.chunk(4):
    (F.cross_entropy(m2(chunk).flatten(0, 1), chunk.flatten()) / 4).backward()        # divide by the number of micro-batches
diff = max((a.grad - b.grad).abs().max().item() for a, b in zip(m1.parameters(), m2.parameters()))
print(f"max gradient difference, big batch vs 4 accumulated micro-batches: {diff:.2e}")

# %% [markdown]
# ## Fix 3: lower precision, and why fp16 needs loss scaling
# fp16 has 5 exponent bits: the smallest normal number is 6e-5 and anything below ~6e-8 becomes zero.
# bf16 has fp32's 8 exponent bits (range to 1e-38) with less precision. Small gradients vanish in fp16 unless you
# multiply the loss by a scale first.

# %%
g = torch.logspace(-10, -2, 9)
print("gradient    fp16        bf16        fp16 after ×65536 loss scale (then unscaled)")
for v in g:
    print(f"{v.item():.0e}   {v.half().item():<10.3g}  {v.bfloat16().item():<10.3g}  {((v * 65536).half().float() / 65536).item():.3g}")
small = torch.full((1000,), 1e-8)
print(f"share of 1e-8 gradients lost: fp16 {(small.half() == 0).float().mean():.0%}, bf16 {(small.bfloat16() == 0).float().mean():.0%}")

# %% [markdown]
# **What to expect.** Weights+gradients+Adam cost 16 bytes per parameter in fp32 (mixed precision with an fp32
# master copy still costs ~16–18). Activations dominate at long sequence lengths and large batches; the T² attention
# term grows fastest. Checkpointing cuts activations several-fold. Accumulated gradients match the big batch to
# floating-point rounding. Gradients of 1e-8 all vanish in fp16 and survive in bf16 or with loss scaling.
