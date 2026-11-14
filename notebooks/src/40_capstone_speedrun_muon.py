# %% [markdown]
# # 40 · Capstone: a GPT-2 (124M) speedrun, Muon vs AdamW
# Train the same 124M-parameter GPT twice on FineWeb-Edu: once with AdamW, once with Muon (orthogonalised
# momentum for every hidden weight matrix, AdamW for embeddings, head and norms). Same data, same steps, same
# schedule. Then answer the question the NanoGPT speedrun community answered: how much faster is Muon?
# Chapters: **The hidden laws of training**, **Keeping big runs stable**, **How labs run experiments**.
#
# **Hardware:** Colab A100, ~80 minutes per run at the default size (≈ 400M tokens each). On an L4, halve
# `STEPS`. Offline or with `QUICK`, it falls back to a tiny character model on Shakespeare.
#
# **You're done when** you have both validation-loss curves on one plot and the number of steps Muon needed to
# reach AdamW's final loss.

# %%
# !pip -q install -U datasets tiktoken
import os, math, time, urllib.request, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, matplotlib.pyplot as plt
QUICK = os.environ.get("QUICK") == "1"
dev = "cuda" if torch.cuda.is_available() else "cpu"
amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev == "cuda" else torch.autocast("cpu", enabled=False)
cfg = dict(d=64, layers=2, heads=4, T=64, bs=8, accum=1, steps=40, eval_every=10) if QUICK else \
      dict(d=768, layers=12, heads=6, T=1024, bs=32, accum=8, steps=1500, eval_every=100)
tokens_per_step = cfg["bs"] * cfg["accum"] * cfg["T"]
N_TOKENS = tokens_per_step * cfg["steps"] + 4_000_000

# %% [markdown]
# ## Data: FineWeb-Edu tokenised with GPT-2's BPE (fallback: Shakespeare characters)

# %%
def fineweb(n_tokens):
    from datasets import load_dataset; import tiktoken
    enc = tiktoken.get_encoding("gpt2"); out = np.empty(n_tokens, dtype=np.uint16); k = 0; buf = []
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    for doc in ds:
        buf.append(doc["text"])
        if len(buf) == 2000:
            for ids in enc.encode_ordinary_batch(buf, num_threads=8):
                ids = ids + [enc.eot_token]; m = min(len(ids), n_tokens - k); out[k:k + m] = ids[:m]; k += m
            buf = []; print(f"\r{k / 1e6:.0f}M tokens", end="")
            if k >= n_tokens: break
    return torch.from_numpy(out[:k].astype(np.int64)), 50304            # vocab padded to a multiple of 64

def shakespeare():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    if not os.path.exists("shakespeare.txt"): urllib.request.urlretrieve(url, "shakespeare.txt")
    text = open("shakespeare.txt").read(); chars = sorted(set(text))
    return torch.tensor([chars.index(c) for c in text]), len(chars)

try:
    if QUICK: raise RuntimeError("quick mode")
    data, V = fineweb(N_TOKENS)
except Exception as e:
    print("Using Shakespeare characters:", e); data, V = shakespeare()
val_data, train_data = data[:2_000_000 if len(data) > 10_000_000 else len(data) // 10], data[2_000_000 if len(data) > 10_000_000 else len(data) // 10:]
def batch(src, bs, T, gen):
    i = torch.randint(len(src) - T - 1, (bs,), generator=gen)
    return torch.stack([src[j:j + T] for j in i]).to(dev), torch.stack([src[j + 1:j + T + 1] for j in i]).to(dev)

# %% [markdown]
# ## The model (a modern GPT: RoPE, RMSNorm, ReLU², QK-norm, untied head)

# %%
def rope(x):
    T, dh = x.shape[-2], x.shape[-1]
    f = 1.0 / 10000 ** (torch.arange(0, dh, 2, device=x.device) / dh); ang = torch.arange(T, device=x.device)[:, None] * f[None]
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * ang.cos() - x2 * ang.sin(), x1 * ang.sin() + x2 * ang.cos()], -1).flatten(-2)
norm = lambda x: F.rms_norm(x, (x.shape[-1],))

class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__(); self.h = h
        self.qkv = nn.Linear(d, 3 * d, bias=False); self.o = nn.Linear(d, d, bias=False)
        self.up = nn.Linear(d, 4 * d, bias=False); self.down = nn.Linear(4 * d, d, bias=False)
        nn.init.zeros_(self.o.weight); nn.init.zeros_(self.down.weight)       # residual branches start at zero
    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(norm(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        q, k = rope(norm(q)), rope(norm(k))                                  # QK-norm, then rotary
        x = x + self.o(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, D))
        return x + self.down(F.relu(self.up(norm(x))).square())             # ReLU² MLP

class GPT(nn.Module):
    def __init__(self, V, d, layers, heads, **_):
        super().__init__(); self.emb = nn.Embedding(V, d); self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.head = nn.Linear(d, V, bias=False); nn.init.zeros_(self.head.weight)
    def forward(self, idx):
        x = norm(self.emb(idx))
        for b in self.blocks: x = b(x)
        logits = self.head(norm(x)).float()
        return 30 * torch.tanh(logits / 30)                                  # logit soft-capping

# %% [markdown]
# ## Muon
# For each hidden 2-D weight: momentum, then replace the update by the nearest semi-orthogonal matrix
# (all singular values → ~1) using five Newton–Schulz iterations. Every direction of the layer moves at the
# same rate, so directions with small gradients aren't starved.

# %%
def newton_schulz5(G, steps=5, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.bfloat16 if G.is_cuda else torch.float32); X = X / (X.norm() + eps)
    tall = G.size(0) > G.size(1)
    if tall: X = X.T
    for _ in range(steps):
        A = X @ X.T; X = a * X + (b * A + c * A @ A) @ X
    return X.T if tall else X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95):
        super().__init__(params, dict(lr=lr, momentum=momentum))
    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None: continue
                buf = self.state[p].setdefault("buf", torch.zeros_like(p.grad)); buf.mul_(g["momentum"]).add_(p.grad)
                u = newton_schulz5(p.grad.add(buf, alpha=g["momentum"]))         # Nesterov
                p.add_(u.to(p.dtype), alpha=-g["lr"] * max(1, p.size(0) / p.size(1)) ** 0.5)

def optimizers(m, kind):
    hidden = [p for n, p in m.named_parameters() if p.dim() == 2 and "blocks" in n]
    other = [p for n, p in m.named_parameters() if not (p.dim() == 2 and "blocks" in n)]
    if kind == "adamw":
        return [torch.optim.AdamW([{"params": hidden, "weight_decay": 0.1}, {"params": other, "weight_decay": 0.0}], lr=6e-4 if not QUICK else 3e-3, betas=(0.9, 0.95), fused=dev == "cuda")]
    return [Muon(hidden, lr=0.02 if not QUICK else 0.03), torch.optim.AdamW(other, lr=6e-3 if not QUICK else 3e-3, betas=(0.9, 0.95), weight_decay=0.0)]

# %% [markdown]
# ## One training run
# Warm-up, then constant, then a linear cooldown over the last 30% (warmup–stable–decay).

# %%
@torch.no_grad()
def val_loss(m, n=20 if not QUICK else 4):
    m.eval(); g = torch.Generator().manual_seed(123); tot = 0
    for _ in range(n):
        x, y = batch(val_data, cfg["bs"], cfg["T"], g)
        with amp: tot += F.cross_entropy(m(x).flatten(0, 1), y.flatten()).item()
    m.train(); return tot / n

def run(kind):
    torch.manual_seed(0); g = torch.Generator().manual_seed(0)
    m = GPT(V, **cfg).to(dev); opts = optimizers(m, kind); base = [[pg["lr"] for pg in o.param_groups] for o in opts]
    S = cfg["steps"]; sched = lambda s: min(1, (s + 1) / (S * 0.02 + 1)) * min(1, (S - s) / (0.3 * S))
    curve, t0 = [], time.time()
    for s in range(S + 1):
        if s % cfg["eval_every"] == 0 or s == S:
            curve.append((s, val_loss(m))); print(f"{kind:6s} step {s:5d}  val {curve[-1][1]:.3f}  {time.time() - t0:.0f}s")
        if s == S: break
        for o, b in zip(opts, base):
            for pg, lr in zip(o.param_groups, b): pg["lr"] = lr * sched(s)
        for _ in range(cfg["accum"]):
            x, y = batch(train_data, cfg["bs"], cfg["T"], g)
            with amp: loss = F.cross_entropy(m(x).flatten(0, 1), y.flatten()) / cfg["accum"]
            loss.backward()
        for o in opts: o.step(); o.zero_grad(set_to_none=True)
    return np.array(curve)

print(f"{sum(p.numel() for p in GPT(V, **cfg).parameters()) / 1e6:.0f}M parameters, {tokens_per_step:,} tokens per step")
curves = {k: run(k) for k in ["adamw", "muon"]}

# %%
for k, c in curves.items(): plt.plot(c[:, 0], c[:, 1], label=k)
plt.xlabel("step"); plt.ylabel("validation loss"); plt.legend(); plt.ylim(None, curves["adamw"][2, 1]); plt.show()
target = curves["adamw"][-1, 1]; hit = next((s for s, l in curves["muon"] if l <= target), None)
print(f"AdamW final {target:.3f}; Muon final {curves['muon'][-1, 1]:.3f}; Muon reached AdamW's final loss at step {hit} of {cfg['steps']}")

# %% [markdown]
# ## My predictions (not run here)
# At the default size on FineWeb-Edu (1,500 steps × 262k tokens ≈ 390M tokens per run):
#
# | Quantity | Prediction |
# |---|---|
# | AdamW final validation loss | 3.45–3.65 |
# | Muon final validation loss | 0.05–0.12 lower than AdamW |
# | Muon reaches AdamW's final loss at | 60–75% of the steps |
# | Muon's extra time per step | under 5% (Newton–Schulz is a few small matmuls) |
#
# Careful: a fair comparison needs a learning-rate sweep for **both** optimisers (try ×0.5 and ×2 of the
# defaults). Many published “optimizer X beats Adam” results disappear once the baseline is tuned. That's the
# point of the stretch goal.
#
# **Stretch:** sweep LR for both at 3 values and re-plot with each at its best; then double the steps and see
# whether the gap shrinks (it's one of my bets).
