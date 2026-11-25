# %% [markdown]
# # 63 · Shrinking the KV cache: MHA, GQA, MQA and MLA
# When a model generates, it keeps the keys and values of every past token in memory: the **KV cache**. For
# long chats with many users, this cache, not the weights, is what fills the GPU. Every big architecture change
# in attention since 2019 is mostly about making it smaller:
#
# | Design | What is cached per token per layer | Used by |
# |---|---|---|
# | **MHA**, multi-head | a key and a value for every head: 2·d numbers | GPT-2, GPT-3, Llama 2 7B |
# | **MQA**, multi-query | one key and one value shared by all heads: 2·d/H | PaLM, Falcon |
# | **GQA**, grouped-query | one key/value pair per group of heads: 2·d·G/H | Llama 2 70B, Llama 3, Mistral |
# | **MLA**, multi-head latent | one small latent vector c; keys and values are rebuilt from it | DeepSeek-V2, V3 |
#
# You'll train all four on the same text at the same size, measure loss against cache size, then check the trick
# that makes MLA cheap at inference: the up-projection can be **absorbed** into the query, so keys never need to
# be rebuilt. Chapters: **Fast inference**, **Attention**, **Long context**.
#
# **Hardware:** CPU, about 6 minutes (QUICK under 2). **You're done when** you have the loss-vs-cache table and
# the absorbed-MLA check prints ✅.

# %%
import os
import math
import time
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

QUICK = os.environ.get("QUICK") == "1"
torch.manual_seed(0)
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

if not os.path.exists("shakespeare.txt"):
    urllib.request.urlretrieve(url, "shakespeare.txt")

text = open("shakespeare.txt").read()
chars = sorted(set(text))
data = torch.tensor([chars.index(c) for c in text])
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
V, T, D, H, LAYERS = len(chars), 128, 128, 8, 4
STEPS = 150 if QUICK else 1000


def batch(split, bs=32):
    d = train_data if split == "train" else val_data
    i = torch.randint(len(d) - T - 1, (bs,))
    return torch.stack([d[j : j + T] for j in i]), torch.stack([d[j + 1 : j + T + 1] for j in i])


# %% [markdown]
# ## 1. One attention class, four designs
# `kv_heads` sets how many distinct key/value heads exist (8 = MHA, 2 = GQA, 1 = MQA). Each KV head is shared by
# `H / kv_heads` query heads. MLA instead compresses the token into a latent of size `latent`, caches only that,
# and rebuilds full per-head keys and values with two up-projections.


# %%
class Attention(nn.Module):
    def __init__(self, kv_heads=H, latent=None):
        super().__init__()
        self.kv_heads, self.latent, self.dh = kv_heads, latent, D // H
        self.q = nn.Linear(D, D, bias=False)
        if latent:
            self.down = nn.Linear(D, latent, bias=False)  # token → latent c (this is what's cached)
            self.up_k = nn.Linear(latent, D, bias=False)  # c → keys for all heads
            self.up_v = nn.Linear(latent, D, bias=False)  # c → values for all heads
        else:
            self.kv = nn.Linear(D, 2 * kv_heads * self.dh, bias=False)
        self.o = nn.Linear(D, D, bias=False)

    def cached_floats(self):
        """Numbers stored in the KV cache per token for this layer."""
        return self.latent if self.latent else 2 * self.kv_heads * self.dh

    def forward(self, x):
        B, T_, _ = x.shape
        q = self.q(x).view(B, T_, H, self.dh).transpose(1, 2)
        if self.latent:
            c = self.down(x)
            k = self.up_k(c).view(B, T_, H, self.dh).transpose(1, 2)
            v = self.up_v(c).view(B, T_, H, self.dh).transpose(1, 2)
        else:
            k, v = self.kv(x).view(B, T_, 2, self.kv_heads, self.dh).permute(2, 0, 3, 1, 4)
            k = k.repeat_interleave(H // self.kv_heads, 1)  # share each KV head across its group
            v = v.repeat_interleave(H // self.kv_heads, 1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, T_, D))


class GPT(nn.Module):
    def __init__(self, **att):
        super().__init__()
        self.tok, self.pos = nn.Embedding(V, D), nn.Embedding(T, D)
        self.atts = nn.ModuleList(Attention(**att) for _ in range(LAYERS))
        self.mlps = nn.ModuleList(
            nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
            for _ in range(LAYERS)
        )
        self.n1 = nn.ModuleList(nn.LayerNorm(D) for _ in range(LAYERS))
        self.n2 = nn.ModuleList(nn.LayerNorm(D) for _ in range(LAYERS))
        self.out = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, V))

    def forward(self, x):
        h = self.tok(x) + self.pos.weight[: x.shape[1]]
        for a, m, n1, n2 in zip(self.atts, self.mlps, self.n1, self.n2):
            h = h + a(n1(h))
            h = h + m(n2(h))
        return self.out(h)


# %% [markdown]
# ## 2. Train all four
# **Predict first:** MQA caches 8× less than MHA. How much worse is its loss? And can MLA, caching 32 numbers
# per token, match GQA, which caches 64?


# %%
def train(**att):
    torch.manual_seed(0)
    m = GPT(**att)
    opt = torch.optim.AdamW(m.parameters(), 2e-3, betas=(0.9, 0.95), weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=STEPS, pct_start=0.1)
    for step in range(STEPS):
        x, y = batch("train")
        loss = F.cross_entropy(m(x).flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return m


@torch.no_grad()
def val_loss(m, k=20):
    torch.manual_seed(123)  # the same validation batches for every model
    return (
        sum(
            F.cross_entropy(m(x).flatten(0, 1), y.flatten()).item()
            for x, y in (batch("val") for _ in range(k))
        )
        / k
    )


designs = {
    "MHA (8 KV heads)": dict(kv_heads=8),
    "GQA (2 KV groups)": dict(kv_heads=2),
    "MQA (1 KV head)": dict(kv_heads=1),
    "MLA (latent 32)": dict(latent=32),
}
results = {}
print(f"{'design':>20} {'params':>9} {'cache/token':>12} {'val loss':>9}")

for name, att in designs.items():
    t0 = time.time()
    m = train(**att)
    params = sum(p.numel() for p in m.parameters())
    cache = sum(a.cached_floats() for a in m.atts)
    results[name] = (cache, val_loss(m), m)
    print(f"{name:>20} {params:9,} {cache:12} {results[name][1]:9.3f}   ({time.time() - t0:.0f}s)")

for name, (cache, loss, _) in results.items():
    plt.scatter(cache, loss, s=60)
    plt.annotate(name, (cache, loss), textcoords="offset points", xytext=(6, 4))

plt.xscale("log", base=2)
plt.xlabel("KV cache: numbers stored per token (all layers)")
plt.ylabel("validation loss (nats/char)")
plt.title("What each design pays in loss for its cache")
plt.show()

# %% [markdown]
# At this size the differences in loss are small, while the cache shrinks 4-8×. That is the whole argument for
# GQA, and why nearly every open model since 2023 uses it. Be careful with the MLA row: at this scale it's
# within noise of the others, and the claim that MLA *beats* MHA (made in the DeepSeek-V2 paper) is about large
# models with a decoupled RoPE path that we've left out.
#
# ## 3. The trick that makes MLA cheap: absorb the up-projection
# Naively, MLA has to rebuild every past key from its latent at every step, which is slow. But the score is
#
# $$q^\top k = (W_q x)^\top (W_{uk}\, c) = \big(W_{uk}^\top W_q x\big)^\top c$$
#
# so you can fold $W_{uk}$ into the query once, and score directly against the cached latents. The same holds
# for values and the output projection. Check that the two give identical attention scores.

# %%
mla = results["MLA (latent 32)"][2].atts[0]
x = torch.randn(1, 10, D)

with torch.no_grad():
    q = mla.q(x).view(1, 10, H, mla.dh).transpose(1, 2)  # (1, H, 10, dh)
    c = mla.down(x)  # (1, 10, latent): the cache
    k = mla.up_k(c).view(1, 10, H, mla.dh).transpose(1, 2)
    naive = q @ k.transpose(-1, -2)
    W_uk = mla.up_k.weight.view(H, mla.dh, mla.latent)  # per head: dh × latent
    q_absorbed = torch.einsum("bhtd,hdl->bhtl", q, W_uk)  # query moved into latent space
    absorbed = q_absorbed @ c.transpose(-1, -2)[:, None]

ok = torch.allclose(naive, absorbed, atol=1e-4)
print(
    ("✅" if ok else "❌")
    + f" absorbed scores match the naive ones (max diff {(naive - absorbed).abs().max():.1e})"
)

# %% [markdown]
# ## 4. What this means for real models
# Bytes of KV cache per token = layers × cached numbers per layer × 2 bytes (bf16). The configurations below
# are the published ones.

# %%
real = {
    "Llama 2 7B (MHA: 32 layers, 32 KV heads × 128)": 32 * 2 * 32 * 128,
    "Llama 3 8B (GQA: 32 layers, 8 KV heads × 128)": 32 * 2 * 8 * 128,
    "DeepSeek-V3 (MLA: 61 layers, latent 512 + 64 RoPE)": 61 * (512 + 64),
}

for name, floats in real.items():
    per_token = floats * 2
    print(
        f"{name:52s} {per_token / 1024:6.0f} KB/token   128k-token chat: {per_token * 128_000 / 1e9:5.1f} GB"
    )

# %% [markdown]
# A single 128k-token conversation with Llama 2 7B's attention would need ~67 GB of cache, more than its
# weights. With MLA, DeepSeek-V3, a model with about 100× more parameters, needs ~9 GB. That's why the KV cache, not FLOPs, shaped
# attention design in the last three years.
#
# ## 5. Field notes: what I believe but can't cite
#
# | Belief | Confidence | Test |
# |---|---|---|
# | **GQA is nearly free because keys are low-rank anyway.** Trained MHA models have key projections whose heads overlap heavily; GQA just stops paying for the redundancy. | high | Take the trained MHA model, stack its 8 key matrices and look at the singular values. |
# | **MQA's loss gap grows with context length**, not with model size: one KV head can't keep separate “what to look for” channels for many distant things at once. | medium | Repeat part 2 with T = 32 and T = 256 and compare the MQA gap. |
# | **Converting a trained MHA model to GQA by averaging heads, then fine-tuning ~5% of the original tokens, recovers almost everything** (this is what “uptraining” does in the GQA paper). Averaging works better than picking one head per group. | high | Average each group's KV heads in the MHA model and fine-tune 100 steps. |
# | **The cache is now worth more than the compute.** Serving teams choose architectures by tokens served per GPU, which is set by cache size × batch; a 1% loss hit for 4× less cache is accepted every time. | high | Compute the users per 80 GB for each row of part 4 at 32k tokens each. |
# | **Quantizing the KV cache to 8 bits is free; to 4 bits it costs more than it looks**, and almost all of the damage is in the keys, not the values, because a few key channels carry outliers. | medium | Quantize k and v separately to 4 bits in `Attention.forward` and compare val loss. |
#
# **Stretch:** add the decoupled RoPE path to MLA (a small shared key with rotary position, concatenated to the
# latent key), and check that the absorption trick still works for the non-RoPE part.
