# %% [markdown]
# # 32 · Ablation lab: what matters in a small GPT
# Train the same small GPT with and without each modern ingredient, over several seeds, and rank what
# actually moved the validation loss. Chapters: **The LLM recipe: what survived**, **Papers vs reality**,
# **Read a training run**.
#
# **You're done when** you have a ranked chart of loss changes with seed error bars, and a note on which
# effects are within seed noise. Before running, write down your predicted ranking.

# %%
import os
import math
import time
import urllib.request
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

QUICK = os.environ.get("QUICK") == "1"
dev = "cuda" if torch.cuda.is_available() else "cpu"
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

if not os.path.exists("shakespeare.txt"):
    urllib.request.urlretrieve(url, "shakespeare.txt")

text = open("shakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
V = len(chars)
data = torch.tensor([stoi[c] for c in text])
n_split = int(0.9 * len(data))
train_data, val_data = data[:n_split], data[n_split:]


def get_batch(d, bs, T, gen=None):
    i = torch.randint(len(d) - T - 1, (bs,), generator=gen)
    return torch.stack([d[j : j + T] for j in i]).to(dev), torch.stack(
        [d[j + 1 : j + T + 1] for j in i]
    ).to(dev)


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


def rope(x):  # rotary position embedding on (B, h, T, dh)
    T, dh = x.shape[-2], x.shape[-1]
    f = 1.0 / 10000 ** (torch.arange(0, dh, 2, device=x.device) / dh)
    ang = torch.arange(T, device=x.device)[:, None] * f[None]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class Block(nn.Module):
    def __init__(self, d, h, c):
        super().__init__()
        self.h, self.c = h, c
        N = RMSNorm if c.get("rmsnorm", True) else nn.LayerNorm
        self.n1, self.n2 = N(d), N(d)
        self.qkv, self.o = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)
        hid = int(8 * d / 3) if c.get("swiglu", True) else 4 * d
        self.up = nn.Linear(d, 2 * hid if c.get("swiglu", True) else hid, bias=False)
        self.down = nn.Linear(hid, d, bias=False)
        self.ffn = None  # set to a module (e.g. a mixture of experts) to replace the MLP

    def mlp(self, x):
        if self.c.get("swiglu", True):
            a, b = self.up(x).chunk(2, -1)
            return self.down(F.silu(a) * b)
        return self.down(F.gelu(self.up(x)))

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        if self.c.get("rope", True):
            q, k = rope(q), rope(k)
        x = x + self.o(
            F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, D)
        )
        return x + (self.ffn if self.ffn is not None else self.mlp)(self.n2(x))


class GPT(nn.Module):
    def __init__(self, d=128, layers=4, heads=4, T=128, **c):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(V, d)
        self.pos = None if c.get("rope", True) else nn.Embedding(T, d)
        self.blocks = nn.ModuleList(Block(d, heads, c) for _ in range(layers))
        self.norm = RMSNorm(d) if c.get("rmsnorm", True) else nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=False)

    def forward(self, idx):
        x = self.tok(idx)
        if self.pos is not None:
            x = x + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))

    def n_params(self):
        return sum(
            p.numel() for n, p in self.named_parameters() if "tok" not in n and "pos" not in n
        )


def train_gpt(
    steps,
    d=128,
    layers=4,
    heads=4,
    T=128,
    bs=32,
    lr=1e-3,
    seed=0,
    warmup=True,
    cosine=True,
    wd=0.1,
    data_src=None,
    **c,
):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    m = GPT(d, layers, heads, T, **c).to(dev)
    decay = [p for p in m.parameters() if p.dim() >= 2]
    no_decay = [p for p in m.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0}],
        lr=lr,
        betas=(0.9, 0.95),
    )
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (min(1, (s + 1) / warm) if warmup else 1)
        * ((0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps))) if cosine else 1),
    )
    src = train_data if data_src is None else data_src
    for s in range(steps):
        x, y = get_batch(src, bs, T, gen)
        loss = F.cross_entropy(m(x).flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sched.step()
    m.eval()
    vg = torch.Generator().manual_seed(1234)
    with torch.no_grad():
        val = np.mean(
            [
                F.cross_entropy(m(x).flatten(0, 1), y.flatten()).item()
                for x, y in (get_batch(val_data, 32, T, vg) for _ in range(10))
            ]
        )
    return val, m


# %% [markdown]
# ## The grid
# The baseline has everything on: RoPE, SwiGLU, RMSNorm, warmup + cosine decay, weight decay, deduplicated data.
# Each ablation removes one thing. "Duplicated data" trains on a stream where a fifth of the text is repeated
# five times, at the same number of steps.

# %%
dup_len = len(train_data) // 5
dup_data = torch.cat([train_data[:dup_len]] * 5)  # same length, 1/5 of the unique text
ABLATIONS = {
    "baseline (all on)": {},
    "no warmup": dict(warmup=False),
    "constant LR (no cosine)": dict(cosine=False),
    "learned positions (no RoPE)": dict(rope=False),
    "GELU MLP (no SwiGLU)": dict(swiglu=False),
    "LayerNorm (no RMSNorm)": dict(rmsnorm=False),
    "no weight decay": dict(wd=0.0),
    "duplicated data": dict(data_src=dup_data),
    "LR 3× higher": dict(lr=3e-3),
}
steps = 60 if QUICK else 2500
seeds = [0] if QUICK else [0, 1, 2]
res = {}

for name, cfg in ABLATIONS.items():
    vals = [
        train_gpt(steps, d=96 if QUICK else 192, layers=2 if QUICK else 4, seed=s, **cfg)[0]
        for s in seeds
    ]
    res[name] = (np.mean(vals), np.std(vals))
    print(f"{name:30s} {np.mean(vals):.4f} ± {np.std(vals):.4f}")

# %%
base, base_sd = res["baseline (all on)"]
rows = sorted(
    ((k, m - base, s) for k, (m, s) in res.items() if k != "baseline (all on)"), key=lambda r: -r[1]
)
plt.figure(figsize=(8, 4))
plt.barh(
    [r[0] for r in rows],
    [r[1] for r in rows],
    xerr=[r[2] for r in rows],
    color=["tab:red" if r[1] > 2 * max(base_sd, r[2]) else "tab:gray" for r in rows],
)
plt.axvline(0, c="k")
plt.xlabel("Δ validation loss vs baseline (positive = removing it hurts)")
plt.title("Grey: within ~2 seed standard deviations")
plt.show()

for k, d, s in rows:
    print(
        f"{k:30s} {d:+.4f}  {'(within seed noise)' if abs(d) < 2 * max(base_sd, s, 1e-9) else ''}"
    )

# %% [markdown]
# **What usually comes out.** The learning rate and its schedule move the loss most. Duplicated data hurts
# visibly. Swapping RoPE, SwiGLU and RMSNorm for their older versions changes little at this scale, often
# within seed noise; their benefits show up mostly at larger scale and longer training, which is why a
# single small ablation is weak evidence either way.
