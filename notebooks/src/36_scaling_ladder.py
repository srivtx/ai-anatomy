# %% [markdown]
# # 36 · A scaling ladder
# Do what labs do before a big run: train both recipes at five small sizes, each with tokens proportional to
# parameters, fit a power law in compute for each, predict a 4× bigger run, then train it and check.
# Chapters: **How labs run experiments**, **Scaling laws**.
#
# **You're done when** you have two fitted laws with bootstrap intervals and a checked prediction.

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
# ## Two recipes
# Recipe A uses a moderate learning rate. Recipe B uses a 3× higher one, which tends to help small models
# (they train faster) and hurt big ones (the optimal LR falls with width). Does the ladder see the crossover?

# %%
RECIPES = {"A: lr 1e-3": dict(lr=1e-3), "B: lr 3e-3": dict(lr=3e-3)}
rungs = (
    [(32, 2), (48, 2), (64, 2), (80, 3), (96, 3)]
    if QUICK
    else [(64, 2), (96, 3), (128, 4), (160, 4), (192, 5)]
)
target = (128, 4) if QUICK else (288, 6)
bs, T = 32, 128


def steps_for(d, L):  # tokens ∝ parameters, roughly compute-optimal at this scale
    n = 12 * L * d * d
    return max(40, int((20 if not QUICK else 0.5) * n / (bs * T)))


ladder = {k: [] for k in RECIPES}

for d, L in rungs:
    for name, cfg in RECIPES.items():
        st = steps_for(d, L)
        val, m = train_gpt(st, d=d, layers=L, bs=bs, T=T, **cfg)
        C = 6 * m.n_params() * st * bs * T
        ladder[name].append((C, val))
        print(f"{name}  d={d} L={L}: C={C:.2e}  val {val:.3f}")

# %%
from scipy.optimize import curve_fit

law = lambda C, a, alpha, e: e + a * (C / 1e12) ** -alpha


def fit(pts, n_boot=200, rng=np.random.default_rng(0)):
    C, Lv = np.array(pts).T
    p, _ = curve_fit(law, C, Lv, p0=(1.0, 0.2, 1.0), maxfev=50000, bounds=([0, 0, 0], [50, 2, 5]))
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(C), len(C))
        try:
            boots.append(
                curve_fit(law, C[i], Lv[i], p0=p, maxfev=20000, bounds=([0, 0, 0], [50, 2, 5]))[0]
            )
        except Exception:
            pass
    return p, boots


td, tL = target
t_steps = steps_for(td, tL)
C_t = 6 * (12 * tL * td * td) * t_steps * bs * T
xs = np.logspace(np.log10(min(c for c, _ in ladder["A: lr 1e-3"]) / 2), np.log10(C_t * 2), 100)

for name, color in zip(RECIPES, ["tab:blue", "tab:orange"]):
    p, boots = fit(ladder[name])
    preds = [law(C_t, *b) for b in boots]
    lo, hi = np.percentile(preds, [5, 95]) if preds else (np.nan, np.nan)
    actual, _ = train_gpt(t_steps, d=td, layers=tL, bs=bs, T=T, **RECIPES[name])
    print(f"{name}: predicted {law(C_t, *p):.3f} [{lo:.3f}, {hi:.3f}], actual {actual:.3f}")
    C, Lv = np.array(ladder[name]).T
    plt.loglog(C, Lv, "o", color=color, label=name)
    plt.loglog(xs, law(xs, *p), "--", color=color)
    plt.loglog([C_t], [actual], "*", ms=14, color=color)

plt.xlabel("training compute (FLOPs)")
plt.ylabel("validation loss")
plt.legend()
plt.title("Stars: the target run, trained after predicting it")
plt.show()

# %% [markdown]
# **What to expect.** With low noise the predictions land within 1–2% of the real loss. Recipe B often wins
# on the smallest rungs and loses ground as size grows: if its fitted curve crosses A's before the target,
# the ladder has told you not to scale it, even though it "won" on most of the runs you trained.
