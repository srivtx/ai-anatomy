# %% [markdown]
# # 15 · State-space models: from S4-style recurrence to Mamba's selective scan
# A linear time-invariant (LTI) recurrence vs an input-dependent (selective) one, on the selective-copying
# task. Plus a parallel scan that matches the sequential loop exactly. Chapter: **State-space models and Mamba**.
#
# **You're done when** the LTI model fails selective copying and the selective scan solves it.

# %%
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

QUICK = os.environ.get("QUICK") == "1"
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# %% [markdown]
# ## Selective copying
# A sequence of length L contains K data tokens at random positions among noise tokens; at the end the model
# must output the K data tokens in order. A fixed (LTI) filter can't know which positions to keep: it has to
# decide based on *content*.

# %%
VOCAB, NOISE, K, L = 8, 0, 4, 32 if QUICK else 64


def task(bs):
    x = torch.zeros(bs, L + K, dtype=torch.long)  # 0 = noise / blank
    tgt = torch.randint(1, VOCAB, (bs, K))
    for b in range(bs):
        pos = torch.randperm(L)[:K].sort().values
        x[b, pos] = tgt[b]
    x[:, L:] = VOCAB  # "now recall" markers
    return x.to(dev), tgt.to(dev)


# %% [markdown]
# ## The two layers
# Discretised SSM: $h_t = \bar A\, h_{t-1} + \bar B\, x_t$, $y_t = C h_t$. In the LTI version $\bar A, \bar B, C$
# are fixed; in the selective version $\Delta_t$, $B_t$, $C_t$ are computed from the input, so the layer can
# choose to ignore a token ($\Delta \to 0$: keep the state) or overwrite the state ($\Delta$ large).


# %%
class SSM(nn.Module):
    def __init__(self, d, n=16, selective=True):
        super().__init__()
        self.sel, self.n = selective, n
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, n + 1).float()).repeat(d, 1)
        )  # S4D-real init
        self.D = nn.Parameter(torch.ones(d))
        if selective:
            self.proj = nn.Linear(d, 2 * n + d)  # input-dependent B, C, Δ
        else:
            self.B = nn.Parameter(torch.randn(d, n) / n**0.5)
            self.C = nn.Parameter(torch.randn(d, n) / n**0.5)
            self.dt = nn.Parameter(torch.full((d,), -3.0))

    def forward(self, x):  # x: (B, T, d)
        Bsz, T, d = x.shape
        A = -self.A_log.exp()  # (d, n), negative = decaying
        if self.sel:
            Bt, Ct, dt = self.proj(x).split([self.n, self.n, d], -1)
            dt = F.softplus(dt)
        else:
            Bt = self.B.expand(Bsz, T, d, self.n)
            Ct = self.C.expand(Bsz, T, d, self.n)
            dt = F.softplus(self.dt).expand(Bsz, T, d)
        h = torch.zeros(Bsz, d, self.n, device=x.device)
        ys = []
        for t in range(T):  # sequential scan (clear, not fast)
            Abar = torch.exp(dt[:, t, :, None] * A)
            Bx = (
                dt[:, t, :, None] * (Bt[:, t, None, :] if self.sel else Bt[:, t]) * x[:, t, :, None]
            )
            h = Abar * h + Bx
            ys.append((h * (Ct[:, t, None, :] if self.sel else Ct[:, t])).sum(-1))
        return torch.stack(ys, 1) + self.D * x


class Model(nn.Module):
    def __init__(self, selective, d=64, layers=2):
        super().__init__()
        self.emb = nn.Embedding(VOCAB + 1, d)
        self.layers = nn.ModuleList(SSM(d, selective=selective) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(layers))
        self.out = nn.Linear(d, VOCAB)

    def forward(self, x):
        h = self.emb(x)
        for n, l in zip(self.norms, self.layers):
            h = h + l(n(h))
        return self.out(h[:, -K:])


def train(selective, steps=300 if QUICK else 2000):
    torch.manual_seed(0)
    m = Model(selective).to(dev)
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    for s in range(steps):
        x, y = task(64)
        loss = F.cross_entropy(m(x).flatten(0, 1), y.flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
    x, y = task(256)
    with torch.no_grad():
        return (m(x).argmax(-1) == y).float().mean().item()


for sel in [False, True]:
    print(
        f"{'selective (Mamba-style)' if sel else 'LTI (S4-style)':24s}: token accuracy {train(sel):.0%}  (chance {1 / (VOCAB - 1):.0%})"
    )

# %% [markdown]
# ## A parallel scan gives the same answer
# The recurrence $h_t = a_t h_{t-1} + b_t$ is associative: combine $(a_1, b_1)$ then $(a_2, b_2)$ into
# $(a_1 a_2,\ a_2 b_1 + b_2)$. That allows an $O(\log T)$-depth parallel scan on a GPU.


# %%
def seq_scan(a, b):
    h, out = torch.zeros_like(b[0]), []
    for t in range(len(a)):
        h = a[t] * h + b[t]
        out.append(h)
    return torch.stack(out)


def parallel_scan(a, b):  # Hillis–Steele inclusive scan
    a, b = a.clone(), b.clone()
    step = 1
    while step < len(a):
        a_prev, b_prev = torch.ones_like(a), torch.zeros_like(b)
        a_prev[step:], b_prev[step:] = a[:-step], b[:-step]
        a, b = a * a_prev, a * b_prev + b
        step *= 2
    return b


a, b = torch.rand(100, 16) * 0.9 + 0.1, torch.randn(100, 16)
print(
    "parallel scan == sequential scan:",
    torch.allclose(seq_scan(a, b), parallel_scan(a, b), atol=1e-5),
)
