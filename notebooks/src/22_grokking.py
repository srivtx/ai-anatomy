# %% [markdown]
# # 22 · Grokking modular addition
# Learn (a + b) mod p from a fraction of all pairs (p = 97 on a GPU). Training accuracy hits 100% early; test accuracy jumps much
# later, and only with weight decay. Then look at the Fourier structure of the embedding.
# Chapter: **What is intelligence?**
#
# **You're done when** you have the delayed-generalisation curve, with and without weight decay.

# %%
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

QUICK = os.environ.get("QUICK") == "1"
dev = "cuda" if torch.cuda.is_available() else "cpu"
p = 23 if QUICK else 97
pairs = torch.cartesian_prod(torch.arange(p), torch.arange(p)).to(dev)
labels = pairs.sum(1) % p
g = torch.Generator().manual_seed(0)
perm = torch.randperm(len(pairs), generator=g).to(dev)
frac = 0.45 if QUICK else 0.4
tr, te = perm[: int(frac * len(pairs))], perm[int(frac * len(pairs)) :]


class Net(nn.Module):
    """One hidden layer on one-hot inputs: h = (E_a[a] + E_b[b])², logits = W h. The square activation makes
    the Fourier solution easy to express (Gromov, 2023); the browser chapter uses the same model."""

    def __init__(self, h=96 if QUICK else 256):
        super().__init__()
        self.ea = nn.Embedding(p, h)
        self.eb = nn.Embedding(p, h)
        self.out = nn.Linear(h, p, bias=False)
        for e in (self.ea, self.eb):
            nn.init.normal_(e.weight, std=1 / 2**0.5)
        nn.init.normal_(self.out.weight, std=1 / h**0.5)

    def forward(self, ab):
        return self.out((self.ea(ab[:, 0]) + self.eb(ab[:, 1])) ** 2)


def run(wd, steps):
    torch.manual_seed(0)
    m = Net().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=wd, betas=(0.9, 0.98))
    hist = []
    for s in range(steps + 1):
        loss = F.cross_entropy(m(pairs[tr]), labels[tr])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if s % (steps // 50) == 0:
            with torch.no_grad():
                hist.append(
                    (
                        s,
                        (m(pairs[tr]).argmax(1) == labels[tr]).float().mean().item(),
                        (m(pairs[te]).argmax(1) == labels[te]).float().mean().item(),
                    )
                )
    return m, np.array(hist)


steps = 3000 if QUICK else 10000
fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

for a, wd in zip(ax, [1.0, 0.0]):
    m, h = run(wd, steps)
    a.plot(h[:, 0], h[:, 1], label="train")
    a.plot(h[:, 0], h[:, 2], label="test")
    a.set_xscale("symlog")
    a.set_title(f"weight decay {wd}")
    a.set_xlabel("step")
    a.legend()
    print(
        f"weight decay {wd}: train {h[-1, 1]:.0%}, test {h[-1, 2]:.0%}; first step with test > 90%: {next((int(s) for s, _, t in h if t > 0.9), None)}"
    )
    models = globals().setdefault("models", {})
    models[wd] = m

plt.show()

# %% [markdown]
# ## The algorithm it found: waves
# Look at each hidden unit's input weights as a function of the number a = 0 … p−1, and project onto Fourier
# modes $\cos(2\pi k a/p)$, $\sin(2\pi k a/p)$. In the grokked network most units become a clean wave at a
# single frequency: they represent numbers as angles, and addition as adding angles (Nanda et al., 2023;
# Gromov, 2023). Different units use different frequencies. The memorising network shows no such structure.


# %%
def purity(m):
    E = (m.ea.weight + m.eb.weight).detach().cpu()  # (p, hidden)
    a = torch.arange(p).float()
    spec = torch.stack(
        [
            (torch.cos(2 * np.pi * k * a / p) @ E) ** 2
            + (torch.sin(2 * np.pi * k * a / p) @ E) ** 2
            for k in range(1, p // 2 + 1)
        ]
    )
    spec = spec / spec.sum(0, keepdim=True)  # (frequencies, hidden units)
    return spec.max(0).values, spec.argmax(0) + 1, E


for wd, m in models.items():
    pur, _, _ = purity(m)
    print(
        f"weight decay {wd}: mean share of each unit's energy in its top frequency {pur.mean():.2f}"
    )

pur, freq, E = purity(models[1.0])
top = pur.argsort(descending=True)[:8]
fig, ax = plt.subplots(2, 4, figsize=(14, 5))

for a_, j in zip(ax.flat, top):
    a_.plot(E[:, j], "o-", ms=3)
    a_.set_title(f"unit {j.item()}: k = {freq[j].item()}, purity {pur[j]:.2f}", fontsize=9)

plt.suptitle("Grokked units: clean waves over a = 0 … p−1")
plt.tight_layout()
plt.show()
