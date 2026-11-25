# %% [markdown]
# # 47 · Can a model teach itself? Expert iteration on addition
# A small network learns 4-digit addition from easy examples only (at most one carry). Then it teaches itself:
# it samples answers to unlabelled problems, a checker keeps the right ones, and it trains on its own successes.
# We measure how far it climbs up the ladder of harder problems (more carries), and test three claims from the
# chapter: more tries per problem helps, a leaky checker hurts a lot, and problems at the model's edge beat a
# random pool. Chapter: **How models teach themselves**.
#
# **Hardware:** CPU, about 2 minutes.
#
# **You're done when** you have accuracy-by-carries curves over rounds for K = 1, K = 16, a leaky checker and an
# edge-of-ability problem pool, plus the supervised upper bound.

# %%
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

QUICK = os.environ.get("QUICK") == "1"
torch.manual_seed(0)
rng = np.random.default_rng(0)
D = 4  # digits per number
ROUNDS = 3 if QUICK else 12
POOL = 1500 if QUICK else 4000


def carries(a, b):
    c = n = 0
    for _ in range(D):
        s = a % 10 + b % 10 + c
        c = s // 10
        n += c
        a //= 10
        b //= 10
    return n


def make(n, maxc=None, exact=None, allowed=None):
    """n random problems, optionally filtered by number of carries."""
    A, B = [], []
    while len(A) < n:
        a, b = rng.integers(0, 10**D, 2)
        k = carries(a, b)
        if (
            (maxc is None or k <= maxc)
            and (exact is None or k == exact)
            and (allowed is None or k in allowed)
        ):
            A.append(a)
            B.append(b)
    return np.array(A), np.array(B)


digits = lambda x, n: [(x // 10**i) % 10 for i in range(n)]
enc = (
    lambda A, B: F.one_hot(
        torch.tensor(np.array([digits(a, D) + digits(b, D) for a, b in zip(A, B)])), 10
    )
    .float()
    .flatten(1)
)
target = lambda A, B: torch.tensor(np.array([digits(a + b, D + 1) for a, b in zip(A, B)]))

# %% [markdown]
# ## 1. A base model that only saw easy problems
# An MLP reads both numbers and outputs a distribution over each of the 5 answer digits. Sampling an answer
# means sampling each digit. Difficulty is the number of carries: 0 to 4.


# %%
class Adder(nn.Module):
    def __init__(self, h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D * 2 * 10, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, (D + 1) * 10),
        )

    def forward(self, x):
        return self.net(x).view(-1, D + 1, 10)


def fit(m, X, Y, epochs=30, lr=2e-3):
    opt = torch.optim.Adam(m.parameters(), lr)
    for _ in range(epochs):
        p = torch.randperm(len(X))
        for i in range(0, len(X), 128):
            j = p[i : i + 128]
            loss = F.cross_entropy(m(X[j]).flatten(0, 1), Y[j].flatten())
            opt.zero_grad()
            loss.backward()
            opt.step()
    return m


TEST = {k: make(400, exact=k) for k in range(D + 1)}


def by_carries(m):
    with torch.no_grad():
        return np.array(
            [
                (m(enc(*TEST[k])).argmax(-1) == target(*TEST[k])).all(1).float().mean().item()
                for k in range(D + 1)
            ]
        )


A0, B0 = make(3000, maxc=1)
base = fit(Adder(), enc(A0, B0), target(A0, B0), epochs=20 if QUICK else 80)
print("base model, exact accuracy by number of carries:", by_carries(base).round(2))

# %% [markdown]
# ## 2. The self-training loop
# Each round: sample K answers per unlabelled problem, keep one the checker accepts, fine-tune on the kept answers
# plus a little of the original easy data (an anchor). `leak` is the chance the checker accepts a wrong answer.
# `edge=True` builds each round's pool from the carry levels where the model is between 5% and 80% accurate.


# %%
def expert_iteration(m, K=1, leak=0.0, edge=False, rounds=ROUNDS):
    hist = [by_carries(m)]
    for r in range(rounds):
        if edge:
            acc = hist[-1]
            allowed = {k for k in range(D + 1) if 0.05 <= acc[k] <= 0.8} or {
                int(np.argmax(acc < 0.8))
            }
            A, B = make(POOL, allowed=allowed)
        else:
            A, B = make(POOL)
        X, Y = enc(A, B), target(A, B)
        with torch.no_grad():
            probs = m(X).softmax(-1)
        solved = torch.zeros(len(X), dtype=torch.bool)
        wrong = torch.zeros(len(X), dtype=torch.bool)
        chosen = torch.zeros_like(Y)
        for _ in range(K):
            s = torch.distributions.Categorical(probs).sample()
            ok = (s == Y).all(1)
            new = ok & ~solved & ~wrong
            chosen[new] = s[new]
            solved |= new
            if leak:
                slip = ~ok & ~solved & ~wrong & (torch.rand(len(X)) < leak)
                chosen[slip] = s[slip]
                wrong |= slip
        keep = solved | wrong
        Ae, Be = make(800, maxc=1)
        fit(
            m,
            torch.cat([X[keep], enc(Ae, Be)]),
            torch.cat([chosen[keep], target(Ae, Be)]),
            epochs=8,
            lr=1e-3,
        )
        hist.append(by_carries(m))
        print(
            f"round {r + 1:2d}: kept {keep.float().mean():.0%} (wrong {wrong.float().mean():.0%})  accuracy by carries {hist[-1].round(2)}"
        )
    return np.array(hist)


# %% [markdown]
# **My predictions, written before running** (~65% confident each):
# 1. K = 16 reaches clearly higher accuracy on 2- and 3-carry problems than K = 1 after the same rounds.
# 2. A checker that lets ~10% of wrong samples through ends *below* K = 1 with a perfect checker on 2+ carries.
# 3. An edge-of-ability pool beats a random pool at equal K, because fewer kept examples are trivially easy.
# 4. Nothing reaches the supervised upper bound on 4-carry problems: the loop can't learn what it never samples.

# %%
runs = {}

for name, kw in [
    ("K = 1", dict(K=1)),
    ("K = 16", dict(K=16)),
    ("K = 16, leaky checker (10%)", dict(K=16, leak=0.1)),
    ("K = 16, problems at the edge", dict(K=16, edge=True)),
]:
    print("\n" + name)
    torch.manual_seed(1)
    runs[name] = expert_iteration(copy.deepcopy(base), **kw)

# %% [markdown]
# ## 3. The upper bound: real labels for every problem
# Same number of fine-tuning steps, but with the true answers for problems of every difficulty.

# %%
sup = copy.deepcopy(base)

for r in range(ROUNDS):
    A, B = make(POOL)
    Ae, Be = make(800, maxc=1)
    fit(
        sup,
        torch.cat([enc(A, B), enc(Ae, Be)]),
        torch.cat([target(A, B), target(Ae, Be)]),
        epochs=8,
        lr=1e-3,
    )

runs["supervised (upper bound)"] = np.array([by_carries(base)] * ROUNDS + [by_carries(sup)])
print("supervised:", by_carries(sup).round(2))

# %%
fig, axs = plt.subplots(1, 3, figsize=(15, 4))

for k, ax in zip([1, 2, 3], axs):
    for name, hist in runs.items():
        if name.startswith("supervised"):
            ax.axhline(hist[-1][k], ls="--", c="gray", label=name)
        else:
            ax.plot(hist[:, k], "o-", ms=3, label=name)
    ax.set_title(f"{k} carr{'y' if k == 1 else 'ies'}")
    ax.set_xlabel("round")
    ax.set_ylim(0, 1)

axs[0].set_ylabel("exact-match accuracy")
axs[0].legend(fontsize=8)
plt.tight_layout()
plt.show()
print({n: h[-1].round(2).tolist() for n, h in runs.items()})

# %% [markdown]
# **What I measured** (full run, CPU, about a minute). Exact-match accuracy after 12 rounds, by number of carries:
#
# | run | 0 | 1 | 2 | 3 | 4 |
# |---|---|---|---|---|---|
# | base model (before self-training) | 0.67 | 0.45 | 0.05 | 0.00 | 0.00 |
# | K = 1 | 0.95 | 0.90 | 0.36 | 0.05 | 0.00 |
# | K = 16 | 0.95 | 0.94 | 0.60 | 0.23 | 0.04 |
# | K = 16, leaky checker (10%) | 0.87 | 0.79 | 0.22 | 0.00 | 0.00 |
# | K = 16, problems at the edge | 0.90 | 0.89 | **0.68** | **0.34** | **0.10** |
# | supervised upper bound | 0.94 | 0.95 | 0.89 | 0.88 | 0.84 |
#
# All four predictions held at this scale:
# 1. **More tries:** K = 16 beat K = 1 by 24 points on 2 carries and 18 on 3.
# 2. **Leaky checker:** letting wrong answers through cost more than K = 16 gained; it ended below K = 1 with a
#    perfect checker, and it even lost accuracy on easy problems the model had already learned.
# 3. **Problems at the edge** beat a random pool on every hard level, with fewer kept examples per round.
# 4. **Ceiling:** even the best self-training run reached 10% on 4 carries against 84% with real labels. The loop
#    climbs one rung at a time, and only as fast as rare successes appear.
#
# (Your base-model numbers may differ a little: they depend on the random easy set.)
#
# **Stretch:** replace the exact checker with majority vote over the K samples (no ground truth at all). On which
# carry levels does it still help, and where does it start reinforcing confident mistakes?
