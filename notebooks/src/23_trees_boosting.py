# %% [markdown]
# # 23 · Trees, forests and gradient boosting from scratch
# A decision tree, a random forest and gradient boosting in NumPy, raced against an MLP on tabular data.
# Chapter: **Classical ML: trees and kNN**.
#
# **You're done when** your boosting matches scikit-learn within ~1% and you can say why trees win here.

# %%
import os
import numpy as np
import matplotlib.pyplot as plt

QUICK = os.environ.get("QUICK") == "1"
rng = np.random.default_rng(0)

# Tabular data with the things trees love: thresholds, interactions, irrelevant and differently-scaled features
n = 2000 if QUICK else 6000
X = np.column_stack(
    [
        rng.normal(size=n),
        rng.uniform(0, 100, n),
        rng.integers(0, 5, n),
        rng.exponential(2, n),
        rng.normal(size=(n, 4)),
    ]
)
logit = (
    2.0 * (X[:, 0] > 0.3)
    + 1.5 * ((X[:, 1] > 60) & (X[:, 2] >= 3))
    - 1.2 * (X[:, 3] > 3)
    + 0.3 * X[:, 4]
)
y = (logit + rng.normal(scale=0.8, size=n) > 1.0).astype(float)
cut = int(0.75 * n)
Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]
print("positives:", y.mean().round(3))

# %% [markdown]
# ## A regression tree (squared-error splits)
# Gradient boosting fits trees to residuals, so we build regression trees; a classifier is a tree on 0/1 targets.


# %%
class Tree:
    def __init__(self, depth=3, min_leaf=10, n_feat=None, rng=None):
        self.depth, self.min_leaf, self.n_feat, self.rng = (
            depth,
            min_leaf,
            n_feat,
            rng or np.random.default_rng(0),
        )

    def fit(self, X, g):
        self.root = self._grow(X, g, self.depth)
        return self

    def _grow(self, X, g, depth):
        if depth == 0 or len(g) < 2 * self.min_leaf:
            return ("leaf", g.mean())
        feats = (
            range(X.shape[1])
            if self.n_feat is None
            else self.rng.choice(X.shape[1], self.n_feat, replace=False)
        )
        best = None
        for f in feats:
            order = np.argsort(X[:, f])
            xs, gs = X[order, f], g[order]
            cs, cs2 = np.cumsum(gs), np.cumsum(gs**2)
            tot, tot2, m = cs[-1], cs2[-1], len(gs)
            k = np.arange(self.min_leaf, m - self.min_leaf)
            k = k[xs[k] != xs[k - 1]]  # only split between distinct values
            if len(k) == 0:
                continue
            left_sse = cs2[k - 1] - cs[k - 1] ** 2 / k
            right_sse = (tot2 - cs2[k - 1]) - (tot - cs[k - 1]) ** 2 / (m - k)
            i = np.argmin(left_sse + right_sse)
            if best is None or left_sse[i] + right_sse[i] < best[0]:
                best = (left_sse[i] + right_sse[i], f, (xs[k[i] - 1] + xs[k[i]]) / 2)
        if best is None:
            return ("leaf", g.mean())
        _, f, t = best
        L = X[:, f] <= t
        return (
            "node",
            f,
            t,
            self._grow(X[L], g[L], depth - 1),
            self._grow(X[~L], g[~L], depth - 1),
        )

    def predict(self, X):
        out = np.empty(len(X))
        for i, x in enumerate(X):
            node = self.root
            while node[0] == "node":
                node = node[3] if x[node[1]] <= node[2] else node[4]
            out[i] = node[1]
        return out


acc = lambda p: ((p > 0.5) == yte).mean()
tree = Tree(depth=6).fit(Xtr, ytr)
print(f"single tree (depth 6): {acc(tree.predict(Xte)):.3f}")

# %% [markdown]
# ## Random forest: average many decorrelated trees


# %%
def forest(n_trees=30 if not QUICK else 10, depth=8):
    preds = []
    for t in range(n_trees):
        r = np.random.default_rng(t)
        idx = r.integers(0, len(Xtr), len(Xtr))  # bootstrap sample
        preds.append(Tree(depth, n_feat=3, rng=r).fit(Xtr[idx], ytr[idx]).predict(Xte))
    return np.mean(preds, 0)


print(f"random forest: {acc(forest()):.3f}")

# %% [markdown]
# ## Gradient boosting: each tree fits the gradient of the log-loss
# With $F$ the current logit and $p = \sigma(F)$, the negative gradient of the log-loss is $y - p$.


# %%
def boost(rounds=150 if not QUICK else 40, lr=0.1, depth=3):
    F_tr = np.full(len(Xtr), np.log(ytr.mean() / (1 - ytr.mean())))
    F_te = np.full(len(Xte), F_tr[0])
    curve = []
    for r in range(rounds):
        resid = ytr - 1 / (1 + np.exp(-F_tr))
        t = Tree(depth).fit(Xtr, resid)
        F_tr += lr * 4 * t.predict(Xtr)
        F_te += lr * 4 * t.predict(Xte)  # ×4: crude Newton step for the logistic loss
        curve.append(acc(1 / (1 + np.exp(-F_te))))
    return curve


curve = boost()
print(f"gradient boosting: {curve[-1]:.3f}")
plt.plot(curve)
plt.xlabel("boosting round")
plt.ylabel("test accuracy")
plt.show()

# %% [markdown]
# ## Against scikit-learn and an MLP

# %%
try:
    from sklearn.ensemble import HistGradientBoostingClassifier

    sk = HistGradientBoostingClassifier(max_iter=200).fit(Xtr, ytr)
    print(f"sklearn HistGradientBoosting: {sk.score(Xte, yte):.3f}")
except ImportError:
    print("pip install scikit-learn to compare")

import torch
import torch.nn as nn

torch.manual_seed(0)
mu, sd = Xtr.mean(0), Xtr.std(0)
xt, xv = torch.tensor((Xtr - mu) / sd, dtype=torch.float32), torch.tensor(
    (Xte - mu) / sd, dtype=torch.float32
)
mlp = nn.Sequential(
    nn.Linear(X.shape[1], 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)
)
opt = torch.optim.Adam(mlp.parameters(), 1e-3)

for ep in range(300 if not QUICK else 60):
    loss = nn.functional.binary_cross_entropy_with_logits(
        mlp(xt).squeeze(-1), torch.tensor(ytr, dtype=torch.float32)
    )
    opt.zero_grad()
    loss.backward()
    opt.step()

print(f"MLP: {acc(torch.sigmoid(mlp(xv)).squeeze(-1).detach().numpy()):.3f}")

# %% [markdown]
# **Why trees win here**: the signal lives in thresholds and interactions of a few features, features have
# wildly different scales, and some are irrelevant. Trees are invariant to monotone rescaling and pick
# thresholds directly; the MLP has to approximate step functions with smooth units. On images and text it's
# the opposite. Grinsztajn et al. (2022) document this across 45 tabular datasets.
