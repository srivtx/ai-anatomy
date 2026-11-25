# %% [markdown]
# # 29 · Hopfield networks, capacity, and why attention is a Hopfield update
# Store binary patterns as energy minima, measure how many fit, then show that the modern (continuous)
# Hopfield update is exactly softmax attention. Chapter: **Energy-based models and memory**.
#
# **You're done when** you have a capacity curve near 0.14·N and modern Hopfield recall equals attention.

# %%
import os
import numpy as np
import matplotlib.pyplot as plt

QUICK = os.environ.get("QUICK") == "1"
rng = np.random.default_rng(0)
N = 100 if QUICK else 200


def store(P):  # Hebbian rule: W = (1/N) Σ p pᵀ, no self-connections
    W = P.T @ P / P.shape[1]
    np.fill_diagonal(W, 0)
    return W


def recall(W, x, sweeps=10):
    x = x.copy()
    for _ in range(sweeps):
        for i in rng.permutation(len(x)):  # asynchronous updates always lower the energy
            x[i] = 1 if W[i] @ x >= 0 else -1
    return x


energy = lambda W, x: -0.5 * x @ W @ x

# %% [markdown]
# ## Restore corrupted patterns

# %%
P = rng.choice([-1, 1], size=(5, N))
W = store(P)
x = P[0].copy()
flip = rng.choice(N, N // 5, replace=False)
x[flip] *= -1
r = recall(W, x)
print(
    f"overlap before {np.mean(x == P[0]):.0%}, after {np.mean(r == P[0]):.0%};  energy {energy(W, x):.1f} → {energy(W, r):.1f}"
)

# %% [markdown]
# ## Capacity
# With random patterns, recall breaks down around $0.14N$ stored patterns (Amit, Gutfreund & Sompolinsky, 1985).

# %%
loads, ok = np.linspace(0.02, 0.3, 8 if QUICK else 15), []

for a in loads:
    K = max(1, int(a * N))
    P = rng.choice([-1, 1], size=(K, N))
    W = store(P)
    trials = min(K, 5 if QUICK else 10)
    ok.append(np.mean([np.mean(recall(W, P[k], 3) == P[k]) > 0.97 for k in range(trials)]))

plt.plot(loads, ok, "o-")
plt.axvline(0.138, ls="--", c="gray")
plt.xlabel("patterns / neurons")
plt.ylabel("fraction recalled almost perfectly")
plt.show()

# %% [markdown]
# ## Modern Hopfield networks = attention
# Ramsauer et al. (2020): with energy $E(\xi) = -\beta^{-1}\log\sum_k e^{\beta\, x_k^\top \xi} + \tfrac12 \xi^\top\xi$,
# one update step is $\xi \leftarrow X^\top \mathrm{softmax}(\beta X \xi)$: a query attending over stored
# keys/values. Capacity becomes exponential in the dimension.

# %%
import torch

d, K, beta = 64, 1000, 8.0
Xs = torch.nn.functional.normalize(torch.randn(K, d), dim=1)
query = Xs[42] + 0.08 * torch.randn(d)  # noisy version of pattern 42 (noise norm ≈ 0.6)
hop = Xs.T @ torch.softmax(beta * Xs @ query, 0)  # one modern-Hopfield update
attn = torch.nn.functional.scaled_dot_product_attention(
    query[None, None, None], Xs[None, None], Xs[None, None], scale=beta
)[0, 0, 0]
print("same result as attention:", torch.allclose(hop, attn, atol=1e-5))
got = (Xs @ hop).argmax().item()
print("retrieved pattern:", got, f"(stored {K} patterns in {d} dimensions, far above 0.14·d)")
assert got == 42
