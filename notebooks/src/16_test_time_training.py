# %% [markdown]
# # 16 · Memory as learning: linear attention, the delta rule and a TTT-MLP
# Three fixed-size memories that read a stream of key→value pairs, then answer queries. Measure recall as the
# number of stored pairs grows. Chapter: **Test-time training**.
#
# **You're done when** you have recall accuracy vs number of stored pairs for each memory.

# %%
import os, torch, torch.nn.functional as F, matplotlib.pyplot as plt
QUICK = os.environ.get("QUICK") == "1"
torch.manual_seed(0)
d = 32

def keys(n):  return F.normalize(torch.randn(n, d), dim=-1)
def vals(n):  return F.normalize(torch.randn(n, d), dim=-1)

def linear_attn(K, Vv):        # S = Σ v kᵀ : every write adds, nothing is ever corrected
    return lambda q: q @ (Vv.T @ K).T

def delta_rule(K, Vv, beta=1.0):
    S = torch.zeros(d, d)
    for k, v in zip(K, Vv): S = S + beta * torch.outer(v - S @ k, k)    # write only the error: overwrite, don't pile up
    return lambda q: q @ S.T

def ttt_mlp(K, Vv, lr=0.5, epochs=3):
    """The memory is a small MLP trained by gradient descent on the (k, v) pairs as they arrive."""
    W1 = torch.zeros(d, 4 * d); W2 = torch.zeros(4 * d, d); W1.normal_(0, 0.1); W1.requires_grad_(True); W2.requires_grad_(True)
    for _ in range(epochs):
        for k, v in zip(K, Vv):
            loss = ((F.gelu(k @ W1) @ W2 + k - v) ** 2).sum()                # residual MLP
            g1, g2 = torch.autograd.grad(loss, [W1, W2])
            with torch.no_grad(): W1 -= lr * g1; W2 -= lr * g2
    return lambda q: (F.gelu(q @ W1) @ W2 + q).detach()

def recall(mem_fn, n, trials=3 if QUICK else 10):
    hits = 0
    for _ in range(trials):
        K, Vv = keys(n), vals(n); read = mem_fn(K, Vv)(K)
        hits += ((read @ Vv.T).argmax(1) == torch.arange(n)).float().mean().item()
    return hits / trials

ns = [8, 16, 32, 64, 96, 128] if not QUICK else [16, 64, 128]
for name, fn in [("linear attention", linear_attn), ("delta rule", delta_rule), ("TTT-MLP", ttt_mlp)]:
    r = [recall(fn, n) for n in ns]; plt.plot(ns, [x * 100 for x in r], "o-", label=name)
    print(f"{name:17s}: " + "  ".join(f"{n} pairs {x:.0%}" for n, x in zip(ns, r)))
plt.axvline(d, ls="--", c="gray"); plt.xlabel("pairs stored"); plt.ylabel("recall accuracy (%)"); plt.legend(); plt.title(f"memory state of size {d}×{d}"); plt.show()

# %% [markdown]
# ## When values change: overwrite vs pile up
# Now 16 keys are written 64 times in total, each time with a fresh value, and we ask for the *latest* value
# of each key. Linear attention adds every old value into the state; the delta rule replaces it.

# %%
def updates(mem_fn, m=16, writes=64, trials=3 if QUICK else 10):
    hits = 0
    for _ in range(trials):
        K = keys(m); idx = torch.randint(0, m, (writes,)); Vw = vals(writes)
        read = mem_fn(K[idx], Vw)(K)
        latest = {int(k): i for i, k in enumerate(idx)}                     # index of the last write per key
        ok = [((read[k] @ Vw.T).argmax().item() == latest[k]) for k in latest]
        hits += sum(ok) / len(ok)
    return hits / trials
for name, fn in [("linear attention", linear_attn), ("delta rule", delta_rule), ("TTT-MLP", ttt_mlp)]:
    print(f"{name:17s}: latest value recalled {updates(fn):.0%}")

# %% [markdown]
# **What to expect.** With distinct random keys, linear attention degrades gracefully as pairs pile up, and
# the delta rule is strong while pairs stay well below d and then falls faster (with β = 1 it overwrites
# whatever overlapping keys stored). When values for the same key change,
# the delta rule recalls the latest value far more often than linear attention, which returns a blend of
# every value ever written. Overwriting is what “memory as learning” buys you.
