# %% [markdown]
# # 21 · Speculative decoding
# Draft k tokens with a small model, verify them with the big model in one pass, and use rejection sampling so
# the output distribution is exactly the big model's. Chapter: **Inference**.
#
# **You're done when** you've measured acceptance rate and speedup, and the token histogram matches plain
# sampling from the big model.

# %%
import os, numpy as np, matplotlib.pyplot as plt
QUICK = os.environ.get("QUICK") == "1"
rng = np.random.default_rng(0)
V = 12

# Two toy "language models": next-token distributions that depend on the previous token.
# The draft is a blurred version of the target, so it's often but not always right.
target_T = rng.dirichlet(np.ones(V) * 0.3, size=V)
draft_T = 0.7 * target_T + 0.3 * rng.dirichlet(np.ones(V), size=V)
target = lambda prev: target_T[prev]; draft = lambda prev: draft_T[prev]

def plain(n, start=0):
    out, prev = [], start
    for _ in range(n): prev = rng.choice(V, p=target(prev)); out.append(prev)
    return out, n                                         # n target forward passes

def speculative(n, k=4, start=0):
    out, prev, passes, proposed, accepted = [], start, 0, 0, 0
    while len(out) < n:
        # 1. draft k tokens autoregressively with the cheap model
        drafts, qs, p_ = [], [], prev
        for _ in range(k): q = draft(p_); t = rng.choice(V, p=q); drafts.append(t); qs.append(q); p_ = t
        # 2. one pass of the big model scores all k positions (+1 for the bonus token)
        passes += 1; ctx = [prev] + drafts
        for i, t in enumerate(drafts):
            p = target(ctx[i]); proposed += 1
            if rng.random() < min(1, p[t] / qs[i][t]):          # accept with prob min(1, p/q)
                out.append(t); accepted += 1
            else:                                                 # reject: sample from the residual (p − q)+
                r = np.maximum(p - qs[i], 0); out.append(rng.choice(V, p=r / r.sum())); break
        else:
            out.append(rng.choice(V, p=target(drafts[-1])))       # all accepted: a free extra token
        prev = out[-1]
    return out[:n], passes, accepted / proposed

N = 20_000 if QUICK else 200_000
a, pa = plain(N)
for k in [1, 2, 4, 8]:
    b, pb, acc = speculative(N, k)
    print(f"k={k}: acceptance {acc:.0%}, target passes per token {pb / N:.2f} (plain: 1.00), speedup in passes {N / pb:.2f}×")

# %% [markdown]
# ## The distribution is unchanged
# Compare bigram statistics of plain and speculative samples.

# %%
def bigrams(seq):
    M = np.zeros((V, V))
    for x, y in zip(seq, seq[1:]): M[x, y] += 1
    return M / M.sum(1, keepdims=True)
b, _, _ = speculative(N, 4)
diff = np.abs(bigrams(a) - bigrams(b)).max()
print(f"max difference in next-token probabilities: {diff:.4f} (sampling noise at this N)")
plt.figure(figsize=(10, 3)); plt.bar(np.arange(V) - 0.2, np.bincount(a, minlength=V) / N, 0.4, label="plain"); plt.bar(np.arange(V) + 0.2, np.bincount(b, minlength=V) / N, 0.4, label="speculative")
plt.xlabel("token"); plt.ylabel("frequency"); plt.legend(); plt.show()
assert diff < 0.05
