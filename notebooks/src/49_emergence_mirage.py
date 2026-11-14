# %% [markdown]
# # 49 · Emergence or mirage? Grade the same models three ways
# Train a family of networks of increasing size on multi-digit addition, then grade every model three ways:
# exact match (all digits right), per-digit accuracy, and the probability assigned to the correct answer. Does
# the ability “emerge” suddenly, or does it only look that way under exact match? Then forecast the big model's
# exact-match score from the small models' per-digit numbers.
# Chapter: **Emergence: real, or the ruler?**
#
# **Hardware:** CPU, about 3 minutes.
#
# **You're done when** you have the three curves against model size and a forecast error for the largest model.

# %%
import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, matplotlib.pyplot as plt
QUICK = os.environ.get("QUICK") == "1"
torch.manual_seed(0); rng = np.random.default_rng(0)
D = 6                                                      # 6-digit + 6-digit → 7-digit answers
digits = lambda x, n: [(x // 10 ** i) % 10 for i in range(n)]
def data(n):
    a, b = rng.integers(0, 10 ** D, n), rng.integers(0, 10 ** D, n)
    X = F.one_hot(torch.tensor(np.array([digits(x, D) + digits(y, D) for x, y in zip(a, b)])), 10).float().flatten(1)
    Y = torch.tensor(np.array([digits(x + y, D + 1) for x, y in zip(a, b)]))
    return X, Y
Xtr, Ytr = data(20000 if QUICK else 60000); Xte, Yte = data(3000)
WIDTHS = [8, 16, 32, 64, 128] if QUICK else [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
EPOCHS = 4 if QUICK else 25

# %%
def model(w): return nn.Sequential(nn.Linear(D * 20, w), nn.ReLU(), nn.Linear(w, w), nn.ReLU(), nn.Linear(w, (D + 1) * 10))
def train(w):
    torch.manual_seed(0); m = model(w); opt = torch.optim.Adam(m.parameters(), 2e-3)
    for _ in range(EPOCHS):
        p = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 256):
            j = p[i:i + 256]; loss = F.cross_entropy(m(Xtr[j]).view(-1, 10), Ytr[j].flatten()); opt.zero_grad(); loss.backward(); opt.step()
    return m
@torch.no_grad()
def grade(m):
    lp = m(Xte).view(-1, D + 1, 10).log_softmax(-1)
    right = lp.argmax(-1) == Yte
    tok_lp = lp.gather(-1, Yte[..., None]).squeeze(-1)
    return {"exact": right.all(1).float().mean().item(), "per_digit": right.float().mean().item(), "prob": tok_lp.mean(1).exp().mean().item()}
rows = []
for w in WIDTHS:
    m = train(w); g = grade(m); g["params"] = sum(p.numel() for p in m.parameters()); rows.append(g)
    print(f"width {w:4d} ({g['params']:>8,} params): exact {g['exact']:.3f}  per-digit {g['per_digit']:.3f}  answer probability {g['prob']:.3f}")

# %% [markdown]
# **My prediction (~70%):** exact match stays near zero for the smaller half of the models and then climbs
# steeply; per-digit accuracy and answer probability climb steadily across the whole range. A forecast of exact
# match as (per-digit accuracy)⁷, fitted only on the smaller models, gets the largest model within 15 points.

# %%
P = np.array([r["params"] for r in rows])
for key, lab in [("exact", "exact match"), ("per_digit", "per-digit accuracy"), ("prob", "probability of the right answer (per digit)")]:
    plt.plot(P, [r[key] for r in rows], "o-", label=lab)
plt.xscale("log"); plt.xlabel("parameters"); plt.ylabel("score"); plt.legend(); plt.title("the same models, three rulers"); plt.show()

# %% [markdown]
# ## Forecast the biggest model from the small ones
# Fit per-digit accuracy against log(parameters) on the smaller half (as a logistic curve), extrapolate, and raise
# it to the 7th power. Note what this assumes: digit errors are independent. Carries make them correlated.

# %%
half = len(rows) // 2 + 1
x, y = np.log10(P[:half]), np.array([r["per_digit"] for r in rows[:half]])
logit = np.log(np.clip(y, 1e-3, 1 - 1e-3) / (1 - np.clip(y, 1e-3, 1 - 1e-3)))
a, b = np.polyfit(x, logit, 1)
for i in range(half, len(rows)):
    pd = 1 / (1 + np.exp(-(a * np.log10(P[i]) + b)))
    print(f"{P[i]:>8,} params: forecast exact {pd ** (D + 1):.3f} (per-digit {pd:.3f}); measured exact {rows[i]['exact']:.3f}")
indep = [r["per_digit"] ** (D + 1) for r in rows]
print("independence check, measured exact vs per-digit^7:", [f"{r['exact']:.2f}/{v:.2f}" for r, v in zip(rows, indep)])

# %% [markdown]
# **What I measured** (full run, CPU, 13 widths, 25 epochs each):
#
# | parameters | exact match | per-digit | answer probability |
# |---|---|---|---|
# | 1.7k – 11.6k | 0.000 | 0.35 → 0.57 | 0.19 → 0.34 |
# | 16k | 0.004 | 0.62 | 0.37 |
# | 28k | 0.022 | 0.75 | 0.52 |
# | 41k | 0.304 | 0.87 | 0.74 |
# | 74k | 0.823 | 0.97 | 0.93 |
# | 361k | 0.982 | 0.997 | 0.99 |
#
# * **Right:** exact match looks emergent: zero for the first seven models, then 2% → 30% → 82% across
#   three neighbouring sizes. Per-digit accuracy and answer probability rise across the whole range, and exact match
#   is almost exactly per-digit⁷ once the models are decent (0.82 vs 0.82, 0.94 vs 0.93), so most of the
#   ‘jump’ is the ruler.
# * **Wrong:** my forecast. Fitting per-digit accuracy on the small half and extrapolating predicted 0.42 exact
#   match for the largest model; it scored 0.98. The per-digit curve itself *accelerated* between 28k and 74k
#   parameters (0.75 → 0.97), faster than the logistic trend from small models. Something real happened there:
#   probably the network finally learned to propagate carries, which is a genuine (if modest) transition hiding
#   under the mirage. So the honest summary is: mostly the metric, partly a real change, and the real part is the
#   part you can't forecast from below.
# * **Lesson:** smooth metrics make emergence *less* surprising, not perfectly predictable. A forecast needs models
#   on both sides of any internal transition, which is what scaling-ladder practice (notebook 36) does.
#
# **Stretch:** make the task genuinely compositional (the answer is right only if the carry chain is computed),
# and look for a size where even per-digit accuracy on the *last* digit jumps. That would be real emergence.
