# %% [markdown]
# # 44 · Error bars on evals, with real models
# Train two classifiers that are genuinely close in quality, then do what most leaderboards skip: confidence
# intervals, a paired test, seed variance, clustered errors, and the “best prompt” trap (here: best of several
# test-time input formats). Ends with a test of my claim that sub-2-point gains on small benchmarks are usually noise.
# Chapter: **Is that +2% real?**, **Evals**.
#
# **Hardware:** CPU, about two minutes.
#
# **You're done when** you can state the gap between the two models with a 95% CI from a paired test, and show
# how much the seed, the test-set size and the format choice each move the result.

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import norm

QUICK = os.environ.get("QUICK") == "1"
rng = np.random.default_rng(0)
X, y = load_digits(return_X_y=True)
X = X / 16.0
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=0, stratify=y)
print("train", len(ytr), "test", len(yte))

# %% [markdown]
# ## 1. Two close models, scored per question
# Keep the per-question outcomes, not just the accuracy. Everything else needs them.


# %%
def fit(kind, seed):
    m = (
        LogisticRegression(C=1.0, max_iter=3000)
        if kind == "A"
        else MLPClassifier((32,), max_iter=300 if not QUICK else 60, random_state=seed)
    )
    return m.fit(Xtr, ytr)


A, B = fit("A", 0), fit("B", 0)
a, b = (A.predict(Xte) == yte).astype(float), (B.predict(Xte) == yte).astype(float)
print(f"A {a.mean():.3f}   B {b.mean():.3f}   raw gap {b.mean() - a.mean():+.3f}")

# %% [markdown]
# ## 2. Confidence intervals: unpaired, paired, bootstrap


# %%
def unpaired(a, b):
    se = np.sqrt(a.mean() * (1 - a.mean()) / len(a) + b.mean() * (1 - b.mean()) / len(b))
    return b.mean() - a.mean(), se


def paired(a, b):
    d = b - a
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d))


def bootstrap(a, b, n=2000):
    idx = rng.integers(0, len(a), (n, len(a)))
    g = b[idx].mean(1) - a[idx].mean(1)
    return np.percentile(g, [2.5, 97.5])


for name, (g, se) in [("unpaired", unpaired(a, b)), ("paired", paired(a, b))]:
    print(f"{name:9s} gap {g:+.3f}  95% CI [{g - 1.96 * se:+.3f}, {g + 1.96 * se:+.3f}]")

print("bootstrap 95% CI", np.round(bootstrap(a, b), 3))

# %% [markdown]
# ## 3. Seeds: the noise most papers never show
# Retrain B with five seeds. If the spread across seeds is as large as the gap, one run proves nothing.

# %%
seeds = range(3 if QUICK else 8)
accB = [(fit("B", s).predict(Xte) == yte).mean() for s in seeds]
print(
    "B across seeds:", np.round(accB, 3), f"→ mean {np.mean(accB):.3f} ± {np.std(accB, ddof=1):.3f}"
)

# %% [markdown]
# ## 4. Test-set size: how big must a benchmark be?
# Subsample the test set at different sizes; the gap estimate's spread shrinks like 1/√N.

# %%
sizes = [50, 100, 200, 400]
spread = []  # subsets of the 899-item test set; spread = []

for n in sizes:
    gaps = [
        b[i].mean() - a[i].mean()
        for i in (rng.choice(len(a), n, replace=False) for _ in range(300))
    ]
    spread.append(np.std(gaps))
    print(
        f"N = {n:4d}: gap {np.mean(gaps):+.3f} ± {np.std(gaps):.3f}; B looks worse in {np.mean(np.array(gaps) < 0):.0%} of draws"
    )

plt.loglog(sizes, spread, "o-", label="measured")
plt.loglog(sizes, spread[0] * np.sqrt(sizes[0] / np.array(sizes)), "--", label="1/√N")
plt.xlabel("questions")
plt.ylabel("std of the gap")
plt.legend()
plt.show()


def questions_needed(gap, sd_diff, power=0.8):
    return int(np.ceil(((norm.ppf(0.975) + norm.ppf(power)) * sd_diff / gap) ** 2))


print(
    "questions needed to detect a 2-point gap with this pair:",
    questions_needed(0.02, (b - a).std()),
)

# %% [markdown]
# ## 5. Clustered errors
# Pretend test items come in groups of 10 near-duplicates (like 10 questions about one passage). We make clusters
# by grouping each image with its 9 nearest neighbours; naive SEs then understate the real uncertainty.

# %%
from sklearn.cluster import KMeans

cl = KMeans(n_clusters=len(a) // 10, n_init=1 if QUICK else 3, random_state=0).fit_predict(Xte)


def clustered_se(d, ids):
    m = d.mean()
    sums = np.array([(d[ids == c] - m).sum() for c in np.unique(ids)])
    return np.sqrt((sums**2).sum()) / len(d)


d = b - a
print(
    f"naive paired SE {d.std(ddof=1) / np.sqrt(len(d)):.4f}   clustered SE {clustered_se(d, cl):.4f}"
)

# %% [markdown]
# ## 6. The best-format trap
# The analogue of trying several prompt templates: score B under several input “formats” (small shifts and
# blur of the test images, all equally valid), report the best, and compare with A scored in one format.

# %%
from scipy.ndimage import shift as nd_shift, gaussian_filter

imgs = Xte.reshape(-1, 8, 8)
formats = {
    "plain": imgs,
    "blur 0.4": gaussian_filter(imgs, (0, 0.4, 0.4)),
    "shift right": nd_shift(imgs, (0, 0, 0.3)),
    "shift down": nd_shift(imgs, (0, 0.3, 0)),
    "sharpen": np.clip(imgs * 1.15 - 0.03, 0, 1),
}
sc = {k: (B.predict(v.reshape(len(v), -1)) == yte).mean() for k, v in formats.items()}
print({k: round(v, 3) for k, v in sc.items()})
print(
    f"honest gap (plain format) {sc['plain'] - a.mean():+.3f}   reported gap (best format) {max(sc.values()) - a.mean():+.3f}"
)

# %% [markdown]
# ## 7. Your turn: the 2-point claim
# My claim in the chapter (~75% confident): most reported gains under 2 points on benchmarks with fewer than 1,000
# items wouldn't survive a paired test with several samples per question. Feed in per-question results from any
# two models (lm-eval-harness writes them with `--log_samples`) and count how many published gaps survive.


# %%
def verdict(scores_a, scores_b, cluster_ids=None):
    d = np.asarray(scores_b, float) - np.asarray(scores_a, float)
    se = (
        clustered_se(d, np.asarray(cluster_ids))
        if cluster_ids is not None
        else d.std(ddof=1) / np.sqrt(len(d))
    )
    lo, hi = d.mean() - 1.96 * se, d.mean() + 1.96 * se
    return f"gap {d.mean():+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}] → " + (
        "B is better" if lo > 0 else "A is better" if hi < 0 else "can't tell"
    )


print(verdict(a, b))

# %% [markdown]
# **What I measured** (CPU, ~1 minute). A (logistic regression) 96.1%, B (small MLP) 96.6%: a +0.4-point gap.
# Paired 95% CI [−0.5, +1.4] points, so we can't tell them apart; the unpaired CI is almost twice as wide
# ([−1.3, +2.2]). B's accuracy moved ±0.3 points across 8 seeds, as large as the gap itself. On 50–200-question
# subsets, B looked *worse* in about a fifth of draws. Clustering barely changed the SE here (0.0047 → 0.0050)
# because near-duplicate digits are mostly all right or all wrong for both models together, which cancels in
# the paired difference; on real benchmarks with shared passages the effect is usually bigger. The format trap
# was small (+0.2 points) because most “formats” hurt; it bites when formats are about equally good, as prompt
# templates usually are.
