# %% [markdown]
# # 53 · Model zoo, part 4: learning without labels
# No y, only X. Find groups (clustering), find the few directions that matter (dimensionality reduction), or find
# what doesn't fit (anomaly detection). You'll write k-means and PCA yourself, then meet Gaussian mixtures,
# hierarchical clustering, DBSCAN, t-SNE and isolation forests.
#
# **You're done when** all checks print ✅ with SOLUTIONS = False. CPU, ~25 minutes.

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, load_digits

SOLUTIONS = os.environ.get("SOLUTIONS") == "1"
rng = np.random.default_rng(0)


def check(name, got, want, tol=1e-3):
    if got is None or got is Ellipsis:
        print(f"⏳ {name}: not done yet")
        return False
    try:
        ok = np.allclose(np.asarray(got, float), np.asarray(want, float), atol=tol)
    except Exception:
        ok = got == want
    print(("✅ " if ok else "❌ ") + name + ("" if ok else f": got {got}, expected {want}"))
    return ok


# %% [markdown]
# ## 1. k-means: “k centres, each point joins the nearest”
# Repeat two steps until nothing changes: **assign** each point to its nearest centre; **update** each centre to the
# mean of its points. It minimises the total squared distance to centres (*inertia*). You choose k.

# %%
X, true = make_blobs(600, centers=4, cluster_std=0.9, random_state=3)


def kmeans(X, k, iters=50, seed=0):
    C = X[np.random.default_rng(seed).choice(len(X), k, replace=False)]
    for _ in range(iters):
        if SOLUTIONS:
            assign = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
            C = np.array(
                [X[assign == j].mean(0) if np.any(assign == j) else C[j] for j in range(k)]
            )
        else:
            return None, None  # Exercise 1: assign step, then update step
    return assign, C


assign, C = kmeans(X, 4)
from sklearn.metrics import adjusted_rand_score

check(
    "exercise 1: k-means recovers the blobs (ARI > 0.9)",
    None if assign is None else bool(adjusted_rand_score(true, assign) > 0.9),
    True,
)

if assign is not None:
    plt.scatter(*X.T, c=assign, s=6)
    plt.scatter(*C.T, c="red", marker="x", s=120)
    plt.title("k-means, k = 4")
    plt.show()

# %% [markdown]
# ### Choosing k: the elbow
# Inertia always falls as k grows; look for the bend where extra clusters stop helping much.

# %%
from sklearn.cluster import KMeans

inertia = [KMeans(k, n_init=5, random_state=0).fit(X).inertia_ for k in range(1, 10)]
plt.plot(range(1, 10), inertia, "o-")
plt.xlabel("k")
plt.ylabel("inertia")
plt.title("the elbow is at k = 4")
plt.show()

# %% [markdown]
# ## 2. Where k-means fails, and what to use instead
# k-means assumes round, similar-sized clusters. On two interleaved moons it cuts straight through them.
# **DBSCAN** grows clusters through dense regions and marks sparse points as noise; **hierarchical (agglomerative)**
# clustering merges the closest groups step by step; **Gaussian mixtures** allow stretched (elliptical) clusters
# and give soft, probabilistic assignments.

# %%
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

M, _ = make_moons(400, noise=0.06, random_state=0)
S = np.r_[rng.normal([0, 0], [3, 0.4], (200, 2)), rng.normal([0, 3], [0.4, 1.5], (200, 2))]
fig, ax = plt.subplots(2, 4, figsize=(15, 6.5))

for row, data, name in [(0, M, "moons"), (1, S, "stretched")]:
    for a, (alg, lab) in zip(
        ax[row],
        [
            ("k-means", KMeans(2, n_init=5, random_state=0).fit_predict(data)),
            ("DBSCAN", DBSCAN(eps=0.2 if name == "moons" else 0.6).fit_predict(data)),
            (
                "agglomerative (single link)",
                AgglomerativeClustering(2, linkage="single").fit_predict(data),
            ),
            ("Gaussian mixture", GaussianMixture(2, random_state=0).fit(data).predict(data)),
        ],
    ):
        a.scatter(*data.T, c=lab, s=5, cmap="tab10")
        a.set_title(f"{alg} on {name}", fontsize=9)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. PCA: “the directions with the most spread”
# Centre the data, then the principal components are the eigenvectors of its covariance matrix, equivalently the
# right singular vectors from an SVD. Project onto the top few to compress, visualise, or denoise.

# %%
digits = load_digits()
D = digits.data


def pca(X, k):
    if SOLUTIONS:
        Xc = X - X.mean(0)
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        return Xc @ Vt[:k].T, (s**2 / np.sum(s**2))[:k]
    # Exercise 2: return (projected data of shape [n, k], explained-variance ratio of each of the k components)
    ...


out = pca(D, 2)
from sklearn.decomposition import PCA

ref = PCA(2).fit(D).explained_variance_ratio_
check("exercise 2: explained variance matches scikit-learn", None if out is None else out[1], ref)

if out is not None:
    plt.scatter(*out[0].T, c=digits.target, cmap="tab10", s=6)
    plt.colorbar()
    plt.title("digits: 64 dimensions → 2 with PCA")
    plt.show()

cum = np.cumsum(PCA().fit(D).explained_variance_ratio_)
print("components needed for 90% of the variance:", int(np.searchsorted(cum, 0.9) + 1), "of 64")

# %% [markdown]
# ### Exercise 3: compress and rebuild
# Keep k components, map back to 64 pixels, and measure the reconstruction error. How many components do you need
# for a mean squared error below 1.0?


# %%
def components_needed(threshold=1.0):
    if SOLUTIONS:
        for k in range(1, 65):
            p = PCA(k).fit(D)
            if np.mean((p.inverse_transform(p.transform(D)) - D) ** 2) < threshold:
                return k
    ...


k = components_needed()
print("components for MSE < 1:", k)
check("exercise 3: between 20 and 40 components", None if k is None else bool(20 <= k <= 40), True)

# %% [markdown]
# ## 4. t-SNE: neighbourhoods, not distances
# t-SNE keeps points that are neighbours in 64-D close in 2-D. Good for looking, bad for measuring: cluster sizes
# and the distances between clusters in the picture mean nothing. UMAP (`pip install umap-learn`) is a faster cousin.

# %%
from sklearn.manifold import TSNE

n = 600 if os.environ.get("QUICK") == "1" else 1797
emb = TSNE(2, perplexity=30, random_state=0, init="pca").fit_transform(D[:n])
plt.scatter(*emb.T, c=digits.target[:n], cmap="tab10", s=6)
plt.title("t-SNE of the digits: clusters PCA couldn't separate")
plt.show()

# %% [markdown]
# ## 5. Anomaly detection
# **Isolation forest:** random splits isolate unusual points in fewer steps. Useful for fraud, faults and data bugs.

# %%
from sklearn.ensemble import IsolationForest

normal = rng.normal(0, 1, (500, 2))
odd = rng.uniform(-6, 6, (20, 2))
A = np.r_[normal, odd]
flag = IsolationForest(contamination=0.04, random_state=0).fit_predict(A) == -1
print(f"flagged {flag.sum()} points; {flag[500:].sum()} of the 20 planted anomalies caught")
plt.scatter(*A.T, c=np.where(flag, "red", "gray"), s=8)
plt.title("isolation forest: red = flagged")
plt.show()

# %% [markdown]
# ### Exercise 4: match the tool to the job
# Return a dict with values from "kmeans", "dbscan", "gmm", "pca", "tsne", "isolation_forest".


# %%
def ex4():
    if SOLUTIONS:
        return {
            "segment_customers_into_5_groups": "kmeans",
            "find_odd_transactions": "isolation_forest",
            "compress_features_before_a_model": "pca",
            "clusters_of_any_shape_with_noise": "dbscan",
            "soft_probabilities_of_membership": "gmm",
            "a_pretty_2d_map_of_embeddings": "tsne",
        }
    ...


check(
    "exercise 4",
    ex4(),
    {
        "segment_customers_into_5_groups": "kmeans",
        "find_odd_transactions": "isolation_forest",
        "compress_features_before_a_model": "pca",
        "clusters_of_any_shape_with_noise": "dbscan",
        "soft_probabilities_of_membership": "gmm",
        "a_pretty_2d_map_of_embeddings": "tsne",
    },
)

# %% [markdown]
# | Method | Finds | You choose | Watch out for |
# |---|---|---|---|
# | k-means | k round clusters | k | different sizes and shapes; scale features first |
# | Gaussian mixture | k elliptical clusters, soft membership | k, covariance type | local optima: run several inits |
# | Agglomerative | a tree of merges | linkage, where to cut | O(n²) memory |
# | DBSCAN | dense regions of any shape + noise | eps, min_samples | varying densities |
# | PCA | linear directions of most variance | number of components | only linear structure |
# | t-SNE / UMAP | neighbourhood maps for viewing | perplexity / n_neighbors | distances in the plot aren't meaningful |
# | Isolation forest | outliers | contamination rate | needs a sense of how rare anomalies are |
