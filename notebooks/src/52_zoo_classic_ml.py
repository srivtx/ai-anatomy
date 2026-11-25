# %% [markdown]
# # 52 · Model zoo, part 3: classic machine learning
# Six families that dominated before deep learning and still win on many tabular problems: k-nearest neighbours,
# naive Bayes, decision trees, random forests, gradient boosting and support vector machines. For each: the idea
# in one line, its anatomy (what it stores, what it learns), a from-scratch piece, and its decision boundary.
#
# **You're done when** all checks print ✅ with SOLUTIONS = False and you can fill in the comparison table at the
# end from memory. CPU, ~25 minutes.

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, load_breast_cancer
from sklearn.model_selection import train_test_split

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


X, y = make_moons(400, noise=0.3, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)


def boundary(ax, predict, title):
    gx, gy = np.meshgrid(np.linspace(-2, 3, 150), np.linspace(-1.5, 2, 150))
    ax.contourf(
        gx, gy, predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape), alpha=0.35, cmap="RdBu"
    )
    ax.scatter(*Xte.T, c=yte, cmap="RdBu", s=8)
    ax.set_title(title, fontsize=10)


# %% [markdown]
# ## 1. k-nearest neighbours: “you are like your neighbours”
# **Stores:** the whole training set. **Learns:** nothing (lazy). **Predicts:** majority label of the k closest points.
# Hyperparameter k: small k = wiggly boundary (overfit), large k = smooth (underfit).


# %%
def knn_predict(Xtr, ytr, Xq, k=5):
    if SOLUTIONS:
        d = ((Xq[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
        nn = np.argsort(d, 1)[:, :k]
        return (ytr[nn].mean(1) > 0.5).astype(int)
    # Exercise 1: distances to all training points, take the k nearest, majority vote (binary labels)
    ...


p = knn_predict(Xtr, ytr, Xte)
check(
    "exercise 1: kNN accuracy above 85%",
    None if p is None else bool(np.mean(p == yte) > 0.85),
    True,
)

# %% [markdown]
# ### Exercise 2: choose k on a validation set, not the test set

# %%
Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.3, random_state=1)


def best_k(ks=(1, 3, 5, 9, 15, 25, 51)):
    if SOLUTIONS:
        return max(ks, key=lambda k: np.mean(knn_predict(Xa, ya, Xv, k) == yv))
    ...  # return the k with the highest validation accuracy


k = best_k()
print("chosen k:", k)
check("exercise 2: a sensible k (not 1)", None if k is None else bool(k > 1), True)

# %% [markdown]
# ## 2. Naive Bayes: “count, assuming features are independent”
# **Learns:** per class, a mean and variance for each feature (Gaussian NB). **Predicts:** the class with the highest
# p(class) × Π p(feature | class). The independence assumption is usually false, and it often works anyway,
# especially for text.

# %%
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB().fit(Xtr, ytr)
print("naive Bayes accuracy", nb.score(Xte, yte))

# %% [markdown]
# ## 3. Decision trees: “twenty questions”
# **Learns:** a tree of yes/no questions on single features (x₁ < 0.3?). Each split is chosen to make the children
# purer. **Gini impurity** of a node with class shares pₖ is 1 − Σpₖ².


# %%
def gini(labels):
    if SOLUTIONS:
        p = np.bincount(labels) / len(labels)
        return 1 - np.sum(p**2)
    ...  # Exercise 3


check(
    "exercise 3: gini",
    (
        None
        if gini(np.array([0, 0, 1, 1])) is None
        else [gini(np.array([0, 0, 1, 1])), gini(np.array([1, 1, 1]))]
    ),
    [0.5, 0.0],
)


def best_split(x, labels):
    """Exercise 4: for one feature x, return the threshold minimising the weighted Gini of the two children."""
    if SOLUTIONS:
        best = (None, 1e9)
        for t in np.unique(x)[:-1]:
            L, R = labels[x <= t], labels[x > t]
            g = (len(L) * gini(L) + len(R) * gini(R)) / len(labels)
            if g < best[1]:
                best = (t, g)
        return best[0]
    ...


check(
    "exercise 4: best threshold",
    best_split(np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0]), np.array([0, 0, 0, 1, 1, 1])),
    3.0,
)

from sklearn.tree import DecisionTreeClassifier, export_text

for depth in [2, 5, None]:
    t = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(Xtr, ytr)
    print(
        f"tree depth {str(depth):4s}: train {t.score(Xtr, ytr):.2f}  test {t.score(Xte, yte):.2f}  leaves {t.get_n_leaves()}"
    )

print(export_text(DecisionTreeClassifier(max_depth=2).fit(Xtr, ytr), feature_names=["x1", "x2"]))

# %% [markdown]
# ## 4. Ensembles of trees
# **Random forest:** many deep trees, each on a bootstrap sample with random feature subsets; average their votes.
# Lowers variance. **Gradient boosting:** many shallow trees in sequence, each fitting the errors (gradients) of the
# ones before. Lowers bias. Boosting (XGBoost, LightGBM, CatBoost) is still the default winner on tabular data.

# %%
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

rf = RandomForestClassifier(200, random_state=0).fit(Xtr, ytr)
gb = GradientBoostingClassifier(random_state=0).fit(Xtr, ytr)
print("random forest", rf.score(Xte, yte), " gradient boosting", gb.score(Xte, yte))

# %% [markdown]
# ### Exercise 5: boosting by hand, for regression
# Start from the mean; repeatedly fit a depth-1 tree (a stump) to the residuals and add a fraction of it.

# %%
from sklearn.tree import DecisionTreeRegressor

xr = np.sort(rng.random(200)) * 6
yr = np.sin(xr) + 0.2 * rng.normal(size=200)


def boost(x, y, rounds=100, lr=0.1):
    pred = np.full_like(y, y.mean())
    for _ in range(rounds):
        if SOLUTIONS:
            stump = DecisionTreeRegressor(max_depth=1).fit(x[:, None], y - pred)
            pred = pred + lr * stump.predict(x[:, None])
        else:
            return None  # fit a stump to the residuals y - pred and add lr × its prediction
    return pred


pb = boost(xr, yr)
check(
    "exercise 5: boosting fits well (MSE < 0.08)",
    None if pb is None else bool(np.mean((pb - yr) ** 2) < 0.08),
    True,
)

if pb is not None:
    plt.scatter(xr, yr, s=6, c="gray")
    plt.plot(xr, pb, "r")
    plt.title("100 stumps added together")
    plt.show()

# %% [markdown]
# ## 5. Support vector machines: “the widest street”
# **Learns:** the boundary with the largest margin to the nearest points (the *support vectors*); only those points
# matter. With the **kernel trick** (RBF kernel) it draws curved boundaries without computing new features.

# %%
from sklearn.svm import SVC

svl, svr = SVC(kernel="linear").fit(Xtr, ytr), SVC(kernel="rbf", C=1).fit(Xtr, ytr)
print(
    "linear SVM", svl.score(Xte, yte), " RBF SVM", svr.score(Xte, yte), " support vectors:",
    svr.n_support_.sum(), "of", len(ytr)
)

# %% [markdown]
# ## 6. Six boundaries side by side

# %%
fig, ax = plt.subplots(2, 3, figsize=(13, 7.5))

for a, (name, m) in zip(
    ax.ravel(),
    [
        ("kNN (k=15)", None),
        ("naive Bayes", nb),
        ("decision tree (depth 5)", DecisionTreeClassifier(max_depth=5).fit(Xtr, ytr)),
        ("random forest", rf),
        ("gradient boosting", gb),
        ("RBF SVM", svr),
    ],
):
    if m is None:
        from sklearn.neighbors import KNeighborsClassifier

        m = KNeighborsClassifier(15).fit(Xtr, ytr)
    boundary(a, m.predict, f"{name}: {m.score(Xte, yte):.2f}")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. On real tabular data
# Breast-cancer diagnosis: 30 numeric features. Scaling matters for kNN and SVM, not for trees.

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

Xb, yb = load_breast_cancer(return_X_y=True)
a, b, c, d = train_test_split(Xb, yb, test_size=0.3, random_state=0)

for name, m in [
    ("logistic", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
    ("kNN (unscaled)", KNeighborsClassifier()),
    ("kNN (scaled)", make_pipeline(StandardScaler(), KNeighborsClassifier())),
    ("naive Bayes", GaussianNB()),
    ("tree", DecisionTreeClassifier(random_state=0)),
    ("random forest", RandomForestClassifier(300, random_state=0)),
    ("boosting", GradientBoostingClassifier(random_state=0)),
    ("RBF SVM (scaled)", make_pipeline(StandardScaler(), SVC())),
]:
    print(f"{name:18s} {m.fit(a, c).score(b, d):.3f}")

# %% [markdown]
# ### Exercise 6: which model?
# Return a dict mapping each situation to one of "knn", "naive_bayes", "tree", "forest", "boosting", "svm".
# * "explain_to_a_doctor": the model must be readable as rules
# * "spam_words": 50,000 word-count features, little training time
# * "kaggle_tabular": best accuracy on a big tabular dataset
# * "few_samples_curvy": 300 samples, a curved boundary, features scaled


# %%
def ex6():
    if SOLUTIONS:
        return {
            "explain_to_a_doctor": "tree",
            "spam_words": "naive_bayes",
            "kaggle_tabular": "boosting",
            "few_samples_curvy": "svm",
        }
    ...


check(
    "exercise 6",
    ex6(),
    {
        "explain_to_a_doctor": "tree",
        "spam_words": "naive_bayes",
        "kaggle_tabular": "boosting",
        "few_samples_curvy": "svm",
    },
)

# %% [markdown]
# ## The comparison table
#
# | Model | Stores / learns | Needs scaling? | Strength | Weakness |
# |---|---|---|---|---|
# | kNN | all training data | yes | no training, any boundary | slow predictions, curse of dimensionality |
# | Naive Bayes | per-class feature statistics | no | tiny, fast, great for text | independence assumption |
# | Decision tree | a tree of thresholds | no | readable, handles mixed types | overfits alone, unstable |
# | Random forest | hundreds of deep trees | no | robust default, little tuning | big, less readable |
# | Gradient boosting | hundreds of shallow trees in sequence | no | usually best on tabular data | more tuning, sequential training |
# | SVM | support vectors + kernel | yes | strong on small/medium data | slow beyond ~100k samples, needs C/γ tuning |
