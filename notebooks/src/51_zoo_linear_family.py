# %% [markdown]
# # 51 · Model zoo, part 2: the linear family
# The simplest models that learn, and still the first thing to try on tabular data: linear regression (two ways),
# polynomial features, ridge and lasso, logistic regression, softmax regression and Poisson regression. You'll
# write each one in NumPy, then meet its scikit-learn twin.
#
# **Anatomy of a linear model:** prediction = link⁻¹(w·x + b). The *link* decides the family: identity → linear
# regression (numbers), sigmoid → logistic (yes/no), softmax → multiclass, exp → Poisson (counts). Same core,
# different output and loss. That's a **generalised linear model (GLM)**.
#
# **You're done when** all checks print ✅ with SOLUTIONS = False. CPU, ~20 minutes.

# %%
import os
import numpy as np
import matplotlib.pyplot as plt

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
    print(
        ("✅ " if ok else "❌ ")
        + name
        + ("" if ok else f": got {np.round(got, 4)}, expected {np.round(want, 4)}")
    )
    return ok


# %% [markdown]
# ## 1. Linear regression: y ≈ Xw + b
# **Loss:** mean squared error. **Two ways to fit:** the closed form (normal equations) w = (XᵀX)⁻¹Xᵀy, exact but
# O(d³); and gradient descent, which scales to anything. Add a column of ones to X so b becomes just another weight.

# %%
n = 200
X = rng.normal(size=(n, 3))
true_w = np.array([2.0, -1.0, 0.5])
y = X @ true_w + 3.0 + 0.1 * rng.normal(size=n)
X1 = np.hstack([X, np.ones((n, 1))])


def fit_closed_form(X1, y):
    if SOLUTIONS:
        return np.linalg.solve(X1.T @ X1, X1.T @ y)
    ...  # Exercise 1: solve the normal equations (use np.linalg.solve, not inv)


check("exercise 1: closed form", fit_closed_form(X1, y), [2.0, -1.0, 0.5, 3.0], tol=0.05)


def fit_gd(X1, y, lr=0.1, steps=500):
    w = np.zeros(X1.shape[1])
    for _ in range(steps):
        if SOLUTIONS:
            grad = 2 / len(y) * X1.T @ (X1 @ w - y)
        else:
            grad = ...  # Exercise 2: gradient of the mean squared error with respect to w
        if grad is Ellipsis:
            return None
        w -= lr * grad
    return w


check("exercise 2: gradient descent", fit_gd(X1, y), [2.0, -1.0, 0.5, 3.0], tol=0.05)

# %% [markdown]
# ## 2. Polynomial features and regularisation
# A linear model on features [x, x², x³, …] fits curves; the model is still linear *in its weights*. With many
# features it overfits, so add a penalty: **ridge** (L2, λ‖w‖²) shrinks all weights; **lasso** (L1, λ‖w‖₁) drives
# some exactly to zero, which selects features.

# %%
x = np.sort(rng.random(20))
yy = np.sin(2 * np.pi * x) + 0.15 * rng.normal(size=20)
P = lambda x, d=12: np.vstack([x**k for k in range(d + 1)]).T


def ridge(Phi, y, lam):
    if SOLUTIONS:
        return np.linalg.solve(Phi.T @ Phi + lam * np.eye(Phi.shape[1]), Phi.T @ y)
    ...  # Exercise 3: closed-form ridge, (ΦᵀΦ + λI)⁻¹Φᵀy


xs = np.linspace(0, 1, 200)

for lam in [1e-9, 1e-4, 1e-1]:
    w = ridge(P(x), yy, lam)
    if w is not None:
        plt.plot(xs, P(xs) @ w, label=f"λ = {lam:g}")

plt.scatter(x, yy, c="k", s=15)
plt.ylim(-2, 2)
plt.legend()
plt.title("degree-12 polynomial, three ridge strengths")
plt.show()
w_big, w_small = ridge(P(x), yy, 1e-9), ridge(P(x), yy, 1e-1)
check(
    "exercise 3: ridge shrinks weights",
    None if w_big is None else bool(np.abs(w_small).sum() < np.abs(w_big).sum()),
    True,
)

from sklearn.linear_model import Lasso

Z = rng.normal(size=(200, 10))
target = 3 * Z[:, 0] - 2 * Z[:, 1] + 0.1 * rng.normal(size=200)
lasso = Lasso(alpha=0.1).fit(Z, target)
print("lasso weights (only features 0 and 1 matter):", np.round(lasso.coef_, 2))

# %% [markdown]
# ## 3. Logistic regression: yes/no
# p(y = 1 | x) = σ(w·x + b), with σ(z) = 1 / (1 + e⁻ᶻ). **Loss:** binary cross-entropy. Its gradient has the same
# shape as linear regression's: Xᵀ(p − y) / n. That's the GLM pattern.

# %%
Xc = np.vstack([rng.normal([-1, -1], 1, (100, 2)), rng.normal([1.5, 1], 1, (100, 2))])
yc = np.r_[np.zeros(100), np.ones(100)]
Xc1 = np.hstack([Xc, np.ones((200, 1))])
sigmoid = lambda z: 1 / (1 + np.exp(-z))


def fit_logistic(X1, y, lr=0.5, steps=2000):
    w = np.zeros(X1.shape[1])
    for _ in range(steps):
        if SOLUTIONS:
            grad = X1.T @ (sigmoid(X1 @ w) - y) / len(y)
        else:
            grad = ...  # Exercise 4: gradient of the mean binary cross-entropy
        if grad is Ellipsis:
            return None
        w -= lr * grad
    return w


w = fit_logistic(Xc1, yc)
acc = None if w is None else np.mean((sigmoid(Xc1 @ w) > 0.5) == yc)
check("exercise 4: logistic accuracy above 90%", None if acc is None else bool(acc > 0.9), True)

if w is not None:
    gx, gy = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))
    pz = sigmoid(np.c_[gx.ravel(), gy.ravel(), np.ones(10000)] @ w).reshape(100, 100)
    plt.contourf(gx, gy, pz, levels=20, cmap="RdBu_r", alpha=0.5)
    plt.scatter(*Xc.T, c=yc, cmap="RdBu_r", s=10)
    plt.title("logistic regression: a straight decision boundary")
    plt.show()

# %% [markdown]
# ## 4. Softmax regression: many classes
# One weight vector per class; softmax turns the K scores into probabilities. **Loss:** cross-entropy. On the
# 8×8 digits, this one-layer model already gets about 90%.

# %%
from sklearn.datasets import load_digits

d = load_digits()
Xd = np.hstack([d.data / 16, np.ones((len(d.data), 1))])
yd = d.target
tr, te = np.arange(len(yd)) < 1200, np.arange(len(yd)) >= 1200


def softmax(Z):
    if SOLUTIONS:
        E = np.exp(Z - Z.max(1, keepdims=True))
        return E / E.sum(1, keepdims=True)
    ...  # Exercise 5: row-wise softmax, numerically stable (subtract each row's max first)


check(
    "exercise 5: softmax",
    softmax(np.array([[1000.0, 1000.0], [0.0, np.log(3)]])),
    [[0.5, 0.5], [0.25, 0.75]],
)

if softmax(np.zeros((1, 2))) is not None:
    W = np.zeros((Xd.shape[1], 10))
    Y = np.eye(10)[yd[tr]]
    for _ in range(300):
        W -= 0.5 * Xd[tr].T @ (softmax(Xd[tr] @ W) - Y) / tr.sum()
    print(
        f"softmax regression on digits: test accuracy {np.mean((Xd[te] @ W).argmax(1) == yd[te]):.1%}"
    )

# %% [markdown]
# ## 5. Poisson regression: counts
# For counts (visits per day, goals per match) predict a rate λ = exp(w·x + b), which is always positive.
# Loss: Poisson negative log-likelihood. Gradient: Xᵀ(λ − y)/n, the GLM pattern again.

# %%
Xp = np.hstack([rng.normal(size=(500, 1)), np.ones((500, 1))])
yp = rng.poisson(np.exp(Xp @ np.array([0.8, 1.0])))
w = np.zeros(2)

for _ in range(2000):
    w -= 0.05 * Xp.T @ (np.exp(Xp @ w) - yp) / len(yp)

print("Poisson regression recovered weights", np.round(w, 2), "(true 0.8, 1.0)")

# %% [markdown]
# ## 6. The same models in scikit-learn

# %%
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression, PoissonRegressor

print("LinearRegression", np.round(LinearRegression().fit(X, y).coef_, 2))
print("LogisticRegression accuracy", LogisticRegression().fit(Xc, yc).score(Xc, yc))
print(
    "multinomial LogisticRegression on digits",
    round(LogisticRegression(max_iter=2000).fit(d.data[tr], yd[tr]).score(d.data[te], yd[te]), 3),
)
print("PoissonRegressor", np.round(PoissonRegressor(alpha=0).fit(Xp[:, :1], yp).coef_, 2))

# %% [markdown]
# ### Exercise 6: pick the family
# For each task, which member of the linear family fits? Answer with a list of: "linear", "logistic", "softmax", "poisson".
# 1. Predict tomorrow's temperature. 2. Will this email be spam? 3. Which of 10 digits is this? 4. How many customers arrive per hour?


# %%
def ex6():
    if SOLUTIONS:
        return ["linear", "logistic", "softmax", "poisson"]
    ...


check("exercise 6", ex6(), ["linear", "logistic", "softmax", "poisson"])

# %% [markdown]
# **The dilemma.** Linear models are fast, interpretable (each weight is “how much this feature matters”) and hard
# to overfit with regularisation, but they can't learn interactions or curves unless you hand-craft features.
# Trees (notebook 52) and neural nets (54) learn those features themselves, at the cost of interpretability and data.
