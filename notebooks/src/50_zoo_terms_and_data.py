# %% [markdown]
# # 50 · Model zoo, part 1: the words and the data
# Before any model: what the words mean (feature, label, parameter, hyperparameter, loss, metric…), what kinds of
# data exist, how to split it, and how to measure results. Every later zoo notebook uses this vocabulary.
#
# **How exercises work.** Each exercise is a function with `...` where your code goes, followed by `check(...)`,
# which prints ✅ or a hint. Set `SOLUTIONS = True` in the next cell to see worked answers run instead.
#
# **Hardware:** CPU, a few minutes. **You're done when** every check prints ✅ with SOLUTIONS = False.

# %%
import os, numpy as np, matplotlib.pyplot as plt
SOLUTIONS = os.environ.get("SOLUTIONS") == "1"      # set to True to run the worked answers
np.random.seed(0)
def check(name, got, want, tol=1e-6):
    if got is None or got is Ellipsis: print(f"⏳ {name}: not done yet"); return False
    try: ok = np.allclose(np.asarray(got, float), np.asarray(want, float), atol=tol)
    except Exception: ok = got == want
    print(("✅ " if ok else "❌ ") + name + ("" if ok else f": got {got}, expected {want}")); return ok

# %% [markdown]
# ## 1. The vocabulary
#
# | Word | Meaning | In a house-price model |
# |---|---|---|
# | **sample** (example, row) | one item of data | one house |
# | **feature** (input, x) | a measured property of a sample | floor area, rooms, distance to centre |
# | **label** (target, y) | what we want to predict | the sale price |
# | **dataset** | many samples, usually a matrix X of shape (n_samples, n_features) and a vector y | 10,000 houses |
# | **model** | a function f(x; θ) that maps features to a prediction | price ≈ w·area + b |
# | **parameters** (weights, θ) | numbers the model *learns* from data | w and b |
# | **hyperparameters** | settings *you* choose before training | learning rate, tree depth, k in kNN |
# | **loss** | how wrong one prediction is; training minimises it | squared error |
# | **metric** | how *you* judge the model; may differ from the loss | mean absolute error in £ |
# | **training / validation / test set** | data to learn from / to tune choices on / touched once at the end | 70% / 15% / 15% |
# | **overfitting** | great on training data, poor on new data: memorised noise | a 30-degree polynomial through 20 points |
# | **underfitting** | poor everywhere: too simple | a constant price for every house |
# | **generalisation** | performance on data the model never saw | the test score |
#
# **Task types:** *regression* (predict a number), *classification* (predict a category), *clustering* (find groups,
# no labels), *generation* (produce new samples), *ranking*, *control* (choose actions: reinforcement learning).

# %% [markdown]
# ## 2. The kinds of data, and their shapes

# %%
from sklearn.datasets import load_iris, load_digits, load_diabetes
iris, digits, diab = load_iris(), load_digits(), load_diabetes()
print("tabular, classification  iris X", iris.data.shape, "y", iris.target.shape, "classes", list(iris.target_names))
print("tabular, regression      diabetes X", diab.data.shape, "y range", diab.target.min(), "–", diab.target.max())
print("images                   digits", digits.images.shape, "→ flattened", digits.data.shape)
text = ["the cat sat", "the dog sat", "a cat ran"]
vocab = sorted({w for s in text for w in s.split()})
bow = np.array([[s.split().count(w) for w in vocab] for s in text])
print("text as bag of words     ", bow.shape, vocab)
t = np.arange(200); series = np.sin(t / 10) + 0.1 * np.random.randn(200)
print("time series              ", series.shape, "→ windows of 20 to predict the next value:", np.lib.stride_tricks.sliding_window_view(series, 21).shape)
edges = np.array([[0, 1], [1, 2], [2, 0], [2, 3]])
print("graph                    ", edges.shape[0], "edges between", edges.max() + 1, "nodes")
fig, ax = plt.subplots(1, 4, figsize=(14, 3))
ax[0].scatter(iris.data[:, 2], iris.data[:, 3], c=iris.target, s=12); ax[0].set_title("tabular (iris petals)")
ax[1].imshow(digits.images[0], cmap="gray_r"); ax[1].set_title(f"image (label {digits.target[0]})")
ax[2].plot(series); ax[2].set_title("time series"); ax[3].imshow(bow, cmap="Blues"); ax[3].set_title("text → counts"); plt.show()

# %% [markdown]
# ### Exercise 1: shapes
# The digits images are 8×8. How many features does one flattened image have, and how many numbers are in the
# whole flattened dataset?

# %%
def ex1():
    if SOLUTIONS: return digits.images.shape[1] * digits.images.shape[2], digits.data.size
    ...  # return (features_per_image, total_numbers)
check("exercise 1", ex1(), (64, 1797 * 64))

# %% [markdown]
# ## 3. Splitting: train, validation, test
# Shuffle, then cut. The test set is a sealed envelope: open it once, at the very end.

# %%
# Exercise 2: return index arrays (train, val, test) for n samples, shuffled with the given seed, in proportions
# 70 / 15 / 15 (round the train and val sizes down; the test set gets the rest).
def split_indices(n, seed=0):
    if SOLUTIONS:
        idx = np.random.default_rng(seed).permutation(n); a = int(0.7 * n); b = a + int(0.15 * n)
        return idx[:a], idx[a:b], idx[b:]
    ...
s = split_indices(1797)
check("exercise 2: sizes", None if s is None else [len(p) for p in s], [1257, 269, 271])
check("exercise 2: no overlap", None if s is None else len(set(np.concatenate(s))), 1797)

# %% [markdown]
# ## 4. Preparing features
# **Standardise** numeric features (subtract the mean, divide by the standard deviation) so no feature dominates
# because of its units. Compute the mean and std on the *training* set only, then apply them to everything:
# computing them on all data leaks test information into training.
#
# **One-hot encode** categories: colour ∈ {red, green, blue} → three 0/1 columns. Never feed category codes 0, 1, 2
# as numbers: the model would think blue (2) is “twice” green (1).

# %%
def standardise(train, other):
    if SOLUTIONS:
        mu, sd = train.mean(0), train.std(0) + 1e-12; return (train - mu) / sd, (other - mu) / sd
    ...  # Exercise 3: return (train_scaled, other_scaled) using the TRAIN mean and std only
tr, te = np.array([[1., 10.], [3., 30.]]), np.array([[2., 20.]])
out = standardise(tr, te); check("exercise 3", None if out is None else out[1], [[0., 0.]])

def one_hot(labels, n_classes):
    if SOLUTIONS: return np.eye(n_classes)[labels]
    ...  # Exercise 4: labels [2, 0] with 3 classes → [[0,0,1],[1,0,0]]
check("exercise 4", one_hot(np.array([2, 0, 1]), 3), [[0, 0, 1], [1, 0, 0], [0, 1, 0]])

# %% [markdown]
# ## 5. Metrics
# **Regression:** MSE (mean squared error, punishes big misses), MAE (mean absolute error, in the label's units),
# R² (share of variance explained; 1 is perfect, 0 is “predict the mean”).
# **Classification:** accuracy, and when classes are imbalanced, precision (of the ones I flagged, how many were
# right?), recall (of the real positives, how many did I catch?), and F1 (their harmonic mean).

# %%
def regression_metrics(y, p):
    if SOLUTIONS:
        mse = np.mean((y - p) ** 2); return mse, np.mean(np.abs(y - p)), 1 - mse / np.var(y)
    ...  # Exercise 5: return (mse, mae, r2)
check("exercise 5", regression_metrics(np.array([1., 2., 3., 4.]), np.array([1., 2., 4., 4.])), (0.25, 0.25, 0.8))

def precision_recall_f1(y, p):
    if SOLUTIONS:
        tp = np.sum((p == 1) & (y == 1)); fp = np.sum((p == 1) & (y == 0)); fn = np.sum((p == 0) & (y == 1))
        pr, rc = tp / (tp + fp), tp / (tp + fn); return pr, rc, 2 * pr * rc / (pr + rc)
    ...  # Exercise 6: binary labels y and predictions p → (precision, recall, f1)
check("exercise 6", precision_recall_f1(np.array([1, 1, 0, 0, 1]), np.array([1, 0, 0, 1, 1])), (2 / 3, 2 / 3, 2 / 3))

# %% [markdown]
# ### Why accuracy lies on imbalanced data
# A fraud detector that always says “not fraud” on data with 1% fraud is 99% accurate and catches nothing.

# %%
y = (np.random.rand(10000) < 0.01).astype(int); lazy = np.zeros_like(y)
print(f"accuracy of always-no: {np.mean(lazy == y):.1%}; recall: {np.sum(lazy & y) / y.sum():.0%}")

# %% [markdown]
# ## 6. Overfitting, seen once
# Fit polynomials of growing degree to 15 noisy points and score them on fresh points.

# %%
x = np.sort(np.random.rand(15)); yy = np.sin(2 * np.pi * x) + 0.2 * np.random.randn(15)
xt = np.random.rand(200); yt = np.sin(2 * np.pi * xt) + 0.2 * np.random.randn(200)
degs = range(0, 13); tr_err, te_err = [], []
for d in degs:
    c = np.polyfit(x, yy, d); tr_err.append(np.mean((np.polyval(c, x) - yy) ** 2)); te_err.append(np.mean((np.polyval(c, xt) - yt) ** 2))
plt.semilogy(degs, tr_err, "o-", label="training error"); plt.semilogy(degs, te_err, "o-", label="test error")
plt.xlabel("polynomial degree (model complexity)"); plt.legend(); plt.title("underfit ← sweet spot → overfit"); plt.show()
print("best degree on test data:", int(np.argmin(te_err)))

# %% [markdown]
# ### Exercise 7: spot the leak
# Which of these pipelines leaks test information into training? Return the letter.
# * **A**: split → fit the scaler on train → scale train and test → train → score on test
# * **B**: scale all data → split → train → score on test
# * **C**: split → train with early stopping on the validation set → score on test

# %%
def ex7():
    if SOLUTIONS: return "B"
    ...
check("exercise 7", ex7(), "B")

# %% [markdown]
# **Next:** notebook 51, the linear family, the simplest models that actually learn.
