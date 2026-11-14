# %% [markdown]
# # 57 · Model zoo, part 8: picking a model, and the dilemmas
# You've met every family. Now the practical question: given *this* data and *these* constraints, what do you try
# first? We race the zoo across five kinds of data, then walk through the dilemmas that decide real choices: bias
# vs variance, accuracy vs interpretability, data size vs capacity, latency vs quality, and “no free lunch”.
#
# **You're done when** all checks print ✅ with SOLUTIONS = False and you can defend a model choice for a new
# problem in two sentences. CPU, ~15 minutes.

# %%
import os, time, numpy as np, matplotlib.pyplot as plt, warnings; warnings.filterwarnings("ignore")
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits, make_moons
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
SOLUTIONS = os.environ.get("SOLUTIONS") == "1"; QUICK = os.environ.get("QUICK") == "1"
rng = np.random.default_rng(0)
def check(name, got, want, tol=1e-3):
    if got is None or got is Ellipsis: print(f"⏳ {name}: not done yet"); return False
    try: ok = np.allclose(np.asarray(got, float), np.asarray(want, float), atol=tol)
    except Exception: ok = got == want
    print(("✅ " if ok else "❌ ") + name + ("" if ok else f": got {got}, expected {want}")); return ok

# %% [markdown]
# ## 1. Always start with a baseline
# A model that ignores the input (predict the most common class, or the mean) tells you what “no learning” scores.
# Anything you build must beat it clearly, or something is wrong.

# %%
def cv_score(model, X, y, folds=5):
    if SOLUTIONS: return cross_val_score(model, X, y, cv=KFold(folds, shuffle=True, random_state=0)).mean()
    ...  # Exercise 1: mean 5-fold cross-validated score (shuffle with random_state=0)
Xb, yb = load_breast_cancer(return_X_y=True)
base = cv_score(DummyClassifier(strategy="most_frequent"), Xb, yb)
check("exercise 1: baseline accuracy on breast cancer", base, np.mean(yb == 1), tol=0.02)

# %% [markdown]
# ## 2. Race the zoo across five kinds of data
# Tabular classification, tabular regression, small images, a curved 2-D problem, and a time series turned into
# windows. Each cell is a 5-fold cross-validated score (accuracy, or R² for regression), plus fit time.

# %%
Xd, yd = load_diabetes(return_X_y=True); Xi, yi = load_digits(return_X_y=True); Xm, ym = make_moons(600, noise=0.3, random_state=0)
t = np.arange(1200); series = np.sin(t / 12) + 0.5 * np.sin(t / 40) + 0.2 * rng.normal(size=1200)
W = np.lib.stride_tricks.sliding_window_view(series, 25); Xt, yt = W[:, :-1], W[:, -1]
sc = lambda m: make_pipeline(StandardScaler(), m)
CLS = {"baseline": DummyClassifier(), "logistic": sc(LogisticRegression(max_iter=3000)), "kNN": sc(KNeighborsClassifier()), "tree": DecisionTreeClassifier(random_state=0),
       "random forest": RandomForestClassifier(200, random_state=0, n_jobs=-1), "boosting": HistGradientBoostingClassifier(random_state=0), "SVM": sc(SVC()), "MLP": sc(MLPClassifier(hidden_layer_sizes=(128,), max_iter=600, random_state=0))}
REG = {"baseline": DummyRegressor(), "logistic": sc(Ridge()), "kNN": sc(KNeighborsRegressor()), "tree": DecisionTreeRegressor(random_state=0),
       "random forest": RandomForestRegressor(200, random_state=0, n_jobs=-1), "boosting": HistGradientBoostingRegressor(random_state=0), "SVM": sc(SVR()), "MLP": sc(MLPRegressor(hidden_layer_sizes=(128,), max_iter=1500, random_state=0))}
TASKS = [("tabular: diagnosis", Xb, yb, CLS), ("tabular: disease progression (R²)", Xd, yd, REG), ("images: digits", Xi, yi, CLS), ("2-D curved: moons", Xm, ym, CLS), ("time series: next value (R²)", Xt, yt, REG)]
if QUICK: TASKS = [(n, X[:300], y[:300], M) for n, X, y, M in TASKS]
table = {}
for tname, X, y, models in TASKS:
    for mname, m in models.items():
        t0 = time.time(); s = cross_val_score(m, X, y, cv=KFold(5, shuffle=True, random_state=0)).mean(); table[(tname, mname)] = (s, time.time() - t0)
names = list(CLS)
print(f"{'':36s}" + "".join(f"{n[:11]:>12s}" for n in names))
for tname, *_ in TASKS:
    row = [table[(tname, n)][0] for n in names]; best = int(np.argmax(row))
    print(f"{tname:36s}" + "".join(f"{('*' if i == best else ' ') + f'{v:.3f}':>12s}" for i, v in enumerate(row)))
print("(* = best; the 'logistic' column is ridge regression for the R² tasks)")

# %% [markdown]
# **What I measured** (5-fold CV, full data). Boosting won *none* of these, and ridge regression won both
# regression tasks:
#
# | Task | Winner | Runner-up |
# |---|---|---|
# | diagnosis (569 rows) | SVM 0.979 | MLP 0.975, logistic 0.972 |
# | disease progression (442 rows) | ridge R² 0.490 | MLP 0.483; single tree −0.135 |
# | digits (1,797 images) | SVM 0.982 | MLP 0.978, kNN 0.977 |
# | moons | MLP 0.908 | kNN 0.902 |
# | time series | ridge R² 0.924 | MLP 0.923 |
#
# The lesson the table hides: these datasets are *small*. Boosting's famous dominance on tabular data shows up from
# a few thousand rows with messy, mixed-type features; on a few hundred clean numeric rows, well-regularised
# linear models and SVMs are just as good or better. And a single unpruned tree is the worst real model
# everywhere: high variance in action.
#
# ### Exercise 2: read the table
# Return the name of the best model for each task from your `table` (look it up programmatically).

# %%
def winners():
    if SOLUTIONS: return {tname: max(names, key=lambda n: table[(tname, n)][0]) for tname, *_ in TASKS}
    ...
w = winners(); print(w)
check("exercise 2: no task is won by the baseline", None if w is None else all(v != "baseline" for v in w.values()), True)

# %% [markdown]
# ## 3. The dilemmas
#
# **Bias vs variance.** Simple models (linear, shallow trees) are wrong in the same way every time: *bias*. Flexible
# models (deep trees, kNN with k = 1, big nets) change a lot with the training sample: *variance*. Ensembles cut
# variance (forests) or bias (boosting); regularisation and more data cut variance.

# %%
from sklearn.model_selection import validation_curve
depths = np.arange(1, 16)
tr, va = validation_curve(DecisionTreeClassifier(random_state=0), Xm, ym, param_name="max_depth", param_range=depths, cv=5)
plt.plot(depths, tr.mean(1), "o-", label="training"); plt.plot(depths, va.mean(1), "o-", label="validation")
plt.xlabel("tree depth"); plt.ylabel("accuracy"); plt.legend(); plt.title("bias (left) vs variance (right)"); plt.show()

# %% [markdown]
# **Data size vs capacity.** With little data, simple models win; with lots, flexible ones overtake. Learning curves
# show where you are, and whether more data would help.

# %%
from sklearn.model_selection import learning_curve
for name, m in [("logistic", sc(LogisticRegression(max_iter=3000))), ("random forest", RandomForestClassifier(200, random_state=0, n_jobs=-1)), ("MLP", sc(MLPClassifier(hidden_layer_sizes=(128,), max_iter=600, random_state=0)))]:
    sizes, _, va = learning_curve(m, Xi, yi, train_sizes=np.linspace(0.05, 1, 6), cv=5); plt.plot(sizes, va.mean(1), "o-", label=name)
plt.xlabel("training examples"); plt.ylabel("validation accuracy"); plt.legend(); plt.title("digits: which model wins depends on how much data you have"); plt.show()

# %% [markdown]
# **Accuracy vs interpretability.** A depth-3 tree or a linear model can be explained to a regulator; a boosted
# ensemble usually can't, without tools like SHAP. **Latency and cost vs quality.** kNN and SVMs are slow to predict
# at scale; a big net may need a GPU per request. **No free lunch.** Averaged over all possible problems, no model
# beats any other: every win comes from an assumption that matches the data. That's why you test, not guess.

# %%
for tname, *_ in TASKS[:1]:
    for n in names: print(f"{n:14s} fit+score time on '{tname}': {table[(tname, n)][1]:.2f}s")

# %% [markdown]
# ## 4. A first-try guide
#
# | Your data | Try first | Then | Why |
# |---|---|---|---|
# | Tabular, thousands to millions of rows | gradient boosting (LightGBM, XGBoost, CatBoost) | linear model as a sanity check |
| Tabular, a few hundred clean rows | regularised linear model or SVM | boosting, compared by cross-validation | little data favours simple models | trees handle mixed types, missing values, interactions |
# | Tabular, must explain | logistic/linear or a shallow tree | boosting + SHAP if allowed | readable weights and rules |
# | Images | a pretrained CNN or ViT, fine-tuned | train from scratch only with lots of data | transfer beats from-scratch on small data |
# | Text | a pretrained transformer (embeddings + linear head, or fine-tune, or prompt an LLM) | TF-IDF + logistic as a baseline | language needs pretraining |
# | Time series | seasonal naive and gradient boosting on lag features | a sequence model if data is large | most forecasting wins come from good features |
# | Graphs | a GNN | node features + boosting | structure matters |
# | No labels | k-means / PCA to explore | self-supervised embeddings | find structure first |
# | Actions and rewards | a bandit, then RL | imitation learning if you have demonstrations | the signal is reward |
# | Generate new data | a pretrained diffusion model or LLM | fine-tune with LoRA | nobody trains these from scratch on a laptop |

# %% [markdown]
# ### Exercise 3: defend a choice
# Return a dict mapping each scenario to one of: "boosting", "linear", "pretrained_cnn", "pretrained_transformer", "kmeans", "bandit".

# %%
def ex3():
    if SOLUTIONS: return {"credit_scoring_regulator_needs_reasons": "linear", "500k_rows_of_sales_data_best_accuracy": "boosting", "3000_labelled_skin_lesion_photos": "pretrained_cnn",
                          "classify_support_tickets_by_topic": "pretrained_transformer", "group_shoppers_with_no_labels": "kmeans", "choose_which_headline_to_show": "bandit"}
    ...
check("exercise 3", ex3(), {"credit_scoring_regulator_needs_reasons": "linear", "500k_rows_of_sales_data_best_accuracy": "boosting", "3000_labelled_skin_lesion_photos": "pretrained_cnn",
                            "classify_support_tickets_by_topic": "pretrained_transformer", "group_shoppers_with_no_labels": "kmeans", "choose_which_headline_to_show": "bandit"})

# %% [markdown]
# **Where to go next:** the website's chapters go deep on each family, and the projects from notebook 01 onward
# build them from scratch: linear regression (01), autograd (02), CNNs (03), VAEs (04), tokenizers (05), a GPT (06)…
