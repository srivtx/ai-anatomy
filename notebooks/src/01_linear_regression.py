# %% [markdown]
# # 01 · Linear regression three ways
# Fit a line with the closed-form solution, with gradient descent, and with stochastic gradient descent,
# all in NumPy; then the same model in PyTorch. Chapter: **Gradient descent**.
#
# **You're done when** your gradient-descent weights match the closed form to 4 decimals and you can
# explain why the loss curve bends.

# %%
import os
import numpy as np
import matplotlib.pyplot as plt

QUICK = os.environ.get("QUICK") == "1"
rng = np.random.default_rng(0)
n, d = 500, 3
X = rng.normal(size=(n, d))
w_true, b_true = np.array([2.0, -1.0, 0.5]), 0.7
y = X @ w_true + b_true + rng.normal(scale=0.3, size=n)
Xb = np.hstack([X, np.ones((n, 1))])  # append a column of ones for the bias

# %% [markdown]
# ## 1. Closed form
# Setting the gradient of the mean squared error to zero gives the normal equations
# $(X^\top X) w = X^\top y$. Solve them directly (never invert the matrix explicitly).

# %%
w_closed = np.linalg.solve(Xb.T @ Xb, Xb.T @ y)
print("closed form:", w_closed.round(4))

# %% [markdown]
# ## 2. Gradient descent
# Loss $L(w) = \frac{1}{n}\lVert Xw - y\rVert^2$, gradient $\nabla L = \frac{2}{n} X^\top (Xw - y)$.
# The largest stable learning rate is $2/\lambda_{max}$ of the Hessian $\frac{2}{n}X^\top X$.

# %%
H = 2 / n * Xb.T @ Xb
lr_max = 2 / np.linalg.eigvalsh(H).max()
print(f"largest stable learning rate ≈ {lr_max:.3f}")


def gd(lr, steps=300):
    w, hist = np.zeros(d + 1), []
    for _ in range(steps):
        r = Xb @ w - y
        hist.append((r**2).mean())
        w -= lr * 2 / n * Xb.T @ r
    return w, hist


w_gd, hist = gd(0.5 * lr_max)
print(
    "gradient descent:", w_gd.round(4), " max |diff| vs closed form:", np.abs(w_gd - w_closed).max()
)

for frac in [0.1, 0.5, 0.95, 1.05]:
    _, h = gd(frac * lr_max, 60)
    plt.semilogy(h, label=f"lr = {frac:.2f} × max")

plt.xlabel("step")
plt.ylabel("MSE")
plt.legend()
plt.title("Too small is slow; above the max it diverges")
plt.show()

# %% [markdown]
# **Why the curve bends**: the error in each eigen-direction of $X^\top X$ shrinks by a constant factor
# $|1 - \text{lr}\cdot\lambda_i|$ per step. Fast directions vanish first (steep drop), then the slowest
# direction dominates (long flat tail).
#
# ## 3. Stochastic gradient descent
# Use a random mini-batch per step. Noisier, but each step is far cheaper on big data.


# %%
def sgd(lr=0.05, batch=16, epochs=20 if not QUICK else 5):
    w = np.zeros(d + 1)
    for _ in range(epochs):
        for idx in np.array_split(rng.permutation(n), n // batch):
            r = Xb[idx] @ w - y[idx]
            w -= lr * 2 / len(idx) * Xb[idx].T @ r
    return w


w_sgd = sgd()
print("SGD:", w_sgd.round(3), " (close, but jitters around the optimum unless the LR decays)")

# %% [markdown]
# ## 4. The same in PyTorch

# %%
import torch

Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
model = torch.nn.Linear(d, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.1)

for step in range(300):
    loss = ((model(Xt).squeeze(-1) - yt) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()

print(
    "torch:", model.weight.data.numpy().round(4), model.bias.data.numpy().round(4),
    f"loss {loss.item():.4f}"
)
assert np.allclose(w_gd, w_closed, atol=1e-3)
