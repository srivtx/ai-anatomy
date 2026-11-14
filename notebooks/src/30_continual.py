# %% [markdown]
# # 30 · Catastrophic forgetting, replay and EWC
# Learn five tasks one after another (MNIST with a different fixed pixel permutation per task), then fill
# in the accuracy matrix for naive fine-tuning, replay and EWC. Chapter: **Learning without forgetting**.
#
# **You're done when** you have an accuracy matrix (task trained × task tested) for each method.

# %%
import os, gzip, urllib.request
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, matplotlib.pyplot as plt
QUICK = os.environ.get("QUICK") == "1"
dev = "cuda" if torch.cuda.is_available() else "cpu"
def load(f):
    if not os.path.exists(f): urllib.request.urlretrieve("https://ossci-datasets.s3.amazonaws.com/mnist/" + f, f)
    return gzip.open(f).read()
X = torch.tensor(np.frombuffer(load("train-images-idx3-ubyte.gz"), np.uint8, offset=16).reshape(-1, 784) / 255.0, dtype=torch.float32)
Y = torch.tensor(np.frombuffer(load("train-labels-idx1-ubyte.gz"), np.uint8, offset=8)).long()
Xte = torch.tensor(np.frombuffer(load("t10k-images-idx3-ubyte.gz"), np.uint8, offset=16).reshape(-1, 784) / 255.0, dtype=torch.float32)
Yte = torch.tensor(np.frombuffer(load("t10k-labels-idx1-ubyte.gz"), np.uint8, offset=8)).long()
n_tr = 3000 if QUICK else 20000
K = 3 if QUICK else 5
g = torch.Generator().manual_seed(0)
perms = [torch.arange(784)] + [torch.randperm(784, generator=g) for _ in range(K - 1)]
tasks = [(X[:n_tr][:, p].to(dev), Y[:n_tr].to(dev), Xte[:2000][:, p].to(dev), Yte[:2000].to(dev)) for p in perms]

def make(): torch.manual_seed(0); return nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10)).to(dev)
acc = lambda m, x, y: (m(x).argmax(1) == y).float().mean().item()

def fisher(m, x, y, n=500):
    F_ = [torch.zeros_like(p) for p in m.parameters()]
    for i in range(0, n, 50):
        m.zero_grad(); F.cross_entropy(m(x[i:i + 50]), y[i:i + 50]).backward()
        for f, p in zip(F_, m.parameters()): f += p.grad.detach() ** 2 / (n / 50)
    return F_

def run(method, epochs=1 if QUICK else 3, replay_per_task=200, lam=2000.0):
    m = make(); opt = torch.optim.Adam(m.parameters(), 1e-3); M = np.zeros((K, K)); memory, anchors = [], []
    for t, (x, y, _, _) in enumerate(tasks):
        for ep in range(epochs):
            for idx in torch.randperm(len(x), device=dev).split(64):
                xb, yb = x[idx], y[idx]
                if method == "replay" and memory:                        # mix in old examples
                    mx = torch.cat([a for a, _ in memory]); my = torch.cat([b for _, b in memory]); j = torch.randint(len(mx), (32,), device=dev)
                    xb, yb = torch.cat([xb, mx[j]]), torch.cat([yb, my[j]])
                loss = F.cross_entropy(m(xb), yb)
                if method == "ewc":
                    for F_, old in anchors: loss = loss + lam / 2 * sum((f * (p - o) ** 2).sum() for f, p, o in zip(F_, m.parameters(), old))
                opt.zero_grad(); loss.backward(); opt.step()
        if method == "replay": memory.append((x[:replay_per_task], y[:replay_per_task]))
        if method == "ewc": anchors.append((fisher(m, x, y), [p.detach().clone() for p in m.parameters()]))
        with torch.no_grad(): M[t] = [acc(m, xt, yt) for _, _, xt, yt in tasks]
    return M

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a, method in zip(ax, ["naive", "replay", "ewc"]):
    M = run(method)
    a.imshow(M, vmin=0, vmax=1, cmap="viridis"); a.set_title(f"{method}: avg final {M[-1].mean():.0%}")
    for i in range(K):
        for j in range(K): a.text(j, i, f"{M[i, j]:.0%}" if j <= i else "", ha="center", va="center", color="w", fontsize=8)
    a.set_xlabel("task tested"); a.set_ylabel("after training task")
    print(f"{method:7s} final accuracy per task:", " ".join(f"{v:.0%}" for v in M[-1]))
plt.show()

# %% [markdown]
# **What to expect.** Naive fine-tuning keeps the newest task and loses much of the older ones. Replaying 200
# stored examples per task recovers most of it. EWC helps too, with a λ that needs tuning: too small and it
# forgets, too large and new tasks stop being learned.
