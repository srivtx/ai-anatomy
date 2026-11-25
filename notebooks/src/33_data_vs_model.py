# %% [markdown]
# # 33 · Data vs model size
# Corrupt a fraction of training labels, then compare two ways to spend effort: a bigger model, or cleaner
# labels. Then find the bad labels automatically. Chapter: **Data beats architecture**.
#
# **You're done when** you have a table of model size × data quality → test accuracy.

# %%
import os
import gzip
import urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

QUICK = os.environ.get("QUICK") == "1"
dev = "cuda" if torch.cuda.is_available() else "cpu"


def load(f):
    if not os.path.exists(f):
        urllib.request.urlretrieve("https://ossci-datasets.s3.amazonaws.com/mnist/" + f, f)
    return gzip.open(f).read()


X = torch.tensor(
    np.frombuffer(load("train-images-idx3-ubyte.gz"), np.uint8, offset=16).reshape(-1, 784) / 255.0,
    dtype=torch.float32,
)
Y = torch.tensor(np.frombuffer(load("train-labels-idx1-ubyte.gz"), np.uint8, offset=8)).long()
Xte = torch.tensor(
    np.frombuffer(load("t10k-images-idx3-ubyte.gz"), np.uint8, offset=16).reshape(-1, 784) / 255.0,
    dtype=torch.float32,
).to(dev)
Yte = (
    torch.tensor(np.frombuffer(load("t10k-labels-idx1-ubyte.gz"), np.uint8, offset=8))
    .long()
    .to(dev)
)
n = 3000 if QUICK else 10000
X, Y = X[:n].to(dev), Y[:n].to(dev)
g = torch.Generator().manual_seed(0)
noisy = Y.clone()
flip = torch.rand(n, generator=g).to(dev) < 0.3
noisy[flip] = (
    Y[flip] + torch.randint(1, 10, (int(flip.sum()),), generator=g).to(dev)
) % 10  # 30% wrong labels


def train(width, labels, epochs=10 if not QUICK else 4, seed=0):
    torch.manual_seed(seed)
    m = nn.Sequential(
        nn.Linear(784, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 10)
    ).to(dev)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    for _ in range(epochs):
        for idx in torch.randperm(n, device=dev).split(64):
            loss = F.cross_entropy(m(X[idx]), labels[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return (m(Xte).argmax(1) == Yte).float().mean().item(), m


print(f"{'':12s}" + "".join(f"{w:>10d}" for w in [64, 256, 1024]) + "   (hidden width)")

for name, lab in [("30% noisy", noisy), ("clean", Y)]:
    print(f"{name:12s}" + "".join(f"{train(w, lab)[0]:>10.1%}" for w in [64, 256, 1024]))

# %% [markdown]
# ## Find the bad labels automatically (confident learning, simplified)
# Train on half, score the other half, swap. Labels the model finds very unlikely are suspicious.

# %%
idx = torch.randperm(n, generator=g).to(dev)
halves = [idx[: n // 2], idx[n // 2 :]]
score = torch.zeros(n, device=dev)

for a, b in [(0, 1), (1, 0)]:
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10)).to(dev)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    for _ in range(3):
        for j in halves[a][torch.randperm(len(halves[a]), device=dev)].split(64):
            loss = F.cross_entropy(m(X[j]), noisy[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        score[halves[b]] = (
            F.softmax(m(X[halves[b]]), 1).gather(1, noisy[halves[b]][:, None]).squeeze(1)
        )

k = int(flip.sum())
suspects = score.argsort()[:k]
print(
    f"of the {k} lowest-scoring labels, {flip[suspects].float().mean():.0%} are truly wrong (base rate 30%)"
)
fixed = noisy.clone()
fixed[suspects] = -1  # drop suspects
keep = fixed >= 0
X_all, Y_all = X, noisy
X, noisy_kept, n_keep = X[keep], noisy[keep], int(keep.sum())
torch.manual_seed(0)
m = nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10)
).to(dev)
opt = torch.optim.Adam(m.parameters(), 1e-3)

for _ in range(10 if not QUICK else 4):
    for j in torch.randperm(n_keep, device=dev).split(64):
        loss = F.cross_entropy(m(X[j]), noisy_kept[j])
        opt.zero_grad()
        loss.backward()
        opt.step()

with torch.no_grad():
    print(f"after dropping suspects: test accuracy {(m(Xte).argmax(1) == Yte).float().mean():.1%}")
