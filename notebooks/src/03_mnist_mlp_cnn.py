# %% [markdown]
# # 03 · MNIST: MLP vs CNN, with ablations
# Train an MLP and a small CNN on handwritten digits, then run a proper ablation (normalisation, init,
# dropout) over three seeds. Chapters: **Training tricks**, **CNNs**, **Research method**.
#
# **You're done when** the CNN is above 99% test accuracy and you have an ablation table with mean ± std.

# %%
import os, gzip, urllib.request, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, matplotlib.pyplot as plt
QUICK = os.environ.get("QUICK") == "1"
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

def load_mnist():
    base = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    out = {}
    for k, f in [("xtr", "train-images-idx3-ubyte.gz"), ("ytr", "train-labels-idx1-ubyte.gz"), ("xte", "t10k-images-idx3-ubyte.gz"), ("yte", "t10k-labels-idx1-ubyte.gz")]:
        if not os.path.exists(f): urllib.request.urlretrieve(base + f, f)
        raw = gzip.open(f).read()
        out[k] = np.frombuffer(raw, np.uint8, offset=16 if k[0] == "x" else 8)
    xtr = torch.tensor(out["xtr"].reshape(-1, 1, 28, 28), dtype=torch.float32) / 255
    xte = torch.tensor(out["xte"].reshape(-1, 1, 28, 28), dtype=torch.float32) / 255
    return xtr, torch.tensor(out["ytr"]).long(), xte, torch.tensor(out["yte"]).long()

xtr, ytr, xte, yte = load_mnist()
if QUICK: xtr, ytr, xte, yte = xtr[:4000], ytr[:4000], xte[:1000], yte[:1000]
mean, std = xtr.mean(), xtr.std()
xtr, xte = ((xtr - mean) / std).to(dev), ((xte - mean) / std).to(dev)
ytr, yte = ytr.to(dev), yte.to(dev)
print(xtr.shape, xte.shape, dev)

# %%
class MLP(nn.Module):
    def __init__(self, norm=True, drop=0.1):
        super().__init__()
        layers, d = [nn.Flatten()], 784
        for h in [512, 256]:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h) if norm else nn.Identity(), nn.ReLU(), nn.Dropout(drop)]; d = h
        self.net = nn.Sequential(*layers, nn.Linear(d, 10))
    def forward(self, x): return self.net(x)

class CNN(nn.Module):
    def __init__(self, norm=True, drop=0.1):
        super().__init__()
        N = (lambda c: nn.BatchNorm2d(c)) if norm else (lambda c: nn.Identity())
        self.f = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), N(32), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), N(32), nn.ReLU(), nn.MaxPool2d(2),
                               nn.Conv2d(32, 64, 3, padding=1), N(64), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), N(64), nn.ReLU(), nn.MaxPool2d(2))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(drop), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(drop), nn.Linear(128, 10))
    def forward(self, x): return self.head(self.f(x))

def init_bad(m):
    # "bad init": tiny weights, so signals shrink layer after layer
    if isinstance(m, (nn.Linear, nn.Conv2d)): nn.init.normal_(m.weight, std=0.01); nn.init.zeros_(m.bias)

def run(make, epochs=3, seed=0, bad_init=False, lr=1e-3, bs=128):
    torch.manual_seed(seed)
    model = make().to(dev)
    if bad_init: model.apply(init_bad)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * (len(xtr) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr * 3, total_steps=steps)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr), device=dev)
        for i in range(0, len(xtr) - bs + 1, bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(model(xtr[idx]), ytr[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(xte[i:i + 1000]).argmax(1) for i in range(0, len(xte), 1000)])
    return (pred == yte).float().mean().item(), model

E = 1 if QUICK else 3
acc_mlp, _ = run(MLP, E); acc_cnn, cnn = run(CNN, E)
print(f"MLP {acc_mlp:.2%}   CNN {acc_cnn:.2%}")

# %% [markdown]
# ## Ablations over seeds
# One run is an anecdote. Change one thing at a time, run three seeds each, and report mean ± std.

# %%
configs = {
    "CNN baseline": dict(make=CNN),
    "no BatchNorm": dict(make=lambda: CNN(norm=False)),
    "bad init (std 0.01)": dict(make=lambda: CNN(norm=False), bad_init=True),
    "no dropout": dict(make=lambda: CNN(drop=0.0)),
    "MLP baseline": dict(make=MLP),
}
seeds = [0, 1] if QUICK else [0, 1, 2]
rows = []
for name, cfg in configs.items():
    accs = [run(epochs=E, seed=s, **cfg)[0] for s in seeds]
    rows.append((name, np.mean(accs), np.std(accs)))
    print(f"{name:22s} {np.mean(accs):.2%} ± {np.std(accs):.2%}")

names, means, sds = zip(*rows)
plt.barh(names, [m * 100 for m in means], xerr=[s * 100 for s in sds]); plt.xlim(min(means) * 100 - 1, 100); plt.xlabel("test accuracy (%)"); plt.title("Ablations, mean ± std over seeds"); plt.show()

# %% [markdown]
# ## Look at the mistakes

# %%
with torch.no_grad(): p = cnn(xte[:2000]).argmax(1)
wrong = (p != yte[:2000]).nonzero().flatten()[:16].cpu()
fig, ax = plt.subplots(2, 8, figsize=(12, 3.5))
for a, i in zip(ax.flat, wrong):
    a.imshow(xte[i, 0].cpu(), cmap="gray"); a.set_title(f"true {yte[i].item()} / said {p[i].item()}", fontsize=8); a.axis("off")
plt.suptitle("Most errors are digits a person might also hesitate over"); plt.show()
