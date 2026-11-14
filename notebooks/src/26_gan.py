# %% [markdown]
# # 26 · GANs: mode collapse and how to fight it
# A GAN on a ring of 8 Gaussians, where you can count how many modes it covers, then on MNIST.
# Chapter: **GANs and normalising flows**.
#
# **You're done when** you have a plot of mode coverage over training and recognisable generated digits.

# %%
import os, gzip, urllib.request, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, matplotlib.pyplot as plt
QUICK = os.environ.get("QUICK") == "1"
dev = "cuda" if torch.cuda.is_available() else "cpu"

centres = torch.tensor([[math.cos(2 * math.pi * k / 8), math.sin(2 * math.pi * k / 8)] for k in range(8)]) * 2
def ring(n): return centres[torch.randint(0, 8, (n,))] + 0.05 * torch.randn(n, 2)
def coverage(x):
    d = torch.cdist(x, centres); near = d.min(1).values < 0.2
    modes = d.argmin(1)[near].unique().numel()
    return modes, near.float().mean().item()

def mlp(i, o, h=128): return nn.Sequential(nn.Linear(i, h), nn.LeakyReLU(0.2), nn.Linear(h, h), nn.LeakyReLU(0.2), nn.Linear(h, o))

def train_gan(loss="standard", steps=3000, seed=0, d_steps=1):
    torch.manual_seed(seed)
    G, D = mlp(8, 2), mlp(2, 1)
    og, od = torch.optim.Adam(G.parameters(), 1e-3, betas=(0.5, 0.999)), torch.optim.Adam(D.parameters(), 1e-3, betas=(0.5, 0.999))
    hist = []
    for step in range(steps):
        for _ in range(d_steps):
            real, fake = ring(256), G(torch.randn(256, 8)).detach()
            if loss == "standard":
                ld = F.softplus(-D(real)).mean() + F.softplus(D(fake)).mean()
            else:  # WGAN-GP: critic with a gradient penalty
                a = torch.rand(256, 1); mix = (a * real + (1 - a) * fake).requires_grad_(True)
                g = torch.autograd.grad(D(mix).sum(), mix, create_graph=True)[0]
                ld = D(fake).mean() - D(real).mean() + 10 * ((g.norm(dim=1) - 1) ** 2).mean()
            od.zero_grad(); ld.backward(); od.step()
        fake = G(torch.randn(256, 8))
        lg = F.softplus(-D(fake)).mean() if loss == "standard" else -D(fake).mean()
        og.zero_grad(); lg.backward(); og.step()
        if step % 100 == 0: hist.append((step, *coverage(G(torch.randn(2000, 8)).detach())))
    return G, hist

steps = 1500 if QUICK else 5000
runs = {"standard, seed 0": train_gan("standard", steps, 0), "standard, seed 1": train_gan("standard", steps, 1),
        "WGAN-GP (5 critic steps)": train_gan("wgan", steps, 0, d_steps=5)}
fig, ax = plt.subplots(1, len(runs) + 1, figsize=(18, 4))
for a, (name, (G, hist)) in zip(ax, runs.items()):
    s = G(torch.randn(2000, 8)).detach(); a.scatter(*s.T, s=2); a.scatter(*centres.T, c="r", marker="x"); a.set_title(name); a.set_xlim(-3, 3); a.set_ylim(-3, 3)
    ax[-1].plot([h[0] for h in hist], [h[1] for h in hist], label=name)
ax[-1].set_xlabel("step"); ax[-1].set_ylabel("modes covered (of 8)"); ax[-1].legend(); plt.show()
for name, (G, hist) in runs.items(): print(f"{name}: covers {hist[-1][1]} of 8 modes")

# %% [markdown]
# ## A DCGAN on MNIST

# %%
f = "train-images-idx3-ubyte.gz"
if not os.path.exists(f): urllib.request.urlretrieve("https://ossci-datasets.s3.amazonaws.com/mnist/" + f, f)
X = torch.tensor(np.frombuffer(gzip.open(f).read(), np.uint8, offset=16).reshape(-1, 1, 28, 28) / 127.5 - 1, dtype=torch.float32)
if QUICK: X = X[:2000]
X = X.to(dev)
G = nn.Sequential(nn.Linear(64, 128 * 7 * 7), nn.Unflatten(1, (128, 7, 7)), nn.BatchNorm2d(128), nn.ReLU(),
                  nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Tanh()).to(dev)
D = nn.Sequential(nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2), nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2), nn.Flatten(), nn.Linear(128 * 7 * 7, 1)).to(dev)
og, od = torch.optim.Adam(G.parameters(), 2e-4, betas=(0.5, 0.999)), torch.optim.Adam(D.parameters(), 2e-4, betas=(0.5, 0.999))
for step in range(100 if QUICK else 6000):
    real = X[torch.randint(0, len(X), (128,), device=dev)]; fake = G(torch.randn(128, 64, device=dev))
    ld = F.softplus(-D(real)).mean() + F.softplus(D(fake.detach())).mean()
    od.zero_grad(); ld.backward(); od.step()
    lg = F.softplus(-D(fake)).mean()
    og.zero_grad(); lg.backward(); og.step()
    if step % 1000 == 0: print(step, f"D {ld.item():.3f}  G {lg.item():.3f}")
with torch.no_grad(): s = G(torch.randn(64, 64, device=dev)).cpu()
plt.figure(figsize=(6, 6)); plt.imshow(s.reshape(8, 8, 28, 28).permute(0, 2, 1, 3).reshape(224, 224), cmap="gray"); plt.axis("off"); plt.show()
