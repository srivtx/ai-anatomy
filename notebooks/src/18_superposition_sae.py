# %% [markdown]
# # 18 · Superposition and sparse autoencoders
# Reproduce Anthropic's toy model of superposition at three sparsities, then train a sparse autoencoder on
# a model's hidden activations and check it recovers the true features. Chapter: **Looking inside: superposition**.
#
# **You're done when** you have feature-geometry plots at three sparsities and SAE features that match the
# true ones with cosine > 0.9.

# %%
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

QUICK = os.environ.get("QUICK") == "1"


def toy(n_feat=5, m=2, sparsity=0.9, steps=3000 if not QUICK else 800, seed=0, decay=0.7):
    torch.manual_seed(seed)
    W = nn.Parameter(torch.randn(m, n_feat) * 0.3)
    b = nn.Parameter(torch.zeros(n_feat))
    imp = decay ** torch.arange(n_feat).float()
    opt = torch.optim.Adam([W, b], 1e-2)
    for _ in range(steps):
        x = torch.rand(1024, n_feat) * (torch.rand(1024, n_feat) > sparsity)
        out = F.relu(x @ W.T @ W + b)
        loss = (imp * (x - out) ** 2).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return W.detach(), b.detach()


fig, ax = plt.subplots(1, 3, figsize=(12, 4))

for a, s in zip(ax, [0.0, 0.7, 0.97]):
    W, b = toy(sparsity=s)
    for i in range(W.shape[1]):
        a.arrow(0, 0, W[0, i].item(), W[1, i].item(), head_width=0.05, color=plt.cm.viridis(i / 5))
        a.annotate(f"f{i + 1}", (W[0, i] * 1.15, W[1, i] * 1.15))
    rep = (W.norm(dim=0) > 0.5).sum().item()
    a.set_title(f"sparsity {s:.0%}: {rep} features in 2 dims")
    a.set_xlim(-1.5, 1.5)
    a.set_ylim(-1.5, 1.5)
    a.set_aspect("equal")

plt.show()

# %% [markdown]
# ## A sparse autoencoder recovers the features
# Train a bigger toy (20 sparse features squeezed into 8 dimensions), record its hidden activations, then
# train an SAE with 40 latents and an L1 penalty. Each decoder column should line up with one true feature.

# %%
n_feat, m = 20, 8
W, b = toy(n_feat=n_feat, m=m, sparsity=0.95, decay=0.95)
x = torch.rand(50_000, n_feat) * (torch.rand(50_000, n_feat) > 0.95)
acts = x @ W.T  # the model's hidden activations


class SAE(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.enc = nn.Linear(d, k)
        self.dec = nn.Linear(k, d)

    def forward(self, a):
        f = F.relu(self.enc(a - self.dec.bias))
        return self.dec(f), f


sae = SAE(m, 40)
opt = torch.optim.Adam(sae.parameters(), 3e-3)

for step in range(3000 if not QUICK else 1500):
    a = acts[torch.randint(0, len(acts), (1024,))]
    rec, f = sae(a)
    loss = F.mse_loss(rec, a) + 3e-3 * f.abs().sum(-1).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    with torch.no_grad():
        sae.dec.weight /= sae.dec.weight.norm(dim=0, keepdim=True)  # unit-norm decoder columns

true_dirs = F.normalize(W, dim=0)  # each true feature's direction in hidden space
learned = F.normalize(sae.dec.weight.detach(), dim=0)
cos = (true_dirs.T @ learned).abs()  # (20 true, 40 learned)
best = cos.max(1).values
represented = W.norm(dim=0) > 0.5
print(f"true features the model represents: {int(represented.sum())} of {n_feat}")
print(
    f"best SAE match per represented feature: mean cosine {best[represented].mean():.3f}, min {best[represented].min():.3f}"
)
plt.figure(figsize=(10, 4))
plt.imshow(cos, aspect="auto", cmap="magma")
plt.xlabel("SAE latent")
plt.ylabel("true feature")
plt.colorbar(label="|cosine|")
plt.show()
