# %% [markdown]
# # 04 · A variational autoencoder with a 2D latent
# Encode digits into a 2-dimensional latent space you can plot, sample from and walk through, and see what
# β (the KL weight) does. Chapter: **Autoencoders and VAEs**.
#
# **You're done when** you have a grid of decoded samples across the latent plane and can explain the
# blur vs sharpness trade-off.

# %%
import os, gzip, urllib.request
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, matplotlib.pyplot as plt
QUICK = os.environ.get("QUICK") == "1"
dev = "cuda" if torch.cuda.is_available() else "cpu"

def mnist(split="train"):
    f = {"train": "train-images-idx3-ubyte.gz", "test": "t10k-images-idx3-ubyte.gz"}[split]
    lf = f.replace("images-idx3", "labels-idx1")
    for name in (f, lf):
        if not os.path.exists(name): urllib.request.urlretrieve("https://ossci-datasets.s3.amazonaws.com/mnist/" + name, name)
    x = np.frombuffer(gzip.open(f).read(), np.uint8, offset=16).reshape(-1, 784) / 255.0
    y = np.frombuffer(gzip.open(lf).read(), np.uint8, offset=8)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y)

x, y = mnist()
if QUICK: x, y = x[:5000], y[:5000]
x = x.to(dev)

class VAE(nn.Module):
    def __init__(self, z=2):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        self.mu, self.logvar = nn.Linear(256, z), nn.Linear(256, z)
        self.dec = nn.Sequential(nn.Linear(z, 256), nn.ReLU(), nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 784))
    def forward(self, x):
        h = self.enc(x); mu, logvar = self.mu(h), self.logvar(h)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()          # reparameterisation trick
        return self.dec(z), mu, logvar

def loss_fn(logits, x, mu, logvar, beta):
    rec = F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / len(x)
    kl = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum() / len(x)    # KL(q(z|x) || N(0, I))
    return rec + beta * kl, rec, kl

def train(beta=1.0, epochs=10 if not QUICK else 2, bs=128):
    torch.manual_seed(0); m = VAE().to(dev); opt = torch.optim.Adam(m.parameters(), 1e-3)
    for ep in range(epochs):
        perm = torch.randperm(len(x), device=dev)
        for i in range(0, len(x), bs):
            xb = x[perm[i:i + bs]]
            logits, mu, logvar = m(xb)
            loss, rec, kl = loss_fn(logits, xb, mu, logvar, beta)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"β={beta} epoch {ep}: recon {rec.item():.1f}  KL {kl.item():.2f}")
    return m

vae = train(1.0)

# %% [markdown]
# ## The latent plane, coloured by digit

# %%
with torch.no_grad(): mu = vae.mu(vae.enc(x[:5000])).cpu()
plt.figure(figsize=(6, 6)); plt.scatter(mu[:, 0], mu[:, 1], c=y[:5000], cmap="tab10", s=3); plt.colorbar(); plt.title("Encoder means (no labels were used)"); plt.show()

# %% [markdown]
# ## Decode a grid across the plane

# %%
g = torch.linspace(-2.5, 2.5, 15)
zz = torch.stack(torch.meshgrid(g, g.flip(0), indexing="xy"), -1).reshape(-1, 2).to(dev)
with torch.no_grad(): imgs = torch.sigmoid(vae.dec(zz)).cpu().reshape(15, 15, 28, 28)
plt.figure(figsize=(8, 8)); plt.imshow(imgs.permute(0, 2, 1, 3).reshape(15 * 28, 15 * 28), cmap="gray"); plt.axis("off"); plt.title("Every point in the plane decodes to a digit-like image"); plt.show()

# %% [markdown]
# ## What β does
# β < 1 lets the encoder use the latent space freely: sharper reconstructions, a lumpier latent space with
# holes that decode to garbage. β > 1 pushes everything towards N(0, I): smoother, more sample-able, blurrier.

# %%
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
for a, beta in zip(ax, [0.1, 1.0, 4.0]):
    m = vae if beta == 1.0 else train(beta)
    with torch.no_grad(): s = torch.sigmoid(m.dec(torch.randn(64, 2, device=dev))).cpu().reshape(8, 8, 28, 28)
    a.imshow(s.permute(0, 2, 1, 3).reshape(8 * 28, 8 * 28), cmap="gray"); a.set_title(f"samples, β = {beta}"); a.axis("off")
plt.show()
