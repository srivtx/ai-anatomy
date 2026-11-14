# %% [markdown]
# # 12 · Diffusion and flow matching
# DDPM on a 2D shape, then on MNIST digits; then flow matching, and how sample quality depends on the
# number of steps. Chapter: **Diffusion**.
#
# **You're done when** generated digits look like digits and you have a quality-vs-steps comparison.

# %%
import os, gzip, urllib.request, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, matplotlib.pyplot as plt
QUICK = os.environ.get("QUICK") == "1"
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# %% [markdown]
# ## 1. A 2D shape
# Data: points on a spiral. The network predicts the noise ε that was added at step t.

# %%
def spiral(n):
    t = torch.rand(n) * 3 * math.pi + 1
    return torch.stack([t * torch.cos(t), t * torch.sin(t)], 1) / 6 + 0.02 * torch.randn(n, 2)

T = 200
betas = torch.linspace(1e-4, 0.02, T); alphas = 1 - betas; abar = torch.cumprod(alphas, 0)

class Eps(nn.Module):
    def __init__(self, d=2, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d + 16, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, d))
    def forward(self, x, t):                       # t in [0, 1]
        f = torch.arange(8, device=x.device) * 1.0
        emb = torch.cat([torch.sin(t[:, None] * 2 ** f * math.pi), torch.cos(t[:, None] * 2 ** f * math.pi)], 1)
        return self.net(torch.cat([x, emb], 1))

model = Eps(); opt = torch.optim.Adam(model.parameters(), 2e-3)
for step in range(1500 if QUICK else 6000):
    x0 = spiral(512); t = torch.randint(0, T, (512,)); eps = torch.randn_like(x0)
    xt = abar[t].sqrt()[:, None] * x0 + (1 - abar[t]).sqrt()[:, None] * eps    # jump straight to step t
    loss = F.mse_loss(model(xt, t / T), eps)
    opt.zero_grad(); loss.backward(); opt.step()
print("final loss", loss.item())

@torch.no_grad()
def ddpm_sample(model, n, d=2):
    x = torch.randn(n, d)
    for t in reversed(range(T)):
        e = model(x, torch.full((n,), t / T))
        x = (x - betas[t] / (1 - abar[t]).sqrt() * e) / alphas[t].sqrt()
        if t > 0: x += betas[t].sqrt() * torch.randn_like(x)
    return x

@torch.no_grad()
def ddim_sample(model, n, steps, d=2):
    x = torch.randn(n, d); ts = torch.linspace(T - 1, 0, steps).long()
    for i, t in enumerate(ts):
        e = model(x, torch.full((n,), t.item() / T))
        x0 = (x - (1 - abar[t]).sqrt() * e) / abar[t].sqrt()
        a_next = abar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0)
        x = a_next.sqrt() * x0 + (1 - a_next).sqrt() * e          # deterministic DDIM step
    return x

fig, ax = plt.subplots(1, 4, figsize=(16, 4))
ax[0].scatter(*spiral(2000).T, s=2); ax[0].set_title("data")
ax[1].scatter(*ddpm_sample(model, 2000).T, s=2); ax[1].set_title(f"DDPM, {T} steps")
for a, k in zip(ax[2:], [20, 5]): a.scatter(*ddim_sample(model, 2000, k).T, s=2); a.set_title(f"DDIM, {k} steps")
for a in ax: a.set_xlim(-2, 2); a.set_ylim(-2, 2)
plt.show()

# %% [markdown]
# ## 2. Flow matching on the same shape
# Learn a velocity field along straight lines from noise to data: $x_t = (1-t)\,x_0 + t\,x_1$, target
# $v = x_1 - x_0$. Sampling is integrating an ODE; straight paths need few steps.

# %%
flow = Eps(); opt = torch.optim.Adam(flow.parameters(), 2e-3)
for step in range(1500 if QUICK else 6000):
    x1 = spiral(512); x0 = torch.randn_like(x1); t = torch.rand(512)
    xt = (1 - t[:, None]) * x0 + t[:, None] * x1
    loss = F.mse_loss(flow(xt, t), x1 - x0)
    opt.zero_grad(); loss.backward(); opt.step()

@torch.no_grad()
def flow_sample(model, n, steps, d=2):
    x = torch.randn(n, d)
    for i in range(steps): x = x + model(x, torch.full((n,), i / steps)) / steps     # Euler
    return x

def mmd(a, b):                                     # a simple distance between sample sets (lower is better)
    k = lambda x, y: torch.exp(-torch.cdist(x, y) ** 2 / 0.1).mean()
    return (k(a, a) + k(b, b) - 2 * k(a, b)).item()

ref = spiral(1500)
for k in [2, 5, 10, 50]:
    print(f"{k:3d} steps: DDIM MMD {mmd(ddim_sample(model, 1500, k), ref):.4f}   flow MMD {mmd(flow_sample(flow, 1500, k), ref):.4f}")

# %% [markdown]
# ## 3. Digits
# Same DDPM objective with a small convolutional network. On a T4 a few minutes gives recognisable digits.

# %%
f = "train-images-idx3-ubyte.gz"
if not os.path.exists(f): urllib.request.urlretrieve("https://ossci-datasets.s3.amazonaws.com/mnist/" + f, f)
X = torch.tensor(np.frombuffer(gzip.open(f).read(), np.uint8, offset=16).reshape(-1, 1, 28, 28) / 127.5 - 1, dtype=torch.float32)
if QUICK: X = X[:2000]
X = F.pad(X, (2, 2, 2, 2), value=-1).to(dev)        # 32×32 for easy down/upsampling

class UNetish(nn.Module):
    def __init__(self, c=64):
        super().__init__()
        self.temb = nn.Sequential(nn.Linear(32, c), nn.SiLU(), nn.Linear(c, c))
        self.d1 = nn.Conv2d(1, c, 3, padding=1); self.d2 = nn.Conv2d(c, c, 4, 2, 1); self.d3 = nn.Conv2d(c, 2 * c, 4, 2, 1)
        self.mid = nn.Conv2d(2 * c, 2 * c, 3, padding=1)
        self.u3 = nn.ConvTranspose2d(2 * c, c, 4, 2, 1); self.u2 = nn.ConvTranspose2d(2 * c, c, 4, 2, 1); self.out = nn.Conv2d(2 * c, 1, 3, padding=1)
    def forward(self, x, t):
        f = torch.arange(16, device=x.device)
        e = torch.cat([torch.sin(t[:, None] * 1000 / 10000 ** (f / 16)), torch.cos(t[:, None] * 1000 / 10000 ** (f / 16))], 1)
        e = self.temb(e)[:, :, None, None]
        h1 = F.silu(self.d1(x) + e); h2 = F.silu(self.d2(h1)); h3 = F.silu(self.d3(h2)); m = F.silu(self.mid(h3))
        u = F.silu(self.u3(m)); u = F.silu(self.u2(torch.cat([u, h2], 1)))
        return self.out(torch.cat([u, h1], 1))

net = UNetish().to(dev); opt = torch.optim.Adam(net.parameters(), 2e-4)
ab = abar.to(dev)
for step in range(200 if QUICK else 8000):
    x0 = X[torch.randint(0, len(X), (128,), device=dev)]; t = torch.randint(0, T, (128,), device=dev); eps = torch.randn_like(x0)
    xt = ab[t].sqrt()[:, None, None, None] * x0 + (1 - ab[t]).sqrt()[:, None, None, None] * eps
    loss = F.mse_loss(net(xt, t / T), eps)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 1000 == 0: print(step, loss.item())

@torch.no_grad()
def sample_digits(n=36, steps=50):
    x = torch.randn(n, 1, 32, 32, device=dev); ts = torch.linspace(T - 1, 0, steps).long()
    for i, t in enumerate(ts):
        e = net(x, torch.full((n,), t.item() / T, device=dev))
        x0 = ((x - (1 - ab[t]).sqrt() * e) / ab[t].sqrt()).clamp(-1, 1)
        a_next = ab[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=dev)
        x = a_next.sqrt() * x0 + (1 - a_next).sqrt() * e
    return x.cpu()
s = sample_digits()
plt.figure(figsize=(6, 6)); plt.imshow(s.reshape(6, 6, 32, 32).permute(0, 2, 1, 3).reshape(192, 192), cmap="gray"); plt.axis("off"); plt.title("DDIM, 50 steps"); plt.show()
