# %% [markdown]
# # 55 · Model zoo, part 6: generative models
# A generative model learns p(x) so it can *produce* new samples. Six families, all trained on the same 2-D
# dataset (two moons) so you can compare them by eye and by a number: a Gaussian mixture, an autoregressive model,
# a VAE, a GAN, a normalizing flow and a diffusion model. Each one makes a different trade-off between sample
# quality, speed, stability and whether you get likelihoods.
#
# **You're done when** all checks print ✅ with SOLUTIONS = False and you have six sample plots with their
# distance to the real data. CPU, ~20 minutes.

# %%
import os, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, matplotlib.pyplot as plt
from sklearn.datasets import make_moons
SOLUTIONS = os.environ.get("SOLUTIONS") == "1"; QUICK = os.environ.get("QUICK") == "1"
torch.manual_seed(0); np.random.seed(0)
def check(name, got, want, tol=1e-3):
    if got is None or got is Ellipsis: print(f"⏳ {name}: not done yet"); return False
    try: ok = np.allclose(np.asarray(got, float), np.asarray(want, float), atol=tol)
    except Exception: ok = got == want
    print(("✅ " if ok else "❌ ") + name + ("" if ok else f": got {got}, expected {want}")); return ok
data = torch.tensor(make_moons(4000, noise=0.05, random_state=0)[0], dtype=torch.float32); data = (data - data.mean(0)) / data.std(0)
STEPS = 400 if QUICK else 4000
def energy_distance(a, b, n=1000):
    """0 when the two sample sets come from the same distribution; larger is worse."""
    a, b = a[:n], b[:n]; return (2 * torch.cdist(a, b).mean() - torch.cdist(a, a).mean() - torch.cdist(b, b).mean()).item()
mlp = lambda i, o, h=128: nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, o))
samples = {}

# %% [markdown]
# ## 1. Gaussian mixture: a sum of blobs
# p(x) = Σₖ πₖ N(x; μₖ, Σₖ). Sampling: pick a component with probability πₖ, then draw from its Gaussian.

# %%
from sklearn.mixture import GaussianMixture
gm = GaussianMixture(16, random_state=0).fit(data.numpy())
def sample_mixture(weights, means, covs, n, rng=np.random.default_rng(0)):
    if SOLUTIONS:
        k = rng.choice(len(weights), n, p=weights)
        return np.array([rng.multivariate_normal(means[j], covs[j]) for j in k])
    ...  # Exercise 1: choose components by weight, then sample each from its Gaussian
s = sample_mixture(gm.weights_, gm.means_, gm.covariances_, 1000)
check("exercise 1: mixture samples look like the data (energy distance < 0.05)", None if s is None else bool(energy_distance(torch.tensor(s, dtype=torch.float32), data) < 0.05), True)
samples["Gaussian mixture"] = torch.tensor(gm.sample(1000)[0], dtype=torch.float32)

# %% [markdown]
# ## 2. Autoregressive: one coordinate at a time
# p(x₁, x₂) = p(x₁) · p(x₂ | x₁). Discretise each coordinate into bins and predict bins with a classifier. This is
# exactly how language models work, with words instead of bins.

# %%
B = 64; lo, hi = data.min() - 0.1, data.max() + 0.1
to_bin = lambda v: ((v - lo) / (hi - lo) * B).long().clamp(0, B - 1); from_bin = lambda b: lo + (b.float() + torch.rand_like(b.float())) / B * (hi - lo)
p1 = torch.bincount(to_bin(data[:, 0]), minlength=B).float() + 1; p1 /= p1.sum()
cond = mlp(B, B); opt = torch.optim.Adam(cond.parameters(), 3e-3)
for _ in range(STEPS // 2):
    idx = torch.randint(0, len(data), (256,)); b1, b2 = to_bin(data[idx, 0]), to_bin(data[idx, 1])
    loss = F.cross_entropy(cond(F.one_hot(b1, B).float()), b2); opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    b1 = torch.multinomial(p1, 1000, replacement=True); b2 = torch.multinomial(F.softmax(cond(F.one_hot(b1, B).float()), -1), 1).squeeze(1)
samples["Autoregressive"] = torch.stack([from_bin(b1), from_bin(b2)], 1)

# %% [markdown]
# ## 3. VAE: encode to a small code, decode back
# The encoder outputs a mean and log-variance; sample z = μ + σ·ε (the **reparameterisation trick**, so gradients
# flow through the sampling); the decoder rebuilds x. Loss = reconstruction + KL(q(z|x) ‖ N(0, I)).

# %%
def reparameterise(mu, logvar, eps):
    if SOLUTIONS: return mu + torch.exp(0.5 * logvar) * eps
    ...  # Exercise 2
check("exercise 2: reparameterisation", reparameterise(torch.tensor([1.0]), torch.tensor([math.log(4.0)]), torch.tensor([0.5])), [2.0])
enc, dec = mlp(2, 4), mlp(2, 2); opt = torch.optim.Adam([*enc.parameters(), *dec.parameters()], 2e-3)
for _ in range(STEPS):
    x = data[torch.randint(0, len(data), (256,))]; mu, logvar = enc(x).chunk(2, -1)
    z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
    loss = F.mse_loss(dec(z), x, reduction="sum") / 256 + 0.05 * (-0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(-1).mean())
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad(): samples["VAE"] = dec(torch.randn(1000, 2))

# %% [markdown]
# ## 4. GAN: a forger and a detective
# The generator turns noise into samples; the discriminator tries to tell real from fake; each trains against the
# other. Sharp samples, no likelihood, notoriously unstable (watch for **mode collapse**: one moon only).

# %%
G, D = mlp(8, 2), mlp(2, 1); og, od = torch.optim.Adam(G.parameters(), 1e-3, betas=(0.5, 0.9)), torch.optim.Adam(D.parameters(), 1e-3, betas=(0.5, 0.9))
def d_loss(real_logits, fake_logits):
    if SOLUTIONS: return F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits)) + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    ...  # Exercise 3: real should be classified 1, fake 0 (use binary_cross_entropy_with_logits)
check("exercise 3: discriminator loss", d_loss(torch.tensor([0.0]), torch.tensor([0.0])), [2 * math.log(2)])
for _ in range(STEPS):
    x = data[torch.randint(0, len(data), (256,))]; fake = G(torch.randn(256, 8))
    ld = F.binary_cross_entropy_with_logits(D(x), torch.ones(256, 1)) + F.binary_cross_entropy_with_logits(D(fake.detach()), torch.zeros(256, 1))
    od.zero_grad(); ld.backward(); od.step()
    lg = F.binary_cross_entropy_with_logits(D(fake), torch.ones(256, 1)); og.zero_grad(); lg.backward(); og.step()
with torch.no_grad(): samples["GAN"] = G(torch.randn(1000, 8))

# %% [markdown]
# ## 5. Normalizing flow: an invertible warp of a Gaussian
# Stack invertible layers f so x = f(z), z ~ N(0, I). Exact likelihood by the change of variables:
# log p(x) = log N(f⁻¹(x)) + log|det ∂f⁻¹/∂x|. Here: affine coupling layers (RealNVP) that alternate coordinates.

# %%
class Coupling(nn.Module):
    def __init__(self, flip): super().__init__(); self.flip = flip; self.net = mlp(1, 2, 64)
    def inverse(self, x):                      # data → noise, with log-det
        a, b = (x[:, 1:], x[:, :1]) if self.flip else (x[:, :1], x[:, 1:])
        s, t = self.net(a).chunk(2, -1); s = torch.tanh(s); b = (b - t) * torch.exp(-s)
        return (torch.cat([b, a], 1) if self.flip else torch.cat([a, b], 1)), -s.sum(-1)
    def forward(self, z):                      # noise → data
        a, b = (z[:, 1:], z[:, :1]) if self.flip else (z[:, :1], z[:, 1:])
        s, t = self.net(a).chunk(2, -1); s = torch.tanh(s); b = b * torch.exp(s) + t
        return torch.cat([b, a], 1) if self.flip else torch.cat([a, b], 1)
flow = nn.ModuleList(Coupling(i % 2 == 1) for i in range(8)); opt = torch.optim.Adam(flow.parameters(), 2e-3)
for _ in range(STEPS):
    x = data[torch.randint(0, len(data), (256,))]; ld = 0
    for layer in reversed(flow): x, l = layer.inverse(x); ld = ld + l
    loss = -(-0.5 * (x ** 2).sum(-1) - math.log(2 * math.pi) + ld).mean(); opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    z = torch.randn(1000, 2)
    for layer in flow: z = layer(z)
    samples["Normalizing flow"] = z
print(f"flow's exact average log-likelihood on the data: {-loss.item():.2f} nats")

# %% [markdown]
# ## 6. Diffusion: learn to remove noise
# Forward process: x_t = √ᾱ_t · x₀ + √(1 − ᾱ_t) · ε, more noise as t grows. A network learns to predict ε from
# (x_t, t). Sampling runs the process backwards from pure noise, a little denoising per step (DDPM).

# %%
TT = 100; betas = torch.linspace(1e-4, 0.05, TT); abar = torch.cumprod(1 - betas, 0)
def noisy(x0, t, eps):
    if SOLUTIONS: return abar[t].sqrt()[:, None] * x0 + (1 - abar[t]).sqrt()[:, None] * eps
    ...  # Exercise 4: the forward process for a batch of timesteps t
check("exercise 4: forward noising", noisy(torch.ones(1, 2), torch.tensor([0]), torch.zeros(1, 2)), abar[0].sqrt() * torch.ones(1, 2))
eps_net = mlp(3, 2, 128); opt = torch.optim.Adam(eps_net.parameters(), 2e-3)
for _ in range(STEPS):
    x0 = data[torch.randint(0, len(data), (256,))]; t = torch.randint(0, TT, (256,)); eps = torch.randn_like(x0)
    xt = abar[t].sqrt()[:, None] * x0 + (1 - abar[t]).sqrt()[:, None] * eps
    loss = F.mse_loss(eps_net(torch.cat([xt, (t.float() / TT)[:, None]], 1)), eps); opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    x = torch.randn(1000, 2)
    for t in reversed(range(TT)):
        e = eps_net(torch.cat([x, torch.full((1000, 1), t / TT)], 1))
        x = (x - betas[t] / (1 - abar[t]).sqrt() * e) / (1 - betas[t]).sqrt()
        if t: x += betas[t].sqrt() * torch.randn_like(x)
    samples["Diffusion"] = x

# %% [markdown]
# ## Six models, one dataset

# %%
fig, ax = plt.subplots(2, 4, figsize=(15, 7)); ax = ax.ravel()
ax[0].scatter(*data[:1000].T, s=3, c="k"); ax[0].set_title("real data")
for a, (name, s) in zip(ax[1:], samples.items()):
    a.scatter(*s.T, s=3); a.set_title(f"{name}\nenergy distance {energy_distance(s, data):.3f}", fontsize=9); a.set_xlim(-2.5, 2.5); a.set_ylim(-2.5, 2.5)
ax[-1].axis("off"); plt.tight_layout(); plt.show()
for name, s in samples.items(): print(f"{name:18s} energy distance {energy_distance(s, data):.3f}")

# %% [markdown]
# **What I measured** (CPU, 4,000 steps each). Energy distance to the real data (lower is better): mixture 0.002,
# autoregressive 0.005, VAE 0.005, GAN 0.005, flow 0.017, diffusion 0.016. On a 2-D toy the humble mixture wins;
# the ranking flips on images, where mixtures and simple flows are hopeless and diffusion dominates. Judge
# generative models on the data they're meant for.
#
# ### Exercise 5: which generative model?
# Values: "mixture", "autoregressive", "vae", "gan", "flow", "diffusion".

# %%
def ex5():
    if SOLUTIONS: return {"generate_text": "autoregressive", "best_image_quality_today": "diffusion", "exact_likelihood_for_anomaly_scores": "flow",
                          "a_smooth_latent_space_to_interpolate": "vae", "one_step_fast_sharp_samples": "gan", "tiny_data_quick_density": "mixture"}
    ...
check("exercise 5", ex5(), {"generate_text": "autoregressive", "best_image_quality_today": "diffusion", "exact_likelihood_for_anomaly_scores": "flow",
                            "a_smooth_latent_space_to_interpolate": "vae", "one_step_fast_sharp_samples": "gan", "tiny_data_quick_density": "mixture"})

# %% [markdown]
# | Family | Sampling | Likelihood | Training | Where it shines |
# |---|---|---|---|---|
# | Mixture model | instant | exact | EM, stable | small, low-dimensional data |
# | Autoregressive | one element at a time (slow) | exact | stable (it's classification) | text, code, audio tokens: all LLMs |
# | VAE | one pass | lower bound | stable | latent spaces; the compressor inside latent diffusion |
# | GAN | one pass | none | unstable, mode collapse | fast sharp images, super-resolution |
# | Normalizing flow | one pass | exact | stable, architecture-constrained | density estimation, likelihood-based scores |
# | Diffusion / flow matching | many denoising steps | bound / via ODE | stable | images, video, audio, molecules |
