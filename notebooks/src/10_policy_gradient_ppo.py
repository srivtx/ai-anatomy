# %% [markdown]
# # 10 · Policy gradients: REINFORCE, a baseline, GAE and PPO
# CartPole's physics written out in 20 lines, then three algorithms on it, each a small change to the last.
# Chapters: **RL basics**, **RLHF with PPO**.
#
# **You're done when** PPO reaches the maximum return of 500 and you have learning curves for all three.

# %%
import os, math, numpy as np, torch, torch.nn as nn, matplotlib.pyplot as plt
QUICK = os.environ.get("QUICK") == "1"

class CartPole:
    """The classic control problem (Barto, Sutton & Anderson, 1983), same constants as Gym's CartPole-v1."""
    g, mc, mp, l, F, dt = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
    def __init__(self, seed=0): self.rng = np.random.default_rng(seed)
    def reset(self): self.s = self.rng.uniform(-0.05, 0.05, 4); self.t = 0; return self.s.astype(np.float32)
    def step(self, a):
        x, xd, th, thd = self.s; f = self.F if a == 1 else -self.F
        tmp = (f + self.mp * self.l * thd ** 2 * math.sin(th)) / (self.mc + self.mp)
        thacc = (self.g * math.sin(th) - math.cos(th) * tmp) / (self.l * (4 / 3 - self.mp * math.cos(th) ** 2 / (self.mc + self.mp)))
        xacc = tmp - self.mp * self.l * thacc * math.cos(th) / (self.mc + self.mp)
        self.s = np.array([x + self.dt * xd, xd + self.dt * xacc, th + self.dt * thd, thd + self.dt * thacc]); self.t += 1
        done = abs(self.s[0]) > 2.4 or abs(self.s[2]) > 12 * math.pi / 180 or self.t >= 500
        return self.s.astype(np.float32), 1.0, done

def net(out): return nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, out))

def rollout(env, pi, vf=None, n_steps=2048):
    obs, acts, rews, dones, logps, vals, ep_returns, ret = [], [], [], [], [], [], [], 0
    o = env.reset()
    for _ in range(n_steps):
        with torch.no_grad():
            logits = pi(torch.tensor(o)); dist = torch.distributions.Categorical(logits=logits); a = dist.sample()
            v = vf(torch.tensor(o)).item() if vf else 0.0
        o2, r, d = env.step(a.item())
        obs.append(o); acts.append(a.item()); rews.append(r); dones.append(d); logps.append(dist.log_prob(a).item()); vals.append(v); ret += r
        o = o2
        if d: ep_returns.append(ret); ret = 0; o = env.reset()
    return dict(obs=torch.tensor(np.array(obs)), acts=torch.tensor(acts), rews=np.array(rews), dones=np.array(dones), logps=torch.tensor(logps), vals=np.array(vals), last_obs=o), ep_returns

def discounted(rews, dones, gamma=0.99):
    out, g = np.zeros_like(rews), 0.0
    for t in reversed(range(len(rews))): g = rews[t] + gamma * g * (1 - dones[t]); out[t] = g
    return out

def gae(b, vf, gamma=0.99, lam=0.95):
    with torch.no_grad(): last_v = vf(torch.tensor(b["last_obs"])).item()
    adv, g, vals = np.zeros_like(b["rews"]), 0.0, np.append(b["vals"], last_v)
    for t in reversed(range(len(b["rews"]))):
        nonterm = 1 - b["dones"][t]
        delta = b["rews"][t] + gamma * vals[t + 1] * nonterm - vals[t]          # one-step TD error
        g = delta + gamma * lam * nonterm * g; adv[t] = g
    return adv, adv + b["vals"]

def train(algo, iters, seed=0):
    torch.manual_seed(seed); env = CartPole(seed)
    pi, vf = net(2), net(1)
    opt = torch.optim.Adam(list(pi.parameters()) + list(vf.parameters()), 3e-4 if algo == "ppo" else 1e-3)
    curve = []
    for it in range(iters):
        b, eps = rollout(env, pi, vf if algo != "reinforce" else None)
        if eps: curve.append(np.mean(eps))
        if algo == "reinforce":                     # weight every log-prob by the return that followed
            adv = torch.tensor(discounted(b["rews"], b["dones"]), dtype=torch.float32); adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            loss = -(torch.distributions.Categorical(logits=pi(b["obs"])).log_prob(b["acts"]) * adv).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        elif algo == "baseline":                    # subtract a learned value: same expectation, less variance
            ret = torch.tensor(discounted(b["rews"], b["dones"]), dtype=torch.float32)
            v = vf(b["obs"]).squeeze(-1); adv = (ret - v.detach()); adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            loss = -(torch.distributions.Categorical(logits=pi(b["obs"])).log_prob(b["acts"]) * adv).mean() + 0.5 * ((v - ret) ** 2).mean() / ret.var()
            opt.zero_grad(); loss.backward(); opt.step()
        else:                                       # PPO: GAE advantages, several epochs, clipped ratio
            adv, ret = gae(b, vf); adv = torch.tensor(adv, dtype=torch.float32); ret = torch.tensor(ret, dtype=torch.float32)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            for _ in range(10):
                for idx in torch.randperm(len(adv)).split(256):
                    dist = torch.distributions.Categorical(logits=pi(b["obs"][idx]))
                    ratio = (dist.log_prob(b["acts"][idx]) - b["logps"][idx]).exp()
                    clip = torch.min(ratio * adv[idx], ratio.clamp(0.8, 1.2) * adv[idx])     # the PPO clip
                    vloss = ((vf(b["obs"][idx]).squeeze(-1) - ret[idx]) ** 2).mean()
                    loss = -clip.mean() + 0.5 * vloss / ret.var() - 0.01 * dist.entropy().mean()
                    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(list(pi.parameters()) + list(vf.parameters()), 0.5); opt.step()
    return curve

iters = 8 if QUICK else 60
curves = {a: train(a, iters) for a in ["reinforce", "baseline", "ppo"]}
for a, c in curves.items(): plt.plot(c, label=a); print(f"{a:10s} final mean return {c[-1]:.0f}")
plt.xlabel("iteration (2048 environment steps each)"); plt.ylabel("mean episode return"); plt.axhline(500, ls="--", c="gray"); plt.legend(); plt.show()
