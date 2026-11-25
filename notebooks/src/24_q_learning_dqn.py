# %% [markdown]
# # 24 · Q-learning: a table, then a network
# Tabular Q-learning on a gridworld with a value map, then a DQN with replay and a target network on
# CartPole. Chapter: **Q-learning and DQN**.
#
# **You're done when** you have a gridworld value map and a DQN that balances CartPole.

# %%
import os
import math
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

QUICK = os.environ.get("QUICK") == "1"
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# %% [markdown]
# ## Tabular Q-learning
# $Q(s,a) \leftarrow Q(s,a) + \alpha\,[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$

# %%
GRID = ["S...#....", ".##.#.##.", "....#..#.", ".#.....#.", ".#.###.#.", ".#...#...", "...#.#.#G"]
H, W = len(GRID), len(GRID[0])
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def step(s, a):
    r, c = s[0] + MOVES[a][0], s[1] + MOVES[a][1]
    if not (0 <= r < H and 0 <= c < W) or GRID[r][c] == "#":
        r, c = s
    return (r, c), (1.0 if GRID[r][c] == "G" else -0.01), GRID[r][c] == "G"


Q = np.zeros((H, W, 4))

for ep in range(500 if QUICK else 2000):
    s, eps = (0, 0), max(0.05, 1 - ep / 500)
    for t in range(200):
        a = np.random.randint(4) if np.random.rand() < eps else Q[s].argmax()
        s2, r, done = step(s, a)
        Q[s][a] += 0.5 * (r + 0.95 * Q[s2].max() * (not done) - Q[s][a])
        s = s2
        if done:
            break

V = Q.max(-1)
V[np.array([[ch == "#" for ch in row] for row in GRID])] = np.nan
plt.imshow(V, cmap="viridis")
plt.colorbar(label="max Q")
plt.title("Learned values: brighter = closer to the goal")
s, path = (0, 0), [(0, 0)]

for _ in range(60):
    s, _, d = step(s, Q[s].argmax())
    path.append(s)
    if d:
        break

plt.plot([p[1] for p in path], [p[0] for p in path], "r.-")
plt.show()
print(
    "greedy path reaches the goal:", GRID[path[-1][0]][path[-1][1]] == "G", "in", len(path) - 1,
    "steps"
)

# %% [markdown]
# ## DQN on CartPole
# Two tricks make Q-learning with a network stable: a **replay buffer** (break the correlation between
# consecutive samples) and a **target network** (a slowly-updated copy used for the bootstrap target).


# %%
class CartPole:
    g, mc, mp, l, Fm, dt = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.s = self.rng.uniform(-0.05, 0.05, 4)
        self.t = 0
        return self.s.astype(np.float32)

    def step(self, a):
        x, xd, th, thd = self.s
        f = self.Fm if a == 1 else -self.Fm
        tmp = (f + self.mp * self.l * thd**2 * math.sin(th)) / (self.mc + self.mp)
        thacc = (self.g * math.sin(th) - math.cos(th) * tmp) / (
            self.l * (4 / 3 - self.mp * math.cos(th) ** 2 / (self.mc + self.mp))
        )
        xacc = tmp - self.mp * self.l * thacc * math.cos(th) / (self.mc + self.mp)
        self.s = np.array(
            [x + self.dt * xd, xd + self.dt * xacc, th + self.dt * thd, thd + self.dt * thacc]
        )
        self.t += 1
        fell = abs(self.s[0]) > 2.4 or abs(self.s[2]) > 12 * math.pi / 180
        return self.s.astype(np.float32), 1.0, fell, self.t >= 500


import copy

qnet = nn.Sequential(
    nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2)
)
target = copy.deepcopy(qnet)  # the slowly-updated copy
opt = torch.optim.Adam(qnet.parameters(), 5e-4)
buf = collections.deque(maxlen=50_000)
env = CartPole()
o, ret, returns, total = env.reset(), 0, [], 4_000 if QUICK else 60_000

for t in range(total):
    eps = max(0.02, 1 - t / (total * 0.3))
    a = random.randrange(2) if random.random() < eps else qnet(torch.tensor(o)).argmax().item()
    o2, r, fell, trunc = env.step(a)
    buf.append((o, a, r, o2, fell))
    ret += r
    o = o2
    if fell or trunc:
        returns.append(ret)
        ret = 0
        o = env.reset()
    if len(buf) >= 1000:
        ob, ac, rw, ob2, dn = map(np.array, zip(*random.sample(buf, 128)))
        with torch.no_grad():
            # Double DQN: the online net picks the action, the target net scores it
            a2 = qnet(torch.tensor(ob2)).argmax(1, keepdim=True)
            y = torch.tensor(rw, dtype=torch.float32) + 0.99 * target(torch.tensor(ob2)).gather(
                1, a2
            ).squeeze(1) * (1 - torch.tensor(dn, dtype=torch.float32))
        q = qnet(torch.tensor(ob)).gather(1, torch.tensor(ac)[:, None]).squeeze(1)
        loss = F.smooth_l1_loss(q, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    if t % 1000 == 0:
        target.load_state_dict(qnet.state_dict())

plt.plot(returns, alpha=0.4)
plt.plot(np.convolve(returns, np.ones(20) / 20, mode="valid"))
plt.xlabel("episode")
plt.ylabel("return")
plt.title("DQN learns, noisily")
plt.show()
print(
    "best 20-episode average:",
    max(np.convolve(returns, np.ones(min(20, len(returns))) / min(20, len(returns)), mode="valid")),
)
