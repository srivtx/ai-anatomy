# %% [markdown]
# # 25 · AlphaZero on tic-tac-toe
# Monte Carlo tree search guided by a small policy-value network that learns only from self-play.
# Chapter: **AlphaZero: search + learning**.
#
# **You're done when** the agent never loses to random play and draws against perfect minimax play.

# %%
import os, math, random, functools
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
QUICK = os.environ.get("QUICK") == "1"
random.seed(0); np.random.seed(0); torch.manual_seed(0)
LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
def winner(b):
    for i, j, k in LINES:
        if b[i] != 0 and b[i] == b[j] == b[k]: return b[i]
    return 0 if 0 in b else 2           # 2 = draw
def moves(b): return [i for i in range(9) if b[i] == 0]

@functools.lru_cache(None)
def minimax(b, player):                 # perfect play, for evaluation only
    w = winner(b)
    if w == 2: return 0, None
    if w != 0: return (1 if w == player else -1), None
    best = (-2, None)
    for m in moves(b):
        nb = list(b); nb[m] = player; v = -minimax(tuple(nb), -player)[0]
        if v > best[0]: best = (v, m)
    return best

class Net(nn.Module):
    def __init__(self):
        super().__init__(); self.body = nn.Sequential(nn.Linear(18, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.p, self.v = nn.Linear(128, 9), nn.Linear(128, 1)
    def forward(self, b, player):       # board from the current player's point of view
        x = torch.tensor([[1.0 if c == player else 0.0 for c in b] + [1.0 if c == -player else 0.0 for c in b]])
        h = self.body(x); return self.p(h)[0], torch.tanh(self.v(h))[0, 0]

def mcts(net, b, player, sims, c=1.5):
    N, Wv, P = {}, {}, {}
    def search(b, player):
        w = winner(b)
        if w == 2: return 0.0
        if w != 0: return 1.0 if w == player else -1.0
        if b not in P:                                       # expand with the network's prior and value
            with torch.no_grad(): logits, v = net(b, player)
            mask = torch.full((9,), -1e9); mask[moves(b)] = 0
            P[b] = F.softmax(logits + mask, -1).numpy(); N[b] = np.zeros(9); Wv[b] = np.zeros(9)
            return v.item()
        n = N[b].sum()
        ucb = [Wv[b][m] / (N[b][m] + 1e-9) * (N[b][m] > 0) + c * P[b][m] * math.sqrt(n + 1) / (1 + N[b][m]) if b[m] == 0 else -1e9 for m in range(9)]
        m = int(np.argmax(ucb)); nb = list(b); nb[m] = player
        v = -search(tuple(nb), -player)
        N[b][m] += 1; Wv[b][m] += v; return v
    for _ in range(sims): search(b, player)
    return N[b] / N[b].sum()

def self_play(net, sims):
    b, player, hist = (0,) * 9, 1, []
    while winner(b) == 0:
        pi = mcts(net, b, player, sims)
        hist.append((b, player, pi))
        m = np.random.choice(9, p=pi) if len(hist) < 3 else int(np.argmax(pi))   # explore early moves
        nb = list(b); nb[m] = player; b, player = tuple(nb), -player
    w = winner(b)
    return [(s, p, pi, 0.0 if w == 2 else (1.0 if w == p else -1.0)) for s, p, pi in hist]

net = Net(); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4); buf = []
iters, games, sims = (6, 10, 25) if QUICK else (40, 30, 60)
for it in range(iters):
    for _ in range(games): buf += self_play(net, sims)
    buf = buf[-5000:]
    for _ in range(100):
        batch = random.sample(buf, min(64, len(buf))); loss = 0
        for s, p, pi, z in batch:
            logits, v = net(s, p)
            loss = loss + (v - z) ** 2 - (torch.tensor(pi, dtype=torch.float32) * F.log_softmax(logits, -1)).sum()
        opt.zero_grad(); (loss / len(batch)).backward(); opt.step()
    print(f"iteration {it}: buffer {len(buf)} positions")

# %% [markdown]
# ## Evaluate: against random play and against perfect play

# %%
def play(agent_first, opponent, sims):
    b, player = (0,) * 9, 1
    while winner(b) == 0:
        agent_turn = (player == 1) == agent_first
        m = int(np.argmax(mcts(net, b, player, sims))) if agent_turn else opponent(b, player)
        nb = list(b); nb[m] = player; b, player = tuple(nb), -player
    w = winner(b); agent = 1 if agent_first else -1
    return "draw" if w == 2 else ("win" if w == agent else "loss")
rand_opp = lambda b, p: random.choice(moves(b)); perfect = lambda b, p: minimax(b, p)[1]
for name, opp in [("random", rand_opp), ("perfect minimax", perfect)]:
    for sims_eval in ([25] if QUICK else [10, 50, 200]):
        res = [play(f, opp, sims_eval) for f in [True, False] for _ in range(10 if QUICK else 25)]
        print(f"vs {name:16s} with {sims_eval:3d} sims: " + ", ".join(f"{k} {res.count(k)}" for k in ["win", "draw", "loss"]))
