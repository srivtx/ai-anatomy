# %% [markdown]
# # 28 · Graph neural networks: message passing by hand
# A graph convolutional network in plain PyTorch on Zachary's karate club (34 people, one club that split
# in two), trained from just two labelled members. Chapter: **Graph neural networks**.
#
# **You're done when** most nodes are classified from two labels and the embedding plot splits the club.

# %%
import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, matplotlib.pyplot as plt
QUICK = os.environ.get("QUICK") == "1"
torch.manual_seed(0)

# Zachary's karate club (1977): edges and which faction each member joined after the split
edges = [(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,10),(0,11),(0,12),(0,13),(0,17),(0,19),(0,21),(0,31),(1,2),(1,3),(1,7),(1,13),(1,17),(1,19),(1,21),(1,30),(2,3),(2,7),(2,8),(2,9),(2,13),(2,27),(2,28),(2,32),(3,7),(3,12),(3,13),(4,6),(4,10),(5,6),(5,10),(5,16),(6,16),(8,30),(8,32),(8,33),(9,33),(13,33),(14,32),(14,33),(15,32),(15,33),(18,32),(18,33),(19,33),(20,32),(20,33),(22,32),(22,33),(23,25),(23,27),(23,29),(23,32),(23,33),(24,25),(24,27),(24,31),(25,31),(26,29),(26,33),(27,33),(28,31),(28,33),(29,32),(29,33),(30,32),(30,33),(31,32),(31,33),(32,33)]
faction = torch.tensor([0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1])
N = 34
A = torch.zeros(N, N)
for i, j in edges: A[i, j] = A[j, i] = 1

# %% [markdown]
# ## The GCN layer
# $H' = \sigma(\hat D^{-1/2} \hat A \hat D^{-1/2} H W)$ with $\hat A = A + I$: every node averages its
# neighbours' features (and its own), then applies a shared linear map. That's message passing.

# %%
Ah = A + torch.eye(N); d = Ah.sum(1); Anorm = Ah / torch.sqrt(d[:, None] * d[None, :])

class GCN(nn.Module):
    def __init__(self, layers=2, h=16):
        super().__init__()
        dims = [N] + [h] * (layers - 1) + [2]
        self.W = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims, dims[1:]))
    def forward(self, X, return_hidden=False):
        H = X
        for k, W in enumerate(self.W):
            H = Anorm @ W(H)
            if k < len(self.W) - 1: H = torch.tanh(H); hidden = H
        return (H, hidden) if return_hidden else H

X = torch.eye(N)                     # no features: each node is just its identity
labelled = torch.tensor([0, 33])     # the instructor and the administrator
def train(layers, steps=200):
    torch.manual_seed(0); m = GCN(layers); opt = torch.optim.Adam(m.parameters(), 0.01, weight_decay=5e-4)
    for _ in range(steps):
        loss = F.cross_entropy(m(X)[labelled], faction[labelled])
        opt.zero_grad(); loss.backward(); opt.step()
    acc = (m(X).argmax(1) == faction).float().mean().item()
    return m, acc

model, acc = train(2)
print(f"2-layer GCN from 2 labels: {acc:.0%} of members placed in the right faction")

# %%
_, H = model(X, return_hidden=True)
H = H.detach()
plt.figure(figsize=(5, 5)); plt.scatter(H[:, 0], H[:, 1], c=faction, cmap="coolwarm", s=60)
for i in range(N): plt.annotate(str(i), H[i, :2], fontsize=7)
plt.title("Hidden embeddings: the club splits"); plt.show()

# %% [markdown]
# ## Over-smoothing
# Each layer averages over neighbours. Stack enough and every node's representation converges to the same
# vector, so accuracy falls.

# %%
for L in [1, 2, 4, 8, 16]:
    print(f"{L:2d} layers: accuracy {train(L)[1]:.0%}")
