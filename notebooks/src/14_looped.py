# %% [markdown]
# # 14 · Recurrent depth: think longer at test time
# Train a weight-shared ("looped") network on 32-bit prefix-sum parity, then test on much longer inputs by
# running more iterations. A fixed-depth network of the same size can't. Chapter: **Looped networks**
# (after Schwarzschild et al., 2021 and Bansal et al., 2022).
#
# **You're done when** you have accuracy vs iterations on inputs longer than any seen in training.

# %%
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

QUICK = os.environ.get("QUICK") == "1"
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


def data(bs, n):  # input bits → running parity (prefix sums mod 2)
    x = torch.randint(0, 2, (bs, n), device=dev).float()
    return x, (x.cumsum(1) % 2)


class Looped(nn.Module):
    """1D conv block applied repeatedly with shared weights; the input is re-injected each iteration ("recall")."""

    def __init__(self, w=64, k=3):
        super().__init__()
        self.inp = nn.Conv1d(1, w, k, padding=k // 2)
        self.block = nn.Sequential(
            nn.Conv1d(2 * w, w, k, padding=k // 2),
            nn.ReLU(),
            nn.Conv1d(w, w, k, padding=k // 2),
            nn.ReLU(),
        )
        self.out = nn.Conv1d(w, 1, k, padding=k // 2)

    def forward(self, x, iters):
        e = self.inp(x[:, None])
        h = torch.zeros_like(e)
        outs = []
        for _ in range(iters):
            h = self.block(torch.cat([h, e], 1))
            outs.append(self.out(h)[:, 0])
        return outs


class Fixed(nn.Module):
    """Same parts, no weight sharing: a fixed number of distinct layers."""

    def __init__(self, depth, w=64, k=3):
        super().__init__()
        self.inp = nn.Conv1d(1, w, k, padding=k // 2)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(2 * w, w, k, padding=k // 2),
                nn.ReLU(),
                nn.Conv1d(w, w, k, padding=k // 2),
                nn.ReLU(),
            )
            for _ in range(depth)
        )
        self.out = nn.Conv1d(w, 1, k, padding=k // 2)

    def forward(self, x, iters=None):
        e = self.inp(x[:, None])
        h = torch.zeros_like(e)
        for b in self.blocks:
            h = b(torch.cat([h, e], 1))
        return [self.out(h)[:, 0]]


n_train, train_iters = (16, 16) if QUICK else (32, 30)
steps = 400 if QUICK else 4000


def train(model, iters):
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
    for s in range(steps):
        x, y = data(64, n_train)
        outs = model(x, iters)
        # supervise a random later iteration, so the net can't rely on a specific stopping point
        k = torch.randint(len(outs) // 2, len(outs), (1,)).item() if len(outs) > 1 else 0
        loss = F.binary_cross_entropy_with_logits(outs[k], y)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


looped = train(Looped().to(dev), train_iters)
fixed = train(Fixed(train_iters // 2).to(dev), None)


@torch.no_grad()
def solved(model, n, iters):
    x, y = data(256, n)
    out = model(x, iters)[-1]
    return ((out > 0).float() == y).all(1).float().mean().item()  # whole sequence right


test_lens = [n_train, 2 * n_train, 4 * n_train]

for n in test_lens:
    its = (
        [train_iters, 2 * train_iters, 4 * train_iters]
        if not QUICK
        else [train_iters, 2 * train_iters]
    )
    print(
        f"length {n:4d}: fixed-depth {solved(fixed, n, None):.0%} | looped "
        + ", ".join(f"{i} iters {solved(looped, n, i):.0%}" for i in its)
    )

# %% [markdown]
# **What to expect.** Each iteration can carry the parity a few positions further, so a 32-bit problem needs
# roughly 32/kernel-reach iterations and a 256-bit one needs many more. The looped net, trained only on short
# inputs, solves longer ones when given more iterations; the fixed-depth net can't, because its depth was
# fixed at training time.
