# %% [markdown]
# # 35 · Hyperparameters that transfer: μP
# Under the standard parameterisation (SP), the best learning rate drifts as a network gets wider, so every
# size needs its own sweep. Under μP (Yang & Hu; Yang et al., 2022) it stays put: tune small, train big.
# A character-level language model (8 characters of context → MLP → next character) at several widths, three things
# you can see: a coordinate check, an LR sweep under SP, the same sweep under μP. Chapter: **Keeping big runs stable**.
#
# **You're done when** you have the two LR-sweep plots side by side and can say how far the optimum moved.

# %%
import os
import math
import urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

QUICK = os.environ.get("QUICK") == "1"
dev = "cuda" if torch.cuda.is_available() else "cpu"
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

if not os.path.exists("shakespeare.txt"):
    urllib.request.urlretrieve(url, "shakespeare.txt")

text = open("shakespeare.txt").read()
chars = sorted(set(text))
V = len(chars)
data = torch.tensor([chars.index(c) for c in text], device=dev)
CTX, EMB = 8, 32


def batch(bs=256, g=None):
    i = torch.randint(len(data) - CTX - 1, (bs,), generator=g, device="cpu").to(dev)
    return torch.stack([data[i + k] for k in range(CTX)], 1), data[i + CTX]


BASE = 64


class Net(nn.Module):
    """emb → W_in (fixed fan-in) → ReLU → W_hid (width×width) → ReLU → W_out → logits."""

    def __init__(self, width, mup):
        super().__init__()
        self.mup, self.width = mup, width
        self.emb = nn.Embedding(V, EMB)
        self.w_in = nn.Linear(CTX * EMB, width)
        self.w_hid = nn.Linear(width, width)
        self.w_out = nn.Linear(width, V, bias=False)
        for l in (self.w_in, self.w_hid):
            nn.init.normal_(l.weight, std=1 / math.sqrt(l.in_features))
            nn.init.zeros_(l.bias)
        if mup:
            nn.init.zeros_(self.w_out.weight)  # μP: readout starts at zero
        else:
            nn.init.normal_(self.w_out.weight, std=1 / math.sqrt(width))

    def forward(self, x):
        h = F.relu(self.w_hid(F.relu(self.w_in(self.emb(x).flatten(1)))))
        return self.w_out(h)


def optimizer(m, lr):
    """Adam. Under μP, layers whose fan-in grows with width (hidden, readout) get lr × BASE/width."""
    if not m.mup:
        return torch.optim.Adam(m.parameters(), lr)
    s = BASE / m.width
    return torch.optim.Adam(
        [
            {"params": [*m.emb.parameters(), *m.w_in.parameters(), m.w_hid.bias], "lr": lr},
            {"params": [m.w_hid.weight, m.w_out.weight], "lr": lr * s},
        ]
    )


# %% [markdown]
# ## Coordinate check
# Train each width for a few steps at one learning rate and measure the typical size (RMS) of the logits.
# Under SP they grow with width: updates to a wide readout add up across many inputs. Under μP they're flat.

# %%
widths = [64, 128, 256, 512] if QUICK else [64, 128, 256, 512, 1024, 2048]

for mup in (False, True):
    rms = []
    for w in widths:
        torch.manual_seed(0)
        m = Net(w, mup).to(dev)
        opt = optimizer(m, 1e-2)
        g = torch.Generator().manual_seed(0)
        for _ in range(5):
            x, y = batch(g=g)
            loss = F.cross_entropy(m(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            rms.append(m(batch(g=g)[0]).pow(2).mean().sqrt().item())
    plt.plot(widths, rms, "o-", label="μP" if mup else "SP")
    print(("μP" if mup else "SP"), [f"{r:.1f}" for r in rms])

plt.xscale("log", base=2)
plt.xlabel("width")
plt.ylabel("logit RMS after 5 steps")
plt.legend()
plt.title("coordinate check")
plt.show()

# %% [markdown]
# ## Learning-rate sweep at three widths, SP vs μP

# %%
lrs = [2**k for k in range(-13, -3)] if not QUICK else [2**k for k in range(-12, -4, 2)]
sweep_widths = [128, 512, 2048] if not QUICK else [64, 256]
STEPS = 400 if not QUICK else 80


def final_loss(width, mup, lr):
    torch.manual_seed(0)
    m = Net(width, mup).to(dev)
    opt = optimizer(m, lr)
    g = torch.Generator().manual_seed(1)
    for _ in range(STEPS):
        x, y = batch(g=g)
        loss = F.cross_entropy(m(x), y)
        if not torch.isfinite(loss):
            return float("nan")
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return np.mean(
            [
                F.cross_entropy(m(x), y).item()
                for x, y in (batch(1024, torch.Generator().manual_seed(9)),)
            ]
        )


fig, ax = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
best = {}

for a, mup in zip(ax, (False, True)):
    for w in sweep_widths:
        L = [final_loss(w, mup, lr) for lr in lrs]
        a.plot(lrs, L, "o-", label=f"width {w}")
        best[(mup, w)] = lrs[int(np.nanargmin(L))]
    a.set_xscale("log", base=2)
    a.set_xlabel("learning rate")
    a.set_title("μP" if mup else "standard (SP)")
    a.legend()

ax[0].set_ylabel(f"loss after {STEPS} steps")
ax[0].set_ylim(None, 3.2)
plt.show()

for mup in (False, True):
    print(
        ("μP" if mup else "SP"), "best LR by width:",
        {w: f"2^{int(math.log2(best[(mup, w)]))}" for w in sweep_widths}
    )

# %% [markdown]
# **What I measured** (CPU run). Coordinate check: under SP the logit RMS after 5 steps went 1.8, 2.1, 1.0, 1.1,
# 3.5, 10.6 for widths 64 to 2048, blowing up past 1024; under μP it was 2.5–2.6 at every width. That's the
# clean, decisive evidence. The LR sweep is subtler at this scale: SP's best LR drifted down (2⁻⁸ at 128 and
# 512, 2⁻⁹ at 2048); μP's best settled at 2⁻⁶ from width 512 on, after sitting at 2⁻⁷ for width 128. μP's
# guarantee is about the limit of large width, and a 128-wide net is far from it. Expect a cleaner picture with
# longer training, a transformer, and widths of 256–4096 on a GPU; that's the stretch goal.
