# %% [markdown]
# # 62 · Where am I? Positional encodings and length extrapolation
# Attention by itself sees a *set*: shuffle the tokens and the output just shuffles too. A transformer only
# knows order through how positions are encoded, and that choice decides what happens on sequences longer than
# any in training. Here five encodings fight on the same task, at 1×, 2×, 4× and 8× the training length.
# Chapters: **Attention**, **Long context**.
#
# | Encoding | Idea | Used by |
# |---|---|---|
# | **NoPE** | nothing at all: the causal mask alone leaks position | some research models |
# | **Learned absolute** | a trained vector per position, added to the token | GPT-2 |
# | **Sinusoidal** | fixed sines and cosines of the position, added | the original 2017 transformer |
# | **RoPE** | rotate queries and keys by an angle ∝ position, so q·k depends on the *distance* | Llama, Qwen, Mistral… |
# | **ALiBi** | no embedding; subtract slope × distance from every attention score | BLOOM, MPT |
#
# **The task: induction.** Some random tokens, then a random block, then the same block again. In the repeat every
# token can be predicted by finding its previous occurrence and copying the token after it. This is the
# *induction head* circuit (Olsson et al. 2022), and it needs to know “the token right before me”, which is
# exactly a positional skill.
#
# **Hardware:** CPU, about 12 minutes (QUICK under 2). **You're done when** you have the accuracy table across
# lengths and the attention-sink measurement.

# %%
import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

QUICK = os.environ.get("QUICK") == "1"
torch.manual_seed(0)
VOCAB, D, HEADS, LAYERS = 64, 64, 4, 2
TRAIN_LEN, MAX_LEN = 64, 512
STEPS = 300 if QUICK else 3000


def induction_batch(bs, length, fixed=False):
    """Random prefix, then a random block, then the same block again. Returns inputs, targets and a mask of the
    positions whose next token is copyable. The block length varies, so 'copy from n tokens back' can't work.
    """
    x = torch.randint(0, VOCAB, (bs, length))
    mask = torch.zeros(bs, length, dtype=torch.bool)
    for i in range(bs):
        b = length // 2 if fixed else int(torch.randint(length // 8, length // 2 + 1, (1,)))
        start = length - 2 * b
        x[i, start + b :] = x[i, start : start + b]
        mask[i, start + b : length - 1] = True
    return x[:, :-1], x[:, 1:], mask[:, :-1]


x, y, mask = induction_batch(1, 24)
print("input   ", x[0].tolist())
print("copyable", mask[0].int().tolist())

# %% [markdown]
# ## 1. One attention layer, five ways to know position
# Everything except the encoding is shared. RoPE rotates pairs of dimensions of q and k by angle
# $p\,\theta_i$ with $\theta_i = 10000^{-2i/d}$; because rotations compose, $q_p\cdot k_s$ depends only on
# $p - s$. ALiBi adds $-m_h\,(p - s)$ to the score of head $h$, with slopes $m_h = 2^{-8h/H}$.


# %%
def rope(x):
    """Rotate pairs of channels of x (B, H, T, dh) by an angle proportional to the position."""
    T, dh = x.shape[-2], x.shape[-1]
    theta = 10000 ** (-torch.arange(0, dh, 2) / dh)
    ang = torch.arange(T)[:, None] * theta[None]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


def alibi_bias(T):
    slopes = 2 ** (-8 * torch.arange(1, HEADS + 1) / HEADS)
    dist = torch.arange(T)[:, None] - torch.arange(T)[None]  # how far back the key is
    return -slopes[:, None, None] * dist.clamp(min=0)


def sinusoid(T, d):
    pos = torch.arange(T)[:, None]
    div = 10000 ** (torch.arange(0, d, 2) / d)
    pe = torch.zeros(T, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos / div), torch.cos(pos / div)
    return pe


class Attention(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.qkv, self.o = nn.Linear(D, 3 * D, bias=False), nn.Linear(D, D, bias=False)
        self.last = None  # attention weights of the last call, for inspection

    def forward(self, x):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, HEADS, D // HEADS).permute(2, 0, 3, 1, 4)
        if self.kind == "rope":
            q, k = rope(q), rope(k)
        s = q @ k.transpose(-1, -2) / math.sqrt(D // HEADS)
        if self.kind == "alibi":
            s = s + alibi_bias(T)
        s = s.masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf"))
        a = s.softmax(-1)
        self.last = a.detach()
        return self.o((a @ v).transpose(1, 2).reshape(B, T, D))


class Model(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.tok = nn.Embedding(VOCAB, D)
        self.pos = nn.Embedding(MAX_LEN, D) if kind == "learned" else None
        self.atts = nn.ModuleList(Attention(kind) for _ in range(LAYERS))
        self.norms = nn.ModuleList(nn.LayerNorm(D) for _ in range(LAYERS))
        self.mlps = nn.ModuleList(
            nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
            for _ in range(LAYERS)
        )
        self.norms2 = nn.ModuleList(nn.LayerNorm(D) for _ in range(LAYERS))
        self.out = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, VOCAB))

    def forward(self, x):
        T = x.shape[1]
        h = self.tok(x)
        if self.kind == "learned":
            h = h + self.pos.weight[:T]
        if self.kind == "sinusoid":
            h = h + sinusoid(T, D)
        for att, n1, mlp, n2 in zip(self.atts, self.norms, self.mlps, self.norms2):
            h = h + att(n1(h))
            h = h + mlp(n2(h))
        return self.out(h)


# %% [markdown]
# ## 2. Train at length 64, test up to 512
# **Predict first:** which encoding do you expect to fall apart beyond 64 tokens, and which to survive?


# %%
def train(kind, steps=STEPS, fixed=False):
    torch.manual_seed(0)
    m = Model(kind)
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=0.01)
    for step in range(steps):
        x, y, mask = induction_batch(64, TRAIN_LEN, fixed)
        loss = F.cross_entropy(m(x)[mask], y[mask])  # train only where the answer is knowable
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return m


@torch.no_grad()
def copy_accuracy(m, length, bs=64, fixed=False):
    x, y, mask = induction_batch(bs, length, fixed)
    pred = m(x).argmax(-1)
    return (pred == y)[mask].float().mean().item()  # only copyable positions count


KINDS = ["nope", "learned", "sinusoid", "rope", "alibi"]
LENGTHS = [64, 128, 256, 512]
models, table = {}, {}
print(f"{'':>10}" + "".join(f"{f'len {n}':>10}" for n in LENGTHS))

for kind in KINDS:
    t0 = time.time()
    models[kind] = train(kind)
    table[kind] = [copy_accuracy(models[kind], n) for n in LENGTHS]
    print(
        f"{kind:>10}" + "".join(f"{a:10.0%}" for a in table[kind]) + f"   ({time.time() - t0:.0f}s)"
    )

for kind in KINDS:
    plt.plot(LENGTHS, table[kind], "o-", label=kind)

plt.xscale("log", base=2)
plt.axvline(TRAIN_LEN, color="gray", lw=1)
plt.xlabel("sequence length (trained at 64)")
plt.ylabel("copy accuracy on the repeat")
plt.legend()
plt.show()

# %% [markdown]
# Read the table row by row. In my run:
#
# - **Learned absolute** and **sinusoidal** were near perfect at length 64 and fell apart beyond it (about 20% at
#   128, single digits at 256). Positions past 64 are vectors the model never trained on, or values it never saw.
# - **ALiBi** learned fastest (above 80% within 500 steps) and kept ~90% at 4× the training length. Its bias
#   depends only on distance, and a penalty for “far away” means the same thing at any length.
# - **RoPE** also sees only relative angles, yet still degrades: its slowly rotating channels reach angles at long
#   range that never occurred in training. This is why real models need position interpolation or YaRN to stretch
#   their context.
# - **NoPE** never learned the task at all in 3,000 steps. Position can be inferred from the causal mask, but at
#   this size the model doesn't find out how.
#
# The exact numbers move with the seed: run two seeds before believing a gap of a few points (notebook 44 shows why).
#
# ## 3. The shortcut I fell into
# My first version of this notebook always used a block of exactly half the sequence, so the copy was always 32
# tokens back. RoPE scored 100% at length 64 and 2% at 128. It had not learned induction at all: it had learned
# “copy the token 31 positions back”, which RoPE makes trivially easy because it encodes relative distance.
# Reproduce it, because this failure is everywhere in real evaluations: **a benchmark with a fixed structure
# rewards the structure, not the skill.**

# %%
shortcut = train("rope", fixed=True)
print("RoPE, fixed block = half the sequence")
print("  same structure, length 64 :", f"{copy_accuracy(shortcut, 64, fixed=True):.0%}")
print("  same structure, length 128:", f"{copy_accuracy(shortcut, 128, fixed=True):.0%}")
print("  varied blocks, length 64  :", f"{copy_accuracy(shortcut, 64):.0%}")

# %% [markdown]
# ## 4. Attention sinks
# Trained transformers often dump a large share of attention onto the **first token**, whatever it is. That
# token acts as a parking spot for heads that have nothing useful to read (Xiao et al. 2023). It matters in
# practice: if a KV cache evicts the first token to save memory, quality collapses. Measure it here.

# %%
with torch.no_grad():
    x, _, _ = induction_batch(16, 256)
    for kind in KINDS:
        models[kind](x)
        shares = []
        for att in models[kind].atts:
            a = att.last[:, :, 32:]  # skip the first queries, which have few keys to choose from
            shares.append(a[..., 0].mean().item())
        uniform = (1 / torch.arange(33, 256).float()).mean().item()
        print(
            f"{kind:>10}: attention on token 0 by layer {[f'{s:.1%}' for s in shares]}   (uniform would be {uniform:.1%})"
        )

# %% [markdown]
# On a task this small the sinks are weak or absent; they grow with model size and training length, and
# softmax is what forces them (attention weights must sum to 1, so a head that wants to “do nothing” needs
# somewhere to put the weight). The stretch below makes one appear.
#
# ## 5. Field notes: what I believe but can't cite
#
# | Belief | Confidence | Test |
# |---|---|---|
# | **Positional encoding matters less than the training length distribution.** A model trained on a mix of lengths extrapolates better with *any* encoding than a model trained at one length with the “best” encoding. | high | Train RoPE on lengths drawn from 16 to 64 and compare with fixed 64. |
# | **NoPE works because the causal mask is a counter:** the first token gets attention 1, the second ½, and so on, and the model reads position from that. It breaks with bidirectional attention. | high | Remove the causal mask (and the copy task becomes trivial: use a different task) and watch NoPE fail. |
# | **RoPE's base (10,000) is a context-length setting in disguise.** Raising the base to 500,000 (as Llama 3 did) slows the rotations so far positions look familiar; most “RoPE can't extrapolate” results are runs with too small a base for the target length. | medium | Re-run RoPE with base 1,000 and 1,000,000 and compare at 512. |
# | **Attention sinks are a symptom of softmax, not a feature the model wants.** Give each head an explicit “attend to nothing” option (an extra zero logit, “softmax-off-by-one”) and the sinks mostly vanish without hurting loss. | medium | Append a zero column to the scores before the softmax; re-measure part 3. |
# | **The induction head forms suddenly**, in a phase change you can see as a bump in the loss curve, and *when* it forms depends on the positional encoding more than on anything else. | medium | Log the loss every 10 steps for each encoding and compare when the drop happens. |
# | **Long-context benchmarks that only test retrieval (needle in a haystack) overrate models.** A model can find one needle at 128k and still fail to *use* two facts that are 50k tokens apart. | high | Make the copy task need two lookups (copy the token after the token after the query). |
#
# **Stretch:** train with a fixed “beginning of sequence” token at position 0 and 4 layers, and look for a sink
# above 30% on that token.
