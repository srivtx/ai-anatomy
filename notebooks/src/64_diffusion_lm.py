# %% [markdown]
# # 64 · A diffusion language model: write all the words at once
# Every model so far writes left to right, one token per step. A **masked diffusion language model** starts from
# a line of blanks and fills them in over a few rounds, in any order, many tokens per round. This is the idea
# behind MDLM (Sahoo et al. 2024), LLaDA (Nie et al. 2025) and the fast “diffusion” coding models. You'll train
# one and a normal left-to-right (autoregressive, AR) model of the same size on the same text, then compare:
#
# 1. **Loss.** How far behind is diffusion on likelihood?
# 2. **Speed vs quality.** Fewer rounds are faster; how much worse is the text?
# 3. **Infilling.** Given the start and the end, fill the middle, which AR can't do directly.
#
# The training rule is short: mask each token with probability t (t random in (0, 1)), predict the masked
# ones, and weight the loss by 1/t. That weighted loss is an upper bound on the negative log-likelihood, so the
# two models can be compared on the same scale. Chapters: **Diffusion and flow matching**, **Next-token prediction**.
#
# **Hardware:** a free Colab T4 (Runtime → Change runtime type). It also runs on a CPU, but slowly: on a 2-core
# CPU the full run took over 40 minutes in my test. QUICK mode takes about 3 minutes on a CPU. **You're done when** you have the loss
# comparison, samples at 4 to 128 rounds with their quality scores, and an infilled line.

# %%
import os
import time
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

QUICK = os.environ.get("QUICK") == "1"
torch.manual_seed(0)
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

if not os.path.exists("shakespeare.txt"):
    urllib.request.urlretrieve(url, "shakespeare.txt")

text = open("shakespeare.txt").read()
chars = sorted(set(text))
V = len(chars)
MASK = V  # one extra token id: the blank
data = torch.tensor([chars.index(c) for c in text])
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
T, D, LAYERS, HEADS = 96, 128, 4, 4
STEPS = 200 if QUICK else 1500
decode = lambda ids: "".join("_" if i == MASK else chars[i] for i in ids)


def batch(split, bs=32):
    d = train_data if split == "train" else val_data
    i = torch.randint(len(d) - T - 1, (bs,))
    return torch.stack([d[j : j + T] for j in i]), torch.stack([d[j + 1 : j + T + 1] for j in i])


# %% [markdown]
# ## 1. One transformer, two ways to use it
# The only architectural difference: the AR model uses a causal mask (each position sees only the past); the
# diffusion model sees everything, including the blanks around it.


# %%
class Net(nn.Module):
    def __init__(self, causal):
        super().__init__()
        self.causal = causal
        self.tok, self.pos = nn.Embedding(V + 1, D), nn.Embedding(T, D)
        layer = nn.TransformerEncoderLayer(
            D, HEADS, 4 * D, dropout=0.0, batch_first=True, norm_first=True, activation="gelu"
        )
        self.body = nn.TransformerEncoder(layer, LAYERS)
        self.out = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, V))

    def forward(self, x):
        mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1]) if self.causal else None
        h = self.tok(x) + self.pos.weight[: x.shape[1]]
        return self.out(self.body(h, mask=mask, is_causal=self.causal))


def ar_loss(m, x, y):
    return F.cross_entropy(m(x).flatten(0, 1), y.flatten())


def diffusion_loss(m, x):
    """Mask each token with prob t, predict the masked ones, weight by 1/t (the MDLM bound, per token)."""
    t = torch.rand(x.shape[0], 1).clamp(min=1e-3)
    masked = torch.rand(x.shape) < t
    logits = m(torch.where(masked, MASK, x))
    ce = F.cross_entropy(logits.transpose(1, 2), x, reduction="none")
    return (ce * masked / t).sum() / x.numel()


def train(m, loss_fn):
    opt = torch.optim.AdamW(m.parameters(), 2e-3, betas=(0.9, 0.95), weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=STEPS, pct_start=0.1)
    t0 = time.time()
    for step in range(STEPS):
        x, y = batch("train")
        loss = loss_fn(m, x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % (STEPS // 4) == 0:
            print(f"  step {step:4d}  loss {loss.item():.3f}  ({time.time() - t0:.0f}s)")
    return m


print("autoregressive")
ar = train(Net(causal=True), ar_loss)
print("diffusion")
dm = train(Net(causal=False), lambda m, x, y: diffusion_loss(m, x))

# %% [markdown]
# ## 2. Likelihood: how far behind is diffusion?
# The AR loss is the exact negative log-likelihood per character. The diffusion number is an upper bound on it
# (averaged over many random t), so the true diffusion NLL is somewhere at or below it.

# %%
with torch.no_grad():
    torch.manual_seed(1)
    ar_val = sum(ar_loss(ar, *batch("val")).item() for _ in range(20)) / 20
    dm_val = sum(diffusion_loss(dm, batch("val")[0]).item() for _ in range(60)) / 60

print(f"AR        NLL      {ar_val:.3f} nats/char")
print(f"diffusion bound ≤ {dm_val:.3f} nats/char   (gap {dm_val - ar_val:+.3f})")

# %% [markdown]
# Expect the diffusion model to be clearly behind on likelihood at this size: it must learn to predict *any* subset of
# tokens from *any* other subset, a much bigger job than “next from previous”. Published results find the gap
# shrinks with scale but has not closed for likelihood; the case for diffusion LMs is speed and flexibility.
#
# ## 3. Sampling: fewer rounds, faster, worse
# Start with 96 blanks. Each round, predict every blank, then keep the most confident predictions and leave the
# rest blank for the next round. With R rounds, about 96/R tokens are fixed per round. To score the text fairly,
# we ask the **AR model** how surprising each sample is (lower is more Shakespeare-like).


# %%
@torch.no_grad()
def diffusion_sample(m, rounds, x=None, temp=0.8):
    x = torch.full((1, T), MASK) if x is None else x.clone()
    for r in range(rounds):
        blank = x == MASK
        left = int(blank.sum())
        if left == 0:
            break
        probs = (m(x) / temp).softmax(-1)
        pick = torch.multinomial(probs[0], 1)[:, 0][None]
        conf = probs.gather(-1, pick[..., None])[..., 0].masked_fill(~blank, -1)
        k = max(
            1, round(left / (rounds - r))
        )  # spread the remaining blanks over the remaining rounds
        keep = conf[0].topk(k).indices
        x[0, keep] = pick[0, keep]
    return x


@torch.no_grad()
def ar_sample(m, n=T, temp=0.8):
    x = torch.tensor([[chars.index("\n")]])
    while x.shape[1] < n:
        p = (m(x[:, -T:])[:, -1] / temp).softmax(-1)
        x = torch.cat([x, torch.multinomial(p, 1)], 1)
    return x


@torch.no_grad()
def ar_judge(x):
    """Average surprise of the AR model reading the sample (nats/char)."""
    return F.cross_entropy(ar(x[:, :-1]).flatten(0, 1), x[:, 1:].flatten()).item()


torch.manual_seed(0)
rounds_list = [4, 8, 16, 32, 96]
scores = []

for R in rounds_list:
    s = [diffusion_sample(dm, R) for _ in range(4)]
    scores.append(sum(ar_judge(x) for x in s) / 4)
    print(f"--- diffusion, {R} rounds (judge {scores[-1]:.2f}) ---\n{decode(s[0][0].tolist())}\n")

ar_samples = [ar_sample(ar) for _ in range(4)]
ar_score = sum(ar_judge(x) for x in ar_samples) / 4
print(
    f"--- autoregressive, 96 steps (judge {ar_score:.2f}) ---\n{decode(ar_samples[0][0].tolist())}"
)

plt.plot(rounds_list, scores, "o-", label="diffusion")
plt.axhline(ar_score, ls="--", color="gray", label="AR (one token per step)")
plt.xscale("log", base=2)
plt.xlabel("rounds (forward passes) to write 96 characters")
plt.ylabel("AR judge: nats/char (lower is better)")
plt.legend()
plt.show()

# %% [markdown]
# The judge is biased toward the AR model (it is the AR model), so read the *shape* of the curve, not the gap to
# the dashed line. Going from 96 rounds to 16 is a 6× speed-up; see how much quality it costs. At 4 rounds, 24
# characters are fixed at once, each chosen without seeing the others, so neighbours that each look likely on
# their own can belong to different words. That **independence within a round** is the core weakness of parallel decoding.
#
# ## 4. Infilling: start and end given, fill the middle

# %%
x, _ = batch("val", 1)
gap = slice(30, 66)
holed = x.clone()
holed[0, gap] = MASK
torch.manual_seed(3)
filled = diffusion_sample(dm, 36, holed)
print("with a hole:", decode(holed[0].tolist()).replace("\n", "⏎"))
print("filled     :", decode(filled[0].tolist()).replace("\n", "⏎"))
print("original   :", decode(x[0].tolist()).replace("\n", "⏎"))

# %% [markdown]
# The filled text has to fit both what comes before and what comes after. An AR model sees only what's before;
# getting it to infill takes special training (fill-in-the-middle), and it still can't look at the suffix while
# writing the first missing word.
#
# ## 5. Field notes: what I believe but can't cite
#
# | Belief | Confidence | Test |
# |---|---|---|
# | **The 1/t weight is what makes it train badly at small scale.** Rounds with tiny t (few blanks) get huge weights and noisy gradients; sampling t from a distribution that avoids tiny values (or clipping the weight) trains faster at the cost of a looser bound. | medium | Clamp t at 0.1 instead of 0.001 and compare the loss curve and the samples. |
# | **Confidence-ordered unmasking is doing most of the work.** Unmasking random positions instead of the most confident ones loses far more quality than halving the number of rounds. | high | Replace `topk(conf)` with a random choice of blanks and redo part 3. |
# | **Diffusion LMs write easy tokens first**: spaces, line breaks and the ends of common words fill in round 1, rare content words last. The model is effectively planning the shape of the text before the content. | high | Record which round each position was filled in and colour the sample by it. |
# | **Diffusion LMs are not slower to train per token of data, they are worse per token of data**; the gap to AR behaves like a constant multiple of compute (several times more) that doesn't close with scale for likelihood. | medium | Train both for 2× and 4× steps and see whether the gap in part 2 shrinks in nats or in ratio. |
# | **The commercial win is latency for code**, where the shape is predictable (indentation, brackets) and many tokens can be fixed per round safely. For open-ended prose the quality per round is much worse. | medium | Train on Python source instead of Shakespeare and repeat part 3: does the curve flatten? |
# | **Reversal is easier for diffusion LMs.** Because they learn to predict any token from any others, “B is A's parent” helps with “A's parent is B” (the reversal curse of notebook 59). LLaDA reports this; I expect it to be the most durable advantage. | medium | Train notebook 59's family facts on the diffusion objective and test reversed questions. |
#
# **Stretch:** semi-autoregressive sampling, as LLaDA does: split the line into 4 blocks, run diffusion inside
# each block from left to right, and compare quality at the same total number of rounds.
