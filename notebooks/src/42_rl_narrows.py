# %% [markdown]
# # 42 · Does RL teach, or only sharpen?
# Reproduce the pass@k crossing (Yue et al., 2025) in three steps: the exact toy from the chapter, a neural
# policy trained with GRPO on problems it has to generalise to, and the same measurement on a real LLM.
# Then test my hypothesis: that the crossing moves to larger k as the RL mix gets more diverse.
# Chapters: **What RL really does to a model**, **GRPO and DPO**.
#
# **Hardware:** parts 1–2 run on a CPU in a few minutes. Part 3 needs a GPU and the Hugging Face hub.
#
# **You're done when** you have pass@k curves before and after RL, the k where they cross, and a plot of
# that crossing point against the number of distinct problem families in the RL data.

# %%
import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

QUICK = os.environ.get("QUICK") == "1"
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
KS = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def pass_at_k(n, c, k):
    """Unbiased estimator (Chen et al., 2021) from n samples with c correct."""
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


# %% [markdown]
# ## 1. The exact toy: one shared strategy distribution
# Eight strategies with different coverage; RL follows the exact policy gradient of expected reward.

# %%
COVER = np.array([0.7, 0.4, 0.25, 0.2, 0.15, 0.1, 0.08, 0.06])
PRIOR = np.array([0.4, 0.18, 0.12, 0.1, 0.08, 0.06, 0.04, 0.02])
rng = np.random.default_rng(4)
Q = np.where(rng.random((300, 8)) < COVER, 0.25 + 0.65 * rng.random((300, 8)), 0.01)
softmax = lambda z: np.exp(z - z.max()) / np.exp(z - z.max()).sum()
curve = lambda pi: [float(np.mean(1 - (1 - Q @ pi) ** k)) for k in KS]
z = np.log(PRIOR)
base = curve(softmax(z))

for _ in range(400):
    pi = softmax(z)
    R = Q.mean(0)
    z += 0.5 * pi * (R - pi @ R)

rl = curve(softmax(z))
plt.semilogx(KS, base, "o-", label="base")
plt.semilogx(KS, rl, "o-", label="after RL")
plt.xlabel("k")
plt.ylabel("pass@k")
plt.legend()
plt.title("Toy: shared strategy distribution")
plt.show()
cross = next((k for k, b, r in zip(KS, base, rl) if b > r), None)
print("pass@1 base %.3f → RL %.3f; base overtakes RL at k = %s" % (base[0], rl[0], cross))

# %% [markdown]
# ## 2. A neural policy that must generalise
# Problems are 14-bit integers. Strategy s solves a problem when some of its bits match a rule; the policy sees
# only 10 of the 14 bits, so like a real model it can't always tell which strategy will work. With F families,
# strategy s only works on problems whose family (x mod F) matches s mod F. The policy network sees the problem, picks a
# strategy, and is rewarded if it works. “Pretraining” = imitation of the prior (strategy picked regardless of
# correctness), so the base model is broad but unskilled. RL = GRPO on training problems; we measure pass@k on
# held-out problems.

# %%
BITS, HID, S = 10, 4, 8  # the policy sees 10 bits; 4 more are hidden from it
NEED = [1, 1, 2, 2, 3, 3, 3, 3]  # strategy s needs NEED[s] bits to match: coverage 2^-NEED


def make_world(families, seed=0):
    g = np.random.default_rng(seed)
    rules = []
    for s in range(S):  # each rule mixes a visible bit with hidden ones
        bits = [int(g.integers(0, BITS))] + [
            int(BITS + h) for h in g.choice(HID, NEED[s] - 1, replace=False)
        ]
        rules.append((np.array(bits), g.integers(0, 2, len(bits))))

    def solves(x, s):
        b = (x[:, None] >> np.arange(BITS + HID)) & 1
        idx, val = rules[s]
        return np.all(b[:, idx] == val, 1) & ((x % families) == (s % families))

    return solves


def feats(x):
    return torch.tensor(((x[:, None] >> np.arange(BITS)) & 1), dtype=torch.float32)


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(BITS, 64), nn.ReLU(), nn.Linear(64, S))

    def forward(self, x):
        return self.net(x)


def evaluate(pol, solves, xs, n=256):
    with torch.no_grad():
        p = F.softmax(pol(feats(xs)), -1).numpy()
    ok = np.stack([solves(xs, s) for s in range(S)], 1).astype(float)  # [problems, strategies]
    succ = (p * ok).sum(1)  # chance one sample solves it
    c = np.random.default_rng(1).binomial(n, np.clip(succ, 0, 1))
    return [float(np.mean([pass_at_k(n, ci, k) for ci in c])) for k in KS]


def run(families, steps=600 if not QUICK else 120, seed=0):
    solves = make_world(families, seed)
    xs = np.random.default_rng(seed).permutation(2 ** (BITS + HID))
    train, test = xs[:12000], xs[12000:13000]
    pol = Policy()
    opt = torch.optim.Adam(pol.parameters(), 3e-3)
    prior = torch.tensor(np.log(PRIOR), dtype=torch.float32)
    for _ in range(300 if not QUICK else 60):  # pretraining: imitate the prior
        x = np.random.choice(train, 128)
        loss = F.kl_div(
            F.log_softmax(pol(feats(x)), -1),
            F.softmax(prior, -1).expand(128, S),
            reduction="batchmean",
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    base = evaluate(pol, solves, test)
    opt = torch.optim.Adam(pol.parameters(), 1e-3)
    G = 8
    for _ in range(steps):  # GRPO: groups of G samples per problem
        x = np.random.choice(train, 32)
        logits = pol(feats(x))
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample((G,))  # [G, 32]
        ok = np.stack([solves(x, s) for s in range(S)], 1)  # [32, S]
        r = torch.tensor(ok[np.arange(32)[None, :], a.numpy()], dtype=torch.float32)
        adv = (r - r.mean(0)) / (r.std(0) + 1e-6)
        loss = -(adv * dist.log_prob(a)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return base, evaluate(pol, solves, test)


b, r = run(families=1)
plt.semilogx(KS, b, "o-", label="base (held-out problems)")
plt.semilogx(KS, r, "o-", label="after GRPO")
plt.xlabel("k")
plt.ylabel("pass@k")
plt.legend()
plt.show()
print("neural policy: pass@1 %.3f → %.3f, pass@256 %.3f → %.3f" % (b[0], r[0], b[-1], r[-1]))

# %% [markdown]
# ## 3. My hypothesis: diversity moves the crossing
# Same setup, more strategy families in the RL mix. Prediction (~65% confidence): the crossing point k* rises,
# and with enough families it disappears (RL also improves pass@256). If k* stays flat, I'm wrong.

# %%
rows = []

for fams in ([1, 2, 4, 8] if not QUICK else [1, 4]):
    b, r = run(fams, seed=fams)
    k_star = next((k for k, bb, rr in zip(KS, b, r) if bb > rr + 0.005), None)
    rows.append((fams, r[0] - b[0], r[-1] - b[-1], k_star))
    print(
        f"families {fams}: Δpass@1 {r[0]-b[0]:+.3f}  Δpass@256 {r[-1]-b[-1]:+.3f}  crossing k* = {k_star}"
    )

# %% [markdown]
# ## 4. On a real model (GPU)
# The same measurement on GSM8K with a base model and an RL-trained sibling. Expect the instruct/RL model to
# win at k = 1 and the gap to shrink or reverse by k = 64–256 (Yue et al. saw reversal for most base/RL pairs).

# %%
if torch.cuda.is_available() and not QUICK:
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from datasets import load_dataset

        qs = load_dataset("openai/gsm8k", "main", split="test").select(range(100))

        def correct_counts(name, n=64):
            tok = AutoTokenizer.from_pretrained(name)
            m = AutoModelForCausalLM.from_pretrained(
                name, torch_dtype=torch.bfloat16, device_map="cuda"
            )
            out = []
            for q in qs:
                gold = q["answer"].split("####")[-1].strip().replace(",", "")
                ids = tok(
                    f"Question: {q['question']}\nAnswer: Let's think step by step.",
                    return_tensors="pt",
                ).to("cuda")
                g = m.generate(
                    **ids,
                    do_sample=True,
                    temperature=0.8,
                    max_new_tokens=256,
                    num_return_sequences=n,
                )
                texts = tok.batch_decode(g[:, ids.input_ids.shape[1] :], skip_special_tokens=True)
                out.append(sum(gold in t.split("\n\n")[0][-40:] for t in texts))
            return out

        for name in ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct"]:
            c = correct_counts(name)
            print(
                name,
                [
                    round(float(np.mean([pass_at_k(64, ci, k) for ci in c])), 3)
                    for k in [1, 4, 16, 64]
                ],
            )
    except Exception as e:
        print("Real-model part skipped:", type(e).__name__, e)
else:
    print(
        "Part 4 needs a GPU (and the Hugging Face hub). Parts 1–3 are the full experiment in miniature."
    )

# %% [markdown]
# **What I measured** (full run, CPU, ~3 minutes). Part 1: pass@1 25% → 40%, base overtakes RL from k = 8.
# Part 2, neural policy on held-out problems: pass@1 39% → 78%, but pass@256 94% → 82%. Part 3, the hypothesis:
#
# | strategy families | Δ pass@1 | Δ pass@256 | crossing k* |
# |---|---|---|---|
# | 1 | +0.39 | −0.12 | 8 |
# | 2 | +0.32 | −0.13 | 8 |
# | 4 | +0.27 | −0.06 | 32 |
# | 8 | +0.15 | −0.01 | 64 |
#
# The crossing moved right as the mix got more diverse, as predicted, and the large-k loss almost vanished. But
# it never turned into a large-k *gain*, so the strong form of my hypothesis (diverse RL expands pass@k) isn't
# supported at this scale; the weak form (diversity protects breadth) is. Diversity also cost pass@1 gains here,
# because each strategy is useful on fewer problems. Run part 4 on a GPU to see where real models sit.
