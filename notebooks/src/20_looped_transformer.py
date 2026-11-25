# %% [markdown]
# # 20 · The looped transformer: one block, run many times
# Notebook 14 looped a small conv net. This one loops a real transformer block: the same weights run T times,
# and every pass gets the input added back in. If the block learns one step of an algorithm, then running it
# longer at test time solves harder problems than any it saw in training. Chapter: **Looped and recurrent-depth
# models** (after Dehghani et al. 2018, *Universal Transformers*; Giannou et al. 2023, *Looped Transformers as
# Programmable Computers*; Geiping et al. 2025, the recurrent-depth model *Huginn*).
#
# The task is **pointer chasing**. Twelve nodes, each pointing at another one. The question for every node is:
# where do you land after k hops? One hop is exactly one attention lookup (“read the pointer of the node I'm
# at”), so a block that learns one hop can do k hops in k passes.
#
# 1. Train one looped block on k ≤ 5 hops, test up to 24 hops.
# 2. Ablate the two details that decide whether it extrapolates: re-injecting the input, and normalising the state.
# 3. Overthinking: what happens if you keep looping past the answer.
# 4. Trees and an adaptive exit: loop until the answer stops changing.
# 5. Field notes: what I believe about looped models that isn't written down anywhere, and how to test it.
#
# **Hardware:** CPU, about 5 minutes (QUICK mode under 2). **You're done when** you have the accuracy-vs-hops
# table, the ablation table and the adaptive-exit numbers.

# %%
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

QUICK = os.environ.get("QUICK") == "1"
torch.manual_seed(0)
N, D, HEADS, KMAX = 12, 96, 4, 5  # nodes, width, heads, most hops seen in training
STEPS = 400 if QUICK else 1500


def permutation_batch(bs):
    """Each row is a random permutation: node i points to f[i]."""
    return torch.argsort(torch.rand(bs, N), 1)


def hops(f, k):
    """Targets after 1, 2, ..., k hops for every node at once."""
    p = torch.arange(N).expand(f.shape[0], N)
    out = []
    for _ in range(k):
        p = torch.gather(f, 1, p)
        out.append(p)
    return out


f = permutation_batch(1)
print("pointers      ", f[0].tolist())
print("after 1 hop   ", hops(f, 1)[-1][0].tolist())
print("after 3 hops  ", hops(f, 3)[-1][0].tolist())

# %% [markdown]
# ## 1. The model
# A pre-norm transformer block (attention + MLP), **one copy of the weights**. Each loop:
#
# $$h_{t+1} = \mathrm{Norm}\big(\mathrm{Block}(h_t + e)\big)$$
#
# where $e$ is the embedded input (pointer token + position). Adding $e$ every pass is **input injection**: the
# block never has to remember the question, only its progress. The norm on the state keeps its size the same
# on loop 30 as on loop 3, so the block sees the kind of input it was trained on. We'll switch each one off in
# part 2 and see what breaks.
#
# For comparison, `Stacked` uses the same block design with **separate weights per layer**, like a normal
# transformer. It has 5× the parameters and simply has no sixth layer to run.


# %%
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.att = nn.MultiheadAttention(D, HEADS, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, x):
        a = self.n1(x)
        x = x + self.att(a, a, a, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class Looped(nn.Module):
    def __init__(self, inject=True, norm=True):
        super().__init__()
        self.tok, self.pos = nn.Embedding(N, D), nn.Embedding(N, D)
        self.block = Block()
        self.state_norm = nn.LayerNorm(D)
        self.out = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, N))
        self.inject, self.norm = inject, norm

    def forward(self, f, loops):
        e = self.tok(f) + self.pos.weight
        h, outs = e, []
        for _ in range(loops):
            h = self.block(h + e if self.inject else h)
            if self.norm:
                h = self.state_norm(h)
            outs.append(self.out(h))  # a prediction after every loop
        return outs


class Stacked(nn.Module):
    def __init__(self, depth=KMAX):
        super().__init__()
        self.tok, self.pos = nn.Embedding(N, D), nn.Embedding(N, D)
        self.blocks = nn.ModuleList(Block() for _ in range(depth))
        self.out = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, N))

    def forward(self, f, loops):
        if loops > len(self.blocks):
            return None  # there is no layer 6
        e = self.tok(f) + self.pos.weight
        h, outs = e, []
        for b in self.blocks[:loops]:
            h = b(h + e)
            outs.append(self.out(h))
        return outs


params = lambda m: sum(p.numel() for p in m.parameters())
print(f"looped: {params(Looped()):,} parameters   stacked: {params(Stacked()):,} parameters")

# %% [markdown]
# ## 2. Train on k ≤ 5, test on k ≤ 24
# Each batch draws a random k from 1 to 5 and runs exactly k loops, with a loss on the prediction after every
# loop (loop t should give the answer after t hops). **Predict first:** at k = 16, more than three times the
# hardest training example, will the looped model be near 100%, near chance, or in between?


# %%
def train(
    model, steps=STEPS, per_step=True, data=permutation_batch, kmax=KMAX, fixed_k=None, log=True
):
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=0.01)
    t0 = time.time()
    for step in range(steps):
        k = fixed_k or int(torch.randint(1, kmax + 1, (1,)))
        f = data(128)
        targets, outs = hops(f, k), model(f, k)
        if per_step:
            loss = (
                sum(F.cross_entropy(o.flatten(0, 1), t.flatten()) for o, t in zip(outs, targets))
                / k
            )
        else:
            loss = F.cross_entropy(outs[-1].flatten(0, 1), targets[-1].flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if log and step % (steps // 4) == 0:
            print(f"  step {step:4d}  loss {loss.item():.4f}  ({time.time() - t0:.0f}s)")
    return model


@torch.no_grad()
def accuracy(model, k, data=permutation_batch, read_at=None):
    f = data(512)
    outs = model(f, read_at or k)
    if outs is None:
        return None
    return (outs[-1].argmax(-1) == hops(f, k)[-1]).float().mean().item()


TEST_K = [1, 3, 5, 6, 8, 10, 12, 16, 24]
print("looped")
looped = train(Looped())
print("stacked")
stacked = train(Stacked())
rows = {"looped (1 block)": looped, "stacked (5 blocks)": stacked}
print(f"\n{'hops k':>20}" + "".join(f"{k:>7}" for k in TEST_K))

for name, m in rows.items():
    accs = [accuracy(m, k) for k in TEST_K]
    print(f"{name:>20}" + "".join("      —" if a is None else f"{a:7.0%}" for a in accs))

# %% [markdown]
# Chance is not 1/12 here: in a random permutation many nodes sit on short cycles, so some guesses are free.
# The stacked model is perfect where it has layers and has nothing to offer beyond them. The looped model keeps
# going well past what it trained on, then fades. Part 2 asks why it fades.
#
# ## 3. Which details matter? Four ablations
# Same training, different switches. Extrapolation is measured at 10, 16 and 24 hops.

# %%
variants = {
    "inject + norm (default)": dict(inject=True, norm=True),
    "inject, no state norm": dict(inject=True, norm=False),
    "no inject, no state norm": dict(inject=False, norm=False),
    "inject + norm, final-loop loss": dict(inject=True, norm=True, per_step=False),
}
ablation = {}
print(f"{'':>32}" + "".join(f"{k:>7}" for k in ("k=5", "k=10", "k=16", "k=24")))

for name, kw in variants.items():
    per_step = kw.pop("per_step", True)
    torch.manual_seed(1)
    m = train(Looped(**kw), per_step=per_step, log=False)
    ablation[name] = [accuracy(m, k) for k in (5, 10, 16, 24)]
    print(f"{name:>32}" + "".join(f"{a:7.0%}" for a in ablation[name]))

# %% [markdown]
# In my run (1500 steps), with injection and the state norm, 16 hops stayed at 94%. Without the norm it fell to
# 21%, and without injection as well the model was lost by 10 hops. Supervising only the last loop cost some
# extrapolation (79% at 16 hops) but much less than dropping the norm. The lesson generalises: **a looped model extrapolates only if
# the state it feeds itself on loop 20 looks like the state it saw on loop 3.** Injection and normalisation are
# two ways of making that true.
#
# ## 4. Overthinking: looping past the answer
# Ask for k hops but read the answer after k + extra loops. **Predict:** does extra thinking help, or hurt?

# %%
for extra in [0, 1, 2, 4]:
    print(
        f"k = 4, read after {4 + extra} loops: {accuracy(looped, 4, read_at=4 + extra):.0%} correct"
    )

# %% [markdown]
# It hurts, and in a perfectly understandable way: the block does one more hop per loop, so after 5 loops it
# answers the 5-hop question. A looped model needs to know **when to stop**. Two fixes: give it a task whose
# answer is a fixed point (extra loops change nothing), or learn a halting signal. Part 5 does the first.
#
# ## 5. Trees, fixed points and an adaptive exit
# Now every node points at its parent and the root points at itself. Ask “where is your root?”: after enough
# hops every node sits on the root and stays there, so the answer is a fixed point. Train on shallow trees
# (depth ≤ 4, 6 loops), then test on a **single chain 11 deep** and loop until the predictions stop changing.


# %%
def tree_batch(bs, max_depth=4):
    f = torch.zeros(bs, N, dtype=torch.long)
    for b in range(bs):
        order = torch.randperm(N).tolist()
        depth = {order[0]: 0}
        f[b, order[0]] = order[0]  # the root points to itself
        for j in range(1, N):
            parents = [v for v in order[:j] if depth[v] < max_depth]
            p = parents[int(torch.randint(len(parents), (1,)))]
            f[b, order[j]], depth[order[j]] = p, depth[p] + 1
    return f


def chain_batch(bs):
    f = torch.zeros(bs, N, dtype=torch.long)
    for b in range(bs):
        o = torch.randperm(N)
        f[b, o[0]] = o[0]
        f[b, o[1:]] = o[:-1]  # every node points to the one before it: depth 11
    return f


torch.manual_seed(2)
tree_model = train(Looped(), steps=STEPS // 2 if QUICK else 800, data=tree_batch, fixed_k=6)

with torch.no_grad():
    f = chain_batch(256)
    root = hops(f, N)[-1]
    preds = [o.argmax(-1) for o in tree_model(f, 30)]

one_hop = [
    min(t + 1, N) / N for t in range(30)
]  # share of nodes within t hops of the root, if 1 hop per loop
measured = [(p == root).float().mean().item() for p in preds]
plt.plot(range(1, 31), measured, "o-", label="looped model, measured")
plt.plot(range(1, 31), one_hop, "--", label="if it did exactly one hop per loop")
plt.axvline(6, color="gray", lw=1)
plt.text(6.3, 0.1, "trained with 6 loops")
plt.xlabel("loops")
plt.ylabel("nodes that found their root")
plt.legend()
plt.show()

stops = []

for b in range(256):
    stop = next(
        (
            t + 1
            for t in range(2, 30)
            if torch.equal(preds[t][b], preds[t - 1][b])
            and torch.equal(preds[t - 1][b], preds[t - 2][b])
        ),
        30,
    )
    stops.append(stop)

print(
    "found their root after 4, 6, 8 loops:", ", ".join(f"{measured[t - 1]:.0%}" for t in (4, 6, 8))
)
solved = sum(torch.equal(preds[s - 1][b], root[b]) for b, s in enumerate(stops)) / 256
print(
    f"adaptive exit (stop after 2 unchanged loops): {sum(stops) / 256:.1f} loops on average, whole chain right {solved:.0%}"
)

# %% [markdown]
# Two things to notice. First, the adaptive exit works: the model stops by itself and gets whole chains right,
# though it never saw a chain deeper than 4. Second, the measured curve rises **faster than one hop per loop**.
# In my run it had about 95% of nodes right after 6 loops, where single hops would give 58%. Attention reads the
# *current state* of the node it points at, not the original pointer, so a node can jump to “wherever my parent
# has got to”. That is pointer jumping, which needs only about log₂(depth) loops. Nobody told it to do this. To
# check, plot the attention weights of loop 3 and see whether nodes attend to their grandparent.
#
# ## 6. Field notes: what I believe but can't cite
# These come from running looped and recurrent models and from reading between the lines of papers. None of
# them is an established result. Each has a confidence and a cheap test, and some are cheap enough to run here.
#
# | Belief | Confidence | How to test it in this notebook |
# |---|---|---|
# | Most “looped models don't extrapolate” results are really **state drift**: the state's norm or direction on loop 20 is unlike anything seen in training. Fix the drift and extrapolation returns. | high | Part 3 already shows it. Also log `h.norm()` per loop with and without the state norm. |
# | **Train with a random loop count**, or the model learns a clock: it uses “which loop am I on?” as information and breaks at other counts. Huginn samples the count from a heavy-tailed distribution for this reason. | high | Train with `fixed_k=5` only and test at 8 and 12. |
# | **Start the block near identity** (zero-init the attention and MLP output projections). A looped block that starts as a random map multiplies its randomness every pass, and early training wastes steps undoing it. | medium | Zero `block.att.out_proj.weight` and `block.mlp[2].weight` at init; compare loss at step 100. |
# | You don't need to backpropagate through every loop. **Truncating the gradient to the last few loops** gives nearly the same model at a fraction of the memory (Huginn backprops through its last 8). The early loops still learn, through the loops that follow. | medium | Wrap all but the last 3 loops in `torch.no_grad()` during training. |
# | **Looping buys reasoning per parameter, not knowledge per parameter.** A looped model ties a deeper stacked model on tasks that need steps, and loses on tasks that need stored facts, because facts need parameters to live in. | medium | Add a task that needs a memorised lookup table and compare looped vs stacked at equal parameters. |
# | A stop-when-it-stops-changing exit **fails on oscillation**: some states fall into a 2-cycle and never settle. Real systems compare against the state two steps back, or use a KL threshold on the output. | medium | Count chains in part 5 where `preds[t]` equals `preds[t-2]` but not `preds[t-1]`. |
# | The **pointer jumping** of part 5 appears only when the loss rewards finishing early. With a single loss at the final loop, the model settles for one hop per loop. | low: a guess | Train `tree_model` with `per_step=False` and redraw the curve. |
#
# **Stretch:** loop a 2-layer block instead of 1 (the “prelude, recurrent core, coda” design of Huginn), or put the
# looped block inside the GPT from notebook 06 and run 1, 2, 4 and 8 loops on the same weights.
