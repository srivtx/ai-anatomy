# %% [markdown]
# # 65 · Model merging: adding skills with weight arithmetic, and where it breaks
# Take a base model, fine-tune one copy for task A and another for task B. The **task vector**
# $\tau_A = \theta_A - \theta_{base}$ is the change fine-tuning made. Often you can *add* task vectors to get one
# model that does both, or *subtract* one to remove a skill, with no training at all (Ilharco et al. 2023,
# *Editing Models with Task Arithmetic*). Many models on open leaderboards are merges. This notebook shows the
# mechanics, the one case where merging is almost free, and the cases where it costs you. Chapters:
# **Distillation, pruning and merging**, **Different networks, same mind**.
#
# 1. A base digit classifier, fine-tuned for **mirrored** digits (task A) and digits **shifted down** (task B).
#    As in the original paper, the shared body is merged and each task keeps its own small output head.
# 2. Merge by averaging, by task arithmetic with a scale λ, and by TIES. Find where the tasks interfere.
# 3. Model soups: average fine-tunes of the *same* task. The easy win.
# 4. Subtract a task vector to remove a skill.
# 5. Why any of it works: walk the straight line between two models and look for a loss barrier.
#
# **Hardware:** CPU, about 2 minutes. **You're done when** you have the merge table, the soup result and the
# two loss-along-the-line curves.

# %%
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

QUICK = os.environ.get("QUICK") == "1"
torch.manual_seed(0)
digits = load_digits()
X = torch.tensor(digits.images, dtype=torch.float32) / 16  # (n, 8, 8), values 0..1
y = torch.tensor(digits.target)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
W = 512 if QUICK else 1024
FT_STEPS = 150 if QUICK else 300

TASKS = {
    "normal": lambda x: x,
    "mirrored (A)": lambda x: torch.flip(x, (2,)),
    "shifted (B)": lambda x: torch.roll(x, 2, 1),
}

fig, ax = plt.subplots(1, 3, figsize=(7, 2.4))

for a, (name, f) in zip(ax, TASKS.items()):
    a.imshow(f(Xte[:1])[0], cmap="gray_r")
    a.set_title(name)
    a.axis("off")

plt.show()

# %% [markdown]
# ## 1. A base model and two fine-tunes


# %%
def body():
    return nn.Sequential(nn.Flatten(), nn.Linear(64, W), nn.ReLU(), nn.Linear(W, W), nn.ReLU())


def fit(b, head, task, steps, lr, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(list(b.parameters()) + list(head.parameters()), lr)
    xs = TASKS[task](Xtr)
    for _ in range(steps):
        idx = torch.randint(len(xs), (256,))  # minibatches, so different seeds give different runs
        loss = F.cross_entropy(head(b(xs[idx])), ytr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return b, head


base_body, base_head = fit(body(), nn.Linear(W, 10), "normal", 600, 1e-3)
body_a, head_a = fit(
    copy.deepcopy(base_body), copy.deepcopy(base_head), "mirrored (A)", FT_STEPS, 1e-4
)
body_b, head_b = fit(
    copy.deepcopy(base_body), copy.deepcopy(base_head), "shifted (B)", FT_STEPS, 1e-4
)
HEADS = {
    "normal": base_head,
    "mirrored (A)": head_a,
    "shifted (B)": head_b,
}  # each task reads with its own head


@torch.no_grad()
def scores(b):
    return {
        t: (HEADS[t](b(f(Xte))).argmax(1) == yte).float().mean().item() for t, f in TASKS.items()
    }


def show(name, b):
    s = scores(b)
    print(f"{name:>30}" + "".join(f"{v:14.0%}" for v in s.values()))
    return s


header = lambda: print(f"{'':>30}" + "".join(f"{t:>14}" for t in TASKS))
header()
show("base", base_body)
show("fine-tuned on A", body_a)
show("fine-tuned on B", body_b)

# %% [markdown]
# Each fine-tune learns its task and partly forgets the normal digits. Now merge the bodies. Weights are
# flattened into one long vector, so the arithmetic below is literally vector arithmetic.

# %%
flat = lambda m: torch.nn.utils.parameters_to_vector(m.parameters()).detach()


def load(vec):
    m = body()
    torch.nn.utils.vector_to_parameters(vec, m.parameters())
    return m


θ0, θa, θb = flat(base_body), flat(body_a), flat(body_b)
τa, τb = θa - θ0, θb - θ0
print(
    f"|θ0| = {θ0.norm():.1f}   |τA| = {τa.norm():.2f}   |τB| = {τb.norm():.2f}   cosine(τA, τB) = {F.cosine_similarity(τa, τb, 0):.3f}"
)


def ties(taus, keep=0.2):
    """TIES-merging (Yadav et al. 2023): keep each vector's largest 20%, elect a sign per weight, average those that agree."""
    trimmed = []
    for t in taus:
        cut = t.abs().kthvalue(int(t.numel() * (1 - keep))).values
        trimmed.append(torch.where(t.abs() >= cut, t, torch.zeros_like(t)))
    stacked = torch.stack(trimmed)
    sign = torch.sign(stacked.sum(0))
    agree = (torch.sign(stacked) == sign) & (stacked != 0)
    return (stacked * agree).sum(0) / agree.sum(0).clamp(min=1)


print()
header()
show("average of A and B", load((θa + θb) / 2))

for lam in [0.6, 0.8, 1.0]:
    show(f"task arithmetic, λ = {lam}", load(θ0 + lam * (τa + τb)))

show("TIES, λ = 1", load(θ0 + ties([τa, τb])))

# %% [markdown]
# In my run the best task-arithmetic λ (0.8) reached about 60% on A and 56% on B: far above the base (41%, 9%),
# far below each specialist (91%, 80%), and normal digits fell to 81%. Plain averaging and TIES protected the
# normal digits but kept little of either new skill. The task vectors are nearly orthogonal (cosine ≈ 0.1), so
# this isn't two changes cancelling out. Look at where each change lives, layer by layer.

# %%
offset = 0

for name, p in base_body.named_parameters():
    n = p.numel()
    a, b = τa[offset : offset + n], τb[offset : offset + n]
    print(
        f"{name:>10}: cosine(τA, τB) = {F.cosine_similarity(a, b, 0):+.3f}   share of |τA|² here: {a.pow(2).sum() / τa.pow(2).sum():.0%}"
    )
    offset += n

# %% [markdown]
# The cosine is small in every layer, so the two changes don't point against each other anywhere. Two other
# effects cost the merge its accuracy. **Dilution:** averaging gives each task half its change, and half a change
# buys much less than half the skill. **Crosstalk:** each fine-tune adapted its head to *its own* features, and
# adding the other task's change moves those features under the head's feet. Raising λ fixes dilution and
# worsens crosstalk, which is why there's a best λ in the middle. In large pretrained models both effects are
# smaller, because each task vector is a tiny nudge spread over many layers.
#
# ## 3. Model soups: the merge that's nearly free
# Fine-tune the base on task A three times with different seeds (different minibatch order), then average the
# three (Wortsman et al. 2022, *Model soups*). There's no conflict this time: all three moved toward the same
# goal. In my run the soup scored within the range of its three parts, at the cost of a single model.

# %%
soup_parts = []

for seed in (1, 2, 3):
    b, _ = fit(
        copy.deepcopy(base_body), copy.deepcopy(head_a), "mirrored (A)", FT_STEPS, 1e-4, seed=seed
    )
    soup_parts.append(b)

singles = [scores(b)["mirrored (A)"] for b in soup_parts]
soup = load(torch.stack([flat(b) for b in soup_parts]).mean(0))
print("three fine-tunes on A:", ", ".join(f"{s:.1%}" for s in singles))
print(
    f"their average (soup): {scores(soup)['mirrored (A)']:.1%}   and on normal digits: {scores(soup)['normal']:.0%}"
)

# %% [markdown]
# ## 4. Removing a skill by subtraction
# $\theta_0 - \lambda\,\tau_A$: push the base *away* from task A. Does the mirrored skill drop while normal
# digits survive?

# %%
header()
show("base", base_body)

for lam in [0.5, 1.0, 2.0]:
    show(f"base − {lam}·τA", load(θ0 - lam * τa))

# %% [markdown]
# Small λ removes some of the skill at little cost; larger λ damages everything, because τA isn't purely “read
# mirrored digits”, it also carries generic changes. Task negation is used this way in practice (for example, to
# reduce toxic output): a small λ, then a check of what else got worse.
#
# ## 5. Why merging works at all: no barrier on the line
# Walk the straight line $(1-\alpha)\theta_A + \alpha\theta_B$ and measure the loss on normal digits. Then do
# the same between the base and a model trained **from a different random start** on the same task.


# %%
@torch.no_grad()
def loss_on_line(va, vb, head, n=21):
    return [
        F.cross_entropy(head(load((1 - a) * va + a * vb)(Xte)), yte).item()
        for a in np.linspace(0, 1, n)
    ]


torch.manual_seed(10)
other_body, other_head = fit(body(), copy.deepcopy(base_head), "normal", 600, 1e-3, seed=10)
alphas = np.linspace(0, 1, 21)
plt.plot(
    alphas, loss_on_line(θa, θb, base_head), label="A → B (both fine-tuned from the same base)"
)
plt.plot(
    alphas,
    loss_on_line(θ0, flat(other_body), base_head),
    label="base → a model from another random start",
)
plt.xlabel("α along the straight line")
plt.ylabel("test loss on normal digits (base head)")
plt.legend()
plt.show()

# %% [markdown]
# The two fine-tunes sit in the **same basin**: every point between them is a reasonable model, which is why
# averaging them doesn't fall apart. A network from another random start is separated by a barrier: its neurons
# are in a different order (the chapter **Different networks, same mind** shows how re-ordering them removes most
# of the barrier). **Rule: only merge models that share a parent**, and the less each one moved, the better.
#
# ## 6. Field notes: what I believe but can't cite
#
# | Belief | Confidence | Test |
# |---|---|---|
# | **Merges work because fine-tuning barely moves big models.** In a 7B model a fine-tune's task vector is tiny next to the weights and spread over many layers, so two of them rarely collide. Merging gets worse the longer and harder you fine-tune. | high | Double `FT_STEPS` or the learning rate and redo the merge table. |
# | **The choice of λ matters more than TIES vs DARE vs averaging.** Leaderboard merge recipes mostly win by tuning λ (and per-layer weights) on the benchmark itself, a quiet form of overfitting to it. | high | Choose λ on half the test set and report on the other half, for each method. |
# | **Soups are the safest merge there is**: same data, same base, different seeds. If a soup doesn't beat its parts, your fine-tunes moved too far apart (learning rate too high). | high | Raise the soup's learning rate 10× and see whether averaging starts to hurt. |
# | **Safety tuning can be removed by subtracting its task vector and added back by adding it**, and both directions work far too easily. That's why post-training safety on open weights is thin. | medium | Treat "normal" as the behaviour to protect: fine-tune it away, then add back part of the original. |
# | **What a task changes decides whether it merges.** Tasks that change how inputs are read (like mirroring) suffer crosstalk; tasks that change only what to do with the same features (new labels, new formats) merge almost perfectly. | medium | Make task B “predict 9 − digit” on normal images (same features, new answers) and redo the merge. |
#
# **Stretch:** merge two LoRA fine-tunes of the model from notebook 08. LoRA updates are low-rank: check whether
# they are even more orthogonal than full fine-tunes, and whether that makes them merge better.
