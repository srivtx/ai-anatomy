# %% [markdown]
# # 27 · Distil, prune and quantise
# Train a big teacher on MNIST, then make the smallest model you can: distil into a tiny student with soft
# labels, prune it, and quantise it to 8 and 4 bits. Measure what each step costs in accuracy and saves in size.
# Chapters: **Distillation**, **Precision**, **Inference**.
#
# **You're done when** you have an accuracy-vs-size table for every variant, and the distilled student beats the
# same student trained on labels alone.

# %%
import os
import gzip
import urllib.request
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

QUICK = os.environ.get("QUICK") == "1"
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


def load_mnist():
    base = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    out = {}
    for k, f in [
        ("xtr", "train-images-idx3-ubyte.gz"),
        ("ytr", "train-labels-idx1-ubyte.gz"),
        ("xte", "t10k-images-idx3-ubyte.gz"),
        ("yte", "t10k-labels-idx1-ubyte.gz"),
    ]:
        if not os.path.exists(f):
            urllib.request.urlretrieve(base + f, f)
        out[k] = np.frombuffer(gzip.open(f).read(), np.uint8, offset=16 if k[0] == "x" else 8)
    f = lambda x: torch.tensor(x.reshape(-1, 784), dtype=torch.float32) / 255
    return (
        f(out["xtr"]),
        torch.tensor(out["ytr"]).long(),
        f(out["xte"]),
        torch.tensor(out["yte"]).long(),
    )


xtr, ytr, xte, yte = load_mnist()

if QUICK:
    xtr, ytr = xtr[:6000], ytr[:6000]

# a small labelled set makes the teacher's soft labels matter more
n_small = 2000 if QUICK else 5000
xtr, ytr, xte, yte = xtr.to(dev), ytr.to(dev), xte.to(dev), yte.to(dev)


def mlp(h):
    return nn.Sequential(
        nn.Linear(784, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 10)
    ).to(dev)


def acc(m, x=xte, y=yte):
    with torch.no_grad():
        return (m(x).argmax(1) == y).float().mean().item()


def size_kb(m, bits=32):
    return sum(p.numel() for p in m.parameters()) * bits / 8 / 1024


def train(m, x, y, epochs, teacher=None, T=4.0, alpha=0.9, lr=1e-3):
    opt = torch.optim.Adam(m.parameters(), lr)
    for _ in range(epochs):
        perm = torch.randperm(len(x), device=dev)
        for i in range(0, len(x), 128):
            b = perm[i : i + 128]
            out = m(x[b])
            loss = F.cross_entropy(out, y[b])
            if teacher is not None:
                with torch.no_grad():
                    soft = F.softmax(teacher(x[b]) / T, -1)
                kd = (
                    F.kl_div(F.log_softmax(out / T, -1), soft, reduction="batchmean") * T * T
                )  # Hinton et al., 2015
                loss = alpha * kd + (1 - alpha) * loss
            opt.zero_grad()
            loss.backward()
            opt.step()
    return m


# %% [markdown]
# ## Teacher, student alone, distilled students
# The teacher sees all the labelled data. The first student sees only a small labelled subset. The second sees the
# same subset plus the teacher's full probability distribution on those images (“dark knowledge”: a 7 that
# looks a bit like a 1). The third never sees a label: the teacher labels the rest of the images for it.

# %%
teacher = train(mlp(512), xtr, ytr, 3 if QUICK else 8)
student_alone = train(mlp(32), xtr[:n_small], ytr[:n_small], 30 if not QUICK else 15)
torch.manual_seed(0)
student_soft = train(
    mlp(32), xtr[:n_small], ytr[:n_small], 30 if not QUICK else 15, teacher=teacher
)

# the setting where distillation pays: the teacher labels data nobody labelled. alpha=1 → soft labels only
torch.manual_seed(0)
student_kd = train(mlp(32), xtr, ytr, 6 if not QUICK else 4, teacher=teacher, alpha=1.0)
rows = [
    ("teacher (512 hidden)", acc(teacher), size_kb(teacher)),
    ("student, labels only", acc(student_alone), size_kb(student_alone)),
    ("student, + soft labels, same images", acc(student_soft), size_kb(student_soft)),
    ("student, distilled on unlabelled data", acc(student_kd), size_kb(student_kd)),
]

# %% [markdown]
# ## Prune: zero the smallest weights
# Magnitude pruning, then a short fine-tune with the mask held fixed. Sparse storage needs indices too, so the
# real saving is smaller than the sparsity suggests unless the hardware supports sparse formats (2:4 on NVIDIA GPUs).


# %%
def prune(m, frac):
    m = copy.deepcopy(m)
    masks = []
    for p in m.parameters():
        if p.dim() == 2:
            th = p.abs().flatten().kthvalue(int(frac * p.numel())).values
            mask = (p.abs() > th).float()
            p.data *= mask
            masks.append((p, mask))
    opt = torch.optim.Adam(m.parameters(), 5e-4)
    for _ in range(3):
        for i in range(0, n_small, 128):
            loss = F.cross_entropy(m(xtr[i : i + 128]), ytr[i : i + 128])
            opt.zero_grad()
            loss.backward()
            opt.step()
            for p, mask in masks:
                p.data *= mask
    return m


for frac in [0.5, 0.8, 0.9]:
    pm = prune(student_kd, frac)
    nz = sum((p != 0).sum().item() for p in pm.parameters())
    rows.append((f"distilled + {int(frac * 100)}% pruned", acc(pm), nz * 32 / 8 / 1024))

# %% [markdown]
# ## Quantise weights to 8 and 4 bits
# Per-output-channel scales for int8; groups of 16 weights per scale for int4 (one scale per group costs
# 16 bits, so effective bits per weight = 4 + 16/16 = 5).


# %%
def quantise(m, bits, group=None):
    m = copy.deepcopy(m)
    qmax = 2 ** (bits - 1) - 1
    for p in m.parameters():
        if p.dim() < 2:
            continue
        w = p.data if group is None else p.data.reshape(-1, group)
        scale = w.abs().amax(1, keepdim=True).clamp(min=1e-8) / qmax
        p.data = (torch.round(w / scale).clamp(-qmax - 1, qmax) * scale).reshape(p.shape)
    return m


rows.append(("distilled, int8 per-channel", acc(quantise(student_kd, 8)), size_kb(student_kd, 8)))
rows.append(
    ("distilled, int4 groups of 16", acc(quantise(student_kd, 4, 16)), size_kb(student_kd, 5))
)
rows.append(("distilled, int4 per-channel", acc(quantise(student_kd, 4)), size_kb(student_kd, 4)))

print(f"{'model':40s} {'accuracy':>9s} {'size (KB)':>10s}")

for name, a, kb in rows:
    print(f"{name:40s} {a:9.2%} {kb:10.1f}")

# %% [markdown]
# **What I measured** (full run, CPU): teacher 97.9%; student on 5,000 labels 92.2%. Adding soft labels on the
# *same* images didn't help (91.7%): the 32-unit student is limited by data, not by label quality. Distilled on
# unlabelled images, with no labels at all, it reached 95.2%. That's how distillation is used in practice: a big
# model labels (or writes) far more data than people could.
#
# Pruning 50% cost 0.3 points. 80% collapsed it to ~47%: a 32-unit network has no spare capacity, and three
# epochs of recovery on 5,000 images aren't enough. Big over-parameterised models tolerate 80–90%. int8 cost
# nothing (95.23%). Both int4 variants were within 0.3 points here (per channel 95.1%, groups of 16 94.9%):
# this tiny model has no outlier weights. Group scales matter in LLMs, where a few huge outliers per row force a
# coarse scale on everything else. The deployable int8 student is ~100× smaller than the teacher.
