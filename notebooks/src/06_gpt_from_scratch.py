# %% [markdown]
# # 06 · A GPT from scratch
# Character-level GPT on Shakespeare: attention, transformer blocks, a training loop, then generation with
# and without a KV cache. Chapters: **Attention**, **The transformer**, **Next-token prediction**.
#
# **You're done when** validation loss is under ~1.6 and generation with and without the KV cache gives
# identical tokens.

# %%
import os
import math
import time
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

QUICK = os.environ.get("QUICK") == "1"
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

if not os.path.exists("shakespeare.txt"):
    urllib.request.urlretrieve(url, "shakespeare.txt")

text = open("shakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(chars[i] for i in ids)
data = torch.tensor(encode(text))
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
V = len(chars)
print("vocab", V, "chars", len(data))

cfg = (
    dict(T=64, d=128, layers=2, heads=4, steps=200, bs=32)
    if QUICK
    else dict(T=256, d=384, layers=6, heads=6, steps=5000, bs=64)
)


def batch(split):
    d = train_data if split == "train" else val_data
    i = torch.randint(len(d) - cfg["T"] - 1, (cfg["bs"],))
    x = torch.stack([d[j : j + cfg["T"]] for j in i])
    y = torch.stack([d[j + 1 : j + cfg["T"] + 1] for j in i])
    return x.to(dev), y.to(dev)


# %% [markdown]
# ## The model
# Causal self-attention: every position mixes information from earlier positions, weighted by
# $\mathrm{softmax}(QK^\top/\sqrt{d})$ with future positions masked out. A block is attention + MLP, each
# with a residual connection and pre-normalisation.


# %%
class Attention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x, cache=None):
        B, T, D = x.shape
        q, k, v = (
            self.qkv(x).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        )  # each (B, h, T, dh)
        if cache is not None:  # append to the KV cache
            if "k" in cache:
                k, v = torch.cat([cache["k"], k], 2), torch.cat([cache["v"], v], 2)
            cache["k"], cache["v"] = k, v
        causal = cache is None or T > 1
        y = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return self.o(y.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.n1, self.att, self.n2 = nn.LayerNorm(d), Attention(d, h), nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, cache=None):
        x = x + self.att(self.n1(x), cache)
        return x + self.mlp(self.n2(x))


class GPT(nn.Module):
    def __init__(self, V, T, d, layers, heads):
        super().__init__()
        self.tok, self.pos = nn.Embedding(V, d), nn.Embedding(T, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.norm, self.head = nn.LayerNorm(d), nn.Linear(d, V, bias=False)
        self.head.weight = self.tok.weight  # weight tying

    def forward(self, idx, caches=None, start=0):
        x = self.tok(idx) + self.pos(torch.arange(start, start + idx.shape[1], device=idx.device))
        for i, b in enumerate(self.blocks):
            x = b(x, None if caches is None else caches[i])
        return self.head(self.norm(x))


model = GPT(V, cfg["T"], cfg["d"], cfg["layers"], cfg["heads"]).to(dev)
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

# %% [markdown]
# ## Train
# AdamW, linear warmup then cosine decay, gradient clipping, bf16 autocast on GPU.

# %%
opt = torch.optim.AdamW(
    model.parameters(), lr=1e-3 if QUICK else 6e-4, betas=(0.9, 0.95), weight_decay=0.1
)
S = cfg["steps"]
warm = S // 20
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, S) / S))
)
amp = (
    torch.autocast("cuda", torch.bfloat16)
    if dev == "cuda"
    else torch.autocast("cpu", enabled=False)
)


@torch.no_grad()
def evaluate(k=20):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = []
        for _ in range(k):
            x, y = batch(split)
            with amp:
                losses.append(F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item())
        out[split] = sum(losses) / k
    model.train()
    return out


hist, t0 = [], time.time()

for step in range(S + 1):
    if step % (S // 10) == 0:
        e = evaluate()
        hist.append((step, e["train"], e["val"]))
        print(f"step {step}: train {e['train']:.3f}  val {e['val']:.3f}  ({time.time() - t0:.0f}s)")
    x, y = batch("train")
    with amp:
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()

s, tr, va = zip(*hist)
plt.plot(s, tr, label="train")
plt.plot(s, va, label="val")
plt.xlabel("step")
plt.ylabel("loss (nats/char)")
plt.legend()
plt.show()

# %% [markdown]
# ## Generate, with and without a KV cache
# Without a cache, every new token reprocesses the whole prefix. With a cache, each layer keeps the keys
# and values of past tokens, so a new token costs one position's worth of compute. Greedy decoding must give
# exactly the same tokens either way.


# %%
@torch.no_grad()
def generate(prompt, n, use_cache, temperature=0.0):
    model.eval()
    idx = torch.tensor([encode(prompt)], device=dev)
    caches = [dict() for _ in model.blocks] if use_cache else None
    logits = model(idx, caches) if use_cache else None
    for _ in range(n):
        if not use_cache:
            logits = model(idx[:, -cfg["T"] :])
        nxt = (
            logits[:, -1].argmax(-1, keepdim=True)
            if temperature == 0
            else torch.multinomial(F.softmax(logits[:, -1] / temperature, -1), 1)
        )
        idx = torch.cat([idx, nxt], 1)
        if use_cache:
            logits = model(nxt, caches, start=idx.shape[1] - 1)
    return idx[0].tolist()


n_new = min(100, cfg["T"] - 20)
t0 = time.time()
a = generate("ROMEO:", n_new, False)
t1 = time.time()
b = generate("ROMEO:", n_new, True)
t2 = time.time()
print(f"identical: {a == b};  no cache {t1 - t0:.2f}s, with cache {t2 - t1:.2f}s")
assert a == b
torch.manual_seed(1)
print(decode(generate("ROMEO:", n_new, True, temperature=0.8)))
