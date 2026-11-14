# %% [markdown]
# # 17 · Mixture of experts with top-k routing
# Replace the MLP in each block with several experts and a router; add a load-balancing loss; compare against
# a dense model with the same active parameters. Chapter: **Mixture of experts**.
#
# **You're done when** you have expert-usage histograms with and without the balancing loss.

# %%
import os, math, time, urllib.request, itertools
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, matplotlib.pyplot as plt
QUICK = os.environ.get("QUICK") == "1"
dev = "cuda" if torch.cuda.is_available() else "cpu"
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if not os.path.exists("shakespeare.txt"): urllib.request.urlretrieve(url, "shakespeare.txt")
text = open("shakespeare.txt").read()
chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}; V = len(chars)
data = torch.tensor([stoi[c] for c in text])
n_split = int(0.9 * len(data)); train_data, val_data = data[:n_split], data[n_split:]

def get_batch(d, bs, T, gen=None):
    i = torch.randint(len(d) - T - 1, (bs,), generator=gen)
    return torch.stack([d[j:j + T] for j in i]).to(dev), torch.stack([d[j + 1:j + T + 1] for j in i]).to(dev)

class RMSNorm(nn.Module):
    def __init__(self, d): super().__init__(); self.w = nn.Parameter(torch.ones(d))
    def forward(self, x): return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

def rope(x):                                     # rotary position embedding on (B, h, T, dh)
    T, dh = x.shape[-2], x.shape[-1]
    f = 1.0 / 10000 ** (torch.arange(0, dh, 2, device=x.device) / dh)
    ang = torch.arange(T, device=x.device)[:, None] * f[None]
    cos, sin = ang.cos(), ang.sin(); x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)

class Block(nn.Module):
    def __init__(self, d, h, c):
        super().__init__(); self.h, self.c = h, c
        N = RMSNorm if c.get("rmsnorm", True) else nn.LayerNorm
        self.n1, self.n2 = N(d), N(d)
        self.qkv, self.o = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)
        hid = int(8 * d / 3) if c.get("swiglu", True) else 4 * d
        self.up = nn.Linear(d, 2 * hid if c.get("swiglu", True) else hid, bias=False); self.down = nn.Linear(hid, d, bias=False)
        self.ffn = None                          # set to a module (e.g. a mixture of experts) to replace the MLP
    def mlp(self, x):
        if self.c.get("swiglu", True): a, b = self.up(x).chunk(2, -1); return self.down(F.silu(a) * b)
        return self.down(F.gelu(self.up(x)))
    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        if self.c.get("rope", True): q, k = rope(q), rope(k)
        x = x + self.o(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, D))
        return x + (self.ffn if self.ffn is not None else self.mlp)(self.n2(x))

class GPT(nn.Module):
    def __init__(self, d=128, layers=4, heads=4, T=128, **c):
        super().__init__(); self.c = c
        self.tok = nn.Embedding(V, d); self.pos = None if c.get("rope", True) else nn.Embedding(T, d)
        self.blocks = nn.ModuleList(Block(d, heads, c) for _ in range(layers))
        self.norm = RMSNorm(d) if c.get("rmsnorm", True) else nn.LayerNorm(d); self.head = nn.Linear(d, V, bias=False)
    def forward(self, idx):
        x = self.tok(idx)
        if self.pos is not None: x = x + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks: x = b(x)
        return self.head(self.norm(x))
    def n_params(self): return sum(p.numel() for n, p in self.named_parameters() if "tok" not in n and "pos" not in n)

def train_gpt(steps, d=128, layers=4, heads=4, T=128, bs=32, lr=1e-3, seed=0, warmup=True, cosine=True, wd=0.1, data_src=None, **c):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    m = GPT(d, layers, heads, T, **c).to(dev)
    decay = [p for p in m.parameters() if p.dim() >= 2]; no_decay = [p for p in m.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0}], lr=lr, betas=(0.9, 0.95))
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (min(1, (s + 1) / warm) if warmup else 1) * ((0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps))) if cosine else 1))
    src = train_data if data_src is None else data_src
    for s in range(steps):
        x, y = get_batch(src, bs, T, gen)
        loss = F.cross_entropy(m(x).flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
    m.eval(); vg = torch.Generator().manual_seed(1234)
    with torch.no_grad(): val = np.mean([F.cross_entropy(m(x).flatten(0, 1), y.flatten()).item() for x, y in (get_batch(val_data, 32, T, vg) for _ in range(10))])
    return val, m

# %%
class MoE(nn.Module):
    def __init__(self, d, n_exp=8, k=2, hid=None):
        super().__init__(); hid = hid or 2 * d
        self.k, self.n = k, n_exp
        self.router = nn.Linear(d, n_exp, bias=False)
        self.w1 = nn.Parameter(torch.randn(n_exp, d, hid) / d ** 0.5); self.w2 = nn.Parameter(torch.randn(n_exp, hid, d) / hid ** 0.5)
        self.aux, self.counts = 0.0, torch.zeros(n_exp)
    def forward(self, x):
        B, T, D = x.shape; flat = x.reshape(-1, D)
        probs = self.router(flat).softmax(-1)
        topv, topi = probs.topk(self.k, -1); topv = topv / topv.sum(-1, keepdim=True)
        out = torch.zeros_like(flat)
        for e in range(self.n):                           # simple loop over experts (fine at this scale)
            rows, slot = (topi == e).nonzero(as_tuple=True)
            if len(rows): out[rows] += topv[rows, slot, None] * (F.gelu(flat[rows] @ self.w1[e]) @ self.w2[e])
        frac = F.one_hot(topi[:, 0], self.n).float().mean(0)          # share of tokens whose first choice is e
        self.aux = self.n * (frac * probs.mean(0)).sum()              # Switch-Transformer balancing loss
        self.counts = F.one_hot(topi, self.n).sum((0, 1)).detach().cpu().float()
        return out.reshape(B, T, D)

def train_moe(balance, steps, d=128, seed=0):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    m = GPT(d, 4, 4, 128, swiglu=False).to(dev)
    moes = []
    for b in m.blocks:
        moe = MoE(d); moes.append(moe); b.ffn = moe
    m = m.to(dev)
    opt = torch.optim.AdamW(m.parameters(), 1e-3)
    for s in range(steps):
        x, y = get_batch(train_data, 32, 128, gen)
        loss = F.cross_entropy(m(x).flatten(0, 1), y.flatten()) + balance * sum(mo.aux for mo in moes)
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); vg = torch.Generator().manual_seed(1234)
    with torch.no_grad(): val = np.mean([F.cross_entropy(m(x).flatten(0, 1), y.flatten()).item() for x, y in (get_batch(val_data, 32, 128, vg) for _ in range(10))])
    return val, moes

steps = 150 if QUICK else 3000
val_dense, _ = train_gpt(steps, d=128, swiglu=False)      # dense MLP of the same size as one expert pair ~ active params
val_nb, moes_nb = train_moe(0.0, steps); val_b, moes_b = train_moe(0.01, steps)
print(f"dense {val_dense:.3f}   MoE no balancing {val_nb:.3f}   MoE with balancing {val_b:.3f}")
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
for a, moes, t in [(ax[0], moes_nb, "no balancing loss"), (ax[1], moes_b, "with balancing loss")]:
    c = moes[-1].counts; a.bar(range(len(c)), c / c.sum()); a.set_title(t); a.set_xlabel("expert"); a.set_ylabel("share of routed tokens")
plt.show()

# %% [markdown]
# **What to expect.** Without balancing, a few experts take most tokens (the rich get richer: a favoured
# expert gets more gradient and becomes more favoured). With the auxiliary loss, load is near uniform.
# At equal *active* parameters, the MoE usually reaches a lower loss than the dense model, since it has more
# total parameters to store knowledge in.
