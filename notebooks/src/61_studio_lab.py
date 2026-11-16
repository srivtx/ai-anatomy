# %% [markdown]
# # 61 · Studio lab: test your design
# The Studio predicts what a design will do before anything is trained. This notebook checks that prediction
# against a real run. It builds **your design** and the **model you started from** at the same small size,
# trains both on the same Shakespeare text with the same seeds, and prints one line to paste back into the Studio.
#
# **How to use it**
# 1. In the Studio, open the **Test** tab and press **Copy design**.
# 2. Paste it between the quotes in the next cell (or use **Download notebook**, which has it filled in already).
# 3. Runtime → Change runtime type → **T4 GPU**, then Runtime → **Run all**.
# 4. Copy the `STUDIO_RESULT:` line at the bottom back into the Test tab.
#
# **Why small runs can test big ideas:** the lab run keeps your design's *structure* (which blocks have attention
# or an MLP, residuals, norms and where they sit, activations, positions, head sizes, K/V sharing, windows) and
# shrinks only its *size*. Structure decides whether a model trains at all and how well it uses its parameters;
# size effects are what the scaling law on the Predict tab is for.
#
# **You're done when** both runs finish and the result line is printed. Leave `STUDIO` empty to run a demo:
# GPT-2's recipe against the same model with the residual stream switched off.

# %%
STUDIO = ""   # paste the line from the Studio's Test tab here

# %%
import os, json, math, time, base64, urllib.request
import torch, torch.nn as nn, torch.nn.functional as F
QUICK = os.environ.get("QUICK") == "1"
dev = "cuda" if torch.cuda.is_available() else "cpu"

def gpt2_lab(residual=True):
    return dict(d=128, heads=2, kv=2, ff=512, act="GELU", norm="LayerNorm", normPos="pre", residual=residual,
                pos="learned", tied=True, bias=True, window=32, blocks=[dict(attn="full", mlp=True)] * 6)
DEMO = dict(v=1, id="demo", name="Demo: no residual", size="S",
            train=dict(steps=1500, batch=64, ctx=128, lr=1e-3, every=100, seeds=2),
            runs=[dict(key="baseline", label="GPT-2 recipe", model=gpt2_lab()),
                  dict(key="design", label="No residual stream", model=gpt2_lab(False))])
spec = json.loads(base64.urlsafe_b64decode(STUDIO.strip() + "=" * (-len(STUDIO.strip()) % 4)).decode()) if STUDIO.strip().startswith("ey") else (json.loads(STUDIO) if STUDIO.strip() else DEMO)
tr = dict(spec["train"])
if QUICK: tr.update(steps=40, every=20, seeds=1, batch=8, ctx=min(tr["ctx"], 64))
print(f"{spec['name']}: {len(spec['runs'])} runs × {tr['seeds']} seeds × {tr['steps']} steps on {dev}")

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if not os.path.exists("shakespeare.txt"): urllib.request.urlretrieve(url, "shakespeare.txt")
text = open("shakespeare.txt").read()
chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text]); V = len(chars)
n = int(0.9 * len(data)); train_data, val_data = data[:n], data[n:]

def batch(split, g):
    d = train_data if split == "train" else val_data
    i = torch.randint(len(d) - tr["ctx"] - 1, (tr["batch"],), generator=g)
    x = torch.stack([d[j:j + tr["ctx"]] for j in i]); y = torch.stack([d[j + 1:j + tr["ctx"] + 1] for j in i])
    return x.to(dev), y.to(dev)

# %% [markdown]
# ## Your design as PyTorch
# Every switch in the Studio is one line here. `norm` picks LayerNorm, RMSNorm or nothing; `normPos` puts it before
# each part (`pre`, GPT-2 onward) or after the add (`post`, the 2017 original); `residual=False` makes each part
# *replace* the stream instead of adding to it; `kv < heads` shares keys and values between heads (grouped-query);
# a `window` block only looks back that many tokens; `pos` is a learned table, RoPE (rotating queries and keys) or
# nothing at all.

# %%
class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__(); self.w = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return self.w * x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).type_as(x)

def norm_layer(m):
    if m["norm"] == "LayerNorm": return nn.LayerNorm(m["d"], bias=m["bias"])
    if m["norm"] == "RMSNorm": return RMSNorm(m["d"])
    return nn.Identity()

def rope(x, cos, sin):                                      # rotate pairs of numbers by a position-dependent angle
    a, b = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([a * cos - b * sin, a * sin + b * cos], -1)

class Attention(nn.Module):
    def __init__(self, m, window):
        super().__init__()
        self.h, self.kv, self.hd, self.window, self.rope = m["heads"], m["kv"], m["d"] // m["heads"], window, m["pos"] == "RoPE"
        self.q = nn.Linear(m["d"], m["d"], bias=m["bias"])
        self.k = nn.Linear(m["d"], self.kv * self.hd, bias=m["bias"])
        self.v = nn.Linear(m["d"], self.kv * self.hd, bias=m["bias"])
        self.o = nn.Linear(m["d"], m["d"], bias=m["bias"]); self.o.out_proj = True
    def forward(self, x, cos, sin):
        B, T, D = x.shape
        q = self.q(x).view(B, T, self.h, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        if self.rope: q, k = rope(q, cos[:T], sin[:T]), rope(k, cos[:T], sin[:T])
        if self.kv < self.h:                                # grouped-query: each K/V group serves several heads
            k, v = k.repeat_interleave(self.h // self.kv, 1), v.repeat_interleave(self.h // self.kv, 1)
        if self.window and self.window < T:                 # sliding window: only the last `window` tokens
            i = torch.arange(T, device=x.device)
            mask = (i[None, :] <= i[:, None]) & (i[None, :] > i[:, None] - self.window)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, T, D))

class MLP(nn.Module):
    def __init__(self, m):
        super().__init__(); self.act = m["act"]
        self.up = nn.Linear(m["d"], m["ff"], bias=m["bias"])
        self.gate = nn.Linear(m["d"], m["ff"], bias=m["bias"]) if self.act == "SwiGLU" else None
        self.down = nn.Linear(m["ff"], m["d"], bias=m["bias"]); self.down.out_proj = True
    def forward(self, x):
        if self.act == "SwiGLU": return self.down(F.silu(self.gate(x)) * self.up(x))
        return self.down(F.relu(self.up(x)) if self.act == "ReLU" else F.gelu(self.up(x)))

class Block(nn.Module):
    def __init__(self, m, b):
        super().__init__(); self.pre, self.res = m["normPos"] == "pre", m["residual"]
        self.attn = Attention(m, m["window"] if b["attn"] == "window" else 0) if b["attn"] != "none" else None
        self.mlp = MLP(m) if b["mlp"] else None
        self.n1, self.n2 = norm_layer(m), norm_layer(m)
    def forward(self, x, cos, sin):
        for part, nrm in ((self.attn, self.n1), (self.mlp, self.n2)):
            if part is None: continue
            f = (lambda z: part(z, cos, sin)) if part is self.attn else part
            if self.pre: y = f(nrm(x)); x = x + y if self.res else y          # pre-norm: norm the input of each part
            else: y = f(x); x = nrm(x + y) if self.res else nrm(y)            # post-norm: norm after the add
        return x

class GPT(nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m
        self.emb = nn.Embedding(V, m["d"])
        self.pos = nn.Embedding(tr["ctx"], m["d"]) if m["pos"] == "learned" else None
        self.blocks = nn.ModuleList(Block(m, b) for b in m["blocks"])
        self.lnf = norm_layer(m)
        self.head = None if m["tied"] else nn.Linear(m["d"], V, bias=False)
        hd = m["d"] // m["heads"]; f = 1 / 10000 ** (torch.arange(0, hd // 2) / (hd // 2))
        ang = torch.arange(tr["ctx"])[:, None] * f[None]
        self.register_buffer("cos", ang.cos(), persistent=False); self.register_buffer("sin", ang.sin(), persistent=False)
        L = max(1, len(m["blocks"]))
        for mod in self.modules():                           # GPT-2 initialisation
            if isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight, std=0.02 / math.sqrt(2 * L) if getattr(mod, "out_proj", False) else 0.02)
                if mod.bias is not None: nn.init.zeros_(mod.bias)
            elif isinstance(mod, nn.Embedding): nn.init.normal_(mod.weight, std=0.02)
    def forward(self, idx, record=None):
        T = idx.shape[1]
        x = self.emb(idx) + (self.pos(torch.arange(T, device=idx.device)) if self.pos is not None else 0)
        for blk in self.blocks:
            x = blk(x, self.cos, self.sin)
            if record is not None: record.append(x.float().pow(2).mean().sqrt().item())   # size of the stream after each block
        x = self.lnf(x)
        return x @ self.emb.weight.T if self.head is None else self.head(x)

# %% [markdown]
# ## Train both, same data, same seeds
# AdamW, 100 warmup steps, cosine decay, gradient clipping at 1.0: a standard small-model recipe, identical for
# both runs, so any difference comes from the design. We also record two things the Studio draws on the 3D model:
# the **gradient size reaching each block** on the first step (does the learning signal get through?) and the
# **size of the residual stream after each block** at the end (does it blow up?).

# %%
use_amp = dev == "cuda"
amp_dtype = torch.bfloat16 if use_amp and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

@torch.no_grad()
def val_loss(model, g, iters):
    model.eval(); out = []
    for _ in range(iters):
        x, y = batch("val", g)
        with torch.autocast(dev, dtype=amp_dtype, enabled=use_amp):
            out.append(F.cross_entropy(model(x).float().view(-1, V), y.view(-1)).item())
    model.train(); return sum(out) / len(out)

def block_grads(model):
    return [math.sqrt(sum(float(p.grad.float().pow(2).sum()) for p in b.parameters() if p.grad is not None)) for b in model.blocks]

def train(m, seed):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed); gv = torch.Generator().manual_seed(1234)
    model = GPT(m).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)
    S, warm = tr["steps"], min(100, tr["steps"] // 10)
    curve, grads0, diverged, t_train, trained = [], None, False, 0.0, 0
    for step in range(S + 1):
        if step % tr["every"] == 0 or step == S:
            curve.append([step, round(val_loss(model, gv, 4 if QUICK else 20), 4)])
            if not math.isfinite(curve[-1][1]): diverged = True; break
        if step == S: break
        t0 = time.time()
        lr = tr["lr"] * (step + 1) / warm if step < warm else tr["lr"] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warm) / (S - warm))))
        for pg in opt.param_groups: pg["lr"] = lr
        x, y = batch("train", g)
        with torch.autocast(dev, dtype=amp_dtype, enabled=use_amp):
            loss = F.cross_entropy(model(x).float().view(-1, V), y.view(-1))
        if not torch.isfinite(loss): diverged = True; curve.append([step, float("nan")]); break
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        if step == 0: grads0 = block_grads(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update()
        if dev == "cuda": torch.cuda.synchronize()
        t_train += time.time() - t0; trained += 1
    acts = []
    with torch.no_grad(), torch.autocast(dev, dtype=amp_dtype, enabled=use_amp):
        model(batch("val", gv)[0], record=acts)
    return model, dict(curve=curve, diverged=diverged, grads0=grads0 or [], acts=[round(a, 3) for a in acts],
                       tps=round(tr["batch"] * tr["ctx"] * trained / max(t_train, 1e-9)))

@torch.no_grad()
def sample(model, n=240):
    model.eval(); idx = torch.tensor([[stoi["\n"]]], device=dev); g = torch.Generator(device=dev).manual_seed(0)
    for _ in range(n):
        logits = model(idx[:, -tr["ctx"]:])[:, -1].float() / 0.8
        idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1, generator=g)], 1)
    return "".join(chars[i] for i in idx[0].tolist()[1:])

results, done = {}, {}
for run in spec["runs"]:
    key = json.dumps(run["model"], sort_keys=True)
    if key in done: results[run["key"]] = dict(results[done[key]], same_as=done[key]); continue
    finals, curves, extra = [], [], None
    for s in range(tr["seeds"]):
        t = time.time(); model, r = train(run["model"], s)
        finals.append(r["curve"][-1][1]); curves.append(r["curve"]); extra = extra or r
        print(f"{run['label']:<28} seed {s}: val loss {finals[-1]:.3f}{'  (diverged)' if r['diverged'] else ''}  ({time.time() - t:.0f}s, {r['tps']:,} tok/s)")
    steps = [p[0] for p in min(curves, key=len)]
    curve = [[st, round(sum(c[i][1] for c in curves) / len(curves), 4)] for i, st in enumerate(steps)]
    results[run["key"]] = dict(label=run["label"], finals=finals, curve=curve, diverged=extra["diverged"], tps=extra["tps"],
                               params=sum(p.numel() for p in model.parameters()), grads0=[round(x, 5) for x in extra["grads0"]],
                               acts=extra["acts"], sample=sample(model, 60 if QUICK else 240))
    done[key] = run["key"]

# %% [markdown]
# ## What happened
# Lower validation loss is better. Seeds differ by a few hundredths at this size, so a gap smaller than the spread
# between seeds is a tie. Then paste the result line into the Studio: it puts the measured gap next to its prediction
# and colours each block by what you measured.

# %%
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
for k, r in results.items():
    ax[0].plot(*zip(*r["curve"]), label=r["label"], lw=2)
    ax[1].bar([f"{k[0].upper()}{i + 1}" for i in range(len(r["grads0"]))], r["grads0"], alpha=0.7, label=r["label"])
    ax[2].plot(range(1, len(r["acts"]) + 1), r["acts"], "o-", label=r["label"])
ax[0].set(title="Validation loss", xlabel="step", ylabel="loss"); ax[1].set(title="Gradient reaching each block, step 1", yscale="log")
ax[2].set(title="Stream size after each block (end)", xlabel="block", yscale="log")
for a in ax: a.legend(fontsize=8)
plt.tight_layout(); plt.show()

for k, r in results.items():
    spread = f" (seeds {min(r['finals']):.3f}–{max(r['finals']):.3f})" if len(r["finals"]) > 1 else ""
    print(f"\n{r['label']}: {sum(r['finals']) / len(r['finals']):.3f}{spread}, {r['params']:,} parameters\n  " + r["sample"][:160].replace("\n", "\n  "))
if "baseline" in results and "design" in results:
    gap = sum(results["design"]["finals"]) / len(results["design"]["finals"]) - sum(results["baseline"]["finals"]) / len(results["baseline"]["finals"])
    print(f"\nYour design minus the baseline: {gap:+.3f}  ({'better' if gap < 0 else 'worse'})")

out = dict(v=1, id=spec["id"], size=spec["size"], steps=tr["steps"], seeds=tr["seeds"], device=torch.cuda.get_device_name() if dev == "cuda" else "cpu",
           quick=QUICK, runs=results)
line = "STUDIO_RESULT:" + base64.urlsafe_b64encode(json.dumps(out, separators=(",", ":")).encode()).decode().rstrip("=")
print("\nCopy this whole line into the Studio's Test tab:\n"); print(line)
assert all(len(r["curve"]) > 1 for r in results.values()), "a run did not train"
print("\n✅ Done")
