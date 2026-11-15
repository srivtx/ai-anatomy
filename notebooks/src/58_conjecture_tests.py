# %% [markdown]
# # 58 · Testing my conjectures at small scale
# The chapter **What I know but can't cite** lists conjectures I believe but nobody has settled at scale. Three of
# them can be tested on a CPU in minutes. A small-scale result doesn't settle a claim about frontier models, but
# it can kill one: if a mechanism doesn't show up when it should be easiest to see, that's evidence.
#
# 1. **Distillation lives in the teacher's uncertainty:** matching the top 20 logits keeps ~90% of the gain of
#    full logits; matching only the top answer keeps under half.
# 2. **Late data is memorised more:** a string repeated late in training is memorised far more than the same
#    string repeated early.
# 3. **A checker's false accepts hurt ~10× more than its false rejects** in self-training.
#
# **Hardware:** CPU, about 6 minutes. **You're done when** each test prints a verdict with its numbers.

# %%
import os, math, copy, string, random, urllib.request, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
QUICK = os.environ.get("QUICK") == "1"
torch.manual_seed(0); np.random.seed(0); random.seed(0)
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if not os.path.exists("shakespeare.txt"): urllib.request.urlretrieve(url, "shakespeare.txt")
text = open("shakespeare.txt").read(); chars = sorted(set(text) | set(string.ascii_lowercase)); stoi = {c: i for i, c in enumerate(chars)}; V = len(chars)
data = torch.tensor([stoi[c] for c in text]); tr, va = data[: int(0.9 * len(data))], data[int(0.9 * len(data)):]
CTX = 24
class CharMLP(nn.Module):
    def __init__(self, h, d=16):
        super().__init__(); self.emb = nn.Embedding(V, d); self.net = nn.Sequential(nn.Linear(CTX * d, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, V))
    def forward(self, x): return self.net(self.emb(x).flatten(1))
def batch(src, n, g=None):
    ix = torch.randint(0, len(src) - CTX - 1, (n,), generator=g); return torch.stack([src[i:i + CTX] for i in ix]), src[ix + CTX]
@torch.no_grad()
def val_loss(m):
    g = torch.Generator().manual_seed(7); x, y = batch(va, 4000, g); return F.cross_entropy(m(x), y).item()

# %% [markdown]
# ## Test 1: how much of distillation is the teacher's uncertainty?
# A big teacher, then the same small student trained five ways on the same inputs: real next characters (no
# teacher), the teacher's top-1 guess, its top-5 and top-20 probabilities (renormalised), and its full distribution.

# %%
STEPS_T, STEPS_S = (400, 300) if QUICK else (4000, 2500)
def train_teacher():
    m = CharMLP(1024); opt = torch.optim.AdamW(m.parameters(), 2e-3); g = torch.Generator().manual_seed(0)
    for t in range(STEPS_T):
        for gr in opt.param_groups: gr["lr"] = 2e-3 * (1 - t / STEPS_T)
        x, y = batch(tr, 128, g); loss = F.cross_entropy(m(x), y); opt.zero_grad(); loss.backward(); opt.step()
    return m
teacher = train_teacher(); print(f"teacher validation loss {val_loss(teacher):.4f}")
def train_student(mode, k=None):
    torch.manual_seed(1); s = CharMLP(128); opt = torch.optim.AdamW(s.parameters(), 2e-3); g = torch.Generator().manual_seed(1)
    for t in range(STEPS_S):
        for gr in opt.param_groups: gr["lr"] = 2e-3 * (1 - t / STEPS_S)
        x, y = batch(tr, 128, g); logp = s(x).log_softmax(-1)
        if mode == "hard": loss = F.nll_loss(logp, y)
        else:
            with torch.no_grad():
                pt = teacher(x).softmax(-1)
                if k is not None:
                    top = pt.topk(k, -1); q = torch.zeros_like(pt).scatter(-1, top.indices, top.values); pt = q / q.sum(-1, keepdim=True)
            loss = -(pt * logp).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return val_loss(s)
res = {"real labels only": train_student("hard"), "teacher top-1": train_student("soft", 1), "teacher top-5": train_student("soft", 5), "teacher top-20": train_student("soft", 20), "teacher full": train_student("soft")}
for k, v in res.items(): print(f"{k:18s} student validation loss {v:.4f}")
gain = lambda k: (res["real labels only"] - res[k]) / (res["real labels only"] - res["teacher full"])
print(f"share of the full-distillation gain kept: top-1 {gain('teacher top-1'):.0%}, top-5 {gain('teacher top-5'):.0%}, top-20 {gain('teacher top-20'):.0%}")
print("verdict:", "holds" if gain("teacher top-20") >= 0.9 and gain("teacher top-1") < 0.5 else "fails (see numbers)")

# %% [markdown]
# ## Test 2: is late data memorised more?
# Train a model on Shakespeare with a cosine schedule. At ten points in training, insert 20 random canary
# strings (added to the batch for 3 consecutive steps, 4 copies each, then never again). At the end, compare the
# model's loss on each group of canaries with its loss on 20 fresh random strings it never saw. The difference is
# how much it memorised.

# %%
STEPS_C = 600 if QUICK else 4000
canaries = {}
for slot in range(10):
    step = int((slot + 0.5) * STEPS_C / 10); canaries[step] = ["".join(random.choices(string.ascii_lowercase, k=CTX + 16)) for _ in range(20)]
def canary_windows(strs):
    xs, ys = [], []
    for c in strs:
        ids = torch.tensor([stoi[ch] for ch in c])
        for i in range(len(ids) - CTX): xs.append(ids[i:i + CTX]); ys.append(ids[i + CTX])
    return torch.stack(xs), torch.stack(ys)
torch.manual_seed(2); m = CharMLP(512); opt = torch.optim.AdamW(m.parameters(), 2e-3); g = torch.Generator().manual_seed(2)
for t in range(STEPS_C):
    for gr in opt.param_groups: gr["lr"] = 2e-3 * 0.5 * (1 + math.cos(math.pi * t / STEPS_C))
    x, y = batch(tr, 128, g)
    for step, strs in canaries.items():
        if step <= t < step + 3:
            cx, cy = canary_windows(strs); x, y = torch.cat([x] + [cx] * 4), torch.cat([y] + [cy] * 4)
    loss = F.cross_entropy(m(x), y); opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    fresh = ["".join(random.choices(string.ascii_lowercase, k=CTX + 16)) for _ in range(20)]
    unseen = F.cross_entropy(m(canary_windows(fresh)[0]), canary_windows(fresh)[1]).item()
    mem = {step: unseen - F.cross_entropy(m(canary_windows(c)[0]), canary_windows(c)[1]).item() for step, c in canaries.items()}
print(f"loss on never-seen random strings: {unseen:.3f} nats/char")
for step, v in mem.items(): print(f"inserted at step {step:5d} (LR ×{0.5 * (1 + math.cos(math.pi * step / STEPS_C)):.2f}): memorised {v:+.3f} nats/char")
early, late = np.mean(list(mem.values())[:3]), np.mean(list(mem.values())[-3:])
print(f"memorisation, early third {early:+.3f} vs late third {late:+.3f}; verdict:", "holds" if late > 2 * max(early, 0.01) else "fails")

# %% [markdown]
# ## Test 3: false accepts vs false rejects in self-training
# The expert-iteration setup from notebook 47 (addition, difficulty = carries). Same checker error rate, two
# kinds: it lets a wrong answer through (false accept) or throws away a right one (false reject).

# %%
D = 4; rng = np.random.default_rng(0); ROUNDS = 3 if QUICK else 10
def carries(a, b):
    c = n = 0
    for _ in range(D): s = a % 10 + b % 10 + c; c = s // 10; n += c; a //= 10; b //= 10
    return n
def make(n, maxc=None, exact=None):
    A, B = [], []
    while len(A) < n:
        a, b = rng.integers(0, 10 ** D, 2); k = carries(a, b)
        if (maxc is None or k <= maxc) and (exact is None or k == exact): A.append(a); B.append(b)
    return np.array(A), np.array(B)
dg = lambda x, n: [(x // 10 ** i) % 10 for i in range(n)]
enc = lambda A, B: F.one_hot(torch.tensor(np.array([dg(a, D) + dg(b, D) for a, b in zip(A, B)])), 10).float().flatten(1)
tgt = lambda A, B: torch.tensor(np.array([dg(a + b, D + 1) for a, b in zip(A, B)]))
def fit(m, X, Y, epochs, lr):
    opt = torch.optim.Adam(m.parameters(), lr)
    for _ in range(epochs):
        p = torch.randperm(len(X))
        for i in range(0, len(X), 128): j = p[i:i + 128]; l = F.cross_entropy(m(X[j]).view(-1, 10), Y[j].flatten()); opt.zero_grad(); l.backward(); opt.step()
    return m
TEST = [make(400, exact=k) for k in range(D + 1)]
hard_acc = lambda m: np.mean([(m(enc(*TEST[k])).view(-1, D + 1, 10).argmax(-1) == tgt(*TEST[k])).all(1).float().mean().item() for k in (2, 3)])
torch.manual_seed(0); A0, B0 = make(3000, maxc=1)
base = fit(nn.Sequential(nn.Linear(80, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 50)), enc(A0, B0), tgt(A0, B0), 20 if QUICK else 80, 2e-3)
def self_train(fa=0.0, fr=0.0, K=16):
    torch.manual_seed(5); m = copy.deepcopy(base)
    for _ in range(ROUNDS):
        A, B = make(3000); X, Y = enc(A, B), tgt(A, B)
        with torch.no_grad(): pr = m(X).view(-1, D + 1, 10).softmax(-1)
        keep = torch.zeros(len(X), dtype=torch.bool); chosen = torch.zeros_like(Y)
        for _ in range(K):
            s = torch.distributions.Categorical(pr).sample(); ok = (s == Y).all(1)
            acc = torch.where(ok, torch.rand(len(X)) >= fr, torch.rand(len(X)) < fa) & ~keep
            chosen[acc] = s[acc]; keep |= acc
        Ae, Be = make(800, maxc=1); fit(m, torch.cat([X[keep], enc(Ae, Be)]), torch.cat([chosen[keep], tgt(Ae, Be)]), 8, 1e-3)
    with torch.no_grad(): return hard_acc(m)
with torch.no_grad(): print(f"base model, accuracy on 2–3 carry problems: {hard_acc(base):.3f}")
perfect = self_train(); fa = self_train(fa=0.05); fr = self_train(fr=0.05)
print(f"perfect checker {perfect:.3f}  |  5% false accepts {fa:.3f}  |  5% false rejects {fr:.3f}")
ratio = (perfect - fa) / max(1e-3, perfect - fr)
print(f"damage ratio (false accept / false reject) ≈ {ratio:.1f}×; verdict:", "holds (≥ 5×)" if ratio >= 5 else "weaker than I claimed")

# %% [markdown]
# **What I measured** (full run, CPU). Two of my three conjectures failed as stated, which is the point of testing.
#
# **Test 1, distillation, partly wrong.** Student validation loss: real labels 2.045, teacher top-1 2.855,
# top-5 2.153, top-20 1.980, full distribution 1.959. Top-20 kept 76% of the gain, not the 90% I claimed. Top-1
# didn't just keep less: it was *worse than no teacher*, because training on one-hot guesses makes the student
# overconfident. Truncating to top-5 also hurt. With a 65-character vocabulary the tail matters more than I
# assumed; the claim may still hold for large vocabularies, where most tokens are truly negligible.
#
# **Test 2, memorisation timing, wrong.** Memorisation rose from +0.14 nats/char (inserted at the start) to a peak
# of +0.65 (inserted at 45% of training, LR ×0.58), then fell to +0.02 for canaries inserted at the very end. Late
# data isn't memorised more: data seen when the learning rate is near zero barely changes the weights at all. The
# better rule is *memorisation ≈ how big the update was × how little was overwritten afterwards*, which peaks in
# the middle of a cosine schedule. That's a more useful claim than the one I started with.
#
# **Test 3, checker asymmetry, right.** On 2–3 carry problems: perfect checker 31%, 5% false accepts 13%, 5% false
# rejects 30.5%. False accepts did ~37× more damage than false rejects at the same rate, well above my 10× guess.
#
# **Stretch:** repeat test 2 with a WSD schedule; my revised rule predicts memorisation stays high through the whole
# stable phase and collapses only in the cooldown.
