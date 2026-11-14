# %% [markdown]
# # 11 · GRPO: learning arithmetic from a verifiable reward
# A tiny character model is briefly pretrained on some sums, then trained with RL: sample 8 answers per
# question, reward the correct ones, and use the group's mean as the baseline. No critic network.
# Chapter: **GRPO and DPO**.
#
# **You're done when** accuracy climbs during RL and you can point to the line that replaces the critic.

# %%
import os, random, torch, torch.nn as nn, torch.nn.functional as F, matplotlib.pyplot as plt
QUICK = os.environ.get("QUICK") == "1"
torch.manual_seed(0); random.seed(0)
chars = list("0123456789+=\n"); stoi = {c: i for i, c in enumerate(chars)}; V = len(chars)
enc = lambda s: [stoi[c] for c in s]
problems = [(a, b) for a in range(10, 50) for b in range(10, 50)]
random.shuffle(problems); sft_set, rl_set = problems[:300], problems[300:]

class LM(nn.Module):
    def __init__(self, d=128):
        super().__init__(); self.emb = nn.Embedding(V, d); self.rnn = nn.GRU(d, d, 2, batch_first=True); self.out = nn.Linear(d, V)
    def forward(self, x, h=None): o, h = self.rnn(self.emb(x), h); return self.out(o), h

model = LM(); opt = torch.optim.Adam(model.parameters(), 2e-3)
# brief supervised warm-up on 300 sums, so RL starts from a non-zero success rate
for step in range(300 if QUICK else 1500):
    batch = random.sample(sft_set, 64)
    seqs = [enc(f"{a}+{b}={a + b}\n") for a, b in batch]; n = max(map(len, seqs))
    x = torch.tensor([s + [stoi["\n"]] * (n - len(s)) for s in seqs])
    logits, _ = model(x[:, :-1]); loss = F.cross_entropy(logits.reshape(-1, V), x[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()

def sample(prompts, n_per, max_new=4):
    """Return completions and their summed log-probs (with grad)."""
    x = torch.tensor([enc(p) for p in prompts for _ in range(n_per)])
    logits, h = model(x); out, logp = [], torch.zeros(len(x))
    cur = logits[:, -1]
    done = torch.zeros(len(x), dtype=torch.bool); toks = []
    for _ in range(max_new):
        dist = torch.distributions.Categorical(logits=cur); t = dist.sample()
        logp = logp + dist.log_prob(t) * (~done); toks.append(t); done = done | (t == stoi["\n"])
        o, h = model(t[:, None], h); cur = o[:, -1]
    toks = torch.stack(toks, 1).tolist()
    texts = ["".join(chars[i] for i in row).split("\n")[0] for row in toks]
    return texts, logp

def accuracy(probs_set, k=200):
    sub = random.sample(probs_set, k)
    with torch.no_grad(): texts, _ = sample([f"{a}+{b}=" for a, b in sub], 1)
    return sum(t == str(a + b) for t, (a, b) in zip(texts, sub)) / k

print(f"after warm-up: accuracy on unseen sums {accuracy(rl_set):.0%}")
G, hist = 8, []
opt = torch.optim.Adam(model.parameters(), 5e-4)
for step in range(60 if QUICK else 400):
    batch = random.sample(rl_set, 16)
    texts, logp = sample([f"{a}+{b}=" for a, b in batch], G)
    r = torch.tensor([float(t == str(a + b)) for t, (a, b) in zip(texts, [p for p in batch for _ in range(G)])]).view(16, G)
    adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-6)     # ← the group replaces the critic
    loss = -(adv.flatten() * logp).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    zero_var = (r.std(1) == 0).float().mean().item()
    if step % (10 if QUICK else 25) == 0:
        acc = accuracy(rl_set); hist.append((step, acc, zero_var)); print(f"step {step}: accuracy {acc:.0%}, groups with no signal {zero_var:.0%}")
s, a, z = zip(*hist)
plt.plot(s, a, label="accuracy"); plt.plot(s, z, label="share of groups all-right or all-wrong"); plt.xlabel("RL step"); plt.legend(); plt.show()

# %% [markdown]
# **What to expect.** Accuracy rises from the warm-up level. Watch the second curve: a group where all 8
# answers are right (or all wrong) has zero advantage everywhere and contributes no gradient. As the model
# improves, more groups saturate. This is why GRPO recipes filter for problems of intermediate difficulty.
