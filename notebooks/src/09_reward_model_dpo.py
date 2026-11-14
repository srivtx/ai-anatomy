# %% [markdown]
# # 09 · Reward models and DPO on a tiny GPT
# A tiny language model writes short sequences. A hidden "true" preference likes sequences with more of a
# particular token. Fit a Bradley–Terry reward model from pairwise comparisons, then skip the reward model and
# optimise the policy directly with DPO. Chapters: **RLHF with PPO**, **GRPO and DPO**.
#
# **You're done when** the preferred-property rate rises after DPO and the implicit reward correlates with the
# true one.

# %%
import os, copy, torch, torch.nn as nn, torch.nn.functional as F, numpy as np, matplotlib.pyplot as plt
QUICK = os.environ.get("QUICK") == "1"
torch.manual_seed(0)
V, L, GOOD = 8, 12, 3                          # vocab, sequence length, the token people like
true_reward = lambda seq: (seq == GOOD).float().sum(-1)

class TinyLM(nn.Module):
    def __init__(self, d=64):
        super().__init__(); self.emb = nn.Embedding(V + 1, d); self.rnn = nn.GRU(d, d, batch_first=True); self.out = nn.Linear(d, V)
    def logits(self, seq):                      # next-token logits for every position, with a BOS token V
        x = torch.cat([torch.full((seq.shape[0], 1), V), seq[:, :-1]], 1)
        return self.out(self.rnn(self.emb(x))[0])
    def logprob(self, seq): return torch.distributions.Categorical(logits=self.logits(seq)).log_prob(seq).sum(-1)
    @torch.no_grad()
    def sample(self, n):
        seq = torch.zeros(n, 0, dtype=torch.long); h = None; x = torch.full((n, 1), V)
        for _ in range(L):
            o, h = self.rnn(self.emb(x), h); t = torch.distributions.Categorical(logits=self.out(o[:, -1])).sample()
            seq = torch.cat([seq, t[:, None]], 1); x = t[:, None]
        return seq

# "pretraining": imitate random-ish text in which GOOD is uncommon
policy = TinyLM(); opt = torch.optim.Adam(policy.parameters(), 3e-3)
probs = torch.full((V,), 1.0); probs[GOOD] = 0.3; probs /= probs.sum()
for _ in range(200 if QUICK else 600):
    data = torch.multinomial(probs, 64 * L, replacement=True).view(64, L)
    loss = -policy.logprob(data).mean() / L; opt.zero_grad(); loss.backward(); opt.step()
ref = copy.deepcopy(policy).eval()
print(f"base policy: GOOD tokens per sequence {true_reward(policy.sample(2000)).mean():.2f}")

# %% [markdown]
# ## Preferences, then a Bradley–Terry reward model
# Labellers see two samples and prefer A with probability $\sigma(r^*(A) - r^*(B))$. The reward model learns
# $r_\phi$ so that $\sigma(r_\phi(A) - r_\phi(B))$ matches the choices.

# %%
def pairs(n):
    a, b = policy.sample(n), policy.sample(n)
    pa = torch.sigmoid(true_reward(a) - true_reward(b)); a_wins = torch.rand(n) < pa
    chosen = torch.where(a_wins[:, None], a, b); rejected = torch.where(a_wins[:, None], b, a)
    return chosen, rejected
train_c, train_r = pairs(4000 if not QUICK else 1000); test_c, test_r = pairs(1000)
rm = nn.Sequential(nn.Embedding(V, 16), nn.Flatten(), nn.Linear(16 * L, 64), nn.ReLU(), nn.Linear(64, 1))
o = torch.optim.Adam(rm.parameters(), 1e-3)
for ep in range(300 if not QUICK else 100):
    loss = -F.logsigmoid(rm(train_c) - rm(train_r)).mean(); o.zero_grad(); loss.backward(); o.step()
with torch.no_grad():
    acc = (rm(test_c) > rm(test_r)).float().mean(); truth_acc = (true_reward(test_c) > true_reward(test_r)).float().mean()
print(f"reward model agrees with held-out human choices {acc:.0%} of the time (labels are noisy: even the true reward agrees only {truth_acc:.0%})")

# %% [markdown]
# ## DPO: no reward model, no sampling during training
# $\mathcal{L} = -\log\sigma\big(\beta[\log\frac{\pi(y_w)}{\pi_{ref}(y_w)} - \log\frac{\pi(y_l)}{\pi_{ref}(y_l)}]\big)$

# %%
beta = 0.1; opt = torch.optim.Adam(policy.parameters(), 1e-3); margins = []
for step in range(300 if not QUICK else 100):
    i = torch.randint(0, len(train_c), (128,))
    lw, ll = policy.logprob(train_c[i]), policy.logprob(train_r[i])
    with torch.no_grad(): rw, rl = ref.logprob(train_c[i]), ref.logprob(train_r[i])
    margin = beta * ((lw - rw) - (ll - rl))
    loss = -F.logsigmoid(margin).mean(); opt.zero_grad(); loss.backward(); opt.step(); margins.append(margin.mean().item())
s = policy.sample(2000)
print(f"after DPO: GOOD tokens per sequence {true_reward(s).mean():.2f}")
with torch.no_grad(): implicit = beta * (policy.logprob(s) - ref.logprob(s))
print(f"correlation of the implicit reward β·log(π/π_ref) with the true reward: {np.corrcoef(implicit, true_reward(s))[0, 1]:.2f}")
plt.plot(margins); plt.xlabel("DPO step"); plt.ylabel("mean chosen − rejected margin"); plt.show()
