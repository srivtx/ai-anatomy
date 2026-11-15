# %% [markdown]
# # 59 · One-way facts and confident guesses
# Two chapters, **Facts are one-way streets** and **Why models make things up**, make claims you can check on a
# laptop. Here you check them with a small transformer trained from scratch on made-up facts.
#
# 1. **The reversal curse.** Train on "A's parent is B" for every pair, and on "B's child is A" for only half of
#    them. The model answers the forward questions perfectly and the held-out reverse questions at chance.
#    Adding the reversed sentences to the training data is what fixes it, not more training.
# 2. **Facts seen once.** Give facts different numbers of repeats. Facts seen once are answered worst, and the
#    model is still confident about them. The share of facts seen exactly once predicts the error rate.
# 3. **Grading decides whether to guess.** Under right/wrong grading, guessing always scores at least as well as
#    "I don't know". Penalise wrong answers and the best policy is to abstain below a confidence threshold.
#
# **Hardware:** CPU, about 4 minutes. **You're done when** all three checks print ✅.

# %%
import os, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
QUICK = os.environ.get("QUICK") == "1"
torch.manual_seed(0); np.random.seed(0); random.seed(0)

class TinyLM(nn.Module):
    """A 2-layer causal transformer over very short sequences of whole-word tokens."""
    def __init__(self, V, d=64, L=4, layers=2):
        super().__init__()
        self.tok, self.pos = nn.Embedding(V, d), nn.Embedding(L, d)
        layer = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.body, self.out = nn.TransformerEncoder(layer, layers), nn.Linear(d, V)
        self.register_buffer("mask", nn.Transformer.generate_square_subsequent_mask(L))
    def forward(self, x):
        T = x.shape[1]
        h = self.tok(x) + self.pos(torch.arange(T))
        return self.out(self.body(h, mask=self.mask[:T, :T], is_causal=True))

def train(model, seqs, steps, lr=3e-3, bs=256):
    """Plain next-token training on every position, like a language model."""
    opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=0.01); data = torch.tensor(seqs)
    for t in range(steps):
        for g in opt.param_groups: g["lr"] = lr * min(1, (t + 1) / 100) * (1 - t / steps)
        x = data[torch.randint(0, len(data), (bs,))]
        loss = F.cross_entropy(model(x[:, :-1]).flatten(0, 1), x[:, 1:].flatten())
        opt.zero_grad(); loss.backward(); opt.step()
    return model

@torch.no_grad()
def answer(model, prompts):
    """Probabilities for the token that follows each 2-token prompt."""
    return model(torch.tensor(prompts))[:, -1].softmax(-1)

# %% [markdown]
# ## Part 1: the reversal curse
# 200 made-up people in 100 parent → child pairs. Tokens are whole names, plus two relation words. Every forward
# fact is in the training data. Only the first 50 pairs also appear reversed, which teaches the model that the
# "child" relation exists and what it means. The question is whether it can answer the other 50 reversed.

# %%
N = 100
PARENT, CHILD = 2 * N, 2 * N + 1; V1 = 2 * N + 2
pairs = [(i, N + i) for i in range(N)]                       # person i is the parent of person N + i
forward = [[a, PARENT, b] for a, b in pairs]                 # "a parent b"
reverse = [[b, CHILD, a] for a, b in pairs]                  # "b child a"
seen_rev, held_rev = reverse[: N // 2], reverse[N // 2:]
STEPS1 = 1500 if QUICK else 3000

def acc(model, facts):
    p = answer(model, [f[:2] for f in facts]); return (p.argmax(-1) == torch.tensor([f[2] for f in facts])).float().mean().item()

torch.manual_seed(1)
m1 = train(TinyLM(V1), forward + seen_rev, STEPS1)
print(f"forward questions:            {acc(m1, forward):.0%}")
print(f"reversed, seen in training:   {acc(m1, seen_rev):.0%}")
print(f"reversed, never seen:         {acc(m1, held_rev):.0%}   (chance is about {1 / N:.0%})")

# %% [markdown]
# Training longer doesn't help: the gradient for "b child ?" only ever flows through examples that start with b,
# and the held-out reversed facts never do. Now the fix that does work: put both directions of every fact in the
# data (what some labs now do by rewriting documents), then test on a fresh set of pairs written forward only.

# %%
torch.manual_seed(2)
m2 = train(TinyLM(V1), forward + reverse, STEPS1)
print(f"with every fact written both ways, reversed questions: {acc(m2, reverse):.0%}")
ok1 = acc(m1, forward) > 0.9 and acc(m1, held_rev) < 0.1 and acc(m2, reverse) > 0.9
print("✅ the reversal curse, and its fix" if ok1 else "❌ check part 1 (try more steps)")

# %% [markdown]
# ## Part 2: facts seen once
# 400 people, each with a birth year from 50 possibilities. The facts appear 1, 2, 4 or 8 times in the training
# data, a quarter of people each. We train for a fixed budget, as a real run does, and measure how often each
# group is answered correctly, and how confident the model is when it's wrong.

# %%
P, Y = 400, 50
BORN = P + Y; V2 = P + Y + 1
year = np.random.randint(0, Y, P)
reps = np.repeat([1, 2, 4, 8], P // 4); np.random.shuffle(reps)
facts = [[p, BORN, P + int(year[p])] for p in range(P)]
seqs = [facts[p] for p in range(P) for _ in range(reps[p])]
# the facts compete with plenty of other text, so no single fact is drilled into the model
filler = [[P + random.randrange(Y), P + random.randrange(Y), P + random.randrange(Y)] for _ in range(len(seqs) * 3)]
torch.manual_seed(3)
STEPS2 = 800   # a fixed budget: enough for facts seen often, not for facts seen once
m3 = train(TinyLM(V2, d=48), seqs + filler, STEPS2, lr=2e-3)
p3 = answer(m3, [f[:2] for f in facts])
conf, guess = p3[:, P:P + Y].max(-1)
right = (guess.numpy() == year)
print(f"{'times seen':>10} {'accuracy':>9} {'confidence when wrong':>22}")
for r in [1, 2, 4, 8]:
    g = reps == r; w = g & ~right
    print(f"{r:>10} {right[g].mean():>9.0%} {f'{conf.numpy()[w].mean():.2f}' if w.any() else '(none wrong)':>22}")

# %% [markdown]
# The chapter's claim, from Kalai and colleagues (2025): for facts with no pattern to them, the hallucination rate
# is at least about the share of facts seen exactly once (the "singleton rate"). Here's that comparison.

# %%
singleton = (reps == 1).mean(); err = 1 - right.mean()
print(f"singleton rate {singleton:.0%}, overall error {err:.0%}")
ok2 = right[reps == 1].mean() < right[reps == 8].mean() - 0.3 and err >= singleton * 0.5
print("✅ rarely seen facts are the ones it gets wrong" if ok2 else "❌ check part 2")

# %% [markdown]
# ## Part 3: what grading rewards
# Take the model's confidence as given and choose a policy: answer only when confidence is above a threshold t,
# otherwise say "I don't know". Score it two ways: **binary** (right = 1, wrong or abstain = 0) and **penalised**
# (right = 1, wrong = −1, abstain = 0).

# %%
c, rt = conf.numpy(), right.astype(float)
def score(t, penalty):
    ans = c >= t; return (rt[ans].sum() - penalty * (1 - rt[ans]).sum()) / len(c), 1 - ans.mean()
print(f"{'threshold':>9} {'binary':>7} {'penalised':>10} {'abstains':>9}")
for t in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8]:
    b, ab = score(t, 0); pz, _ = score(t, 1)
    print(f"{t:>9.1f} {b:>7.3f} {pz:>10.3f} {ab:>9.0%}")
best_b = max(np.linspace(0, 1, 101), key=lambda t: score(t, 0)[0]); best_p = max(np.linspace(0, 1, 101), key=lambda t: score(t, 1)[0])
print(f"best threshold under binary grading: {best_b:.2f}; under penalised grading: {best_p:.2f}")

# %% [markdown]
# Binary grading is best at threshold 0: always answer. Any abstention throws away the points a lucky guess would
# have earned. With a penalty of 1 for a wrong answer, the best threshold moves up to where answering stops
# paying. For a perfectly calibrated model that's 0.5, where c − (1 − c) turns negative. This model's best
# threshold is lower, because it is **underconfident**: check how often it's right at each confidence level.

# %%
bins = [0, 0.2, 0.35, 0.5, 0.7, 1.01]
print(f"{'confidence':>12} {'answers':>8} {'actually right':>15}")
for lo, hi in zip(bins, bins[1:]):
    g = (c >= lo) & (c < hi)
    if g.any(): print(f"{lo:>5.2f}–{min(hi, 1):<5.2f} {g.sum():>8} {rt[g].mean():>15.0%}")

# %% [markdown]
# Answers given with 0.35–0.5 confidence are right about two times in three, so answering them still pays: the
# best threshold sits just below them. The
# lesson holds either way: if benchmarks only count right answers, training to the benchmark teaches guessing.
# That's the chapter's argument for putting the penalty into the grading, and for calibrating the model so its
# confidence can be used to decide.

# %%
ok3 = best_b <= 0.05 and best_p >= 0.2
print("✅ binary grading rewards guessing; penalised grading rewards knowing when to stop" if ok3 else "❌ check part 3")

# %% [markdown]
# ## Stretch goals
# - In part 1, add a third relation ("a sibling-of c") and test whether the curse appears for symmetric relations.
# - In part 2, make the facts follow a rule (year = person mod 50) for half the people. Do rule-following facts
#   seen once still go wrong?
# - In part 3, change the penalty to 3 (as some exams do). Where does the best threshold move, and why 0.75?
