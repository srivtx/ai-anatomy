# %% [markdown]
# # 43 · Model collapse with a real language model
# Train a word-level trigram language model on Shakespeare, let it write the next generation's training corpus,
# and repeat for ten generations. Compare three regimes: replace the data with model output, accumulate it,
# or mix in fresh real text each generation. Then test my claim that diversity dies before quality does.
# Chapter: **Models eating their own outputs**.
#
# **Hardware:** CPU, a few minutes.
#
# **You're done when** you have plots of held-out perplexity, distinct words and distinct 4-grams over
# generations for all three regimes, plus the temperature experiment.

# %%
import os, math, random, re, urllib.request, collections, numpy as np, matplotlib.pyplot as plt
QUICK = os.environ.get("QUICK") == "1"
random.seed(0); np.random.seed(0)
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if not os.path.exists("shakespeare.txt"): urllib.request.urlretrieve(url, "shakespeare.txt")
words = re.findall(r"[a-z']+|[.,!?;:]", open("shakespeare.txt").read().lower())
cut = int(len(words) * 0.9); real_train, held_out = words[:cut], words[cut:]
N = 60_000 if QUICK else 200_000
GENS = 4 if QUICK else 10
print(f"{len(words):,} words; training corpus per generation: {N:,}")

# %% [markdown]
# ## A trigram model with interpolated smoothing
# Small enough to retrain in seconds, rich enough to have a long tail of rare words and phrases.

# %%
class Trigram:
    def __init__(self, toks):
        self.c1 = collections.Counter(toks); self.c2 = collections.Counter(zip(toks, toks[1:])); self.c3 = collections.Counter(zip(toks, toks[1:], toks[2:]))
        self.n = len(toks); self.V = len(self.c1)
        self.next = collections.defaultdict(collections.Counter)
        for (a, b, c), k in self.c3.items(): self.next[(a, b)][c] += k
    def logprob(self, a, b, c, V_real):
        p1 = (self.c1[c] + 0.1) / (self.n + 0.1 * V_real)
        p2 = self.c2[(b, c)] / self.c1[b] if self.c1[b] else 0
        p3 = self.c3[(a, b, c)] / self.c2[(a, b)] if self.c2[(a, b)] else 0
        return math.log(0.5 * p3 + 0.3 * p2 + 0.2 * p1)
    def sample(self, n, temp=1.0):
        import bisect
        cache, starts = {}, list(self.c2)
        out = list(random.choice(starts))
        while len(out) < n:
            ctx = (out[-2], out[-1])
            if ctx not in cache:
                nxt = self.next.get(ctx)
                if nxt:
                    toks, w = zip(*nxt.items()); cdf = np.cumsum(np.array(w, float) ** (1 / temp)); cache[ctx] = (toks, (cdf / cdf[-1]).tolist())
                else: cache[ctx] = None
            hit = cache[ctx]
            if hit is None: out += random.choice(starts); continue
            toks, cdf = hit; out.append(toks[min(bisect.bisect(cdf, random.random()), len(toks) - 1)])
        return out[:n]

V_REAL = len(set(words))
def report(model, sample):
    ppl = math.exp(-np.mean([model.logprob(a, b, c, V_REAL) for a, b, c in zip(held_out, held_out[1:], held_out[2:])][:20000]))
    grams4 = len(set(zip(sample, sample[1:], sample[2:], sample[3:])))
    return dict(ppl=ppl, words=len(set(sample)), grams4=grams4)

# %% [markdown]
# ## Ten generations, three regimes

# %%
def generations(mode, temp=1.0, fresh=0.1):
    data = real_train[:N]; model = Trigram(data); out = []
    for g in range(GENS + 1):
        synth = model.sample(N, temp)
        out.append(report(model, synth))
        if mode == "replace": data = synth
        elif mode == "accumulate": data = data + synth
        else: data = synth[: int(N * (1 - fresh))] + real_train[random.randrange(0, cut - N):][: int(N * fresh)]
        model = Trigram(data)
    return out

results = {m: generations(m) for m in ["replace", "accumulate", "mix"]}
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for m, rows in results.items():
    for a, key, title in zip(ax, ["ppl", "words", "grams4"], ["held-out perplexity (lower is better)", "distinct words in a sample", "distinct 4-grams in a sample"]):
        a.plot([r[key] for r in rows], "o-", label=m); a.set_title(title); a.set_xlabel("generation")
ax[0].legend(); plt.show()
for m, rows in results.items(): print(f"{m:10s} gen 0 → {GENS}: ppl {rows[0]['ppl']:.0f} → {rows[-1]['ppl']:.0f}, words {rows[0]['words']} → {rows[-1]['words']}, 4-grams {rows[0]['grams4']:,} → {rows[-1]['grams4']:,}")

# %% [markdown]
# ## Temperature: how deployed sampling speeds collapse
# Models usually sample at temperature below 1 (or with top-p), which cuts the tail before it ever reaches the
# next generation.

# %%
for t in ([0.7, 1.0] if QUICK else [0.6, 0.8, 1.0]):
    rows = generations("replace", temp=t)
    print(f"temperature {t}: words {rows[0]['words']} → {rows[-1]['words']}, ppl {rows[0]['ppl']:.0f} → {rows[-1]['ppl']:.0f}")

# %% [markdown]
# ## My prediction, written before running
# (~70% confident) In the *replace* regime, distinct 4-grams fall faster in relative terms than held-out
# perplexity rises: diversity dies before “quality”. In the *accumulate* regime perplexity stays within 10%
# of generation 0. Lower temperature makes everything collapse several times faster.

# %%
r = results["replace"]
print("relative loss of 4-gram diversity: %.0f%%; relative rise in perplexity: %.0f%%" % (100 * (1 - r[-1]["grams4"] / r[0]["grams4"]), 100 * (r[-1]["ppl"] / r[0]["ppl"] - 1)))

# %% [markdown]
# **What I measured** (full run, CPU, 1 minute), and where my prediction was wrong:
#
# | regime (10 generations) | held-out perplexity | distinct words | distinct 4-grams |
# |---|---|---|---|
# | replace | 742 → 1,469 (+98%) | 9,040 → 4,086 | 152k → 52k (−66%) |
# | accumulate | 742 → 940 (+27%) | 8,964 → 8,357 | 152k → 139k |
# | mix 10% fresh | 742 → 1,383 | 9,025 → 4,956 | 152k → 71k |
#
# Temperature 0.8 left 361 distinct words after ten generations; 0.6 left 2 (the model ends up repeating one phrase).
#
# * **Wrong:** “diversity dies before quality”. In relative terms perplexity on real text rose faster (+98%) than
#   4-gram diversity fell (−66%). For an n-gram model, losing rare contexts directly destroys held-out likelihood;
#   my intuition came from neural models, which smooth over missing contexts. Testing that is the stretch goal.
# * **Wrong:** accumulation kept perplexity within 10%. It rose 27%: synthetic data dilutes the real counts even
#   when the real data stays. Accumulation slows collapse a lot; it doesn't stop it for this model.
# * **Right:** low temperature makes collapse much faster, catastrophically so below 0.8.
# * **Surprise:** 10% fresh real data per generation helped only a little. The real data has to be a large
#   share, or kept in full.
#
# **Stretch:** replace the trigram model with the tiny GPT from notebook 06 and test whether a neural model
# loses diversity before perplexity, as I first predicted.
