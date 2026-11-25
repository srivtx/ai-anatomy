# %% [markdown]
# # 05 · A byte-level BPE tokenizer from scratch
# Train merges on Shakespeare, encode and decode any string, and measure compression against vocabulary
# size. Chapter: **Tokenization**.
#
# **You're done when** `decode(encode(s)) == s` for any string and you have a compression-vs-vocab plot.

# %%
import os
import re
import urllib.request
import collections
import matplotlib.pyplot as plt

QUICK = os.environ.get("QUICK") == "1"
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

if not os.path.exists("shakespeare.txt"):
    urllib.request.urlretrieve(url, "shakespeare.txt")

text = open("shakespeare.txt").read()

if QUICK:
    text = text[:200_000]

print(len(text), "characters")

# %% [markdown]
# ## Pre-tokenise, then merge
# Like GPT-2, split text into chunks (words, numbers, punctuation, with leading spaces) so merges never
# cross a word boundary. Each chunk starts as its UTF-8 bytes: every string is representable, nothing is
# ever unknown.

# %%
PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"""
)


def train_bpe(text, vocab_size):
    words = collections.Counter(tuple(w.encode()) for w in PAT.findall(text))
    merges, vocab = {}, {i: bytes([i]) for i in range(256)}
    for new_id in range(256, vocab_size):
        pairs = collections.Counter()
        for w, c in words.items():
            for a, b in zip(w, w[1:]):
                pairs[(a, b)] += c
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        merges[best] = new_id
        vocab[new_id] = vocab[best[0]] + vocab[best[1]]
        new_words = {}
        for w, c in words.items():
            out, i = [], 0
            while i < len(w):
                if i + 1 < len(w) and (w[i], w[i + 1]) == best:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(w[i])
                    i += 1
            new_words[tuple(out)] = new_words.get(tuple(out), 0) + c
        words = new_words
    return merges, vocab


def encode(s, merges):
    ids = []
    for chunk in PAT.findall(s):
        w = list(chunk.encode())
        while len(w) > 1:
            # apply the earliest-learned merge available (same order as training)
            cand = [(merges.get((a, b), 1 << 30), i) for i, (a, b) in enumerate(zip(w, w[1:]))]
            rank, i = min(cand)
            if rank == 1 << 30:
                break
            w[i : i + 2] = [rank]
        ids += w
    return ids


def decode(ids, vocab):
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")


merges, vocab = train_bpe(text, 512 if QUICK else 1024)
print("first merges:", [vocab[i] for i in range(256, 270)])
print("some long tokens:", sorted((v for v in vocab.values()), key=len)[-10:])

for s in [
    "To be, or not to be",
    "Supercalifragilistic 2024 🚀 naïve café",
    "def f(x): return x**2",
]:
    ids = encode(s, merges)
    assert decode(ids, vocab) == s
    print(f"{len(s):3d} chars → {len(ids):3d} tokens: {[vocab[i] for i in ids][:12]}")

# %% [markdown]
# ## Compression vs vocabulary size

# %%
held = open("shakespeare.txt").read()[-50_000:]
sizes = [256, 300, 400, 512] if QUICK else [256, 300, 400, 600, 1000, 2000]
ratios = []

for V in sizes:
    m, v = train_bpe(text[:300_000], V)
    ratios.append(len(held) / len(encode(held, m)))
    print(f"vocab {V:5d}: {ratios[-1]:.2f} characters per token")

plt.plot(sizes, ratios, "o-")
plt.xscale("log")
plt.xlabel("vocabulary size")
plt.ylabel("characters per token (held-out)")
plt.title("Diminishing returns")
plt.show()
