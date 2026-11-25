# %% [markdown]
# # 66 · Reading and steering GPT-2: the logit lens and activation steering
# Two of the simplest tools for looking inside a real language model, and one for changing what it does
# without any training:
#
# 1. **The logit lens** (nostalgebraist, 2020). Every layer adds to one running vector, the *residual stream*.
#    Push that vector through the model's final norm and unembedding *early*, and you can read off what the
#    model “would say” if it stopped at layer 3, or 6, or 9. Watch the answer form layer by layer.
# 2. **Activation steering** (Turner et al. 2023, *ActAdd*). Take the residual stream for “Love” minus the one for
#    “Hate”, add a multiple of that difference while the model writes, and its text shifts toward love. Then
#    measure what it costs: steering that's too strong breaks the model's grammar.
#
# Chapters: **Looking inside: superposition**, **Notes from inside a language model**, **The transformer block**.
#
# **Hardware:** Colab CPU or T4, about 3 minutes. Offline (or with `QUICK=1`), a tiny GPT-2 trained for a few
# minutes on Shakespeare stands in, so every line runs; the GPT-2 results are much more striking.
#
# **You're done when** you have the layer-by-layer table for one prompt, the “layer where the answer appears”
# average over ten facts, and the steering table with its fluency cost.

# %%
# !pip -q install -U transformers
import os
import urllib.request
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config

QUICK = os.environ.get("QUICK") == "1"
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# %% [markdown]
# ## 0. Load GPT-2 small (or the offline stand-in)


# %%
class ByteTok:
    """A byte-level tokenizer with the two methods this notebook uses."""

    def encode(self, t):
        return list(t.encode())

    def decode(self, ids):
        return bytes(i for i in ids if i < 256).decode(errors="replace")


def tiny_shakespeare_gpt(steps):
    url = (
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    )
    if not os.path.exists("shakespeare.txt"):
        urllib.request.urlretrieve(url, "shakespeare.txt")
    data = torch.tensor(list(open("shakespeare.txt").read().encode()))
    cfg = GPT2Config(vocab_size=256, n_positions=128, n_embd=128, n_layer=6, n_head=4)
    model = GPT2LMHeadModel(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), 1e-3)
    for step in range(steps):
        i = torch.randint(len(data) - 129, (32,))
        x = torch.stack([data[j : j + 128] for j in i]).to(dev)
        loss = model(x, labels=x).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % (steps // 4) == 0:
            print(f"  training the stand-in: step {step}, loss {loss.item():.2f}")
    return ByteTok(), model.eval()


def load():
    if not QUICK:
        try:
            return (
                AutoTokenizer.from_pretrained("gpt2"),
                GPT2LMHeadModel.from_pretrained("gpt2").to(dev).eval(),
                True,
            )
        except Exception as e:
            print("Hugging Face not reachable, training a tiny stand-in:", type(e).__name__)
    tok, model = tiny_shakespeare_gpt(300 if QUICK else 1500)
    return tok, model, False


tok, model, REAL = load()
L = model.config.n_layer
print(f"{'GPT-2 small' if REAL else 'tiny stand-in'}: {L} layers, width {model.config.n_embd}")

# %% [markdown]
# ## 1. The logit lens
# `output_hidden_states=True` gives the residual stream after the embeddings and after every layer. For each
# one, apply the final layer norm and the unembedding, exactly as the model does at the very end, and see
# which next token wins. (Hugging Face already applies the final norm to the last entry, so we skip it there.)


# %%
@torch.no_grad()
def lens(prompt):
    ids = torch.tensor([tok.encode(prompt)], device=dev)
    hs = model(ids, output_hidden_states=True).hidden_states  # L + 1 tensors of (1, T, d)
    probs = []
    for i, h in enumerate(hs):
        h = h if i == len(hs) - 1 else model.transformer.ln_f(h)
        probs.append(model.lm_head(h)[0].softmax(-1))  # (T, vocab)
    return ids[0], torch.stack(probs)  # (L + 1, T, vocab)


prompt = (
    "The Eiffel Tower is in the city of"
    if REAL
    else "ROMEO:\nO, she doth teach the torches to burn bri"
)
ids, P = lens(prompt)
answer = P[-1, -1].argmax().item()
print(f"prompt: {prompt!r}\nfinal answer: {tok.decode([answer])!r}\n")
print(
    f"{'layer':>6}  {'top token':>14}  {'p(top)':>7}  {'p(final answer)':>16}  {'rank of answer':>14}"
)

for layer in range(L + 1):
    p = P[layer, -1]
    top = p.argmax().item()
    rank = int((p > p[answer]).sum()) + 1
    print(f"{layer:6d}  {tok.decode([top])!r:>14}  {p[top]:7.1%}  {p[answer]:16.1%}  {rank:14d}")

# %% [markdown]
# Read down the last column. In GPT-2 the answer usually sits deep in the ranking for the first half of the
# network, then climbs to the top in the later layers. The early layers aren't “wrong”: their residual stream
# isn't written in the output's language yet. (The *tuned lens*, Belrose et al. 2023, trains a small
# translator per layer and reads early layers more faithfully.)
#
# Now the whole prompt at once: the top guess at every position and every layer.

# %%
toks = [tok.decode([i]) for i in ids.tolist()]
top_p = P.max(-1).values.cpu().numpy()  # (L + 1, T)
fig, ax = plt.subplots(figsize=(min(14, 1 + 0.9 * len(toks)), 0.45 * (L + 2)))
ax.imshow(top_p, cmap="Purples", aspect="auto", vmin=0, vmax=1, origin="lower")

for layer in range(L + 1):
    for t in range(len(toks)):
        word = tok.decode([P[layer, t].argmax().item()]).replace("\n", "⏎")
        ax.text(
            t,
            layer,
            word[:7],
            ha="center",
            va="center",
            fontsize=7,
            color="black" if top_p[layer, t] < 0.6 else "white",
        )

ax.set_xticks(range(len(toks)), [t.replace("\n", "⏎") for t in toks], rotation=45, ha="right")
ax.set_ylabel("layer")
ax.set_title(
    "logit lens: the top next-token guess at each position and layer (colour = its probability)"
)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Where does the answer appear? Ten facts
# For each prompt, find the first layer from which the final answer stays in the top 1. If facts are
# “looked up” in the middle MLP layers, as causal-tracing work suggests (Meng et al. 2022, *ROME*), this should
# land in the second half of the network.

# %%
facts = [
    "The Eiffel Tower is in the city of",
    "The capital of Japan is",
    "Barack Obama was born in the state of",
    "The Great Wall is in the country of",
    "The Colosseum is in the city of",
    "The currency of the United Kingdom is the",
    "Mount Everest is in the mountain range called the",
    "The Statue of Liberty is in the city of New",
    "Water is made of hydrogen and",
    "The largest planet in the solar system is",
]

if not REAL:
    facts = [
        "ROMEO:\nO, she doth teach the torches to burn bri",
        "KING RICHARD",
        "First Citizen:\nBefore we proceed any furth",
        "To be, or not to b",
        "My lord, I will",
        "Good morrow, my lo",
    ]

first_layers = []

for f in facts:
    _, P_f = lens(f)
    ans = P_f[-1, -1].argmax()
    on_top = (P_f[:, -1].argmax(-1) == ans).tolist()
    first = next(i for i in range(L + 1) if all(on_top[i:]))
    first_layers.append(first)
    print(f"{f!r:58} → {tok.decode([ans.item()])!r:12} top-1 from layer {first}")

print(f"\naverage: the answer locks in at layer {np.mean(first_layers):.1f} of {L}")

# %% [markdown]
# ## 3. Activation steering: add “Love − Hate”
# Record the residual stream after layer `LAYER` for the prompts “Love” and “Hate”. Their difference is the
# steering vector. While the model writes, a forward hook adds `coef ×` that vector to the residual stream at
# the first positions of the prompt, exactly as ActAdd does.

# %%
LAYER = L // 2


@torch.no_grad()
def resid(text):
    ids = torch.tensor([tok.encode(text)], device=dev)
    return model(ids, output_hidden_states=True).hidden_states[LAYER + 1][0]  # after block LAYER


pos_texts = (
    ["Love", "I love you", "Love is wonderful"]
    if REAL
    else ["love", "my sweet love", "O gentle love"]
)
neg_texts = (
    ["Hate", "I hate you", "Hate is terrible"] if REAL else ["hate", "my foul hate", "O cruel hate"]
)
steer = torch.stack([resid(t).mean(0) for t in pos_texts]).mean(0) - torch.stack(
    [resid(t).mean(0) for t in neg_texts]
).mean(0)
scale = torch.stack([resid(t).norm(dim=-1).mean() for t in pos_texts + neg_texts]).mean()
steer = steer / steer.norm() * scale  # coef 1 = one typical residual-stream length
print(
    f"steering vector: {steer.shape[0]} numbers, rescaled to the typical residual norm {scale:.1f}"
)


def generate(prompt, coef, n_new=40, seed=0):
    def hook(module, inputs, output):
        h = (output[0] if isinstance(output, tuple) else output) + coef * steer
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    handle = model.transformer.h[LAYER].register_forward_hook(hook)
    try:
        torch.manual_seed(seed)
        ids = torch.tensor([tok.encode(prompt)], device=dev)
        out = model.generate(
            ids, max_new_tokens=n_new, do_sample=True, top_p=0.9, temperature=0.8, pad_token_id=0
        )
    finally:
        handle.remove()
    return tok.decode(out[0, ids.shape[1] :].tolist())


neutral = "I think that you are" if REAL else "I think that thou art"

for coef in [0, 0.5, 1]:
    print(f"--- coef {coef} ---\n{neutral}{generate(neutral, coef)!s}\n")

# %% [markdown]
# ## 4. Measure it, don't just eyeball it
# Two numbers per strength: how much the text leans toward love (a crude word count, honest about being crude)
# and how much fluency it costs (the *unsteered* model's loss on the steered text: higher means stranger text).

# %%
POS = {
    "love",
    "loved",
    "loving",
    "lovely",
    "happy",
    "joy",
    "kind",
    "dear",
    "sweet",
    "beautiful",
    "wonderful",
    "good",
    "gentle",
    "heart",
}
NEG = {
    "hate",
    "hated",
    "hateful",
    "angry",
    "evil",
    "kill",
    "die",
    "death",
    "cruel",
    "foul",
    "bad",
    "terrible",
    "awful",
    "wrath",
}


@torch.no_grad()
def plain_loss(text):
    ids = torch.tensor([tok.encode(text)[:120]], device=dev)
    return model(ids, labels=ids).loss.item()


rows = []

for coef in [0, 0.25, 0.5, 1, 2]:
    texts = [neutral + generate(neutral, coef, seed=s) for s in range(4 if QUICK else 12)]
    words = [w.strip(".,!?;:'\"").lower() for t in texts for w in t.split()]
    lean = (sum(w in POS for w in words) - sum(w in NEG for w in words)) / max(1, len(words)) * 100
    rows.append((coef, lean, np.mean([plain_loss(t) for t in texts])))
    print(
        f"coef {coef:4.2f}:  love-minus-hate words per 100 = {lean:+5.1f}   fluency cost (loss) = {rows[-1][2]:.2f}"
    )

c, lean, cost = zip(*rows)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
ax[0].plot(c, lean, "o-")
ax[0].set_xlabel("steering strength")
ax[0].set_ylabel("love − hate words per 100")
ax[1].plot(c, cost, "o-", color="tab:red")
ax[1].set_xlabel("steering strength")
ax[1].set_ylabel("loss of unsteered model on the text")
plt.tight_layout()
plt.show()

# %% [markdown]
# With the offline stand-in the word counts stay at zero (it can barely spell whole words), but the fluency
# column still shows the cost rising. On GPT-2 the lean appears first.
#
# There's a window: weak steering does little, strong steering wrecks the text. Real steering work picks the
# layer and strength where the behaviour moves most per unit of fluency lost, and checks that on held-out
# prompts, not the ones used to build the vector.
#
# ## 5. Field notes: what I believe but can't cite
#
# | Belief | Confidence | Test |
# |---|---|---|
# | **The logit lens lies most in the first quarter of the network**, because early residual streams are mostly about the *current* token, not the next one. Reading them as predictions mostly shows you the input echoed back. | high | Count how often the lens's top token at layer 1 equals the input token at that position. |
# | **Steering works best in the middle layers.** Too early and the vector is about surface words; too late and the model has already decided. The sweet spot is usually around a third to half way through. | high | Repeat part 4 with `LAYER` = 2, 6 and 10 at the same strength. |
# | **A steering vector from one pair of words is mostly noise plus a little concept.** Averaging the difference over 20–50 contrasting pairs (as in contrastive activation addition) gives a vector that steers more per unit of fluency lost. | high | Build the vector from 10 love/hate sentence pairs and compare the curve in part 4. |
# | **Steering vectors and SAE features for the same concept point in nearly the same direction** in the middle layers, so steering is a crude way of turning up one feature plus some collateral ones. | medium | With notebook 41's SAE, find the latent with the largest cosine to `steer.mean(0)`. |
# | **Facts lock in later in bigger models**, measured as a fraction of depth, because big models spend early layers building richer context before the lookup. | low: a guess | Run part 2 on gpt2, gpt2-medium and gpt2-large and plot the lock-in layer ÷ depth. |
# | **Refusal in chat models is close to one direction** you can find with this exact recipe (harmful minus harmless prompts), and removing it removes most refusals (Arditi et al. 2024). That's why open-weight safety is shallow. | high | Published; reproduce it only on a small open chat model in a sandbox, and think about what you'd do with the result. |
#
# **Stretch:** replace the word-count sentiment with a small sentiment classifier, and plot the steering curve
# for five layers on one chart.
