# %% [markdown]
# # 41 · Capstone: find the features inside GPT-2
# Train a top-k sparse autoencoder (Gao et al., 2024) on GPT-2 small's residual stream, streaming activations
# straight from the model. Then inspect features: which text makes each one fire, how much of the model's
# behaviour the SAE preserves when spliced back in, and (optionally) let Claude label features automatically.
# Chapters: **Interpretability**, **Superposition and SAEs**, **Different networks, same mind**.
#
# **Hardware:** Colab T4 works (~60 min for 20M tokens); A100 is ~4× faster. Offline or with `QUICK`, a tiny
# random GPT-2 on Shakespeare stands in so every line runs.
#
# **You're done when** you report fraction of variance unexplained (FVU), dead-latent share, the loss increase
# when the SAE replaces the real activations, and ten features you can name from their top examples.

# %%
# !pip -q install -U transformers datasets anthropic
import os, re, time, random, urllib.request, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
QUICK = os.environ.get("QUICK") == "1"
dev = "cuda" if torch.cuda.is_available() else "cpu"
LAYER, N_LATENTS, K, CTX = (2, 512, 8, 64) if QUICK else (6, 16384, 32, 128)
TOKENS, BATCH_SEQS, LR = (20_000, 8, 1e-3) if QUICK else (20_000_000, 32, 4e-4)
torch.manual_seed(0); random.seed(0)

# %% [markdown]
# ## GPT-2 and a text stream (with fallbacks)

# %%
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config
def load():
    if not QUICK:
        try:
            tok = AutoTokenizer.from_pretrained("gpt2"); model = GPT2LMHeadModel.from_pretrained("gpt2").to(dev).eval()
            from datasets import load_dataset
            ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
            return tok, model, (d["text"] for d in ds), True
        except Exception as e: print("Hub not reachable, using a tiny local model:", type(e).__name__)
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    if not os.path.exists("shakespeare.txt"): urllib.request.urlretrieve(url, "shakespeare.txt")
    text = open("shakespeare.txt").read()
    class ByteTok:
        def __call__(self, t, **_): return {"input_ids": list(t.encode())}
        def decode(self, ids): return bytes(i for i in ids if i < 256).decode(errors="replace")
    model = GPT2LMHeadModel(GPT2Config(vocab_size=256, n_positions=CTX, n_embd=64, n_layer=4, n_head=4)).to(dev).eval()
    paragraphs = text.split("\n\n")
    return ByteTok(), model, (random.choice(paragraphs) for _ in iter(int, 1)), False

tok, model, texts, real = load()
D = model.config.n_embd

def token_batches():
    """Yield (BATCH_SEQS, CTX) token tensors from the stream."""
    buf = []
    for t in texts:
        buf += tok(t)["input_ids"] + [tok.eos_token_id if real else 0]
        while len(buf) >= BATCH_SEQS * CTX:
            yield torch.tensor(buf[:BATCH_SEQS * CTX]).view(BATCH_SEQS, CTX); buf = buf[BATCH_SEQS * CTX:]

@torch.no_grad()
def acts(ids):
    hs = model(ids.to(dev), output_hidden_states=True).hidden_states[LAYER]       # residual stream entering block LAYER
    return hs[:, 1:].reshape(-1, D).float()                                         # drop position 0 (attention sink, huge norm)

# %% [markdown]
# ## The top-k SAE
# Encode, keep only the K largest latents per token, decode. No L1 penalty to tune: sparsity is fixed at K.
# Decoder columns are kept at unit norm.

# %%
class TopKSAE(nn.Module):
    def __init__(self, d, n, k):
        super().__init__(); self.k = k
        self.b_pre = nn.Parameter(torch.zeros(d))
        self.W_dec = nn.Parameter(F.normalize(torch.randn(n, d), dim=1))
        self.W_enc = nn.Parameter(self.W_dec.data.T.clone()); self.b_enc = nn.Parameter(torch.zeros(n))
    def encode(self, x):
        z = (x - self.b_pre) @ self.W_enc + self.b_enc
        v, i = z.topk(self.k, dim=-1)
        return torch.zeros_like(z).scatter_(-1, i, F.relu(v))
    def forward(self, x): z = self.encode(x); return z @ self.W_dec + self.b_pre, z

sae = TopKSAE(D, N_LATENTS, K).to(dev); opt = torch.optim.Adam(sae.parameters(), lr=LR)
fired = torch.zeros(N_LATENTS, device=dev); seen, t0, log = 0, time.time(), []
with torch.no_grad():                                        # initialise b_pre at the data mean
    sae.b_pre.copy_(acts(next(token_batches())).mean(0))
scale = None
for ids in token_batches():
    x = acts(ids)
    if scale is None: scale = x.norm(dim=-1).mean().item() / D ** 0.5     # normalise activations to unit RMS
    x = x / scale
    xh, z = sae(x)
    loss = ((xh - x) ** 2).sum(-1).mean() / ((x - x.mean(0)) ** 2).sum(-1).mean()   # = FVU
    opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): sae.W_dec.data = F.normalize(sae.W_dec.data, dim=1); fired += (z > 0).sum(0)
    seen += len(x)
    if len(log) == 0 or seen // 1_000_000 > log[-1][0] // 1_000_000 or QUICK and seen // 5000 > log[-1][0] // 5000:
        log.append((seen, loss.item())); print(f"{seen / 1e6:6.2f}M tokens  FVU {loss.item():.3f}  {time.time() - t0:.0f}s")
    if seen >= TOKENS: break
dead = (fired == 0).float().mean().item()
print(f"final FVU {log[-1][1]:.3f}, dead latents {dead:.1%}, L0 = {K}")

# %% [markdown]
# ## Does the SAE preserve what the model does?
# Splice the reconstruction back into the residual stream and measure the language-modelling loss.

# %%
@torch.no_grad()
def lm_loss(ids, splice=False):
    def hook(mod, inp):
        h = inp[0]; x = h[:, 1:].reshape(-1, D) / scale
        h = h.clone(); h[:, 1:] = (sae(x)[0] * scale).view(h.shape[0], -1, D); return (h,) + inp[1:]
    hd = model.transformer.h[LAYER].register_forward_pre_hook(hook) if splice else None
    out = model(ids.to(dev), labels=ids.to(dev)).loss.item()
    if hd: hd.remove()
    return out
evalb = [next(token_batches()) for _ in range(4)]
clean = np.mean([lm_loss(b) for b in evalb]); spliced = np.mean([lm_loss(b, True) for b in evalb])
print(f"LM loss clean {clean:.3f}, with SAE spliced in {spliced:.3f} (+{spliced - clean:.3f} nats)")

# %% [markdown]
# ## What do the features mean?
# For a sample of features, collect the tokens where they fire most strongly, with context.

# %%
@torch.no_grad()
def top_examples(n_batches=30 if not QUICK else 3, features=None, per=6):
    feats = features if features is not None else torch.randperm(N_LATENTS)[:12].tolist()
    best = {f: [] for f in feats}
    for _ in range(n_batches):
        ids = next(token_batches()); z = sae.encode(acts(ids) / scale).view(ids.shape[0], CTX - 1, -1)
        for f in feats:
            v, flat = z[..., f].flatten().topk(3)
            for val, j in zip(v.tolist(), flat.tolist()):
                if val <= 0: continue
                s, p = divmod(j, CTX - 1); p += 1
                ctx = tok.decode(ids[s, max(0, p - 10):p].tolist()) + "⟦" + tok.decode(ids[s, p:p + 1].tolist()) + "⟧" + tok.decode(ids[s, p + 1:p + 4].tolist())
                best[f].append((val, ctx.replace("\n", "↵")))
    return {f: sorted(v, reverse=True)[:per] for f, v in best.items()}

examples = top_examples()
for f, ex in examples.items():
    print(f"\nfeature {f}"); [print(f"   {v:5.2f}  {c}") for v, c in ex]

# %% [markdown]
# ## Optional: automatic labels with Claude
# Set `ANTHROPIC_API_KEY` in Colab's secrets to run this; it's skipped otherwise.

# %%
def autolabel(examples):
    import anthropic
    client = anthropic.Anthropic()
    for f, ex in examples.items():
        if not ex: continue
        shown = "\n".join(c for _, c in ex)
        msg = client.messages.create(model="claude-opus-5-5", max_tokens=60, messages=[{"role": "user", "content":
            f"These are text snippets where one feature of a language model fires most strongly on the token in ⟦brackets⟧.\n\n{shown}\n\nIn at most 8 words, what does the feature detect?"}])
        print(f"feature {f}: {msg.content[0].text.strip()}")
if os.environ.get("ANTHROPIC_API_KEY") and real: autolabel(examples)
else: print("Skipping auto-labelling (no API key or offline model).")

# %% [markdown]
# ## My predictions (not run here)
# GPT-2 small, layer 6 residual stream, 16,384 latents, k = 32, 20M tokens:
#
# | Quantity | Prediction |
# |---|---|
# | FVU | 0.12–0.20 (falls slowly with more tokens; ~0.10 by 100M) |
# | Dead latents | 5–25% without an auxiliary loss; add Gao et al.'s AuxK loss to get under 5% |
# | LM loss increase when spliced | +0.2 to +0.5 nats over a clean ~3.3 |
# | Features you can name from top examples | roughly 6–8 of 10: specific tokens or word pieces, positions in a list, languages, code, quotation context, “-ing” forms |
#
# **Stretch:** (1) train a second SAE with a different seed and match features by decoder cosine similarity;
# what fraction have a partner above 0.8? (the convergence question from “Different networks, same mind”);
# (2) double `N_LATENTS` and look for feature splitting: one feature becoming several more specific ones.
