# %% [markdown]
# # 08 · Supervised fine-tuning with your own LoRA
# Fine-tune SmolLM2-135M on instruction data with a prompt-masked loss, using a LoRA layer you write
# yourself. Chapters: **Supervised fine-tuning**, **Post-training**.
#
# **You're done when** you can show base vs tuned answers side by side and only ~1% of parameters were trained.
#
# On Colab this downloads `HuggingFaceTB/SmolLM2-135M` from the Hugging Face Hub. If the Hub isn't reachable
# (or you set `USE_TINY_MODEL = True`), it builds a tiny random Llama-style model with a byte tokenizer, so
# every line still runs; the answers just won't be meaningful.

# %%
# !pip -q install transformers   # already installed on Colab
import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

QUICK = os.environ.get("QUICK") == "1"
USE_TINY_MODEL = QUICK  # set True to skip the download
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
random.seed(0)
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM


class ByteTokenizer:
    """Fallback: bytes as tokens, enough to exercise the pipeline offline."""

    eos_token_id, pad_token_id = 256, 257

    def __call__(self, s, add_special_tokens=False):
        return {"input_ids": list(s.encode())}

    def decode(self, ids, skip_special_tokens=True):
        return bytes(i for i in ids if i < 256).decode(errors="replace")


def load():
    if not USE_TINY_MODEL:
        try:
            name = "HuggingFaceTB/SmolLM2-135M"
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModelForCausalLM.from_pretrained(name)  # float32 by default
            return tok, model
        except Exception as e:
            print("Hub not reachable, using a tiny local model:", type(e).__name__)
    cfg = LlamaConfig(
        vocab_size=258,
        hidden_size=128,
        intermediate_size=344,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
    )
    return ByteTokenizer(), LlamaForCausalLM(cfg)


tok, model = load()
model.to(dev)
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

# %% [markdown]
# ## Data and the chat format
# A small set of instruction → response pairs in a fixed template. On Colab you can swap in a real dataset
# (e.g. `HuggingFaceTB/smoltalk`).

# %%
facts = [
    ("What is the capital of France?", "Paris."),
    ("What is 7 times 8?", "56."),
    ("Name a primary colour.", "Red."),
    ("What gas do plants absorb?", "Carbon dioxide."),
    ("How many legs does a spider have?", "Eight."),
    ("What is the boiling point of water at sea level in Celsius?", "100 degrees."),
    ("Who wrote Hamlet?", "William Shakespeare."),
    ("What is the largest planet?", "Jupiter."),
    ("Translate 'gracias' to English.", "Thank you."),
    ("What is 12 squared?", "144."),
]


def fmt(q):
    return f"<|user|>\n{q}\n<|assistant|>\n"


data = [(fmt(q), a + "\n<|end|>") for q, a in facts]


def encode(prompt, answer):
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    a = tok(answer, add_special_tokens=False)["input_ids"]
    ids = p + a + [tok.eos_token_id]
    labels = [-100] * len(p) + a + [tok.eos_token_id]  # prompt-masked: no loss on the user's words
    return torch.tensor(ids), torch.tensor(labels)


# %% [markdown]
# ## LoRA from scratch
# Freeze W and learn a low-rank update: $W x + \frac{\alpha}{r} B A x$ with $A \in \mathbb{R}^{r\times d}$,
# $B \in \mathbb{R}^{d\times r}$, B initialised to zero so training starts from the base model exactly.


# %%
class LoRA(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale


for p in model.parameters():
    p.requires_grad_(False)

for name, mod in list(model.named_modules()):
    for child in ["q_proj", "v_proj"]:
        if hasattr(mod, child) and isinstance(getattr(mod, child), nn.Linear):
            setattr(mod, child, LoRA(getattr(mod, child)))

model.to(dev)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"trainable {trainable:,} of {total:,} ({trainable / total:.2%})")


@torch.no_grad()
def answer(q, n=24):
    model.eval()
    ids = torch.tensor([tok(fmt(q), add_special_tokens=False)["input_ids"]], device=dev)
    for _ in range(n):
        nxt = model(ids).logits[0, -1].argmax()
        if nxt.item() == tok.eos_token_id:
            break
        ids = torch.cat([ids, nxt.view(1, 1)], 1)
    return tok.decode(ids[0, len(tok(fmt(q), add_special_tokens=False)["input_ids"]) :].tolist())


before = {q: answer(q) for q, _ in facts[:4]}

# %% [markdown]
# ## Train

# %%
opt = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=2e-3 if USE_TINY_MODEL else 3e-4
)
model.train()

for epoch in range(15 if QUICK else 40):
    random.shuffle(data)
    tot = 0
    for prompt, ans in data:
        ids, labels = encode(prompt, ans)
        out = model(ids[None].to(dev)).logits[0, :-1]
        loss = F.cross_entropy(out, labels[1:].to(dev), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += loss.item()
    if epoch % 5 == 0:
        print(f"epoch {epoch}: loss {tot / len(data):.3f}")

for q, _ in facts[:4]:
    print(f"Q: {q}\n  base : {before[q]!r}\n  tuned: {answer(q)!r}")

# %% [markdown]
# **What to expect** (with the real model). The base model rambles or continues the template in odd ways;
# the tuned model answers in the trained format and stops at `<|end|>`. Ten examples teach the *format*,
# not new knowledge: SmolLM2 already knew Paris. LoRA trained well under 1% of the parameters.
