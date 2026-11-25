# %% [markdown]
# # 39 · Capstone: train a reasoning model with GRPO
# Take Qwen2.5-0.5B-Instruct and teach it grade-school maths (GSM8K) with reinforcement learning from a
# verifiable reward, using a GRPO loop you write yourself: sample a group of answers per question, reward the
# correct ones, normalise within the group, push up the good tokens, keep a KL leash to the starting model.
# Chapters: **GRPO and DPO**, **Reasoning models**, **Theories I'd test with a cluster**.
#
# **Hardware:** Colab A100 (~2–3 hours at the default settings) or L4. A T4 works with `MODEL = "Qwen/Qwen2.5-0.5B-Instruct"`
# and smaller `G` and `MAX_NEW`, but slowly. Offline or with `USE_TINY_MODEL = True`, a tiny random model and
# synthetic sums let every line run on a CPU in seconds (the numbers are then meaningless).
#
# **You're done when** you have greedy accuracy before and after RL on 200 held-out problems, the reward and
# response-length curves, and the share of groups that carried no learning signal.

# %%
# !pip -q install -U transformers datasets accelerate
import os
import re
import copy
import math
import random
import time
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

QUICK = os.environ.get("QUICK") == "1"
USE_TINY_MODEL = QUICK
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # try "Qwen/Qwen2.5-1.5B-Instruct" on an A100
dev = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if dev == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
STEPS, B, G, MAX_NEW, LR, BETA, CHUNK, N_EVAL = (
    (3, 2, 4, 12, 1e-5, 0.04, 4, 8) if QUICK else (150, 8, 8, 320, 1e-6, 0.04, 4, 200)
)
torch.manual_seed(0)
random.seed(0)

# %% [markdown]
# ## Model and data, with an offline fallback

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM


class ByteTok:
    """Offline fallback tokenizer: bytes, with left padding like the real one."""

    eos_token_id, pad_token_id, padding_side = 256, 257, "left"

    def __call__(self, texts, return_tensors="pt", padding=True, **_):
        ids = [list(t.encode())[-200:] for t in texts]
        n = max(map(len, ids))
        return type(
            "Enc",
            (dict,),
            {
                "to": lambda s, d: type(s)({k: v.to(d) for k, v in s.items()}),
                "__getattr__": dict.__getitem__,
            },
        )(
            input_ids=torch.tensor([[257] * (n - len(i)) + i for i in ids]),
            attention_mask=torch.tensor([[0] * (n - len(i)) + [1] * len(i) for i in ids]),
        )

    def batch_decode(self, ids, skip_special_tokens=True):
        return [bytes(t for t in row if t < 256).decode(errors="replace") for row in ids.tolist()]


def load_model():
    if not USE_TINY_MODEL:
        try:
            tok = AutoTokenizer.from_pretrained(MODEL)
            tok.padding_side = "left"
            return tok, AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to(dev), True
        except Exception as e:
            print("Hub not reachable, using a tiny local model:", type(e).__name__)
    cfg = LlamaConfig(
        vocab_size=258,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        eos_token_id=256,
        pad_token_id=257,
    )
    return ByteTok(), LlamaForCausalLM(cfg).to(dev), False


def load_data(real):
    if real:
        try:
            from datasets import load_dataset

            ds = load_dataset("openai/gsm8k", "main")
            gold = lambda a: a.split("####")[-1].strip().replace(",", "")
            return [(x["question"], gold(x["answer"])) for x in ds["train"]], [
                (x["question"], gold(x["answer"])) for x in ds["test"]
            ]
        except Exception as e:
            print("GSM8K not reachable, using synthetic problems:", type(e).__name__)
    mk = lambda: (
        lambda a, b: (
            f"Sam has {a} marbles and finds {b} more. How many marbles does Sam have now?",
            str(a + b),
        )
    )(random.randint(2, 60), random.randint(2, 60))
    return [mk() for _ in range(2000)], [mk() for _ in range(300)]


tok, model, real = load_model()
train, test = load_data(real)
ref = copy.deepcopy(model).eval().requires_grad_(False)  # the KL anchor

if real and dev == "cuda":
    model.gradient_checkpointing_enable()  # trade compute for memory

print(
    f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters, {len(train)} training problems"
)

SYSTEM = (
    "Solve the problem. Think step by step, then end with a line of the form 'Answer: <number>'."
)


def prompt(q):
    if real:
        return tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"{q}\nAnswer: "


def extract(text):
    m = re.findall(r"Answer:\s*\$?(-?[\d,]*\.?\d+)", text) or re.findall(r"-?\d[\d,]*\.?\d*", text)
    return m[-1].replace(",", "").rstrip(".") if m else None


def reward(text, gold):
    a = extract(text)
    try:
        return float(a is not None and abs(float(a) - float(gold)) < 1e-6)
    except ValueError:
        return 0.0


# %% [markdown]
# ## Rollouts, log-probabilities and evaluation


# %%
@torch.no_grad()
def rollout(questions, n, greedy=False):
    enc = tok(
        [prompt(q) for q in questions for _ in range(n)], return_tensors="pt", padding=True
    ).to(dev)
    out = model.generate(
        **enc,
        do_sample=not greedy,
        temperature=1.0,
        top_p=1.0,
        max_new_tokens=MAX_NEW,
        pad_token_id=tok.pad_token_id,
    )
    comp = out[:, enc["input_ids"].shape[1] :]
    is_end = (comp == tok.eos_token_id) | (comp == tok.pad_token_id)
    first = torch.where(
        is_end.any(1),
        is_end.float().argmax(1),
        torch.full((len(comp),), comp.shape[1], device=comp.device),
    )
    cmask = (torch.arange(comp.shape[1], device=comp.device)[None] <= first[:, None]).float()
    texts = tok.batch_decode(
        comp.masked_fill(cmask == 0, tok.pad_token_id), skip_special_tokens=True
    )
    return enc, comp, cmask, texts


def token_logps(m, ids, attn, n_comp):
    logits = m(input_ids=ids, attention_mask=attn).logits[:, -n_comp - 1 : -1].float()
    return torch.gather(logits.log_softmax(-1), 2, ids[:, -n_comp:, None]).squeeze(-1)


def evaluate(problems):
    model.eval()
    hits, lens = 0, []
    for i in range(0, len(problems), 16):
        qs = problems[i : i + 16]
        _, _, cm, texts = rollout([q for q, _ in qs], 1, greedy=True)
        hits += sum(reward(t, g) for t, (_, g) in zip(texts, qs))
        lens += cm.sum(1).tolist()
    return hits / len(problems), float(np.mean(lens))


eval_set = test[:N_EVAL]
acc0, len0 = evaluate(eval_set)
print(f"before RL: greedy accuracy {acc0:.1%}, mean length {len0:.0f} tokens")

# %% [markdown]
# ## The GRPO loop
# For each question, G samples. Advantage = (reward − group mean) / group std: the group is the baseline, no
# critic. Loss per token = −advantage · log π(token) + β · KL(π‖π_ref), averaged over all completion tokens
# in the batch (the token-level normalisation from DAPO, which stops long answers from being under-weighted).
# One gradient step per batch keeps us exactly on-policy, so no clipping ratio is needed.

# %%
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
log = []

for step in range(STEPS):
    t0 = time.time()
    batch = random.sample(train, B)
    model.eval()
    enc, comp, cmask, texts = rollout([q for q, _ in batch], G)
    r = torch.tensor(
        [reward(t, g) for t, (_, g) in zip(texts, [p for p in batch for _ in range(G)])], device=dev
    ).view(B, G)
    adv = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)).flatten()
    ids = torch.cat([enc["input_ids"], comp], 1)
    attn = torch.cat([enc["attention_mask"].float(), cmask], 1).long()
    model.train()
    opt.zero_grad(set_to_none=True)
    n_tok = cmask.sum()
    for i in range(0, len(ids), CHUNK):  # micro-batches: logits are huge
        sl = slice(i, i + CHUNK)
        lp = token_logps(model, ids[sl], attn[sl], comp.shape[1])
        with torch.no_grad():
            lr_ = token_logps(ref, ids[sl], attn[sl], comp.shape[1])
        kl = torch.exp(lr_ - lp) - (lr_ - lp) - 1  # k3 estimator, always ≥ 0
        loss = ((-adv[sl, None] * lp + BETA * kl) * cmask[sl]).sum() / n_tok
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    log.append(
        dict(
            step=step,
            reward=r.mean().item(),
            no_signal=(r.std(1) == 0).float().mean().item(),
            length=cmask.sum(1).mean().item(),
            secs=time.time() - t0,
        )
    )
    if step % (1 if QUICK else 10) == 0:
        print({k: round(v, 3) for k, v in log[-1].items()})

acc1, len1 = evaluate(eval_set)
print(
    f"after RL:  greedy accuracy {acc1:.1%} (was {acc0:.1%}), mean length {len1:.0f} tokens (was {len0:.0f})"
)

# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))
k = lambda key: (
    np.convolve([d[key] for d in log], np.ones(5) / 5, mode="valid")
    if len(log) >= 5
    else [d[key] for d in log]
)
ax[0].plot(k("reward"))
ax[0].set_title("training reward (5-step average)")
ax[1].plot(k("length"))
ax[1].set_title("completion length (tokens)")
ax[2].plot(k("no_signal"))
ax[2].set_title("groups with zero advantage")
[a.set_xlabel("step") for a in ax]
plt.tight_layout()
plt.show()

# %% [markdown]
# ## My predictions (not run here: no GPU in the authoring environment)
# For Qwen2.5-0.5B-Instruct with these settings on an A100:
#
# | Quantity | Prediction |
# |---|---|
# | Greedy accuracy before RL (this prompt, strict extraction) | 30–42% |
# | After 150 steps | 40–50%; a large part of the early gain is format (always ending with `Answer:`) |
# | Training reward | rises fastest in the first ~40 steps, then slowly |
# | Completion length | dips first (dropping rambling), then creeps up |
# | Groups with no signal | 40–60%, mostly all-wrong groups early, all-right groups late |
# | Wall-clock | ~50–70 s per step, ~2–3 h total |
#
# If your numbers land outside these ranges, that's interesting: check whether extraction is failing
# (print 10 completions) before concluding anything about RL.
#
# **Stretch:** (1) filter training problems to those with 1–7 of 8 correct (dynamic sampling, as in DAPO) and
# measure the speed-up; (2) compute pass@64 before and after on 50 problems to test the “RL sharpens rather
# than expands” bet from the predictions chapter; (3) set `BETA = 0` and watch whether length runs away.
