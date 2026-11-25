# %% [markdown]
# # 45 · How long should a model think?
# Part 1 rebuilds the chapter's model of reasoning (serial steps, slips, self-checks, context rot,
# second-guessing) and tests my formula for the best fixed budget. Part 2 runs real budget forcing (s1-style
# “Wait”) on an open reasoning model with GSM8K, so you can see whether real models peak and decline the same way.
# Chapter: **How long should a model think?**, **Chain of thought and System 2**.
#
# **Hardware:** part 1 on a CPU in a minute; part 2 needs a GPU (T4 works for a 0.5–1.5B model).
#
# **You're done when** you have accuracy-vs-budget curves for forced and stop-when-done thinking, the best budget
# for several settings compared with d / (1 − c), and (on a GPU) a real curve from budget forcing.

# %%
import os
import numpy as np
import matplotlib.pyplot as plt

QUICK = os.environ.get("QUICK") == "1"
TRIALS = 600 if QUICK else 3000
BUDGETS = np.unique(np.round(np.logspace(np.log10(8), np.log10(2048), 18)).astype(int))


def think_curve(eps=0.03, check=0.3, rot=400, flip=0.004, stop=False, mean_steps=30, seed=2):
    g = np.random.default_rng(seed)
    probs = np.maximum(2, np.round(g.exponential(mean_steps, TRIALS)).astype(int))
    out = []
    for B in BUDGETS:
        r = np.random.default_rng(seed + 101)
        ok = 0  # common random numbers across budgets
        for d in probs:
            t = done = 0
            wrong = bad = False
            while t < B and done < d:
                t += 1
                if bad and r.random() < check:
                    bad = False
                    continue
                if bad:
                    wrong = True
                bad = r.random() < eps * (1 + t / rot)
                done += 1
            if done < d:
                continue
            correct = not (wrong or bad)
            if not stop and B > t:  # second-guessing as a two-state Markov chain
                n, pi_c, lam = B - t, (flip / 3) / (flip + flip / 3), 1 - flip - flip / 3
                p_ok = pi_c + (1 - pi_c) * lam**n if correct else pi_c * (1 - lam**n)
                correct = r.random() < p_ok
            ok += correct
        out.append(ok / len(probs))
    return np.array(out)


# %% [markdown]
# ## 1. Forced vs stop-when-done

# %%
forced, stop = think_curve(), think_curve(stop=True)
plt.semilogx(BUDGETS, stop, "o-", label="stops when done")
plt.semilogx(BUDGETS, forced, "o-", label="forced to use the budget")
plt.xlabel("budget (steps)")
plt.ylabel("accuracy")
plt.legend()
plt.show()
print(
    "forced peak %.2f at %d steps; forced at %d: %.2f; stop-when-done at %d: %.2f"
    % (forced.max(), BUDGETS[forced.argmax()], BUDGETS[-1], forced[-1], BUDGETS[-1], stop[-1])
)

# %% [markdown]
# ## 2. Testing my formula for the best budget
# Prediction from the chapter (~60% confident): the best fixed budget grows like d / (1 − c), problem depth over
# the self-check miss rate, until context rot dominates. Sweep both and compare the measured peak with the formula.

# %%
rows = []

for d in ([20, 60] if QUICK else [15, 30, 60]):
    for c in ([0.2, 0.7] if QUICK else [0.1, 0.4, 0.7]):
        f = think_curve(check=c, mean_steps=d)
        best = BUDGETS[f.argmax()]
        rows.append((d, c, best, d / (1 - c)))
        print(
            f"depth {d:3d}  check {c:.1f}: best budget {best:5d}   d/(1−c) = {d / (1 - c):6.1f}   ratio {best / (d / (1 - c)):.2f}"
        )

ratios = [b / f for _, _, b, f in rows]
print(
    "ratio best / formula: mean %.2f, spread (max/min) %.1f×"
    % (np.mean(ratios), max(ratios) / min(ratios))
)

# %% [markdown]
# ## 3. Real budget forcing (GPU)
# Cut the model's thinking at a budget, or append “Wait” when it tries to stop early, then force an answer.

# %%
import torch

if torch.cuda.is_available() and not QUICK:
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from datasets import load_dataset

        name = "Qwen/Qwen2.5-1.5B-Instruct"
        tok = AutoTokenizer.from_pretrained(name)
        m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="cuda")
        qs = load_dataset("openai/gsm8k", "main", split="test").select(range(80))

        def run(q, budget):
            msgs = [
                {
                    "role": "user",
                    "content": q
                    + "\nThink step by step, then give the final number after 'Answer:'.",
                }
            ]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            used = 0
            while used < budget:
                ids = tok(text, return_tensors="pt").to("cuda")
                out = m.generate(
                    **ids,
                    max_new_tokens=budget - used,
                    do_sample=False,
                    stop_strings=["Answer:"],
                    tokenizer=tok,
                )
                new = tok.decode(out[0, ids.input_ids.shape[1] :], skip_special_tokens=True)
                used += out.shape[1] - ids.input_ids.shape[1]
                text += new.replace("Answer:", "")
                if used < budget:
                    text += "\nWait, let me double-check. "
            ids = tok(text + "\nAnswer:", return_tensors="pt").to("cuda")
            return tok.decode(
                m.generate(**ids, max_new_tokens=12, do_sample=False)[0, ids.input_ids.shape[1] :],
                skip_special_tokens=True,
            )

        for budget in [64, 128, 256, 512, 1024, 2048]:
            acc = np.mean(
                [
                    q["answer"].split("####")[-1].strip().replace(",", "")
                    in run(q["question"], budget).replace(",", "")
                    for q in qs
                ]
            )
            print(f"budget {budget:5d} tokens: {acc:.0%}")
    except Exception as e:
        print("Real-model part skipped:", type(e).__name__, e)
else:
    print(
        "Part 3 needs a GPU. My prediction: accuracy rises to a peak around 256–512 tokens for a 1.5B instruct model on GSM8K, then falls a few points as forced 'Wait's make it second-guess."
    )

# %% [markdown]
# **What I measured** (CPU, 10 seconds), and my formula failed.
# Forced thinking peaked at 48% around 41 steps and fell to 25% at 2,048; stopping when done reached 58%.
# The sweep refutes my d / (1 − c) rule: the best budget grew with problem depth (41 → 57 → ~100 steps for
# depth 15 → 30 → 60) but barely moved with the self-check rate. The ratio to my formula ranged over 4.5×.
# Better self-checking raises the peak; it doesn't move it much. In this model the best budget is set by how
# long problems are (a quantile of the length distribution) traded against second-guessing, not by the check
# rate. A better guess to test next: best budget ≈ the length that finishes ~70–80% of problems.
