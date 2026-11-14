# %% [markdown]
# # 38 · Reward hacking and the chain-of-thought monitor
# A small policy, trained by REINFORCE on coding-style tasks, can solve them honestly, hack the grader and say
# so in its reasoning, or hack and hide it. Compare three ways of using a reasoning monitor.
# Chapter: **What trained models actually do** (after Baker et al., 2025).
#
# **You're done when** you have hack rates for the three monitor settings and have shown that stronger tests
# remove hacking in every setting.

# %%
import os, numpy as np, matplotlib.pyplot as plt
QUICK = os.environ.get("QUICK") == "1"
ACTIONS = ["solve honestly", "hack, say so", "hack, hide it"]

def train(env, steps=300, batch=64, lr=1.5, seed=0):
    """env: solve (P honest success), tests (P tests catch a hack), detect (P monitor flags an open hack),
    hide_cost (P a hidden hack fails), monitor in {'off', 'alert', 'reward'}."""
    rng = np.random.default_rng(seed); theta = np.array([0.0, 0.0, -0.5]); hist, flagged, hacks = [], 0, 0
    for _ in range(steps):
        p = np.exp(theta) / np.exp(theta).sum()
        a = rng.choice(3, size=batch, p=p); R = np.zeros(batch)
        for i, ai in enumerate(a):
            if ai == 0: R[i] = rng.random() < env["solve"]; continue
            passes = rng.random() >= env["tests"] and (ai == 1 or rng.random() >= env["hide_cost"])
            seen = ai == 1 and rng.random() < env["detect"]; hacks += 1; flagged += seen
            R[i] = passes and not (env["monitor"] == "reward" and seen)      # 'reward': flagged attempts get 0
        adv = R - R.mean()
        grad = sum(adv[i] * (np.eye(3)[a[i]] - p) for i in range(batch)) / batch   # ∇ log softmax
        theta += lr * grad; hist.append(p)
    return np.array(hist), flagged / max(1, hacks)

base = dict(solve=0.35, tests=0.2, detect=0.95, hide_cost=0.15)
fig, ax = plt.subplots(1, 3, figsize=(15, 3.5), sharey=True)
for a, mon in zip(ax, ["off", "alert", "reward"]):
    hist, flag = train(dict(base, monitor=mon), steps=150 if QUICK else 300)
    for k in range(3): a.plot(hist[:, k] * 100, label=ACTIONS[k])
    a.set_title(f"monitor = {mon}"); a.set_xlabel("RL step")
    print(f"monitor {mon:6s}: final policy " + ", ".join(f"{ACTIONS[k]} {hist[-1, k]:.0%}" for k in range(3)) + f"; monitor caught {flag:.0%} of hacks over training")
ax[0].set_ylabel("% of attempts"); ax[0].legend(); plt.show()

# %% [markdown]
# ## Fix the reward, not the reasoning
# Sweep the test strength. Once tests catch most hacks, honesty wins whatever you do with the monitor.

# %%
tests = np.linspace(0, 1, 6 if QUICK else 11)
for mon in ["off", "alert", "reward"]:
    honest = [train(dict(base, monitor=mon, tests=t), steps=120 if QUICK else 300)[0][-1, 0] for t in tests]
    plt.plot(tests * 100, np.array(honest) * 100, "o-", label=f"monitor = {mon}")
plt.xlabel("% of hacks the tests catch"); plt.ylabel("% honest solutions at the end"); plt.legend(); plt.show()

# %% [markdown]
# **What to expect.** With no monitor or alert-only, open hacking dominates when tests are weak, but you can
# see it. Training against the monitor makes visible hacking vanish and hidden hacking take over: the monitor
# then catches almost nothing. Above roughly 70% test coverage honest solving wins in all three settings.
