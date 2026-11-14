import { h, loop, rng } from "../lib/core.js";
import { slider, stats, predict, code, callout, section, prose, button, table } from "../lib/widgets.js";
import { xyChart } from "../lib/plot.js";
import { rlWorld, softmax, passAtK, entropy, rlStep, STRATEGIES } from "../lib/hidden.js";

const KS = [1, 2, 4, 8, 16, 32, 64, 128, 256];

export default function (root, { id }) {
  const w = rlWorld();
  let logits = [...w.logits0], step = 0, ent = 0, kl = 0, r = rng(1), hist = [];
  const pi0 = softmax(w.logits0);
  const st = stats([{ label: "RL steps" }, { label: "pass@1 (one try)" }, { label: "pass@64 (best of 64)" }, { label: "Strategy entropy" }]);
  const curves = xyChart({ width: 480, height: 290, xLog: true, xLabel: "k (attempts)", yLabel: "pass@k", xDomain: [1, 256], yDomain: [0.2, 1], fmtY: (v) => `${Math.round(v * 100)}%`, label: "pass@k for the base model and the RL-trained model" });
  const time = xyChart({ width: 480, height: 290, xLabel: "RL step", yLabel: "solved", yDomain: [0.2, 1], fmtY: (v) => `${Math.round(v * 100)}%`, label: "pass@1 and pass@64 over RL training" });
  const barsEl = h("div.rln-bars");

  function render() {
    const pi = softmax(logits);
    const p1 = passAtK(w, pi, 1), p64 = passAtK(w, pi, 64);
    st.set(0, String(step));
    st.set(1, `${(p1 * 100).toFixed(1)}%`, p1 > passAtK(w, pi0, 1) + 0.01 ? "good" : "");
    const b64 = passAtK(w, pi0, 64);
    st.set(2, `${(p64 * 100).toFixed(1)}%`, p64 < b64 - 0.01 ? "bad" : "good");
    st.set(3, `${entropy(pi).toFixed(2)} nats`, entropy(pi) < 0.6 ? "bad" : "");
    curves.update([
      { name: "Base model", color: "var(--c1)", points: KS.map((k) => [k, passAtK(w, pi0, k)]), dots: true, r: 3.5 },
      { name: `After ${step} RL steps`, color: "var(--c2)", points: KS.map((k) => [k, passAtK(w, pi, k)]), dots: true, r: 3.5 },
    ]);
    time.update([
      { name: "pass@1", color: "var(--c2)", points: hist.map((x) => [x[0], x[1]]) },
      { name: "pass@64", color: "var(--c6)", points: hist.map((x) => [x[0], x[2]]) },
    ], { xDomain: [0, Math.max(50, step)] });
    barsEl.replaceChildren(...STRATEGIES.map((name, s) => h("div.rln-row",
      h("span.rln-name", name),
      h("span.rln-track", h("span.rln-base", { style: { width: `${pi0[s] * 100}%` } }), h("span.rln-now", { style: { width: `${pi[s] * 100}%` } })),
      h("span.rln-val", `${(pi0[s] * 100).toFixed(0)}% → ${(pi[s] * 100).toFixed(0)}%`))));
  }
  function record() { const pi = softmax(logits); hist.push([step, passAtK(w, pi, 1), passAtK(w, pi, 64)]); }
  function reset() { logits = [...w.logits0]; step = 0; r = rng(1); hist = []; record(); render(); }

  let running = false, acc = 0;
  const lp = loop((dt) => {
    if (!running) return;
    acc += dt;
    while (acc > 0.03) { acc -= 0.03; logits = rlStep(w, logits, { ent, kl }, r); step++; if (step % 4 === 0) record(); }
    render();
    if (step >= 600) { running = false; playBtn.textContent = "Train with RL"; }
  }, curves.el);
  const playBtn = button("Train with RL", () => { running = !running; playBtn.textContent = running ? "Pause" : "Train with RL"; }, "primary");

  root.append(
    section(null, prose(`<p>Reinforcement learning on verifiable rewards (math answers, passing tests) is how reasoning models got good in 2024–2026. A finding that surprised many people, and that's still argued about: <b>RL mostly makes a model more reliable at things it could already sometimes do, and can make it worse at things it rarely did</b>. Yue et al. (2025) measured it as pass@k: with one attempt the RL model wins; with hundreds of attempts the <i>base</i> model often solves more problems.</p>
<p>Here is a model of why. A “model” chooses one of eight problem-solving strategies; each of ${w.P} problems can only be cracked by some of them. RL pushes probability toward whatever earns the most reward on average. Rare strategies that alone solve a few odd problems get squeezed out.</p>`)),
    section("Train it and watch pass@k cross",
      h("div.stage",
        h("div.stage-controls",
          playBtn,
          button("Reset", reset),
          slider({ label: "Entropy bonus", min: 0, max: 0.1, step: 0.005, value: ent, fmt: (v) => v.toFixed(3), onInput: (v) => (ent = v) }),
          slider({ label: "KL leash to the base model", min: 0, max: 0.5, step: 0.02, value: kl, fmt: (v) => v.toFixed(2), onInput: (v) => (kl = v) }),
        ),
        h("div.stage-body", st.el, h("div.split", curves.el, time.el), h("div", h("div.rln-head", h("b", "How often each strategy gets used"), h("span.rln-key", h("i.rln-kb"), "base ", h("i.rln-kn"), "now")), barsEl)),
      ),
      callout("tip", "Things to try", "<p>Press train with no bonus and no leash. pass@1 climbs from 25% to almost 40%, the strategy bars collapse onto “Standard algebra”, and the orange pass@k curve drops below the blue one for k ≥ 4: the base model, sampled enough times, solves more. Reset, then add a KL leash or an entropy bonus: less gain at k = 1, but the crossing moves right or disappears. Every RL recipe you read about (clip-higher in DAPO, KL in PPO, entropy tricks, resets in ProRL) is a position on this trade-off.</p>"),
      predict({
        chapter: id,
        question: "An RL-trained model beats its base model at pass@1. What happens at pass@256?",
        options: ["RL wins by even more", "They're equal", "The base model can win", "Neither solves anything new"],
        answer: 2,
        explain: "<p>RL sharpens the distribution: it moves mass to answers that already worked. Problems that only a rare path solved lose that path. With enough samples the broader base model covers more problems. Whether long, diverse RL can overcome this and teach genuinely new strategies is open: NVIDIA's ProRL (2025) reported expanded pass@k after long training with KL resets; most short-RL studies see the crossing.</p>",
      }),
    ),
    section("What practitioners know about RL that papers underplay",
      table(["Observation", "Status", "Why it happens"], [
        ["Entropy collapses early, and most of the gain comes in the first few hundred steps", "Published (Cui et al. 2025), widely seen", "The policy gradient feeds on high-probability winners; once the distribution is sharp, groups stop disagreeing and the signal vanishes"],
        ["Problems the model solves 0% or 100% of the time teach nothing", "Published (GRPO, DAPO)", "Group-normalised advantages are zero when every sample gets the same reward; filtering for ~20–80% solve rates is a big efficiency win"],
        ["RL'd models get worse at calibrated probabilities", "Published (GPT-4 report)", "Reward for being right, not for knowing when you might be wrong, pushes confidence to the extremes"],
        ["Responses grow longer during RL whether or not it helps", "Widely seen", "Length correlates with reward on hard problems; length-normalised losses (Dr. GRPO) cut the bias"],
        ["A few hundred carefully chosen problems can do most of what 100k do", "Published (LIMO, s1, one-shot RLVR)", "RL is eliciting and sharpening, not adding knowledge, so you need coverage of skills, not volume"],
        ["Spurious or even random rewards still help some base models", "Published for Qwen2.5-Math (Shao et al. 2025); not for others", "Any sharpening toward the model's own frequent behaviour (code-style reasoning) can raise pass@1 if that behaviour was already good"],
      ]),
      callout("note", "My intuition (unverified, ~65% confident)", "<p>The crossing is a property of <i>narrow</i> RL, not of RL. When RL environments are diverse enough that no single strategy wins on average, and the run is long with periodic resets of the reference model, sharpening per environment coexists with breadth overall, and pass@k at large k improves too. The prediction: the crossing point k* grows roughly with the number of distinct environment families in the RL mix. Notebook 42 sets up the experiment.</p>"),
    ),
    section("In code",
      code({
        "Unbiased pass@k": `import numpy as np

def pass_at_k(n: int, c: int, k: int) -> float:
    """Chen et al. (2021): n samples, c correct. Unbiased, numerically stable."""
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

# per problem: generate n = 256 samples, count correct, average pass_at_k over problems
scores = {k: np.mean([pass_at_k(256, c, k) for c in correct_counts]) for k in [1, 4, 16, 64, 256]}`,
        "Keep entropy alive (clip-higher)": `# DAPO-style asymmetric clipping: let low-probability tokens grow faster than
# high-probability ones shrink, which slows the entropy collapse.
ratio = torch.exp(logp_new - logp_old)
eps_low, eps_high = 0.2, 0.28
loss = -torch.min(ratio * adv, torch.clamp(ratio, 1 - eps_low, 1 + eps_high) * adv)
loss = (loss * mask).sum() / mask.sum()          # token-level mean, not per-sequence

# and drop zero-signal groups before the update
keep = (group_rewards.std(dim=1) > 0)`,
        "Track it": `# log these every RL step: they warn you before the eval does
entropy = -(probs * logprobs).sum(-1)[mask].mean()
solve_rate = rewards.view(G, -1).mean(1)
log({"entropy": entropy, "frac_all_wrong": (solve_rate == 0).float().mean(),
     "frac_all_right": (solve_rate == 1).float().mean(), "len": lengths.float().mean()})`,
      }),
    ),
  );
  reset();
  lp.start();
}
