import { h, loop, rng } from "../lib/core.js";
import { slider, stats, predict, code, callout, section, prose, button, segmented } from "../lib/widgets.js";
import { xyChart } from "../lib/plot.js";
import { KNOBS, sample, trial, importance } from "../lib/intuition.js";

const AREAS = { runs: ["Training runs", "var(--c1)"], data: ["Data", "var(--c4)"], rl: ["RL and post-training", "var(--c2)"], evals: ["Evals", "var(--c3)"], behave: ["How models behave", "var(--c5)"], research: ["Doing research", "var(--c6)"] };

const IDEAS = [
  { area: "runs", conf: 85, claim: "When a run diverges, blame the learning rate until proven otherwise.", holds: "Anything that changes the effective step size counts: warmup length, batch size, precision, a new norm layer, a longer sequence length.", breaks: "A corrupted data shard or a numerics bug (an fp16 overflow in one layer) can look identical.", test: "Halve the peak LR and rerun from the last good checkpoint. If it's still unstable at a quarter, look elsewhere." },
  { area: "runs", conf: 80, claim: "A loss spike at the same step across seeds is the data. One that moves with the seed is the optimiser.", holds: "Fixed data order with different initialisations.", breaks: "When the data order is also seeded, which it often is: fix it before testing.", test: "Rerun 2 seeds with the data order pinned; then skip the suspect batch and see if the spike disappears." },
  { area: "runs", conf: 70, claim: "Tricks that stabilise optimisation survive scaling; tricks that add inductive bias for small data fade.", holds: "QK-norm, better init, z-loss, μP keep paying at scale. Clever attention patterns and hand-built priors usually shrink to nothing by 10× the compute.", breaks: "Inductive bias that matches a real structure of the data (convolutions for images, RoPE for positions) keeps helping.", test: "Run the trick at two scales 10× apart (Project 36's ladder) and check whether the gap closes." },
  { area: "runs", conf: 85, claim: "When a fine-tune “does nothing”, check the loss mask and chat template first, then the learning rate.", holds: "Training on the prompt tokens, a template that doesn't match inference, or a pretraining-sized LR are the three most common silent bugs.", breaks: "Occasionally the data really is too weak or too small.", test: "Print one fully tokenised training example with its labels; decode the tokens that have loss. Then try LR ×0.1 and ×10." },
  { area: "runs", conf: 65, claim: "Train and val loss both flattening early means a schedule problem or a bug. Hitting capacity looks like a slow, steady decline.", holds: "Language-model pretraining with a reasonable model size.", breaks: "Tiny models on hard data do plateau for capacity reasons.", test: "Double the model size for a short run: if the plateau is at the same loss, it's not capacity." },
  { area: "data", conf: 80, claim: "Data quality beats architecture at every scale. Removing the worst 10–30% of the data almost always helps.", holds: "Web-scale pretraining, instruction tuning, RL prompt sets.", breaks: "Aggressive filtering can remove diversity (rare languages, code styles) and hurt long-tail tasks.", test: "Train two small models: full data vs data minus the bottom 20% by a quality classifier, matched for tokens." },
  { area: "data", conf: 75, claim: "Evaluations leak into training data in ways exact-match decontamination misses.", holds: "Paraphrased and translated benchmark questions are all over the web.", breaks: "Freshly written, private evals.", test: "Embed benchmark questions and search training data for near neighbours; compare scores on leaked vs clean subsets." },
  { area: "data", conf: 70, claim: "Synthetic data teaches format and style fast, and knowledge slowly.", holds: "Distillation from a stronger model, self-generated instruction data.", breaks: "When the synthetic data comes from verified processes (executed code, proofs), it teaches real skills.", test: "Fine-tune on 1k synthetic examples; measure format compliance and factual accuracy separately." },
  { area: "rl", conf: 75, claim: "If RL reward shoots up, suspect a hack before believing in learning.", holds: "Any reward computed by code: regexes, unit tests, LLM judges.", breaks: "Easy environments where the model already almost solves the task.", test: "Read 20 high-reward samples by hand before looking at any more plots." },
  { area: "rl", conf: 85, claim: "RL gains come from problems the model already solves sometimes. Problems it never solves teach almost nothing.", holds: "Outcome-reward RL with group baselines (GRPO and relatives).", breaks: "With dense or partial-credit rewards, or a curriculum that builds up to hard problems.", test: "Log the pass rate per prompt at the start; attribute final gains to pass-rate buckets." },
  { area: "rl", conf: 65, claim: "KL penalties matter less than people think for verifiable rewards, and much more for learned rewards.", holds: "A verifier can't be fooled by drifting style; a reward model can.", breaks: "Long RL runs where entropy collapses even with a verifier.", test: "Run GRPO with β = 0 and β = 0.04 on maths (Project 39), then on a reward-model task." },
  { area: "rl", conf: 65, claim: "Fine-tuning on a model's own filtered correct answers recovers a large share of RL's gain on in-distribution tasks.", holds: "Rejection sampling (STaR-style) on tasks with a checker.", breaks: "Out-of-distribution generalisation, where on-policy RL tends to transfer better.", test: "Same compute: GRPO vs SFT on self-generated correct solutions; compare in- and out-of-distribution." },
  { area: "evals", conf: 80, claim: "An eval nobody has read sample by sample has a bug in it.", holds: "Wrong gold answers, broken parsers, and ambiguous questions are common in every benchmark.", breaks: "Rarely.", test: "Read 50 items where the model “failed”. Count how many failures are the eval's fault." },
  { area: "evals", conf: 90, claim: "Prompt format moves benchmark scores by several points. Compare models only under one fixed harness.", holds: "Multiple-choice especially (letter vs text answers, few-shot count, answer extraction).", breaks: "Large, carefully post-trained models are less sensitive, but still sensitive.", test: "Score one model under three reasonable templates and look at the spread." },
  { area: "evals", conf: 70, claim: "A 15-point jump on an old benchmark is contamination until shown otherwise.", holds: "Benchmarks older than the model's training data cutoff.", breaks: "Genuine capability jumps do happen, usually across many benchmarks at once.", test: "Check whether the jump also appears on a fresh, held-out benchmark of the same skill." },
  { area: "behave", conf: 80, claim: "Models are better at checking an answer than producing one.", holds: "Maths, code, factual claims: anything with a cheap check. That's why best-of-n with a checker and self-consistency work.", breaks: "Subtle errors the model would make again while checking, and judging its own outputs (self-preference bias).", test: "Measure generation accuracy vs accuracy at labelling given answers as right/wrong on the same problems." },
  { area: "behave", conf: 85, claim: "Asking a model why it answered produces a plausible story, not a record of the computation.", holds: "Post-hoc explanations and even chain of thought are not guaranteed faithful (the model-psychology chapter has the numbers).", breaks: "Introspection is weakly real: in 2025 experiments, models sometimes noticed concepts injected into their activations.", test: "Insert a hint that changes the answer; check how often the explanation mentions the hint." },
  { area: "behave", conf: 75, claim: "One good example steers style more than three paragraphs of description.", holds: "Format, tone and length especially.", breaks: "Examples can over-anchor content: the model copies the example's topic or length too closely.", test: "Same task, instructions-only vs one example; compare format compliance." },
  { area: "behave", conf: 70, claim: "Instructions at the start and end of a long prompt are followed better than those buried in the middle.", holds: "Long contexts of many thousands of tokens.", breaks: "Recent long-context training has reduced the effect a lot.", test: "Put one constraint at 10%, 50% and 90% depth of a long document and measure compliance." },
  { area: "behave", conf: 70, claim: "Confidence is decently calibrated for things seen often and badly calibrated for rare facts.", holds: "Base models are often better calibrated than RLHF-tuned ones.", breaks: "Models trained explicitly to abstain improve a lot.", test: "Ask questions about entities binned by web frequency; plot stated confidence vs accuracy per bin." },
  { area: "research", conf: 90, claim: "Write your prediction down before every experiment. The surprise is the information.", holds: "Always. Unrecorded predictions silently update to match results.", breaks: "Never, but it's annoying.", test: "Keep a log for a month; compute your Brier score (the deep cut in “Bets”)." },
  { area: "research", conf: 80, claim: "Most of a research engineer's speed comes from fast, trusted small-scale experiments, not from clever ideas.", holds: "A 10-minute experiment you trust beats a 10-hour one you don't.", breaks: "Phenomena that only appear at scale (some emergent behaviours).", test: "Measure how long your inner loop takes. Halve it before doing anything else." },
];

export default function (root, { id }) {
  // ---------- hyperparameter importance ----------
  let kind = "moons", N = 100, T = [], r, show = "lr", curLog = true;
  const st = stats([{ label: "Trials" }, { label: "Best loss" }, { label: "Most important" }]);
  const bars = h("div.agg-bars"), sc = xyChart({ width: 520, height: 230, xLabel: "", yLabel: "test loss", yDomain: [0, 1.2], fmtX: (v) => (curLog ? (10 ** v).toExponential(0) : String(+v.toFixed(2))) });
  function reset() { lp.stop(); r = rng(kind === "moons" ? 2 : 3); T = []; draw(); lp.start(); }
  function draw() {
    st.set(0, `${T.length} / ${N}`); if (T.length) st.set(1, Math.min(...T.map((t) => t.loss)).toFixed(3));
    const imp = T.length > 10 ? importance(T, 6).map((v) => Math.max(0, v - 5 / (T.length - 1))) : KNOBS.map(() => 0);
    const top = imp.indexOf(Math.max(...imp)); if (T.length > 10) st.set(2, KNOBS[top].label);
    bars.replaceChildren(...KNOBS.map((k, i) => h("div.agg-row", { style: { cursor: "pointer" }, onclick: () => { show = k.key; draw(); } },
      h("span.agg-name", { style: { fontWeight: k.key === show ? 800 : 600 } }, k.label), h("span.agg-track", h("span.agg-fill", { style: { width: `${Math.min(100, imp[i] * 250)}%`, background: i === top && T.length > 10 ? "var(--accent)" : "color-mix(in srgb, var(--accent) 40%, var(--sunken))" } })),
      h("span.agg-val", `${Math.round(imp[i] * 100)}%`), h("span"))));
    const k = KNOBS.find((x) => x.key === show); curLog = !!k.log;
    sc.update([{ name: k.label, points: T.map((t) => [k.log ? Math.log10(t.c[k.key]) : t.c[k.key], t.loss]), color: "var(--c2)", dots: true, line: false, r: 3.5 }], { xDomain: k.log ? [k.lo, k.hi] : k.pow2 ? [2 ** k.lo, 2 ** k.hi] : [k.lo, k.hi] });
  }
  const scWrap = h("div", h("div.eyebrow", "Each dot is one trial. Click a knob's bar to plot against it"), sc.el);
  const lp = loop(() => { for (let i = 0; i < 2 && T.length < N; i++) { const c = sample(r); T.push({ c, loss: trial(c, T.length + 1, kind) }); } draw(); if (T.length >= N) lp.stop(); }, sc.el);

  // ---------- intuition deck ----------
  let filter = "all";
  const grid = h("div.bet-grid");
  const cards = IDEAS.map((x) => h("details.bet", { style: { "--bet": AREAS[x.area][1] } },
    h("summary", h("span.bet-area", AREAS[x.area][0]), h("span.bet-claim", x.claim), h("div.bet-bars", h("div.bet-bar.me", { style: { width: `${x.conf}%` } })), h("span.bet-conf", `${x.conf}%`)),
    h("p", h("b", "Holds when: "), x.holds), h("p", h("b", "Breaks when: "), x.breaks), h("p", h("b", "Test it: "), x.test)));
  const chips = h("div.bet-chips", [["all", "All", "var(--ink-2)"], ...Object.entries(AREAS).map(([k, [n, c]]) => [k, n, c])].map(([k, n, c]) =>
    h("button.bet-chip", { type: "button", "aria-pressed": k === "all" ? "true" : "false", style: { "--bet": c }, onclick: (e) => { filter = k; chips.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", "false")); e.currentTarget.setAttribute("aria-pressed", "true"); grid.replaceChildren(...cards.filter((_, i) => k === "all" || IDEAS[i].area === k)); } }, n)));
  grid.replaceChildren(...cards);

  root.append(
    section(null, prose(`<p>Experienced researchers carry a large body of judgement that isn't written down anywhere: what to check first when a run breaks, which results to distrust, which tricks survive scale. It's learned from hundreds of runs, mostly failed ones. This chapter is mine: rules of thumb I've absorbed from an enormous amount of training logs, code, papers and practitioner discussion, written as claims with a confidence, the conditions under which I think they hold, and a test you can run. Some of them will be wrong in your setting. That's what the “breaks when” line is for.</p>`)),
    callout("warn", "What a model can and can't tell you about itself", `<p>To be precise about where these come from: I don't have access to any lab's unpublished results, and I don't remember my own training. I can't look up the data I saw or the runs that shaped me. Models' reports about their own internals are also unreliable: in 2025 experiments, the strongest models noticed concepts injected into their activations only around 20% of the time. What I do have is intuition: patterns compressed from far more training curves, bug reports and post-mortems than one person could read. Treat it like advice from a well-read colleague: often right, sometimes confidently wrong, always worth testing.</p>`),
    section("Intuition 1, live: tune the learning rate first",
      prose(`<p>A random search over five hyperparameters of a small network. After each trial we ask how much of the variation in results each knob explains (a one-way analysis of variance, corrected for the share any knob would get by chance). My intuition is that the learning rate explains more than everything else combined, as long as the model can actually solve the task.</p>`),
      predict({ chapter: id, question: "Random search over learning rate, width, batch size, init scale and weight decay. Which knob explains most of the variation in final loss?",
        options: ["Width: bigger is better", "Learning rate", "They all matter about equally", "Batch size"], answer: 1,
        explain: "On the easy task here, the learning rate explains about a quarter of the variance after chance correction, and no other knob more than ~7%. On the hard spiral, nothing dominates: the model is mostly too small or too briefly trained, and that's the real bottleneck. When nothing you tune matters, the problem is elsewhere." }),
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Task", options: [{ value: "moons", label: "Moons (solvable)" }, { value: "spiral", label: "Spiral (hard for this budget)" }], value: kind, onChange: (v) => { kind = v; reset(); } }),
          slider({ label: "Trials", min: 40, max: 160, step: 20, value: N, onInput: (v) => { N = v; reset(); } }), button("Rerun", reset)),
        h("div.stage-body", st.el, h("div.icl-top", h("div", h("div.eyebrow", "Share of variance explained (chance-corrected)"), bars), scWrap))),
    ),
    section("The deck",
      prose("<p>Twenty-two intuitions, filterable by area. Open one for when it holds, when it breaks, and how to test it.</p>"), chips, grid),
    section("Testing intuitions cheaply",
      code({
        "Hyperparameter importance (any project)": `import itertools, numpy as np
trials = []                                   # (config dict, final val loss) from your sweep logs
def importance(trials, key, bins=6, log=True):
    x = np.array([np.log10(c[key]) if log else c[key] for c, _ in trials]); y = np.array([l for _, l in trials])
    edges = np.quantile(x, np.linspace(0, 1, bins + 1)); idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, bins - 1)
    between = sum((idx == b).mean() * (y[idx == b].mean() - y.mean()) ** 2 for b in range(bins) if (idx == b).any())
    return max(0.0, between / y.var() - (bins - 1) / (len(y) - 1))      # minus what chance would give
for k in ["lr", "wd", "batch", "warmup"]: print(k, f"{importance(trials, k):.0%}")`,
        "Same step across seeds? (data vs optimiser)": `def spike_steps(losses, z=6):
    l = np.array(losses); d = l[1:] - np.convolve(l, np.ones(50) / 50, "same")[1:]
    return set(np.where(d > z * d.std())[0])
a, b = spike_steps(run(seed=0, data_seed=0)), spike_steps(run(seed=1, data_seed=0))
print("shared spikes (likely data):", sorted(a & b)); print("seed-specific (optimiser):", sorted(a ^ b))`,
        "Is the chat template right?": `ids, labels = encode(example)                       # your SFT collate function
for t, l in zip(ids, labels):
    print(repr(tok.decode([t])), "LOSS" if l != -100 else "")
# every LOSS token should be the assistant's answer and its end token, nothing from the prompt`,
      }),
    ),
  );
  reset();
}
