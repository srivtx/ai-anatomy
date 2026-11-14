import { h, loop } from "../lib/core.js";
import { stats, predict, code, callout, section, prose, table, button, stepper, segmented } from "../lib/widgets.js";
import { xyChart } from "../lib/plot.js";
import { dataset } from "../lib/nn.js";
import { rng } from "../lib/core.js";
import { run, mean, sd } from "../lib/research.js";

const LRS = [0.001, 0.003, 0.01, 0.03, 0.1];
const SWEEP_DATA = { train: dataset("circle", 200, { noise: 0.15, seed: 21 }), test: dataset("circle", 600, { noise: 0.15, seed: 121 }) };
// small, noisy data so regularisers matter: 80 spiral points, 10% of labels flipped
const ABL_DATA = (() => {
  const train = dataset("spiral", 80, { noise: 0.05, seed: 21 }), r = rng(5);
  train.forEach((p) => { if (r() < 0.1) p.y = 1 - p.y; });
  return { train, test: dataset("spiral", 600, { noise: 0.05, seed: 121 }) };
})();
const COMP = [
  { key: "width", name: "Wider layers (12 → 32)" },
  { key: "wd", name: "Weight decay" },
  { key: "aug", name: "Input jitter (augmentation)" },
];
const cfgOf = (mask) => ({ width: mask & 1 ? 32 : 12, wd: mask & 2 ? 1e-3 : 0, aug: mask & 4 ? 0.06 : 0, lr: 0.02, steps: 800, act: "relu" });
const pm = (m, s) => `${(m * 100).toFixed(1)} ± ${(s * 100).toFixed(1)}`;
const sgn = (d) => `${d >= 0 ? "+" : "−"}${Math.abs(d * 100).toFixed(1)}`;

/** Run a list of jobs a few at a time per frame so the page stays responsive. */
function runner(el, onProgress, onDone) {
  let queue = [], done = 0, total = 0;
  const lp = loop(() => {
    const t0 = performance.now();
    while (queue.length && performance.now() - t0 < 14) { queue.shift()(); done++; }
    onProgress(done, total);
    if (!queue.length) { lp.stop(); onDone(); }
  }, el);
  return { start(jobs) { queue = jobs; done = 0; total = jobs.length; lp.start(); }, get busy() { return lp.running; } };
}

export default function (root, { id }) {
  // ---------- 1. tune the baseline too ----------
  let idea = "sin", baseLR = 0;
  const res = { base: LRS.map(() => []), idea: LRS.map(() => []) };
  const sc = xyChart({ height: 280, xLog: true, xLabel: "learning rate", yLabel: "test accuracy (%)", xDomain: [0.0008, 0.13], yDomain: [50, 100] });
  const sst = stats([{ label: "The paper's comparison" }, { label: "Both tuned the same way" }, { label: "Verdict" }]);
  const sprog = h("p.canvas-cap", "");
  const IDEAS = { sin: { name: "New idea: sin activations", cfg: { act: "sin" }, seed: 0 }, placebo: { name: "New idea: placebo (identical network, other seeds)", cfg: { act: "tanh" }, seed: 10 } };
  function drawSweep() {
    const best = (rows) => rows.reduce((b, r, i) => (r.length && mean(r) > (b.v ?? -1) ? { v: mean(r), i } : b), {});
    const ser = (rows, name, color) => [
      { name, color, points: rows.map((r, i) => [LRS[i], r.length ? mean(r) * 100 : NaN]).filter((p) => isFinite(p[1])), dots: true, r: 5 },
      { color, points: rows.flatMap((r, i) => r.map((a) => [LRS[i] * (1 + (Math.random() - 0.5) * 0.12), a * 100])), dots: true, r: 2.5, line: false },
    ];
    sc.update([...ser(res.base, "Baseline (tanh): mean of 3 seeds", "var(--c2)"), ...ser(res.idea, IDEAS[idea].name, "var(--accent)")], { marks: [{ x: LRS[baseLR], label: "baseline's LR in the paper" }] });
    const bi = best(res.idea), bb = best(res.base), b0 = res.base[baseLR];
    if (bi.v == null || !b0.length || bb.v == null) { for (let i = 0; i < 3; i++) sst.set(i, "–", ""); return; }
    const claim = bi.v - mean(b0), fair = bi.v - bb.v, noise = Math.max(sd(res.idea[bi.i]), sd(res.base[bb.i]));
    sst.set(0, `${sgn(claim)} points`, claim > 0.03 ? "good" : "");
    sst.set(1, `${sgn(fair)} points`, Math.abs(fair) < Math.max(0.01, 2 * noise) ? "warn" : fair > 0 ? "good" : "bad");
    sst.set(2, Math.abs(fair) < Math.max(0.01, 2 * noise) ? "No real gain: within seed noise" : fair > 0 ? "Real gain survives" : "Baseline is better", Math.abs(fair) < Math.max(0.01, 2 * noise) ? "warn" : fair > 0 ? "good" : "bad");
  }
  const sweep = runner(sc.el, (d, t) => { sprog.textContent = d < t ? `Training ${d + 1} of ${t} networks…` : `${t} networks trained in your browser: 5 learning rates × 3 seeds × 2 methods, 300 steps each.`; drawSweep(); }, drawSweep);
  function startSweep() {
    res.base = LRS.map(() => []); res.idea = LRS.map(() => []);
    const jobs = [];
    LRS.forEach((lr, i) => [1, 2, 3].forEach((seed) => {
      jobs.push(() => res.base[i].push(run({ act: "tanh", lr, seed, steps: 300 }, SWEEP_DATA).acc));
      jobs.push(() => res.idea[i].push(run({ ...IDEAS[idea].cfg, lr, seed: seed + IDEAS[idea].seed, steps: 300 }, SWEEP_DATA).acc));
    }));
    sweep.start(jobs);
  }

  // ---------- 2. ablations ----------
  const abl = Array.from({ length: 8 }, () => []);
  let nSeeds = 5;
  const ablEl = h("div"), cmpEl = h("div");
  const aprog = h("p.canvas-cap", "");
  function drawAbl() {
    const have = abl.every((a) => a.length > 0);
    const rows = Array.from({ length: 8 }, (_, m) => m).sort((a, b) => a.toString(2).replace(/0/g, "").length - b.toString(2).replace(/0/g, "").length || a - b).map((m) => {
      const a = abl[m];
      return [COMP.filter((_, i) => m & (1 << i)).map((c) => c.name.split(" (")[0]).join(" + ") || "Bare baseline",
        a.length ? h("span.mono", pm(mean(a), sd(a))) : "…",
        a.length ? h("div.bar-t", { style: { minWidth: "140px" } }, h("span.bar-f", { style: { width: `${Math.max(0, (mean(a) - 0.5) / 0.5) * 100}%`, background: m === 7 ? "var(--accent)" : m === 0 ? "var(--ink-3)" : "var(--c4)" } })) : ""];
    });
    ablEl.replaceChildren(table(["Components on", `Test accuracy (%), mean ± sd over ${nSeeds} seeds`, ""], rows));
    if (!have) return cmpEl.replaceChildren();
    const M = abl.map(mean);
    cmpEl.replaceChildren(table(["Component", "Add it to the bare baseline", "Remove it from the full recipe"], COMP.map((c, i) => {
      const add = M[1 << i] - M[0], rem = M[7] - M[7 ^ (1 << i)];
      const tone = (d) => h("span.mono", { style: { color: d > 0.02 ? "var(--good)" : d < -0.02 ? "var(--bad)" : "var(--ink-2)", fontWeight: 700 } }, `${sgn(d)} points`);
      return [c.name, tone(add), tone(rem)];
    })), h("p.canvas-cap", "“Remove it” shows how much the full recipe loses without that component: positive means the component was helping. When the two columns disagree, the components interact, and a single ablation table tells only half the story."));
  }
  const ablRun = runner(ablEl, (d, t) => { aprog.textContent = d < t ? `Training ${d + 1} of ${t} networks…` : `${t} networks trained: every combination of 3 components × ${nSeeds} seeds, 800 steps each, on 80 noisy points.`; drawAbl(); }, drawAbl);
  function startAbl(seeds) {
    nSeeds = seeds; abl.forEach((a) => (a.length = 0));
    const jobs = [];
    for (let s = 1; s <= seeds; s++) for (let m = 0; m < 8; m++) jobs.push(() => abl[m].push(run({ ...cfgOf(m), seed: s }, ABL_DATA).acc));
    ablRun.start(jobs);
  }

  root.append(
    section(null, prose(`<p>Most ideas in machine learning don't work, and many that seem to work are artefacts of how they were tested. Research is mostly the discipline of not fooling yourself: compare against a baseline tuned as carefully as your method, run several seeds, change one thing at a time, and say in advance what result would change your mind. Below, every number comes from real networks trained in your browser.</p>`)),
    section("Tune the baseline too",
      prose("<p>A common way to “beat” a baseline: sweep the learning rate for your new method and pick its best value, but run the baseline at the setting from an older paper. Here both methods get the same sweep. Move the marker to choose which learning rate the baseline got in the paper.</p>"),
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "The new idea", options: [{ value: "sin", label: "sin activations" }, { value: "placebo", label: "Placebo" }], value: idea, onChange: (v) => { idea = v; startSweep(); } }),
          segmented({ label: "Baseline's LR in the paper", options: LRS.map((lr, i) => ({ value: i, label: String(lr) })), value: baseLR, onChange: (v) => { baseLR = v; drawSweep(); } }),
          button("Re-run", startSweep, "primary")),
        h("div.stage-body", sst.el, sc.el, sprog)),
      callout("tip", "What you should see", "<p>At the paper's learning rate (0.001, too low for 300 steps) the baseline scores around 70%, and the new idea at its best rate scores about 92%: a headline gain of 20+ points. Tune the baseline the same way and it reaches about 92% too. The placebo, which is literally the same network, “wins” by the same margin. Melis et al. (2018) found the same at scale: well-tuned LSTMs beat many newer language-model architectures that had been compared against under-tuned ones.</p>"),
    ),
    section("Ablations: which part mattered?",
      prose("<p>A recipe with three components beats the baseline. Which component did the work? Train every combination and compare two ways of asking: add each component to the bare baseline, or remove each from the full recipe.</p>"),
      h("div.stage",
        h("div.stage-controls", button("Run with 5 seeds", () => startAbl(5), "primary"), button("Run with 2 seeds", () => startAbl(2))),
        h("div.stage-body", ablEl, aprog, cmpEl)),
      callout("note", "What this run typically shows", "<p>Each component helps on its own. Together, weight decay and jitter over-regularise, so the full recipe can be <i>worse</i> than two of its parts, and removing either regulariser from it looks like a big gain. Width matters a lot when added alone and barely at all when removed from the full recipe. Neither column is “the” importance of a component. Also notice the spread: with 2 seeds the standard deviations are unreliable and rankings flip between runs.</p>"),
      predict({
        chapter: id,
        question: "Your new method beats the baseline by 0.8 points on one seed. What should you do first?",
        options: ["Write the paper", "Run more seeds for both, with the same tuning budget for each, and check whether 0.8 is bigger than the seed-to-seed spread", "Try it on a bigger model", "Remove the baseline from the comparison"],
        answer: 1,
        explain: "<p>Seed-to-seed variation on small benchmarks is often a point or more. Without several seeds and a fairly tuned baseline, a 0.8-point gain is indistinguishable from noise. Bouthillier et al. (2021) measured that data order, initialisation and augmentation randomness alone move results by amounts comparable to many published improvements.</p>",
      }),
    ),
    section("The research loop",
      stepper([
        { title: "1. Ask a sharp question", html: "<p>“Does X reduce validation loss at fixed compute on this setup?” beats “Is X good?”. Decide the metric and the budget before you start.</p>" },
        { title: "2. Write down your prediction", html: "<p>Write the number you expect, and what you'll conclude if you don't get it. It stops you from rationalising whatever comes out.</p>" },
        { title: "3. Smallest experiment that could change your mind", html: "<p>Minutes, not days. A small model and a small dataset where the effect should still show. Scale up only what survives.</p>" },
        { title: "4. Reproduce the baseline first", html: "<p>Before changing anything, match a known number. If you can't reproduce the baseline, your comparison means nothing.</p>" },
        { title: "5. Change one thing; equal tuning", html: "<p>Give the baseline the same hyperparameter sweep as your method. Fix seeds, log every config, keep the code that made each plot.</p>" },
        { title: "6. Seeds, intervals, and the data", html: "<p>3 seeds minimum, more for small effects; report mean ± spread. Then read actual examples where the methods differ: plots hide bugs that examples reveal.</p>" },
        { title: "7. Check it scales, then write it up", html: "<p>Many tricks help small models and vanish at scale (or reverse). Test 2–3 sizes and see whether the gain grows, holds or shrinks. Report the negative results too.</p>" },
      ]),
      table(["Habit", "What it prevents"], [
        ["Same tuning budget for every method", "Wins that come from tuning, not the idea"],
        ["≥3 seeds, report spread", "Mistaking noise for a gain"],
        ["Held-out test set touched once", "Overfitting to the validation set through many experiments"],
        ["Compare at equal compute (FLOPs or wall-clock)", "“Better” methods that are just bigger or slower"],
        ["Ablate both ways (add-in and leave-out)", "Crediting the wrong component"],
        ["Plot the loss curves, not just final numbers", "Missing instabilities, divergence, unfinished training"],
        ["Check scaling at 2–3 sizes", "Tricks that only help tiny models"],
        ["Log git commit + config + seed per run", "Results nobody (including you) can reproduce"],
      ]),
      callout("note", "The bitter lesson, as a research heuristic", "<p>Rich Sutton's 2019 essay observed that over 70 years, general methods that scale with compute (search and learning) beat methods built on human knowledge of the domain. In practice: prefer changes that make better use of more data and compute, and be suspicious of clever tricks whose benefit shrinks as models grow.</p>"),
    ),
    section("In code",
      code({
        "Fair sweep": `import itertools, numpy as np
def sweep(train_fn, lrs=(1e-3, 3e-3, 1e-2, 3e-2, 1e-1), seeds=(0, 1, 2)):
    """Same grid, same seeds, for every method."""
    res = {lr: [train_fn(lr=lr, seed=s) for s in seeds] for lr in lrs}
    best = max(res, key=lambda lr: np.mean(res[lr]))
    return best, np.mean(res[best]), np.std(res[best], ddof=1), res

lr_b, base, base_sd, _ = sweep(lambda **k: train(method="baseline", **k))
lr_n, new, new_sd, _ = sweep(lambda **k: train(method="new", **k))
print(f"baseline {base:.3f}±{base_sd:.3f} (lr {lr_b})  new {new:.3f}±{new_sd:.3f} (lr {lr_n})")`,
        "Full ablation grid": `components = ["width", "weight_decay", "jitter"]
results = {}
for mask in itertools.product([0, 1], repeat=len(components)):
    cfg = dict(zip(components, mask))
    results[mask] = [train(**cfg, seed=s) for s in range(5)]

M = {k: np.mean(v) for k, v in results.items()}
full, bare = (1,) * len(components), (0,) * len(components)
for i, c in enumerate(components):
    add = M[tuple(int(j == i) for j in range(len(components)))] - M[bare]
    rem = M[full] - M[tuple(0 if j == i else 1 for j in range(len(components)))]
    print(f"{c:14s} add-in {add:+.3f}   leave-out {rem:+.3f}")`,
        "Bootstrap interval on a difference": `def paired_bootstrap(a, b, n=10_000, rng=np.random.default_rng(0)):
    """a, b: per-example scores of two methods on the same test items."""
    d = np.asarray(b) - np.asarray(a)
    means = [rng.choice(d, size=len(d)).mean() for _ in range(n)]
    return d.mean(), np.percentile(means, [2.5, 97.5])`,
      }, "Project 32 builds this harness and runs a full ablation study with seeds and bootstrap intervals."),
    ),
  );
  drawSweep(); drawAbl();
  startSweep(); startAbl(5);
}
