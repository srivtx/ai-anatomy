import { h, rng } from "../lib/core.js";
import { slider, stats, predict, code, callout, section, prose, table, stepper, button } from "../lib/widgets.js";
import { xyChart } from "../lib/plot.js";
import { gauss } from "../lib/nn.js";

// true loss curves L(C) = E + a · (C / C0)^-alpha, C in FLOPs
const E = 1.7, C0 = 1e18;
const lossA = (C) => E + 2.2 * (C / C0) ** -0.15;               // baseline
function lossB(C, cross) {
  // the new method: better at small scale, smaller exponent, crossing the baseline `cross` decades above C0·1e2
  const Cx = C0 * 10 ** (2 + cross), alphaB = 0.12, aB = (lossA(Cx) - E) * (Cx / C0) ** alphaB;
  return E + aB * (C / C0) ** -alphaB;
}
/** Fit L = e + a·C^-alpha by scanning e and doing a straight-line fit in log space. */
function fit(pts) {
  let best = null;
  const lo = Math.min(...pts.map((p) => p[1]));
  for (let e = 1.0; e < lo - 0.02; e += 0.02) {
    const X = pts.map((p) => Math.log(p[0] / C0)), Y = pts.map((p) => Math.log(p[1] - e));
    const n = X.length, mx = X.reduce((a, b) => a + b) / n, my = Y.reduce((a, b) => a + b) / n;
    const sxx = X.reduce((a, x) => a + (x - mx) ** 2, 0), sxy = X.reduce((a, x, i) => a + (x - mx) * (Y[i] - my), 0);
    const slope = sxy / sxx, icpt = my - slope * mx;
    const sse = pts.reduce((a, p, i) => a + (e + Math.exp(icpt + slope * X[i]) - p[1]) ** 2, 0);
    if (!best || sse < best.sse) best = { sse, f: (C) => e + Math.exp(icpt + slope * Math.log(C / C0)) };
  }
  return best.f;
}

export default function (root, { id }) {
  const o = { cross: 1.5, noise: 0.01, rungs: 5, target: 3 };
  const ch = xyChart({ height: 320, xLog: true, xLabel: "training compute (FLOPs)", yLabel: "validation loss", xDomain: [1e18, 1e24], yDomain: [1.9, 4.0], fmtX: (v) => `1e${Math.round(Math.log10(v))}` });
  const st = stats([{ label: "Truth at the target" }, { label: "This ladder predicts" }, { label: "Ladder right, over 400 simulated ladders" }, { label: "“Compare at the biggest rung”, right" }]);
  let seed = 3;
  function ladder(r) {
    const Cs = Array.from({ length: o.rungs }, (_, i) => C0 * 10 ** ((2 * i) / Math.max(1, o.rungs - 1)));   // 1e18 … 1e20
    const noisy = (f) => Cs.map((C) => [C, f(C) + gauss(r) * o.noise]);
    return { A: noisy(lossA), B: noisy((C) => lossB(C, o.cross)) };
  }
  function draw() {
    const T = C0 * 10 ** (2 + o.target), truthB = lossB(T, o.cross) < lossA(T);
    // Monte Carlo of the decision
    const r = rng(11);
    let okFit = 0, okBig = 0;
    for (let k = 0; k < 400; k++) {
      const L = ladder(r), fa = fit(L.A), fb = fit(L.B);
      if ((fb(T) < fa(T)) === truthB) okFit++;
      if ((L.B.at(-1)[1] < L.A.at(-1)[1]) === truthB) okBig++;
    }
    const L = ladder(rng(seed)), fa = fit(L.A), fb = fit(L.B), predB = fb(T) < fa(T);
    const xs = Array.from({ length: 61 }, (_, i) => 1e18 * 10 ** (i / 10));
    ch.update([
      { name: "Baseline: true curve", color: "var(--c2)", points: xs.map((C) => [C, lossA(C)]), dash: true },
      { name: "New method: true curve", color: "var(--accent)", points: xs.map((C) => [C, lossB(C, o.cross)]), dash: true },
      { name: "Baseline: ladder runs and fit", color: "var(--c2)", points: xs.map((C) => [C, fa(C)]) },
      { name: "New method: ladder runs and fit", color: "var(--accent)", points: xs.map((C) => [C, fb(C)]) },
      { color: "var(--c2)", points: L.A, dots: true, line: false, r: 5 },
      { color: "var(--accent)", points: L.B, dots: true, line: false, r: 5 },
    ], { xDomain: [1e18, 1e24], yDomain: [1.9, 4.0], marks: [{ x: T, label: "target run" }, { x: 1e20, label: "largest rung" }] });
    st.set(0, truthB ? "New method wins" : "Baseline wins");
    st.set(1, predB ? "New method wins" : "Baseline wins", predB === truthB ? "good" : "bad");
    st.set(2, `${Math.round(okFit / 4)}%`, okFit / 400 > 0.9 ? "good" : okFit / 400 > 0.7 ? "warn" : "bad");
    st.set(3, `${Math.round(okBig / 4)}%`, okBig / 400 > 0.9 ? "good" : okBig / 400 > 0.7 ? "warn" : "bad");
  }

  root.append(
    section(null, prose(`<p>A frontier lab can afford only a handful of full-size training runs a year. Every architecture change, data mix and hyperparameter in those runs was decided on much smaller experiments. How labs make those decisions, and how often they're wrong, is rarely written down. The core tools are <b>scaling ladders</b> (the same recipe at several small sizes, fitted and extrapolated), <b>proxy models</b> for data choices, and evaluation methods that still have signal at small scale.</p>`)),
    section("The scaling ladder: decide with 1% of the compute",
      prose("<p>A new trick beats the baseline at small scale. Will it still win at a target 1,000× larger? The lab trains both at a few sizes (the rungs of the ladder), fits L = E + a·C<sup>−α</sup> to each, and compares the extrapolations. Here the new method has a better constant but a smaller exponent, so its advantage shrinks with scale and eventually reverses. That's a common real-world pattern, not a contrived one.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Where the true curves cross (decades above the ladder)", min: -1, max: 4, step: 0.1, value: o.cross, fmt: (v) => v.toFixed(1), onInput: (v) => { o.cross = v; draw(); } }),
          slider({ label: "Target run (decades above the largest rung)", min: 1, max: 4, step: 0.5, value: o.target, fmt: (v) => `${(10 ** v).toLocaleString()}×`, onInput: (v) => { o.target = v; draw(); } }),
          slider({ label: "Run-to-run noise in loss", min: 0, max: 0.04, step: 0.002, value: o.noise, fmt: (v) => v.toFixed(3), onInput: (v) => { o.noise = v; draw(); } }),
          slider({ label: "Rungs on the ladder", min: 3, max: 8, value: o.rungs, onInput: (v) => { o.rungs = v; draw(); } }),
          button("New ladder", () => { seed++; draw(); }, "primary")),
        h("div.stage-body", st.el, ch.el, h("p.canvas-cap", "Dashed: the true curves, unknown to the lab. Dots: the ladder runs, with noise. Solid: power laws fitted to the dots and extrapolated."))),
      callout("tip", "What you should notice", "<ul style='margin:0;padding-left:18px'><li>Comparing at the biggest rung you trained says the new method wins whenever the crossover is beyond your ladder: right only if the target is before the crossover.</li><li>Fitting the curves can see the crossover coming, but only when noise is small relative to the change in slope. With 0.02–0.03 noise in loss, which is typical across seeds at small scale, the extrapolation is often wrong. More rungs and wider spacing help more than more seeds per rung.</li><li>The further you extrapolate, the less you should trust any fit. Labs add intermediate “checkpoint” runs at 10× and 100× before committing.</li></ul>"),
    ),
    section("How an idea travels through a lab",
      stepper([
        { title: "1. Tiny test (hours)", html: "<p>A 10–100M-parameter model on a few billion tokens. The goal is to kill bad ideas cheaply. Watch the loss curve, not just the final number. Use the lab's standard small-model recipe so results are comparable with everything else ever tried.</p>" },
        { title: "2. Ladder (days)", html: "<p>4–6 sizes spanning 1–2 orders of magnitude of compute, each at its compute-optimal token count (or the lab's chosen over-training ratio). Fit and extrapolate both baseline and change. Also check that the change doesn't hurt throughput: a 3% loss gain at 10% lower tokens/second is a net loss at fixed compute.</p>" },
        { title: "3. Downstream evals with known noise", html: "<p>Loss can improve while capabilities don't. Run the eval suite on every rung, but only trust evals whose signal at this scale exceeds their noise (see below).</p>" },
        { title: "4. The baseline moves", html: "<p>Changes that survive get merged into the baseline recipe, and future ideas are tested against the new one. Ideas that interacted with an old baseline (e.g. a regulariser that helped only because data was repeated) can stop helping, so labs periodically re-ablate.</p>" },
        { title: "5. Medium run, then the big run", html: "<p>A 10–100× intermediate run checks the extrapolation. The big run is launched with monitoring of loss, gradient norms, logit maxima and per-domain losses, with automatic checkpoints and on-call engineers.</p>" },
        { title: "6. Postmortem", html: "<p>Compare the big run's loss with the ladder's prediction. Teams that do this keep calibrating their own extrapolations; Llama 3's report (2024) describes predicting the 405B model's benchmark accuracy from small runs to within a few points.</p>" },
      ]),
    ),
    section("Evals that still have signal at small scale",
      prose("<p>Small models score near chance on hard benchmarks, so accuracy jumps around between checkpoints and tells you nothing. Labs use smoother proxies, and pick evals by their <b>signal-to-noise ratio</b>: how far apart different models score, divided by how much one model's score wobbles between nearby checkpoints (Heineman et al., 2025, AI2).</p>"),
      table(["Technique", "Why it helps"], [
        ["Per-token loss / bits-per-byte on the correct answer", "Continuous, so small improvements show up before accuracy moves"],
        ["Cloze format instead of multiple-choice letters", "Small models can't do the A/B/C/D format at all; scoring each answer's likelihood works earlier"],
        ["Average over checkpoints near the end", "Removes checkpoint-to-checkpoint noise"],
        ["Choose evals with high signal-to-noise", "Drop benchmarks that are noise at your scale"],
        ["Held-out domain losses (code, maths, multilingual)", "Catches a data change that helps one domain and hurts another"],
        ["Proxy models for data decisions", "DataDecide (Magnusson et al., 2025): rankings of pretraining datasets at ~150M parameters predicted the 1B ranking about 80% of the time"],
      ]),
      predict({
        chapter: id,
        question: "A data-filtering change improves a 150M model's validation loss by 1%, and its MMLU score is unchanged at 25.4%. What should you conclude?",
        options: ["The change doesn't help capabilities", "Nothing yet from MMLU: at 150M it's at chance (25%) and has no signal; look at loss on the answers or a cloze-format eval, and check the ladder", "The change hurts at scale", "Use a bigger learning rate"],
        answer: 1,
        explain: "<p>A 150M model can't do MMLU's multiple-choice format, so its accuracy is noise around 25%. Smoother metrics (likelihood of the correct answer) or evals with signal at that scale are what labs use. Whether the 1% loss gain holds at scale is the ladder's job.</p>",
      }),
    ),
    section("Infrastructure that makes this possible",
      table(["Piece", "Why labs insist on it"], [
        ["Config-as-code, one command per run", "Any result can be re-run exactly; diffs between runs are diffs between configs"],
        ["Deterministic, resumable data loader", "Rewinds, ablations and debugging need to replay exact batches"],
        ["Standard small-model recipes (“the 1B baseline”)", "Every experiment is comparable with years of earlier ones"],
        ["Automatic eval on every checkpoint", "Curves, not points; catches regressions as they happen"],
        ["Warmup–stable–decay schedules", "Branch one run into several finished checkpoints for scaling studies"],
        ["Compute accounting per experiment", "Teams have budgets; ideas compete for FLOPs"],
        ["A results log with negative results", "Stops the next person re-trying the same dead end"],
      ]),
    ),
    section("In code",
      code({
        "Fit and extrapolate a ladder": `import numpy as np
from scipy.optimize import curve_fit

def law(C, E, a, alpha): return E + a * (C / 1e18) ** -alpha

def extrapolate(C, L, target, n_boot=1000, rng=np.random.default_rng(0)):
    p, _ = curve_fit(law, C, L, p0=(1.7, 2.0, 0.15), maxfev=20000)
    preds = []
    for _ in range(n_boot):                               # bootstrap over rungs for an interval
        i = rng.integers(0, len(C), len(C))
        try:
            q, _ = curve_fit(law, C[i], L[i], p0=p, maxfev=20000); preds.append(law(target, *q))
        except RuntimeError: pass
    return law(target, *p), np.percentile(preds, [5, 95])

C = np.array([1e18, 3e18, 1e19, 3e19, 1e20])
base, base_ci = extrapolate(C, loss_baseline, target=1e23)
new, new_ci = extrapolate(C, loss_new, target=1e23)
print(f"baseline {base:.3f} {base_ci}, new {new:.3f} {new_ci}")   # overlapping intervals: not decided`,
        "Chinchilla-style rung sizes": `def rung(compute, tokens_per_param=20):
    # C ≈ 6 N D and D = k N  →  N = sqrt(C / (6 k))
    N = (compute / (6 * tokens_per_param)) ** 0.5
    return int(N), int(tokens_per_param * N)
for C in [1e18, 3e18, 1e19, 3e19, 1e20]:
    print(f"{C:.0e} FLOPs → {rung(C)[0] / 1e6:.0f}M params, {rung(C)[1] / 1e9:.1f}B tokens")`,
        "Eval signal-to-noise": `def snr(scores_by_model, last_k=5):
    """scores_by_model: {model: [score at each of the final checkpoints]}"""
    finals = np.array([np.mean(s[-last_k:]) for s in scores_by_model.values()])
    noise = np.mean([np.std(s[-last_k:]) for s in scores_by_model.values()])
    signal = finals.max() - finals.min()                  # spread between models
    return signal / noise                                  # keep evals with high values at your scale`,
      }, "Project 36 builds a five-rung ladder on a free GPU, fits it, and checks the prediction with one bigger run."),
    ),
  );
  draw();
}
