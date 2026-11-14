import { h } from "../lib/core.js";
import { slider, stats, predict, code, callout, section, prose, table, dilemma } from "../lib/widgets.js";
import { xyChart, fmtSI } from "../lib/plot.js";

// Chinchilla's parametric fit (Hoffmann et al. 2022, approach 3), loss in nats per token.
const E = 1.69, A = 406.4, B = 410.7, ALPHA = 0.34, BETA = 0.28;
const loss = (N, D) => E + A / N ** ALPHA + B / D ** BETA;
function optimal(C) {
  let best = null;
  for (let lg = 6; lg <= 14; lg += 0.01) { const N = 10 ** lg, D = C / (6 * N), l = loss(N, D); if (!best || l < best.l) best = { N, D, l }; }
  return best;
}

export default function (root, { id }) {
  let logC = 23.0, tpp = 20;
  const iso = xyChart({ height: 300, xLog: true, xLabel: "model size N (parameters)", yLabel: "predicted loss (nats/token)" });
  const st = stats([{ label: "Compute-optimal size" }, { label: "Training tokens" }, { label: "Tokens per parameter" }, { label: "Predicted loss" }]);
  function drawIso() {
    const C = 10 ** logC, series = [], opt = [];
    const budgets = [-2, -1, 0, 1, 2].map((k) => logC + k);
    const colors = ["c6", "c5", "c1", "c3", "c4"];
    budgets.forEach((lc, i) => {
      const Cb = 10 ** lc, pts = [];
      for (let lg = 6.5; lg <= 13; lg += 0.05) { const N = 10 ** lg, D = Cb / (6 * N); if (D > 1e8) pts.push([N, loss(N, D)]); }
      series.push({ name: `C = 1e${lc.toFixed(1)}`, color: `var(--${colors[i]})`, points: pts });
      const o = optimal(Cb); opt.push([o.N, o.l]);
    });
    series.push({ name: "Optimum for each budget", color: "var(--ink)", points: opt, dots: true, line: false });
    const o = optimal(C);
    iso.update(series, { yDomain: [1.8, 4], xDomain: [10 ** 6.5, 1e13], marks: [{ x: o.N, label: "optimal size" }] });
    st.set(0, fmtSI(o.N)); st.set(1, fmtSI(o.D)); st.set(2, (o.D / o.N).toFixed(0)); st.set(3, o.l.toFixed(3));
  }

  // overtraining for cheaper inference
  const ot = stats([{ label: "Model size at this ratio" }, { label: "Loss penalty vs optimal" }, { label: "Inference cost vs optimal model" }]);
  function drawOT() {
    const C = 10 ** logC, o = optimal(C), N = Math.sqrt(C / (6 * tpp)), D = C / (6 * N);
    ot.set(0, fmtSI(N)); const pen = loss(N, D) - o.l;
    ot.set(1, `+${pen.toFixed(3)} nats`, pen < 0.02 ? "good" : pen > 0.1 ? "bad" : "warn");
    ot.set(2, `${((N / o.N) * 100).toFixed(0)}%`, N < o.N ? "good" : "warn");
  }

  root.append(
    section(null, prose(`<p>In 2020 researchers at OpenAI noticed something remarkable: a language model's loss falls as a smooth <b>power law</b> in model size, data and compute, over many orders of magnitude (Kaplan et al., 2020). You can train small models, fit a line on a log-log plot, and predict how good a model 1,000× bigger will be before you build it. That predictability is why labs were willing to spend hundreds of millions on single training runs.</p>
<p>DeepMind's <b>Chinchilla</b> paper (Hoffmann et al., 2022) fitted <code>L(N, D) = E + A/N^α + B/D^β</code>: an irreducible floor E (the entropy of text), plus a term that shrinks with parameters N, plus one that shrinks with training tokens D. Training costs about C = 6·N·D FLOPs. So for a fixed budget, how should you split it?</p>`)),
    section("Spend a compute budget",
      prose("<p>Each curve fixes the compute and varies the split: tiny models trained on huge data on the left, huge models trained on little data on the right. Every curve has a valley. The dots trace the optimum as the budget grows.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Compute budget (FLOPs)", min: 19, max: 26.5, step: 0.1, value: logC, fmt: (v) => `1e${v.toFixed(1)}`, onInput: (v) => { logC = v; drawIso(); drawOT(); }, hint: "GPT-3: 3e23 · Llama 3 70B: ~6e24" })),
        h("div.stage-body", st.el, iso.el)),
      callout("tip", "Things to try", "<ul style='margin:0;padding-left:18px'><li>Set the budget to 3.1e23 (GPT-3's). The fit says the best model is around 25B parameters on ~2T tokens, not 175B on 300B: GPT-3 was far too big for its data. Chinchilla (70B on 1.4T tokens) beat the 280B Gopher with the same compute.</li><li>Note how flat each valley is near the bottom: being 2× off the optimal size costs very little.</li><li>Watch the tokens-per-parameter readout: this published fit gives 40 to 120. Chinchilla's other two methods gave about 20, and a 2024 replication (Besiroglu et al.) found the fit's constants were slightly off; corrected, it also lands near 20. “About 20 tokens per parameter” is the rule of thumb everyone quotes.</li></ul>"),
      predict({
        chapter: id,
        question: "You get 4× more compute. Under Chinchilla scaling, how should you grow the model?",
        options: ["4× bigger, same data", "Same size, 4× the data", "About 2× bigger and about 2× the data", "16× bigger"],
        answer: 2,
        explain: "<p>Compute-optimal N and D both grow roughly as √C, so 4× compute means about 2× parameters and 2× tokens. Kaplan's earlier analysis had suggested putting most of it into model size; Chinchilla's more careful learning-rate schedules showed data matters as much.</p>",
      }),
    ),
    section("But inference costs money too",
      prose("<p>Chinchilla minimises <i>training</i> compute. A model served to millions of people spends far more compute on inference over its lifetime, and inference cost scales with N. So labs now deliberately <b>overtrain</b> smaller models: Llama 3 8B saw 15 trillion tokens, about 1,900 tokens per parameter, almost 100× the Chinchilla ratio.</p>"),
      h("div.stage",
        h("div.stage-controls", slider({ label: "Tokens per parameter", min: 1, max: 3000, step: 1, value: tpp, onInput: (v) => { tpp = v; drawOT(); }, hint: "Chinchilla ≈ 20 · Llama 3 8B ≈ 1,900" })),
        h("div.stage-body", ot.el)),
      dilemma("A fixed training budget. Compute-optimal, or small and overtrained?", [
        { name: "Compute-optimal (≈20 tokens/param)", pros: ["Lowest loss for the training budget", "Best for a research model or a one-off"], cons: ["Bigger model: every query costs more", "Needs more GPUs to serve"] },
        { name: "Overtrained small model", pros: ["Much cheaper and faster to serve", "Fits on smaller hardware, even phones"], cons: ["Slightly worse at the same training cost", "Needs enormous amounts of data"] },
      ], "For anything deployed at scale, overtrain. The loss penalty is small and inference savings are large. Data, not compute, becomes the limit: hence the data chapter next."),
    ),
    section("What the laws don't say",
      table(["Caveat", "Detail"], [
        ["Loss is not ability", "Smooth loss curves can hide abrupt jumps in specific skills. Whether “emergent abilities” are real or an artefact of all-or-nothing metrics is debated (Wei et al. 2022; Schaeffer et al. 2023)."],
        ["Data runs out", "High-quality text on the web is estimated at tens of trillions of tokens. Repeating data up to ~4 epochs costs little (Muennighoff et al. 2023); beyond that returns fall fast."],
        ["Constants depend on everything", "Architecture, data mix and tokenizer change A, B, α, β. Fits only transfer within a setup."],
        ["New axes", "Test-time compute (thinking longer), RL compute and data quality each have their own scaling curves, covered in later chapters."],
      ]),
    ),
    section("In code",
      code({
        NumPy: `import numpy as np
from scipy.optimize import curve_fit

# losses measured from your own small runs (project 07)
N = np.array([1e6, 3e6, 1e7, 3e7, 1e8])
L = np.array([4.12, 3.71, 3.35, 3.08, 2.86])

def law(n, a, alpha, c):
    return a * n ** -alpha + c

(a, alpha, c), _ = curve_fit(law, N, L, p0=[50, 0.3, 1.5], maxfev=20000)
print(f"L(N) = {a:.1f} * N^-{alpha:.3f} + {c:.2f}")
print("prediction for 1e9 params:", law(1e9, a, alpha, c))`,
      }, "Project 07 trains six sizes, fits this curve and checks the prediction on a model it hasn't trained yet."),
    ),
  );
  drawIso(); drawOT();
}
