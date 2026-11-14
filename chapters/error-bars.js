import { h } from "../lib/core.js";
import { slider, segmented, stats, predict, code, callout, section, prose, table } from "../lib/widgets.js";
import { xyChart } from "../lib/plot.js";
import { evalNoise } from "../lib/hidden.js";

export default function (root, { id }) {
  let N = 500, delta = 0.02, k = 1, formats = 1, pending = 0;
  const st = stats([{ label: "Standard error of the gap" }, { label: "Reruns where B looks worse" }, { label: "Reruns that are “significant”" }, { label: "Average reported gain" }]);
  const chart = xyChart({ height: 270, xLabel: "measured gain of B over A (percentage points)", yLabel: "share of reruns", fmtX: (v) => `${v > 0 ? "+" : ""}${v}`, fmtY: (v) => `${Math.round(v * 100)}%`, label: "Distribution of the measured gain across reruns" });

  function run() {
    const reruns = Math.max(80, Math.min(400, Math.round(3e6 / (N * k * (formats + 1)))));
    const o = evalNoise({ N, delta, k, formats, reruns });
    const se = o.reduce((a, x) => a + x.sePaired, 0) / o.length;
    const wrong = o.filter((x) => x.diff < 0).length / o.length;
    const sig = o.filter((x) => x.diff > 1.96 * x.sePaired).length / o.length;
    const mean = o.reduce((a, x) => a + x.diff, 0) / o.length;
    st.set(0, `±${(se * 100).toFixed(2)} pts`, se > delta ? "bad" : se > delta / 2 ? "warn" : "good");
    st.set(1, `${Math.round(wrong * 100)}%`, wrong > 0.1 ? "bad" : wrong > 0.02 ? "warn" : "good");
    st.set(2, `${Math.round(sig * 100)}%`, sig < 0.5 ? "warn" : "good");
    st.set(3, `+${(mean * 100).toFixed(1)} pts`, mean > delta * 1.5 + 0.005 ? "bad" : "");
    const lo = -8, hi = 12, bw = 1, bins = new Array(Math.round((hi - lo) / bw)).fill(0);
    for (const x of o) { const b = Math.floor((x.diff * 100 - lo) / bw); if (b >= 0 && b < bins.length) bins[b]++; }
    chart.update([{ name: "reruns", color: "var(--c1)", points: bins.map((c, i) => [lo + (i + 0.5) * bw, c / o.length]), dots: true, r: 3 }],
      { xDomain: [lo, hi], yDomain: [0, Math.max(0.1, ...bins.map((c) => c / o.length)) * 1.1], marks: [{ x: 0 }, { x: delta * 100, label: delta ? "true gain" : "true gain = 0" }] });
  }
  const later = () => { cancelAnimationFrame(pending); pending = requestAnimationFrame(run); };
  const log = (label, min, max, val, fmt, set) => slider({ label, min: Math.log10(min), max: Math.log10(max), step: 0.05, value: Math.log10(val), fmt: (x) => fmt(10 ** x), onInput: (x) => { set(10 ** x); later(); } });

  root.append(
    section(null, prose(`<p>Model B scores 2 points higher than model A on a benchmark. Is B better? Leaderboards, papers and launch posts treat that as a yes. People who run evals for a living know it often isn't, and almost nobody prints error bars. Here, B really is better by a true gain you choose. Every rerun draws a fresh benchmark of N questions from the same population, as a new version of the test would, and samples answers again. The chart shows what you'd see across 300 such reruns.</p>`)),
    section("How sure can you be?",
      h("div.stage",
        h("div.stage-controls",
          log("Questions in the benchmark", 50, 10000, N, (v) => Math.round(v).toLocaleString(), (v) => (N = Math.round(v))),
          slider({ label: "True gain of B", min: 0, max: 0.06, step: 0.005, value: delta, fmt: (v) => `${(v * 100).toFixed(1)} pts`, onInput: (v) => { delta = v; later(); } }),
          segmented({ label: "Samples per question", options: [{ value: 1, label: "1" }, { value: 4, label: "4" }, { value: 16, label: "16" }], value: k, onChange: (v) => { k = v; later(); } }),
          slider({ label: "Prompt formats tried for B (best reported)", min: 1, max: 10, value: formats, onInput: (v) => { formats = v; later(); } }),
        ),
        h("div.stage-body", st.el, chart.el),
      ),
      callout("tip", "Things to try", "<p>At 500 questions and a true gain of 2 points, B looks <i>worse</i> in about one rerun in four, and a significance test agrees with the truth less than a tenth of the time. Push questions to 5,000 or sample 16 answers per question: the bell narrows around the truth. Now set the true gain to zero and try 8 prompt formats for B, reporting the best: B “wins” by several points with no real difference. That's how many leaderboard gaps are made.</p>"),
      predict({
        chapter: id,
        question: "A benchmark has 500 questions and a model scores around 60%. Roughly how big is the 95% confidence interval on its score?",
        options: ["±0.5 points", "±1 point", "±4 points", "±10 points"],
        answer: 2,
        explain: "<p>Standard error = √(0.6 × 0.4 / 500) ≈ 2.2 points, and 95% is ±1.96 SE ≈ ±4.3 points. Comparing two models on the same questions (a paired test) shrinks it somewhat, and sampling several answers per question shrinks it more, but a 1–2 point gap on a 500-question benchmark is usually noise. Many popular benchmarks have only a few hundred items (GPQA Diamond has 198, AIME has 30 a year).</p>",
      }),
    ),
    section("Eval habits from people who've been burned",
      table(["Habit", "Why"], [
        ["Report a 95% interval, or at least the standard error", "Miller (2024), “Adding error bars to evals”: the numbers above, applied to real benchmarks"],
        ["Compare models on the same questions with a paired test", "Most question difficulty is shared, so differences have less noise than raw scores"],
        ["Cluster standard errors when questions come in groups", "Ten questions about one passage are not ten independent samples; naive SEs can be 2–3× too small"],
        ["Fix the prompt, template and parser before looking at results", "Choosing them afterwards inflates gains (the formats slider)"],
        ["Run several seeds for anything you train", "Seed-to-seed variance in fine-tuning often exceeds the effect you're testing"],
        ["Read 50 failures by hand", "Half of a ‘regression’ is often a parser that chokes on the new model's formatting"],
        ["Hold out a private test set and check for contamination", "Public benchmarks leak into web crawls; scores on a fresh copy can drop 5–10 points"],
      ]),
      callout("note", "My intuition (~75% confident)", "<p>For frontier models in late 2026, a large share of reported gains of under 2 points on benchmarks with fewer than 1,000 items would not survive a paired test with 4 samples per question. Notebook 44 lets you test this on any two models whose per-question outputs you have.</p>"),
    ),
    section("In code",
      code({
        "Paired difference with CI": `import numpy as np

def paired_gap(a, b):
    """a, b: per-question scores (0/1, or mean over k samples) on the SAME questions."""
    d = np.asarray(b, float) - np.asarray(a, float)
    se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), (d.mean() - 1.96 * se, d.mean() + 1.96 * se)

gap, (lo, hi) = paired_gap(scores_a, scores_b)
print(f"B − A = {gap:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")`,
        "Clustered SE": `def clustered_se(d, cluster_ids):
    """Questions that share a passage/task are correlated: sum within clusters first."""
    d, ids = np.asarray(d, float), np.asarray(cluster_ids)
    mean = d.mean()
    sums = np.array([(d[ids == c] - mean).sum() for c in np.unique(ids)])
    return np.sqrt((sums ** 2).sum()) / len(d)`,
        "How many questions do I need?": `from scipy.stats import norm

def questions_needed(gap, sd_diff=0.5, alpha=0.05, power=0.8):
    """sd_diff: std of per-question differences (≈0.5 for single 0/1 samples, less with k samples)."""
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return int(np.ceil((z * sd_diff / gap) ** 2))

questions_needed(0.02)    # → about 4,900 questions to reliably detect 2 points`,
      }),
    ),
  );
  run();
}
