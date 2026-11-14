import { h } from "../lib/core.js";
import { slider, stats, predict, code, callout, section, prose, table, stepper, bars } from "../lib/widgets.js";
import { xyChart } from "../lib/plot.js";

// MMLU scores as reported in each model's paper (few-shot setups differ slightly between reports)
const MMLU = [[2020.4, 43.9, "GPT-3 175B"], [2021.95, 60.0, "Gopher 280B"], [2022.25, 67.6, "Chinchilla 70B"], [2022.3, 69.3, "PaLM 540B"], [2023.2, 86.4, "GPT-4"], [2024.55, 87.3, "Llama 3.1 405B"]];

export default function (root, { id }) {
  // ---------- 1. noise and paired comparisons ----------
  const o = { n: 200, a: 0.72, b: 0.76, agree: 0.7 };
  const nc = xyChart({ width: 460, height: 260, xLog: true, xLabel: "questions in the eval", yLabel: "95% interval on the difference (± points)", xDomain: [20, 5000] });
  const nst = stats([{ label: "Measured difference" }, { label: "± (unpaired)" }, { label: "± (paired, same questions)" }, { label: "Verdict" }]);
  function drawNoise() {
    // per-question correctness of the two models; "agree" is how much their mistakes overlap
    const pa = o.a, pb = o.b, lo = Math.max(0, pa + pb - 1), hi = Math.min(pa, pb), pBoth = lo + o.agree * (hi - lo);
    const varPair = pa + pb - 2 * pBoth - (pa - pb) ** 2, varUn = pa * (1 - pa) + pb * (1 - pb);
    const ci = (v, n) => 1.96 * Math.sqrt(v / n) * 100;
    const ns = Array.from({ length: 41 }, (_, i) => Math.round(20 * 250 ** (i / 40)));
    nc.update([{ name: "Unpaired", color: "var(--c2)", points: ns.map((n) => [n, ci(varUn, n)]) }, { name: "Paired (same questions)", color: "var(--accent)", points: ns.map((n) => [n, ci(varPair, n)]) }], { xDomain: [20, 5000], marks: [{ x: o.n, label: "n" }] });
    const d = (pb - pa) * 100, cp = ci(varPair, o.n);
    nst.set(0, `${d >= 0 ? "+" : ""}${d.toFixed(1)} points`); nst.set(1, `± ${ci(varUn, o.n).toFixed(1)}`); nst.set(2, `± ${cp.toFixed(1)}`);
    nst.set(3, Math.abs(d) > cp ? "Real difference (paired)" : "Can't tell yet", Math.abs(d) > cp ? "good" : "warn");
  }

  // ---------- 2. judge position bias ----------
  const j = { w: 0.6, bias: 0.25, acc: 0.85 };
  const jEl = h("div");
  function drawJudge() {
    // judge: with probability bias it picks whichever answer is shown first; otherwise it picks the better one with accuracy acc
    const fair = j.acc * j.w + (1 - j.acc) * (1 - j.w);
    const aFirst = j.bias + (1 - j.bias) * fair;      // A always shown first
    const bFirst = (1 - j.bias) * fair;               // B shown first: bias favours B
    const swapped = (aFirst + bFirst) / 2;
    jEl.replaceChildren(h("div.eyebrow", "Estimated win rate of model A"), bars([
      { label: "True rate A is better", value: j.w * 100, color: "var(--ink-3)" },
      { label: "Judge, A always first", value: aFirst * 100, color: "var(--bad)" },
      { label: "Judge, both orders averaged", value: swapped * 100, color: "var(--good)" },
    ], { max: 100, fmt: (v) => `${v.toFixed(1)}%` }), h("p.canvas-cap", "Averaging both orders removes position bias. It can't fix the judge's own error rate, which pulls estimates towards 50%: validate judges against human labels."));
  }

  // ---------- 3. saturation ----------
  const sat = xyChart({ height: 260, xLabel: "year", yLabel: "MMLU accuracy (%)", xDomain: [2020, 2025], yDomain: [20, 100] });

  root.append(
    section(null, prose(`<p>You can't improve what you can't measure, and in ML it's easy to measure the wrong thing, or the right thing too noisily to tell. An <b>eval</b> is a set of inputs, a way to get the model's outputs, and a way to grade them. Good evals are the most valuable long-lived asset in an applied AI project: models and prompts change every few months, the eval set stays and tells you whether each change helped.</p>`)),
    section("Is that difference real?",
      prose("<p>Two models on the same eval. Model B scores 4 points higher. With how many questions does that mean anything? A key trick: compare models on the <b>same</b> questions (a paired comparison). Their errors overlap, so the noise in the <i>difference</i> is much smaller than the noise in each score.</p>"),
      h("div.lab",
        h("div.stage",
          h("div.stage-controls",
            slider({ label: "Questions", min: 20, max: 5000, step: 10, value: o.n, onInput: (v) => { o.n = v; drawNoise(); } }),
            slider({ label: "Model A accuracy", min: 0.3, max: 0.95, step: 0.01, value: o.a, fmt: (v) => `${Math.round(v * 100)}%`, onInput: (v) => { o.a = v; drawNoise(); } }),
            slider({ label: "Model B accuracy", min: 0.3, max: 0.95, step: 0.01, value: o.b, fmt: (v) => `${Math.round(v * 100)}%`, onInput: (v) => { o.b = v; drawNoise(); } }),
            slider({ label: "How much their mistakes overlap", min: 0, max: 1, step: 0.05, value: o.agree, fmt: (v) => `${Math.round(v * 100)}%`, onInput: (v) => { o.agree = v; drawNoise(); } })),
          h("div.stage-body", nst.el)),
        nc.el),
      callout("tip", "Rules of thumb", "<ul style='margin:0;padding-left:18px'><li>The standard error of an accuracy is √(p(1−p)/n): about ±9 points (95%) at n = 100, ±3 at n = 1,000. Small evals can only detect big changes.</li><li>Always compare models on the same items and look at the per-item differences; when the models' mistakes overlap a lot, the interval shrinks dramatically.</li><li>Report intervals. “72% vs 76% on 100 questions” is not a result.</li></ul>"),
    ),
    section("LLM judges and their biases",
      prose("<p>For open-ended outputs, a strong model often grades pairs of answers. It's cheap and correlates well with people, but judges have biases: they favour the answer shown first (or second), longer answers, and sometimes their own model family. The cheapest fix for position bias: judge each pair twice, in both orders.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "True rate A is better", min: 0.2, max: 0.8, step: 0.01, value: j.w, fmt: (v) => `${Math.round(v * 100)}%`, onInput: (v) => { j.w = v; drawJudge(); } }),
          slider({ label: "Judge's position bias", min: 0, max: 0.5, step: 0.01, value: j.bias, fmt: (v) => `${Math.round(v * 100)}%`, onInput: (v) => { j.bias = v; drawJudge(); } }),
          slider({ label: "Judge accuracy (when unbiased)", min: 0.5, max: 1, step: 0.01, value: j.acc, fmt: (v) => `${Math.round(v * 100)}%`, onInput: (v) => { j.acc = v; drawJudge(); } })),
        h("div.stage-body", jEl)),
    ),
    section("Benchmarks saturate",
      prose("<p>MMLU, 57 subjects of multiple-choice exam questions, was hard when released in 2020. Within four years top models approached the estimated expert level of about 90%, and the remaining errors are partly wrong answer keys. Every popular benchmark follows this path, which is why new, harder ones keep appearing (GPQA, Humanity's Last Exam, SWE-bench, ARC-AGI-2) and why your own task-specific eval matters more than leaderboards.</p>"),
      sat.el,
      predict({
        chapter: id,
        question: "A new model is 10 points better on a public benchmark but no better on your own eval of real user tasks. What's the most likely explanation?",
        options: ["Your eval is broken", "The benchmark measures something different from your task, or it leaked into training data; trust the eval built from your real tasks", "The model is worse", "Benchmarks are always right"],
        answer: 1,
        explain: "<p>Public benchmarks are useful for broad comparisons but can be contaminated, saturated or simply about different skills. A task-specific eval with real inputs and clear grading is the ground truth for your use case.</p>",
      }),
    ),
    section("Building an eval that lasts",
      stepper([
        { title: "1. Collect real inputs", html: "<p>Start from real user requests and real failures, not invented examples. 50–200 good items beat 10,000 synthetic ones at first.</p>" },
        { title: "2. Define success precisely", html: "<p>For each item: an exact answer, a test to run, or a rubric a person (or judge model) can apply consistently. If two people disagree on the grade, fix the rubric.</p>" },
        { title: "3. Grade with code where possible", html: "<p>Exact match, unit tests, JSON schema checks, regexes. Use model judges for what code can't check, and check the judge against human grades.</p>" },
        { title: "4. Read the transcripts", html: "<p>The score tells you <i>whether</i>; the transcripts tell you <i>why</i>. Look at failures every time.</p>" },
        { title: "5. Keep it alive", html: "<p>Add every new failure from production. Keep a held-out slice you never tune against. Version it like code.</p>" },
      ]),
      table(["Metric", "Use for"], [
        ["Accuracy / exact match", "Classification, short answers"],
        ["pass@k", "Code and maths: chance at least one of k samples is correct"],
        ["Win rate vs a baseline (judged)", "Open-ended quality"],
        ["Rubric score", "Multi-criteria tasks (correctness, completeness, tone)"],
        ["Task success rate + cost + latency", "Agents: did it finish, at what price, how fast"],
      ]),
    ),
    section("In code",
      code({
        "Paired comparison": `import numpy as np
a = np.array(correct_model_a); b = np.array(correct_model_b)       # 0/1 per question, same questions
diff = b - a
mean, se = diff.mean(), diff.std(ddof=1) / np.sqrt(len(diff))
print(f"B − A = {mean:+.3f} ± {1.96 * se:.3f} (95%)")`,
        "Unbiased pass@k": `from math import comb
def pass_at_k(n, c, k):
    """n samples generated, c of them correct: probability at least one of k random samples is correct."""
    if n - c < k: return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)            # Chen et al., 2021`,
        "Judge both orders": `def judged_win_rate(judge, pairs):
    wins = 0.0
    for q, a, b in pairs:
        wins += (judge(q, first=a, second=b) == "first") * 0.5
        wins += (judge(q, first=b, second=a) == "second") * 0.5      # swapped order
    return wins / len(pairs)`,
      }),
    ),
  );
  drawNoise(); drawJudge();
  sat.update([
    { name: "Reported MMLU (few-shot)", color: "var(--accent)", points: MMLU.map(([x, y]) => [x, y]), dots: true, r: 5 },
    { name: "Estimated expert level (~89.8%)", color: "var(--ink-3)", points: [[2020, 89.8], [2025, 89.8]], dash: true },
    { name: "Random guessing (25%)", color: "var(--c2)", points: [[2020, 25], [2025, 25]], dash: true },
  ], { xDomain: [2020, 2025], yDomain: [20, 100] });
}
