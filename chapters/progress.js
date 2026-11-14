import { h } from "../lib/core.js";
import { slider, stats, bars, predict, code, callout, section, prose, table } from "../lib/widgets.js";
import { xyChart } from "../lib/plot.js";

const L10 = Math.log10;

export default function (root, { id }) {
  const p = { hw: 4.5, algo: 3, post: 1.5, year: 2026, width: 2.5 };
  let pending = 0;
  const st = stats([{ label: "Effective compute vs 2019" }, { label: "Hardware and spend (orders of mag.)" }, { label: "Algorithms, data, RL (orders of mag.)" }, { label: "A benchmark goes 10% → 90% in" }]);
  const chart = xyChart({ height: 270, yLog: true, xLabel: "year", yLabel: "× the 2019 frontier", xDomain: [2019, 2030], fmtX: (v) => String(Math.round(v)), fmtY: (v) => (v >= 1e3 ? `1e${Math.round(L10(v))}` : String(Math.round(v))), label: "Effective training compute over time" });
  const barBox = h("div");

  function run() {
    const yrs = p.year - 2019;
    const g = p.hw * p.algo * p.post;
    const ooms = { hw: yrs * L10(p.hw), algo: yrs * L10(p.algo), post: Math.max(0, p.year - 2024) * L10(p.post) }; // RL and test-time scaling took off in 2024
    const total = ooms.hw + ooms.algo + ooms.post;
    st.set(0, `10^${total.toFixed(1)}×`);
    st.set(1, `${ooms.hw.toFixed(1)} OOM`);
    st.set(2, `${(ooms.algo + ooms.post).toFixed(1)} OOM`);
    st.set(3, `${(p.width / L10(g)).toFixed(1)} years`, p.width / L10(g) < 2 ? "warn" : "");
    const xs = Array.from({ length: 45 }, (_, i) => 2019 + i * 0.25);
    chart.update([
      { name: "Hardware and spend only", color: "var(--c2)", points: xs.map((x) => [x, Math.pow(p.hw, x - 2019)]) },
      { name: "+ better algorithms and data", color: "var(--c1)", points: xs.map((x) => [x, Math.pow(p.hw * p.algo, x - 2019)]) },
      { name: "+ post-training and test-time compute", color: "var(--c4)", points: xs.map((x) => [x, Math.pow(p.hw * p.algo, x - 2019) * Math.pow(p.post, Math.max(0, x - 2024))]) },
    ], { marks: [{ x: p.year, label: String(p.year) }] });
    barBox.replaceChildren(bars([
      { label: "Hardware and spend", value: ooms.hw, color: "var(--c2)" },
      { label: "Algorithms and data", value: ooms.algo, color: "var(--c1)" },
      { label: "Post-training, test-time", value: ooms.post, color: "var(--c4)" },
    ], { max: Math.max(1, total), fmt: (v) => `${v.toFixed(1)} OOM` }));
  }
  const later = () => { cancelAnimationFrame(pending); pending = requestAnimationFrame(run); };

  root.append(
    section(null, prose(`<p>Every year the best models get noticeably better, and it's tempting to credit one thing: bigger GPUs, a clever architecture, a new training trick. The honest answer is that progress is a product of several multipliers, each growing every year, and products of exponentials grow very fast.</p>
<p>The standard way to account for it is <b>effective compute</b>: how much compute you would have needed, with 2019 methods, to get today's result. Physical compute grows because chips improve and labs spend more. Algorithmic progress (architectures, optimisers, data quality, schedules) means each FLOP buys more. Since 2024, a third multiplier has become large: post-training with RL and spending compute at test time. Slide the rates and see what they compound to. (The third multiplier starts in 2024 here.)</p>`)),
    section("Compound the multipliers",
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Physical training compute, × per year", min: 1, max: 8, step: 0.1, value: p.hw, fmt: (v) => `${v.toFixed(1)}×`, onInput: (v) => { p.hw = v; later(); } }),
          slider({ label: "Algorithms and data, × per year", min: 1, max: 6, step: 0.1, value: p.algo, fmt: (v) => `${v.toFixed(1)}×`, onInput: (v) => { p.algo = v; later(); } }),
          slider({ label: "Post-training and test-time, × per year", min: 1, max: 4, step: 0.1, value: p.post, fmt: (v) => `${v.toFixed(1)}×`, onInput: (v) => { p.post = v; later(); } }),
          slider({ label: "Year", min: 2020, max: 2030, step: 1, value: p.year, fmt: String, onInput: (v) => { p.year = v; later(); } }),
          slider({ label: "Benchmark difficulty span (orders of magnitude)", min: 1, max: 6, step: 0.5, value: p.width, fmt: (v) => v.toFixed(1), onInput: (v) => { p.width = v; later(); } }),
        ),
        h("div.stage-body", st.el, chart.el, barBox),
      ),
      callout("tip", "Things to try", "<p>The defaults use published estimates for the first two rates (Epoch AI: frontier training compute up about 4–5× a year; Ho et al., 2024: the compute needed for a given language-model performance halves roughly every 8 months, about 3× a year, with wide error bars). The third rate is my rough guess, not a measurement. Together they give about 20× a year in effective compute. Now look at the last tile: a benchmark whose easy and hard questions span 2.5 orders of magnitude of effective compute goes from 10% to 90% in under two years. That's why benchmarks now saturate faster than people can build them. Set algorithms to 1×: progress still happens, but a benchmark lasts twice as long.</p>"),
      predict({
        chapter: id,
        question: "Algorithmic progress halves the compute needed every ~8 months. Physical compute grows ~4.5× a year. Over 4 years, which contributes more orders of magnitude?",
        options: ["Algorithms, by far", "Physical compute, but algorithms are close behind", "They're exactly equal", "Neither: data is the only thing that matters"],
        answer: 1,
        explain: "<p>Physical: 4 × log₁₀(4.5) ≈ 2.6 orders of magnitude. Algorithms: 4 × log₁₀(2.8) ≈ 1.8. Both are huge, and they multiply. The uncertainty on the algorithmic rate is wide (the 95% interval spans 5 to 14 months for a halving), so ‘close behind’ could be ‘level’.</p>",
      }),
    ),
    section("Where gains actually came from, 2023–2026",
      prose(`<p>Effective compute is an accounting tool. Here is what sits inside it, with my best estimate of how much each lever moved real capability over the last few model generations. The status column tells you how much of that is measured.</p>`),
      table(["Lever", "What changed", "My estimate of its share", "Status"], [
        ["More pretraining compute", "Bigger runs on more tokens; overtraining small models far past Chinchilla", "Large, but a shrinking share", "Published trend (Epoch AI)"],
        ["Data quality", "Model-based filtering, deduplication, rewriting the web, more code and math", "Large: the most underrated lever", "Published ablations (FineWeb-Edu, DCLM)"],
        ["Distillation", "Small models trained on big models' outputs or logits", "Most of the gains at small sizes", "Published practice (Gemma, Llama 3.2, many others)"],
        ["RL with verifiable rewards", "Reasoning models trained on checkable math, code and tool tasks", "The largest single jump in reasoning since 2024", "Published (o1, R1, many reports)"],
        ["Test-time compute", "Thinking longer, sampling more, checking answers", "Large on hard reasoning, small on knowledge", "Published (scaling curves)"],
        ["Architecture", "MoE, attention variants, better normalisation", "Real but modest; mostly efficiency", "Published (each is a few-× efficiency)"],
        ["Tooling and scaffolding", "Agents with code execution, search, memory", "Large on agentic tasks; invisible on static benchmarks", "Partly measured (agent horizons)"],
        ["Evals and environments", "Better tests and RL environments to train against", "Indirect but decisive: you improve what you can measure", "My view"],
      ]),
      callout("note", "Three things I believe about progress that are hard to prove", "<p><b>1. Data and evaluation are the real bottlenecks, not ideas (~70%).</b> Most of a lab's effort goes into data pipelines, environments and evals; architecture changes are rarer and smaller than papers suggest. If you could see inside a lab's commit history, I'd bet most changes touch data, not models.</p><p><b>2. The data wall is softer than forecasts say (~60%).</b> Published estimates put the stock of quality human text at a few hundred trillion tokens, used up within years. But RL environments, synthetic data with verifiers, and multimodal data each add supply, and repeated high-quality data loses value slowly up to several epochs. The wall will show up as rising cost per point, not a stop.</p><p><b>3. Benchmarks overstate progress on what they measure and understate it on what they don't (~65%).</b> Labs optimise toward the benchmarks people watch, which inflates those; meanwhile, abilities nobody measures (judgment in long tasks, taste, knowing when to ask) improve without a number attached. Reading transcripts is still the best way to know what changed.</p>"),
    ),
    section("In code",
      code({
        "Effective-compute arithmetic": `import math
hw, algo, post = 4.5, 2.8, 1.5                 # × per year (the last is a guess)
for years in [1, 2, 4, 7]:
    eff = (hw * algo * post) ** years
    print(f"{years} yr: 10^{math.log10(eff):.1f}× effective compute")

# how long a benchmark lasts: its span (orders of magnitude of effective compute from 10% to 90%)
span = 2.5
print(f"10% → 90% in {span / math.log10(hw * algo * post):.1f} years")`,
        "Measure algorithmic progress yourself": `# train the same small model with a 2019 recipe and a 2025 recipe; find the compute
# the old recipe needs to match the new one's loss
old = {"norm": "post-LN", "opt": "Adam", "sched": "linear-warmup-invsqrt", "data": "raw-web"}
new = {"norm": "pre-RMSNorm", "opt": "AdamW", "sched": "wsd", "data": "filtered-web", "pos": "rope"}
loss_new = train(size=50e6, tokens=1e9, **new)
for tokens in [1e9, 2e9, 4e9, 8e9]:
    if train(size=50e6, tokens=tokens, **old) <= loss_new:
        print(f"the old recipe needs {tokens/1e9:.0f}× the tokens"); break`,
      }),
    ),
  );
  run();
}
