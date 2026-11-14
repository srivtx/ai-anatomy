import { h } from "../lib/core.js";
import { slider, segmented, stats, predict, code, callout, section, prose, table } from "../lib/widgets.js";
import { xyChart } from "../lib/plot.js";
import { emergeMetrics } from "../lib/hidden.js";
import { rng } from "../lib/core.js";

export default function (root, { id }) {
  const p = { L: 8, models: 8, N: 200, sharp: 0, metric: "exact" };
  let pending = 0;
  const st = stats([{ label: "Per-token accuracy" }, { label: "Your metric" }, { label: "Steepest rise per 10×" }, { label: "Verdict" }]);
  const chart = xyChart({ height: 300, xLabel: "training compute (log₁₀ FLOPs)", yLabel: "score", xDomain: [18, 25], yDomain: [0, 1], fmtX: (v) => `1e${Math.round(v)}`, fmtY: (v) => `${Math.round(v * 100)}%`, label: "Benchmark score against training compute" });

  function run() {
    const r = rng(11 + p.models * 7 + p.N);
    const opts = { sharp: p.sharp };
    const xs = Array.from({ length: 141 }, (_, i) => 18 + i * 0.05);
    const metricOf = (m) => (p.metric === "exact" ? m.exact : p.metric === "partial" ? m.partial : Math.exp(m.logp / p.L));
    const curve = xs.map((x) => [x, metricOf(emergeMetrics(x, p.L, opts))]);
    const tok = xs.map((x) => [x, emergeMetrics(x, p.L, opts).token]);
    const mx = Array.from({ length: p.models }, (_, i) => 18.5 + (6 * i) / Math.max(1, p.models - 1));
    const measured = mx.map((x) => {
      const truth = metricOf(emergeMetrics(x, p.L, opts));
      let k = 0; for (let j = 0; j < p.N; j++) if (r() < truth) k++;
      return [x, p.metric === "exact" ? k / p.N : truth];
    });
    const slope = (c) => { let m = 0, at = c[0][0]; for (let i = 20; i < c.length; i++) { const d = c[i][1] - c[i - 20][1]; if (d > m) { m = d; at = c[i][0]; } } return [m, at]; };
    const [ms, mat] = slope(curve), [ts] = slope(tok);
    st.set(0, `${Math.round(emergeMetrics(mx[0], p.L, opts).token * 100)}% → ${Math.round(emergeMetrics(mx.at(-1), p.L, opts).token * 100)}%`);
    st.set(1, `${Math.round(measured[0][1] * 100)}% → ${Math.round(measured.at(-1)[1] * 100)}%`);
    st.set(2, `+${Math.round(ms * 100)} points`);
    const real = ts > 0.45, mirage = !real && ms > 1.3 * ts;
    st.set(3, real ? "a real transition" : mirage ? "sudden: the metric" : "smooth", real ? "warn" : mirage ? "bad" : "good");
    chart.update([
      { name: "Per-token accuracy (what the model is actually getting better at)", color: "var(--c4)", points: tok },
      { name: p.metric === "exact" ? `Exact match on ${p.L}-token answers` : p.metric === "partial" ? "Partial credit (share of tokens right)" : "Probability of the right answer, per token", color: "var(--c1)", points: curve },
      { name: "Models you actually trained and measured", color: "var(--c2)", points: measured, dots: true, r: 5, line: false },
    ]);
  }
  const later = () => { cancelAnimationFrame(pending); pending = requestAnimationFrame(run); };

  root.append(
    section(null, prose(`<p>Some abilities seem to appear out of nowhere: models below a certain size score zero, and a slightly bigger one suddenly scores well. This was called <b>emergence</b> (Wei et al., 2022), and it's an unnerving idea: if abilities switch on unpredictably, you can't forecast what the next model will do.</p>
<p>Then Schaeffer et al. (2023) pointed out that much of it is the <b>ruler, not the model</b>. If an answer is 8 tokens long and graded all-or-nothing, a model that gets each token right with probability p scores p⁸. p can rise smoothly while p⁸ sits near zero and then shoots up. Below you can make emergence appear and disappear by changing only the metric, and then add a genuine phase change to see what real emergence looks like through the same lens.</p>`)),
    section("Make emergence appear and disappear",
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "How you grade", options: [{ value: "exact", label: "Exact match" }, { value: "partial", label: "Partial credit" }, { value: "prob", label: "Probability of the right answer" }], value: p.metric, onChange: (v) => { p.metric = v; run(); } }),
          slider({ label: "Answer length (tokens)", min: 1, max: 30, step: 1, value: p.L, fmt: String, onInput: (v) => { p.L = v; later(); } }),
          slider({ label: "Models trained (sizes measured)", min: 3, max: 25, step: 1, value: p.models, fmt: String, onInput: (v) => { p.models = v; later(); } }),
          slider({ label: "Test questions per model", min: 20, max: 2000, step: 20, value: p.N, fmt: String, onInput: (v) => { p.N = v; later(); } }),
          slider({ label: "A genuine phase change (circuit forms)", min: 0, max: 8, step: 0.5, value: p.sharp, fmt: (v) => (v ? `sharpness ${v}` : "off"), onInput: (v) => { p.sharp = v; later(); } }),
        ),
        h("div.stage-body", st.el, chart.el),
      ),
      callout("tip", "Things to try", "<p>With exact match on 8-token answers, the score sits near zero until about 1e21.5 and then climbs fast, even though per-token accuracy (yellow) has been rising all along. Switch to partial credit or answer probability: the cliff is gone. Make answers longer: the ‘emergence point’ moves right and gets sharper, which is why long-answer tasks look most emergent. Train only 3–4 model sizes: the gaps between them make any rise look like a jump. Then turn on a genuine phase change: now even per-token accuracy jumps, and no choice of metric smooths it away. That's the test that separates the two.</p>"),
      predict({
        chapter: id,
        question: "A capability looks emergent under exact match. You re-score with the probability the model assigns to the correct answer. That curve is smooth. What have you learned?",
        options: ["The capability is fake", "The model was improving steadily; the jump came from the all-or-nothing metric", "The capability will stop improving", "Nothing: the two metrics measure unrelated things"],
        answer: 1,
        explain: "<p>A smooth continuous metric underneath a jumpy discrete one means the model's underlying skill grew steadily, and you could have forecast the jump from smaller models. If even the continuous metric jumped, you'd have a real transition, like an induction head forming during training (Olsson et al., 2022).</p>",
      }),
    ),
    section("What's settled and what isn't",
      table(["Claim", "Status", "Notes"], [
        ["Many ‘emergent’ benchmark jumps are smooth under continuous metrics", "Published (Schaeffer et al., 2023)", "Exact match and multiple-choice accuracy are the usual culprits"],
        ["Some transitions are real, inside training", "Published (induction heads; grokking)", "Loss itself shows a bump or a plateau and drop"],
        ["Abilities appear once pretraining loss crosses a threshold, whatever the model size", "Published (Du et al., 2024)", "Loss, not parameters or FLOPs, is the better x-axis"],
        ["You can forecast a jump by measuring tiny success rates with many samples", "Published (Hu et al., 2023: PassUntil)", "Needs thousands of samples per question on small models"],
        ["Agentic abilities (long tasks) will look emergent for the same reason", "My view", "Success on a 50-step task is roughly p⁵⁰; small per-step gains become big jumps"],
      ]),
      callout("note", "My view (~70% confident)", "<p>Most of the emergence people will report in the next few years will be in <b>agents</b>, and most of it will be the same mirage at a larger scale: a long task succeeds only if every step does, so its score behaves like p raised to the number of steps. A model that goes from 95% to 98% per step goes from 8% to 36% on a 50-step task, which will feel like a new ability. The forecastable quantity is the per-step reliability. The caveat that keeps me from going higher: some steps (noticing you're stuck, choosing to backtrack) may be genuinely all-or-nothing skills, and those could produce real phase changes.</p><p><b>Tested at toy scale (notebook 49):</b> 13 networks learning 6-digit addition. Exact match went 0 → 2% → 30% → 82% across three neighbouring sizes, and it equalled per-digit accuracy to the 7th power, so the jump was mostly the ruler. But my forecast from the small half of the models predicted 42% for the largest and it scored 98%: per-digit accuracy itself sped up mid-range, probably when the networks learned to carry. Smooth metrics make emergence less mysterious, not fully forecastable.</p>"),
    ),
    section("In code",
      code({
        "Score three ways (Hugging Face)": `def three_scores(model, tok, prompt, answer):
    ids = tok(prompt + answer, return_tensors="pt").input_ids
    n = len(tok(answer).input_ids)
    with torch.no_grad(): logp = model(ids).logits[0, -n - 1:-1].log_softmax(-1)
    tgt = ids[0, -n:]
    tok_lp = logp.gather(-1, tgt[:, None]).squeeze(-1)
    greedy_right = (logp.argmax(-1) == tgt).float()
    return {"exact": greedy_right.prod().item(),        # all-or-nothing
            "partial": greedy_right.mean().item(),      # share of tokens right
            "prob": tok_lp.mean().exp().item()}         # per-token probability of the answer`,
        "Fit and forecast from small models": `# per-token log-prob of the right answer is usually linear-ish in log-compute: fit it on small models
x, y = np.log10(flops_small), np.array(per_token_logprob_small)
a, b = np.polyfit(x, y, 1)
for f in [1e22, 1e23, 1e24]:
    p_tok = np.exp(a * np.log10(f) + b)
    print(f"{f:.0e}: per-token {p_tok:.2f} → exact match on 8 tokens ≈ {p_tok ** 8:.2f}")`,
      }),
    ),
  );
  run();
}
