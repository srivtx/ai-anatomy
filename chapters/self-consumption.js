import { h } from "../lib/core.js";
import { slider, segmented, stats, predict, code, callout, section, prose, button, table } from "../lib/widgets.js";
import { xyChart } from "../lib/plot.js";
import { selfConsume } from "../lib/hidden.js";

export default function (root, { id }) {
  let mode = "replace", fresh = 0.2, temp = 1, gen = 0, res = null;
  const st = stats([{ label: "Generation" }, { label: "Words still alive (of 400)" }, { label: "Rare words alive (ranks 201–400)" }, { label: "KL(real ‖ model)" }]);
  const rank = xyChart({ width: 480, height: 300, xLog: true, yLog: true, xLabel: "word rank", yLabel: "probability", xDomain: [1, 400], yDomain: [1e-5, 0.3], fmtY: (v) => (v >= 0.01 ? v.toFixed(2) : v.toExponential(0)), label: "Rank-frequency of real language and of the model" });
  const time = xyChart({ width: 480, height: 300, xLabel: "generation", yLabel: "words alive", xDomain: [0, 30], yDomain: [0, 400], label: "Vocabulary surviving over generations" });

  function run() { res = selfConsume({ G: 30, mode, fresh, temp }); show(); }
  function show() {
    const s = res.hist[gen];
    st.set(0, String(gen));
    st.set(1, String(s.alive), s.alive < 150 ? "bad" : s.alive < 250 ? "warn" : "good");
    st.set(3, s.kl.toFixed(2), s.kl > 1 ? "bad" : s.kl > 0.5 ? "warn" : "good");
    // model at generation `gen` isn't stored; rerun quickly to that generation
    const m = gen === 30 ? res.model : selfConsume({ G: gen, mode, fresh, temp }).model;
    const rareAlive = m.slice(200).filter((p) => p > 0).length;
    st.set(2, `${rareAlive} of 200`, rareAlive < 50 ? "bad" : rareAlive < 120 ? "warn" : "good");
    const sorted = res.truth.map((p, i) => [i + 1, p]);
    rank.update([
      { name: "Real language (Zipf)", color: "var(--c1)", points: sorted },
      { name: `Model, generation ${gen}`, color: "var(--c2)", points: m.map((p, i) => [i + 1, p]).filter((x) => x[1] > 0), dots: true, r: 3.5, line: false },
    ]);
    time.update([{ name: "words alive", color: "var(--c2)", points: res.hist.slice(0, gen + 1).map((x) => [x.g, x.alive]) }]);
  }
  const gSlider = slider({ label: "Generation", min: 0, max: 30, value: 0, onInput: (v) => { gen = v; show(); } });

  root.append(
    section(null, prose(`<p>The web is filling with model-written text, and the next models train on the web. What happens when models learn from models? Shumailov et al. (Nature, 2024) called the result <b>model collapse</b>: each generation forgets a little of the tail of the distribution, the rare words, unusual facts and minority styles, and the loss compounds.</p>
<p>The cleanest version fits on a page: a “language” of 400 words with realistic Zipf frequencies. Each generation's model is fitted to 2,000 words of training data, then writes the next generation's training data. A word the model never sampled gets probability zero and can never come back.</p>`)),
    section("Generations of models, live",
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Training data each generation", options: [{ value: "replace", label: "Only the last model's output" }, { value: "accumulate", label: "Keep all data, add synthetic" }, { value: "mix", label: "Synthetic + some fresh real" }], value: mode, onChange: (v) => { mode = v; run(); } }),
          slider({ label: "Fresh real data (mix mode)", min: 0.05, max: 0.5, step: 0.05, value: fresh, fmt: (v) => `${Math.round(v * 100)}%`, onInput: (v) => { fresh = v; run(); } }),
          slider({ label: "Sampling temperature", min: 0.9, max: 1.1, step: 0.02, value: temp, fmt: (v) => v.toFixed(2), onInput: (v) => { temp = v; run(); } }),
          gSlider,
          button("Play 30 generations", () => { let g = 0; const t = setInterval(() => { gen = g; gSlider.value = g; show(); if (++g > 30) clearInterval(t); }, 120); }, "primary"),
        ),
        h("div.stage-body", st.el, h("div.split", rank.el, time.el)),
      ),
      callout("tip", "Things to try", "<p>Play 30 generations with “only the last model's output”: the orange dots peel away from the right end of the curve (the rare words); by generation 30 only about 60 of the 400 words survive. Switch to “keep all data”: collapse stops, because the original real data never leaves (Gerstgrasser et al., 2024). Then nudge temperature below 1: sampling slightly greedily, as deployed models do, speeds the collapse dramatically.</p>"),
      predict({
        chapter: id,
        question: "A lab trains each new model on a mix of web data, which is increasingly model-written, plus all its older data. Does it collapse?",
        options: ["Yes, inevitably", "Not if the real data stays in the mix", "Only if the models are small", "Only at temperature above 1"],
        answer: 1,
        explain: "<p>Collapse needs the real data to be <i>replaced</i>. If you accumulate, the synthetic share grows but the original distribution stays anchored, and the error stays bounded. The practical danger is subtler: losing provenance (not knowing which data was real), and sampling settings (low temperature, top-p) that cut the tail before it ever reaches the next model.</p>",
      }),
    ),
    section("Synthetic data that works, and why",
      prose(`<p>Collapse sounds like an argument against synthetic data, yet frontier labs now train heavily on it. The difference is <b>information</b>. Sampling from a model and training on the result adds nothing the model didn't already have, and finite samples lose the tail. Synthetic data helps when something outside the model adds information: a verifier, a filter, a stronger teacher, or the environment.</p>`),
      table(["Kind of synthetic data", "Where the new information comes from", "Collapse risk"], [
        ["Self-generated text, trained on as-is", "Nowhere", "High: this is the setup above"],
        ["Rejection sampling with a verifier (math, code)", "The checker, which keeps only correct answers", "Low for correctness, but diversity still narrows (the RL chapter)"],
        ["Distillation from a stronger model", "The teacher's extra knowledge", "Low until students become teachers in a loop"],
        ["Rephrasing real documents (WRAP, Nemotron-CC)", "The real documents; the model only changes the form", "Low: the facts are anchored"],
        ["Environment rollouts (agents, games, simulators)", "The environment's responses", "Low: the world pushes back"],
        ["Persona-driven or seed-driven generation", "Diversity of the seeds", "Medium: diversity depends on the seed pool"],
      ]),
      callout("note", "My intuition (~70% confident)", "<p>The quantity that matters is <i>bits of external verification per synthetic token</i>. When it's near zero, every generation loses tail mass at a rate set by the sample size; when it's large, synthetic data behaves like real data. A testable consequence: for verifier-filtered synthetic math, adding more unfiltered self-samples hurts held-out diversity long before it hurts accuracy, so measure distinct-solution counts, not just accuracy. Notebook 43 lets you watch both.</p>"),
    ),
    section("In code",
      code({
        "Recursive training (any LM)": `def generation_loop(train, sample, real_data, gens=10, n=100_000, mode="accumulate", fresh=0.1):
    data = list(real_data[:n])
    model = train(data)
    for g in range(gens):
        synth = sample(model, n, temperature=1.0)       # temperature < 1 accelerates collapse
        if mode == "replace":
            data = synth
        elif mode == "accumulate":
            data = data + synth
        else:  # mix in fresh real data every generation
            data = synth[: int(n * (1 - fresh))] + random.sample(real_data, int(n * fresh))
        model = train(data)
        yield g, model`,
        "Measure the tail, not just loss": `from collections import Counter

def tail_report(samples, real, top=1000):
    real_freq = Counter(w for doc in real for w in doc.split())
    rare = {w for w, _ in real_freq.most_common()[top:]}          # words beyond the top 1000
    gen = Counter(w for doc in samples for w in doc.split())
    return {
        "distinct words": len(gen),
        "rare-word share": sum(c for w, c in gen.items() if w in rare) / sum(gen.values()),
        "distinct 4-grams": len({tuple(d.split()[i:i + 4]) for d in samples for i in range(len(d.split()) - 3)}),
    }`,
      }),
    ),
  );
  run();
}
