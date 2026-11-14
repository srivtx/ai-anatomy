import { h, rng, loop } from "../lib/core.js";
import { slider, stats, predict, callout, section, prose, table, button, bars, code } from "../lib/widgets.js";
import { plane } from "../lib/plot.js";
import { makeData, ARCH, trainer, acc, suspicion } from "../lib/datafirst.js";

const RUNS = [
  { key: "small", short: "Small, noisy labels", label: "Small model, noisy labels", arch: ARCH.small, clean: false, chunk: 6 },
  { key: "big", short: "6× bigger, noisy labels", label: "6× bigger model, noisy labels", arch: ARCH.big, clean: false, chunk: 2 },
  { key: "clean", short: "Small, labels fixed", label: "Small model, labels fixed", arch: ARCH.small, clean: true, chunk: 6 },
];

export default function (root, { id }) {
  let noise = 0.3, data, cur = 0, ts = [], results = {};
  const planes = RUNS.map((rn) => plane({ width: 300, height: 300, label: rn.label }));
  const caps = RUNS.map(() => h("p.canvas-cap"));
  const st = stats([{ label: "Mislabelled training points" }, { label: "Now training" }]);
  const barsEl = h("div");
  function reset() {
    lp.stop(); data = makeData(noise); cur = 0; results = {};
    ts = RUNS.map((rn) => trainer(rn.clean ? data.train.map((p) => ({ ...p, y: p.clean })) : data.train, rn.arch, { epochs: 250 }));
    sus = null; drawAll(); drawReview();
  }
  function drawPlane(i) {
    const pl = planes[i], m = ts[i].model, started = ts[i].epoch > 0, train = RUNS[i].clean ? data.train.map((p) => ({ ...p, y: p.clean })) : data.train;
    pl.paint(() => {
      pl.clear();
      if (started) pl.field((x, y) => 1 / (1 + Math.exp(-m.forward([x, y])[0])), { res: 50 });
      pl.scatter(train, { r: 2.6 });
    });
    caps[i].textContent = started ? `${RUNS[i].label}: ${ts[i].done ? "done" : `epoch ${ts[i].epoch}`}${results[RUNS[i].key] != null ? ` · test accuracy ${(results[RUNS[i].key] * 100).toFixed(1)}%` : ""}` : `${RUNS[i].label}: waiting`;
  }
  function drawAll() {
    RUNS.forEach((_, i) => drawPlane(i));
    const nNoisy = data.train.filter((p) => p.y !== p.clean).length;
    st.set(0, `${nNoisy} of ${data.train.length}`, nNoisy ? "warn" : "good"); st.set(1, cur < RUNS.length ? RUNS[cur].label : "All done");
    barsEl.replaceChildren(h("div.eyebrow", "Accuracy on a clean test set"), bars(RUNS.map((rn) => ({ label: rn.short, value: (results[rn.key] ?? 0) * 100, color: rn.clean ? "var(--good)" : rn.key === "big" ? "var(--c2)" : "var(--accent)" })), { max: 100, fmt: (v) => (v ? `${v.toFixed(1)}%` : "–") }));
  }
  const lp = loop(() => {
    if (cur >= RUNS.length) { lp.stop(); return; }
    const t = ts[cur]; t.run(RUNS[cur].chunk);
    if (t.done) { results[RUNS[cur].key] = acc(t.model, data.test); cur++; }
    drawAll();
  }, planes[0].el);

  // ---------- find the bad labels ----------
  let sus = null, budget = 80, after = null;
  const rp = plane({ width: 380, height: 380, label: "Training data, most suspicious labels ringed" });
  const rst = stats([{ label: "Reviewed" }, { label: "Errors found, reviewing the most suspicious first" }, { label: "Errors found, reviewing at random" }, { label: "Test accuracy after fixing" }]);
  function drawReview() {
    const ready = !!sus;
    rp.paint(() => {
      rp.clear();
      rp.scatter(data.train, { r: 2.6 });
      if (ready) for (const { p } of sus.slice(0, budget)) { const c = rp.ctx; c.strokeStyle = p.y !== p.clean ? rp.colors.bad : rp.colors.ink; c.lineWidth = 1.6; c.beginPath(); c.arc(rp.X(p.x[0]), rp.Y(p.x[1]), 6, 0, Math.PI * 2); c.stroke(); }
    });
    if (!ready) { ["–", "–", "–", "–"].forEach((v, i) => rst.set(i, v)); return; }
    const top = sus.slice(0, budget).map((x) => x.p), hit = top.filter((p) => p.y !== p.clean).length;
    const nNoisy = data.train.filter((p) => p.y !== p.clean).length, randHit = Math.round((budget * nNoisy) / data.train.length);
    rst.set(0, `${budget} labels`); rst.set(1, `${hit} (${Math.round((hit / budget) * 100)}% of reviews)`, "good"); rst.set(2, `≈ ${randHit} (${Math.round((nNoisy / data.train.length) * 100)}%)`);
    rst.set(3, after != null ? `${(after * 100).toFixed(1)}%` : "press Retrain", after != null ? "good" : "");
  }
  function findSuspicious() { sus = suspicion(data.train, ARCH.small); after = null; drawReview(); }
  function retrain() {
    if (!sus) findSuspicious();
    const S = new Set(sus.slice(0, budget).map((x) => x.p)), fixed = data.train.map((p) => (S.has(p) ? { ...p, y: p.clean } : p));
    const t = trainer(fixed, ARCH.small, { epochs: 250 }); t.run(250); after = acc(t.model, data.test); drawReview();
  }

  root.append(
    section(null, prose(`<p>A concept can be beautiful and still lose to messy reality. The most common reality in applied ML is that the data is worse than you think: wrong labels, duplicates, leakage, a training set that doesn't look like what users send. Practitioners repeat the same lesson for a reason: <b>when a model underperforms, look at the data before you touch the architecture.</b> This chapter makes that concrete with a real experiment that trains in your browser.</p>`)),
    section("Same budget, two ways to spend it",
      prose("<p>A two-spiral classification problem. A fraction of the training labels are flipped, as happens when annotators are tired or rules are ambiguous. You can spend effort on a <b>bigger model</b> (six times the parameters) or on <b>fixing the labels</b>. Three runs, trained one after another, scored on a clean test set:</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Label noise", min: 0, max: 0.45, step: 0.05, value: noise, fmt: (v) => `${Math.round(v * 100)}% of labels wrong`, onInput: (v) => { noise = v; reset(); } }),
          button("Run the experiment", () => { if (cur >= RUNS.length) reset(); lp.start(); }, "primary"), button("Reset", reset)),
        h("div.stage-body", st.el,
          h("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(220px,1fr))", gap: "14px" } }, planes.map((pl, i) => h("div", pl.el, caps[i]))),
          barsEl)),
      callout("tip", "What you should see", "<ul style='margin:0;padding-left:18px'><li>At 30% noise the small model lands somewhere around 75–82% and the six-times-bigger model within a few points of it (sometimes better, sometimes worse). The same small model with fixed labels gets about 98%. Fixing the data beats six times the parameters by a mile.</li><li>The bigger model isn't “smarter” about noise. With enough capacity, it starts fitting the wrong labels too: look for little islands of the wrong colour.</li><li>Set noise to 0: now all three are close. Model size matters when the data is good and the problem needs the capacity. Order your work accordingly.</li></ul>"),
    ),
    section("Find the bad labels without knowing the truth",
      prose("<p>In real projects nobody tells you which labels are wrong, and reviewing everything is expensive. A practical trick (the idea behind <b>confident learning</b> and tools like cleanlab): train on half the data, score the other half, and swap. Labels the model finds very unlikely, without having trained on them, are the most suspicious. Review those first. Red rings are labels that really were wrong.</p>"),
      h("div.stage",
        h("div.stage-controls",
          button("Rank labels by suspicion", findSuspicious, "primary"),
          slider({ label: "Review budget", min: 10, max: 200, step: 10, value: budget, fmt: (v) => `${v} labels`, onInput: (v) => { budget = v; after = null; drawReview(); } }),
          button("Fix the reviewed labels and retrain", retrain)),
        h("div.stage-body", rst.el, h("div.canvas-box", rp.el))),
      callout("note", "Why this works", "<p>A model trained on mostly-correct data learns the real pattern; points that contradict that pattern stand out. Cross-fitting matters: a model scoring its <i>own</i> training points has memorised them and can't see their errors. Reviewing by suspicion typically finds two to three times as many errors per hour of human time as random review. The same idea scales to LLM data: use a model to flag low-quality or mislabelled examples, then have people check the flagged ones.</p>"),
      predict({
        chapter: id,
        question: "Your fine-tuned classifier is stuck at 85% and adding layers doesn't help. What's the most likely first win?",
        options: ["A transformer instead of an MLP", "Look at the errors: sample 50 misclassified examples and read them. Often many are label errors, ambiguous cases or a data pipeline bug", "Train for 10× longer", "Lower the learning rate"],
        answer: 1,
        explain: "<p>Reading errors is the highest-leverage hour in applied ML. You'll usually find label noise, duplicated or leaked examples, a preprocessing mismatch, or a category definition that annotators disagree on. Each of those has a fix that no architecture change can provide.</p>",
      }),
    ),
    section("The data problems that bite in practice",
      table(["Problem", "Symptom", "What actually fixes it"], [
        ["Label noise", "Training accuracy plateaus below 100%; errors look “reasonable”", "Clearer labelling guidelines, multiple annotators, review by suspicion"],
        ["Duplicates across train and test", "Test score suspiciously close to training score; drops in production", "Deduplicate before splitting (exact and near-duplicate)"],
        ["Leakage", "Amazing validation scores that collapse on new data", "Split by time, user or source; remove features computed after the outcome"],
        ["Train/serve skew", "Offline metrics great, live metrics poor", "Use the same preprocessing code for training and serving; log real inputs"],
        ["Distribution shift", "Quality decays over weeks", "Monitor input statistics; refresh data; evaluate on recent samples"],
        ["Class imbalance", "High accuracy but the rare class is never predicted", "Report per-class recall, reweight or resample, pick thresholds per class"],
        ["Low-quality web text (LLMs)", "Model learns boilerplate, spam, broken formatting", "Heuristic filters + quality classifiers + dedup (see the data chapter)"],
      ]),
    ),
    section("In code",
      code({
        "Find suspicious labels": `import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import HistGradientBoostingClassifier

# out-of-fold probabilities: every example is scored by a model that never trained on it
proba = cross_val_predict(HistGradientBoostingClassifier(), X, y, cv=5, method="predict_proba")
p_given_label = proba[np.arange(len(y)), y]          # how plausible is the label we have?
suspects = np.argsort(p_given_label)                 # most suspicious first
review_queue = suspects[:200]                        # send these to a human
# the cleanlab library does a more careful version (per-class thresholds, joint noise estimate)`,
        "Data vs model experiment": `for noise in [0.0, 0.1, 0.3]:
    y_noisy = flip_labels(y_train, noise)
    for name, model in [("small", make_mlp(24)), ("big", make_mlp(96))]:
        for labels, tag in [(y_noisy, "noisy"), (y_train, "clean")]:
            acc = fit(model(), X_train, labels).score(X_test, y_test)
            print(f"{noise:.0%} {name:5s} {tag:5s} {acc:.3f}")`,
      }, "Project 33 runs this grid on real data (CIFAR-10 features) with several seeds."),
    ),
  );
  reset();
}
