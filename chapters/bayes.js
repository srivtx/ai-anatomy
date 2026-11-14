import { h, rng } from "../lib/core.js";
import { slider, stats, predict, code, callout, section, prose, table } from "../lib/widgets.js";
import { xyChart } from "../lib/plot.js";
import { gauss, sigmoid } from "../lib/nn.js";

const SPAM = ["win a free prize now", "free money click now", "claim your free gift card", "you won cash prize click", "cheap pills limited offer", "urgent claim your reward now", "free entry win big money", "act now limited time offer free", "congratulations you won a free trip", "click here for cheap loans"];
const HAM = ["are we still meeting for lunch", "can you review my pull request", "the meeting moved to tuesday", "thanks for the notes from class", "lunch now or after the call", "here is the report you asked for", "did you push the fix to main", "call me when you get home", "the train is late again", "can you send the slides before the meeting"];
const words = (s) => s.toLowerCase().match(/[a-z']+/g) || [];
function trainNB() {
  const count = { spam: new Map(), ham: new Map() }, tot = { spam: 0, ham: 0 };
  for (const [cls, docs] of [["spam", SPAM], ["ham", HAM]]) for (const d of docs) for (const w of words(d)) { count[cls].set(w, (count[cls].get(w) || 0) + 1); tot[cls]++; }
  const vocab = new Set([...count.spam.keys(), ...count.ham.keys()]);
  // log P(word | class) with add-one (Laplace) smoothing
  const lp = (cls, w) => Math.log(((count[cls].get(w) || 0) + 1) / (tot[cls] + vocab.size));
  return { lp, vocab };
}

export default function (root, { id }) {
  // 1. base rates
  let prev = 1, sens = 90, fpr = 9;
  const grid = h("div", { style: { display: "grid", gridTemplateColumns: "repeat(40, 1fr)", gap: "2px", maxWidth: "620px" }, role: "img", "aria-label": "1,000 people: who has the condition and who tests positive" });
  const st = stats([{ label: "People who test positive" }, { label: "…of whom actually sick" }, { label: "P(sick | positive)" }]);
  function drawGrid() {
    const sick = Math.round(prev * 10), tp = Math.round((sick * sens) / 100), fp = Math.round(((1000 - sick) * fpr) / 100);
    const cells = [];
    for (let i = 0; i < 1000; i++) {
      const isSick = i < sick, pos = isSick ? i < tp : i - sick < fp;
      const bg = isSick ? (pos ? "var(--bad)" : "color-mix(in srgb, var(--bad) 35%, var(--surface))") : pos ? "var(--warn)" : "var(--sunken)";
      cells.push(h("span", { style: { aspectRatio: "1", borderRadius: "2px", background: bg } }));
    }
    grid.replaceChildren(...cells);
    st.set(0, String(tp + fp)); st.set(1, String(tp));
    const post = tp / Math.max(1, tp + fp);
    st.set(2, `${(post * 100).toFixed(1)}%`, post < 0.3 ? "bad" : post > 0.8 ? "good" : "warn");
  }

  // 2. naive Bayes
  const nb = trainNB();
  const input = h("input.ta", { value: "claim your free lunch now", "aria-label": "Message to classify" });
  const out = h("div"), nst = stats([{ label: "P(spam | message)" }, { label: "Verdict" }]);
  function classify() {
    const ws = words(input.value);
    let lo = Math.log(0.5) - Math.log(0.5); // equal priors
    const rows = ws.map((w) => { const l = nb.lp("spam", w) - nb.lp("ham", w); lo += l; return [w, nb.vocab.has(w) ? l.toFixed(2) : "unseen (≈0)", l > 0.3 ? h("span.pill.bad", "spammy") : l < -0.3 ? h("span.pill.good", "hammy") : h("span.pill", "neutral")]; });
    const p = sigmoid(lo);
    nst.set(0, `${(p * 100).toFixed(1)}%`); nst.set(1, p > 0.5 ? "spam" : "not spam", p > 0.5 ? "bad" : "good");
    out.replaceChildren(table(["Word", "log P(w|spam) − log P(w|ham)", ""], rows));
  }
  input.addEventListener("input", classify);

  // 3. calibration
  let over = 2.5, T = 1;
  const rel = xyChart({ height: 260, xLabel: "confidence the model reports", yLabel: "how often it's actually right", xDomain: [0, 1], yDomain: [0, 1] });
  const cst = stats([{ label: "Expected calibration error" }, { label: "Average confidence" }, { label: "Accuracy" }]);
  function drawCal() {
    const r = rng(4), bins = Array.from({ length: 10 }, () => ({ n: 0, ok: 0, conf: 0 }));
    let accSum = 0, confSum = 0;
    for (let i = 0; i < 4000; i++) {
      const z = gauss(r) * 1.2 + 0.8;          // the true log-odds of being right
      const right = r() < sigmoid(z);
      const conf = sigmoid((z * over) / T);    // an overconfident model exaggerates them
      const c = Math.max(conf, 1 - conf), ok = conf >= 0.5 ? right : !right;
      const b = Math.min(9, Math.floor(c * 10));
      bins[b].n++; bins[b].ok += ok; bins[b].conf += c; accSum += ok; confSum += c;
    }
    const pts = bins.filter((b) => b.n > 20).map((b) => [b.conf / b.n, b.ok / b.n]);
    const ece = bins.reduce((a, b) => a + (b.n ? (b.n / 4000) * Math.abs(b.ok / b.n - b.conf / b.n) : 0), 0);
    rel.update([{ name: "Perfect calibration", color: "var(--ink-3)", points: [[0, 0], [1, 1]], dash: true }, { name: "This model", color: "var(--c1)", points: pts, dots: true }]);
    cst.set(0, `${(ece * 100).toFixed(1)}%`, ece < 0.02 ? "good" : ece > 0.08 ? "bad" : "warn");
    cst.set(1, `${((confSum / 4000) * 100).toFixed(1)}%`); cst.set(2, `${((accSum / 4000) * 100).toFixed(1)}%`);
  }

  root.append(
    section(null, prose(`<p>Bayes' rule is how to update a belief with evidence: <code>P(H | E) = P(E | H) · P(H) / P(E)</code>. The <b>prior</b> P(H) is what you believed before; the <b>likelihood</b> P(E | H) is how well the hypothesis predicts what you saw; the <b>posterior</b> is what you should believe now. It sounds obvious, and people (and models) get it wrong all the time.</p>`)),
    section("The base-rate trap",
      prose("<p>A test for a rare condition is 90% sensitive (it catches 90% of sick people) and has a 9% false-positive rate. You test positive. How worried should you be? Each square below is one of 1,000 people.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "How common the condition is", min: 0.1, max: 30, step: 0.1, value: prev, fmt: (v) => `${v.toFixed(1)}%`, onInput: (v) => { prev = v; drawGrid(); } }),
          slider({ label: "Sensitivity", min: 50, max: 100, value: sens, fmt: (v) => `${v}%`, onInput: (v) => { sens = v; drawGrid(); } }),
          slider({ label: "False-positive rate", min: 0, max: 30, step: 0.5, value: fpr, fmt: (v) => `${v}%`, onInput: (v) => { fpr = v; drawGrid(); } })),
        h("div.stage-body", st.el, grid,
          h("div.legend", h("span", h("i", { style: { background: "var(--bad)" } }), "sick, tests positive"), h("span", h("i", { style: { background: "color-mix(in srgb, var(--bad) 35%, var(--surface))" } }), "sick, missed"), h("span", h("i", { style: { background: "var(--warn)" } }), "healthy, false alarm"), h("span", h("i", { style: { background: "var(--sunken)" } }), "healthy, negative")))),
      predict({
        chapter: id,
        question: "With 1% prevalence, 90% sensitivity and a 9% false-positive rate, what's the chance you're sick after a positive test?",
        options: ["About 90%", "About 50%", "About 9%", "About 1%"],
        answer: 2,
        explain: "<p>Of 1,000 people, 10 are sick and 9 of them test positive. Of the 990 healthy, about 89 test positive. So 9 of 98 positives are real: about 9%. The rare prior dominates. The same trap hits fraud detectors, content moderation and any classifier hunting for something rare: <b>precision</b> collapses when positives are scarce.</p>",
      }),
    ),
    section("Naive Bayes: a spam filter in ten lines",
      prose("<p>Assume (naively) that words are independent given the class. Then the log-odds of spam is just a sum: one number per word, learned by counting. This was the first spam filter that really worked (Paul Graham's <i>A Plan for Spam</i>, 2002). It's trained here on 20 tiny messages; type your own.</p>"),
      h("div.stage", h("div.stage-body", input, nst.el, out)),
      callout("tip", "Things to try", "<p>Type “free lunch meeting”: <i>free</i> pulls towards spam, <i>lunch</i> and <i>meeting</i> pull back. The verdict is a tug of war of word scores, each one interpretable. Words it has never seen get the same smoothed probability in both classes, so they count for nothing.</p>"),
    ),
    section("Calibration: does 90% mean 90%?",
      prose("<p>A model is <b>calibrated</b> if, among all the times it says 90%, it's right 90% of the time. Modern deep nets are usually <b>overconfident</b>. The fix is often one number: <b>temperature scaling</b> divides the logits by T > 1, fitted on validation data (Guo et al., 2017). The same T is the temperature knob on an LLM.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Model overconfidence", min: 0.5, max: 4, step: 0.1, value: over, fmt: (v) => `${v.toFixed(1)}×`, onInput: (v) => { over = v; drawCal(); } }),
          slider({ label: "Temperature T", min: 0.5, max: 4, step: 0.1, value: T, fmt: (v) => v.toFixed(1), onInput: (v) => { T = v; drawCal(); }, hint: "Set it equal to the overconfidence" })),
        h("div.stage-body", cst.el, rel.el)),
      table(["Idea", "Where it lives in modern ML"], [
        ["Prior", "Weight decay, pretraining as a prior for fine-tuning, the reference model in RLHF"],
        ["Posterior uncertainty", "Ensembles, MC dropout, Bayesian neural nets (rarely at LLM scale)"],
        ["Calibration", "Temperature scaling; LLMs' token probabilities are well calibrated before RLHF, less after"],
        ["In-context learning", "One theory: transformers do implicit Bayesian inference over tasks from the prompt"],
      ]),
    ),
    section("In code",
      code({
        "scikit-learn": `from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import calibration_curve

vec = CountVectorizer()
X = vec.fit_transform(messages)
nb = MultinomialNB(alpha=1.0).fit(X, labels)        # alpha=1: Laplace smoothing
print(nb.predict_proba(vec.transform(["claim your free lunch now"])))

prob_true, prob_pred = calibration_curve(y_val, model.predict_proba(X_val)[:, 1], n_bins=10)`,
        "Temperature scaling": `import torch
T = torch.ones(1, requires_grad=True)
opt = torch.optim.LBFGS([T], lr=0.1)
def closure():
    opt.zero_grad()
    loss = torch.nn.functional.cross_entropy(val_logits / T, val_labels)
    loss.backward()
    return loss
opt.step(closure)        # then use logits / T at test time`,
      }),
    ),
  );
  drawGrid(); classify(); drawCal();
}
