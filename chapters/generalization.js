import { h, rng } from "../lib/core.js";
import { slider, stats, predict, code, callout, section, prose, dilemma, table } from "../lib/widgets.js";
import { plane, xyChart } from "../lib/plot.js";
import { gauss } from "../lib/nn.js";

const truth = (x) => 0.75 * Math.sin(Math.PI * x) + 0.25 * x;

/** Solve (A + lam·I) w = b by Gaussian elimination with partial pivoting. */
function solve(A, b, lam = 0) {
  const n = b.length, M = A.map((row, i) => [...row.map((v, j) => v + (i === j ? lam : 0)), b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    const d = M[c][c] || 1e-12;
    for (let r = 0; r < n; r++) if (r !== c) { const f = M[r][c] / d; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
  }
  return M.map((row, i) => row[n] / (row[i] || 1e-12));
}
// Legendre polynomials: a well-conditioned basis on [-1, 1]
function legendre(x, deg) {
  const P = [1, x];
  for (let n = 1; n < deg; n++) P.push(((2 * n + 1) * x * P[n] - n * P[n - 1]) / (n + 1));
  return P.slice(0, deg + 1);
}
function fitPoly(pts, deg, lam) {
  const X = pts.map((p) => legendre(p.x, deg));
  const A = Array.from({ length: deg + 1 }, (_, i) => Array.from({ length: deg + 1 }, (_, j) => X.reduce((a, row) => a + row[i] * row[j], 0)));
  const b = Array.from({ length: deg + 1 }, (_, i) => X.reduce((a, row, k) => a + row[i] * pts[k].y, 0));
  const w = solve(A, b, lam * pts.length);
  return (x) => legendre(x, deg).reduce((a, v, i) => a + v * w[i], 0);
}
const mse = (f, pts) => pts.reduce((a, p) => a + (f(p.x) - p.y) ** 2, 0) / pts.length;
function sample(n, noise, seed) {
  const r = rng(seed);
  return Array.from({ length: n }, () => { const x = r() * 2 - 1; return { x, y: truth(x) + gauss(r) * noise }; });
}

// Random ReLU features with the minimum-norm fit: the setting where double descent is easiest to see.
function randomFeatures(train, test, p, seed) {
  const r = rng(seed);
  const feats = Array.from({ length: p }, () => { const t = r() * 2.2 - 1.1, sg = r() < 0.5 ? -1 : 1; return (x) => Math.max(0, sg * (x - t)); }); // a kink somewhere in the data range
  const Phi = train.map((pt) => feats.map((f) => f(pt.x)));
  const n = train.length, y = train.map((pt) => pt.y);
  let predict;
  if (p <= n) {
    const A = Array.from({ length: p }, (_, i) => Array.from({ length: p }, (_, j) => Phi.reduce((a, row) => a + row[i] * row[j], 0)));
    const w = solve(A, Array.from({ length: p }, (_, i) => Phi.reduce((a, row, k) => a + row[i] * y[k], 0)), 1e-9);
    predict = (x) => feats.reduce((a, f, i) => a + f(x) * w[i], 0);
  } else {
    // minimum-norm interpolator: w = Φᵀ (Φ Φᵀ)⁻¹ y
    const K = Phi.map((ri) => Phi.map((rj) => ri.reduce((a, v, k) => a + v * rj[k], 0)));
    const alpha = solve(K, y, 1e-9);
    predict = (x) => { const fx = feats.map((f) => f(x)); return Phi.reduce((a, row, k) => a + alpha[k] * row.reduce((s2, v, j) => s2 + v * fx[j], 0), 0); };
  }
  return Math.min(5, mse(predict, test));
}

export default function (root, { id }) {
  let deg = 3, nTrain = 15, noise = 0.2, lam = 0, seed = 1;
  let train, val;
  const pl = plane({ width: 520, height: 340, xr: [-1.05, 1.05], yr: [-1.6, 1.6], label: "Training points, the true function and the fit" });
  const st = stats([{ label: "Training error" }, { label: "Validation error" }, { label: "Gap" }]);
  const curve = xyChart({ width: 460, height: 320, xLabel: "polynomial degree", yLabel: "mean squared error", yLog: true });
  function resample() { train = sample(nTrain, noise, seed); val = sample(200, noise, seed + 1000); }
  function draw() {
    const f = fitPoly(train, deg, lam);
    pl.clear(); pl.axes();
    pl.curve(truth, { color: pl.colors.ink3 || pl.colors["ink-3"], lw: 1.5, dash: [6, 5] });
    pl.curve((x) => Math.max(-3, Math.min(3, f(x))), { color: pl.colors.accent, lw: 2.5, n: 400 });
    pl.scatter(val.slice(0, 60).map((p) => ({ x: [p.x, p.y] })), { color: pl.colors.line, r: 2.5, ring: false });
    pl.scatter(train.map((p) => ({ x: [p.x, p.y] })), { color: pl.colors.c2, r: 4.5 });
    const tr = mse(f, train), va = mse(f, val);
    st.set(0, tr.toFixed(4)); st.set(1, va.toFixed(4), va < noise * noise * 1.6 ? "good" : va > 0.3 ? "bad" : "warn"); st.set(2, `${(va / Math.max(tr, 1e-6)).toFixed(1)}×`);
    const tp = [], vp = [];
    for (let d = 0; d <= 20; d++) { const g = fitPoly(train, d, lam); tp.push([d, Math.max(1e-4, mse(g, train))]); vp.push([d, Math.min(10, mse(g, val))]); }
    curve.update([{ name: "Training", color: "var(--c2)", points: tp, dots: false }, { name: "Validation", color: "var(--c1)", points: vp }], { yDomain: [1e-4, 10], marks: [{ x: deg, label: `degree ${deg}` }] });
  }
  const redo = () => { resample(); draw(); };

  // double descent
  const dd = xyChart({ height: 240, xLabel: "number of random ReLU features (p)", yLabel: "test error (median of 8 draws)", yLog: true, xLog: true });
  function drawDD() {
    const tr = sample(20, 0.15, 77), te = sample(200, 0.15, 78), pts = [];
    for (const p of [2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 30, 40, 60, 100, 150, 250, 400]) {
      const es = []; for (let k = 0; k < 8; k++) es.push(randomFeatures(tr, te, p, 100 + k));
      es.sort((a, b) => a - b);
      pts.push([p, (es[3] + es[4]) / 2]); // median over 8 random feature draws
    }
    dd.update([{ name: "Test error", color: "var(--c1)", points: pts, dots: true, r: 3 }], { marks: [{ x: 20, label: "p = number of training points" }] });
  }

  root.append(
    section(null, prose(`<p>A model that scores 100% on its training data might have learned the pattern, or it might have memorised the answers. The only way to tell is to test it on data it has <b>never seen</b>. The gap between training and held-out performance is the central problem of machine learning, and its name is <b>generalisation</b>.</p>`)),
    section("Fit a curve, then check it on new data",
      prose("<p>Orange dots are the training set, drawn from the dashed curve plus noise. Grey dots are a validation set the fit never sees. Raise the polynomial degree and watch the fit chase the noise.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Polynomial degree", min: 0, max: 20, value: deg, onInput: (v) => { deg = v; draw(); } }),
          slider({ label: "Training points", min: 5, max: 80, value: nTrain, onInput: (v) => { nTrain = v; redo(); } }),
          slider({ label: "Noise", min: 0, max: 0.5, step: 0.05, value: noise, fmt: (v) => v.toFixed(2), onInput: (v) => { noise = v; redo(); } }),
          slider({ label: "Regularisation λ", min: -8, max: 0, step: 0.5, value: -8, fmt: (v) => (v <= -8 ? "off" : `1e${v}`), onInput: (v) => { lam = v <= -8 ? 0 : 10 ** v; draw(); } }),
          slider({ label: "Dataset seed", min: 1, max: 30, value: 1, onInput: (v) => { seed = v; redo(); } })),
        h("div.stage-body", st.el, h("div.lab", h("div.canvas-box", pl.el), curve.el))),
      callout("tip", "Things to try", "<ul style='margin:0;padding-left:18px'><li>Degree 0 and 1 <b>underfit</b>: both errors are high. Degree 3 to 5 is the sweet spot. Degree 15 <b>overfits</b>: training error near zero, validation error explodes. The validation curve is the famous U.</li><li>Keep degree 15 and raise the training points to 80. More data is the most reliable cure for overfitting.</li><li>Keep degree 15 and turn on λ around 1e-3. Regularisation penalises large weights, so the fit smooths out without changing the model's size.</li></ul>"),
      predict({
        chapter: id,
        question: "Degree 15, 15 training points: training error is almost zero and validation is terrible. Which change helps most?",
        options: ["Train for longer", "Use more training points", "Lower the noise in the validation set", "Use a higher degree"],
        answer: 1,
        explain: "<p>With as many parameters as points, the fit can pass through every point, noise included. More data pins the curve down between points. Regularisation (and, for neural nets, early stopping, dropout and weight decay) helps too, but nothing beats more data from the same distribution.</p>",
      }),
    ),
    section("Double descent: the U-curve isn't the whole story",
      prose("<p>Classical statistics says error rises forever once a model is too big. Deep learning broke that: networks with far more parameters than data points generalise well. Here's why, in a model small enough to compute in your browser: fit 20 points with p random ReLU features, picking the <b>smallest-norm</b> solution when there are many.</p>"),
      dd.el,
      prose("<p>Error peaks exactly where p equals the number of points: the model has just enough capacity to interpolate, and only one very wiggly way to do it. Past that, many interpolating solutions exist and the minimum-norm one is smooth, so error <b>falls again</b>. Gradient descent from small initial weights is biased towards such simple solutions, which is part of why huge networks work. This is <b>double descent</b> (Belkin et al. 2019; Nakkiran et al. 2019).</p>"),
      table(["Tool", "What it does", "Where you see it"], [
        ["Train / validation / test split", "Honest measurement; test is touched once at the end", "Everywhere"],
        ["Weight decay (L2)", "Pulls weights towards zero", "AdamW's wd=0.1 in LLM training"],
        ["Dropout", "Randomly zeroes activations so no unit can be relied on alone", "Older nets; rarely in big LLM pretraining"],
        ["Early stopping", "Stop when validation error stops improving", "Fine-tuning"],
        ["Data augmentation", "Flips, crops, paraphrases: more data for free", "Vision, speech"],
        ["More data", "The real fix", "Scaling laws, next level"],
      ]),
      dilemma("Your model is overfitting. Shrink it, or add data?", [
        { name: "Smaller model / more regularisation", pros: ["Cheap and immediate"], cons: ["Caps how good the model can ever be"] },
        { name: "More (or augmented) data", pros: ["Raises the ceiling", "Lets you grow the model too"], cons: ["Collecting and cleaning data is slow and costly"] },
      ], "Modern practice: make the model as big as compute allows, then get enough data for it. Regularise lightly, and watch the validation curve."),
    ),
    section("In code",
      code({
        "scikit-learn": `from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.3, random_state=0)
for degree in range(0, 16):
    model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=1e-3))
    model.fit(X_tr, y_tr)
    print(degree, model.score(X_tr, y_tr), model.score(X_val, y_val))`,
      }),
    ),
  );
  resample();
  pl.paint(draw);
  drawDD();
}
