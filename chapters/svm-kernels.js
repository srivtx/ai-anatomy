import { h, rng } from "../lib/core.js";
import { slider, segmented, stats, predict, code, callout, section, prose, table } from "../lib/widgets.js";
import { plane } from "../lib/plot.js";
import { dataset } from "../lib/nn.js";

const KERNELS = {
  linear: () => (a, b) => 1 + a[0] * b[0] + a[1] * b[1], // the +1 plays the role of a bias
  poly: () => (a, b) => (1 + a[0] * b[0] + a[1] * b[1]) ** 2,
  rbf: (g) => (a, b) => Math.exp(-g * ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)),
};

/** Kernelised Pegasos: stochastic sub-gradient descent on the SVM objective. */
function trainSVM(pts, K, lambda, T = 4000, seed = 1) {
  const r = rng(seed), n = pts.length, y = pts.map((p) => (p.y ? 1 : -1));
  const G = pts.map((a) => pts.map((b) => K(a.x, b.x)));
  const alpha = new Float64Array(n);
  for (let t = 1; t <= T; t++) {
    const i = r.int(n);
    let s = 0;
    for (let j = 0; j < n; j++) if (alpha[j]) s += alpha[j] * y[j] * G[j][i];
    if ((y[i] * s) / (lambda * t) < 1) alpha[i]++;
  }
  const f = (x) => { let s = 0; for (let j = 0; j < n; j++) if (alpha[j]) s += alpha[j] * y[j] * K(pts[j].x, x); return s / (lambda * T); };
  return { f, sv: pts.filter((_, j) => alpha[j] > 0 && Math.abs(f(pts[j].x)) <= 1.05) };
}

export default function (root, { id }) {
  const o = { kind: "circle", kernel: "rbf", gamma: 3, C: 0 };
  let pts, model;
  const pl = plane({ width: 460, height: 420, label: "SVM decision boundary, margins and support vectors" });
  const lift = plane({ width: 320, height: 300, xr: [-1.2, 1.2], yr: [-0.1, 1.6], label: "The circle data lifted into a third dimension" });
  const st = stats([{ label: "Training accuracy" }, { label: "Support vectors" }, { label: "Kernel" }]);
  function train() {
    pts = dataset(o.kind, 120, { noise: 0.1, seed: 4 });
    model = trainSVM(pts, KERNELS[o.kernel](o.gamma), 1 / (10 ** o.C * pts.length));
    draw();
  }
  function draw() {
    pl.clear();
    pl.field((x, y) => { const v = model.f([x, y]); return Math.abs(v) < 1 ? 0.5 + v * 0.25 : v > 0 ? 0.92 : 0.08; }, { res: 64, alpha: 0.55 });
    pl.axes();
    pl.scatter(pts, { r: 3.5 });
    for (const p of model.sv) { pl.ctx.beginPath(); pl.ctx.arc(pl.X(p.x[0]), pl.Y(p.x[1]), 8, 0, 7); pl.ctx.strokeStyle = pl.colors.ink; pl.ctx.lineWidth = 1.6; pl.ctx.stroke(); }
    const acc = pts.filter((p) => (model.f(p.x) > 0) === !!p.y).length / pts.length;
    st.set(0, `${Math.round(acc * 100)}%`, acc > 0.95 ? "good" : acc < 0.75 ? "bad" : "");
    st.set(1, String(model.sv.length));
    st.set(2, { linear: "x · x′", poly: "(1 + x · x′)²", rbf: `exp(−${o.gamma}·|x − x′|²)` }[o.kernel]);
  }
  function drawLift() {
    const d = dataset("circle", 120, { noise: 0.1, seed: 4 });
    lift.clear(); lift.axes();
    lift.line(0, lift.Y(0.27), lift.W, lift.Y(0.27), { color: lift.colors.ink, lw: 2, dash: [6, 4] });
    lift.scatter(d.map((p) => ({ x: [p.x[0], p.x[0] ** 2 + p.x[1] ** 2], y: p.y })), { r: 3.5 });
    lift.text("z = x² + y²", 10, 14, { color: lift.colors["ink-2"], size: 12 });
  }

  root.append(
    section(null, prose(`<p>A perceptron stops at the first line that separates the data. A <b>support vector machine</b> (Cortes and Vapnik, 1995) asks for the best one: the line with the widest empty <b>margin</b> on both sides. Only the points touching the margin, the <b>support vectors</b>, decide where it goes. Delete any other point and nothing changes.</p>
<p>The second idea is better still. Data that no line can split in 2D often <i>can</i> be split after mapping it into more dimensions. The <b>kernel trick</b> computes dot products in that bigger space without ever building it.</p>`)),
    section("Lift the circle",
      prose("<p>Add a third coordinate z = x² + y², the squared distance from the centre. The inner class sits low and the outer ring sits high, and a flat plane (the dashed line, seen from the side) separates them. A kernel does this implicitly: K(a, b) = φ(a) · φ(b).</p>"),
      h("div.canvas-box", { style: { maxWidth: "420px" } }, lift.el)),
    section("Train an SVM",
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Kernel", options: [{ value: "linear", label: "Linear" }, { value: "poly", label: "Polynomial (degree 2)" }, { value: "rbf", label: "RBF (Gaussian)" }], value: o.kernel, onChange: (v) => { o.kernel = v; train(); } }),
          segmented({ label: "Data", options: [{ value: "circle", label: "Circle" }, { value: "moons", label: "Moons" }, { value: "linear", label: "Blobs" }, { value: "xor", label: "XOR" }], value: o.kind, onChange: (v) => { o.kind = v; train(); } }),
          slider({ label: "RBF width γ", min: 0.5, max: 30, step: 0.5, value: o.gamma, onInput: (v) => { o.gamma = v; if (o.kernel === "rbf") train(); }, hint: "Big γ: each point's influence is tiny" }),
          slider({ label: "Softness (log C)", min: -2, max: 3, step: 0.5, value: o.C, fmt: (v) => `C = ${(10 ** v).toPrecision(2)}`, onInput: (v) => { o.C = v; train(); }, hint: "Small C: wide margin, tolerate mistakes" })),
        h("div.stage-body", st.el, h("div.canvas-box", pl.el),
          h("p.canvas-cap", "The pale band is the margin (|f| < 1). Ringed points are support vectors."))),
      callout("tip", "Things to try", "<ul style='margin:0;padding-left:18px'><li>Circle with the linear kernel fails; the degree-2 polynomial kernel is exactly the lift above, and it succeeds.</li><li>RBF with γ = 30 draws a little bubble around every point: overfitting. γ = 1 is smooth.</li><li>Lower C: the margin widens and more points become support vectors, because the model is allowed to get some wrong.</li></ul>"),
      predict({
        chapter: id,
        question: "The RBF kernel corresponds to a feature space with how many dimensions?",
        options: ["2", "3", "As many as training points", "Infinitely many"],
        answer: 3,
        explain: "<p>The Gaussian kernel's Taylor expansion has terms of every degree, so its implicit feature space is infinite-dimensional, yet each K(a, b) costs one exponential. The catch: training needs the n × n matrix of all pairs, so kernel SVMs stall beyond ~100k points. Deep nets learn their features instead, and scale with data.</p>",
      }),
    ),
    section("Where kernels went",
      table(["Then", "Now"], [
        ["Hand-picked kernel (RBF, polynomial)", "Learned features from a neural network"],
        ["Max-margin hinge loss", "Cross-entropy, though margins return in contrastive and metric learning"],
        ["Kernel = similarity of two inputs", "Attention: softmax(q · k) is a kernel smoother over the context"],
        ["Neural tangent kernel", "Theory: an infinitely wide net trained by GD behaves like a kernel machine"],
      ]),
    ),
    section("In code",
      code({
        "scikit-learn": `from sklearn.svm import SVC
from sklearn.datasets import make_circles

X, y = make_circles(300, noise=0.08, factor=0.4)
for kernel in ["linear", "poly", "rbf"]:
    clf = SVC(kernel=kernel, degree=2, gamma=3, C=1.0).fit(X, y)
    print(kernel, clf.score(X, y), "support vectors:", len(clf.support_))`,
      }),
    ),
  );
  train();
  pl.paint(draw);
  lift.paint(drawLift);
}
