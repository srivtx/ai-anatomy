import { h, loop } from "../lib/core.js";
import { segmented, slider, stats, predict, code, callout, section, prose, button, table } from "../lib/widgets.js";
import { plane, xyChart } from "../lib/plot.js";
import { dataset, sigmoid } from "../lib/nn.js";

export default function (root, { id }) {
  let pts = dataset("linear", 40, { seed: 4 }), algo = "perceptron", addClass = 1, lr = 0.2;
  let w = [0.2, -1], b = 0.1, i = 0, epoch = 0, mistakes = 0, lastMistakes = null, cur = -1;
  const pl = plane({ width: 520, height: 420, label: "Points and the learned boundary", onPoint: (x, y) => { pts.push({ x: [x, y], y: addClass }); draw(); } });
  const st = stats([{ label: "Epoch" }, { label: "Mistakes last epoch" }, { label: "Accuracy" }, { label: "Cross-entropy loss" }]);
  const msg = h("p.canvas-cap");

  const score = (p) => w[0] * p.x[0] + w[1] * p.x[1] + b;
  function stepOne() {
    if (!pts.length) return;
    const p = pts[i % pts.length];
    cur = i % pts.length;
    if (algo === "perceptron") {
      const yy = p.y ? 1 : -1;
      if (yy * score(p) <= 0) { w[0] += lr * yy * p.x[0]; w[1] += lr * yy * p.x[1]; b += lr * yy; mistakes++; }
    } else {
      const e = sigmoid(score(p)) - p.y; // gradient of cross-entropy w.r.t. the score
      w[0] -= lr * e * p.x[0]; w[1] -= lr * e * p.x[1]; b -= lr * e;
      if ((score(p) > 0) !== !!p.y) mistakes++;
    }
    i++;
    if (i % pts.length === 0) { epoch++; lastMistakes = mistakes; mistakes = 0; }
  }
  function metrics() {
    let ok = 0, ce = 0;
    for (const p of pts) { const s = score(p); if ((s > 0) === !!p.y) ok++; const q = Math.min(1 - 1e-9, Math.max(1e-9, sigmoid(s))); ce -= p.y ? Math.log(q) : Math.log(1 - q); }
    return { acc: pts.length ? ok / pts.length : 0, ce: pts.length ? ce / pts.length : 0 };
  }
  function draw() {
    pl.clear();
    if (algo === "logistic") pl.field((x, y) => sigmoid(w[0] * x + w[1] * y + b), { res: 60, alpha: 0.5 });
    else pl.field((x, y) => (w[0] * x + w[1] * y + b > 0 ? 0.85 : 0.15), { res: 120, alpha: 0.35 });
    pl.axes();
    // the boundary w·x + b = 0 and the weight vector
    if (Math.abs(w[1]) > 1e-6) pl.curve((x) => -(w[0] * x + b) / w[1], { color: pl.colors.ink, lw: 2.5 });
    else if (Math.abs(w[0]) > 1e-6) pl.line(pl.X(-b / w[0]), 0, pl.X(-b / w[0]), pl.H, { color: pl.colors.ink, lw: 2.5 });
    const n = Math.hypot(...w) || 1;
    pl.line(pl.X(0), pl.Y(0), pl.X((w[0] / n) * 0.4), pl.Y((w[1] / n) * 0.4), { color: pl.colors.accent, lw: 3 });
    pl.text("w", pl.X((w[0] / n) * 0.47), pl.Y((w[1] / n) * 0.47), { color: pl.colors.accent, size: 14, weight: 700, align: "center" });
    pl.scatter(pts, { r: 4.5 });
    if (cur >= 0 && pts[cur]) { const p = pts[cur]; pl.ctx.beginPath(); pl.ctx.arc(pl.X(p.x[0]), pl.Y(p.x[1]), 9, 0, 7); pl.ctx.strokeStyle = pl.colors.ink; pl.ctx.lineWidth = 2; pl.ctx.stroke(); }
    const m = metrics();
    st.set(0, String(epoch));
    st.set(1, lastMistakes == null ? "–" : String(lastMistakes), lastMistakes === 0 ? "good" : "");
    st.set(2, `${Math.round(m.acc * 100)}%`, m.acc === 1 ? "good" : "");
    st.set(3, m.ce.toFixed(3));
    msg.textContent = algo === "perceptron" && lastMistakes === 0 ? "Converged: a full pass with zero mistakes. The perceptron stops learning here." : `Weights: w = (${w[0].toFixed(2)}, ${w[1].toFixed(2)}), b = ${b.toFixed(2)}`;
  }
  let acc = 0;
  const lp = loop((dt) => {
    acc += dt;
    while (acc > 0.025) { acc -= 0.025; stepOne(); }
    draw();
    if ((algo === "perceptron" && lastMistakes === 0) || epoch > 300) lp.stop();
  }, pl.el);
  const reset = () => { w = [0.2, -1]; b = 0.1; i = 0; epoch = 0; mistakes = 0; lastMistakes = null; cur = -1; draw(); };

  // loss comparison
  const lossChart = xyChart({ height: 220, xLabel: "probability given to the right answer", yLabel: "loss", xDomain: [0.01, 1], yDomain: [0, 4.6] });
  const P = Array.from({ length: 100 }, (_, k) => 0.01 + (k / 99) * 0.99);
  lossChart.update([
    { name: "Cross-entropy: −log p", color: "var(--c1)", points: P.map((p) => [p, -Math.log(p)]) },
    { name: "Squared error: (1 − p)²", color: "var(--c2)", points: P.map((p) => [p, (1 - p) ** 2]), dash: true },
  ]);

  root.append(
    section(null, prose(`<p>In 1958 Frank Rosenblatt built the <b>Mark I Perceptron</b>, a room-sized machine that learned to tell shapes apart. Its unit of computation was a cartoon of a neuron: multiply each input by a weight, add them up, and fire if the sum crosses a threshold.</p>
<p><code>output = 1 if w₁x₁ + w₂x₂ + b > 0 else 0</code></p>
<p>Geometrically that's a <b>line</b> (a plane, in more dimensions). The weight vector <b>w</b> points at the side that fires. Learning means moving the line until the classes fall on the right sides.</p>`)),
    section("Train a perceptron",
      prose("<p>The <b>perceptron rule</b> is almost too simple: look at one point; if it's on the wrong side, nudge w towards it (or away, for the other class). Click the plane to add your own points.</p>"),
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Learning rule", options: [{ value: "perceptron", label: "Perceptron (1958)" }, { value: "logistic", label: "Logistic regression" }], value: algo, onChange: (v) => { algo = v; reset(); } }),
          segmented({ label: "Click adds", options: [{ value: 1, label: "Blue" }, { value: 0, label: "Orange" }], value: 1, onChange: (v) => (addClass = v) }),
          slider({ label: "Learning rate", min: 0.02, max: 1, step: 0.02, value: lr, fmt: (v) => v.toFixed(2), onInput: (v) => (lr = v) }),
          button("Train", () => lp.start(), "primary"),
          button("Step once", () => { lp.stop(); stepOne(); draw(); }),
          button("Reset weights", () => { lp.stop(); reset(); }),
          button("Load XOR", () => { lp.stop(); pts = dataset("xor", 60, { seed: 2 }); reset(); }),
          button("Clear points", () => { lp.stop(); pts = []; reset(); })),
        h("div.stage-body", st.el, h("div.canvas-box", pl.el), msg)),
      callout("tip", "Things to try", "<ul style='margin:0;padding-left:18px'><li>Add a blue point deep in orange territory. The perceptron now never converges: it can only draw one line.</li><li>Switch to logistic regression. Instead of a hard edge you get a <i>probability</i> that fades across the boundary, and it keeps improving its confidence even after every point is correct.</li><li>Load XOR. No single line can split it. Hold that thought for the next chapter.</li></ul>"),
      predict({
        chapter: id,
        question: "If some straight line separates the two classes perfectly, what does the perceptron rule guarantee?",
        options: ["Nothing; it may cycle forever", "It finds a separating line after a finite number of mistakes", "It finds the line with the widest margin", "It finds the line that minimises squared error"],
        answer: 1,
        explain: "<p>That's the <b>perceptron convergence theorem</b> (Novikoff, 1962): the number of mistakes is at most (R/γ)², where R is the data radius and γ the margin. It finds <i>a</i> line, not the best one. Maximising the margin is what support vector machines added in the 1990s.</p>",
      }),
    ),
    section("From hard decisions to probabilities",
      prose("<p>The perceptron's step function has no useful gradient: nudging w a little either changes nothing or flips an answer. <b>Logistic regression</b> swaps the step for a smooth <b>sigmoid</b>, σ(z) = 1 / (1 + e<sup>−z</sup>), so the output is a probability and every point contributes a gradient.</p><p>The loss is <b>cross-entropy</b>: −log of the probability you gave the right answer. Compare it with squared error:</p>"),
      lossChart.el,
      prose("<p>Cross-entropy punishes confident mistakes without limit: giving the right answer 1% costs 4.6, while squared error caps out at 1. That steep wall is what makes models fix their worst errors first. It's the same loss a GPT minimises, over a vocabulary of 100,000 classes instead of 2.</p>"),
      table(["", "Perceptron", "Logistic regression"], [
        ["Output", "0 or 1", "A probability"],
        ["Learns from", "Mistakes only", "Every point, weighted by how wrong it is"],
        ["Loss", "Implicit (count of mistakes)", "Cross-entropy"],
        ["Separable data", "Stops at the first separating line", "Keeps sharpening forever (weights grow)"],
        ["Legacy", "The neuron in every neural network", "The output layer of nearly every classifier"],
      ]),
    ),
    section("In code",
      code({
        "Perceptron": `import numpy as np

def perceptron(X, y, epochs=100, lr=1.0):        # y in {-1, +1}
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(epochs):
        mistakes = 0
        for xi, yi in zip(X, y):
            if yi * (xi @ w + b) <= 0:          # wrong side (or on the line)
                w += lr * yi * xi
                b += lr * yi
                mistakes += 1
        if mistakes == 0:
            break                               # converged
    return w, b`,
        "Logistic regression": `def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic(X, y, steps=1000, lr=0.5):          # y in {0, 1}
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(steps):
        p = sigmoid(X @ w + b)
        grad = p - y                            # d(cross-entropy)/d(score): beautifully simple
        w -= lr * X.T @ grad / len(y)
        b -= lr * grad.mean()
    return w, b`,
      }),
    ),
  );
  pl.paint(draw);
}
