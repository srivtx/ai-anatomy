import { h, loop } from "../lib/core.js";
import { segmented, slider, stats, predict, code, callout, section, prose, button, liveChart, table } from "../lib/widgets.js";
import { plane } from "../lib/plot.js";
import { MLP, dataset, bceLogit, sigmoid } from "../lib/nn.js";

const ARCHS = { "0": [], "2": [2], "4": [4], "8": [8], "8x8": [8, 8], "16x16": [16, 16] };

export default function (root, { id }) {
  let kind = "xor", arch = "0", act = "tanh", lr = 0.03, seed = 4;
  let data, net, epoch, loss;
  const pl = plane({ width: 440, height: 440, label: "Decision boundary of the network" });
  const hid = plane({ width: 300, height: 300, xr: [-1.1, 1.1], yr: [-1.1, 1.1], label: "The data as the hidden layer sees it" });
  const thumbs = h("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(84px, 1fr))", gap: "6px" } });
  let thumbPlanes = [];
  const st = stats([{ label: "Epoch" }, { label: "Loss" }, { label: "Accuracy" }, { label: "Parameters" }]);
  const chart = liveChart({ series: [{ name: "Training loss", color: "var(--accent)" }], height: 110, points: 150, yMax: 0.8, fmt: (v) => v.toFixed(1) });
  const hidWrap = h("div", { style: { marginTop: "12px" } }, h("div.eyebrow", "The same data in hidden space (x = neuron 1, y = neuron 2)"), h("div.canvas-box", { style: { maxWidth: "300px" } }, hid.el), h("p.canvas-cap", "The output layer draws a straight line here."));

  function reset() {
    data = dataset(kind, 200, { noise: kind === "spiral" ? 0.05 : 0.08, seed: 3 });
    const sizes = [2, ...ARCHS[arch], 1];
    net = new MLP(sizes, { act, seed });
    epoch = 0; loss = 0;
    chart.reset();
    const n1 = ARCHS[arch][0] || 0;
    thumbPlanes = Array.from({ length: Math.min(n1, 16) }, () => plane({ width: 120, height: 120, label: "One hidden neuron's response" }));
    thumbs.replaceChildren(...thumbPlanes.map((t, k) => h("div", h("div.canvas-box", t.el), h("p.canvas-cap", { style: { margin: "2px 0 0", textAlign: "center" } }, `neuron ${k + 1}`))));
    hidWrap.hidden = n1 !== 2;
    st.set(3, String(net.params));
    draw();
  }
  function train() {
    loss = 0;
    let ok = 0;
    for (const p of data) {
      const z = net.forward(p.x)[0];
      const r = bceLogit(z, p.y);
      loss += r.loss; ok += (z > 0) === !!p.y;
      net.backward([r.d]);
    }
    net.step(lr, data.length);
    loss /= data.length; epoch++;
    return ok / data.length;
  }
  let lastAcc = 0;
  function draw() {
    pl.clear();
    pl.field((x, y) => sigmoid(net.forward([x, y])[0]), { res: 64, alpha: 0.55, contours: [0.5] });
    pl.axes();
    pl.scatter(data, { r: 3.5 });
    thumbPlanes.forEach((t, k) => {
      t.clear();
      t.field((x, y) => { net.forward([x, y]); const v = net.cache.ys[1][k]; return act === "tanh" ? (v + 1) / 2 : act === "relu" ? Math.min(1, v / 2) : v; }, { res: 30, alpha: 0.7, a: t.colors.surface, mid: t.colors.c6, b: t.colors.c6 });
    });
    if (!hidWrap.hidden) {
      hid.clear(); hid.axes();
      const toHidden = (p) => { net.forward(p.x); const v = net.cache.ys[1]; return act === "tanh" ? [v[0], v[1]] : [v[0] * 0.6 - 1, v[1] * 0.6 - 1]; };
      hid.scatter(data.map((p) => ({ x: toHidden(p), y: p.y })), { r: 3 });
      // the output layer is a straight line in this space
      const W = net.W[1], b = net.b[1][0];
      if (act === "tanh" && Math.abs(W[1]) > 1e-6) hid.curve((x) => -(W[0] * x + b) / W[1], { color: hid.colors.ink, lw: 2 });
    }
    st.set(0, String(epoch));
    st.set(1, epoch ? loss.toFixed(3) : "–", loss < 0.1 && epoch ? "good" : "");
    st.set(2, epoch ? `${Math.round(lastAcc * 100)}%` : "–", lastAcc > 0.97 ? "good" : lastAcc < 0.7 && epoch > 300 ? "bad" : "");
  }
  let frame = 0;
  const lp = loop(() => {
    for (let k = 0; k < 6; k++) lastAcc = train();
    if (++frame % 3 === 0) chart.push([loss]);
    draw();
    if (epoch > 4000) lp.stop();
  }, pl.el);

  root.append(
    section(null, prose(`<p>XOR is the simplest function a perceptron can't learn: output 1 when exactly one input is on. Plot it and the ones sit in opposite corners. No single line separates them.</p>
<p>In 1969 Marvin Minsky and Seymour Papert proved this kind of limit rigorously in their book <i>Perceptrons</i>. Funding for neural networks largely dried up: the first <b>AI winter</b>. The fix, a <b>hidden layer</b> trained by backpropagation, was popularised by Rumelhart, Hinton and Williams in 1986. Here it is, live.</p>`)),
    section("Bend space until a line works",
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Data", options: [{ value: "xor", label: "XOR" }, { value: "circle", label: "Circle" }, { value: "moons", label: "Moons" }, { value: "spiral", label: "Spiral" }], value: kind, onChange: (v) => { kind = v; reset(); lp.start(); } }),
          segmented({ label: "Hidden layers", options: [{ value: "0", label: "none" }, { value: "2", label: "2" }, { value: "4", label: "4" }, { value: "8", label: "8" }, { value: "8x8", label: "8+8" }, { value: "16x16", label: "16+16" }], value: arch, onChange: (v) => { arch = v; reset(); lp.start(); } }),
          segmented({ label: "Activation", options: [{ value: "tanh", label: "tanh" }, { value: "relu", label: "ReLU" }, { value: "sigmoid", label: "sigmoid" }], value: act, onChange: (v) => { act = v; reset(); lp.start(); } }),
          slider({ label: "Learning rate", min: 0.002, max: 0.1, step: 0.002, value: lr, fmt: (v) => v.toFixed(3), onInput: (v) => (lr = v) }),
          button("Train", () => lp.start(), "primary"),
          button("New random init", () => { seed++; reset(); lp.start(); }),
          button("Pause", () => lp.stop())),
        h("div.stage-body", st.el,
          h("div.lab",
            h("div", h("div.canvas-box", pl.el), h("p.canvas-cap", "Blue regions: the network says class 1. Orange: class 0.")),
            h("div", h("div.eyebrow", "First hidden layer, one neuron each"), thumbs,
              h("p.canvas-cap", "Each neuron is itself a soft line. The output combines them."),
              hidWrap)),
          chart.el)),
      callout("tip", "Things to try", "<ul style='margin:0;padding-left:18px'><li>XOR with <b>no</b> hidden layer: loss is stuck at 0.69, which is ln 2: the network gives up and says 50/50 everywhere.</li><li>XOR with <b>2</b> hidden neurons: look at the right-hand view. The hidden layer moves the four corners so a straight line splits them. That's what “learning a representation” means.</li><li>Spiral needs depth. Compare 8 vs 8+8. Then try ReLU: the boundary becomes piecewise straight, made of folds.</li><li>Press New random init a few times on 2 hidden neurons: sometimes it gets stuck. Bigger networks almost never do.</li></ul>"),
      predict({
        chapter: id,
        question: "What's the smallest number of hidden neurons that can solve XOR (one hidden layer, smooth activation)?",
        options: ["1", "2", "4", "It needs two hidden layers"],
        answer: 1,
        explain: "<p>Two. One neuron learns roughly “x₁ OR x₂”, the other “x₁ AND x₂”, and the output computes OR-but-not-AND. The universal approximation theorem (Cybenko, 1989) goes much further: one hidden layer with enough neurons can approximate any continuous function. It says nothing about how many neurons, or whether training finds them. Depth makes many functions exponentially cheaper.</p>",
      }),
    ),
    section("Why the hidden layer works",
      prose(`<p>A layer computes <code>h = act(W·x + b)</code>. The <code>W·x + b</code> part can only rotate, stretch and shift. The <b>non-linear activation</b> is what folds and bends. Without it, any stack of layers collapses into one linear map, and you're back to a single line.</p>
<p>So a deep network is a sequence of space-bending steps, ending in a final layer that just draws a line (or a hyperplane). Everything else in this course, from CNNs to transformers, is a cleverer way to arrange those bends.</p>`),
      table(["Activation", "Formula", "Notes"], [
        ["sigmoid", "1 / (1 + e^−x)", "Historical. Saturates at both ends, so gradients vanish in deep stacks."],
        ["tanh", "(e^x − e^−x) / (e^x + e^−x)", "Zero-centred sigmoid. Common in RNNs."],
        ["ReLU", "max(0, x)", "Cheap and doesn't saturate for x > 0. Made deep nets trainable (2010s)."],
        ["GELU / SiLU", "x · Φ(x), x · σ(x)", "Smooth ReLUs. Used in GPT, BERT and Llama."],
        ["SwiGLU", "SiLU(xW) ⊙ (xV)", "A gated MLP. The standard in modern LLM feed-forward blocks."],
      ]),
    ),
    section("In code",
      code({
        PyTorch: `import torch, torch.nn as nn

X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])

model = nn.Sequential(nn.Linear(2, 2), nn.Tanh(), nn.Linear(2, 1))
opt = torch.optim.Adam(model.parameters(), lr=0.05)
for step in range(2000):
    loss = nn.functional.binary_cross_entropy_with_logits(model(X), y)
    opt.zero_grad(); loss.backward(); opt.step()

print(torch.sigmoid(model(X)).round())   # tensor([[0.], [1.], [1.], [0.]])`,
      }),
    ),
  );
  reset();
  pl.paint(draw);
}
