import { h, rng } from "../lib/core.js";
import { walk } from "../lib/walks.js";
import { slider, segmented, toggle, stats, predict, code, callout, section, prose, table, dilemma } from "../lib/widgets.js";
import { xyChart } from "../lib/plot.js";
import { gauss } from "../lib/nn.js";

// A deep stack: h <- [norm](h + act(W h)) or h <- act(W h). Forward and backward
// written out by hand so we can read every layer's activation and gradient size.
function simulate({ depth, width, act, scale, residual, norm, seed = 1 }) {
  const r = rng(seed), B = 32;
  const f = act === "tanh" ? Math.tanh : (x) => (x > 0 ? x : 0);
  const df = act === "tanh" ? (y) => 1 - y * y : (y) => (y > 0 ? 1 : 0);
  const base = act === "relu" ? Math.sqrt(2 / width) : Math.sqrt(1 / width); // He / Xavier
  const Ws = Array.from({ length: depth }, () => Float64Array.from({ length: width * width }, () => gauss(r) * base * scale));
  let H = Array.from({ length: B }, () => Float64Array.from({ length: width }, () => gauss(r)));
  const cache = [], actStd = [];
  for (let l = 0; l < depth; l++) {
    const W = Ws[l], Hin = H, Z = [], A = [], N = [];
    const out = Hin.map((x) => {
      const a = new Float64Array(width);
      for (let j = 0; j < width; j++) { let s = 0; for (let i = 0; i < width; i++) s += x[i] * W[i * width + j]; a[j] = f(s); }
      A.push(a);
      let y = residual ? x.map((v, j) => v + a[j]) : a;
      if (norm) { const m = y.reduce((p, v) => p + v, 0) / width, sd = Math.sqrt(y.reduce((p, v) => p + (v - m) ** 2, 0) / width + 1e-5); y = y.map((v) => (v - m) / sd); N.push(sd); }
      Z.push(y);
      return y;
    });
    cache.push({ Hin, A, Z, N });
    H = out;
    const all = H.flatMap((v) => [...v]), m = all.reduce((p, v) => p + v, 0) / all.length;
    actStd.push(Math.sqrt(all.reduce((p, v) => p + (v - m) ** 2, 0) / all.length));
  }
  // backward from a random linear readout
  const probe = Float64Array.from({ length: width }, () => gauss(r));
  let G = H.map(() => Float64Array.from(probe));
  const gradNorm = new Array(depth);
  for (let l = depth - 1; l >= 0; l--) {
    const { Hin, A, Z, N } = cache[l], W = Ws[l];
    let gW = 0;
    const gWm = new Float64Array(width * width);
    G = G.map((g, b) => {
      if (norm) { const y = Z[b], mg = g.reduce((p, v) => p + v, 0) / width, mgy = g.reduce((p, v, j) => p + v * y[j], 0) / width; g = g.map((v, j) => (v - mg - y[j] * mgy) / N[b]); }
      const gz = g.map((v, j) => v * df(A[b][j]));
      const x = Hin[b], gin = new Float64Array(width);
      for (let i = 0; i < width; i++) { let s = 0; for (let j = 0; j < width; j++) { s += W[i * width + j] * gz[j]; gWm[i * width + j] += x[i] * gz[j]; } gin[i] = s + (residual ? g[i] : 0); }
      return gin;
    });
    for (const v of gWm) gW += v * v;
    gradNorm[l] = Math.sqrt(gW) / B;
  }
  return { actStd, gradNorm };
}

export default function (root, { id }) {
  const o = { depth: 30, width: 48, act: "tanh", scale: 1, residual: false, norm: false };
  const st = stats([{ label: "Activation std, last layer" }, { label: "Gradient, first layer ÷ last layer" }, { label: "Verdict" }]);
  const aChart = xyChart({ width: 460, height: 260, xLabel: "layer", yLabel: "activation std", yLog: true });
  const gChart = xyChart({ width: 460, height: 260, xLabel: "layer", yLabel: "weight-gradient size", yLog: true });
  function run() {
    const { actStd, gradNorm } = simulate(o);
    aChart.update([{ name: "Activation std", color: "var(--c1)", points: actStd.map((v, i) => [i + 1, Math.max(v, 1e-8)]) }], { yDomain: [1e-4, 1e3] });
    gChart.update([{ name: "Gradient", color: "var(--c2)", points: gradNorm.map((v, i) => [i + 1, Math.max(v, 1e-12)]) }]);
    const last = actStd[actStd.length - 1], ratio = gradNorm[0] / Math.max(gradNorm[gradNorm.length - 1], 1e-30);
    st.set(0, last < 1e-3 ? last.toExponential(1) : last.toFixed(3), last < 0.05 || last > 20 ? "bad" : "good");
    st.set(1, ratio < 1e-3 || ratio > 1e3 ? ratio.toExponential(1) : ratio.toFixed(2), ratio < 0.01 || ratio > 100 ? "bad" : "good");
    const ok = !(last < 0.05 || last > 20 || ratio < 0.01 || ratio > 100);
    st.set(2, ok ? "Trainable" : last < 0.05 || ratio < 0.01 ? "Signal vanishes" : "Signal explodes", ok ? "good" : "bad");
  }
  const S = (label, key, min, max, step, fmt, hint) => slider({ label, min, max, step, value: o[key], fmt, hint, onInput: (v) => { o[key] = v; run(); } });

  root.append(
    section(null, prose(`<p>Stack 30 layers and initialise them carelessly, and the network can't learn at all. Each layer multiplies the signal by roughly the same factor; 30 multiplications turn 0.8 into 0.001 and 1.2 into 237. The forward signal dies or explodes, and so does the gradient flowing back. For about twenty years this made deep networks untrainable.</p>
<p>The fixes, roughly in the order they were discovered: <b>careful initialisation</b>, <b>ReLU</b>, <b>normalisation</b> layers, and <b>residual connections</b>. Every modern transformer uses all four.</p>`)),
    section("Send a signal through 30 layers",
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Activation", options: [{ value: "tanh", label: "tanh (Xavier init)" }, { value: "relu", label: "ReLU (He init)" }], value: o.act, onChange: (v) => { o.act = v; run(); } }),
          S("Init scale (× the textbook value)", "scale", 0.3, 2, 0.05, (v) => `${v.toFixed(2)}×`, "1× is Xavier for tanh, He for ReLU"),
          S("Depth", "depth", 2, 60, 1),
          toggle({ label: "Residual connections (h + f(h))", onChange: (v) => { o.residual = v; run(); } }),
          toggle({ label: "LayerNorm after each layer", onChange: (v) => { o.norm = v; run(); } })),
        h("div.stage-body", st.el, h("div.lab", aChart.el, gChart.el))),
      callout("tip", "Things to try", "<ul style='margin:0;padding-left:18px'><li>tanh at 0.7×: the activations shrink layer after layer and the first layer's gradient is thousands of times smaller than the last's. The early layers would never learn.</li><li>ReLU at 1×: He initialisation (√(2/fan-in)) keeps the signal level. Drop it to 0.7× (that's Xavier for ReLU) and it decays; at 1.4× it explodes.</li><li>Pick any bad scale, then turn on LayerNorm: activations are pinned to std 1. Turn on residuals too: gradients get a direct path to the first layer.</li></ul>"),
      predict({
        chapter: id,
        question: "Why do residual connections (output = x + f(x)) let you train 100+ layer networks?",
        options: ["They add more parameters", "The gradient can flow straight through the + to early layers, so it can't vanish multiplicatively", "They act as dropout", "They normalise the activations"],
        answer: 1,
        explain: "<p>The derivative of x + f(x) is 1 + f′(x). Even if f′ is tiny, the 1 carries the gradient back untouched. Each block only has to learn a small <i>correction</i> to its input. ResNet (He et al., 2015) used this to train 152 layers; every transformer block is two residual branches (attention and MLP) added onto a <b>residual stream</b>.</p>",
      }),
    ),
    section("The toolkit",
      table(["Trick", "Fixes", "Detail"], [
        ["Xavier / He init", "Signal scale at the start", "Weights ~ N(0, 1/fan-in) for tanh, N(0, 2/fan-in) for ReLU"],
        ["ReLU family", "Saturation (tanh/sigmoid gradients near 0)", "Gradient is exactly 1 for positive inputs"],
        ["BatchNorm", "Drifting activation scales", "Normalise over the batch; great for CNNs, awkward for sequences"],
        ["LayerNorm / RMSNorm", "Same, per example", "Normalise over features; RMSNorm (no mean) is standard in LLMs"],
        ["Pre-norm placement", "Unstable deep transformers", "x + f(norm(x)) instead of norm(x + f(x))"],
        ["Residual connections", "Vanishing gradients with depth", "y = x + f(x)"],
        ["Dropout", "Overfitting", "Zero random activations in training; scale up the rest"],
        ["Gradient clipping", "Loss spikes from rare huge gradients", "Rescale if the global norm exceeds 1.0"],
        ["Warmup", "Early instability with Adam", "Ramp the learning rate up over the first steps"],
        ["μP (maximal update parametrisation)", "Hyperparameters that don't transfer across sizes", "Scale init and LR with width so the best LR for a small model works for a big one"],
      ]),
      dilemma("BatchNorm or LayerNorm?", [
        { name: "BatchNorm", pros: ["Regularises a little (batch noise)", "Fuses into the conv at inference"], cons: ["Depends on batch size; breaks with batch 1", "Different behaviour in train and eval mode", "Awkward for variable-length sequences"] },
        { name: "LayerNorm / RMSNorm", pros: ["Same in training and inference", "Works for any batch size and sequence"], cons: ["Slightly more compute per token"] },
      ], "Vision CNNs: BatchNorm. Transformers and anything sequential: LayerNorm, now usually RMSNorm in pre-norm position."),
    ),
    walk("training-tricks"),
    section("In code",
      code({
        PyTorch: `import torch.nn as nn

class Block(nn.Module):                    # a pre-norm residual MLP block, transformer style
    def __init__(self, d, p_drop=0.1):
        super().__init__()
        self.norm = nn.RMSNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(p_drop))
    def forward(self, x):
        return x + self.ff(self.norm(x))   # residual: gradient highway

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")   # He init
        nn.init.zeros_(m.bias)

model = nn.Sequential(*[Block(512) for _ in range(48)])
model.apply(init_weights)
# in the loop: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`,
      }),
    ),
  );
  run();
}
