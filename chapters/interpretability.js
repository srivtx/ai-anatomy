import { h, rng, loop } from "../lib/core.js";
import { walk } from "../lib/walks.js";
import { slider, stats, predict, code, callout, section, prose, table, button, stepper, bars } from "../lib/widgets.js";
import { plane } from "../lib/plot.js";
import { gauss } from "../lib/nn.js";

const N = 5, M = 2; // five features, two hidden dimensions

export default function (root, { id }) {
  let S = 0.97, decay = 0.7, W, b, mW, vW, mb, vb, t, r, steps, lossAvg;
  const imp = () => Array.from({ length: N }, (_, i) => decay ** i);
  const pl = plane({ width: 380, height: 380, xr: [-1.6, 1.6], yr: [-1.6, 1.6], label: "Feature directions in the 2D hidden space" });
  const st = stats([{ label: "Training steps" }, { label: "Loss" }, { label: "Features represented" }, { label: "Superposition" }]);
  const normEl = h("div"), neuronEl = h("div");
  function init(seed = 1) {
    r = rng(seed);
    W = Array.from({ length: M }, () => Array.from({ length: N }, () => gauss(r) * 0.3));
    b = new Array(N).fill(0);
    mW = W.map((row) => row.map(() => 0)); vW = W.map((row) => row.map(() => 0)); mb = b.map(() => 0); vb = b.map(() => 0);
    t = 0; steps = 0; lossAvg = null; draw();
  }
  function trainStep(B = 128, lr = 0.01) {
    const I = imp(), gW = W.map((row) => row.map(() => 0)), gb = new Array(N).fill(0);
    let loss = 0;
    for (let s = 0; s < B; s++) {
      // each feature is present with probability (1 − S), with a uniform random magnitude
      const x = Array.from({ length: N }, () => (r() < 1 - S ? r() : 0));
      const hd = W.map((row) => row.reduce((a, w, i) => a + w * x[i], 0));
      const pre = Array.from({ length: N }, (_, i) => W.reduce((a, row, j) => a + row[i] * hd[j], 0) + b[i]);
      const out = pre.map((v) => Math.max(0, v));
      const d = out.map((o, i) => (pre[i] > 0 ? -2 * I[i] * (x[i] - o) : 0));
      out.forEach((o, i) => (loss += I[i] * (x[i] - o) ** 2));
      const dh = hd.map((_, j) => d.reduce((a, di, i) => a + di * W[j][i], 0));
      for (let j = 0; j < M; j++) for (let i = 0; i < N; i++) gW[j][i] += d[i] * hd[j] + dh[j] * x[i];
      d.forEach((di, i) => (gb[i] += di));
    }
    t++;
    const adam = (p, g, m, v, k) => { const gg = g / B; m[k] = 0.9 * m[k] + 0.1 * gg; v[k] = 0.999 * v[k] + 0.001 * gg * gg; p[k] -= (lr * m[k] / (1 - 0.9 ** t)) / (Math.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8); };
    for (let j = 0; j < M; j++) for (let i = 0; i < N; i++) adam(W[j], gW[j][i], mW[j], vW[j], i);
    for (let i = 0; i < N; i++) adam(b, gb[i], mb, vb, i);
    steps++; lossAvg = lossAvg == null ? loss / B : 0.98 * lossAvg + 0.02 * (loss / B);
  }
  function draw() {
    const I = imp(), norms = Array.from({ length: N }, (_, i) => Math.hypot(W[0][i], W[1][i])), cols = ["c1", "c2", "c3", "c4", "c5"];
    pl.paint(() => {
      pl.clear(); pl.axes();
      const ctx = pl.ctx;
      ctx.strokeStyle = pl.colors.line; ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.arc(pl.X(0), pl.Y(0), pl.X(1) - pl.X(0), 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]);
      for (let i = 0; i < N; i++) {
        const x = W[0][i], y = W[1][i], c = pl.colors[cols[i]];
        ctx.strokeStyle = c; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(pl.X(0), pl.Y(0)); ctx.lineTo(pl.X(x), pl.Y(y)); ctx.stroke();
        pl.dot(pl.X(x), pl.Y(y), 6, c);
        pl.text(`f${i + 1}`, pl.X(x * 1.13) - 6, pl.Y(y * 1.13) + 4, { size: 12, color: pl.colors.ink });
      }
    });
    normEl.replaceChildren(h("div.eyebrow", "Length of each feature's direction (≈1 = represented)"), bars(norms.map((n, i) => ({ label: `f${i + 1} (importance ${I[i].toFixed(2)})`, value: n, color: `var(--${cols[i]})` })), { max: 1.3, fmt: (v) => v.toFixed(2) }));
    neuronEl.replaceChildren(h("div.eyebrow", "Hidden unit 1 responds to…"), bars(W[0].map((w, i) => ({ label: `f${i + 1}`, value: Math.abs(w), color: `var(--${cols[i]})` })), { max: 1.3, fmt: (v) => v.toFixed(2) }));
    const rep = norms.filter((n) => n > 0.5).length;
    st.set(0, String(steps)); st.set(1, lossAvg == null ? "–" : lossAvg.toFixed(4)); st.set(2, `${rep} of ${N}`, rep > M ? "good" : "");
    st.set(3, rep > M ? `yes: ${rep} features in ${M} dimensions` : "no", rep > M ? "warn" : "");
  }
  const lp = loop(() => { for (let k = 0; k < 40; k++) trainStep(); draw(); if (steps >= 4000) lp.stop(); }, pl.el);

  root.append(
    section(null, prose(`<p>We can read every weight of a neural network and still not know what it's doing. <b>Mechanistic interpretability</b> tries to reverse-engineer networks into understandable parts: which directions in activation space mean what, and which circuits compute what. A central obstacle is <b>superposition</b>: networks represent more features than they have neurons by packing them into overlapping directions, so individual neurons respond to several unrelated things (they're “polysemantic”). Anthropic's toy-models paper (Elhage et al., 2022) showed when and why this happens. You can reproduce it here.</p>`)),
    section("Five features, two dimensions",
      prose("<p>A tiny autoencoder must squeeze 5 input features through a 2-dimensional bottleneck and reconstruct them (with a ReLU at the output). Features differ in <b>importance</b> (how much reconstructing them counts) and all share the same <b>sparsity</b> (how often each is zero). The arrows are the directions the model assigns to each feature.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Sparsity (chance a feature is off)", min: 0, max: 0.99, step: 0.01, value: S, fmt: (v) => `${Math.round(v * 100)}%`, onInput: (v) => { S = v; } }),
          slider({ label: "Importance decay", min: 0.5, max: 1, step: 0.05, value: decay, fmt: (v) => v.toFixed(2), onInput: (v) => { decay = v; } }),
          button("Train", () => { if (steps >= 4000) init(Math.floor(Math.random() * 1e4)); lp.start(); }, "primary"), button("Pause", () => lp.stop()), button("Reset", () => { lp.stop(); init(Math.floor(Math.random() * 1e4)); })),
        h("div.stage-body", st.el, h("div.lab", h("div.canvas-box", pl.el), h("div", normEl, h("div", { style: { marginTop: "14px" } }, neuronEl))))),
      callout("tip", "Things to try (reset after changing a slider)", "<ul style='margin:0;padding-left:18px'><li><b>Sparsity 0%</b> (features always on): the model represents just the two most important features, at right angles, and ignores the rest. Like PCA: no superposition.</li><li><b>Sparsity 95%+</b> (the default): all five features get directions, arranged in a pentagon. At around 90% you typically see four. Because features are rarely on at the same time, the interference from overlapping directions is usually harmless, and the ReLU with a negative bias filters out the small leakage.</li><li>Look at hidden unit 1: it responds to several features at once. That's a polysemantic neuron, and it's why reading single neurons in real models is so confusing.</li></ul>"),
      predict({
        chapter: id,
        question: "Why does sparsity make superposition worthwhile?",
        options: ["Sparse features are more important", "When features are rarely active together, overlapping directions rarely interfere, so the gain from representing extra features outweighs the occasional noise", "Sparsity reduces the number of features", "ReLU only works on sparse inputs"],
        answer: 1,
        explain: "<p>It's a compression trade-off. Real-world concepts (a specific city, a programming language, a person's name) are each absent from almost every context, so large models have strong incentives to store far more features than dimensions. This is the motivation for sparse autoencoders, which try to pull the packed features back apart.</p>",
      }),
    ),
    section("Pulling features back out: sparse autoencoders",
      stepper([
        { title: "1. Collect activations", html: "<p>Run the model over lots of text and record activation vectors at one layer (e.g. the residual stream).</p>" },
        { title: "2. Train a wide, sparse autoencoder", html: "<p>Encode each activation into many more features than dimensions (e.g. 16× or more), with a penalty that makes only a few active at once, then decode back. Each decoder column is a candidate feature direction.</p>" },
        { title: "3. Interpret features", html: "<p>For each feature, look at the texts where it fires most. Many are crisply interpretable: a feature for Python code, for the Golden Gate Bridge, for sycophantic praise.</p>" },
        { title: "4. Test causally", html: "<p>Clamp a feature high or to zero and see if behaviour changes as predicted. Anthropic's “Golden Gate Claude” (2024) amplified one feature until the model brought the bridge into every answer.</p>" },
      ]),
      table(["Method", "Question it answers"], [
        ["Linear probes", "Is this information linearly present at this layer?"],
        ["Logit lens / tuned lens", "What would the model predict if it stopped at this layer?"],
        ["Activation patching", "Which components cause this behaviour? (swap activations between runs and see what changes)"],
        ["Sparse autoencoders / transcoders", "What features does the model use?"],
        ["Circuit analysis, attribution graphs", "How do features combine to produce an output?"],
        ["Steering vectors", "Can we change behaviour by adding a direction?"],
      ]),
      callout("note", "Why it matters", "<p>If we can read what a model represents and computes, we can check for problems that behavioural tests miss: hidden goals, deceptive reasoning, or knowledge the model shouldn't use. That's still an aspiration; current tools explain parts of the computation for particular prompts, not the whole model.</p>"),
    ),
    walk("interpretability"),
    section("In code",
      code({
        "Toy superposition model": `class ToyModel(nn.Module):
    def __init__(self, n_features=5, n_hidden=2):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_hidden, n_features) * 0.3)
        self.b = nn.Parameter(torch.zeros(n_features))
    def forward(self, x):                                   # x: (batch, n_features)
        return F.relu(x @ self.W.T @ self.W + self.b)

def batch(n, sparsity, n_features=5):
    x = torch.rand(n, n_features)
    return x * (torch.rand(n, n_features) > sparsity)

importance = 0.7 ** torch.arange(5)
loss = (importance * (x - model(x)) ** 2).sum(-1).mean()`,
        "Sparse autoencoder": `class SAE(nn.Module):
    def __init__(self, d_model, n_features):
        super().__init__()
        self.enc = nn.Linear(d_model, n_features); self.dec = nn.Linear(n_features, d_model)
    def forward(self, a):
        f = F.relu(self.enc(a - self.dec.bias))             # sparse feature activations
        return self.dec(f), f
recon, f = sae(acts)
loss = F.mse_loss(recon, acts) + l1_coef * f.abs().sum(-1).mean()`,
      }, "Project 18 reproduces the toy-models phase diagram and trains an SAE that recovers the hidden features."),
    ),
  );
  init();
}
