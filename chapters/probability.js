import { h } from "../lib/core.js";
import { slider, segmented, stats, predict, code, callout, section, prose, table, bars } from "../lib/widgets.js";
import { xyChart } from "../lib/plot.js";

const FACES = ["1", "2", "3", "4", "5", "6"];
const P_PRESETS = {
  fair: { label: "Fair die", p: [1, 1, 1, 1, 1, 1] },
  loaded: { label: "Loaded die", p: [5, 1, 1, 1, 1, 1] },
  sure: { label: "Almost certain", p: [97, 0.6, 0.6, 0.6, 0.6, 0.6] },
  coin: { label: "Coin (1 or 2)", p: [1, 1, 0, 0, 0, 0] },
};
const Q_PRESETS = {
  uniform: { label: "Uniform guess", q: () => [1, 1, 1, 1, 1, 1] },
  match: { label: "Matches the truth", q: (p) => p },
  over: { label: "Overconfident on 1", q: () => [20, 0.2, 0.2, 0.2, 0.2, 0.2] },
  wrong: { label: "Wrong favourite (6)", q: () => [1, 1, 1, 1, 1, 5] },
};
const norm = (v) => { const s = v.reduce((a, b) => a + b, 0); return v.map((x) => x / s); };
const H = (p) => -p.reduce((a, x) => a + (x > 0 ? x * Math.log2(x) : 0), 0);
const CE = (p, q) => -p.reduce((a, x, i) => a + (x > 0 ? x * Math.log2(Math.max(q[i], 1e-12)) : 0), 0);

export default function (root, { id }) {
  let pKey = "loaded", qKey = "uniform", smooth = 0;
  const st = stats([{ label: "Entropy H(p)" }, { label: "Cross-entropy H(p, q)" }, { label: "KL(p ‖ q): wasted bits" }]);
  const pBars = h("div"), qBars = h("div"), codeTbl = h("div");
  function draw() {
    const p = norm(P_PRESETS[pKey].p), q0 = norm(Q_PRESETS[qKey].q(P_PRESETS[pKey].p));
    const q = q0.map((x) => (1 - smooth) * x + smooth / 6);
    pBars.replaceChildren(h("div.eyebrow", "Truth p"), bars(p.map((x, i) => ({ label: `face ${FACES[i]}`, value: x * 100, color: "var(--c1)" })), { max: 100, fmt: (v) => `${v.toFixed(1)}%` }));
    qBars.replaceChildren(h("div.eyebrow", "Model q"), bars(q.map((x, i) => ({ label: `face ${FACES[i]}`, value: x * 100, color: "var(--c2)" })), { max: 100, fmt: (v) => `${v.toFixed(1)}%` }));
    const hp = H(p), ce = CE(p, q);
    st.set(0, `${hp.toFixed(3)} bits`); st.set(1, `${ce.toFixed(3)} bits`); st.set(2, `${(ce - hp).toFixed(3)} bits`, ce - hp < 0.01 ? "good" : ce - hp > 1 ? "bad" : "warn");
    codeTbl.replaceChildren(table(["Face", "p", "Ideal code length −log₂ p", "Length your model assigns −log₂ q"],
      p.map((x, i) => [FACES[i], x.toFixed(3), x > 0 ? `${(-Math.log2(x)).toFixed(2)} bits` : "never happens", `${(-Math.log2(Math.max(q[i], 1e-12))).toFixed(2)} bits`])));
  }

  // maximum likelihood and a prior
  let n = 10, heads = 7, prior = 0;
  const lik = xyChart({ height: 230, xLabel: "θ = probability of heads", yLabel: "relative belief", xDomain: [0, 1], yDomain: [0, 1.05] });
  const mst = stats([{ label: "Maximum likelihood θ" }, { label: "With the prior (MAP)" }]);
  function drawLik() {
    const xs = Array.from({ length: 201 }, (_, i) => i / 200);
    const ll = (t, a) => (heads + a) * Math.log(Math.max(t, 1e-12)) + (n - heads + a) * Math.log(Math.max(1 - t, 1e-12));
    const curve = (a) => { const v = xs.map((t) => ll(t, a)); const m = Math.max(...v); return xs.map((t, i) => [t, Math.exp(v[i] - m)]); };
    const series = [{ name: "Likelihood of the data", color: "var(--c1)", points: curve(0) }];
    if (prior > 0) series.push({ name: `Posterior with a Beta(${prior + 1}, ${prior + 1}) prior`, color: "var(--c2)", points: curve(prior), dash: true });
    lik.update(series);
    mst.set(0, (heads / n).toFixed(3));
    mst.set(1, ((heads + prior) / (n + 2 * prior)).toFixed(3));
  }
  const headsSl = slider({ label: "Heads observed", min: 0, max: n, value: heads, onInput: (v) => { heads = v; drawLik(); } });

  root.append(
    section(null, prose(`<p>Almost every loss in deep learning is a statement about probability. A classifier outputs a <b>distribution</b> over classes; a language model outputs a distribution over the next token. Training pushes those distributions towards the data. To understand <i>how far</i> apart two distributions are, Claude Shannon gave us a unit in 1948: the <b>bit</b>.</p>`)),
    section("Entropy, cross-entropy and KL",
      prose("<p><b>Entropy</b> H(p) is the average surprise of outcomes from p: the fewest bits per outcome any code could use. A fair die needs log₂ 6 ≈ 2.58 bits; a die that almost always lands on 1 needs nearly none.</p><p><b>Cross-entropy</b> H(p, q) is the bits you actually spend if the world follows p but you design your code (your model) for q. It's always at least H(p). The excess is the <b>KL divergence</b>: the price of being wrong.</p>"),
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "The truth", options: Object.entries(P_PRESETS).map(([k, v]) => ({ value: k, label: v.label })), value: pKey, onChange: (v) => { pKey = v; draw(); } }),
          segmented({ label: "Your model", options: Object.entries(Q_PRESETS).map(([k, v]) => ({ value: k, label: v.label })), value: qKey, onChange: (v) => { qKey = v; draw(); } }),
          slider({ label: "Smooth the model towards uniform", min: 0, max: 1, step: 0.05, value: 0, fmt: (v) => `${Math.round(v * 100)}%`, onInput: (v) => { smooth = v; draw(); } })),
        h("div.stage-body", st.el, h("div.lab", pBars, qBars), codeTbl)),
      callout("tip", "Things to try", "<ul style='margin:0;padding-left:18px'><li>Truth: almost certain. Model: wrong favourite. KL is huge: the model keeps saying face 1 is unlikely, so it pays ~5 bits every time it happens.</li><li>Now add smoothing. Hedging towards uniform caps the damage. That's <b>label smoothing</b>, and why models shouldn't output exactly 0%.</li><li>Model: matches the truth. KL is exactly 0, and cross-entropy equals entropy: the floor no model can beat.</li></ul>"),
      predict({
        chapter: id,
        question: "A language model's training loss stops falling at 1.8 nats per token, even with more compute. What's the most likely reason?",
        options: ["A bug in the optimizer", "Text itself has irreducible entropy: you can't predict the next word perfectly", "The model is too big", "Cross-entropy can't go below 2"],
        answer: 1,
        explain: "<p>Cross-entropy = entropy of the data + KL to the model. Training can only shrink the KL. The entropy of language is a floor; scaling-law fits even estimate it (the constant E ≈ 1.69 nats in the Chinchilla fit). Note PyTorch uses natural log, so losses are in <b>nats</b>: 1 nat = 1.44 bits.</p>",
      }),
    ),
    section("Maximum likelihood: where losses come from",
      prose("<p>You flip a coin 10 times and see 7 heads. What's θ, the probability of heads? <b>Maximum likelihood</b> picks the θ that makes your data most probable: here θ = 0.7. Taking the log turns the product of probabilities into a sum, and flipping the sign gives the <b>negative log-likelihood</b>, which is exactly cross-entropy. Minimising cross-entropy <i>is</i> maximum likelihood.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Flips", min: 1, max: 200, value: n, onInput: (v) => { n = v; heads = Math.min(heads, n); headsSl.querySelector("input").max = n; headsSl.value = heads; drawLik(); } }),
          headsSl,
          slider({ label: "Prior strength (pseudo-flips each way)", min: 0, max: 50, value: 0, onInput: (v) => { prior = v; drawLik(); } })),
        h("div.stage-body", mst.el, lik.el)),
      prose("<p>With 3 flips and 3 heads, maximum likelihood says θ = 1: the coin <i>never</i> lands tails. That's overfitting in its purest form. A <b>prior</b> (pretend you already saw some heads and tails) pulls the estimate back to sanity; that's <b>MAP</b> estimation, and L2 weight decay is exactly a Gaussian prior on the weights. Raise the flips and the data overwhelms any prior.</p>"),
      table(["Loss you use", "Is really", "Assumes"], [
        ["Mean squared error", "Negative log-likelihood", "Gaussian noise around the prediction"],
        ["Binary cross-entropy", "Negative log-likelihood", "A Bernoulli (coin) outcome"],
        ["Softmax cross-entropy", "Negative log-likelihood", "A categorical outcome (next token)"],
        ["+ weight decay", "Negative log-posterior", "A Gaussian prior on weights"],
        ["KL term in a VAE", "Distance from the prior", "Latents should look like N(0, 1)"],
        ["KL penalty in RLHF", "Stay close to the reference model", "Don't drift into reward hacking"],
      ]),
    ),
    section("In code",
      code({
        PyTorch: `import torch, torch.nn.functional as F

logits = torch.tensor([[2.0, 0.5, -1.0]])      # model output for one example
target = torch.tensor([0])
loss = F.cross_entropy(logits, target)          # = -log softmax(logits)[target], in nats

p = torch.tensor([0.7, 0.2, 0.1])
q = torch.softmax(logits, -1)[0]
entropy = -(p * p.log()).sum()
cross = -(p * q.log()).sum()
kl = (p * (p / q).log()).sum()                  # cross - entropy
print(entropy, cross, kl, cross - entropy)`,
      }),
    ),
  );
  draw();
  drawLik();
}
