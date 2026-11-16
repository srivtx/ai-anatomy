import { h, rng, loop } from "../lib/core.js";
import { moeLab } from "../lib/labs.js";
import { goal, whyChain, sticker } from "../lib/why.js";
import { walk } from "../lib/walks.js";
import { slider, stats, predict, code, callout, section, prose, table, button, bars } from "../lib/widgets.js";
import { plane, fmtSI } from "../lib/plot.js";
import { gauss, softmax, sigmoid, dataset } from "../lib/nn.js";

const EXPERT_COLORS = ["c1", "c2", "c3", "c4", "c5", "c6", "accent", "warm"];

export default function (root, { id }) {
  // ---------- 1. a mixture of linear experts trained live ----------
  let E = 6;
  const data = dataset("circle", 300, { noise: 0.08, seed: 5 });
  let W, V, r, steps; // W: experts (E × 3: w1, w2, b). V: gate (E × 3)
  const pl = plane({ width: 380, height: 380, label: "Prediction" }), pr = plane({ width: 380, height: 380, label: "Which expert" });
  const mst = stats([{ label: "Steps" }, { label: "Accuracy" }, { label: "Experts in use" }]);
  function init(seed = 3) {
    r = rng(seed); steps = 0;
    W = Array.from({ length: E }, () => [gauss(r), gauss(r), gauss(r) * 0.1]);
    V = Array.from({ length: E }, () => [gauss(r) * 0.5, gauss(r) * 0.5, 0]);
    drawMix();
  }
  const gate = (x) => softmax(V.map((v) => v[0] * x[0] + v[1] * x[1] + v[2]));
  const expertOut = (x) => W.map((w) => w[0] * x[0] + w[1] * x[1] + w[2]);
  const predictZ = (x) => { const g = gate(x), o = expertOut(x); return o.reduce((s, oe, e) => s + g[e] * oe, 0); };
  function trainStep(lr = 0.05) {
    const gW = W.map(() => [0, 0, 0]), gV = V.map(() => [0, 0, 0]);
    for (const { x, y } of data) {
      const g = gate(x), o = expertOut(x), z = o.reduce((s, oe, e) => s + g[e] * oe, 0), dz = sigmoid(z) - y;
      for (let e = 0; e < E; e++) {
        const dO = dz * g[e]; gW[e][0] += dO * x[0]; gW[e][1] += dO * x[1]; gW[e][2] += dO;
        const dl = dz * g[e] * (o[e] - z); // softmax gate gradient: g_e (o_e − Σ g o)
        gV[e][0] += dl * x[0]; gV[e][1] += dl * x[1]; gV[e][2] += dl;
      }
    }
    for (let e = 0; e < E; e++) for (let k = 0; k < 3; k++) { W[e][k] -= (lr * gW[e][k]) / data.length * 10; V[e][k] -= (lr * gV[e][k]) / data.length * 10; }
    steps++;
  }
  function drawMix() {
    pl.paint(() => { pl.clear(); pl.field((x, y) => sigmoid(predictZ([x, y]))); pl.scatter(data); });
    pr.paint(() => {
      pr.clear();
      const res = 70, ctx = pr.ctx, cw = pr.W / res, ch = pr.H / res;
      for (let j = 0; j < res; j++) for (let i = 0; i < res; i++) {
        const [x, y] = pr.inv((i + 0.5) * cw, (j + 0.5) * ch), g = gate([x, y]), e = g.indexOf(Math.max(...g));
        ctx.globalAlpha = 0.25 + 0.55 * g[e]; ctx.fillStyle = pr.colors[EXPERT_COLORS[e]]; ctx.fillRect(i * cw, j * ch, cw + 0.5, ch + 0.5);
      }
      ctx.globalAlpha = 1;
      pr.scatter(data, { color: pr.colors.ink, r: 2, ring: false });
    });
    let ok = 0; const used = new Set();
    for (const { x, y } of data) { ok += (predictZ(x) > 0) === !!y; const g = gate(x); used.add(g.indexOf(Math.max(...g))); }
    mst.set(0, String(steps)); mst.set(1, `${Math.round((ok / data.length) * 100)}%`, ok / data.length > 0.95 ? "good" : ""); mst.set(2, `${used.size} of ${E}`);
  }
  const lp = loop(() => { for (let k = 0; k < 10; k++) trainStep(); drawMix(); if (steps >= 1500) lp.stop(); }, pl.el);

  // ---------- 2. routing, capacity and balance ----------
  const o = { Ex: 8, k: 2, cf: 1.25, skew: 1.5 };
  let bias, rr;
  const rst = stats([{ label: "Token assignments dropped" }, { label: "Busiest expert vs average" }, { label: "Switch load-balancing loss" }]);
  const barsEl = h("div");
  function resetBias() { bias = new Array(o.Ex).fill(0); routeDraw(); }
  function route() {
    rr = rng(17);
    const N = 512, load = new Array(o.Ex).fill(0), probSum = new Array(o.Ex).fill(0), C = Math.ceil((o.cf * N * o.k) / o.Ex);
    let dropped = 0;
    for (let t = 0; t < N; t++) {
      const logits = Array.from({ length: o.Ex }, (_, e) => (o.skew * (o.Ex - 1 - e)) / (o.Ex - 1) + gauss(rr) * 1.2);
      const p = softmax(logits); p.forEach((v, e) => (probSum[e] += v / N));
      // DeepSeek-V3 style: the bias only affects which experts are chosen, not the mixing weights
      const order = logits.map((l, e) => [l + bias[e], e]).sort((a, b) => b[0] - a[0]).slice(0, o.k);
      // tokens over capacity are dropped: they skip this layer through the residual connection
      for (const [, e] of order) { if (load[e] >= C) dropped++; else load[e]++; }
    }
    return { load, probSum, C, dropped, N };
  }
  function routeDraw() {
    const { load, probSum, C, dropped, N } = route();
    const mean = (N * o.k) / o.Ex, f = load.map((l) => l / (N * o.k));
    barsEl.replaceChildren(h("div.eyebrow", `Tokens processed per expert (capacity ${C} each)`), bars(load.map((l, e) => ({ label: `Expert ${e + 1}`, value: l, color: l >= C ? "var(--bad)" : "var(--accent)" })), { max: Math.max(C, ...load), fmt: (v) => (v >= C ? `${v} · full` : String(v)) }));
    rst.set(0, `${((dropped / (N * o.k)) * 100).toFixed(1)}%`, dropped / (N * o.k) > 0.05 ? "bad" : "good");
    rst.set(1, `${(Math.max(...load) / mean).toFixed(2)}×`);
    rst.set(2, (o.Ex * f.reduce((s, fe, e) => s + fe * probSum[e], 0)).toFixed(3));
  }
  function balance() {
    // aux-loss-free balancing: nudge each expert's routing bias against its load
    for (let it = 0; it < 30; it++) {
      const N = 512, mean = (N * o.k) / o.Ex;
      // measure demand before capacity, so the update sees the overflow
      const rr2 = rng(17), demand = new Array(o.Ex).fill(0);
      for (let t = 0; t < N; t++) {
        const logits = Array.from({ length: o.Ex }, (_, e) => (o.skew * (o.Ex - 1 - e)) / (o.Ex - 1) + gauss(rr2) * 1.2);
        logits.map((l, e) => [l + bias[e], e]).sort((a, b) => b[0] - a[0]).slice(0, o.k).forEach(([, e]) => demand[e]++);
      }
      bias = bias.map((b, e) => b - 0.05 * Math.sign(demand[e] - mean));
    }
    routeDraw();
  }

  // ---------- 3. total vs active parameters ----------
  const c = { d: 4096, ff: 1408, L: 48, Ex: 64, k: 6, shared: 2 };
  const pst = stats([{ label: "Total parameters" }, { label: "Active per token" }, { label: "Active fraction" }]);
  function drawParams() {
    const attn = 4 * c.d * c.d, exp = 3 * c.d * c.ff, router = c.d * c.Ex;
    const total = c.L * (attn + (c.Ex + c.shared) * exp + router), active = c.L * (attn + (c.k + c.shared) * exp + router);
    pst.set(0, fmtSI(total)); pst.set(1, fmtSI(active), "good"); pst.set(2, `${((active / total) * 100).toFixed(1)}%`);
  }

  root.append(
    section(null, goal("<p>Bigger models know more, but every parameter used costs compute on every token. We want <b>lots of parameters, with only a few of them used per token</b>.</p>", ["many expert sub-networks", "a router that picks the right ones for each token", "load spread fairly across experts"]),
      prose(`<p>A dense model uses every parameter for every token. A <b>mixture of experts</b> (MoE) replaces each feed-forward layer with many smaller feed-forward “experts” and a <b>router</b> that sends each token to just a few of them. You get the knowledge capacity of a huge model for the compute cost of a small one. Most frontier and leading open models are now MoEs: Mixtral, DeepSeek-V3, Qwen3's largest models, Llama 4. The idea is old (Jacobs, Jordan, Nowlan and Hinton, 1991); making it work at scale is mostly about routing and balance.</p>`),
      h("div.vx-row", sticker("experts", { title: "Two of eight", caption: "the router (purple) sends this token to two experts; the other six do nothing", px: 3 }))),
    section("Experts carve up the input space",
      prose("<p>The original mixture of experts, training live: six <b>linear</b> classifiers, each of which alone can only draw a straight line, and a softmax <b>gate</b> that decides how much to trust each one at each point. Together they learn a circle. The right panel colours each point by the expert the gate trusts most.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Number of experts", min: 1, max: 8, value: E, onInput: (v) => { E = v; lp.stop(); init(); } }),
          button("Train", () => lp.start(), "primary"), button("Pause", () => lp.stop()), button("New seed", () => { lp.stop(); init(Math.floor(Math.random() * 1e4)); lp.start(); })),
        h("div.stage-body", mst.el, h("div.lab", h("div", pl.el, h("p.canvas-cap", "The mixture's prediction")), h("div", pr.el, h("p.canvas-cap", "Each colour is one expert's territory"))))),
      callout("tip", "Things to try", "<ul style='margin:0;padding-left:18px'><li>With 1 expert it's plain logistic regression: one straight line, about 50–60% accuracy at best.</li><li>With 6, each expert specialises on one side of the circle and the gate hands over between them: a smooth piecewise-linear boundary.</li><li>Sometimes an expert gets no territory at all: it's “dead”. In LLM-scale MoEs this happens too, which is why routers need load balancing.</li></ul>"),
    ),
    section("Routing tokens in a transformer",
      prose("<p>In an LLM, the router is a linear layer that scores every expert for every token; the token goes to its <b>top-k</b> experts and their outputs are mixed by the router's weights. On real hardware each expert lives on a different GPU and can only accept a fixed number of tokens per batch: its <b>capacity</b>, set by a capacity factor times the average share. Popular experts overflow and the extra tokens are dropped (they skip the layer through the residual connection).</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Experts", min: 4, max: 16, value: o.Ex, onInput: (v) => { o.Ex = v; resetBias(); } }),
          slider({ label: "Top-k", min: 1, max: 4, value: o.k, onInput: (v) => { o.k = v; routeDraw(); } }),
          slider({ label: "Capacity factor", min: 1, max: 2, step: 0.05, value: o.cf, fmt: (v) => v.toFixed(2), onInput: (v) => { o.cf = v; routeDraw(); } }),
          slider({ label: "Router's favouritism", min: 0, max: 4, step: 0.1, value: o.skew, fmt: (v) => v.toFixed(1), onInput: (v) => { o.skew = v; routeDraw(); } }),
          button("Balance (30 bias updates)", balance, "primary"), button("Reset biases", resetBias)),
        h("div.stage-body", rst.el, barsEl)),
      callout("tip", "Things to try", "<ul style='margin:0;padding-left:18px'><li>With favouritism on, the first experts overflow and many assignments are dropped. Raising the capacity factor reduces drops but wastes memory and compute on padding for the idle experts.</li><li>Press Balance. Each update lowers the routing bias of overloaded experts and raises it for idle ones, until load evens out. This is DeepSeek-V3's <b>auxiliary-loss-free</b> balancing. The older approach (Switch Transformer) adds a loss term, E · Σ f<sub>i</sub>·P<sub>i</sub>, which is smallest (1.0) when load is uniform, but that extra loss also fights the main objective.</li></ul>"),
      predict({
        chapter: id,
        question: "Without any balancing, what tends to happen to the router early in training?",
        options: ["It spreads tokens evenly by itself", "A few experts get slightly better first, so they get more tokens, so they improve faster: a rich-get-richer collapse onto a few experts", "Every expert ends up identical", "It routes randomly forever"],
        answer: 1,
        explain: "<p>Routing collapse is self-reinforcing, so every practical MoE adds a balancing mechanism: auxiliary losses, bias adjustment, noisy routing, or expert-choice routing (each expert picks its top tokens instead, which balances by construction).</p>",
      }),
    ),
    section("Route a token, in 3D",
      prose("<p>One MoE layer. Pick a token and watch the router light up its top-k experts; tower height is the gate weight. Then change the number of experts and k, and watch total and used parameters come apart.</p>"),
      moeLab()),
    section("Total vs active parameters",
      prose("<p>MoEs are described by two numbers. <b>Total</b> parameters must all sit in GPU memory. <b>Active</b> parameters are what each token actually uses, and they set the compute per token. Fine-grained designs (DeepSeekMoE) use many small experts plus a few <b>shared</b> experts every token visits.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Layers", min: 12, max: 96, value: c.L, onInput: (v) => { c.L = v; drawParams(); } }),
          slider({ label: "Model width", min: 1024, max: 8192, step: 256, value: c.d, onInput: (v) => { c.d = v; drawParams(); } }),
          slider({ label: "Expert hidden size", min: 256, max: 16384, step: 128, value: c.ff, onInput: (v) => { c.ff = v; drawParams(); } }),
          slider({ label: "Routed experts", min: 1, max: 256, value: c.Ex, onInput: (v) => { c.Ex = v; drawParams(); } }),
          slider({ label: "Experts per token (k)", min: 1, max: 16, value: c.k, onInput: (v) => { c.k = v; drawParams(); } }),
          slider({ label: "Shared experts", min: 0, max: 4, value: c.shared, onInput: (v) => { c.shared = v; drawParams(); } })),
        h("div.stage-body", pst.el, h("p.canvas-cap", "Counts attention (4·d²), SwiGLU experts (3·d·hidden each) and the router per layer; ignores embeddings and norms."))),
      table(["Model", "Experts", "Total / active parameters"], [
        ["Mixtral 8×7B (2023)", "8, top-2", "47B / 13B"],
        ["DeepSeek-V3 (2024)", "256 routed + 1 shared, top-8", "671B / 37B"],
        ["Qwen3-235B-A22B (2025)", "128, top-8", "235B / 22B"],
        ["Llama 4 Maverick (2025)", "128 routed + 1 shared, top-1", "400B / 17B"],
      ]),
      table(["Trade-off", "Dense", "MoE"], [
        ["Compute per token", "All parameters", "Only active parameters"],
        ["Memory", "Matches compute", "Must hold every expert: heavy for local inference"],
        ["Training", "Simple", "Routing instability, load balance, all-to-all communication between GPUs"],
        ["Batch-size-1 inference", "Efficient", "Still loads many experts' weights: memory-bandwidth bound"],
      ]),
    ),
    walk("moe"),
    section("Why the router needs a referee",
      whyChain({
        claim: "Mixture-of-experts models are trained with something that keeps expert usage even.",
        steps: [
          { q: "Why does the router need help?", a: "<p>Early on, a few experts are slightly better, so the router sends them more tokens, so they learn faster and get better still. Left alone, most tokens go to a handful of experts and the rest never learn: wasted parameters.</p>" },
          { q: "Why is uneven load a hardware problem too?", a: "<p>Experts live on different GPUs. If one expert gets 3× its share, its GPU does 3× the work while the others wait, and tokens past its capacity get dropped or delayed.</p>" },
          { q: "Then why not route at random?", a: "<p>Then experts can't specialise, which was the point. The fix is a compromise: route by content, but pay a penalty when usage gets uneven. Most models add a balancing loss; DeepSeek-V3 nudges a per-expert bias instead.</p>" },
        ],
        axiom: { name: "Rich get richer: positive feedback runs away", html: "<p>When success brings more of what causes success, small differences grow until one side takes nearly everything (winner-take-all markets, preferential attachment). Any learner that chooses its own training data has this loop, and needs something pushing back.</p>" },
      })),
    section("In code",
      code({
        "Top-k MoE layer": `class MoE(nn.Module):
    def __init__(self, d, hidden, n_experts=8, k=2):
        super().__init__()
        self.router = nn.Linear(d, n_experts, bias=False)
        self.experts = nn.ModuleList(SwiGLU(d, hidden) for _ in range(n_experts))
        self.k = k
    def forward(self, x):                               # x: (tokens, d)
        weights, idx = self.router(x).topk(self.k, -1)  # pick k experts per token
        weights = weights.softmax(-1)
        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):       # real kernels group tokens by expert instead
            tok, slot = (idx == e).nonzero(as_tuple=True)
            if len(tok): out[tok] += weights[tok, slot, None] * expert(x[tok])
        return out`,
      }, "Project 17 trains a small MoE language model and plots each expert's load over training, with and without balancing."),
    ),
  );
  init(); resetBias(); drawParams();
}
