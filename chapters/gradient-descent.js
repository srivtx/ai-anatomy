import { h, loop, rng } from "../lib/core.js";
import { gdLab } from "../lib/labs.js";
import { goal, whyChain, sticker } from "../lib/why.js";
import { walk } from "../lib/walks.js";
import { slider, segmented, stats, predict, code, dilemma, callout, section, prose, button, table } from "../lib/widgets.js";
import { plane, xyChart } from "../lib/plot.js";
import { gauss } from "../lib/nn.js";

const LANDS = {
  bowl: { name: "Narrow valley", f: (x, y) => 0.15 * x * x + 3 * y * y, start: [-1.8, 0.7], note: "Steep across, shallow along: the shape of most real loss surfaces near a minimum." },
  banana: { name: "Rosenbrock banana", f: (x, y) => (1 - x) ** 2 + 6 * (y - x * x) ** 2, start: [-1.5, -0.6], note: "A curved valley. Finding the floor is easy; following it to (1, 1) is hard." },
  twin: { name: "Two valleys", f: (x, y) => (x * x - 1) ** 2 + 0.6 * y * y + 0.35 * x + 0.36, start: [1.7, 1.4], note: "Non-convex: a shallow local minimum on the right, a deeper one on the left." },
  saddle: { name: "Saddle", f: (x, y) => 0.6 * x * x - y * y + 0.25 * y ** 4 + 1, start: [-1.7, 0.02], note: "A ridge you can stand on. The gradient across it is almost zero." },
};
const grad = (f, x, y, e = 1e-4) => [(f(x + e, y) - f(x - e, y)) / (2 * e), (f(x, y + e) - f(x, y - e)) / (2 * e)];

const OPTS = [
  { id: "sgd", name: "Gradient descent", color: "var(--c2)" },
  { id: "momentum", name: "Momentum (0.9)", color: "var(--c1)" },
  { id: "adam", name: "Adam", color: "var(--c3)" },
];

function makeRunner(land, start, lr, noise, seed) {
  const r = rng(seed);
  return OPTS.map((o) => ({ ...o, p: [...start], v: [0, 0], m: [0, 0], s: [0, 0], t: 0, path: [[...start]], dead: false }))
    .map((st) => Object.assign(st, {
      step() {
        if (st.dead) return;
        let [gx, gy] = grad(land.f, st.p[0], st.p[1]);
        gx += gauss(r) * noise; gy += gauss(r) * noise; // minibatch noise
        st.t++;
        const g = [gx, gy];
        for (let k = 0; k < 2; k++) {
          if (st.id === "sgd") st.p[k] -= lr * g[k];
          else if (st.id === "momentum") { st.v[k] = 0.9 * st.v[k] + g[k]; st.p[k] -= lr * st.v[k]; }
          else {
            st.m[k] = 0.9 * st.m[k] + 0.1 * g[k]; st.s[k] = 0.999 * st.s[k] + 0.001 * g[k] * g[k];
            const mh = st.m[k] / (1 - 0.9 ** st.t), sh = st.s[k] / (1 - 0.999 ** st.t);
            st.p[k] -= (lr * mh) / (Math.sqrt(sh) + 1e-8)
          }
        }
        if (!st.p.every(Number.isFinite) || Math.abs(st.p[0]) > 40 || Math.abs(st.p[1]) > 40) st.dead = true;
        st.path.push([...st.p]);
      },
    }));
}

export default function (root, { id }) {
  let landId = "bowl", lr = 0.08, noise = 0, start = LANDS.bowl.start, runners, steps = 0;
  const pl = plane({ width: 720, height: 440, xr: [-2.6, 2.6], yr: [-1.6, 1.8], label: "Loss landscape with optimizer paths", onPoint: (x, y) => { start = [x, y]; reset(true); } });
  const st = stats(OPTS.map((o) => ({ label: o.name })));
  const note = h("p.canvas-cap");
  let range;
  function computeRange() {
    const f = LANDS[landId].f;
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i <= 40; i++) for (let j = 0; j <= 40; j++) { const v = f(pl.xr[0] + (i / 40) * 5.2, pl.yr[0] + (j / 40) * 3.4); lo = Math.min(lo, v); hi = Math.max(hi, v); }
    range = [lo, hi];
  }
  function reset(run) {
    runners = makeRunner(LANDS[landId], start, lr, noise, 9);
    steps = 0;
    note.textContent = LANDS[landId].note + " Click anywhere to start from there.";
    computeRange();
    draw();
    if (run) lp.start();
  }
  function draw() {
    const f = LANDS[landId].f;
    pl.clear();
    pl.field((x, y) => Math.log(1 + f(x, y) - range[0]) / Math.log(1 + range[1] - range[0]), { res: 80, a: pl.colors.surface, mid: pl.colors.c1, b: pl.colors.c6, alpha: 0.45, contours: 10 });
    const ctx = pl.ctx;
    for (const rn of runners) {
      const col = pl.colors[rn.color.slice(6, -1)];
      ctx.beginPath();
      rn.path.forEach(([x, y], i) => (i ? ctx.lineTo(pl.X(x), pl.Y(y)) : ctx.moveTo(pl.X(x), pl.Y(y))));
      ctx.strokeStyle = col; ctx.lineWidth = 2.2; ctx.stroke();
      const [x, y] = rn.p;
      if (!rn.dead) { pl.dot(pl.X(x), pl.Y(y), 6.5, pl.colors.surface); pl.dot(pl.X(x), pl.Y(y), 5, col); }
    }
    pl.dot(pl.X(start[0]), pl.Y(start[1]), 4, pl.colors.ink);
    runners.forEach((rn, i) => {
      const v = f(...rn.p);
      st.set(i, rn.dead ? "diverged" : v.toFixed(3), rn.dead ? "bad" : v - range[0] < 0.02 ? "good" : "");
    });
  }
  let acc = 0;
  const lp = loop((dt) => {
    acc += dt;
    while (acc > 1 / 30) { acc -= 1 / 30; runners.forEach((r) => r.step()); steps++; }
    draw();
    if (steps >= 250) lp.stop();
  }, pl.el);

  const lrSlider = slider({ label: "Learning rate", min: -3, max: 0, step: 0.05, value: Math.log10(lr), fmt: (v) => (10 ** v).toFixed(v < -2 ? 4 : 3), onInput: (v) => { lr = 10 ** v; reset(false); } });

  // learning-rate schedule chart
  const sched = xyChart({ height: 200, xLabel: "training step", yLabel: "learning rate", fmtY: (v) => v.toExponential(0) });
  let warm = 200, minFrac = 0.1;
  function drawSched() {
    const N = 2000, peak = 3e-4, pts = [], pts2 = [];
    for (let s = 0; s <= N; s += 10) {
      const cos = s < warm ? (peak * s) / warm : peak * (minFrac + (1 - minFrac) * 0.5 * (1 + Math.cos((Math.PI * (s - warm)) / (N - warm))));
      const wsd = s < warm ? (peak * s) / warm : s < N * 0.8 ? peak : peak * (1 - (s - N * 0.8) / (N * 0.2)) + peak * minFrac * ((s - N * 0.8) / (N * 0.2));
      pts.push([s, cos]); pts2.push([s, wsd]);
    }
    sched.update([{ name: "Warmup + cosine", color: "var(--c1)", points: pts }, { name: "Warmup-stable-decay", color: "var(--c2)", points: pts2, dash: true }], { yDomain: [0, 3.3e-4] });
  }

  root.append(
    section(null, goal("<p>We have a model with millions of knobs and one number, the loss, that says how wrong it is. We want to <b>turn every knob the right way at once</b>, without trying them one by one.</p>", ["the direction each knob should move", "a step that's neither too big nor too small", "something that copes with long, narrow valleys"]),
      prose(`<p>Gradient descent has one idea: the gradient points uphill, so step the other way. For a straight-line fit the loss is a perfect bowl and that's all you need. Real networks have millions of dimensions and surfaces full of narrow valleys, ridges and plateaus. How you step matters as much as which way.</p>
<p>Three optimizers race below from the same starting point. <b>Gradient descent</b> takes <code>lr × gradient</code>. <b>Momentum</b> keeps a running velocity, so it speeds up along consistent directions and cancels zig-zags. <b>Adam</b> also divides each coordinate's step by its recent gradient size, so steep and shallow directions move at similar speeds.</p>`),
      h("div.vx-row", sticker("bowl", { title: "A loss surface", caption: "cut open: the ball rolls down the slope to the lowest point", px: 4 }))),
    section("Race them",
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Landscape", options: Object.entries(LANDS).map(([k, v]) => ({ value: k, label: v.name })), value: landId, onChange: (v) => { landId = v; start = LANDS[v].start; reset(true); } }),
          lrSlider,
          slider({ label: "Gradient noise (small batches)", min: 0, max: 2, step: 0.1, value: 0, fmt: (v) => (v ? v.toFixed(1) : "none"), onInput: (v) => { noise = v; reset(false); } }),
          button("Race", () => reset(true), "primary")),
        h("div.stage-body", st.el, h("div.canvas-box.wide", pl.el), note,
          h("div.legend", OPTS.map((o) => h("span", h("i", { style: { background: o.color } }), o.name))))),
      callout("tip", "Things to try", "<ul style='margin:0;padding-left:18px'><li><b>Narrow valley</b>: raise the learning rate until gradient descent starts to zig-zag, then a bit more until it diverges. Momentum and Adam survive longer.</li><li><b>Two valleys</b>: which optimizer escapes the shallow right-hand valley? Now add noise. Noise from small batches is a feature: it shakes you out of sharp, shallow minima.</li><li><b>Saddle</b>: plain gradient descent creeps for a long time where the slope is almost flat. In high dimensions, saddles are far more common than bad local minima.</li></ul>"),
      predict({
        chapter: id,
        question: "On the narrow valley, why can't plain gradient descent just use a bigger learning rate to go faster?",
        options: ["Bigger steps are always unstable", "The steep direction overshoots and diverges long before the shallow direction gets fast", "The gradient is zero along the valley", "It runs out of momentum"],
        answer: 1,
        explain: "<p>Stability is set by the <b>steepest</b> direction (step must be below about 2 / curvature), and speed by the <b>shallowest</b>. Their ratio, the condition number, is how slow plain GD is. Momentum and Adam attack that ratio in different ways: momentum accumulates speed along the floor, Adam rescales each coordinate.</p>",
      }),
    ),
    section("Roll the ball yourself",
      prose("<p>The loss as terrain (blue is low). Set the learning rate and watch the path. In the narrow valley, find the largest rate that still works, then switch on momentum and compare.</p>"),
      gdLab()),
    section("Schedules: warm up, then cool down",
      prose("<p>Large models use a learning rate that changes over training. <b>Warmup</b> starts tiny because Adam's statistics are garbage in the first steps and early updates are huge. Then it <b>decays</b>, because near the end you want to settle into the valley floor, not bounce across it. Cosine decay was the default for years; <b>warmup-stable-decay</b> (WSD) holds the rate flat and decays only at the end, so you can branch off checkpoints from the flat part.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Warmup steps", min: 0, max: 600, step: 20, value: warm, onInput: (v) => { warm = v; drawSched(); } }),
          slider({ label: "Final LR (fraction of peak)", min: 0, max: 0.5, step: 0.05, value: minFrac, fmt: (v) => `${Math.round(v * 100)}%`, onInput: (v) => { minFrac = v; drawSched(); } })),
        h("div.stage-body", sched.el)),
      table(["Optimizer", "Update", "Memory per parameter", "Used for"], [
        ["SGD", "p -= lr·g", "none", "Classic vision (ResNets), with momentum"],
        ["Momentum", "v = βv + g; p -= lr·v", "1 value", "Still strong for CNNs"],
        ["Adam / AdamW", "per-coordinate scaled step, decoupled weight decay", "2 values", "Almost every transformer"],
        ["Muon", "orthogonalise the update matrix (Newton-Schulz)", "1 value", "Newer LLM training runs; often faster than AdamW"],
        ["Lion, Sophia, Shampoo, SOAP", "sign updates, second-order preconditioning", "1 to many", "Research and some large runs"],
      ]),
      dilemma("Adam or SGD?", [
        { name: "AdamW", pros: ["Works with little tuning", "Handles very different scales across layers", "Standard for transformers"], cons: ["2× extra memory for the optimizer state", "Can generalise slightly worse on some vision tasks"] },
        { name: "SGD + momentum", pros: ["Minimal memory", "Well understood, generalises well on CNNs"], cons: ["Very sensitive to the learning rate", "Struggles with transformers"] },
      ], "For language models, AdamW (or a newer optimizer like Muon) and a warmup + decay schedule. Optimizer state is often bigger than the model weights in memory, which is why ZeRO and 8-bit optimizers exist."),
    ),
    walk("gradient-descent"),
    section("Why downhill works, and when it doesn't",
      whyChain({
        claim: "Stepping against the gradient is the fastest way down, for a small enough step.",
        steps: [
          { q: "Why the gradient?", a: "<p>Close to any point, a smooth loss looks like a tilted plane: L(θ + δ) ≈ L(θ) + ∇L · δ. For a step of fixed length, the dot product ∇L · δ is most negative when δ points straight against ∇L.</p>" },
          { q: "Why only for a small step?", a: "<p>The plane is only an approximation. The real surface curves, and a big step overshoots. In a valley with curvature λ, any learning rate above 2/λ makes the error grow every step instead of shrink.</p>" },
          { q: "Why does Adam help in narrow valleys?", a: "<p>In a narrow valley the steep direction has huge curvature and the floor direction tiny curvature. One learning rate can't suit both. Adam divides each coordinate's step by the size of its recent gradients, so steep and flat directions take similar-sized steps.</p>" },
        ],
        axiom: { name: "Taylor's theorem: smooth functions are locally flat", html: "<p>If a function has a derivative, then close to any point it equals its tangent plane plus an error that shrinks like the square of the step. Every gradient method rests on this: we only know the slope where we stand, and trust it for a short distance.</p>" },
      })),
    section("In code",
      code({
        "PyTorch": `model = MyModel()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

def lr_at(step, warmup=200, total=2000, peak=3e-4, floor=0.1):
    if step < warmup:
        return peak * step / warmup
    t = (step - warmup) / (total - warmup)
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t)))

for step, (x, y) in enumerate(loader):
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    loss = loss_fn(model(x), y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # guard against spikes
    opt.step()`,
        "Adam by hand": `m = beta1 * m + (1 - beta1) * g          # running mean of gradients
v = beta2 * v + (1 - beta2) * g * g      # running mean of squared gradients
m_hat = m / (1 - beta1 ** t)             # bias correction for the first steps
v_hat = v / (1 - beta2 ** t)
p = p - lr * m_hat / (sqrt(v_hat) + eps) - lr * wd * p   # AdamW: decay applied separately`,
      }),
    ),
  );
  drawSched();
  reset(false);
  pl.paint(draw);
}
