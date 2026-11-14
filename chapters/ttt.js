import { h, rng } from "../lib/core.js";
import { slider, segmented, stats, predict, code, callout, section, prose, table, stepper } from "../lib/widgets.js";
import { xyChart } from "../lib/plot.js";
import { gauss } from "../lib/nn.js";

const D = 16;
const unit = (v) => { const n = Math.hypot(...v) || 1; return v.map((x) => x / n); };
const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);

const METHODS = [
  { key: "softmax", name: "Softmax attention (keeps every pair)", color: "var(--ink-3)", dash: true },
  { key: "linear", name: "Linear attention: M += v·kᵀ", color: "var(--c2)" },
  { key: "delta", name: "Delta rule / TTT-linear: M += β(v − Mk)·kᵀ", color: "var(--accent)" },
  { key: "gated", name: "Gated delta: decay, then delta", color: "var(--c3)" },
];

// write n key-value pairs into each memory, then query every key; returns per-position correctness
function experiment(n, rho, beta, decay, seed) {
  const r = rng(seed), common = unit(Array.from({ length: D }, () => gauss(r)));
  const keys = Array.from({ length: n }, () => unit(Array.from({ length: D }, (_, i) => Math.sqrt(rho) * common[i] + Math.sqrt(1 - rho) * gauss(r) / Math.sqrt(D))));
  const vals = Array.from({ length: n }, () => unit(Array.from({ length: D }, () => gauss(r))));
  const mats = { linear: zeros(), delta: zeros(), gated: zeros() };
  for (let t = 0; t < n; t++) {
    const k = keys[t], v = vals[t];
    addOuter(mats.linear, v, k, 1);
    const pd = matvec(mats.delta, k); addOuter(mats.delta, v.map((x, i) => x - pd[i]), k, beta);
    scale(mats.gated, decay); const pg = matvec(mats.gated, k); addOuter(mats.gated, v.map((x, i) => x - pg[i]), k, beta);
  }
  const read = {
    softmax: (q) => { const w = keys.map((k) => Math.exp(12 * dot(q, k))), z = w.reduce((a, b) => a + b, 0); return Array.from({ length: D }, (_, i) => vals.reduce((s, v, j) => s + (w[j] / z) * v[i], 0)); },
    linear: (q) => matvec(mats.linear, q), delta: (q) => matvec(mats.delta, q), gated: (q) => matvec(mats.gated, q),
  };
  const out = {};
  for (const m of METHODS) out[m.key] = keys.map((k, i) => { const y = read[m.key](k); let best = -Infinity, bi = -1; vals.forEach((v, j) => { const c = dot(y, v); if (c > best) { best = c; bi = j; } }); return bi === i ? 1 : 0; });
  return out;
}
// K variables written repeatedly with new values; query each variable's latest value
function updates(n, beta, decay, seed, K = 8) {
  const r = rng(seed), keys = Array.from({ length: K }, () => unit(Array.from({ length: D }, () => gauss(r))));
  const M = { linear: zeros(), delta: zeros(), gated: zeros() }, all = [], latest = new Array(K).fill(-1), kvs = [];
  for (let t = 0; t < n; t++) {
    const i = Math.floor(r() * K), k = keys[i], v = unit(Array.from({ length: D }, () => gauss(r)));
    all.push(v); latest[i] = all.length - 1; kvs.push([k, v]);
    addOuter(M.linear, v, k, 1);
    const pd = matvec(M.delta, k); addOuter(M.delta, v.map((x, j) => x - pd[j]), k, beta);
    scale(M.gated, decay); const pg = matvec(M.gated, k); addOuter(M.gated, v.map((x, j) => x - pg[j]), k, beta);
  }
  const pick = (y) => { let b = -Infinity, bi = -1; all.forEach((v, j) => { const c = dot(y, v); if (c > b) { b = c; bi = j; } }); return bi; };
  const res = { softmax: 0, linear: 0, delta: 0, gated: 0 }; let q = 0;
  keys.forEach((k, i) => {
    if (latest[i] < 0) return; q++;
    for (const m of ["linear", "delta", "gated"]) res[m] += pick(matvec(M[m], k)) === latest[i];
    const w = kvs.map(([kk]) => Math.exp(12 * dot(k, kk))), z = w.reduce((a, b) => a + b, 0);
    res.softmax += pick(Array.from({ length: D }, (_, d) => kvs.reduce((s2, [, v], j) => s2 + (w[j] / z) * v[d], 0))) === latest[i];
  });
  for (const m in res) res[m] /= q;
  return res;
}
const zeros = () => Array.from({ length: D }, () => new Float64Array(D));
const matvec = (M, x) => M.map((row) => dot(row, x));
const addOuter = (M, a, b, s) => { for (let i = 0; i < D; i++) for (let j = 0; j < D; j++) M[i][j] += s * a[i] * b[j]; };
const scale = (M, s) => { for (const row of M) for (let j = 0; j < D; j++) row[j] *= s; };

export default function (root, { id }) {
  const o = { rho: 0, beta: 0.6, decay: 0.97, n: 24, mode: "updates" };
  const capTitle = h("div.eyebrow");
  const capC = xyChart({ height: 280, xLabel: "writes into memory", yLabel: "fraction recalled correctly", xDomain: [2, 64], yDomain: [0, 1] });
  const posC = xyChart({ width: 460, height: 260, xLabel: "position of the pair in the stream (old → recent)", yLabel: "recalled", yDomain: [0, 1] });
  const st = stats([{ label: "Memory state (every method)" }, { label: "Softmax KV cache at this length" }]);
  function draw() {
    const ns = [2, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64], T = o.mode === "updates" ? 12 : 6;
    const curves = Object.fromEntries(METHODS.map((m) => [m.key, []]));
    for (const n of ns) {
      const acc = Object.fromEntries(METHODS.map((m) => [m.key, 0]));
      for (let s = 0; s < T; s++) {
        if (o.mode === "updates") { const e = updates(n, o.beta, o.decay, 100 + s); for (const m of METHODS) acc[m.key] += e[m.key] / T; }
        else { const e = experiment(n, o.rho, o.beta, o.decay, 100 + s); for (const m of METHODS) acc[m.key] += e[m.key].reduce((a, b) => a + b, 0) / n / T; }
      }
      for (const m of METHODS) curves[m.key].push([n, acc[m.key]]);
    }
    capC.update(METHODS.map((m) => ({ name: m.name, color: m.color, points: curves[m.key], dash: m.dash, dots: !m.dash, r: 3 })), { xDomain: [2, 64], yDomain: [0, 1], marks: o.mode === "updates" ? [] : [{ x: D, label: `d = ${D}` }] });
    capTitle.textContent = o.mode === "updates" ? "8 variables, each overwritten with new values over time. Can the memory return each variable's latest value?" : "Every pair has its own key. Can the memory return each pair's value?";
    // recency profile at the chosen n, averaged over trials and smoothed into 8 bins
    const B = 8, prof = Object.fromEntries(METHODS.map((m) => [m.key, new Array(B).fill(0)])), cnt = new Array(B).fill(0);
    for (let s = 0; s < 12; s++) {
      const e = experiment(o.n, o.rho, o.beta, o.decay, 300 + s);
      for (let i = 0; i < o.n; i++) { const b = Math.min(B - 1, Math.floor((i / o.n) * B)); cnt[b]++; for (const m of METHODS) prof[m.key][b] += e[m.key][i]; }
    }
    posC.update(METHODS.map((m) => ({ name: m.name.split(":")[0].split(" (")[0], color: m.color, dash: m.dash, points: prof[m.key].map((v, b) => [(b + 0.5) / B, v / cnt[b]]) })), { xDomain: [0, 1], yDomain: [0, 1] });
    st.set(0, `${D}×${D} = ${D * D} numbers`); st.set(1, `${2 * o.n * D} numbers (grows)`);
  }

  root.append(
    section(null, prose(`<p>Attention remembers everything by keeping every key and value (the KV cache), which grows with the context. Recurrent models squeeze history into a fixed-size state. The question is <i>how</i> to write into that state. One beautiful answer: treat the state as the <b>weights of a small model</b>, and update it with a step of gradient descent on every token as you read. Learning doesn't stop at training time; it continues inside the forward pass. That's <b>test-time training</b> (TTT), and it unifies a family of fast-memory layers: linear attention, DeltaNet, Gated DeltaNet, TTT layers and Titans.</p>`)),
    section("Write pairs into a 16×16 memory, then try to read them back",
      prose("<p>Each method receives a stream of random (key, value) pairs and must store them in a 16×16 matrix M. Afterwards we query every key and check whether the memory returns the right value. Softmax attention is the reference: it simply keeps all the pairs.</p>"),
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Task", options: [{ value: "updates", label: "Variables that get updated" }, { value: "distinct", label: "Many distinct pairs" }], value: o.mode, onChange: (v) => { o.mode = v; draw(); } }),
          slider({ label: "Key similarity (distinct-pairs task)", min: 0, max: 0.9, step: 0.05, value: o.rho, fmt: (v) => v.toFixed(2), onInput: (v) => { o.rho = v; draw(); } }),
          slider({ label: "Write strength β (delta rules)", min: 0.1, max: 1, step: 0.05, value: o.beta, fmt: (v) => v.toFixed(2), onInput: (v) => { o.beta = v; draw(); } }),
          slider({ label: "Decay per step (gated)", min: 0.8, max: 1, step: 0.005, value: o.decay, fmt: (v) => v.toFixed(3), onInput: (v) => { o.decay = v; draw(); } }),
          slider({ label: "Stream length (bottom chart)", min: 8, max: 64, step: 4, value: o.n, onInput: (v) => { o.n = v; draw(); } })),
        h("div.stage-body", st.el, capTitle, capC.el, h("div.lab", posC.el, h("p.canvas-cap", "Distinct-pairs task: which pairs survive? Linear attention blurs everything roughly equally. The delta rule keeps the most recent ones best. Gating forgets old pairs on purpose to make room.")))),
      callout("tip", "Things to try", "<ul style='margin:0;padding-left:18px'><li><b>Variables that get updated</b> (the default): eight keys are written again and again with new values, like a program reassigning variables. <b>Linear attention</b> can only add, so each key returns a blend of every value it ever had: accuracy falls towards chance as writes pile up. The <b>delta rule</b> first reads what the memory holds for k, then writes only the error (v − Mk), which replaces the old value. It stays near 90% after 64 writes. Even plain softmax attention blends old and new values here; real transformers rely on position information to prefer the latest.</li><li>Switch to <b>many distinct pairs</b> with random keys: now plain linear attention does as well as or better than the delta rule. Random 16-dimensional keys are almost orthogonal, so adding causes little crosstalk, while a strong delta write (β near 1) overwrites directions that other keys share. Lower β to 0.3 and raise key similarity to 0.5: the delta rule pulls ahead again. No update rule wins everywhere; that's why modern layers learn β per token.</li><li>No fixed-size memory matches softmax attention at recalling many <i>distinct</i> items. That trade-off is why hybrid models keep a few attention layers.</li></ul>"),
      predict({
        chapter: id,
        question: "Why is the delta rule called “test-time training”?",
        options: ["It is trained on the test set before deployment", "Its state update is a gradient descent step on a reconstruction loss, computed for each token during inference: the memory learns while it reads", "It only runs at test time", "It uses a test-time dropout"],
        answer: 1,
        explain: "<p>The memory M is a linear model mapping keys to values. Each new token is a training example (k, v), and M takes one gradient step on ½‖Mk − v‖². Replace the linear model with a small MLP and you get TTT-MLP (Sun et al., 2024); add momentum and learned forgetting and you get Titans (Behrouz et al., 2024).</p>",
      }),
    ),
    section("One family, different update rules",
      table(["Layer", "State update for each token", "Interpretation"], [
        ["Linear attention", "M ← M + v kᵀ", "Gradient step on −vᵀMk: always adds, never corrects"],
        ["RetNet / decayed linear attention", "M ← γM + v kᵀ", "Same, with a fixed forgetting rate"],
        ["Mamba-2 / GLA", "M ← α_t M + v kᵀ", "Input-dependent forgetting"],
        ["DeltaNet", "M ← M + β(v − Mk) kᵀ", "One gradient step on ½‖Mk − v‖²: error-correcting"],
        ["Gated DeltaNet", "M ← α_t M + β(v − α_t Mk) kᵀ", "Forget and correct; used in Qwen3-Next's hybrid layers"],
        ["TTT-MLP, Titans", "θ ← θ − η ∇ℓ(f_θ(k), v), with momentum and decay", "The memory is a small neural network trained online"],
      ]),
      stepper([
        { title: "Training the outer model", html: "<p>The outer network (projections that produce k, v, q and the learning rate β) is trained normally, with backpropagation through the inner updates. It learns <i>what</i> to write and <i>how fast</i>.</p>" },
        { title: "Inside the forward pass", html: "<p>For each token the inner memory takes a gradient step. The sequence is processed chunk by chunk so the updates can use matrix multiplies (the same trick as Mamba-2 and FlashLinearAttention).</p>" },
        { title: "Reading", html: "<p>The output for a token is the memory applied to its query: y = M q. The memory has “learned” the context.</p>" },
      ]),
    ),
    section("Test-time training in the wider sense",
      prose("<p>The same idea works at a larger grain: when a model faces a new task with a few examples, briefly <b>fine-tune it on those examples</b> before answering. On the ARC abstract-reasoning benchmark, test-time fine-tuning of an 8B model (with augmented versions of each puzzle's examples) multiplied accuracy several times over the same model without it (Akyürek et al., 2024). It's expensive per query, but it turns few-shot examples into actual learning rather than just context.</p>"),
    ),
    section("In code",
      code({
        "Delta rule (sequential)": `def delta_rule(q, k, v, beta):
    # q, k, v: (T, d) with k L2-normalised; beta: (T,) in (0, 1)
    M = torch.zeros(v.shape[1], k.shape[1])
    out = []
    for t in range(q.shape[0]):
        pred = M @ k[t]                                      # what memory currently says for this key
        M = M + beta[t] * torch.outer(v[t] - pred, k[t])     # write only the error
        out.append(M @ q[t])
    return torch.stack(out)`,
        "Test-time fine-tuning": `def solve_with_ttt(model, demos, query, steps=20, lr=1e-4):
    tuned = copy.deepcopy(model)               # or attach a fresh LoRA adapter
    opt = torch.optim.AdamW(tuned.parameters(), lr=lr)
    for _ in range(steps):
        for x, y in augment(demos):            # e.g. rotations, colour permutations
            loss = tuned.loss(x, y); opt.zero_grad(); loss.backward(); opt.step()
    return tuned.generate(query)`,
      }, "Project 16 compares these memories on associative recall and trains a small DeltaNet layer."),
    ),
  );
  draw();
}
