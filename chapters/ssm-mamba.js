import { h, rng } from "../lib/core.js";
import { slider, segmented, stats, predict, code, callout, section, prose, table, stepper } from "../lib/widgets.js";
import { xyChart, plane, fmtBytes } from "../lib/plot.js";
import { gauss } from "../lib/nn.js";

const T = 120;

export default function (root, { id }) {
  // ---------- 1. a linear recurrence is a convolution ----------
  let a = 0.9;
  const r0 = rng(2);
  const xin = Array.from({ length: T }, (_, t) => (t === 10 ? 1 : t === 45 ? 0.6 : t >= 70 && t < 85 ? 0.3 : 0) + gauss(r0) * 0.02);
  const rc = xyChart({ height: 220, xLabel: "time step", yLabel: "value", xDomain: [0, T] });
  const kc = xyChart({ width: 460, height: 200, xLabel: "lag k", yLabel: "kernel aᵏ", xDomain: [0, 40], yDomain: [0, 1] });
  function drawRec() {
    let hh = 0; const y = [], yc = [];
    for (let t = 0; t < T; t++) { hh = a * hh + (1 - a) * xin[t]; y.push([t, hh]); }
    // the same output as an explicit convolution with kernel (1 − a)·aᵏ
    for (let t = 0; t < T; t++) { let s = 0; for (let k = 0; k <= t; k++) s += (1 - a) * a ** k * xin[t - k]; yc.push([t, s]); }
    rc.update([
      { name: "Input x", color: "var(--ink-3)", points: xin.map((v, t) => [t, v]) },
      { name: "Recurrence: h = a·h + (1 − a)·x", color: "var(--accent)", points: y },
      { name: "Convolution with the kernel (identical)", color: "var(--c2)", points: yc, dash: true },
    ], { xDomain: [0, T] });
    kc.update([{ name: "kernel", color: "var(--accent)", points: Array.from({ length: 41 }, (_, k) => [k, a ** k]) }], { xDomain: [0, 40], yDomain: [0, 1] });
  }

  // ---------- 2. selectivity: remember the last marked value, ignore filler ----------
  let la = 0.85;
  const r1 = rng(9);
  const stream = Array.from({ length: T }, (_, t) => { const imp = t % 17 === 3; return { v: imp ? Math.round((r1() * 2 - 1) * 10) / 10 : (r1() * 2 - 1) * 0.9, imp }; });
  const sc = xyChart({ height: 260, xLabel: "time step", yLabel: "value", xDomain: [0, T], yDomain: [-1.1, 1.1] });
  const sst = stats([{ label: "Error, fixed decay (LTI, like S4)" }, { label: "Error, selective (Mamba)" }]);
  function drawSel() {
    let hL = 0, hS = 0, eL = 0, eS = 0, target = 0;
    const L = [], S = [], tgt = [], marks = [];
    stream.forEach(({ v, imp }, t) => {
      if (imp) { target = v; marks.push([t, v]); }
      hL = la * hL + (1 - la) * v;           // the same decay for every input
      const dt = imp ? 1 : 0;                // Mamba: the step size Δ depends on the input. Δ→0 keeps the state; Δ large overwrites it
      const A = Math.exp(-dt * 8);
      hS = A * hS + (1 - A) * v;
      L.push([t, hL]); S.push([t, hS]); tgt.push([t, target]);
      if (t > 3) { eL += (hL - target) ** 2; eS += (hS - target) ** 2; }
    });
    sc.update([
      { name: "Target: the last marked value", color: "var(--ink-3)", points: tgt },
      { name: "Fixed decay", color: "var(--c2)", points: L },
      { name: "Selective", color: "var(--accent)", points: S },
      { name: "Marked inputs", color: "var(--good)", points: marks, dots: true, line: false, r: 4 },
    ], { xDomain: [0, T], yDomain: [-1.1, 1.1] });
    sst.set(0, (eL / (T - 4)).toFixed(3), "bad"); sst.set(1, (eS / (T - 4)).toFixed(4), "good");
  }

  // ---------- 3. memory at inference ----------
  let n = 32768;
  const mc = xyChart({ height: 260, xLog: true, yLog: true, xLabel: "context length (tokens)", yLabel: "inference memory per sequence", fmtY: fmtBytes });
  const mst = stats([{ label: "Transformer KV cache" }, { label: "Mamba state" }, { label: "Hybrid (1 attention layer in 8)" }]);
  // an 8B-class model: 32 layers, model width 4096; KV with 8 grouped KV heads of 128; Mamba: expand 2, state 16, bf16
  const kv = (len) => 2 * 32 * 8 * 128 * len * 2, ssm = 32 * (2 * 4096) * 16 * 2 + 32 * 2 * 4096 * 4 * 2, hyb = (len) => kv(len) / 8 + ssm * 7 / 8;
  function drawMem() {
    const xs = Array.from({ length: 41 }, (_, i) => 2 ** (8 + i * 0.25));
    mc.update([
      { name: "Attention (KV cache)", color: "var(--c2)", points: xs.map((x) => [x, kv(x)]) },
      { name: "Hybrid", color: "var(--c4)", points: xs.map((x) => [x, hyb(x)]) },
      { name: "SSM (fixed state)", color: "var(--accent)", points: xs.map((x) => [x, ssm]) },
    ], { yDomain: [3e6, 3e10], marks: [{ x: n, label: "" }] });
    mst.set(0, fmtBytes(kv(n))); mst.set(1, fmtBytes(ssm), "good"); mst.set(2, fmtBytes(hyb(n)));
  }

  // ---------- 4. Vision Mamba scan orders ----------
  const G = 8;
  let order = "row";
  const vp = plane({ width: 360, height: 360, xr: [0, G], yr: [0, G], label: "Scan order over image patches" });
  const vst = stats([{ label: "Vertical neighbours: median distance in the sequence" }, { label: "Horizontal neighbours" }]);
  const orders = {
    row: () => [...Array(G * G).keys()].map((i) => [i % G, Math.floor(i / G)]),
    snake: () => [...Array(G * G).keys()].map((i) => { const y = Math.floor(i / G), x = i % G; return [y % 2 ? G - 1 - x : x, y]; }),
    col: () => [...Array(G * G).keys()].map((i) => [Math.floor(i / G), i % G]),
    hilbert: () => {
      const out = [];
      for (let d = 0; d < G * G; d++) {
        let x = 0, y = 0, t = d;
        for (let s = 1; s < G; s *= 2) {
          const rx = 1 & (t / 2), ry = 1 & (t ^ rx);
          if (ry === 0) { if (rx === 1) { x = s - 1 - x; y = s - 1 - y; } [x, y] = [y, x]; }
          x += s * rx; y += s * ry; t = Math.floor(t / 4);
        }
        out.push([x, y]);
      }
      return out;
    },
  };
  function drawScan() {
    const seq = orders[order](), pos = new Map(seq.map((p, i) => [p.join(), i]));
    vp.paint(() => {
      vp.clear();
      for (let i = 0; i <= G; i++) { vp.line(vp.X(i), vp.Y(0), vp.X(i), vp.Y(G), { color: vp.colors.line, lw: 1 }); vp.line(vp.X(0), vp.Y(i), vp.X(G), vp.Y(i), { color: vp.colors.line, lw: 1 }); }
      const ctx = vp.ctx;
      for (let i = 1; i < seq.length; i++) {
        const [x0, y0] = seq[i - 1], [x1, y1] = seq[i], t = i / seq.length;
        ctx.strokeStyle = t < 0.5 ? vp.colors.c2 : vp.colors.accent;
        ctx.globalAlpha = 0.35 + 0.65 * Math.abs(t - 0.5) * 2;
        ctx.lineWidth = 2.5; ctx.beginPath(); ctx.moveTo(vp.X(x0 + 0.5), vp.Y(G - y0 - 0.5)); ctx.lineTo(vp.X(x1 + 0.5), vp.Y(G - y1 - 0.5)); ctx.stroke();
      }
      ctx.globalAlpha = 1;
      vp.dot(vp.X(seq[0][0] + 0.5), vp.Y(G - seq[0][1] - 0.5), 6, vp.colors.good);
      vp.text("start", vp.X(seq[0][0] + 0.5) + 8, vp.Y(G - seq[0][1] - 0.5) - 8, { size: 11, color: vp.colors["ink-2"] });
    });
    const dv = [], dh = [];
    for (let y = 0; y < G; y++) for (let x = 0; x < G; x++) {
      if (y + 1 < G) dv.push(Math.abs(pos.get(`${x},${y}`) - pos.get(`${x},${y + 1}`)));
      if (x + 1 < G) dh.push(Math.abs(pos.get(`${x},${y}`) - pos.get(`${x + 1},${y}`)));
    }
    const med = (v) => v.sort((p, q) => p - q)[Math.floor(v.length / 2)], near = (v) => v.filter((d) => d <= 3).length / v.length;
    vst.set(0, `${med(dv)} steps · ${Math.round(near(dv) * 100)}% within 3`, near(dv) < 0.3 ? "bad" : "good");
    vst.set(1, `${med(dh)} steps · ${Math.round(near(dh) * 100)}% within 3`, near(dh) < 0.3 ? "bad" : "good");
  }

  // parallel scan worked example
  const as = [0.5, 0.9, 0.8, 0.5, 0.9, 0.7, 0.6, 0.9], bs = [1, 0.2, 0.4, 0.1, 0.8, 0.3, 0.5, 0.2];
  const fmt = ([A, B]) => `(${A.toFixed(2)}, ${B.toFixed(2)})`;
  const levels = [as.map((x, i) => [x, bs[i]])];
  for (let s = 1; s < 8; s *= 2) { const prev = levels.at(-1); levels.push(prev.map((p, i) => (i >= s ? [prev[i - s][0] * p[0], p[0] * prev[i - s][1] + p[1]] : p))); }
  const scanTable = (lv) => table(["t", ...as.map((_, i) => String(i + 1))], [["(A, B)", ...levels[lv].map(fmt)], ["h_t = B (h₀ = 0)", ...levels[lv].map((p, i) => (lv === 3 || i < 2 ** lv ? p[1].toFixed(3) : "…"))]], { boldFirst: true });

  root.append(
    section(null, prose(`<p>Attention compares every token with every other: cost grows with the square of the length, and the KV cache grows with every token generated. <b>State space models</b> go back to the idea of an RNN, a fixed-size state updated once per token, but with a structure that trains in parallel like a transformer. <b>Mamba</b> (Gu and Dao, 2023) added the missing ingredient, <b>selectivity</b>, and made SSMs competitive with transformers on language at small and medium scale. Today they're mostly used in <b>hybrids</b> that keep a few attention layers.</p>`)),
    section("A linear recurrence is a convolution",
      prose("<p>The simplest SSM: h<sub>t</sub> = a·h<sub>t−1</sub> + (1 − a)·x<sub>t</sub>. Because it's linear, you can unroll it: the output is the input convolved with a kernel that decays as a<sup>k</sup>. That's the trick behind S4 (Gu et al., 2021): run it as a <b>convolution</b> during training (parallel over the whole sequence) and as a <b>recurrence</b> during generation (constant work per token). Real SSMs use thousands of such channels, each with its own decay, in vector-and-matrix form.</p>"),
      h("div.stage",
        h("div.stage-controls", slider({ label: "Decay a", min: 0.5, max: 0.99, step: 0.01, value: a, fmt: (v) => v.toFixed(2), onInput: (v) => { a = v; drawRec(); } })),
        h("div.stage-body", rc.el, h("div.lab", kc.el, h("p.canvas-cap", "a near 1 means long memory: the kernel decays slowly and the output is smooth and laggy. a small means short memory: it follows the input closely and forgets quickly. The two curves above overlap exactly.")))),
    ),
    section("Selectivity: decide what to remember",
      prose("<p>A fixed decay treats every input the same. That can't do a simple task: <i>remember the last marked value and ignore the filler</i>. Mamba makes the step size Δ (and the input and output projections) <b>functions of the current input</b>. For filler, Δ ≈ 0, so the state is left untouched. For a marked token, Δ is large, so the state is overwritten. It's a gate, like an LSTM's, but still linear in h, so it can be computed with a parallel scan.</p>"),
      h("div.stage",
        h("div.stage-controls", slider({ label: "Fixed decay of the LTI model", min: 0.3, max: 0.995, step: 0.005, value: la, fmt: (v) => v.toFixed(3), onInput: (v) => { la = v; drawSel(); } })),
        h("div.stage-body", sst.el, sc.el)),
      callout("tip", "Things to try", "<p>No setting of the fixed decay works. Small decay: the state is dominated by the latest filler. Decay near 1: the filler averages out, but so does every marked value, and it reacts too slowly. Only an input-dependent decay solves both. This is the <b>selective copying</b> task from the Mamba paper.</p>"),
      prose("<p>The price: with input-dependent coefficients, the recurrence is no longer a fixed convolution. Mamba instead uses a <b>parallel scan</b>. The update h<sub>t</sub> = A<sub>t</sub>·h<sub>t−1</sub> + B<sub>t</sub> composes associatively: (A₁, B₁) then (A₂, B₂) equals (A₁A₂, A₂B₁ + B₂). Associativity means you can combine pairs, then pairs of pairs, in log₂(n) parallel steps:</p>"),
      stepper(levels.map((_, lv) => ({ title: lv === 0 ? "Inputs: one (A, B) per step" : `Level ${lv}: combine each element with the one ${2 ** (lv - 1)} back`, html: `<div style="overflow-x:auto">${scanTable(lv).outerHTML}</div>` }))),
      h("p.canvas-cap", "After log₂(8) = 3 levels every position holds the composition of everything before it, so B is the state h_t. Mamba's kernel does this in GPU SRAM without ever writing the full expanded state to HBM."),
    ),
    section("Constant memory at inference",
      prose("<p>Generation cost is where SSMs shine. A transformer must keep keys and values for every past token. An SSM keeps a fixed-size state, however long the context.</p>"),
      h("div.stage",
          h("div.stage-controls", slider({ label: "Context length", min: 10, max: 20, step: 0.5, value: 15, fmt: (v) => `${Math.round(2 ** v).toLocaleString()} tokens`, onInput: (v) => { n = Math.round(2 ** v); drawMem(); } })),
          h("div.stage-body", mst.el, mc.el)),
      h("p.canvas-cap", "An 8B-class model, 32 layers, bf16; KV cache with 8 grouped KV heads of 128; Mamba with expansion 2, state size 16 and a short conv. Per sequence: multiply by the batch size."),
      predict({
        chapter: id,
        question: "A pure SSM is asked to copy back a random 2,000-token phone book it read earlier. What's the likely problem?",
        options: ["None, its state is unbounded", "Its fixed-size state can't hold 2,000 arbitrary tokens exactly, so recall fails where attention, which keeps every token, succeeds", "It's too slow", "It can't read numbers"],
        answer: 1,
        explain: "<p>Compression is the trade-off. Jelassi et al. (2024, “Repeat After Me”) showed transformers learn to copy long strings far more easily than SSMs of similar size. That's the main reason production models use <b>hybrids</b>: mostly SSM or linear-attention layers for cheap long context, plus a few full attention layers for exact recall.</p>",
      }),
    ),
    section("Vision Mamba: in which order do you read an image?",
      prose("<p>An SSM reads a sequence in order, but an image has no natural order. Vision Mamba (Vim, Zhu et al., 2024) scans the patches forwards and backwards; VMamba (Liu et al., 2024) scans in four directions and merges the results. The scan order decides which patches are close in the sequence, and so which ones the state connects easily.</p>"),
      h("div.stage",
        h("div.stage-controls", segmented({ label: "Scan order", options: [{ value: "row", label: "Row by row" }, { value: "snake", label: "Snake" }, { value: "col", label: "Column by column" }, { value: "hilbert", label: "Hilbert curve" }], value: order, onChange: (v) => { order = v; drawScan(); } })),
        h("div.stage-body", vst.el, h("div.canvas-box", { style: { maxWidth: "380px" } }, vp.el), h("p.canvas-cap", "Orange: first half of the sequence, violet: second half. Lower numbers mean neighbours in the image stay near each other in the sequence."))),
      callout("note", "Why several scans", "<p>Row order keeps horizontal neighbours adjacent but puts every vertical pair 8 steps apart (G steps for a G×G grid; 14 for a 224-pixel image with 16-pixel patches). Column order does the reverse. The Hilbert curve keeps most neighbours close in both directions, at the cost of a few pairs that land far apart where its quadrants meet. Multi-direction scanning gets the best of each and also lets every patch see context from all sides, which a single causal scan can't.</p>"),
    ),
    section("The family",
      table(["Model", "Idea"], [
        ["S4 (2021)", "Structured state matrices initialised to remember history well (HiPPO); trains as a long convolution"],
        ["Mamba (2023)", "Input-dependent Δ, B, C; hardware-aware parallel scan"],
        ["Mamba-2 (2024)", "“State space duality”: a restricted SSM is a form of linear attention, so it can use matrix multiplies and tensor cores"],
        ["RWKV, RetNet, GLA, DeltaNet", "Linear-attention RNNs with decays or gates; trained in parallel, run recurrently"],
        ["Griffin / Hawk (2024)", "Gated linear recurrences, mixed with local attention in Griffin"],
        ["Jamba, Zamba, Samba, Nemotron-H", "Hybrids: mostly recurrent layers, a few attention layers (Jamba: 1 attention in 8, plus MoE)"],
        ["xLSTM (2024)", "The LSTM revisited with exponential gating and matrix memory"],
      ]),
    ),
    section("In code",
      code({
        "Selective scan (reference)": `def selective_scan(x, delta, A, B, C):
    # x, delta: (L, D)   A: (D, N)   B, C: (L, N)   -- a sequential reference, not the fast kernel
    h = torch.zeros(x.shape[1], A.shape[1])
    ys = []
    for t in range(x.shape[0]):
        dA = torch.exp(delta[t, :, None] * A)              # (D, N): input-dependent decay
        dB = delta[t, :, None] * B[t][None, :]             # (D, N)
        h = dA * h + dB * x[t, :, None]                    # state update
        ys.append(h @ C[t])                                # (D,)
    return torch.stack(ys)`,
      }, "Project 15 implements this, checks it against a parallel scan, and trains a small Mamba on a selective copying task."),
    ),
  );
  drawRec(); drawSel(); drawMem(); drawScan();
}
