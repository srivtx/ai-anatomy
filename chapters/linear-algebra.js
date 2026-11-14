import { h } from "../lib/core.js";
import { slider, stats, predict, code, callout, section, prose, table, button } from "../lib/widgets.js";
import { plane } from "../lib/plot.js";

const PRESETS = {
  Identity: [1, 0, 0, 1], "Scale ×1.5": [1.5, 0, 0, 1.5], "Stretch x": [2, 0, 0, 0.6], "Rotate 30°": [Math.cos(Math.PI / 6), -Math.sin(Math.PI / 6), Math.sin(Math.PI / 6), Math.cos(Math.PI / 6)],
  Shear: [1, 1, 0, 1], Reflect: [-1, 0, 0, 1], "Collapse (rank 1)": [1, 0.5, 2, 1],
};
// a little face so you can see orientation and flips
const FACE = [
  ...Array.from({ length: 40 }, (_, i) => [0.5 + 0.38 * Math.cos((i / 40) * 2 * Math.PI), 0.5 + 0.38 * Math.sin((i / 40) * 2 * Math.PI)]),
  [0.36, 0.6], [0.37, 0.62], [0.64, 0.6], [0.65, 0.62],
  ...Array.from({ length: 12 }, (_, i) => [0.32 + (i / 11) * 0.36, 0.38 - 0.1 * Math.sin((i / 11) * Math.PI)]),
];

function eig([a, b, c, d]) {
  const tr = a + d, det = a * d - b * c, disc = tr * tr - 4 * det;
  if (disc < -1e-9) return { complex: true, re: tr / 2, im: Math.sqrt(-disc) / 2 };
  const s = Math.sqrt(Math.max(0, disc)), l1 = (tr + s) / 2, l2 = (tr - s) / 2;
  const vec = (l) => (Math.abs(b) > 1e-9 ? [b, l - a] : Math.abs(c) > 1e-9 ? [l - d, c] : l === a ? [1, 0] : [0, 1]);
  return { complex: false, vals: [l1, l2], vecs: [vec(l1), vec(l2)] };
}

export default function (root, { id }) {
  let M = [1, 0.5, 0, 1];
  const pl = plane({ width: 460, height: 460, xr: [-3, 3], yr: [-3, 3], label: "The plane before and after the matrix" });
  const st = stats([{ label: "Determinant (area scale)" }, { label: "Rank" }, { label: "Eigenvalues" }]);
  const sl = ["a", "b", "c", "d"].map((k, i) => slider({ label: `${k}  (${["row 1, col 1", "row 1, col 2", "row 2, col 1", "row 2, col 2"][i]})`, min: -2.5, max: 2.5, step: 0.05, value: M[i], fmt: (v) => v.toFixed(2), onInput: (v) => { M[i] = v; draw(); } }));
  const T = (x, y) => [M[0] * x + M[1] * y, M[2] * x + M[3] * y];

  function draw() {
    const c = pl.colors;
    pl.clear();
    for (let k = -6; k <= 6; k++) { pl.line(pl.X(k), 0, pl.X(k), pl.H, { color: c.sunken, lw: 1 }); pl.line(0, pl.Y(k), pl.W, pl.Y(k), { color: c.sunken, lw: 1 }); }
    // transformed grid
    for (let k = -6; k <= 6; k++) {
      const [ax, ay] = T(k, -6), [bx, by] = T(k, 6), [cx, cy] = T(-6, k), [dx, dy] = T(6, k);
      pl.line(pl.X(ax), pl.Y(ay), pl.X(bx), pl.Y(by), { color: c.line, lw: 1 });
      pl.line(pl.X(cx), pl.Y(cy), pl.X(dx), pl.Y(dy), { color: c.line, lw: 1 });
    }
    // unit square -> parallelogram
    const sq = [[0, 0], [1, 0], [1, 1], [0, 1]].map(([x, y]) => T(x, y));
    const ctx = pl.ctx;
    ctx.beginPath(); sq.forEach(([x, y], i) => (i ? ctx.lineTo(pl.X(x), pl.Y(y)) : ctx.moveTo(pl.X(x), pl.Y(y)))); ctx.closePath();
    ctx.fillStyle = c["accent"] + "33"; ctx.fill();
    for (const [x, y] of FACE) { const [u, v] = T(x, y); pl.dot(pl.X(u), pl.Y(v), 2.2, c.ink); }
    const e = eig(M);
    if (!e.complex) e.vecs.forEach((v) => { const n = Math.hypot(...v) || 1; pl.line(pl.X((-v[0] / n) * 4), pl.Y((-v[1] / n) * 4), pl.X((v[0] / n) * 4), pl.Y((v[1] / n) * 4), { color: c.c4, lw: 1.5, dash: [6, 5] }); });
    const arrow = (x, y, col, label) => { pl.line(pl.X(0), pl.Y(0), pl.X(x), pl.Y(y), { color: col, lw: 3 }); pl.dot(pl.X(x), pl.Y(y), 4.5, col); pl.text(label, pl.X(x) + 8, pl.Y(y) - 8, { color: col, size: 13, weight: 700 }); };
    arrow(M[0], M[2], c.c1, "î → column 1");
    arrow(M[1], M[3], c.c2, "ĵ → column 2");
    const det = M[0] * M[3] - M[1] * M[2];
    st.set(0, det.toFixed(2), Math.abs(det) < 0.02 ? "bad" : det < 0 ? "warn" : "");
    const rank = Math.abs(det) > 1e-3 ? 2 : M.some((v) => Math.abs(v) > 1e-3) ? 1 : 0;
    st.set(1, String(rank), rank < 2 ? "bad" : "");
    st.set(2, e.complex ? `${e.re.toFixed(2)} ± ${e.im.toFixed(2)}i (rotates)` : e.vals.map((v) => v.toFixed(2)).join(", "));
  }
  const setM = (v) => { M = [...v]; sl.forEach((s2, i) => (s2.value = M[i])); draw(); };

  // dot product
  let ang = 50, la = 2, lb = 1.5;
  const dp = plane({ width: 360, height: 260, xr: [-2.6, 2.6], yr: [-0.8, 2.6], label: "Two vectors and their dot product" });
  const dst = stats([{ label: "a · b" }, { label: "cosine similarity" }]);
  function drawDot() {
    const c = dp.colors, t = (ang * Math.PI) / 180, b = [lb * Math.cos(t), lb * Math.sin(t)], a = [la, 0];
    dp.clear(); dp.axes();
    const proj = b[0];
    dp.line(dp.X(b[0]), dp.Y(b[1]), dp.X(proj), dp.Y(0), { color: c["ink-3"], dash: [4, 4] });
    dp.line(dp.X(0), dp.Y(0), dp.X(proj), dp.Y(0), { color: c.c4, lw: 6 });
    for (const [v, col, l] of [[a, c.c1, "a"], [b, c.c2, "b"]]) { dp.line(dp.X(0), dp.Y(0), dp.X(v[0]), dp.Y(v[1]), { color: col, lw: 3 }); dp.text(l, dp.X(v[0]) + 6, dp.Y(v[1]) - 8, { color: col, size: 14, weight: 700 }); }
    const dot = la * lb * Math.cos(t);
    dst.set(0, dot.toFixed(2), dot < -0.01 ? "bad" : dot > 0.01 ? "good" : "");
    dst.set(1, Math.cos(t).toFixed(2));
  }

  root.append(
    section(null, prose(`<p>A neural network is mostly matrix multiplication. Every layer takes a vector in and multiplies it by a matrix of weights. So it's worth seeing what a matrix actually <i>does</i>: it's a machine that moves every point of space at once, keeping straight lines straight and the origin fixed.</p>
<p>The trick to reading a 2×2 matrix: its <b>columns are where the two basis arrows land</b>. Column 1 is where î = (1, 0) goes, column 2 is where ĵ = (0, 1) goes. Everything else follows, because every point is some mix of î and ĵ.</p>`)),
    section("Drive a matrix",
      h("div.stage",
        h("div.stage-controls", sl, h("div.row", Object.entries(PRESETS).map(([k, v]) => button(k, () => setM(v))))),
        h("div.stage-body", st.el, h("div.lab", h("div.canvas-box", pl.el),
          h("div.prose", { html: `<p><b>Shaded parallelogram</b>: where the unit square lands. Its area is |determinant|.</p><p><b>Dashed yellow lines</b>: eigenvector directions, the lines the matrix only stretches and never turns. The eigenvalue is the stretch factor.</p><p><b>The face</b> shows orientation. A negative determinant flips it like a mirror.</p><p>A matrix with rank 1 squashes the whole plane onto a line. Information is lost: you can't undo it. That's what <b>low rank</b> means, and LoRA exploits it.</p>` })))),
      callout("tip", "Things to try", "<ul style='margin:0;padding-left:18px'><li>Rotate 30°: the eigenvalues turn complex. No direction survives a rotation unturned.</li><li>Collapse (rank 1): determinant 0, and the face is flattened onto a line.</li><li>Shear: one eigenvector direction (along x) with eigenvalue 1, twice. Points slide but the x-axis stays put.</li></ul>"),
      predict({
        chapter: id,
        question: "A layer's weight matrix has determinant 0. What does that tell you about the layer?",
        options: ["It outputs all zeros", "Some different inputs produce identical outputs: information is lost", "It's the identity", "Its gradients are zero"],
        answer: 1,
        explain: "<p>Determinant 0 means the map squashes space into a lower dimension, so it's not invertible: many inputs land on the same output. Big networks are full of approximately low-rank matrices, and that's useful: it's why you can compress them, and why LoRA can fine-tune with tiny rank-8 updates.</p>",
      }),
    ),
    section("The dot product: how similar are two vectors?",
      prose("<p>a · b = a₁b₁ + a₂b₂ = |a| |b| cos θ. It's the length of b's shadow on a (yellow), times |a|. Positive when they point the same way, zero at right angles, negative when opposed. Divide by both lengths and you get <b>cosine similarity</b>: the score behind embeddings, search and every attention head.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Angle between a and b", min: 0, max: 180, value: ang, fmt: (v) => `${v}°`, onInput: (v) => { ang = v; drawDot(); } }),
          slider({ label: "|a|", min: 0.2, max: 2.4, step: 0.1, value: la, fmt: (v) => v.toFixed(1), onInput: (v) => { la = v; drawDot(); } }),
          slider({ label: "|b|", min: 0.2, max: 2.4, step: 0.1, value: lb, fmt: (v) => v.toFixed(1), onInput: (v) => { lb = v; drawDot(); } })),
        h("div.stage-body", dst.el, h("div.canvas-box", { style: { maxWidth: "520px" } }, dp.el))),
    ),
    section("Where this shows up",
      table(["Idea", "In ML"], [
        ["Matrix × vector", "A dense layer: y = W x + b"],
        ["Matrix × matrix", "A whole batch through a layer at once; GPUs are built for this"],
        ["Dot product", "Attention scores q·k, embedding similarity, logits = h · W_vocab"],
        ["Rank / low rank", "LoRA (W + BA with rank 8), compression, MLA's compressed KV cache"],
        ["Eigenvalues", "Whether repeated multiplication explodes or vanishes (RNNs, deep nets, SSMs)"],
        ["SVD", "PCA, model compression, analysing what a layer has learned"],
        ["Orthogonal matrices", "Rotations that preserve length: RoPE, the Muon optimizer"],
      ]),
      prose("<p>Cost to remember: multiplying an (m × k) by a (k × n) matrix takes about <b>2·m·k·n</b> floating-point operations. A 4096 × 4096 layer applied to 2,048 tokens is 69 GFLOPs. Count those up across a model and you get the compute bills in the scaling-laws chapter.</p>"),
    ),
    section("In code",
      code({
        NumPy: `import numpy as np
M = np.array([[1.0, 0.5],
              [0.0, 1.0]])         # a shear
points = np.random.randn(2, 100)    # 100 points, one per column
moved = M @ points                  # every point at once

np.linalg.det(M)                    # area scale: 1.0
vals, vecs = np.linalg.eig(M)       # stretch factors and directions
U, S, Vt = np.linalg.svd(M)         # rotate, scale, rotate
low_rank = (U[:, :1] * S[:1]) @ Vt[:1]   # best rank-1 approximation`,
      }),
    ),
  );
  pl.paint(draw);
  dp.paint(drawDot);
}
