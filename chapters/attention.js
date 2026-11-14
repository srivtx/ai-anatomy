import { h } from "../lib/core.js";
import { slider, segmented, stats, predict, code, callout, section, prose, table, toggle, bars } from "../lib/widgets.js";
import { plane, heatmap } from "../lib/plot.js";
import { softmax } from "../lib/nn.js";

// 1. a single query over four keys, in 2D so you can see every vector
const KEYS = [
  { w: "cat", k: [1.0, 0.25], v: [0.9, 0.2], c: "c1" },
  { w: "sat", k: [-0.2, 1.0], v: [-0.3, 0.9], c: "c2" },
  { w: "mat", k: [0.85, -0.45], v: [0.6, -0.8], c: "c3" },
  { w: "the", k: [-0.9, -0.3], v: [-0.8, -0.4], c: "c4" },
];

// 2. attention heads on a sentence: patterns like those found in trained models
const SENT = ["The", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "too", "tired", "."];
const HEADS = {
  prev: { label: "Previous-token head", logit: (i, j) => (j === i - 1 ? 5 : j === i ? 1 : 0) },
  coref: { label: "Coreference head", logit: (i, j) => (i === 7 && j === 1 ? 6 : i === 7 && j === 5 ? 3.5 : i === 10 && j === 1 ? 4 : j === i ? 2 : 0) },
  sink: { label: "First-token (sink) head", logit: (i, j) => (j === 0 ? 4 : 0.3 * (j === i)) },
  content: { label: "Content-word head", logit: (i, j) => ([1, 3, 5, 10].includes(j) ? 2.5 : 0) },
};

export default function (root, { id }) {
  let ang = 20, len = 1.5;
  const pl = plane({ width: 380, height: 380, xr: [-1.6, 1.6], yr: [-1.6, 1.6], label: "A query vector, four key vectors and the output" });
  const st = stats([{ label: "Highest weight" }, { label: "Entropy of the weights" }]);
  const wEl = h("div");
  function drawOne() {
    const q = [len * Math.cos((ang * Math.PI) / 180), len * Math.sin((ang * Math.PI) / 180)];
    const scores = KEYS.map((x) => (q[0] * x.k[0] + q[1] * x.k[1]) / Math.sqrt(2));
    const w = softmax(scores);
    const out = KEYS.reduce((a, x, i) => [a[0] + w[i] * x.v[0], a[1] + w[i] * x.v[1]], [0, 0]);
    pl.clear(); pl.axes();
    KEYS.forEach((x, i) => {
      pl.line(pl.X(0), pl.Y(0), pl.X(x.k[0]), pl.Y(x.k[1]), { color: pl.colors[x.c], lw: 1 + w[i] * 6 });
      pl.text(`k: ${x.w}`, pl.X(x.k[0]) + 6, pl.Y(x.k[1]) - 8, { color: pl.colors[x.c], size: 12 });
      pl.dot(pl.X(x.v[0]), pl.Y(x.v[1]), 3 + w[i] * 6, pl.colors[x.c]);
    });
    pl.line(pl.X(0), pl.Y(0), pl.X(q[0]), pl.Y(q[1]), { color: pl.colors.ink, lw: 3 });
    pl.text("query", pl.X(q[0]) + 6, pl.Y(q[1]) + 10, { color: pl.colors.ink, size: 12, weight: 700 });
    pl.dot(pl.X(out[0]), pl.Y(out[1]), 9, pl.colors.ink); pl.dot(pl.X(out[0]), pl.Y(out[1]), 6, pl.colors.surface);
    pl.text("output", pl.X(out[0]) + 10, pl.Y(out[1]), { color: pl.colors.ink, size: 12 });
    const top = w.indexOf(Math.max(...w));
    st.set(0, `${KEYS[top].w} (${(w[top] * 100).toFixed(0)}%)`);
    st.set(1, `${(-w.reduce((a, p) => a + p * Math.log2(p), 0)).toFixed(2)} bits`);
    wEl.replaceChildren(bars(KEYS.map((x, i) => ({ label: `${x.w}: q·k/√d = ${scores[i].toFixed(2)}`, value: w[i] * 100, color: `var(--${x.c})` })), { max: 100, fmt: (v) => `${v.toFixed(1)}%` }));
  }

  let head = "coref", causal = true, focus = 7;
  const hmEl = h("div"), focusEl = h("div.chips", { style: { fontSize: "16px" } });
  function drawHeads() {
    const M = SENT.map((_, i) => {
      const logits = SENT.map((_, j) => (causal && j > i ? -Infinity : HEADS[head].logit(i, j)));
      const fin = logits.map((x) => (x === -Infinity ? -1e9 : x));
      return softmax(fin).map((p, j) => (causal && j > i ? null : p));
    });
    hmEl.replaceChildren(heatmap(M, { rows: SENT, cols: SENT, fmt: (v) => v.toFixed(2), showValues: false, cell: 30, onCell: (i) => { focus = i; drawHeads(); } }));
    focusEl.replaceChildren(h("b", { style: { marginRight: "8px", font: "600 13px var(--body)" } }, `“${SENT[focus]}” attends to:`), ...SENT.map((w, j) => h("span.chip", { class: j === focus ? "chip sel" : "chip", style: { background: M[focus][j] == null ? "transparent" : `color-mix(in srgb, var(--accent) ${Math.round(M[focus][j] * 100)}%, transparent)`, opacity: M[focus][j] == null ? 0.35 : 1 } }, w)));
  }

  // RoPE
  let m = 5, n = 2, theta = 0.4;
  const rp = plane({ width: 340, height: 300, xr: [-1.3, 1.3], yr: [-1.15, 1.15], label: "Query and key rotated by their positions" });
  const rst = stats([{ label: "Score at positions (m, n)" }, { label: "Score at (m + 7, n + 7)" }, { label: "Depends only on" }]);
  const rot = (v, a) => [v[0] * Math.cos(a) - v[1] * Math.sin(a), v[0] * Math.sin(a) + v[1] * Math.cos(a)];
  function drawRope() {
    const q = [0.9, 0.3], k = [0.7, -0.5];
    const score = (a, b) => { const qa = rot(q, a * theta), kb = rot(k, b * theta); return qa[0] * kb[0] + qa[1] * kb[1]; };
    const qm = rot(q, m * theta), kn = rot(k, n * theta);
    rp.clear(); rp.axes();
    for (const [v, col, lab] of [[qm, rp.colors.c1, `q at position ${m}`], [kn, rp.colors.c2, `k at position ${n}`]]) { rp.line(rp.X(0), rp.Y(0), rp.X(v[0]), rp.Y(v[1]), { color: col, lw: 3 }); rp.text(lab, rp.X(v[0]) + 6, rp.Y(v[1]), { color: col, size: 12 }); }
    rst.set(0, score(m, n).toFixed(3)); rst.set(1, score(m + 7, n + 7).toFixed(3), "good"); rst.set(2, `m − n = ${m - n}`);
  }

  root.append(
    section(null, prose(`<p>In “The animal didn't cross the street because <b>it</b> was too tired”, the word <i>it</i> only makes sense once you know it refers to the animal. <b>Attention</b> lets every token look at every other token and pull in what it needs. It's the core of the transformer (Vaswani et al., 2017, “Attention Is All You Need”), and it's three small steps:</p>
<ol><li>Each token makes a <b>query</b> (what am I looking for?), a <b>key</b> (what do I contain?) and a <b>value</b> (what will I hand over?), by multiplying its vector by three learned matrices.</li><li>Score every key against the query with a dot product, divide by √d, and softmax into weights that sum to 1.</li><li>Output the weighted average of the values.</li></ol>`)),
    section("One query, four keys",
      prose("<p>Turn the query and watch the weights follow the keys it points towards. The dots are the values; the ringed dot is the output: a blend of values, weighted by attention. Lengthen the query and the softmax gets sharper, like lowering a temperature.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Query direction", min: -180, max: 180, value: ang, fmt: (v) => `${v}°`, onInput: (v) => { ang = v; drawOne(); } }),
          slider({ label: "Query length (sharpness)", min: 0, max: 6, step: 0.1, value: len, fmt: (v) => v.toFixed(1), onInput: (v) => { len = v; drawOne(); } })),
        h("div.stage-body", st.el, h("div.lab", h("div.canvas-box", pl.el), wEl))),
      predict({
        chapter: id,
        question: "Why divide the scores by √d (the vector size) before the softmax?",
        options: ["To make attention faster", "Dot products of random d-dimensional vectors grow like √d, which would make softmax saturate into one-hot weights with vanishing gradients", "To normalise the values", "It's arbitrary"],
        answer: 1,
        explain: "<p>If each component has variance 1, a dot product of two d-dimensional vectors has variance d. At d = 128, raw scores of ±20 would push softmax to put ~100% on one key, and its gradient would be almost zero everywhere else. Dividing by √d keeps the scores around unit size. Set the query length to 6 above to see the saturated regime.</p>",
      }),
    ),
    section("Heads on a real sentence",
      prose("<p>A transformer layer runs many attention <b>heads</b> in parallel, each with its own query, key and value matrices, so each can specialise. Researchers looking inside trained models find recurring kinds of head (Clark et al. 2019; Olsson et al. 2022). The patterns below are hand-set to show those kinds. Click any row to inspect that token.</p>"),
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Head", options: Object.entries(HEADS).map(([k, v]) => ({ value: k, label: v.label })), value: head, onChange: (v) => { head = v; drawHeads(); } }),
          toggle({ label: "Causal mask (GPT-style: no peeking ahead)", checked: true, onChange: (v) => { causal = v; drawHeads(); } })),
        h("div.stage-body", focusEl, hmEl, h("p.canvas-cap", "Rows: the token doing the looking (query). Columns: the token looked at (key). Darker = more attention."))),
      callout("tip", "What the heads do", "<ul style='margin:0;padding-left:18px'><li><b>Previous token</b>: copies information one step along. Paired with a later head it forms an <b>induction head</b> (“last time A was followed by B, so predict B”), a key mechanism behind in-context learning.</li><li><b>Coreference</b>: <i>it</i> looks back at <i>animal</i>.</li><li><b>Sink</b>: dumps attention on the first token when there's nothing useful to attend to. Real models do this a lot, which is why streaming methods keep the first tokens in the cache.</li><li>Turn the causal mask off: that's an encoder like BERT, where every token sees the whole sentence.</li></ul>"),
    ),
    section("Where am I? Rotary position embeddings",
      prose("<p>Attention by itself ignores order: shuffle the tokens and the weights shuffle with them. Models need position information. <b>RoPE</b> (Su et al., 2021), used by Llama, Qwen and most open models, rotates each query and key by an angle proportional to its position. Because a rotation of m·θ dotted with one of n·θ only depends on the difference, the score depends on <b>relative</b> position.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Query position m", min: 0, max: 20, value: m, onInput: (v) => { m = v; drawRope(); } }),
          slider({ label: "Key position n", min: 0, max: 20, value: n, onInput: (v) => { n = v; drawRope(); } }),
          slider({ label: "Rotation speed θ", min: 0.05, max: 1, step: 0.05, value: theta, fmt: (v) => v.toFixed(2), onInput: (v) => { theta = v; drawRope(); } })),
        h("div.stage-body", rst.el, h("div.canvas-box", { style: { maxWidth: "460px" } }, rp.el))),
      prose("<p>Real RoPE splits each vector into many 2D pairs that rotate at different speeds, from fast (fine position) to very slow (long range). Stretching those speeds is how models extend their context window; see the long-context chapter.</p>"),
    ),
    section("The cost",
      table(["", "Self-attention", "RNN"], [
        ["Compute per layer", "O(n² · d): every pair of tokens", "O(n · d²)"],
        ["Steps that must run in sequence (training)", "1: all positions in parallel", "n"],
        ["Path between two distant tokens", "1 hop", "n hops"],
        ["Memory while generating", "KV cache grows with n", "Fixed-size state"],
      ]),
    ),
    section("In code",
      code({
        PyTorch: `import torch, torch.nn.functional as F

def attention(x, Wq, Wk, Wv, n_heads):
    B, T, C = x.shape
    q, k, v = (x @ W for W in (Wq, Wk, Wv))                         # (B, T, C)
    split = lambda t: t.view(B, T, n_heads, C // n_heads).transpose(1, 2)
    q, k, v = split(q), split(k), split(v)                           # (B, H, T, d)
    scores = q @ k.transpose(-2, -1) / (q.size(-1) ** 0.5)           # (B, H, T, T)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))                 # causal
    weights = scores.softmax(-1)
    out = weights @ v                                                # (B, H, T, d)
    return out.transpose(1, 2).reshape(B, T, C)

# in practice, one fused call (FlashAttention under the hood):
# out = F.scaled_dot_product_attention(q, k, v, is_causal=True)`,
      }, "Project 06 builds a full GPT around this."),
    ),
  );
  pl.paint(drawOne);
  drawHeads();
  rp.paint(drawRope);
}
