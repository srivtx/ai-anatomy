import { h, s } from "../lib/core.js";
import { goal, whyChain, sticker } from "../lib/why.js";
import { walk } from "../lib/walks.js";
import { slider, segmented, stats, predict, code, callout, section, prose, table, stepper, bars } from "../lib/widgets.js";
import { fmtSI, fmtBytes } from "../lib/plot.js";

const PRESETS = {
  "GPT-2 small": { d: 768, L: 12, h: 12, kv: 12, ff: 3072, vocab: 50257, ctx: 1024, mlp: "gelu", tied: true, pos: true },
  "Llama 3 8B": { d: 4096, L: 32, h: 32, kv: 8, ff: 14336, vocab: 128256, ctx: 8192, mlp: "swiglu", tied: false, pos: false },
  "Llama 3 70B": { d: 8192, L: 80, h: 64, kv: 8, ff: 28672, vocab: 128256, ctx: 8192, mlp: "swiglu", tied: false, pos: false },
  "GPT-3 175B": { d: 12288, L: 96, h: 96, kv: 96, ff: 49152, vocab: 50257, ctx: 2048, mlp: "gelu", tied: true, pos: true },
};

function count(c) {
  const hd = c.d / c.h;
  const attn = c.d * c.d * 2 + 2 * c.d * c.kv * hd;            // Wq, Wo + Wk, Wv (fewer with grouped-query attention)
  const mlp = (c.mlp === "swiglu" ? 3 : 2) * c.d * c.ff;
  const norms = 2 * c.d;
  const emb = c.vocab * c.d, head = c.tied ? 0 : c.vocab * c.d, pos = c.pos ? c.ctx * c.d : 0;
  const total = c.L * (attn + mlp + norms) + emb + head + pos;
  const nonEmb = c.L * (attn + mlp);
  const kvPerTok = 2 * c.L * c.kv * hd;                       // keys and values, every layer
  // forward FLOPs per token: 2 per weight (multiply + add), plus attention over the context
  const flopsTok = 2 * nonEmb + 2 * c.d * c.vocab + 4 * c.L * c.d * c.ctx / 2;
  return { attn: attn * c.L, mlp: mlp * c.L, emb: emb + head + pos, norms: norms * c.L, total, kvPerTok, flopsTok };
}

const LAYERS = [
  ["tokens", 390, "Token IDs"], ["emb", 340, "Embedding: ID → vector"], ["n1", 265, "RMSNorm"], ["attn", 220, "Multi-head attention"], ["add1", 205, "+"],
  ["n2", 140, "RMSNorm"], ["mlp", 95, "MLP (SwiGLU)"], ["add2", 80, "+"], ["head", 8, "Unembed → logits"],
];

export default function (root, { id }) {
  let cfg = { ...PRESETS["Llama 3 8B"] }, bytes = 2, batch = 1, tokens = 15e12;
  const st = stats([{ label: "Parameters" }, { label: "Weights in memory" }, { label: "Forward FLOPs per token" }, { label: "KV cache, full context" }, { label: "Training compute (6·N·D)" }]);
  const brk = h("div");
  function calc() {
    const r = count(cfg);
    st.set(0, fmtSI(r.total)); st.set(1, fmtBytes(r.total * bytes)); st.set(2, fmtSI(r.flopsTok));
    st.set(3, fmtBytes(r.kvPerTok * bytes * cfg.ctx * batch), r.kvPerTok * bytes * cfg.ctx * batch > r.total * bytes ? "warn" : "");
    st.set(4, `${(6 * r.total * tokens).toExponential(1)} FLOPs`);
    brk.replaceChildren(h("div.eyebrow", "Where the parameters are"), bars([
      { label: "MLP blocks", value: r.mlp, color: "var(--c1)" }, { label: "Attention blocks", value: r.attn, color: "var(--c2)" },
      { label: "Embeddings + head", value: r.emb, color: "var(--c3)" }, { label: "Norms", value: r.norms, color: "var(--c4)" },
    ], { max: r.total, fmt: (v) => `${((v / r.total) * 100).toFixed(1)}%` }));
  }
  const sl = {};
  const S = (label, key, min, max, step = 1, fmt) => (sl[key] = slider({ label, min, max, step, value: cfg[key], fmt, onInput: (v) => { cfg[key] = v; if (key === "h") { cfg.kv = Math.min(cfg.kv, v); } calc(); } }));
  const setPreset = (k) => { cfg = { ...PRESETS[k] }; for (const key of Object.keys(sl)) sl[key].value = cfg[key]; calc(); };

  // the block diagram
  const svg = s("svg", { viewBox: "0 0 600 430", role: "img", "aria-label": "A transformer block: residual stream with attention and MLP branches" });
  let active = null;
  const SX = 130; // x of the residual stream
  function drawBlock() {
    svg.replaceChildren(s("line", { x1: SX, x2: SX, y1: 405, y2: 20, style: "stroke:var(--accent);stroke-width:6;opacity:.35" }));
    // the repeated part
    svg.append(s("rect", { x: 14, y: 50, width: 572, height: 268, rx: 14, style: "fill:none;stroke:var(--line);stroke-dasharray:6 5" }), s("text", { x: 580, y: 312, "text-anchor": "end", class: "node-sub", text: "× N layers" }));
    // branch wires: read the stream on the left, write back at the + above
    for (const [readY, outY] of [[265, 205], [140, 80]]) {
      svg.append(s("path", { d: `M${SX},${readY + 15} L330,${readY + 15}`, class: "wire" }));
      svg.append(s("path", { d: `M540,${readY - 30} C585,${readY - 30} 585,${outY + 15} ${SX + 16},${outY + 15}`, class: "wire", style: "stroke:var(--ink-3)" }));
    }
    for (const [k, y, label] of LAYERS) {
      const branch = ["n1", "attn", "n2", "mlp"].includes(k), x = branch ? 330 : 30, w = branch ? 210 : 250;
      if (k.startsWith("add")) { svg.append(s("circle", { cx: SX, cy: y + 15, r: 15, class: `node-box${active === k ? " hot" : ""}`, style: active === k ? "stroke-width:3" : "" }), s("text", { x: SX, y: y + 20, "text-anchor": "middle", class: "node-label", text: "+" })); continue; }
      svg.append(s("rect", { x, y, width: w, height: 30, rx: 8, class: `node-box${active === k ? " hot" : ""}`, style: active === k ? "stroke-width:3" : "" }), s("text", { x: x + 12, y: y + 20, class: "node-label", text: label }));
    }
  }
  const steps = [
    { title: "Tokens become vectors", html: "<p>Each token ID picks a row of the embedding matrix: a vector of size d (4,096 in Llama 3 8B). This vector is the start of the <b>residual stream</b>: a running scratchpad that every layer reads from and adds to.</p>", onEnter: () => { active = "emb"; drawBlock(); } },
    { title: "Normalise, then attend", html: "<p>Each block first normalises a copy of the stream (pre-norm), then runs attention: every position gathers information from earlier positions. Attention is the only place tokens exchange information.</p>", onEnter: () => { active = "attn"; drawBlock(); } },
    { title: "Add it back", html: "<p>The attention output is <b>added</b> to the stream, not substituted for it. The block only writes a correction. This residual design is what lets 80+ layers train.</p>", onEnter: () => { active = "add1"; drawBlock(); } },
    { title: "Think per token: the MLP", html: "<p>Then a two-layer MLP (usually 3–4× wider than d) processes each position independently. Interpretability work suggests MLPs store much of a model's factual knowledge as key-value lookups. They also hold most of the parameters.</p>", onEnter: () => { active = "mlp"; drawBlock(); } },
    { title: "Repeat N times, then read out", html: "<p>Stack N identical blocks (different weights). Finally, project the stream at each position onto the vocabulary: one logit per token, softmaxed into the next-token distribution. Often the unembedding shares weights with the embedding (weight tying).</p>", onEnter: () => { active = "head"; drawBlock(); } },
  ];

  root.append(
    section(null, goal("<p>We have attention (tokens talk) and MLPs (each token thinks). We want to <b>stack many of them</b> and still train the stack on thousands of GPUs, with every token in parallel.</p>", ["a block that can be repeated 100 times", "a clear path for gradients through all of it", "no step that waits for the previous token during training"]),
      prose(`<p>A GPT is a stack of identical <b>transformer blocks</b> between an embedding at the bottom and a readout at the top. Each block has two sub-layers: <b>attention</b> (tokens talk to each other) and an <b>MLP</b> (each token thinks on its own). Both read from and write back into a shared <b>residual stream</b>. That's the whole architecture; the rest is scale.</p>`),
      h("div.vx-row", sticker("tensor", { title: "Mostly matrix multiplies", caption: "nearly all of a transformer's arithmetic is this one operation", px: 4 }), sticker("attn", { title: "Plus attention", caption: "the one place tokens look at each other", px: 3 }))),
    section("Walk through one block",
      h("div.lab", h("div", { style: { overflowX: "auto" } }, h("div", { style: { minWidth: "460px" } }, svg)), stepper(steps))),
    section("Size any model",
      prose("<p>Pick a real configuration or build your own. The formulas are exact for these architectures (biases ignored); try matching the published sizes.</p>"),
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Preset", options: Object.keys(PRESETS).map((k) => ({ value: k, label: k })), value: "Llama 3 8B", onChange: setPreset }),
          S("d_model", "d", 256, 16384, 256), S("Layers", "L", 1, 128), S("Query heads", "h", 1, 128), S("KV heads (GQA)", "kv", 1, 128, 1, (v) => `${v}`),
          S("MLP width", "ff", 512, 65536, 512), S("Vocabulary", "vocab", 1000, 256000, 1000), S("Context", "ctx", 512, 131072, 512),
          segmented({ label: "Precision", options: [{ value: 4, label: "fp32" }, { value: 2, label: "bf16" }, { value: 1, label: "fp8/int8" }, { value: 0.5, label: "int4" }], value: 2, onChange: (v) => { bytes = v; calc(); } }),
          slider({ label: "Sequences in the batch (for the KV cache)", min: 1, max: 64, value: batch, onInput: (v) => { batch = v; calc(); } }),
          slider({ label: "Training tokens", min: 9, max: 13.3, step: 0.1, value: Math.log10(tokens), fmt: (v) => fmtSI(10 ** v), onInput: (v) => { tokens = 10 ** v; calc(); } })),
        h("div.stage-body", st.el, brk)),
      callout("tip", "Things to try", "<ul style='margin:0;padding-left:18px'><li>Llama 3 8B: about 8.0B parameters, and two-thirds of them are in the MLPs.</li><li>Set KV heads equal to query heads (no GQA) on Llama 3 70B and raise the batch to 32: the KV cache becomes bigger than the model. <b>Grouped-query attention</b> shares each key/value head across several query heads to shrink it.</li><li>GPT-2 small is 124M parameters, and almost a third of that is the embedding table.</li></ul>"),
      predict({
        chapter: id,
        question: "Roughly how many FLOPs does a forward pass cost per token, for a model with N parameters (short context)?",
        options: ["N", "2N", "6N", "N²"],
        answer: 1,
        explain: "<p>Each weight is used once per token, in a multiply and an add: 2N. Training adds the backward pass (about 4N), giving the famous <b>6N FLOPs per training token</b>. Attention over long contexts adds a term proportional to layers × d × context, which starts to matter past tens of thousands of tokens.</p>",
      }),
    ),
    section("Three shapes of transformer",
      table(["Shape", "Attention", "Trained to", "Examples"], [
        ["Encoder-only", "Bidirectional: every token sees all", "Fill in masked words", "BERT, embedding models, classifiers"],
        ["Decoder-only", "Causal: only the past", "Predict the next token", "GPT, Claude, Llama, Gemini"],
        ["Encoder-decoder", "Encoder bidirectional; decoder causal + cross-attention to the encoder", "Map an input sequence to an output sequence", "T5, original Transformer, Whisper"],
      ]),
      prose("<p>Decoder-only won for general models because one objective (next token) on all text scales simply, and any task can be phrased as a continuation. Encoders still win for cheap embeddings and classification; encoder-decoders for speech recognition and some translation.</p>"),
    ),
    walk("transformer"),
    section("Why add, not replace",
      whyChain({
        claim: "Each block adds its output to its input (x + attention(x)) instead of replacing it.",
        steps: [
          { q: "Why add instead of replace?", a: "<p>With x + f(x), the gradient of the output with respect to x is 1 + f′(x). The 1 is a direct path: even if f's gradient is tiny, the signal from the loss reaches the bottom layer intact.</p>" },
          { q: "Why would it vanish otherwise?", a: "<p>Through a plain stack, the gradient is a product of one factor per layer. Factors a little below 1 shrink to almost nothing over 100 layers (0.9¹⁰⁰ ≈ 0.00003); a little above 1 and they explode.</p>" },
          { q: "Why normalise before each sub-layer?", a: "<p>Adding outputs block after block makes x grow. Normalising each sub-layer's input keeps it working on vectors of a steady size, so one learning rate suits every depth.</p>" },
        ],
        axiom: { name: "Long products drift to 0 or infinity", html: "<p>Multiply n numbers each a little off 1 and the result is exponentially small or large: (1 ± ε)ⁿ ≈ e^(±εn). Sums don't do that. Residual connections turn the gradient's long product into a sum over paths, and that's what lets depth train.</p>" },
      })),
    section("In code",
      code({
        PyTorch: `import torch, torch.nn as nn, torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, d, n_heads, d_ff):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.w1, self.w3 = nn.Linear(d, d_ff, bias=False), nn.Linear(d, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d, bias=False)
        self.h = n_heads

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, C))          # attention writes to the stream
        y = self.n2(x)
        return x + self.w2(F.silu(self.w1(y)) * self.w3(y))            # SwiGLU MLP writes to the stream`,
      }),
    ),
  );
  drawBlock(); calc();
}
