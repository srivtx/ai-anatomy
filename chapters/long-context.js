import { h } from "../lib/core.js";
import { slider, segmented, stats, predict, code, callout, section, prose, table } from "../lib/widgets.js";
import { xyChart, heatmap, fmtSI } from "../lib/plot.js";

const N = 16;
const MASKS = {
  full: { label: "Full causal", m: (i, j) => j <= i },
  window: { label: "Sliding window (4)", m: (i, j) => j <= i && i - j < 4 },
  sink: { label: "Sink + window", m: (i, j) => j <= i && (i - j < 4 || j < 2) },
  strided: { label: "Strided sparse", m: (i, j) => j <= i && (i - j < 3 || j % 4 === 3) },
  global: { label: "Global tokens", m: (i, j) => j <= i && (i - j < 3 || j === 0 || j === 8 || i === 8) },
};

export default function (root, { id }) {
  // 1. where the FLOPs go as context grows (Llama-3-8B-like shapes)
  const cost = xyChart({ height: 240, xLog: true, yLog: true, xLabel: "context length (tokens)", yLabel: "FLOPs per generated token" });
  (function () {
    const d = 4096, L = 32, dense = 2 * 7e9, pts = [], att = [];
    for (let e = 10; e <= 20; e += 0.5) { const n = 2 ** e; pts.push([n, dense]); att.push([n, 4 * L * d * n]); }
    cost.update([{ name: "Weights (MLP + projections)", color: "var(--c1)", points: pts }, { name: "Attention over the context", color: "var(--c2)", points: att }], { marks: [{ x: 26700, label: "attention takes over (~27k)" }] });
  })();

  // 2. RoPE frequencies and extension methods
  let base = 10000, train = 8192, target = 131072, method = "none";
  const rope = xyChart({ height: 260, yLog: true, xLabel: "dimension pair (fast → slow rotation)", yLabel: "wavelength in tokens" });
  const rst = stats([{ label: "Dimension pairs seeing unfamiliar angles" }, { label: "Effective base" }]);
  function drawRope() {
    const d = 128, scale = target / train;
    const effBase = method === "ntk" ? base * Math.pow(scale, d / (d - 2)) : base;
    const pts = [], seen = [];
    let ood = 0;
    for (let i = 0; i < d / 2; i++) {
      let theta = Math.pow(effBase, (-2 * i) / d);
      if (method === "pi") theta /= scale; // position interpolation: squeeze all positions into the trained range
      const wl = (2 * Math.PI) / theta;
      pts.push([i, wl]);
      // a pair is in trouble if, at the target length, it reaches rotation angles it never reached in training
      const trainedMax = Math.min(2 * Math.PI, train * Math.pow(base, (-2 * i) / d));
      if (Math.min(2 * Math.PI, target * theta) > trainedMax + 1e-9) ood++;
    }
    rope.update([{ name: "Wavelength per pair", color: "var(--c6)", points: pts }, { name: "Trained context", color: "var(--c3)", points: [[0, train], [63, train]], dash: true }, { name: "Target context", color: "var(--c2)", points: [[0, target], [63, target]], dash: true }]);
    rst.set(0, `${ood} of 64`, ood === 0 ? "good" : ood > 10 ? "bad" : "warn");
    rst.set(1, fmtSI(effBase));
  }

  // 3. attention patterns
  let mask = "window";
  const maskEl = h("div"), mst = stats([{ label: "Attended pairs" }, { label: "vs full causal" }]);
  function drawMask() {
    const M = Array.from({ length: N }, (_, i) => Array.from({ length: N }, (_, j) => (MASKS[mask].m(i, j) ? 1 : j <= i ? 0.05 : null)));
    maskEl.replaceChildren(heatmap(M, { showValues: false, cell: 20, rows: Array.from({ length: N }, (_, i) => i), cols: Array.from({ length: N }, (_, i) => i) }));
    const n = M.flat().filter((v) => v === 1).length;
    mst.set(0, String(n)); mst.set(1, `${Math.round((n / ((N * (N + 1)) / 2)) * 100)}%`);
  }

  root.append(
    section(null, prose(`<p>In 2020 GPT-3 read 2,048 tokens at once. Frontier models now read one to ten million. Long context lets a model take a whole codebase, a book or hours of video as its prompt. Three things get in the way: attention's <b>quadratic cost</b>, <b>positions</b> the model never saw in training, and the <b>KV cache</b> that has to hold every past token. And even when it all fits, models don't use the middle of a long context as well as the ends.</p>`)),
    section("The quadratic wall",
      prose("<p>For each new token, the weights cost the same no matter how long the context is, but attention has to look at every past token. For an 8B-parameter model, attention overtakes everything else at around 27,000 tokens of context (FlashAttention makes it faster, but not less quadratic).</p>"),
      cost.el),
    section("Positions the model has never seen",
      prose("<p>RoPE rotates query and key pairs at 64 different speeds. Fast pairs complete many turns within the training length, so the model has seen every angle. Slow pairs, with wavelengths longer than the training context, only ever saw part of a turn. Run the model past its training length and those slow pairs reach angles it has never seen, and quality collapses. Extension methods change the speeds:</p>"),
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Method", options: [{ value: "none", label: "Do nothing" }, { value: "pi", label: "Position interpolation" }, { value: "ntk", label: "NTK-aware base scaling" }], value: method, onChange: (v) => { method = v; drawRope(); } }),
          slider({ label: "RoPE base", min: 3, max: 6.7, step: 0.1, value: 4, fmt: (v) => fmtSI(10 ** v), onInput: (v) => { base = 10 ** v; drawRope(); }, hint: "Llama 3 uses 500,000" }),
          slider({ label: "Trained context", min: 11, max: 17, step: 1, value: 13, fmt: (v) => fmtSI(2 ** v), onInput: (v) => { train = 2 ** v; drawRope(); } }),
          slider({ label: "Target context", min: 12, max: 21, step: 1, value: 17, fmt: (v) => fmtSI(2 ** v), onInput: (v) => { target = 2 ** v; drawRope(); } })),
        h("div.stage-body", rst.el, rope.el)),
      callout("tip", "What the methods do", "<ul style='margin:0;padding-left:18px'><li><b>Position interpolation</b> (Chen et al., 2023) slows every pair by the same factor, squeezing new positions into the trained range. Nothing is out of range, but nearby tokens become harder to tell apart, so it needs some fine-tuning.</li><li><b>NTK-aware scaling</b> raises the base, which slows the slow pairs a lot and the fast pairs hardly at all. Local detail is kept.</li><li><b>YaRN</b> blends the two per frequency band and adjusts attention temperature. It's what most long-context open models use, along with a short fine-tune on long documents.</li></ul>"),
    ),
    section("Look at fewer tokens",
      prose("<p>The other approach is to stop attending to everything. Pick a pattern; each row is a query position and dark cells are the keys it may see.</p>"),
      h("div.stage",
        h("div.stage-controls", segmented({ label: "Attention pattern", options: Object.entries(MASKS).map(([k, v]) => ({ value: k, label: v.label })), value: mask, onChange: (v) => { mask = v; drawMask(); } })),
        h("div.stage-body", mst.el, maskEl)),
      table(["Technique", "Idea", "Used in"], [
        ["Sliding window", "Each layer sees the last w tokens; stacking layers widens the reach", "Mistral 7B, Gemma's local layers"],
        ["Interleaved local / global", "Most layers local, every k-th layer full", "Gemma 2/3, many recent models"],
        ["Attention sinks", "Always keep the first few tokens plus a window", "StreamingLLM for endless streams"],
        ["Grouped-query / multi-query attention", "Share K and V across query heads: a smaller cache", "Llama 2/3, most modern models"],
        ["Multi-head latent attention (MLA)", "Cache one small latent per token and expand K, V from it", "DeepSeek V2/V3"],
        ["Native sparse attention", "Learn which blocks of the past to attend to", "DeepSeek NSA, 2025"],
        ["Linear attention / SSMs", "Replace attention with a fixed-size recurrent state", "Mamba hybrids, see later chapters"],
      ]),
    ),
    section("Fitting is not the same as using",
      prose("<p>Models retrieve facts placed at the start or end of a long prompt more reliably than ones buried in the middle (“Lost in the Middle”, Liu et al., 2023). Newer models are much better at needle-in-a-haystack tests, but harder long-context tasks that require combining many scattered facts still degrade with length. Practical advice: put instructions and the question <b>after</b> long documents, and retrieve only what's relevant rather than pasting everything.</p>"),
      predict({
        chapter: id,
        question: "A model trained on 8k contexts is run on 32k tokens with no changes. What usually happens?",
        options: ["It works, just slower", "Quality collapses: perplexity explodes once positions pass the training length", "It automatically truncates to 8k", "It works better because it sees more"],
        answer: 1,
        explain: "<p>Positions beyond the training range give attention scores the model never learned to handle, and output degrades into garbage, sometimes abruptly. Context extension needs either a position scheme designed to extrapolate or a rescaling (PI, NTK, YaRN) plus a little training at the new length.</p>",
      }),
    ),
    section("In code",
      code({
        PyTorch: `import torch

def rope_frequencies(head_dim, base=500_000.0, scale=1.0, method="ntk"):
    if method == "ntk":                      # raise the base: slow pairs slow down most
        base = base * scale ** (head_dim / (head_dim - 2))
    inv = base ** (-torch.arange(0, head_dim, 2) / head_dim)
    if method == "pi":                       # position interpolation: slow everything equally
        inv = inv / scale
    return inv                               # angle for position p is p * inv

def apply_rope(x, pos, inv):                 # x: (..., T, head_dim)
    ang = pos[:, None] * inv[None, :]        # (T, head_dim/2)
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)`,
      }),
    ),
  );
  drawRope(); drawMask();
}
