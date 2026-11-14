import { h, rng } from "../lib/core.js";
import { slider, stats, predict, code, callout, section, prose, table, bars } from "../lib/widgets.js";
import { xyChart, heatmap } from "../lib/plot.js";
import { gauss } from "../lib/nn.js";

const HID = 8;
const r0 = rng(12);
const Wx = Array.from({ length: 27 * HID }, () => gauss(r0) * 0.9);
const Wh0 = Array.from({ length: HID * HID }, () => gauss(r0) / Math.sqrt(HID));
const charId = (c) => (c >= "a" && c <= "z" ? c.charCodeAt(0) - 97 : 26);
function runRNN(text, scale) {
  let hs = new Float64Array(HID);
  const out = [];
  for (const ch of text) {
    const x = charId(ch), n = new Float64Array(HID);
    for (let j = 0; j < HID; j++) { let s = Wx[x * HID + j]; for (let i = 0; i < HID; i++) s += hs[i] * Wh0[i * HID + j] * scale; n[j] = Math.tanh(s); }
    hs = n; out.push(n);
  }
  return out;
}

export default function (root, { id }) {
  // 1. hidden state over time
  let text = "the cat sat on the mat", scale = 1.2;
  const input = h("input.ta", { value: text, maxlength: 24, "aria-label": "Text to feed the RNN" });
  const hm = h("div"), diffEl = h("div");
  function drawState() {
    text = (input.value.toLowerCase() || "a").slice(0, 24);
    const H = runRNN(text, scale);
    const M = Array.from({ length: HID }, (_, j) => H.map((v) => (v[j] + 1) / 2));
    hm.replaceChildren(heatmap(M, { rows: Array.from({ length: HID }, (_, j) => `h${j + 1}`), cols: [...text].map((c) => (c === " " ? "␣" : c)), showValues: false, cell: 26 }));
    // flip the first character and measure how different each later state is
    const alt = (text[0] === "x" ? "q" : "x") + text.slice(1), H2 = runRNN(alt, scale);
    diffEl.replaceChildren(bars(H.map((v, t) => ({ label: `step ${t + 1} “${text[t] === " " ? "␣" : text[t]}”`, value: Math.hypot(...v.map((x, j) => x - H2[t][j])), color: "var(--c2)" })), { max: 2.5, fmt: (v) => v.toFixed(3) }));
  }
  input.addEventListener("input", drawState);

  // 2. gradients through time
  let w = 0.9, f = 0.95;
  const gchart = xyChart({ height: 240, xLabel: "steps back in time", yLabel: "gradient size |∂h_T / ∂h_t|", yLog: true });
  const gst = stats([{ label: "Plain RNN after 30 steps" }, { label: "LSTM cell path after 30 steps" }]);
  function drawGrad() {
    const r = rng(5), T = 60, hsArr = [];
    let hh = 0;
    for (let t = 0; t < T; t++) { hh = Math.tanh(w * hh + gauss(r) * 0.6); hsArr.push(hh); }
    const vanilla = [], lstm = [];
    let g = 1;
    for (let k = 1; k <= 50; k++) { g *= w * (1 - hsArr[T - k] ** 2); vanilla.push([k, Math.max(Math.abs(g), 1e-30)]); lstm.push([k, f ** k]); }
    gchart.update([{ name: "Plain RNN: product of w · tanh′", color: "var(--c2)", points: vanilla }, { name: "LSTM cell: product of forget gates", color: "var(--c1)", points: lstm }], { yDomain: [1e-12, 1e3] });
    const v30 = vanilla[29][1], l30 = lstm[29][1];
    gst.set(0, v30.toExponential(1), v30 < 1e-4 || v30 > 1e3 ? "bad" : "good"); gst.set(1, l30.toExponential(1), l30 < 1e-4 ? "bad" : "good");
  }

  // 3. one LSTM step
  const g = { c: 0.8, f: 0.9, i: 0.3, cand: -0.6, o: 0.7 };
  const lst = stats([{ label: "New cell state c" }, { label: "Output h" }]);
  const lbars = h("div");
  function drawLSTM() {
    const c = g.f * g.c + g.i * g.cand, hh = g.o * Math.tanh(c);
    lst.set(0, c.toFixed(3)); lst.set(1, hh.toFixed(3));
    lbars.replaceChildren(bars([
      { label: "kept from old c (f·c)", value: Math.abs(g.f * g.c), color: "var(--c1)" },
      { label: "written in (i·g)", value: Math.abs(g.i * g.cand), color: "var(--c3)" },
      { label: "|new c|", value: Math.abs(c), color: "var(--c6)" },
      { label: "|output h|", value: Math.abs(hh), color: "var(--c2)" },
    ], { max: 2, fmt: (v) => v.toFixed(2) }));
  }
  const GS = (label, key, min, max, hint) => slider({ label, min, max, step: 0.05, value: g[key], fmt: (v) => v.toFixed(2), hint, onInput: (v) => { g[key] = v; drawLSTM(); } });

  root.append(
    section(null, prose(`<p>Text, audio and time series arrive in order, and their length varies. A <b>recurrent neural network</b> reads one element at a time and carries a <b>hidden state</b>: a vector summarising everything it has seen so far. At each step, <code>h_t = tanh(W_h · h_{t−1} + W_x · x_t)</code>. The same weights are reused at every step, so it works for any length.</p>`)),
    section("Watch the state evolve",
      prose("<p>Type below. Each column is the 8-number hidden state after reading that character (darker = higher). The bars underneath show how much the state <i>would</i> change if only the <b>first</b> character were different: that's how much the network still remembers it.</p>"),
      h("div.stage",
        h("div.stage-controls", h("div", { style: { flex: "1 1 260px" } }, input),
          slider({ label: "Recurrent weight scale", min: 0.2, max: 3, step: 0.1, value: scale, fmt: (v) => `${v.toFixed(1)}×`, onInput: (v) => { scale = v; drawState(); } })),
        h("div.stage-body", hm, h("div.eyebrow", "Influence of the first character on each later state"), diffEl)),
      callout("tip", "Things to try", "<p>At small scales the first character's influence fades within a few steps: short memory. Large scales keep it longer but make the state saturate at ±1 (black or white) and become chaotic. Training has to find a narrow good range, and gradients make that hard, as the next section shows.</p>"),
    ),
    section("Why plain RNNs forget: gradients through time",
      prose("<p>To learn that step 3 mattered for an error at step 50, the gradient must flow back through 47 steps, multiplying by the recurrent weight and the tanh slope at each. A product of 47 numbers below 1 vanishes; above 1, it explodes. This is the <b>vanishing gradient</b> problem (Hochreiter 1991; Bengio et al. 1994).</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Plain RNN recurrent weight w", min: 0.3, max: 3, step: 0.05, value: w, fmt: (v) => v.toFixed(2), onInput: (v) => { w = v; drawGrad(); } }),
          slider({ label: "LSTM forget gate f", min: 0.5, max: 1, step: 0.005, value: f, fmt: (v) => v.toFixed(3), onInput: (v) => { f = v; drawGrad(); } })),
        h("div.stage-body", gst.el, gchart.el)),
      predict({
        chapter: id,
        question: "How does the LSTM keep gradients alive over long gaps?",
        options: ["It uses a larger learning rate", "Its cell state is updated additively (c = f·c + i·g), so with f ≈ 1 the gradient passes almost unchanged", "It has no tanh anywhere", "It processes the sequence backwards"],
        answer: 1,
        explain: "<p>The cell state is a conveyor belt: each step mostly copies it forward (times f) and adds a little. Its gradient is a product of forget gates, which the network learns to hold near 1 for things worth remembering. It's the same idea as residual connections, eighteen years earlier (Hochreiter and Schmidhuber, 1997).</p>",
      }),
    ),
    section("Open up an LSTM cell",
      prose("<p>Three gates, each a sigmoid between 0 and 1, decide what to <b>forget</b> from the cell, what to <b>write</b> into it, and what to <b>output</b>. Normally the network computes them from the input and the previous output; here you set them by hand.</p>"),
      h("div.stage",
        h("div.stage-controls", GS("Previous cell c", "c", -2, 2), GS("Forget gate f", "f", 0, 1, "1 = keep everything"), GS("Input gate i", "i", 0, 1), GS("Candidate g (tanh)", "cand", -1, 1), GS("Output gate o", "o", 0, 1)),
        h("div.stage-body", lst.el, lbars)),
      table(["Model", "Memory", "Training", "Status"], [
        ["Plain RNN", "Short; gradients vanish", "Sequential, slow", "Historical"],
        ["LSTM / GRU", "Long, gated", "Sequential, slow", "Ran Google Translate and Siri until ~2017"],
        ["Transformer", "Whole context at once (attention)", "Parallel over the sequence", "Dominant since 2017"],
        ["SSMs / Mamba, RWKV, xLSTM", "Recurrent state, gated or selective", "Parallel via scans", "The comeback: linear-time inference (see the Mamba chapter)"],
      ]),
      prose("<p>Why attention won: an RNN must squeeze the whole past into one fixed-size vector and can only be trained one step after another. Attention lets every position look directly at every other (path length 1) and trains in parallel on GPUs. The price is cost that grows with the square of the length, which is why recurrent ideas keep returning.</p>"),
    ),
    section("In code",
      code({
        PyTorch: `import torch, torch.nn as nn

class CharLSTM(nn.Module):
    def __init__(self, vocab, d=256):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.lstm = nn.LSTM(d, d, num_layers=2, batch_first=True)
        self.head = nn.Linear(d, vocab)
    def forward(self, x, state=None):
        out, state = self.lstm(self.emb(x), state)   # out: every step's hidden state
        return self.head(out), state                 # next-character logits

# keep gradients in check when training RNNs:
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`,
      }),
    ),
  );
  drawState(); drawGrad(); drawLSTM();
}
