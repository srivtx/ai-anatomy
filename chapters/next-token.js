import { h } from "../lib/core.js";
import { slider, stats, predict, code, callout, section, prose, table, button, bars } from "../lib/widgets.js";
import { xyChart } from "../lib/plot.js";
import { CORPUS } from "../lib/corpus.js";

// A character n-gram model: count what follows each context of length n-1.
const TEXT = CORPUS.replace(/\s+/g, " ");
const split = Math.floor(TEXT.length * 0.85);
const TRAIN = TEXT.slice(0, split), HELD = TEXT.slice(split);
const ALPHA = [...new Set(TEXT)].sort();

function buildOne(order) {
  const counts = new Map();
  for (let i = order - 1; i < TRAIN.length; i++) {
    const ctx = TRAIN.slice(i - order + 1, i), c = TRAIN[i];
    let m = counts.get(ctx); if (!m) counts.set(ctx, (m = new Map()));
    m.set(c, (m.get(c) || 0) + 1);
  }
  return counts;
}
const TABLES = [null, ...Array.from({ length: 8 }, (_, i) => buildOne(i + 1))];
/** The model of order n is the stack of count tables 1..n, used with back-off. */
const build = (order) => ({ order, size: TABLES[order].size });
/** P(next | context): use the longest context seen in training ("back off" to shorter ones), add-k smoothed. */
function dist(model, order, text, k) {
  for (let o = order; o >= 1; o--) {
    const ctx = o > 1 ? text.slice(-(o - 1)) : "";
    if (ctx.length < o - 1) continue;
    const m = TABLES[o].get(ctx);
    if (!m) continue;
    const tot = [...m.values()].reduce((a, b) => a + b, 0);
    return { probs: ALPHA.map((c) => ((m.get(c) || 0) + k) / (tot + k * ALPHA.length)), seen: tot, used: o };
  }
  return { probs: ALPHA.map(() => 1 / ALPHA.length), seen: 0, used: 0 };
}
function crossEntropy(model, order, text, k) {
  let bits = 0, n = 0;
  for (let i = Math.max(order - 1, 1); i < text.length; i++) {
    const { probs } = dist(model, order, text.slice(0, i), k);
    bits -= Math.log2(probs[ALPHA.indexOf(text[i])] || 1e-12); n++;
  }
  return bits / n;
}
function shape(probs, T, topK, topP) {
  let p = probs.map((x) => Math.pow(Math.max(x, 1e-12), 1 / T));
  const order = p.map((x, i) => [x, i]).sort((a, b) => b[0] - a[0]);
  const keep = new Set(); let cum = 0; const tot = p.reduce((a, b) => a + b, 0);
  for (const [x, i] of order) { if (keep.size >= topK || (cum >= topP && keep.size)) break; keep.add(i); cum += x / tot; }
  p = p.map((x, i) => (keep.has(i) ? x : 0));
  const z = p.reduce((a, b) => a + b, 0);
  return p.map((x) => x / z);
}

export default function (root, { id }) {
  let order = 4, T = 1, topK = 40, topP = 1, k = 0.01, model = build(order), text = "The keeper ";
  const st = stats([{ label: "Contexts memorised" }, { label: "Training loss (bits/char)" }, { label: "Held-out loss (bits/char)" }]);
  const out = h("p.big-out"), distEl = h("div");
  const curve = xyChart({ width: 460, height: 280, xLabel: "n (context = n − 1 characters)", yLabel: "bits per character" });
  function evalCurve() {
    const tr = [], he = [];
    for (let n = 1; n <= 8; n++) { const m = build(n); tr.push([n, crossEntropy(m, n, TRAIN.slice(0, 600), k)]); he.push([n, crossEntropy(m, n, HELD, k)]); }
    curve.update([{ name: "Training text", color: "var(--c2)", points: tr, dots: true }, { name: "Held-out text", color: "var(--c1)", points: he, dots: true }], { marks: [{ x: order, label: `n = ${order}` }] });
  }
  function drawDist() {
    const { probs, seen, used } = dist(model, order, text, k);
    const p = shape(probs, T, topK, topP);
    const top = p.map((x, i) => [x, i]).sort((a, b) => b[0] - a[0]).slice(0, 8);
    distEl.replaceChildren(h("div.eyebrow", `Next character after “…${used > 1 ? text.slice(-(used - 1)).replace(/ /g, "·") : ""}” (seen ${seen}× in training${used < order ? `, backed off to n = ${used}` : ""})`),
      bars(top.map(([x, i]) => ({ label: JSON.stringify(ALPHA[i]).replace(/ /g, "·"), value: x * 100, color: "var(--accent)" })), { max: 100, fmt: (v) => `${v.toFixed(1)}%` }));
    out.textContent = text;
  }
  function stats_() {
    st.set(0, TABLES.slice(1, order + 1).reduce((a, t) => a + t.size, 0).toLocaleString());
    st.set(1, crossEntropy(model, order, TRAIN.slice(0, 600), k).toFixed(2));
    const he = crossEntropy(model, order, HELD, k);
    st.set(2, he.toFixed(2), he < 2.6 ? "good" : he > 3.4 ? "bad" : "warn");
  }
  function sample(n) {
    for (let i = 0; i < n; i++) {
      const p = shape(dist(model, order, text, k).probs, T, topK, topP);
      let u = Math.random(), c = ALPHA[0];
      for (let j = 0; j < p.length; j++) if ((u -= p[j]) <= 0) { c = ALPHA[j]; break; }
      text += c;
    }
    drawDist();
  }
  const rebuild = () => { model = build(order); stats_(); evalCurve(); drawDist(); };

  root.append(
    section(null, prose(`<p>Every GPT is trained on one task: given the text so far, output a probability for every possible next token. Generation is just that, in a loop: pick a token from the distribution, append it, repeat. The simplest model that does this is the <b>n-gram</b>: look up the last n − 1 characters in a big table of counts. It was the state of the art in speech recognition and machine translation for decades.</p>`)),
    section("A language model you can read",
      prose("<p>This model counted every character context in the lighthouse story (85% of it; the last 15% is held out). Choose n, then generate. Above the output you see the model's distribution for the next character, after temperature, top-k and top-p have reshaped it.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "n (order)", min: 1, max: 8, value: order, onInput: (v) => { order = v; rebuild(); } }),
          slider({ label: "Temperature", min: 0.1, max: 2, step: 0.05, value: T, fmt: (v) => v.toFixed(2), onInput: (v) => { T = v; drawDist(); } }),
          slider({ label: "top-k", min: 1, max: 40, value: topK, onInput: (v) => { topK = v; drawDist(); } }),
          slider({ label: "top-p", min: 0.05, max: 1, step: 0.05, value: topP, fmt: (v) => v.toFixed(2), onInput: (v) => { topP = v; drawDist(); } }),
          slider({ label: "Smoothing k", min: 0.001, max: 0.5, step: 0.001, value: k, fmt: (v) => v.toFixed(3), onInput: (v) => { k = v; stats_(); evalCurve(); drawDist(); }, hint: "Probability given to never-seen continuations" }),
          button("Generate 1", () => sample(1)), button("Generate 80", () => sample(80), "primary"), button("Start over", () => { text = "The keeper "; drawDist(); })),
        h("div.stage-body", st.el, distEl, out)),
      callout("tip", "Things to try", "<ul style='margin:0;padding-left:18px'><li>n = 1 knows only letter frequencies: “eta oinshr” soup. n = 3 makes word-like chunks. n = 5 makes real words and phrases.</li><li>n = 8 copies whole sentences from the story: training loss near 0, held-out loss worse. The model <b>memorised</b>. When it meets a context it never saw it has to <b>back off</b> to shorter ones (the label above the bars tells you when). That's overfitting again.</li><li>Temperature 0.1 is nearly greedy: it loops on the most common phrase. Temperature 2 is noise. top-p 0.5 keeps only the likeliest options: coherent but repetitive.</li></ul>"),
      h("div.lab", curve.el, h("div.prose", { html: "<p>Training loss keeps falling as n grows; held-out loss stops improving after about n = 4. Short contexts underfit, and long ones have mostly never been seen, so the model backs off and the extra length only helps it memorise. An n-gram's table grows exponentially with n. <b>Neural</b> language models fix this by <b>sharing statistical strength</b>: similar contexts get similar internal representations, so the model can generalise to a sentence it has never seen. Bengio et al. showed this with a neural model in 2003; transformers take it to contexts of a million tokens.</p>" })),
      predict({
        chapter: id,
        question: "A model gives the correct next character a probability of 1/4 on average (geometric mean). What's its loss in bits per character?",
        options: ["0.25", "2", "4", "0.5"],
        answer: 1,
        explain: "<p>Cross-entropy is −log₂ of the probability given to what actually happened: −log₂(1/4) = 2 bits. Perplexity is 2 to that power: 4, meaning the model is as uncertain as if it were choosing uniformly among 4 options. Good English models reach under 1 bit per character. LLM losses are usually reported in nats per token.</p>",
      }),
    ),
    section("Sampling knobs, side by side",
      table(["Knob", "What it does", "Typical use"], [
        ["Temperature T", "Divide logits by T before softmax. T < 1 sharpens, T > 1 flattens", "0 to 0.3 for code and facts, 0.7 to 1 for writing"],
        ["top-k", "Keep only the k most likely tokens", "Cuts off the absurd long tail"],
        ["top-p (nucleus)", "Keep the smallest set whose total probability ≥ p", "Adapts: few options when confident, many when not"],
        ["min-p", "Drop tokens below p × the top token's probability", "Popular in open-source samplers"],
        ["Greedy", "Always the single most likely token", "Deterministic, but prone to loops"],
        ["Beam search", "Track the k best whole sequences", "Translation; rarely used for chat"],
      ]),
    ),
    section("In code",
      code({
        PyTorch: `import torch

@torch.no_grad()
def generate(model, ids, n_new, temperature=0.8, top_k=50, top_p=0.95):
    for _ in range(n_new):
        logits = model(ids)[:, -1, :] / temperature          # only the last position
        if top_k:
            kth = torch.topk(logits, top_k).values[:, -1, None]
            logits[logits < kth] = -float("inf")
        probs = torch.softmax(logits, -1)
        if top_p < 1:
            sp, si = probs.sort(descending=True)
            cut = sp.cumsum(-1) - sp > top_p                   # tokens past the nucleus
            sp[cut] = 0
            probs = torch.zeros_like(probs).scatter(-1, si, sp)
        nxt = torch.multinomial(probs / probs.sum(-1, keepdim=True), 1)
        ids = torch.cat([ids, nxt], dim=1)
    return ids`,
      }),
    ),
  );
  rebuild();
}
