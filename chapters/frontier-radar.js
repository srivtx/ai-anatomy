import { h, rng } from "../lib/core.js";
import { slider, stats, predict, code, callout, section, prose, table, button, segmented } from "../lib/widgets.js";
import { plane } from "../lib/plot.js";
import { STATUS } from "../lib/cuts.js";
import { RADAR, AREAS } from "../lib/radar.js";
import { CORPUS } from "../lib/corpus.js";

// ---------- byte-level entropy model: order-2 characters with interpolated smoothing ----------
function byteModel(text) {
  const c1 = new Map(), c2 = new Map(), c0 = new Map();
  const bump = (m, k, ch) => { let t = m.get(k); if (!t) m.set(k, (t = new Map())); t.set(ch, (t.get(ch) || 0) + 1); };
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    c0.set(ch, (c0.get(ch) || 0) + 1);
    if (i >= 1) bump(c1, text[i - 1], ch);
    if (i >= 2) bump(c2, text.slice(i - 2, i), ch);
  }
  const alphabet = [...new Set(text + "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:'\"!?-()\n")];
  const tot0 = [...c0.values()].reduce((a, b) => a + b, 0);
  const dist = (t) => { if (!t) return null; let n = 0; for (const v of t.values()) n += v; return { t, n }; };
  /** entropy (bits) of the next character after ctx */
  return (ctx) => {
    const d2 = dist(c2.get(ctx.slice(-2))), d1 = dist(c1.get(ctx.slice(-1)));
    let H = 0;
    for (const ch of alphabet) {
      const p0 = ((c0.get(ch) || 0) + 0.5) / (tot0 + 0.5 * alphabet.length);
      const p1 = d1 ? (d1.t.get(ch) || 0) / d1.n : p0, p2 = d2 ? (d2.t.get(ch) || 0) / d2.n : p1;
      const w2 = d2 ? d2.n / (d2.n + 3) : 0, w1 = d1 ? d1.n / (d1.n + 3) : 0;
      const p = w2 * p2 + (1 - w2) * (w1 * p1 + (1 - w1) * p0);
      if (p > 0) H -= p * Math.log2(p);
    }
    return H;
  };
}

// a toy "language model" over two-token city names, for parallel-decoding conflicts
const CITIES = [["New", "York", 0.4], ["Los", "Angeles", 0.35], ["San", "Francisco", 0.25]];

export default function (root, { id }) {
  // ---------- 1. the radar ----------
  let area = "all", sel = RADAR.findIndex((r) => r.name.startsWith("Diffusion"));
  const pl = plane({ width: 760, height: 480, xr: [-0.02, 1.06], yr: [0.16, 1.0], label: "Frontier ideas by evidence and potential impact", onPoint: (x, y) => {
    let best = -1, bd = 1e9;
    RADAR.forEach((r, i) => { if (area !== "all" && r.area !== area) return; const d = (r.evidence - x) ** 2 + ((r.impact - y) * 1.2) ** 2; if (d < bd) { bd = d; best = i; } });
    if (best >= 0 && bd < 0.01) { sel = best; drawRadar(); drawCard(); }
  } });
  const card = h("div.radar-card");
  function drawRadar() {
    pl.paint(() => {
      pl.clear();
      const ctx = pl.ctx, c = pl.colors;
      // quadrants
      ctx.fillStyle = c.sunken; ctx.globalAlpha = 0.55; ctx.fillRect(pl.X(0.5), pl.Y(1), pl.X(1.04) - pl.X(0.5), pl.Y(0.56) - pl.Y(1)); ctx.globalAlpha = 1;
      pl.line(pl.X(0.5), 0, pl.X(0.5), pl.H, { color: c.line, dash: [4, 5] });
      pl.line(0, pl.Y(0.56), pl.W, pl.Y(0.56), { color: c.line, dash: [4, 5] });
      pl.text("WATCH CLOSELY", 10, 18, { size: 10.5, color: c["ink-3"], mono: true });
      pl.text("USE NOW", pl.W - 10, 18, { size: 10.5, color: c["ink-3"], mono: true, align: "right" });
      pl.text("LONG SHOTS", 10, pl.H - 30, { size: 10.5, color: c["ink-3"], mono: true });
      pl.text("NICE TO HAVE", pl.W - 10, pl.H - 30, { size: 10.5, color: c["ink-3"], mono: true, align: "right" });
      pl.text("evidence →  (one paper … used in frontier models)", pl.W / 2, pl.H - 10, { size: 11, color: c["ink-2"], align: "center" });
      // greedy label placement: try right, left, above, below; skip boxes that collide
      const placed = [];
      ctx.font = "600 12px Inter, system-ui, sans-serif";
      const order = RADAR.map((_, i) => i).sort((a, b) => (b === sel) - (a === sel));
      for (const i of order) {
        const r = RADAR[i], on = area === "all" || r.area === area;
        const x = pl.X(r.evidence), y = pl.Y(r.impact);
        const short = r.short || r.name.replace(/ \(.*\)/, "").replace(/ and .*/, "").replace(/: .*/, "").replace(/ with .*/, "").replace(/ for .*/, "").replace(/ learned .*/, "").replace(/ at scale/, "").replace(/ everywhere/, "");
        const w = ctx.measureText(short).width + 4, hh = 15;
        const cands = [[x + 11, y - hh / 2], [x - 11 - w, y - hh / 2], [x - w / 2, y - 12 - hh], [x - w / 2, y + 10]];
        const fit = cands.find(([bx, by]) => bx > 2 && bx + w < pl.W - 2 && by > 24 && by + hh < pl.H - 40 && !placed.some((p) => bx < p[0] + p[2] && bx + w > p[0] && by < p[1] + p[3] && by + hh > p[1]));
        r._lab = on && fit ? [fit[0], fit[1], w, hh, short] : null;
        if (r._lab) placed.push(r._lab);
        placed.push([x - 8, y - 8, 16, 16]);
      }
      RADAR.forEach((r, i) => {
        const on = area === "all" || r.area === area;
        ctx.globalAlpha = on ? 1 : 0.15;
        const col = c[{ established: "good", promising: "c1", speculative: "c5" }[r.status]];
        const x = pl.X(r.evidence), y = pl.Y(r.impact);
        if (i === sel) { pl.dot(x, y, 13, c.surface); ctx.strokeStyle = c.ink; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x, y, 11, 0, 7); ctx.stroke(); }
        pl.dot(x, y, 7.5, c.surface); pl.dot(x, y, 6, col);
        if (r._lab) pl.text(r._lab[4], r._lab[0] + 2, r._lab[1] + r._lab[3] / 2, { size: 12, weight: i === sel ? 700 : 500, color: i === sel ? c.ink : c["ink-2"] });
        ctx.globalAlpha = 1;
      });
    });
  }
  function drawCard() {
    const r = RADAR[sel], [label, color] = STATUS[r.status];
    card.replaceChildren(
      h("div.radar-head", h("h3", r.name), h("span.idea-pill", { style: { color } }, label), h("span.radar-meta", `${AREAS[r.area]} · ${r.year}`)),
      h("p", r.claim),
      h("p", h("b", "What's known: "), r.known),
      h("p", h("b", "What to watch: "), r.watch),
      code({ Sketch: r.code }));
  }

  // ---------- 2. diffusion LMs: parallel decoding conflicts ----------
  let steps = 1, samples = [];
  const dst = stats([{ label: "Samples that are real cities" }, { label: "Forward passes per sample" }]);
  const chips = h("div.chips");
  function sampleCities(n = 24) {
    const r = rng(Math.floor(Math.random() * 1e6));
    const pick = (ps) => { let u = r(), i = 0; while (i < ps.length - 1 && (u -= ps[i]) > 0) i++; return i; };
    const P = CITIES.map((c) => c[2]);
    samples = Array.from({ length: n }, () => {
      if (steps === 2) { const i = pick(P); return [CITIES[i][0], CITIES[i][1]]; }  // one token per step: the second is conditioned on the first
      return [CITIES[pick(P)][0], CITIES[pick(P)][1]];                               // both at once: each from its own marginal
    });
    drawCities();
  }
  function drawCities() {
    const ok = (s) => CITIES.some((c) => c[0] === s[0] && c[1] === s[1]);
    chips.replaceChildren(...samples.map((s) => h("span.chip", { class: `chip ${ok(s) ? "ok" : "bad"}` }, `${s[0]} ${s[1]}`)));
    const k = samples.filter(ok).length, exact = steps === 2 ? 1 : CITIES.reduce((a, c) => a + c[2] ** 2, 0);
    dst.set(0, `${k} of ${samples.length} (expected ${Math.round(exact * 100)}%)`, k / samples.length > 0.9 ? "good" : "bad");
    dst.set(1, steps === 2 ? "2 (one token each)" : "1 (both tokens at once)");
  }

  // ---------- 3. byte patches ----------
  const H = byteModel(CORPUS);
  let thr = 3.0, text = "The lighthouse keeper wrote down the colour of the water. Then a quokka named Zbigniew arrived at 07:45.";
  const patchEl = h("div.patches"), pst = stats([{ label: "Bytes" }, { label: "Patches" }, { label: "Average patch" }, { label: "Typical BPE tokens (≈ bytes ÷ 4)" }]);
  const ta = h("textarea.ta", { rows: 3, oninput: (e) => { text = e.target.value; drawPatches(); } }, text);
  function drawPatches() {
    const t = text.slice(0, 400), groups = [];
    let cur = [];
    for (let i = 0; i < t.length; i++) {
      const e = H(t.slice(Math.max(0, i - 2), i));
      if (i > 0 && e > thr && cur.length) { groups.push(cur); cur = []; }
      cur.push([t[i], e]);
    }
    if (cur.length) groups.push(cur);
    patchEl.replaceChildren(...groups.map((g, gi) => h("span.patch", { class: `patch p${gi % 2}` }, g.map(([ch, e]) => h("span.pc", { title: `${e.toFixed(2)} bits` }, h("span.pch", ch === " " ? "␣" : ch), h("span.pbar", { style: { height: `${Math.min(1, e / 5) * 18 + 1}px`, background: e > thr ? "var(--accent)" : "var(--ink-3)" } }))))));
    const bytes = new TextEncoder().encode(t).length;
    pst.set(0, String(bytes)); pst.set(1, String(groups.length)); pst.set(2, `${(bytes / Math.max(1, groups.length)).toFixed(1)} bytes`); pst.set(3, `≈ ${Math.round(bytes / 4)}`);
  }

  root.append(
    section(null, prose(`<p>The useful half-life of “state of the art” is months. This chapter is a map of ideas that were being tested at the edge in 2024–2026, with an honest read of the evidence for each: what was claimed, what was reproduced, what to watch. Treat it as a snapshot written from memory: numbers are approximate, and some of these will look naive in a year. The habit it teaches will last longer: for every new idea, ask <i>how much evidence, at what scale, compared against what baseline</i>.</p>`)),
    section("The radar",
      h("div.stage",
        h("div.stage-controls", segmented({ label: "Area", options: [{ value: "all", label: "All" }, ...Object.entries(AREAS).map(([value, label]) => ({ value, label }))], value: area, onChange: (v) => { area = v; const f = RADAR.findIndex((r) => v === "all" || r.area === v); if (RADAR[sel].area !== v && v !== "all") sel = f; drawRadar(); drawCard(); } })),
        h("div.stage-body",
          h("div.legend", Object.values(STATUS).map(([l, c]) => h("span", h("i", { style: { background: c } }), l)), h("span", "Click a dot to read about it")),
          h("div.canvas-box", pl.el), card)),
      callout("note", "How to read the axes", "<p><b>Evidence</b> runs from a single paper at small scale (left) to adoption in frontier models (right). <b>Impact</b> is a judgement of how much practice would change if the idea fully works. The top-left quadrant is where careers are made and where most ideas quietly fail.</p>"),
    ),
    section("Why diffusion LMs decode carefully",
      prose("<p>A diffusion language model predicts every masked position at once, but each prediction is a separate distribution for that position. Ask it to fill “I flew to ___ ___” and it knows the first slot is New, Los or San and the second is York, Angeles or Francisco, without knowing which go together. Unmask both in one step and you sample from each slot independently.</p>"),
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Decoding", options: [{ value: 1, label: "Both tokens in one step" }, { value: 2, label: "One token per step" }], value: steps, onChange: (v) => { steps = v; sampleCities(); } }),
          button("Sample again", () => sampleCities(), "primary")),
        h("div.stage-body", dst.el, chips)),
      callout("tip", "The trade-off, from first principles", "<p>With probabilities 0.40, 0.35 and 0.25, independent sampling produces a real city only 0.40² + 0.35² + 0.25² ≈ 35% of the time. Real diffusion LMs avoid this by unmasking the most confident positions first and spending more steps where tokens depend on each other, so speed (fewer steps) trades directly against coherence. Autoregressive models never face this: each token is sampled conditioned on all previous ones.</p>"),
    ),
    section("No tokenizer: cut bytes where the next one is surprising",
      prose("<p>Byte-level models avoid tokenizers but make sequences ~4× longer. The Byte Latent Transformer groups bytes into patches with a small byte-level model: a new patch starts wherever the next byte is hard to predict. Predictable stretches (the rest of a common word) become long, cheap patches; surprising ones (names, numbers) get more compute. The model below is a tiny character model trained on this course's small story corpus. Edit the text.</p>"),
      h("div.stage",
        h("div.stage-controls", slider({ label: "Entropy threshold (bits)", min: 1, max: 4, step: 0.1, value: thr, fmt: (v) => v.toFixed(1), onInput: (v) => { thr = v; drawPatches(); } })),
        h("div.stage-body", pst.el, ta, patchEl, h("p.canvas-cap", "Bars under each character show the model's uncertainty about it before seeing it. A purple bar crosses the threshold and starts a new patch."))),
      predict({
        chapter: id,
        question: "With entropy patching, where should a byte-level model spend the most compute?",
        options: ["Evenly on every byte", "On the start of words, names and numbers, where the next byte is hard to predict", "Only on spaces", "On the most common words"],
        answer: 1,
        explain: "<p>Compute follows information. Inside “lighthouse”, once you've seen “lighth”, the rest is nearly free. A rare name or a time like 07:45 is unpredictable byte by byte, so it gets more, shorter patches and more of the big model's attention.</p>",
      }),
    ),
    section("How to keep up without drowning",
      table(["Habit", "Why"], [
        ["Read the ablations and baselines before the abstract", "Most gains disappear against a tuned baseline (see the research-method chapter)"],
        ["Ask “at what scale?”", "Many ideas help at 100M parameters and vanish at 10B"],
        ["Look for independent reproductions", "One lab's result is a hypothesis; three labs' is a finding"],
        ["Check cost: FLOPs, memory, wall-clock", "A 1% gain at 2× cost is a loss for most users"],
        ["Reproduce the smallest claim yourself", "An afternoon with a tiny model teaches more than ten summaries"],
        ["Track adoption in open recipes", "When DeepSeek, Qwen, Llama or OLMo ship it, it probably works"],
      ]),
    ),
    section("In code",
      code({
        "Entropy patching": `import torch
def patch_boundaries(byte_ids, small_lm, threshold=2.6):
    """small_lm: a byte-level LM returning logits (T, 256). Returns patch start indices."""
    with torch.no_grad():
        logp = small_lm(byte_ids[None])[0].log_softmax(-1)
    H = -(logp.exp() * logp).sum(-1) / torch.log(torch.tensor(2.0))   # bits, entropy of byte t+1 given <= t
    starts = [0] + [t + 1 for t in range(len(byte_ids) - 1) if H[t] > threshold]
    return starts`,
        "Confidence-ordered diffusion decoding": `@torch.no_grad()
def diffusion_decode(model, prompt, L=64, steps=16, MASK=126336):
    x = torch.cat([prompt, torch.full((L,), MASK)])
    for s in range(steps):
        masked = x == MASK
        probs = model(x[None]).logits[0].softmax(-1)
        conf, tok = probs.max(-1)
        n_unmask = int(masked.sum() / (steps - s))            # spread the remaining work over the steps
        conf = torch.where(masked, conf, torch.tensor(-1.0))
        idx = conf.topk(n_unmask).indices
        x[idx] = tok[idx]
    return x[len(prompt):]`,
      }, "Each radar card above has its own sketch; the deeper cuts below add a few more."),
    ),
  );
  drawRadar(); drawCard(); sampleCities(); drawPatches();
}
