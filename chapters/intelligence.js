import { h, loop, rng } from "../lib/core.js";
import { stats, predict, code, callout, section, prose, table, button, segmented, bars } from "../lib/widgets.js";
import { plane, xyChart } from "../lib/plot.js";
import { CORPUS } from "../lib/corpus.js";
import { grokker } from "../lib/grok.js";
import { PUZZLES, search, same } from "../lib/arc.js";

/** Adaptive order-k character model: the code length (bits) an arithmetic coder would need. */
function codeLength(text, order) {
  const counts = new Map(), A = 256;
  let bits = 0;
  const lower = new Map(); // for order>0 blend with order-0 to avoid zero-probability symbols
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    const t0 = lower.get(ch) || 0, n0 = i;
    const p0 = (t0 + 0.5) / (n0 + 0.5 * A);
    let p = p0;
    if (order > 0) {
      // interpolate down from the longest context: p_k = (c + w * p_{k-1}) / (n + w)
      for (let k = 1; k <= order; k++) {
        if (i < k) break;
        const t = counts.get(k + "\u0000" + text.slice(i - k, i));
        if (!t) continue;
        const c = t.get(ch) || 0, w = 2 * t.size || 1;
        p = (c + w * p) / (t.n + w);
      }
    }
    bits -= Math.log2(p);
    lower.set(ch, t0 + 1);
    for (let k = 1; k <= order && k <= i; k++) {
      const key = k + "\u0000" + text.slice(i - k, i);
      let t = counts.get(key); if (!t) { t = new Map(); t.n = 0; counts.set(key, t); }
      t.set(ch, (t.get(ch) || 0) + 1); t.n++;
    }
  }
  return bits;
}
async function gzipBytes(text) {
  try {
    const cs = new Blob([text]).stream().pipeThrough(new CompressionStream("gzip"));
    return (await new Response(cs).arrayBuffer()).byteLength;
  } catch { return null; }
}
const PAL = ["var(--sunken)", "var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)", "var(--c5)", "var(--accent)"];
const grid = (g, cls = "") => h("div.agrid-cells", { class: `agrid-cells ${cls}`, style: { gridTemplateColumns: `repeat(${g[0].length}, 1fr)`, width: `${g[0].length * 22}px` } }, g.flat().map((v) => h("span", { style: { background: PAL[v] } })));

export default function (root, { id }) {
  // ---------- 1. compression ----------
  let variant = "orig";
  const cst = stats([{ label: "Characters" }, { label: "Best model" }, { label: "Compressed to" }]);
  const cbars = h("div");
  function variantText() {
    const r = rng(7);
    if (variant === "words") { const w = CORPUS.split(/\s+/); for (let i = w.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [w[i], w[j]] = [w[j], w[i]]; } return w.join(" "); }
    if (variant === "chars") { const c = [...CORPUS]; for (let i = c.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [c[i], c[j]] = [c[j], c[i]]; } return c.join(""); }
    return CORPUS;
  }
  async function drawComp() {
    const t = variantText(), n = t.length;
    const rows = [0, 1, 2, 3, 4].map((k) => ({ label: k ? `${k} char${k > 1 ? "s" : ""} of context` : "Letter frequencies", value: codeLength(t, k) / n, color: k === 4 ? "var(--accent)" : "var(--c4)" }));
    const gz = await gzipBytes(t);
    const all = [{ label: "Raw bytes", value: 8, color: "var(--ink-3)" }, ...(gz ? [{ label: "gzip", value: (gz * 8) / n, color: "var(--c2)" }] : []), ...rows];
    cbars.replaceChildren(h("div.eyebrow", "Bits per character (lower is better)"), bars(all, { max: 8, fmt: (v) => v.toFixed(2) }));
    const best = rows.reduce((a, b) => (b.value < a.value ? b : a));
    cst.set(0, String(n)); cst.set(1, best.label); cst.set(2, `${Math.round((best.value / 8) * 100)}% of raw`, "good");
  }

  // ---------- 2. grokking ----------
  let wd = 1, G, curve, done = false;
  const gst = stats([{ label: "Step" }, { label: "Train accuracy" }, { label: "Test accuracy (unseen sums)" }, { label: "Weight norm" }, { label: "Wave purity of units" }]);
  const gc = xyChart({ width: 520, height: 280, xLabel: "training step", yLabel: "accuracy (%)", xDomain: [0, 3000], yDomain: [0, 100] });
  const wp = plane({ width: 360, height: 280, xr: [0, 1], yr: [0, 1], label: "First-layer weights of 12 hidden units, plotted over the input value 0…22" });
  function resetGrok() { gl.stop(); G = grokker({ wd }); curve = { tr: [], te: [] }; done = false; sampleGrok(); drawGrok(); }
  function sampleGrok() { const a = G.evalSet(G.train), b = G.evalSet(G.test); curve.tr.push([G.t, a.acc * 100]); curve.te.push([G.t, b.acc * 100]); }
  function drawGrok() {
    gc.update([{ name: "Train (45% of all pairs)", color: "var(--c2)", points: curve.tr }, { name: "Test (the other 55%)", color: "var(--accent)", points: curve.te }], { xDomain: [0, 3000], yDomain: [0, 100] });
    const f = G.fourier();
    gst.set(0, String(G.t)); gst.set(1, `${curve.tr.at(-1)[1].toFixed(0)}%`); gst.set(2, `${curve.te.at(-1)[1].toFixed(0)}%`, curve.te.at(-1)[1] > 90 ? "good" : curve.tr.at(-1)[1] > 99 ? "warn" : "");
    gst.set(3, G.norm().toFixed(1)); gst.set(4, f.purity.toFixed(2), f.purity > 0.6 ? "good" : "");
    wp.paint(() => {
      wp.clear();
      const c = wp.colors, cols = 4, rows = 3, cw = wp.W / cols, ch = wp.H / rows;
      for (let u = 0; u < 12; u++) {
        const w = G.unit(u * 8), m = Math.max(1e-6, ...w.map(Math.abs)), x0 = (u % cols) * cw, y0 = Math.floor(u / cols) * ch;
        wp.box(x0 + 4, y0 + 4, cw - 8, ch - 8, { fill: c.sunken, stroke: "transparent", r: 8 });
        const ctx = wp.ctx; ctx.beginPath();
        w.forEach((v, a) => { const x = x0 + 10 + (a / (G.p - 1)) * (cw - 20), y = y0 + ch / 2 - (v / m) * (ch / 2 - 12); a ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
        ctx.strokeStyle = f.units[u * 8].purity > 0.6 ? c.accent : c["ink-3"]; ctx.lineWidth = 2; ctx.stroke();
        w.forEach((v, a) => wp.dot(x0 + 10 + (a / (G.p - 1)) * (cw - 20), y0 + ch / 2 - (v / m) * (ch / 2 - 12), 2, ctx.strokeStyle));
        wp.text(`k=${f.units[u * 8].k}`, x0 + cw - 10, y0 + 14, { size: 10, color: c["ink-3"], align: "right", mono: true });
      }
    });
  }
  const gl = loop(() => {
    for (let i = 0; i < 12 && G.t < 3000; i++) { G.step(); if (G.t % 50 === 0) sampleGrok(); }
    drawGrok();
    if (G.t >= 3000) { gl.stop(); done = true; }
  }, gc.el);

  // ---------- 3. ARC-style program search ----------
  let pi = 0;
  const pzEl = h("div"), solEl = h("div");
  function drawPuzzle() {
    const p = PUZZLES[pi];
    pzEl.replaceChildren(h("div.arc-row", ...p.train.map(([a, b], k) => h("div.arc-pair", h("div.eyebrow", `Example ${k + 1}`), h("div.arc-io", grid(a), h("span.arc-arrow", "→"), grid(b)))),
      h("div.arc-pair.test", h("div.eyebrow", "Test"), h("div.arc-io", grid(p.test[0]), h("span.arc-arrow", "→"), h("span.arc-q", "?")))));
    solEl.replaceChildren(h("p.canvas-cap", "Try to work out the rule yourself, then let the search have a go."));
  }
  function solve() {
    const p = PUZZLES[pi], t0 = performance.now(), res = search(p.train, 2), ms = performance.now() - t0;
    if (!res.prog) return solEl.replaceChildren(h("div.arc-result.bad", h("b", `No program found after trying ${res.tried} programs (${ms.toFixed(1)} ms).`), h("p", p.note || "The rule isn't expressible with these primitives."), h("div.arc-io", h("span", "Answer:"), grid(p.test[1]))));
    const out = res.apply(p.test[0]), ok = same(out, p.test[1]);
    solEl.replaceChildren(h("div.arc-result", { class: `arc-result ${ok ? "good" : "bad"}` },
      h("b", `Found after ${res.tried} programs (${ms.toFixed(1)} ms): `), h("code", res.prog.join(" → ")),
      h("div.arc-io", grid(out), h("span", ok ? "✓ matches the answer" : "✗ wrong answer"))));
  }

  root.append(
    section(null, prose(`<p>There's no agreed definition of intelligence, but there are useful working ones. Three run through modern AI: intelligence as <b>prediction</b> (and therefore compression), as <b>generalisation</b> beyond the examples seen, and as <b>efficient skill acquisition</b>, learning a new task from very little. Each has a live experiment below, and each exposes something current models do well and something they don't.</p>`)),
    section("To predict is to compress",
      prose("<p>Shannon showed that a model assigning probability <i>p</i> to the next symbol can encode it in −log₂ <i>p</i> bits with arithmetic coding. So a better predictor is literally a better compressor. Here, simple character models compress this course's short story; each sees a little more context. Then shuffle the text: what survives tells you what structure each model was using.</p>"),
      h("div.stage",
        h("div.stage-controls", segmented({ label: "Text", options: [{ value: "orig", label: "The story" }, { value: "words", label: "Words shuffled" }, { value: "chars", label: "Characters shuffled" }], value: variant, onChange: (v) => { variant = v; drawComp(); } })),
        h("div.stage-body", cst.el, cbars)),
      callout("note", "At scale", "<p>Delétang et al. (2023, “Language modeling is compression”) used Chinchilla 70B, trained only on text, as a compressor: it shrank ImageNet image patches to about 43% of their size and LibriSpeech audio to about 16%, beating PNG (~58%) and FLAC (~30%). Predicting well forces a model to capture regularities, and regularities are what generalise. The Hutter Prize pays for exactly this: compress 1 GB of Wikipedia better. (Note the catch: the model's own size isn't counted there; a fair compressor must also pay for its weights.)</p>"),
    ),
    section("Grokking: memorise first, understand later",
      prose("<p>A small network learns (a + b) mod 23 from 45% of all pairs. It memorises the training pairs within a few hundred steps, with test accuracy near chance. Keep training with weight decay and, long after training accuracy hit 100%, test accuracy climbs to near 100%: it has found the actual algorithm. Power et al. (2022) named this <b>grokking</b>.</p>"),
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Optimiser", options: [{ value: 1, label: "AdamW, weight decay 1.0" }, { value: 0, label: "No weight decay" }], value: wd, onChange: (v) => { wd = v; resetGrok(); gl.start(); } }),
          button("Train 3,000 steps", () => { if (done) resetGrok(); gl.start(); }, "primary"), button("Reset", resetGrok)),
        h("div.stage-body", gst.el, h("div.lab", gc.el, h("div.canvas-box", wp.el)),
          h("p.canvas-cap", "Right: how 12 hidden units respond to each input value. Memorising units look like noise; once the network groks, they become clean waves, each at one frequency k (purple)."))),
      callout("tip", "Why it happens, from first principles", "<p>Many weight settings fit the training set. Memorisation is found first because it's easy to reach, but it needs large weights. Weight decay slowly pushes towards the smallest-norm solution that still fits, and for modular addition that's a general algorithm: represent numbers as waves (cos and sin of 2πka/p) and add angles, using the identity cos(x + y) = cos x cos y − sin x sin y. Nanda et al. (2023) reverse-engineered exactly this circuit in a grokked transformer. Without weight decay nothing pushes the network off the memorised solution, and test accuracy stays at chance.</p>"),
    ),
    section("Learning a new skill from three examples",
      prose("<p>François Chollet defines intelligence as <b>skill-acquisition efficiency</b>: how well you learn new tasks from little data, given what you already know. His ARC benchmark gives a few input→output grids and asks for the rule. People find most tasks easy. Below, a brute-force program search over 11 grid operations, with compositions of up to two plus an inferred recolouring, tries to find the rule.</p>"),
      h("div.stage",
        h("div.stage-controls", segmented({ label: "Puzzle", options: PUZZLES.map((p, i) => ({ value: i, label: p.name })), value: pi, onChange: (v) => { pi = v; drawPuzzle(); } }), button("Search for a program", solve, "primary")),
        h("div.stage-body", pzEl, solEl)),
      callout("note", "What the search shows", "<p>When the toolbox contains the right concepts, search finds the rule in microseconds from two examples. When it doesn't (counting), no amount of search helps. The hard part of intelligence is having, or inventing, the right building blocks. ARC-AGI-1 (2019) was largely beaten by late 2024 with test-time training and massive sampling (OpenAI's o3 reported 76–88% at very high compute). ARC-AGI-2 (2025) was built to resist that: at launch, pure LLMs scored near 0% and top reasoning systems in the low single digits, while panels of people solved every task. Scores rose quickly through 2025, and the gap was still clear at the time of writing.</p>"),
      predict({
        chapter: id,
        question: "A model scores 90% on a benchmark of 10,000 questions. What can you conclude about its intelligence?",
        options: ["It's 90% as intelligent as a human", "Only that it does these kinds of questions well; whether it can learn genuinely new tasks efficiently is a separate question", "It will score 90% on any new benchmark", "Nothing: benchmarks are meaningless"],
        answer: 1,
        explain: "<p>Benchmark scores measure skill on a distribution, which can be bought with training data. Chollet's argument: intelligence is the efficiency of turning prior knowledge and a little experience into new skill. That's why ARC-style tests use novel tasks, few examples, and limit compute.</p>",
      }),
    ),
    section("Open problems",
      table(["Problem", "What's known", "Why it's hard"], [
        ["Sample efficiency", "Children learn words from a handful of exposures; LLMs need trillions of tokens", "We don't know which priors make human learning so efficient"],
        ["Continual learning", "Deployed models are frozen; fine-tuning forgets (see the chapter)", "Stability vs plasticity; safety of self-modification"],
        ["Reliable reasoning", "RL on verifiable tasks made big gains in maths and code", "Many valuable domains have no cheap verifier"],
        ["Grounding and world models", "Video and robotics models learn some physics", "Text alone underdetermines the physical world"],
        ["Knowing what you know", "Models are often calibrated on multiple choice, less so on free text", "Hallucination is partly a training-incentive problem"],
        ["Long-horizon agency", "Task length that agents complete at 50% success has doubled roughly every 7 months (METR, 2025)", "Errors compound; feedback is sparse"],
        ["Measuring intelligence", "ARC, held-out novel tasks, real-world task suites", "Every benchmark can be overfitted once it's a target"],
      ]),
    ),
    section("In code",
      code({
        "LLM as a compressor": `import torch, math
def bits_per_byte(model, tok, text):
    ids = tok(text, return_tensors="pt").input_ids
    with torch.no_grad():
        logits = model(ids).logits[0, :-1]
    nll = torch.nn.functional.cross_entropy(logits, ids[0, 1:], reduction="sum").item()   # nats
    return nll / math.log(2) / len(text.encode())       # an arithmetic coder would need this many bits per byte`,
        "Grokking (PyTorch)": `p, frac = 97, 0.4
pairs = torch.cartesian_prod(torch.arange(p), torch.arange(p)); labels = pairs.sum(1) % p
perm = torch.randperm(len(pairs)); tr = perm[: int(frac * len(pairs))]; te = perm[int(frac * len(pairs)):]
model = nn.Sequential(nn.Embedding(p, 128), nn.Flatten(), nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, p))
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98))
for step in range(20_000):                                        # full batch
    loss = F.cross_entropy(model(pairs[tr]), labels[tr])
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0:
        with torch.no_grad(): acc = (model(pairs[te]).argmax(-1) == labels[te]).float().mean()
        print(step, loss.item(), acc.item())                       # test accuracy jumps thousands of steps later`,
        "Program search": `import itertools
def search(train_pairs, prims, depth=2):
    for d in range(1, depth + 1):
        for names in itertools.product(prims, repeat=d):
            f = lambda g: functools.reduce(lambda x, n: prims[n](x), names, g)
            if all(f(x) == y for x, y in train_pairs):
                return names                       # the first (shortest) consistent program
    return None`,
      }, "Project 22 reproduces grokking and plots the Fourier features of the embedding."),
    ),
  );
  drawComp(); resetGrok(); drawPuzzle();
}
