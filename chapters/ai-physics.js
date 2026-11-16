import { h } from "../lib/core.js";
import { slider, stats, predict, code, callout, section, prose, table, stepper, button } from "../lib/widgets.js";
import { flowCanvas } from "../lib/flow.js";
import { goal, whyChain, stackMap, sticker } from "../lib/why.js";

function canvas(width, height, label) {
  const fc = flowCanvas({ width, height, label });
  fc.paint = (fn) => { fc.redraw = fn; fn(); };
  return fc;
}
const sci = (x) => { if (!isFinite(x)) return "∞"; const e = Math.floor(Math.log10(x)), m = x / 10 ** e; return e < 4 ? x.toFixed(x < 10 ? 1 : 0) : `${m.toFixed(1)} × 10^${e}`.replace(/\^(-?\d+)/, (_, d) => d.split("").map((c) => "⁰¹²³⁴⁵⁶⁷⁸⁹"["0123456789".indexOf(c)] ?? "⁻").join("")); };
const dur = (s) => {
  const u = [[3.15e7 * 1.38e10, "ages of the universe"], [3.15e7, "years"], [86400, "days"], [3600, "hours"], [60, "minutes"], [1, "seconds"], [1e-3, "ms"], [1e-6, "µs"]];
  for (const [k, n] of u) if (s >= k) { const v = s / k; return `${v >= 1000 ? sci(v) : v.toFixed(v < 10 ? 1 : 0)} ${n}`; }
  return `${(s * 1e9).toFixed(1)} ns`;
};

// two 8×8 templates: a vertical bar ("1") and a ring ("0")
const ONE = new Set([3, 4].flatMap((c) => [1, 2, 3, 4, 5, 6].map((r) => r * 8 + c)));
const ZERO = new Set([]);
for (let r = 1; r < 7; r++) for (let c = 1; c < 7; c++) if ((r === 1 || r === 6) ? c > 1 && c < 6 : c === 1 || c === 6) ZERO.add(r * 8 + c);
const W1 = Array.from({ length: 64 }, (_, i) => (ONE.has(i) ? 1 : -0.4)), W0 = Array.from({ length: 64 }, (_, i) => (ZERO.has(i) ? 1 : -0.4));

export default function (root, { id }) {
  // ---------- 1. the function machine ----------
  const fm = { x: 2, w: 1.5, b: -1 };
  const fc = canvas(880, 250, "A function machine: input, two knobs, output, and its graph");
  function drawFn() {
    const y = fm.w * fm.x + fm.b;
    fc.paint(() => {
      fc.clear(); const c = fc.colors;
      fc.box(20, 95, 90, 60, { fill: c.sunken, stroke: c.line });
      fc.text("input x", 65, 80, { size: 11, color: c["ink-3"], align: "center", mono: true });
      fc.text(fm.x.toFixed(1), 65, 125, { size: 22, align: "center", mono: true });
      fc.line(112, 125, 150, 125, { color: c["ink-3"], lw: 2 });
      fc.box(150, 70, 190, 110, { fill: c.surface, stroke: c.accent, lw: 2.5, r: 12 });
      fc.text(`y = ${fm.w.toFixed(1)} · x ${fm.b < 0 ? "−" : "+"} ${Math.abs(fm.b).toFixed(1)}`, 245, 108, { size: 15, align: "center", mono: true });
      fc.text("the rule", 245, 86, { size: 10.5, color: c["ink-3"], align: "center", mono: true });
      [[205, "w"], [285, "b"]].forEach(([x, n]) => { fc.dot(x, 150, 11, c.accent); fc.text(n, x, 150, { size: 12, color: c["accent-ink"], align: "center", mono: true }); });
      fc.text("knobs (weights)", 245, 172, { size: 10, color: c["ink-3"], align: "center", mono: true });
      fc.line(342, 125, 380, 125, { color: c["ink-3"], lw: 2 });
      fc.box(380, 95, 90, 60, { fill: c.sunken, stroke: c.line });
      fc.text("output y", 425, 80, { size: 11, color: c["ink-3"], align: "center", mono: true });
      fc.text(y.toFixed(1), 425, 125, { size: 22, align: "center", mono: true, color: c.c2 });
      // graph
      const gx = 520, gy = 20, gw = 340, gh = 210, X = (v) => gx + ((v + 4) / 8) * gw, Y = (v) => gy + gh / 2 - (v / 8) * gh;
      fc.box(gx, gy, gw, gh, { fill: c.surface, stroke: c.line, r: 8 });
      fc.line(gx, Y(0), gx + gw, Y(0), { color: c.line }); fc.line(X(0), gy, X(0), gy + gh, { color: c.line });
      fc.line(X(-4), Y(fm.w * -4 + fm.b), X(4), Y(fm.w * 4 + fm.b), { color: c.accent, lw: 2.5 });
      fc.line(X(fm.x), Y(0), X(fm.x), Y(y), { color: c.c2, dash: [4, 4] });
      fc.dot(X(fm.x), Y(Math.max(-4, Math.min(4, y))), 6, c.c2);
      fc.text("every x gets exactly one y: that's what makes it a function", gx + gw / 2, gy + gh - 12, { size: 11, color: c["ink-3"], align: "center" });
    });
  }

  // ---------- 2. a picture is a point in 64 dimensions ----------
  const px = new Set(ONE);
  const cells = Array.from({ length: 64 }, (_, i) => h("button", { type: "button", "aria-label": `pixel row ${Math.floor(i / 8) + 1} column ${(i % 8) + 1}`, onclick: () => { px.has(i) ? px.delete(i) : px.add(i); drawPad(); } }));
  const pad = h("div.pad", cells);
  const vec = h("div.vec", Array.from({ length: 64 }, () => h("i")));
  const wv = (W) => h("div.vec.w", W.map((v) => h("i", { style: { "--c": v > 0 ? "var(--c2)" : "color-mix(in srgb, var(--c1) 45%, var(--surface))" } })));
  const pst = stats([{ label: "Score: looks like a 1" }, { label: "Score: looks like a 0" }, { label: "The function's answer" }, { label: "Multiply-adds used" }]);
  function drawPad() {
    cells.forEach((b, i) => b.classList.toggle("on", px.has(i)));
    [...vec.children].forEach((d, i) => d.classList.toggle("on", px.has(i)));
    const x = Array.from({ length: 64 }, (_, i) => (px.has(i) ? 1 : 0));
    const s1 = x.reduce((a, v, i) => a + v * W1[i], 0), s0 = x.reduce((a, v, i) => a + v * W0[i], 0);
    const p1 = 1 / (1 + Math.exp(-(s1 - s0) / 2));
    pst.set(0, s1.toFixed(1)); pst.set(1, s0.toFixed(1));
    pst.set(2, px.size === 0 ? "blank: no evidence" : `${p1 > 0.5 ? "a 1" : "a 0"} (${Math.round(Math.max(p1, 1 - p1) * 100)}% sure)`, Math.abs(p1 - 0.5) > 0.3 ? "good" : "warn");
    pst.set(3, "128  (64 per score)");
  }
  const preset = (set) => { px.clear(); set.forEach((i) => px.add(i)); drawPad(); };

  // ---------- 3. how many multiply-adds ----------
  const cm = { N: 175e9, D: 300e9 };
  const cst = stats([{ label: "Training: 6 × params × tokens" }, { label: "By hand, 1 per second" }, { label: "One CPU core (~100 GFLOP/s)" }, { label: "AlexNet's GTX 580 (1.6 TFLOP/s)" }, { label: "One H100 (989 TFLOP/s bf16)" }, { label: "10,000 H100s at 40% use" }]);
  const nSl = slider({ label: "Parameters (weights)", min: 6, max: 12.3, step: 0.1, value: Math.log10(cm.N), fmt: (v) => sci(10 ** v), onInput: (v) => { cm.N = 10 ** v; drawCount(); } });
  const dSl = slider({ label: "Training tokens", min: 8, max: 13.5, step: 0.1, value: Math.log10(cm.D), fmt: (v) => sci(10 ** v), onInput: (v) => { cm.D = 10 ** v; drawCount(); } });
  function drawCount() {
    const F = 6 * cm.N * cm.D;
    cst.set(0, `${sci(F)} FLOPs`);
    cst.set(1, dur(F / 2), "bad"); cst.set(2, dur(F / 1e11), "bad"); cst.set(3, dur(F / 1.58e12), "warn");
    cst.set(4, dur(F / 989e12), "warn"); cst.set(5, dur(F / (1e4 * 989e12 * 0.4)), "good");
    ccv.paint(() => {
      ccv.clear(); const c = ccv.colors;
      ccv.box(10, 8, 780, 254, { fill: c.surface, stroke: c.line, r: 10 });
      ccv.text(`Time to train: ${sci(F)} FLOPs (log scale, each gridline is 1,000×)`, 24, 26, { size: 12 });
      const x0 = 200, x1 = 770, lo = -4, hi = 26, X = (sec) => x0 + ((Math.log10(Math.max(sec, 1e-4)) - lo) / (hi - lo)) * (x1 - x0);
      const marks = [[1, "1 s"], [3600, "1 hour"], [86400 * 30, "1 month"], [3.15e7 * 80, "a lifetime"], [4.35e17, "age of the universe"]];
      marks.forEach(([t, nm], i) => { ccv.line(X(t), 44, X(t), 222, { color: c["ink-3"], lw: 1, dash: [3, 4] }); ccv.text(nm, X(t), 236 + (i % 2) * 14, { size: 9.5, color: c["ink-3"], align: "center" }); });
      const rows = [["By hand, 1 per second", F / 2, c.bad], ["One CPU core", F / 1e11, c.bad], ["GTX 580 (2012)", F / 1.58e12, c.warn], ["One H100", F / 989e12, c.warn], ["10,000 H100s", F / (1e4 * 989e12 * 0.4), c.good]];
      rows.forEach(([nm, t, col], i) => {
        const y = 52 + i * 34;
        ccv.text(nm, x0 - 10, y + 9, { size: 11, align: "right", color: c["ink-2"] });
        const w = Math.max(3, Math.min(x1, X(t)) - x0);
        ccv.box(x0, y, w, 18, { fill: `color-mix(in srgb, ${col} 65%, transparent)`, stroke: "transparent", r: 4 });
        const lab = dur(t), inside = w > 140;
        ccv.text(lab, inside ? x0 + w - 8 : x0 + w + 8, y + 9, { size: 10.5, align: inside ? "right" : "left", color: c.ink });
      });
    });
  }
  // CPU and GPU dies, schematic: where each spends its transistors
  const dieCv = canvas(800, 250, "Schematic of a CPU die and a GPU die: the CPU spends its area on a few big cores with large control logic and caches, the GPU on thousands of small arithmetic units");
  dieCv.paint(() => {
    dieCv.clear(); const c = dieCv.colors;
    const alu = `color-mix(in srgb, ${c.good} 60%, transparent)`, ctl = `color-mix(in srgb, ${c.c4} 65%, transparent)`, cache = `color-mix(in srgb, ${c.c1} 40%, transparent)`;
    const panel = (x, title, sub) => { dieCv.box(x, 8, 380, 210, { fill: c.surface, stroke: c.line, r: 10 }); dieCv.text(title, x + 14, 26, { size: 12 }); dieCv.text(sub, x + 14, 42, { size: 10, color: c["ink-3"] }); };
    panel(10, "CPU: a few big cores", "make one stream of instructions finish fast");
    for (let k = 0; k < 4; k++) {                          // four cores, each mostly control and cache
      const x = 24 + (k % 2) * 176, y = 54 + Math.floor(k / 2) * 64;
      dieCv.box(x, y, 168, 58, { fill: c.sunken, stroke: c.line, lw: 1, r: 4 });
      dieCv.box(x + 4, y + 4, 92, 50, { fill: ctl, stroke: "transparent", r: 3 });
      dieCv.text("control, prediction,", x + 50, y + 22, { size: 9, align: "center", color: c.ink });
      dieCv.text("out-of-order", x + 50, y + 35, { size: 9, align: "center", color: c.ink });
      for (let a = 0; a < 4; a++) dieCv.box(x + 100 + (a % 2) * 33, y + 4 + Math.floor(a / 2) * 26, 30, 22, { fill: alu, stroke: "transparent", r: 3 });
    }
    dieCv.box(24, 184, 344, 26, { fill: cache, stroke: "transparent", r: 4 }); dieCv.text("big shared cache", 196, 197, { size: 10, align: "center" });
    panel(410, "GPU: thousands of small units", "make millions of independent steps finish in bulk");
    for (let r = 0; r < 8; r++) {
      const y = 54 + r * 16;
      dieCv.box(424, y, 10, 13, { fill: ctl, stroke: "transparent", r: 2 });
      for (let k = 0; k < 28; k++) dieCv.box(438 + k * 12, y, 10, 13, { fill: alu, stroke: "transparent", r: 2 });
    }
    dieCv.box(424, 184, 352, 26, { fill: cache, stroke: "transparent", r: 4 }); dieCv.text("smaller cache per unit of math", 600, 197, { size: 10, align: "center" });
    // key
    [[alu, "arithmetic (multiply-add units)"], [ctl, "control: deciding what to run"], [cache, "cache memory"]].forEach(([col, nm], i) => {
      const x = 24 + i * 230; dieCv.box(x, 230, 12, 12, { fill: col, stroke: "transparent", r: 2 }); dieCv.text(nm, x + 18, 236, { size: 10.5, color: c["ink-2"] });
    });
    dieCv.text("schematic, not to scale", 776, 236, { size: 9.5, color: c["ink-3"], align: "right" });
  });
  const ccv = canvas(800, 270, "How long training takes by hand, on one CPU core, on a 2012 GPU, on one H100 and on ten thousand, on a log time scale with a second, an hour, a month, a lifetime and the age of the universe marked");
  const setModel = (N, D) => { cm.N = N; cm.D = D; nSl.value = Math.log10(N); dSl.value = Math.log10(D); drawCount(); };

  root.append(
    section(null,
      goal("<p>Before any physics, pin down the job. Everything in this level exists to do one thing extremely fast and extremely cheaply: <b>multiply two numbers and add the result to a running total</b>, about a quadrillion times a second. This chapter shows where that job comes from. Once the job is clear, every later design choice, from silicon to CUDA, falls out as the cheapest way to do it.</p>",
        ["Do ~10¹⁵ multiply-adds every second", "Feed them numbers from memory just as fast", "Stay under ~1,000 watts per chip, or it melts"]),
      stackMap(id),
      h("div.vx-row", sticker("sand", { title: "Sand", caption: "silicon dioxide, the raw material", px: 3 }), sticker("ingot", { title: "A crystal", caption: "melted and pulled as one perfect lattice", px: 3 }), sticker("wafer", { title: "A wafer", caption: "sliced thin and printed with chips", px: 3 }), sticker("chip", { title: "A chip", caption: "billions of switches doing multiply-adds", px: 3 })),
    ),
    section("A function is a rule: one input in, one output out",
      prose("<p>A <b>function</b> is any rule that takes an input and gives back exactly one output. “Double it” is a function: 3 in, 6 out. “The temperature tomorrow, given today's weather” is a function too, even though nobody can write its rule down. So is “is this photo a cat?”: every photo has one right answer, and we don't know the formula.</p><p>A neural network is a function with <b>knobs</b>, numbers called <b>weights</b> that change what the rule does. Training means turning the knobs until the outputs match the examples. Below is the smallest possible one: one input, two knobs, one output.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Input x", min: -3, max: 3, step: 0.1, value: fm.x, fmt: (v) => v.toFixed(1), onInput: (v) => { fm.x = v; drawFn(); } }),
          slider({ label: "Knob w (slope)", min: -2, max: 2, step: 0.1, value: fm.w, fmt: (v) => v.toFixed(1), onInput: (v) => { fm.w = v; drawFn(); } }),
          slider({ label: "Knob b (shift)", min: -3, max: 3, step: 0.1, value: fm.b, fmt: (v) => v.toFixed(1), onInput: (v) => { fm.b = v; drawFn(); } })),
        h("div.stage-body", h("div.wide-canvas", fc.el))),
      callout("tip", "The work inside is one operation", "<p>Look at what the machine actually does: <b>multiply</b> x by w, then <b>add</b> b. A big model does exactly this, just with billions of knobs and thousands of inputs at once. Hardware people call it a <b>multiply-accumulate</b> (MAC), or a fused multiply-add (FMA). Hold on to it: it's the one thing a GPU is built to do.</p>"),
    ),
    section("A picture is one point in 64 dimensions",
      prose("<p>Draw on the 8×8 grid. To a computer the drawing is just 64 numbers in a row, 1 for ink and 0 for blank: that row is the <b>input vector</b>. One number is a point on a line; two numbers are a point on a page; three are a point in a room. Sixty-four numbers are a point in a 64-dimensional space. Nobody can picture that, but the arithmetic is identical: distances, directions and dot products work the same in any number of dimensions.</p><p>The function below has two outputs. Each one multiplies your 64 pixels by its own 64 weights (orange = “ink here is evidence”, blue = “ink here is evidence against”) and adds them up.</p>"),
      h("div.stage",
        h("div.stage-controls", h("div.btn-row", button("Draw a 1", () => preset(ONE)), button("Draw a 0", () => preset(ZERO)), button("Something in between", () => preset(new Set([...ONE].filter((i) => i % 8 === 3).concat([...ZERO].filter((i) => i % 8 < 4))))), button("Clear", () => preset([])))),
        h("div.stage-body",
          h("div.fn-row",
            h("div", { style: { flex: "0 0 auto" } }, h("div.fn-k", "Your drawing (click pixels)"), pad),
            h("div", h("div.fn-k", "The same drawing as a vector of 64 numbers"), vec, h("div.fn-k", "Weights for “looks like a 1”"), wv(W1), h("div.fn-k", "Weights for “looks like a 0”"), wv(W0))),
          pst.el)),
      prose("<p>That's a real (if tiny) classifier: the knobs here were set by hand, and training would set them from examples instead. The sizes grow fast:</p>"),
      table(["Input", "How many numbers", "So it's a point in…"], [
        ["This drawing", "8 × 8 = 64", "64 dimensions"],
        ["A CIFAR-10 image (Alex Krizhevsky's dataset)", "32 × 32 × 3 colours = 3,072", "3,072 dimensions"],
        ["An ImageNet photo as AlexNet saw it", "224 × 224 × 3 = 150,528", "150,528 dimensions"],
        ["A 4K photo", "3840 × 2160 × 3 ≈ 25 million", "25 million dimensions"],
        ["One token inside GPT-3", "12,288", "12,288 dimensions, at every layer"],
        ["GPT-3's knobs", "175 billion weights", "training searches a 175-billion-dimensional landscape"],
      ]),
    ),
    section("Count the multiply-adds",
      prose("<p>A layer that turns 4,096 numbers into 4,096 numbers does 4,096 × 4,096 ≈ 16.8 million multiply-adds per token. A model with N weights does about 2N floating-point operations (FLOPs: one multiply plus one add per weight) per token going forward, and training costs about three times that once the backward pass is included: <b>6 × N × tokens</b>. Pick a model and see how long that takes at different speeds.</p>"),
      h("div.stage",
        h("div.stage-controls", nSl, dSl, h("div.btn-row", button("GPT-2 (1.5B, ~10B tokens, est.)", () => setModel(1.5e9, 1e10)), button("GPT-3 (175B, 300B tokens)", () => setModel(175e9, 300e9)), button("Llama 3 70B (15T tokens)", () => setModel(70e9, 15e12)), button("Llama 3.1 405B (15.6T)", () => setModel(405e9, 15.6e12)))),
        h("div.stage-body", h("div.wide-canvas", ccv.el), cst.el)),
      callout("note", "What the numbers say", "<p>GPT-3's 3 × 10²³ FLOPs would take a person far longer than the age of the universe, one CPU core about a hundred thousand years, and one H100 about ten years. It only becomes weeks when you run <b>thousands of chips, each doing a million billion operations a second</b>. That is the whole hardware problem, and it's why the rest of this level exists. (The FLOP counts are the standard 6ND estimate; real runs also lose time to communication and restarts.)</p>"),
      predict({
        chapter: id,
        question: "A layer turns a 4,096-number vector into another 4,096-number vector (a 4096 × 4096 weight matrix). How many multiply-adds is that for one token?",
        options: ["8,192: one per input and output", "About 16.8 million: every output is a weighted sum of every input", "4,096", "About 4 billion"],
        answer: 1,
        explain: "<p>Each of the 4,096 outputs is a sum over all 4,096 inputs, so 4,096 × 4,096 = 16,777,216 multiply-adds. And those 4,096 sums don't depend on each other: they can all be done at the same time. That independence is the key to the next question, why a GPU.</p>",
      }),
    ),
    section("Why it's a GPU and not a faster CPU",
      h("div.stage", h("div.stage-body", h("div.wide-canvas", dieCv.el))),
      prose("<p>A CPU spends most of its transistors making <b>one</b> stream of instructions finish quickly: big caches, branch predictors, out-of-order logic. A GPU spends them on <b>thousands of arithmetic units</b> that each do the same simple step on different numbers. The work above is millions of independent multiply-adds, the GPU's shape exactly. Graphics happened to need the same shape first (every pixel of a frame is computed by the same small program), which is why gaming cards were ready when AI needed them. Dig into why, all the way down:</p>"),
      whyChain({
        claim: "AI runs on GPUs.",
        steps: [
          { q: "Why a GPU?", a: "<p>Because almost all of AI's work is multiply-adds that don't depend on each other, and a GPU has thousands of multiply-add units working at once. An H100 has 528 tensor cores and 16,896 regular cores.</p>" },
          { q: "Why not build one CPU core that's a thousand times faster?", a: "<p>Because single-core speed hit a wall around 2005. Clock frequency stopped climbing at about 4 GHz and has barely moved since (see <a href='/learn?c=power-wall'>the power wall</a>).</p>" },
          { q: "Why did clocks stop?", a: "<p>Heat. A chip's switching power is roughly C × V² × f: capacitance, voltage squared, frequency. Faster clocks need higher voltage, so power rises much faster than speed. At ~100 watts per square centimetre you can't get the heat out.</p>" },
          { q: "Why does switching make heat at all?", a: "<p>Every bit flip charges or drains a tiny capacitor (a transistor's gate plus its wires) through resistance. Charging one draws C·V² of energy from the supply: half is lost as heat on the way in, the other half when it drains again. Every 0 → 1 → 0 turns C·V² into heat (see <a href='/learn?c=transistor'>the transistor</a>).</p>" },
          { q: "Why can't we lower the voltage and avoid the heat?", a: "<p>Below about 0.7 V a transistor can't switch cleanly between on and off. Electrons have thermal energy, and some always leak over the barrier: each 60 mV of gate voltage only buys 10× more current (see <a href='/learn?c=transistor'>the transistor</a>).</p>" },
        ],
        axiom: { name: "Thermodynamics: energy is conserved, and heat jiggles every electron", html: "<p>Moving charge through a resistance turns electrical energy into heat; at temperature T every electron carries random energy of order kT (k is Boltzmann's constant). Neither can be engineered away. So the only way to do more arithmetic per second within a heat budget is to do <b>more of it at once, more slowly and more cheaply</b>. That's a GPU.</p>" },
      }),
    ),
    section("How AI met the GPU",
      prose("<p>The fit between neural networks and GPUs was spotted by a few people at the University of Toronto, years before it was obvious. It's worth knowing the order things happened in, because it was a chain of practical frustrations, not a grand plan.</p>"),
      stepper([
        { title: "2009: a grad student hits the compute wall", html: "<p>Alex Krizhevsky, a PhD student in Geoffrey Hinton's lab, tried to train models of millions of tiny 32×32 colour images for his master's thesis. Training was so slow he had to split the work across networked machines just to finish. Along the way he built the labelled sets <b>CIFAR-10</b> and <b>CIFAR-100</b>, still used everywhere.</p>" },
        { title: "2009: GPUs, and his own CUDA code", html: "<p>Reading about Hinton's earlier restricted Boltzmann machines, he saw people speeding them up on graphics cards. The same month as his thesis he wrote a report on a convolutional network written in CUDA. His GPU convolution code, which became <b>cuda-convnet</b>, was unusually fast at a time when very few people could program GPUs well.</p>" },
        { title: "2010: refining the recipe on small images", html: "<p>A 2010 paper trained convolutional networks on CIFAR-10 with <b>ReLUs</b>, noting they work well and are very cheap to compute. ReLU became a core piece of AlexNet. The big result came out of about three years of grinding on small images first.</p>" },
        { title: "2011: Ilya sees what the code could do", html: "<p>Ilya Sutskever believed neural networks would keep improving with more data and compute. When ImageNet (1.2 million labelled photos) arrived, he realised Alex's code could take it on, and persuaded him to try, with Hinton as advisor. His intuition: training a network is <i>automatically programming a parallel computer</i>, so networks and GPUs belong together.</p>" },
        { title: "2012: two gaming cards in a bedroom", html: "<p>A single <b>GTX 580</b> had only 3 GB of memory, too little for the network, so Alex split it across two cards (half the neurons on each, talking only at certain layers) and trained it at his parents' house for about five to six days. AlexNet's top-5 error on ImageNet was <b>15.3%</b>, against <b>26.2%</b> for the next best entry. Hinton later summed it up: Ilya thought they should do it, Alex made it work, and he got the Nobel Prize.</p><p class='muted'>Alex has rarely spoken publicly about his motivations, so the “why” here is reconstructed from his papers and from people who worked with him.</p>" },
      ]),
      table(["", "GTX 580 (AlexNet, 2012)", "H100 SXM (2022)", "Ratio"], [
        ["Transistors", "3.0 billion (40 nm)", "80 billion (TSMC 4N)", "≈ 27×"],
        ["Arithmetic", "1.58 TFLOP/s fp32", "67 TFLOP/s fp32; 989 TFLOP/s bf16 on tensor cores", "≈ 40× fp32, ≈ 600× bf16"],
        ["Memory", "3 GB at 192 GB/s", "80 GB at 3.35 TB/s", "≈ 27× size, ≈ 17× speed"],
        ["Power", "244 W", "700 W", "≈ 3×"],
      ]),
      callout("note", "Notice the last column", "<p>Compute grew far faster than memory speed, and power grew least of all. That imbalance (math is cheap, moving data is expensive, heat is the limit) is the physics this level is about. Next, the bottom of the stack: <a href='/learn?c=electrons'>why sand can switch</a>.</p>"),
    ),
    section("In code",
      code({
        "A function with knobs": `import numpy as np

def f(x, w, b):           # one input, two knobs
    return w * x + b      # a multiply and an add: one MAC

# the drawing classifier: 64 inputs, 2 outputs, 128 knobs
x = drawing.reshape(64)            # 8x8 pixels -> a point in 64 dimensions
W = np.stack([w_one, w_zero], 1)   # (64, 2)
scores = x @ W                     # 2 weighted sums = 128 multiply-adds
p = np.exp(scores) / np.exp(scores).sum()`,
        "Counting the work": `def train_flops(params, tokens):
    return 6 * params * tokens            # forward 2N + backward 4N per token

F = train_flops(175e9, 300e9)             # GPT-3: 3.15e23
for name, rate in [("CPU core", 1e11), ("GTX 580", 1.58e12), ("H100", 989e12)]:
    days = F / rate / 86400
    print(f"{name:10s} {days:12.3g} days")

# one 4096 -> 4096 layer, one token
print(4096 * 4096)                        # 16,777,216 multiply-adds, all independent`,
        "Same maths on a GPU": `import time, torch
x = torch.randn(8192, 4096, device="cuda", dtype=torch.bfloat16)   # 8192 tokens
W = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize(); t = time.time()
y = x @ W                                  # 137 billion multiply-adds
torch.cuda.synchronize()
print(2 * 8192 * 4096 * 4096 / (time.time() - t) / 1e12, "TFLOP/s")`,
      }, "Project 60 runs the counting and the GPU timing on a free Colab T4, then builds an adder out of NAND gates."),
    ),
  );
  drawFn(); drawPad(); drawCount();
}
