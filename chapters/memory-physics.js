import { h, loop, rng } from "../lib/core.js";
import { slider, stats, predict, code, callout, section, prose, table, segmented, stepper, toggle, button } from "../lib/widgets.js";
import { flowCanvas } from "../lib/flow.js";
import { goal, whyChain, stackMap, sticker } from "../lib/why.js";
import { walk } from "../lib/walks.js";

function canvas(width, height, label) {
  const fc = flowCanvas({ width, height, label });
  fc.paint = (fn) => { fc.redraw = fn; fn(); };
  return fc;
}
// Horowitz, ISSCC 2014, 45 nm at 0.9 V (as tabulated by Han et al., 2016), picojoules
const E = { add32f: 0.9, mul32f: 3.7, reg: 1, sram: 5, dram: 640 };

export default function (root, { id }) {
  // ---------- 1. a DRAM array that forgets ----------
  const R = 8, C = 24, N = R * C;
  const dr = { refresh: true, temp: 45, t: 0, since: 0 };
  const rnd = rng(5);
  const cells = Array.from({ length: N }, (_, i) => {
    const z = Math.sqrt(-2 * Math.log(Math.max(1e-9, rnd()))) * Math.cos(2 * Math.PI * rnd());
    return { bit: (i * 7 + (i >> 3)) % 3 ? 1 : 0, tau: 3000 * Math.exp(1.25 * z), q: 1, lost: false };   // retention time constant in ms, lognormal
  });
  const dc = canvas(880, 250, "A DRAM array: each square is a capacitor holding charge");
  const dst = stats([{ label: "Time since last refresh" }, { label: "Bits read wrong" }, { label: "Weakest cell holds for" }]);
  function resetD() { dr.t = 0; dr.since = 0; cells.forEach((c) => { c.q = c.bit; c.lost = false; }); drawD(); }
  function stepD(ms) {
    dr.t += ms; dr.since += ms;
    const k = 2 ** ((dr.temp - 45) / 10);                         // leakage roughly doubles every 10 °C
    for (const c of cells) if (c.bit) { c.q *= Math.exp(-(ms * k) / c.tau); if (c.q < 0.5) c.lost = true; }
    if (dr.refresh && dr.since >= 64) { dr.since = 0; cells.forEach((c) => { if (!c.lost) c.q = c.bit; }); }
  }
  function drawD() {
    const lost = cells.filter((c) => c.lost).length, k = 2 ** ((dr.temp - 45) / 10), weak = Math.min(...cells.map((c) => c.tau)) / k * Math.LN2;
    dst.set(0, `${dr.since.toFixed(0)} ms ${dr.refresh ? "(refresh every 64 ms)" : "(refresh off)"}`, dr.refresh ? "good" : "warn");
    dst.set(1, `${lost} of ${cells.filter((c) => c.bit).length} ones`, lost ? "bad" : "good");
    dst.set(2, `${weak.toFixed(0)} ms at ${dr.temp} °C`, weak > 64 ? "good" : "bad");
    dc.paint(() => {
      dc.clear(); const c = dc.colors, s = 30;
      cells.forEach((cl, i) => {
        const x = 30 + (i % C) * (s + 4), y = 26 + Math.floor(i / C) * (s - 4), hh = s - 8;
        dc.box(x, y, s, hh, { fill: c.sunken, stroke: cl.lost ? c.bad : c.line, lw: cl.lost ? 2 : 1, r: 4 });
        if (cl.bit) dc.box(x + 2, y + 2 + (hh - 4) * (1 - cl.q), s - 4, (hh - 4) * cl.q, { fill: cl.lost ? c.bad : cl.q > 0.5 ? c.c4 : c.warn, stroke: "transparent", r: 2 });
      });
      dc.line(30, 26 + R * (s - 4) + 8, 30 + C * (s + 4), 26 + R * (s - 4) + 8, { color: c.line });
      dc.text("each square: one capacitor. Yellow height = charge. A 1 is read wrong once its charge falls below half (red).", 30, 26 + R * (s - 4) + 24, { size: 11.5, color: c["ink-2"] });
      dc.text(`simulated time ${(dr.t / 1000).toFixed(2)} s`, 850, 12, { size: 11, color: c["ink-3"], align: "right", mono: true });
    });
  }
  const anim = loop((dt) => { const sim = dt * 400; for (let q = 0; q < 8; q++) stepD(sim / 8); drawD(); }, dc.el);
  const playBtn = button("▶ Run time", () => { anim.running ? anim.stop() : anim.start(); playBtn.textContent = anim.running ? "❚❚ Pause" : "▶ Run time"; });

  // ---------- 2. energy per multiply-add ----------
  const en = { src: "dram", reuse: 1 };
  const ec = canvas(880, 200, "Energy of one fp32 multiply-add, split into arithmetic and data movement");
  const est = stats([{ label: "Arithmetic (multiply + add)" }, { label: "Fetching the two inputs" }, { label: "Share spent moving data" }]);
  function drawE() {
    const comp = E.mul32f + E.add32f, mv = (2 * E[en.src]) / en.reuse, tot = comp + mv;
    est.set(0, `${comp.toFixed(1)} pJ`); est.set(1, `${mv < 10 ? mv.toFixed(2) : mv.toFixed(0)} pJ`, mv > comp ? "bad" : "good");
    est.set(2, `${((mv / tot) * 100).toFixed(mv / tot > 0.99 ? 1 : 0)}%`, mv / tot > 0.5 ? "bad" : "good");
    ec.paint(() => {
      ec.clear(); const c = ec.colors, L = (v) => Math.max(2, (Math.log10(v) + 2) / 5 * 560);
      const rows = [["32-bit integer add", 0.1, c.c3], ["32-bit float add", E.add32f, c.c3], ["32-bit float multiply", E.mul32f, c.c3], ["Read 32 bits from registers", E.reg, c.c1], ["Read 32 bits from on-chip SRAM", E.sram, c.c1], ["Read 32 bits from off-chip DRAM", E.dram, c.bad]];
      rows.forEach(([n, v, col], i) => {
        const y = 14 + i * 30;
        ec.text(n, 210, y + 10, { size: 12, align: "right", color: c["ink-2"] });
        ec.box(220, y, L(v), 20, { fill: col, stroke: "transparent", r: 4 });
        ec.text(`${v} pJ`, 228 + L(v), y + 10, { size: 12, mono: true });
      });
      ec.text("log scale: each step right is 10× more energy · 45 nm figures from Horowitz (2014); ratios hold roughly today", 220, 194, { size: 10.5, color: c["ink-3"] });
    });
  }

  // ---------- 3. the hierarchy, measured ----------
  const hc = canvas(880, 300, "The H100 memory hierarchy: capacity against latency");
  function drawH() {
    hc.paint(() => {
      hc.clear(); const c = hc.colors, x0 = 80, x1 = 850, y0 = 20, y1 = 250;
      const X = (b) => x0 + ((Math.log10(b) - 4) / 8) * (x1 - x0), Y = (cy) => y1 - ((Math.log10(cy)) / 3) * (y1 - y0);
      for (let e = 4; e <= 12; e += 2) { hc.line(X(10 ** e), y0, X(10 ** e), y1, { color: c.line }); hc.text(["10 KB", "1 MB", "100 MB", "10 GB", "1 TB"][(e - 4) / 2], X(10 ** e), y1 + 14, { size: 10.5, color: c["ink-3"], align: "center", mono: true }); }
      for (const cy of [1, 10, 100, 1000]) { hc.line(x0, Y(cy), x1, Y(cy), { color: c.line }); hc.text(`${cy} cycles`, x0 - 8, Y(cy), { size: 10.5, color: c["ink-3"], align: "right", mono: true }); }
      const pts = [["registers", 256e3, 1, "256 KB per SM"], ["shared memory / L1", 256e3, 30, "256 KB per SM"], ["L2 cache", 50e6, 300, "50 MB, two halves"], ["HBM3 (DRAM)", 80e9, 600, "80 GB, 3.35 TB/s"]];
      let prev = null;
      for (const [n, b, cy, note] of pts) {
        const p = [X(b), Y(cy)];
        if (prev) hc.line(prev[0], prev[1], p[0], p[1], { color: c.accent, lw: 2, dash: [5, 5] });
        prev = p;
      }
      for (const [n, b, cy, note] of pts) { const dy = cy === 1 ? -22 : 0; hc.dot(X(b), Y(cy), 8, c.accent); hc.text(n, X(b) + 14, Y(cy) - 8 + dy, { size: 12.5 }); hc.text(note, X(b) + 14, Y(cy) + 9 + dy, { size: 10.5, color: c["ink-3"] }); }
      hc.text("capacity →", x1, y1 + 32, { size: 10.5, color: c["ink-3"], align: "right", mono: true });
      hc.text("latency ↑ (log)", x0, 10, { size: 10.5, color: c["ink-3"], mono: true });
    });
  }

  // ---------- 4. light per clock cycle ----------
  const lt = { ghz: 1.75 };
  const lst = stats([{ label: "Light travels in one clock cycle" }, { label: "Round trip SM → HBM (~3 cm each way) at light speed" }, { label: "Measured HBM latency" }]);
  function drawL() {
    const cyc = 1 / lt.ghz, cm = 30 * cyc;
    lst.set(0, `${cm.toFixed(1)} cm (in vacuum)`); lst.set(1, `${(6 / 30 / cyc).toFixed(2)} cycles`, "good"); lst.set(2, `~550–700 cycles ≈ ${Math.round(600 * cyc)} ns`, "bad");
    ltc.paint(() => {
      ltc.clear(); const c = ltc.colors, U = 22;          // 22 px per centimetre
      ltc.box(10, 8, 780, 132, { fill: c.surface, stroke: c.line, r: 10 });
      ltc.text("Distance, to scale (1 cm = the bar below)", 24, 24, { size: 12 });
      ltc.line(24, 40, 24 + U, 40, { color: c.ink, lw: 2 }); ltc.text("1 cm", 24 + U + 6, 40, { size: 9.5, color: c["ink-3"], mono: true });
      // the package: die in the middle, HBM stacks on two sides
      const px = 30, py = 62, die = 2.85 * U;
      ltc.box(px, py, die + 2.4 * U, die, { fill: c.sunken, stroke: c.line, r: 4 });
      ltc.box(px + 1.2 * U, py, die, die, { fill: `color-mix(in srgb, ${c.c1} 40%, transparent)`, stroke: c.c1, lw: 1, r: 3 });
      ltc.text("GPU", px + 1.2 * U + die / 2, py + die / 2, { size: 10, align: "center" });
      for (const k of [0, 1, 2]) { ltc.box(px + 2, py + 2 + k * die / 3, 1.1 * U - 4, die / 3 - 4, { fill: `color-mix(in srgb, ${c.c4} 55%, transparent)`, stroke: "transparent", r: 2 }); ltc.box(px + 1.2 * U + die + 2, py + 2 + k * die / 3, 1.1 * U - 4, die / 3 - 4, { fill: `color-mix(in srgb, ${c.c4} 55%, transparent)`, stroke: "transparent", r: 2 }); }
      ltc.text("HBM", px + 0.55 * U, py + die + 10, { size: 9.5, align: "center", color: c["ink-3"] });
      // how far light gets in one cycle
      const ax = px + 1.2 * U + die / 2, reach = cm * U, end = Math.min(776, ax + reach), ay = 52;
      ltc.line(ax, ay, end, ay, { color: c.good, lw: 2.5 });
      if (ax + reach <= 776) { ltc.line(end, ay, end - 7, ay - 5, { color: c.good, lw: 2.5 }); ltc.line(end, ay, end - 7, ay + 5, { color: c.good, lw: 2.5 }); }
      else ltc.text("→ off the edge", 776, ay + 12, { size: 9.5, color: c.good, align: "right" });
      ltc.text(`light in one clock tick at ${lt.ghz.toFixed(2)} GHz: ${cm.toFixed(1)} cm`, Math.min(end, 776) - 6, ay - 12, { size: 11, color: c.good, align: "right" });
      ltc.text("The whole package is ~6 cm across: light could cross it several times per tick.", 240, 112, { size: 10.5, color: c["ink-2"] });
      // where the time really goes, on a log scale
      ltc.box(10, 150, 780, 162, { fill: c.surface, stroke: c.line, r: 10 });
      ltc.text("Waiting time in clock ticks (log scale)", 24, 166, { size: 12 });
      const x0 = 190, x1 = 770, L = (v) => x0 + ((Math.log10(v) + 1) / 4) * (x1 - x0);
      for (const t of [0.1, 1, 10, 100, 1000]) { ltc.line(L(t), 178, L(t), 300, { color: c.line, lw: 1 }); ltc.text(String(t), L(t), 304, { size: 9, color: c["ink-3"], align: "center", mono: true }); }
      const rows = [["Light: SM → HBM → SM", 6 / 30 / cyc, c.good], ["One multiply-add", 4, c.c1], ["Shared memory / L1", 30, c.c3], ["L2 cache", 300, c.c6], ["HBM (measured)", 600, c.bad]];
      rows.forEach(([nm, v, col], i) => {
        const y = 190 + i * 21;
        ltc.text(nm, x0 - 10, y + 6, { size: 10.5, align: "right", color: c["ink-2"] });
        ltc.box(L(0.1), y, Math.max(3, L(v) - L(0.1)), 12, { fill: `color-mix(in srgb, ${col} 70%, transparent)`, stroke: "transparent", r: 3 });
        ltc.text(v < 1 ? v.toFixed(2) : `~${v}`, L(v) + 6, y + 6, { size: 10, mono: true, color: c.ink });
      });
    });
  }
  const ltc = canvas(800, 320, "How far light travels in one clock tick against the size of a GPU package, and memory waiting times on a log scale");

  root.append(
    section(null,
      goal("<p>The multipliers from the last chapter need two fresh numbers every cycle, and a model's weights are billions of numbers. We need somewhere to <b>keep billions of bits</b>, and a way to <b>deliver them at terabytes per second</b>. This chapter shows how bits are stored, and the uncomfortable fact that shapes every AI chip: fetching a number from memory costs far more energy and time than doing math with it.</p>",
        ["Store tens of gigabytes", "Deliver terabytes per second", "Not spend all the power on delivery"]),
      stackMap(id),
      h("div.vx-row", sticker("dram", { arg: true, title: "A DRAM bit holding a 1", caption: "one transistor (gate on the left) and a tall capacitor full of charge", px: 4 }), sticker("dram", { arg: false, title: "…a while later", caption: "the charge leaks away, so every row is re-written every 64 ms", px: 4 }), sticker("package", { title: "A GPU with HBM", caption: "the GPU die in the middle, six stacks of DRAM dies beside it on a silicon interposer", px: 3 })),
    ),
    section("Three ways to hold a bit",
      stepper([
        { title: "A flip-flop (registers): ~20 transistors", html: "<p>Two gates feeding each other in a loop hold a value as long as power is on, and extra transistors let a clock edge capture a new one. Fastest of all, read in under a cycle, but big: this is what registers are made of. An H100 has 256 KB of registers per SM.</p>" },
        { title: "SRAM (caches, shared memory): 6 transistors", html: "<p>Two inverters in a loop (4 transistors) plus 2 access transistors. Holds its value while powered, reads in a few nanoseconds. About 0.02 µm² per bit in a modern process, and that area has almost stopped shrinking at recent nodes, so on-chip memory is precious.</p>" },
        { title: "DRAM (main memory): 1 transistor + 1 capacitor", html: "<p>The bit is a small charge on a capacitor (a few femtofarads, around 10,000–50,000 electrons) behind one access transistor: about 10× denser than SRAM. But the charge leaks, so every cell must be <b>refreshed</b>, read and rewritten, every 64 ms. Reading drains the cell too, so every read is followed by a write-back.</p>" },
        { title: "Why DRAM is a separate chip", html: "<p>Good capacitors need deep trenches or tall cylinders and a different manufacturing process from fast logic transistors. Building both on one die makes both worse, so DRAM lives on its own chips, which means the data has to leave the GPU die to get them. That's the expensive part.</p>" },
      ]),
      h("div.stage",
        h("div.stage-controls", playBtn, button("Reset", resetD),
          toggle({ label: "Refresh every 64 ms", checked: dr.refresh, onChange: (v) => { dr.refresh = v; drawD(); } }),
          slider({ label: "Temperature", min: 25, max: 95, step: 5, value: dr.temp, fmt: (v) => `${v} °C`, onInput: (v) => { dr.temp = v; drawD(); } })),
        h("div.stage-body", dst.el, h("div.wide-canvas", dc.el))),
      callout("tip", "Try it", "<p>Run time with refresh on: the yellow bars sag and get topped up. Turn refresh off and the weakest cells lose their 1s first, then more and more. Heat it to 85–95 °C and leakage roughly doubles every 10 °C, which is why DRAM above 85 °C must be refreshed twice as often (and why “Rowhammer” attacks, which disturb neighbouring cells, work better on hot, dense chips). Retention times here are illustrative.</p>"),
    ),
    section("The energy of moving a number",
      prose("<p>Here is the table that explains modern hardware. Mark Horowitz measured the energy of basic operations in a 45 nm process: adding two numbers costs a fraction of a picojoule, while reading one 32-bit number from off-chip DRAM costs <b>640 pJ</b>, about 170 times a floating-point multiply. Newer processes shrank all of these, but arithmetic shrank faster than wires, so the gap has grown, not closed.</p>"),
      h("div.stage", h("div.stage-body", h("div.wide-canvas", ec.el))),
      prose("<p>Now do one multiply-add, fetching both inputs from somewhere. The only way to make data movement cheap is <b>reuse</b>: fetch a number once and use it many times while it sits close by. A matrix multiply can do this (each number in a tile meets a whole row or column of the other matrix); an elementwise operation can't.</p>"),
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Inputs come from", options: [{ value: "dram", label: "DRAM" }, { value: "sram", label: "On-chip SRAM" }, { value: "reg", label: "Registers" }], value: en.src, onChange: (v) => { en.src = v; drawE2(); } }),
          slider({ label: "Times each fetched number is reused", min: 0, max: 3, step: 0.05, value: 0, fmt: (v) => String(Math.round(10 ** v)), onInput: (v) => { en.reuse = Math.round(10 ** v); drawE2(); } })),
        h("div.stage-body", est.el)),
      predict({
        chapter: id,
        question: "Inputs come from DRAM. How many times must each fetched number be reused before moving data costs less than the fp32 multiply-add itself (4.6 pJ)?",
        options: ["About 2", "About 30", "About 300", "About 10,000"],
        answer: 2,
        explain: "<p>Two inputs at 640 pJ each is 1,280 pJ; divide by r and compare with 4.6: r ≈ 280. Remarkably close to the H100's own balance point: 989 TFLOP/s ÷ 3.35 TB/s ≈ 295 operations per byte loaded. Below that a kernel is <b>memory-bound</b>: the multipliers wait. That's the roofline from <a href='/learn?c=kernels'>the kernels chapter</a>, and it's set by physics, not by bad engineering.</p>",
      }),
      prose("<p>See reuse happen: count every trip to DRAM in a tiny matrix multiply, naive and tiled.</p>"),
      walk("memory-physics"),
    ),
    section("Why far away means slow and costly",
      whyChain({
        claim: "Reading a number from DRAM costs ~170× the energy of multiplying it, and ~500× the time.",
        steps: [
          { q: "Why does distance cost energy?", a: "<p>A wire is a capacitor: its capacitance grows with its length, about 0.2 fF per micrometre on chip. Sending a bit means charging that wire to the supply voltage, which costs C·V². Across a millimetre of chip that's already more than an add; off the chip, through a package and a circuit board, it's hundreds of times more.</p>" },
          { q: "Why does DRAM itself cost so much per read?", a: "<p>A DRAM cell's charge is tiny, so reading it means opening a whole row (thousands of cells at once) onto long bit lines and amplifying the faint voltage difference with sense amplifiers, then writing it all back. You pay for the whole row even if you wanted 4 bytes.</p>" },
          { q: "Why is it ~500 cycles if light covers the round trip in under one?", a: "<p>Distance isn't what makes it slow. Light covers the few centimetres to the HBM and back in a fraction of a cycle. The time goes to the on-chip network hopping from the SM to the right memory controller, queueing behind other requests, and the DRAM protocol itself: opening a row takes about 15 ns, then reading the column and sending data back through several clocked stages.</p>" },
          { q: "Why can't we just build more SRAM and skip DRAM?", a: "<p>SRAM costs 6 transistors and about 10× the area of DRAM per bit. 80 GB of it would fill 15–30 of the largest dies that can be made, just for the bit cells. Cerebras does build a whole-wafer chip with 44 GB of SRAM, but for normal chips on-chip memory is tens to hundreds of megabytes, and every byte competes with arithmetic for the same silicon.</p>" },
        ],
        axiom: { name: "Charging a capacitance costs C·V², and capacitance grows with size", html: "<p>Moving information means moving charge, and moving charge onto a wire costs energy proportional to the wire's length. Nothing can carry a bit without changing some physical state, so <b>communication always costs more than computation over any real distance</b>. Every design in the next two chapters, from caches to tiling to tensor cores, is a way to move fewer bits shorter distances.</p>" },
      }),
      h("div.stage",
        h("div.stage-controls", slider({ label: "Clock speed", min: 1, max: 5, step: 0.05, value: lt.ghz, fmt: (v) => `${v.toFixed(2)} GHz`, onInput: (v) => { lt.ghz = v; drawL(); } })),
        h("div.stage-body", h("div.wide-canvas", ltc.el), lst.el)),
    ),
    section("The hierarchy: small and close, big and far",
      prose("<p>Since fast memory must be small and close, and big memory must be slow and far, every processor stacks them. Each level is roughly 10× bigger and several times slower than the one above. A good program keeps the numbers it's using right now in the top levels. These are approximate H100 figures from NVIDIA's documentation and independent microbenchmarks; latency depends on clocks and how it's measured.</p>"),
      h("div.stage", h("div.stage-body", h("div.wide-canvas", hc.el))),
      table(["Level", "Size on an H100", "Latency", "Who manages it"], [
        ["Registers", "64K × 32-bit = 256 KB per SM (33 MB total)", "~1 cycle", "the compiler"],
        ["Shared memory / L1", "256 KB per SM, up to 228 KB as shared", "~30 cycles", "you (shared) or hardware (L1)"],
        ["L2 cache", "50 MB, split in two partitions", "~300 cycles near, ~470 far", "hardware"],
        ["HBM3", "80 GB in 5 active stacks", "~550–700 cycles", "you: what to load, and when"],
      ]),
      callout("note", "HBM: going wide instead of fast", "<p>Normal DRAM sits centimetres away on a circuit board and sends data over a few dozen fast pins. HBM stacks 8 to 12 DRAM dies on top of each other, wired vertically with thousands of through-silicon vias, and sits millimetres from the GPU on a silicon interposer. Each stack has a 1,024-bit-wide interface at a modest clock. The H100's five stacks give a 5,120-bit bus: short wires and many slow lanes move more data per joule than a few fast ones. HBM is now one of the most expensive parts of an AI chip.</p>"),
    ),
    section("In code",
      code({
        "Energy per multiply-add": `E = dict(add=0.9, mul=3.7, reg=1, sram=5, dram=640)   # pJ, 45 nm (Horowitz 2014)

def pj_per_mac(source, reuse):
    return E["mul"] + E["add"] + 2 * E[source] / reuse

for reuse in [1, 10, 100, 300, 1000]:
    print(reuse, round(pj_per_mac("dram", reuse), 1))
# matmul of n x n tiles: each loaded number is reused ~n times`,
        "Measure latency (CUDA)": `// pointer chasing: each load depends on the last, so nothing overlaps
__global__ void chase(const unsigned* next, unsigned start, int steps, unsigned* out, long long* cycles) {
    unsigned p = start;
    long long t0 = clock64();
    for (int i = 0; i < steps; ++i) p = next[p];      // one dependent load at a time
    long long t1 = clock64();
    *out = p; *cycles = (t1 - t0) / steps;            // cycles per load
}
// fill next[] with a random cycle over N elements: small N fits in L1, then L2, then only HBM`,
        "Measure bandwidth (PyTorch)": `import torch, time
x = torch.empty(2**28, device="cuda", dtype=torch.float32)  # 1 GiB
y = torch.empty_like(x)
torch.cuda.synchronize(); t = time.time()
for _ in range(20): y.copy_(x)                     # read 1 GiB + write 1 GiB
torch.cuda.synchronize()
print(20 * 2 * x.numel() * 4 / (time.time() - t) / 1e9, "GB/s")   # T4: ~250; H100: ~3,000`,
      }, "Project 60 runs the bandwidth test on a Colab GPU."),
    ),
  );
  function drawE2() { drawE(); }
  resetD(); drawE(); drawH(); drawL();
}
