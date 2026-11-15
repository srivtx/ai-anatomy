import { h } from "../lib/core.js";
import { slider, segmented, button, stats, predict, code, callout, section, prose, table } from "../lib/widgets.js";
import { flowCanvas } from "../lib/flow.js";
import { xyChart } from "../lib/plot.js";
import { goal, sticker } from "../lib/why.js";

function canvas(width, height, label) {
  const fc = flowCanvas({ width, height, label });
  fc.paint = (fn) => { fc.redraw = fn; fn(); };
  return fc;
}

// ---------------------------------------------------------------------------------------------
// The numbers. An H100 SXM at its dense bf16 peak: 132 SMs × 4 tensor cores × 512 FMAs per clock × 1.83 GHz
// × 2 FLOPs = 989 TFLOP/s, at 700 W. Everything below is derived from those and labelled where estimated.
// ---------------------------------------------------------------------------------------------
const CLK = 1.83e9, SMS = 132, TC = 4 * SMS, FMA_CLK = 512, WATTS = 700;
const FMA_S = TC * FMA_CLK * CLK;                 // 4.95e14 multiply-adds per second
const E_FMA = WATTS / FMA_S;                      // ~1.4 pJ per FMA, whole chip, all-in
const C_NODE = 0.2e-15, VDD = 0.75, QE = 1.602e-19;
const E_TOGGLE = C_NODE * VDD * VDD;              // 0.11 fJ drawn from the supply per 0→1→0
const ELECTRONS = (C_NODE * VDD) / QE;            // ~940 electrons
const LANDAUER = 1.380649e-23 * 300 * Math.LN2;  // 2.87e-21 J
const N = 4096, TM = 128, TN = 256, BK = 64;
const BLOCKS = (N / TM) * (N / TN), TILE_FMA = TM * TN * N, WAVES = Math.ceil(BLOCKS / SMS);

const SI = ["", "k", "M", "G", "T", "P", "E"];
function big(x) { let i = 0; while (x >= 1000 && i < SI.length - 1) { x /= 1000; i++; } return `${x >= 100 ? x.toFixed(0) : x >= 10 ? x.toFixed(1) : x.toFixed(2)}${SI[i] ? " " + SI[i] : ""}`.replace(/\.0+(\s|$)/, "$1"); }
function unit(x, u, steps) {   // steps: [[scale, name]] largest first
  for (const [s, n] of steps) if (x >= s * 0.999) { const v = x / s; return `${v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1).replace(/\.0$/, "") : v.toFixed(2).replace(/\.?0+$/, "")} ${n}`; }
  const [s, n] = steps[steps.length - 1]; return `${(x / s).toPrecision(2)} ${n}`;
}
const SUP = "⁰¹²³⁴⁵⁶⁷⁸⁹";
const sci = (x) => { const e = Math.floor(Math.log10(x)); return `${(x / 10 ** e).toFixed(1)} × 10${String(e).split("").map((d) => SUP[+d]).join("")}`; };
const fmtM = (m) => unit(m, "m", [[1e-2, "cm"], [1e-3, "mm"], [1e-6, "µm"], [1e-9, "nm"], [1e-12, "pm"]]);
const fmtS = (s) => unit(s, "s", [[1, "s"], [1e-3, "ms"], [1e-6, "µs"], [1e-9, "ns"], [1e-12, "ps"], [1e-15, "fs"]]);
const fmtJ = (j) => unit(j, "J", [[1, "J"], [1e-3, "mJ"], [1e-6, "µJ"], [1e-9, "nJ"], [1e-12, "pJ"], [1e-15, "fJ"], [1e-18, "aJ"], [1e-21, "zJ"]]);

// ---------------------------------------------------------------------------------------------
// The ten zoom levels. size: how big the thing doing this step is; time: how long this step takes;
// energy: what it costs; count: how many run at once on one H100.
// ---------------------------------------------------------------------------------------------
const LEVELS = [
  { name: "One line of Python", short: "Python", size: 5e-2, time: WAVES * TILE_FMA / (4 * FMA_CLK * CLK), energy: N ** 3 * E_FMA, count: "1 line → 1 kernel",
    link: "ai-physics", what: `<p><code>y = x @ W</code> for one 4,096-wide layer and 4,096 tokens: <b>${big(N ** 3)} multiply-adds</b>. Python doesn't do any of them. It hands the whole job to a library (cuBLAS) that launches one kernel on the GPU and returns before it's finished. The hardware doing the work is the whole package: the die and its 5 HBM stacks, about 5 cm across.</p>`,
    why: "AI is a function with billions of knobs, so it is mostly multiply-adds. Every one of them costs energy (bedrock: thermodynamics)." },
  { name: "The kernel: a grid of blocks", short: "Kernel", size: 28.5e-3, time: TILE_FMA / (4 * FMA_CLK * CLK), energy: SMS * TILE_FMA * E_FMA, count: `${BLOCKS} blocks, ${SMS} at a time`,
    link: "cuda", what: `<p>The output is cut into <b>${BLOCKS} tiles of ${TM}×${TN}</b>, one per thread block. ${SMS} SMs take ${SMS} blocks at a time, so the grid runs in <b>${WAVES} waves</b> (each step here is one wave). The last wave has only ${BLOCKS - (WAVES - 1) * SMS} blocks, so ${WAVES * SMS - BLOCKS} SMs sit idle for it: the “tail effect”. The die is 28.5 mm on a side, 814 mm².</p>`,
    why: "Blocks must be able to run in any order, because they can't wait for each other (bedrock: deadlock)." },
  { name: "One SM: one block's tile", short: "SM", size: 1.7e-3, time: (TM * TN * BK) / (4 * FMA_CLK * CLK), energy: TM * TN * BK * E_FMA, count: `${SMS} SMs`,
    link: "gpu-design", what: `<p>One SM owns one ${TM}×${TN} tile. It walks along the shared dimension in steps of ${BK}: copy a ${TM}×${BK} slice of x and a ${BK}×${TN} slice of W (48 KB) from HBM into shared memory, then do <b>${big(TM * TN * BK)} multiply-adds</b> on it. Each number fetched is reused ${TM}–${TN} times, which is the only reason the math isn't starved. One step here is one such slice. An SM is roughly 1–2 mm across.</p>`,
    why: "Fetching a number costs far more than multiplying it, so each fetched number must be reused hundreds of times (bedrock: C·V²)." },
  { name: "One warp: one matrix instruction", short: "Warp", size: 0.7e-3, time: 2048 / (FMA_CLK * CLK), energy: 2048 * E_FMA, count: `up to ${64 * SMS} warps resident`,
    link: "gpu-design", what: "<p>32 threads issue <b>one instruction</b> together: <code>mma.sync.m16n8k16</code>, D = A·B + C on a 16×16 and a 16×8 tile, 2,048 multiply-adds. Each thread holds a few elements of each tile in its registers; the warp as a whole holds the tiles. (Hopper's fastest path, <code>wgmma</code>, has 4 warps issue a 64×256×16 product at once; same idea, bigger.) Each SM has 4 sub-partitions, each with its own scheduler, registers and tensor core.</p>",
    why: "One instruction drives 32 lanes so the cost of fetching and decoding it is shared 32 ways; thousands of warps wait in line so memory delays are hidden (bedrock: Little's law)." },
  { name: "One tensor core", short: "Tensor core", size: 0.25e-3, time: 1 / CLK, energy: FMA_CLK * E_FMA, count: `${TC} tensor cores`,
    link: "gpu-design", what: `<p>Each clock tick (0.55 ns at 1.83 GHz) a tensor core finishes <b>${FMA_CLK} multiply-adds</b>. Think of it as rows of multipliers feeding adder trees: operands come in once and are shared across a whole row, so little energy goes on moving them. NVIDIA doesn't publish its internal layout; this drawing is the idea, not the floorplan. Size is an estimate.</p>`,
    why: "Doing a small matrix product inside one unit reuses each operand many times without leaving the unit (bedrock: C·V²)." },
  { name: "One multiply-add", short: "FMA", size: 1e-5, time: 1 / CLK, energy: E_FMA, count: `${big(TC * FMA_CLK)} per clock`,
    link: "logic", what: `<p>a × b + c, with a and b in bf16 (1 sign, 8 exponent, 7 fraction bits) and c in fp32. Add the exponents, multiply the 8-bit significands, shift c to line up, add, normalise, round. A rough estimate is ~10,000 transistors per lane, about 10 µm across. The whole chip spends <b>${(E_FMA * 1e12).toFixed(1)} pJ per multiply-add</b> at peak (700 W ÷ ${big(FMA_S)} per second): that includes memory, wires, clocks and control, not just this lane.</p>`,
    why: "Numbers are bits because only two states survive noise; a multiplier's size grows with bits², so fewer bits (bf16, fp8) means far cheaper math (bedrock: information needs states noise can't confuse)." },
  { name: "One full adder", short: "Full adder", size: 0.5e-6, time: 15e-12, energy: 1e-15, count: "tens of millions",
    link: "logic", what: "<p>The 8×8 significand multiply is 64 AND gates making partial products and a few dozen <b>full adders</b> summing them. A full adder adds three bits (a, b, carry in) and gives a sum bit and a carry out. Here 1 + 1 + 0: sum 0, carry 1. Built as 2 XORs, 2 ANDs and an OR (or 9 NANDs, or 28 transistors in standard CMOS). ~15 ps, ~1 fJ: estimates.</p>",
    why: "Any logic at all can be built from NAND, and binary addition is just XOR plus carry." },
  { name: "One NAND gate", short: "NAND", size: 0.18e-6, time: 6e-12, energy: 2 * E_TOGGLE, count: "billions",
    link: "transistor", what: `<p>Four transistors. Two p-type in parallel to the supply, two n-type in series to ground. Inputs A = 1 and B = 1 switch both n-types on and both p-types off, so the output wire is drained to 0 V. The output drives a little capacitance (wire plus the next gates' inputs, ~0.2 fF), and how fast the gate can fill or empty it sets its delay: a few picoseconds.</p>`,
    why: "CMOS pairs every pull-up with a pull-down, so a gate only draws current while switching." },
  { name: "One transistor", short: "Transistor", size: 50e-9, time: 1.5e-12, energy: E_TOGGLE, count: "80 billion",
    link: "transistor", what: `<p>A FinFET on TSMC's 4N process: a silicon fin a few nm wide and ~50 nm tall, with the gate wrapped over three sides. Gate at 0 V: the fin is a barrier, off. Gate at +0.75 V: electrons gather under the gate oxide on the fin's faces and form a channel, on. 80 billion of them on 814 mm² is one per <b>~100 × 100 nm</b> on average (wiring included). Charging one ~0.2 fF node costs C·V² ≈ <b>${(E_TOGGLE * 1e18).toFixed(0)} aJ</b> from the supply.</p>`,
    why: "Current over the barrier grows only 10× per 60 mV of gate, so the supply can't drop below ~0.7 V (bedrock: the Boltzmann factor)." },
  { name: "The electrons", short: "Electrons", size: 0.5e-9, time: 0.15e-12, energy: VDD * QE, count: `~${Math.round(ELECTRONS / 10) * 10} per bit`,
    link: "electrons", what: `<p>Turning that wire from 0 to 1 means moving <b>~${Math.round(ELECTRONS / 10) * 10} electrons</b> (Q = C·V = 0.2 fF × 0.75 V). That's it: a bit is a crowd of under a thousand electrons sitting on a wire or not. They cross a ~15 nm channel in ~0.15 ps. Silicon atoms are 0.235 nm apart, so the fin is only ~25 atoms wide. The physics floor for erasing a bit (Landauer: kT ln 2 = 2.9 zJ) is <b>~${(Math.round(E_TOGGLE / LANDAUER / 1000) * 1000).toLocaleString("en")}× smaller</b> than what we spend.</p>`,
    why: "Electrons fill energy bands in pairs (Pauli), which leaves silicon a gap that heat, dopants and a gate voltage can bridge (bedrock: quantum mechanics)." },
];

// ---------------------------------------------------------------------------------------------
// The eight bedrocks from the level, and the GPU features each one forces
// ---------------------------------------------------------------------------------------------
const AX = [
  ["thermo", "Energy is conserved; heat jiggles electrons", "ai-physics"],
  ["pauli", "Pauli exclusion: electrons fill bands", "electrons"],
  ["boltz", "Boltzmann factor e^(−E/kT)", "transistor"],
  ["noise", "Information needs states noise can't confuse", "logic"],
  ["cv2", "Charging a wire costs C·V²", "memory-physics"],
  ["second", "Heat only flows downhill", "power-wall"],
  ["little", "Little's law: in flight = rate × delay", "gpu-design"],
  ["dead", "Circular waiting never ends", "cuda"],
];
const FEAT = [
  ["cmos", "80 billion silicon CMOS switches"],
  ["volt", "Supply near 0.7 V, power capped at 700 W"],
  ["wide", "Thousands of slow lanes at ~1.8 GHz"],
  ["lowp", "Low precision: bf16, fp8, fp4"],
  ["warp", "32 lanes share one instruction"],
  ["tc", "Tensor cores: reuse inside the unit"],
  ["smem", "Registers and shared memory beside the math"],
  ["hbm", "HBM stacked millimetres from the die"],
  ["flight", "Tens of thousands of threads in flight"],
  ["blocks", "Independent blocks; sync between kernels"],
];
const EDGES = {
  thermo: [["volt", "every multiply-add ends as heat, so the chip is sized to a power budget"], ["lowp", "fewer bits flipped per number means less energy per number"]],
  pauli: [["cmos", "the band gap is what lets silicon be both conductor and insulator, which is what a switch is"]],
  boltz: [["volt", "60 mV per 10× of current puts a floor under the threshold, and so under the supply"], ["wide", "with the voltage stuck, a faster clock costs heat the chip can't shed"]],
  noise: [["cmos", "digital gates restore 0 and 1 at every stage, so noise never adds up"], ["lowp", "a multiplier's area and energy grow as bits², so fp8 is ~16× cheaper than fp32"]],
  cv2: [["lowp", "fewer bits per number means fewer wires to charge when moving it"], ["warp", "fetching and decoding an instruction is shared by 32 lanes"], ["tc", "operands are moved once and used many times inside the unit"], ["smem", "a short wire to a register costs a tiny fraction of a long one to DRAM"], ["hbm", "millimetres of wire instead of centimetres of board"]],
  second: [["volt", "heat must flow out through the lid, at most ~100 W/cm²"], ["wide", "the only way to use more transistors is to run many of them slowly"]],
  little: [["flight", "to keep 3.35 TB/s flowing through ~500 ns of delay, ~1.7 MB must be in flight"], ["smem", "a 256 KB register file per SM holds all those waiting threads, so switching is free"]],
  dead: [["blocks", "a block waiting for one that isn't scheduled would wait forever, so blocks never wait for each other"]],
};

export default function (root, { id }) {
  // ---------- 1. the zoom ----------
  const zm = { i: 0 };
  const zc = canvas(880, 410, "Zooming from a line of Python to electrons");
  const zst = stats([{ label: "Size of the thing" }, { label: "Time for one step" }, { label: "Energy for one step" }, { label: "How many at once (H100)" }]);
  const zwhat = h("div.zoom-what");
  const zsl = slider({ label: "Zoom level", min: 0, max: LEVELS.length - 1, value: 0, fmt: (v) => `${v + 1}. ${LEVELS[v].short}`, onInput: (v) => go(v) });
  const zin = button("Zoom in ↓", () => go(zm.i + 1), "primary"), zout = button("↑ Zoom out", () => go(zm.i - 1));
  function go(i) {
    zm.i = Math.max(0, Math.min(LEVELS.length - 1, i));
    zsl.value = zm.i;
    zin.disabled = zm.i === LEVELS.length - 1; zout.disabled = zm.i === 0;
    drawZ();
  }
  function ruler(y, label, lo, hi, val, all, fmt) {
    const c = zc.colors, x0 = 120, x1 = 860, X = (v) => x0 + ((Math.log10(v) - lo) / (hi - lo)) * (x1 - x0);
    zc.text(label, 20, y, { size: 11, color: c["ink-3"] });
    zc.line(x0, y, x1, y, { color: c.line, lw: 2 });
    for (let d = lo; d <= hi; d++) { zc.line(X(10 ** d), y - 4, X(10 ** d), y + 4, { color: c.line }); if ((d - lo) % 2 === 0) zc.text(fmt(10 ** d), X(10 ** d), y + 14, { size: 9.5, align: "center", color: c["ink-3"], mono: true }); }
    all.forEach((v) => zc.dot(X(v), y, 3, c["ink-3"]));
    zc.dot(X(val), y, 7, c.accent);
    zc.text(fmt(val), X(val), y - 15, { size: 11.5, align: "center", color: c.ink, mono: true });
  }
  function drawZ() {
    const L = LEVELS[zm.i];
    zst.set(0, fmtM(L.size)); zst.set(1, fmtS(L.time)); zst.set(2, fmtJ(L.energy)); zst.set(3, L.count);
    zwhat.innerHTML = `<h4>${zm.i + 1}. ${L.name}</h4>${L.what}<p class="muted"><b>Why it's built this way:</b> ${L.why} <a href="/learn?c=${L.link}">Dig into it →</a></p>`;
    zc.paint(() => {
      zc.clear();
      ruler(28, "size", -10, -1, L.size, LEVELS.map((l) => l.size), fmtM);
      ruler(72, "time per step", -13, -3, L.time, LEVELS.map((l) => l.time), fmtS);
      zc.line(20, 98, 860, 98, { color: zc.colors.line, lw: 1, dash: [3, 4] });
      DRAW[zm.i](zc, 20, 108, 840, 296);
    });
  }

  // ---------- 2. one token ----------
  const tk = { n: 70e9, b: 1 };
  const MODELS = [{ value: 8e9, label: "8B" }, { value: 70e9, label: "70B" }, { value: 405e9, label: "405B" }];
  const HBM_J_BYTE = 4e-12 * 8, IDLE_W = 150, BW = 3.35e12, PEAK = 989e12;
  function token(n, b) {
    const gpus = Math.ceil((2 * n) / 70e9), step = Math.max((2 * n) / (gpus * BW), (b * 2 * n) / (gpus * PEAK));
    const math = n * E_FMA, mem = (2 * n * HBM_J_BYTE) / b, idle = (gpus * IDLE_W * step) / b;
    return { gpus, step, math, mem, idle, total: math + mem + idle, bound: (2 * n) / (gpus * BW) >= (b * 2 * n) / (gpus * PEAK) ? "memory" : "math" };
  }
  const tst = stats([{ label: "Multiply-adds per token" }, { label: "H100s to hold the weights" }, { label: "Time per step (all users)" }, { label: "Energy per token" }, { label: "Electrons through 0.75 V" }]);
  const tchart = xyChart({ width: 880, height: 260, xLog: true, yLog: true, xLabel: "users sharing each pass over the weights (batch)", yLabel: "joules per token", xDomain: [1, 1024], yDomain: [1e-3, 100] });
  const tnote = h("p.muted");
  function drawT() {
    const r = token(tk.n, tk.b);
    tst.set(0, big(tk.n)); tst.set(1, String(r.gpus)); tst.set(2, `${fmtS(r.step)} (${r.bound}-bound)`, r.bound === "math" ? "good" : "warn");
    tst.set(3, fmtJ(r.total), r.total < 0.3 ? "good" : r.total < 3 ? "warn" : "bad"); tst.set(4, sci(r.total / (VDD * QE)));
    const xs = Array.from({ length: 41 }, (_, i) => 2 ** (i / 4));
    tchart.update([
      { name: "Total", color: "var(--accent)", points: xs.map((b) => [b, token(tk.n, b).total]) },
      { name: "Reading the weights from HBM", color: "var(--c2)", points: xs.map((b) => [b, token(tk.n, b).mem]), dash: true },
      { name: "The multiply-adds", color: "var(--c3)", points: xs.map((b) => [b, token(tk.n, b).math]), dash: true },
      { name: "Staying on (idle power, leakage)", color: "var(--c5)", points: xs.map((b) => [b, token(tk.n, b).idle]), dash: true },
    ], { xDomain: [1, 1024], yDomain: [1e-3, 100], marks: [{ x: tk.b, label: `batch ${tk.b}` }] });
    const share = r.mem / r.total;
    tnote.innerHTML = `At batch ${tk.b}, <b>${Math.round(share * 100)}%</b> of each token's energy goes on moving weights, ${Math.round((r.idle / r.total) * 100)}% on just being switched on and ${Math.round((r.math / r.total) * 100)}% on the arithmetic. A phone battery (~15 Wh = 54 kJ) would pay for about <b>${big(54e3 / r.total)}</b> tokens at this rate.`;
  }

  // ---------- 3. the eight bedrocks ----------
  const ax = { k: "cv2" };
  const gc = canvas(880, 420, "Eight laws of physics and logic, and the GPU features they force");
  const gtext = h("div.ax-text");
  const axBtns = AX.map(([k, t]) => h("button.btn.small", { type: "button", "aria-pressed": k === ax.k ? "true" : "false", onclick: () => { ax.k = k; drawG(); } }, t.split(":")[0].replace(/ e\^.*/, "")));
  function drawG() {
    axBtns.forEach((b, i) => b.setAttribute("aria-pressed", AX[i][0] === ax.k ? "true" : "false"));
    const on = new Set((EDGES[ax.k] || []).map(([f]) => f)), axis = AX.find((a) => a[0] === ax.k);
    gtext.innerHTML = `<p><b>${axis[1]}</b> forces:</p><ul>${EDGES[ax.k].map(([f, why]) => `<li><b>${FEAT.find((x) => x[0] === f)[1]}</b>: ${why}.</li>`).join("")}</ul><p class="muted">Where it came from: <a href="/learn?c=${axis[2]}">the ${axis[2].replace("-", " ")} chapter</a>'s “why?” chain.</p>`;
    gc.paint(() => {
      gc.clear(); const c = gc.colors, ay = (i) => 18 + i * 49, fy = (i) => 14 + i * 40.5;
      gc.text("BEDROCK", 20, 8, { size: 10, color: c["ink-3"] }); gc.text("WHAT IT FORCES IN A GPU", 520, 8, { size: 10, color: c["ink-3"] });
      for (const [k, list] of Object.entries(EDGES)) {
        const a = AX.findIndex((x) => x[0] === k);
        for (const [f] of list) {
          const j = FEAT.findIndex((x) => x[0] === f), sel = k === ax.k;
          gc.line(330, ay(a) + 17, 520, fy(j) + 15, { color: sel ? c.accent : `color-mix(in srgb, ${c.line} 70%, transparent)`, lw: sel ? 2.5 : 1 });
        }
      }
      AX.forEach(([k, t], i) => {
        const sel = k === ax.k;
        gc.box(20, ay(i), 310, 34, { fill: sel ? `color-mix(in srgb, ${c.accent} 18%, ${c.surface})` : c.surface, stroke: sel ? c.accent : c.line, lw: sel ? 2 : 1.2, r: 8 });
        gc.text(t, 32, ay(i) + 17, { size: 12, color: sel ? c.ink : c["ink-2"] });
      });
      FEAT.forEach(([k, t], j) => {
        const lit = on.has(k);
        gc.box(520, fy(j), 340, 30, { fill: lit ? `color-mix(in srgb, ${c.good} 16%, ${c.surface})` : c.surface, stroke: lit ? c.good : c.line, lw: lit ? 2 : 1.2, r: 8 });
        gc.text(t, 532, fy(j) + 15, { size: 12, color: lit ? c.ink : c["ink-3"] });
      });
    });
  }

  root.append(
    section(null,
      goal("<p>You've dug down eight times, one layer per chapter. Now go back up in one go. We follow <b>one multiply-add</b> from a line of Python to the electrons that carry it, with real numbers for an H100 at every step: how big, how fast, how much energy, how many at once. The goal is to see the whole machine as one thing, and to see which law of physics shaped each part.</p>",
        ["See every layer at its real scale", "Know what one token really costs, and where the energy goes", "Name the law behind each part of a GPU"]),
      h("div.vx-row",
        sticker("package", { title: "The package", caption: "level 1: ~5 cm", px: 3 }),
        sticker("chip", { title: "The die", caption: "level 2: 28.5 mm", px: 3 }),
        sticker("finfet", { title: "A FinFET", caption: "level 9: ~50 nm", px: 3 }),
        sticker("lattice", { arg: "n", title: "The crystal", caption: "level 10: atoms 0.235 nm apart", px: 3 })),
    ),
    section("Zoom: ten levels, one multiply-add",
      prose(`<p>Start at the top and zoom in. The two rulers are logarithmic: each tick is 10×. The whole trip spans <b>8 orders of magnitude in size</b> (5 cm to under a nanometre) and <b>9 in time</b>. The numbers come from the H100's published specs (${SMS} SMs, ${TC} tensor cores, 989 TFLOP/s dense bf16, 700 W) and simple physics; the ones NVIDIA doesn't publish are marked as estimates.</p>`),
      h("div.stage",
        h("div.stage-controls", zsl, h("div.btn-row", zout, zin)),
        h("div.stage-body", zst.el, h("div.wide-canvas", zc.el), zwhat)),
      callout("tip", "Read the rulers", "<p>Going down, every level is smaller and faster, but there are vastly more of them. 80 billion transistors each switching in picoseconds is why one chip does ~10¹⁵ multiply-adds per second. And the energy column tells the real story: the arithmetic in one lane is cheap, but the chip as a whole spends ~1.4 pJ per multiply-add, mostly moving data and keeping everything clocked. That gap is why most of a GPU's design is about not moving things.</p>"),
    ),
    section("What one token costs",
      prose("<p>A language model produces each token by running every weight once: one multiply-add per weight. Pick a model and how many users share each pass over the weights. The model counts three costs: the arithmetic (~1.4 pJ per multiply-add, the chip-wide figure), reading the weights from HBM (~4 pJ per bit, a typical HBM3 figure), and the chip's power just for being on (~150 W per GPU here, an assumption: clocks, leakage, idle units).</p>"),
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Model (bf16 weights)", options: MODELS, value: tk.n, onChange: (v) => { tk.n = v; drawT(); } }),
          slider({ label: "Batch: users per pass", min: 0, max: 10, value: 0, fmt: (v) => String(2 ** v), onInput: (v) => { tk.b = 2 ** v; drawT(); } })),
        h("div.stage-body", tst.el, tchart.el, tnote)),
      callout("note", "What the chart shows", "<p>At batch 1 the GPU reads every weight from HBM to make one token, and moving those bytes (2 per weight) costs ~45× more than multiplying them, and the chip's idle power is paid for the whole slow pass. Share each read across many users and the moving cost per token falls in proportion, until the arithmetic itself is the floor. This is why serving systems batch hard (<a href='/learn?c=memory-physics'>memory physics</a>), and why the tensor cores sit idle during single-user chat. Real measured energy per token is higher than this model: networking, CPUs, cooling and imperfect kernels add more.</p>"),
      predict({
        chapter: id,
        question: "One user chats with a 70B model on H100s (batch 1). What limits how fast tokens come out?",
        options: ["The tensor cores' 989 TFLOP/s", "Reading 140 GB of weights from HBM for every token", "The PCIe link to the CPU", "The clock speed"],
        answer: 1,
        explain: "<p>Each token needs every weight once: 140 GB in bf16. Split across 2 H100s at 3.35 TB/s each, that's ~21 ms per token, ~48 tokens/s. The arithmetic for that token (140 GFLOP) would take the tensor cores ~0.07 ms. The math units are idle >99% of the time; memory bandwidth is the speed limit. Batching, quantisation and speculative decoding all attack this.</p>",
      }),
    ),
    section("Eight laws, one machine",
      prose("<p>Each chapter's “why?” chain ended at a law nobody can engineer around. Here they are side by side, with the GPU features each one forces. Pick a law to light up its consequences. Notice how many arrows leave <b>C·V²</b>: most of a GPU is a machine for not moving data.</p>"),
      h("div.stage",
        h("div.stage-controls", h("div.btn-row", axBtns)),
        h("div.stage-body", h("div.wide-canvas", gc.el), gtext)),
      table(["Layer", "The question we kept asking", "Where “why?” stopped"], [
        ["The job", "Why do AI chips burn so much power?", "Thermodynamics: every bit operation ends as heat"],
        ["The material", "Why can silicon be switched at all?", "Pauli exclusion leaves a band gap"],
        ["The switch", "Why not just lower the voltage?", "The Boltzmann factor: 60 mV per 10× of current"],
        ["The arithmetic", "Why binary?", "Two well-separated states are the most noise can't blur"],
        ["The bottleneck", "Why is memory slow and costly?", "C·V²: a longer wire is a bigger capacitor"],
        ["The constraint", "Why did clocks stop at ~4 GHz?", "Heat only flows downhill, through a finite path"],
        ["The machine", "Why does a GPU need 50,000 threads?", "Little's law: in flight = rate × delay"],
        ["The program", "Why can't blocks wait for each other?", "Circular waiting never ends"],
      ]),
      predict({
        chapter: id,
        question: "Today's chips spend ~0.1 fJ charging a wire to store one bit. How far is that above the physics floor for erasing a bit (Landauer, kT ln 2 at room temperature)?",
        options: ["About 2×", "About 40×", "About 40,000×", "About 10¹²×"],
        answer: 2,
        explain: `<p>kT ln 2 at 300 K is 2.9 × 10⁻²¹ J. A 0.2 fF node at 0.75 V costs C·V² ≈ 1.1 × 10⁻¹⁶ J: ~${(Math.round(E_TOGGLE / LANDAUER / 1000) * 1000).toLocaleString("en")}× more. There's huge room in principle, but the Boltzmann factor keeps the voltage from falling much further, and the wire's capacitance can't shrink much more either. The remaining headroom is mostly in <i>not moving</i> data, which is where new architectures attack.</p>`,
      }),
      predict({
        chapter: id,
        question: "Roughly how many electrons make the difference between a 0 and a 1 on a wire inside an H100?",
        options: ["1", "About 10", "About 1,000", "About a billion"],
        answer: 2,
        explain: `<p>Q = C·V. A typical node (wire plus the gates it drives) is ~0.2 fF; at 0.75 V that's 1.5 × 10⁻¹⁶ coulombs, ~${Math.round(ELECTRONS / 10) * 10} electrons. Few enough that random thermal motion of single electrons matters for the smallest transistors, many enough that a 0 and a 1 are still unmistakable.</p>`,
      }),
    ),
    section("Where to go from here",
      prose("<p>You now have the whole stack, from the job down to electrons and back. Each next step goes deeper at one layer:</p>"),
      table(["If you want to…", "Go to"], [
        ["Write fast kernels: tiling, coalescing, bank conflicts, roofline", "<a href='/learn?c=gpu-from-scratch'>GPU kernels from scratch</a>"],
        ["See why serving is memory-bound and how batching and KV caches work", "<a href='/learn?c=memory-physics'>Why memory is far and slow</a>, then the Systems level"],
        ["Measure all of this on a real GPU", "Colab project 60: from electrons to CUDA, measured"],
        ["Start again from the top with the AlexNet story", "<a href='/learn?c=ai-physics'>What AI asks of the electrons</a>"],
      ], { boldFirst: false }),
    ),
    section("In code",
      code({
        "The zoom numbers": `# Everything from an H100 SXM's published specs plus Q = CV and E = CV²
SMS, TC_PER_SM, FMA_PER_CLK, CLK, WATTS = 132, 4, 512, 1.83e9, 700
fma_per_s = SMS * TC_PER_SM * FMA_PER_CLK * CLK          # 4.95e14, ×2 = 989 TFLOP/s
print(f"{WATTS / fma_per_s * 1e12:.2f} pJ per multiply-add, whole chip")

N, TM, TN = 4096, 128, 256
blocks = (N // TM) * (N // TN)                           # 512 tiles of 128×256
waves = -(-blocks // SMS)                                # 4 waves on 132 SMs
t_tile = TM * TN * N / (TC_PER_SM * FMA_PER_CLK * CLK)   # one SM's tile: ~36 µs
print(f"{blocks} blocks in {waves} waves: {waves * t_tile * 1e6:.0f} µs vs ideal {N**3 / fma_per_s * 1e6:.0f} µs")

C, V, q, kT = 0.2e-15, 0.75, 1.602e-19, 1.380649e-23 * 300
print(f"{C * V / q:.0f} electrons per bit, {C * V * V * 1e18:.0f} aJ per toggle")
print(f"{C * V * V / (kT * 0.6931):,.0f}x the Landauer limit")
print(f"footprint per transistor: {(814e-6 / 80e9) ** 0.5 * 1e9:.0f} nm on a side")`,
        "Energy per token": `def joules_per_token(params, batch, hbm_pj_per_bit=4, fma_pj=1.41, idle_w=150):
    gpus = -(-2 * params // 70e9)                         # bf16 weights, ~70 GB usable each
    t_mem  = 2 * params / (gpus * 3.35e12)                # read every weight once per step
    t_math = batch * 2 * params / (gpus * 989e12)
    step = max(t_mem, t_math)
    math = params * fma_pj * 1e-12                        # one multiply-add per weight per token
    move = 2 * params * 8 * hbm_pj_per_bit * 1e-12 / batch
    idle = gpus * idle_w * step / batch
    return math + move + idle, step

for b in [1, 8, 64, 256]:
    j, step = joules_per_token(70e9, b)
    print(f"batch {b:>3}: {j:.3f} J/token, {b / step:,.0f} tokens/s")`,
      }),
    ),
  );
  go(0); drawT(); drawG();
}

// ---------------------------------------------------------------------------------------------
// One drawing per zoom level, inside the box (x, y, w, h)
// ---------------------------------------------------------------------------------------------
const tint = (c, k, p) => `color-mix(in srgb, ${c[k]} ${p}%, ${c.surface})`;
function grid(cv, x, y, cols, rows, cw, ch, fill, stroke) {
  for (let r = 0; r < rows; r++) for (let q = 0; q < cols; q++) cv.box(x + q * cw, y + r * ch, cw - 1.5, ch - 1.5, { fill: typeof fill === "function" ? fill(r, q) : fill, stroke: stroke || "transparent", lw: 1, r: Math.min(3, cw / 4) });
}
function zoomTo(cv, fx, fy, fw, fh, tx, ty, tw, th) {   // dashed lines from a highlighted region to where the next level is drawn
  const c = cv.colors;
  cv.box(fx, fy, fw, fh, { fill: "transparent", stroke: c.accent, lw: 2.5, r: 3 });
  cv.line(fx + fw, fy, tx, ty, { color: c.accent, lw: 1, dash: [4, 4] });
  cv.line(fx + fw, fy + fh, tx, ty + th, { color: c.accent, lw: 1, dash: [4, 4] });
}
function bits(cv, x, y, groups, cell, label) {
  const c = cv.colors; let xx = x;
  for (const [n, k, name] of groups) { for (let i = 0; i < n; i++) cv.box(xx + i * cell, y, cell - 2, cell, { fill: tint(c, k, 45), stroke: "transparent", r: 2 }); cv.text(name, xx + (n * cell) / 2, y + cell + 10, { size: 9.5, align: "center", color: c["ink-3"] }); xx += n * cell + 4; }
  cv.text(label, x - 8, y + cell / 2, { size: 12, align: "right", mono: true });
}
function gate(cv, x, y, name, on) {
  const c = cv.colors;
  cv.box(x, y, 70, 40, { fill: on ? tint(c, "accent", 22) : c.surface, stroke: on ? c.accent : c.line, lw: 2, r: 10 });
  cv.text(name, x + 35, y + 20, { size: 12.5, align: "center" });
}
function wire(cv, pts, on) {
  const c = cv.colors;
  for (let i = 1; i < pts.length; i++) cv.line(pts[i - 1][0], pts[i - 1][1], pts[i][0], pts[i][1], { color: on ? c.accent : c["ink-3"], lw: on ? 2.5 : 1.5 });
}

const DRAW = [
  // 1. python
  (cv, x, y) => {
    const c = cv.colors, s = 150, gy = y + 60;
    cv.box(x + 10, y + 4, 520, 34, { fill: c.sunken, stroke: "transparent", r: 8 });
    cv.text("y = x @ W        # 4096 tokens × 4096 features", x + 24, y + 21, { size: 13, mono: true });
    const mats = [["x", "4096 tokens × 4096", "c1"], ["W", "4096 × 4096 weights", "c2"], ["y", "4096 × 4096 outputs", "c3"]];
    mats.forEach(([n, d, k], i) => {
      const mx = x + 20 + i * 230;
      cv.box(mx, gy, s, s, { fill: tint(c, k, 30), stroke: c[k], lw: 1.5, r: 4 });
      cv.text(n, mx + s / 2, gy + s / 2, { size: 26, align: "center", weight: 700 });
      cv.text(d, mx + s / 2, gy + s + 16, { size: 11, align: "center", color: c["ink-2"] });
      if (i < 2) cv.text(i ? "=" : "@", mx + s + 40, gy + s / 2, { size: 24, align: "center", color: c["ink-2"] });
    });
    zoomTo(cv, x + 480, gy, s / 32 * 2, s / 32, x + 700, gy + 10, 130, 120);
    cv.box(x + 700, gy + 10, 130, 120, { fill: c.surface, stroke: c.line, r: 8 });
    cv.text("each output =", x + 765, gy + 42, { size: 11, align: "center", color: c["ink-2"] });
    cv.text("4,096 multiply-adds", x + 765, gy + 62, { size: 12, align: "center" });
    cv.text("× 16.8 million outputs", x + 765, gy + 82, { size: 11, align: "center", color: c["ink-2"] });
    cv.text(`= ${big(N ** 3)}`, x + 765, gy + 106, { size: 13, align: "center", weight: 700, mono: true });
    cv.text("Python hands all of it to one GPU kernel and moves on", x + 20, y + 270, { size: 12, color: c["ink-3"] });
  },
  // 2. kernel grid and waves
  (cv, x, y) => {
    const c = cv.colors, cw = 30, ch = 8.4, cols = N / TN, rows = N / TM, gx = x + 20, gy = y + 6, wk = ["c1", "c2", "c3", "c4"];
    grid(cv, gx, gy, cols, rows, cw, ch, (r, q) => tint(c, wk[Math.floor((r * cols + q) / SMS)], 55));
    cv.box(gx - 1, gy - 1, cw + 1, ch + 1, { fill: "transparent", stroke: c.ink, lw: 2, r: 2 });
    cv.text(`y split into ${rows} × ${cols} = ${BLOCKS} tiles of ${TM}×${TN}; colour = which wave runs it`, gx, gy + rows * ch + 14, { size: 11, color: c["ink-2"] });
    for (let w = 0; w < WAVES; w++) {
      const n = Math.min(SMS, BLOCKS - w * SMS), ly = gy + 20 + w * 56;
      cv.box(x + 520, ly, 18, 18, { fill: tint(c, wk[w], 55), stroke: "transparent", r: 4 });
      cv.text(`wave ${w + 1}: ${n} blocks`, x + 548, ly + 9, { size: 12.5 });
      cv.bar(x + 548, ly + 26, 270, 9, n / SMS, c[wk[w]]);
      cv.text(n < SMS ? `${SMS - n} of ${SMS} SMs idle: the tail` : `all ${SMS} SMs busy`, x + 548, ly + 46, { size: 10.5, color: c["ink-3"] });
    }
  },
  // 3. SM
  (cv, x, y) => {
    const c = cv.colors, sx = x + 20, sy = y + 6, sw = 560, sh = 280;
    cv.box(sx, sy, sw, sh, { fill: c.sunken, stroke: c.line, r: 12 });
    cv.text("one SM", sx + 14, sy + 14, { size: 11, color: c["ink-3"] });
    for (let q = 0; q < 4; q++) {
      const qx = sx + 14 + (q % 2) * 270, qy = sy + 28 + Math.floor(q / 2) * 112;
      cv.box(qx, qy, 260, 104, { fill: c.surface, stroke: q === 0 ? c.accent : c.line, lw: q === 0 ? 2.5 : 1.2, r: 8 });
      cv.text(`sub-partition ${q}`, qx + 10, qy + 13, { size: 10.5, color: c["ink-3"] });
      cv.box(qx + 10, qy + 26, 100, 22, { fill: tint(c, "c5", 35), stroke: "transparent", r: 5 }); cv.text("warp scheduler", qx + 60, qy + 37, { size: 10, align: "center" });
      cv.box(qx + 10, qy + 54, 100, 40, { fill: tint(c, "c2", 35), stroke: "transparent", r: 5 }); cv.text("registers 64 KB", qx + 60, qy + 74, { size: 10, align: "center" });
      cv.box(qx + 120, qy + 26, 130, 34, { fill: tint(c, "accent", 40), stroke: "transparent", r: 5 }); cv.text("tensor core", qx + 185, qy + 43, { size: 11, align: "center" });
      grid(cv, qx + 120, qy + 66, 16, 2, 8.1, 13, tint(c, "c3", 45)); cv.text("32 FP32 lanes", qx + 185, qy + 98, { size: 9, align: "center", color: c["ink-3"] });
    }
    cv.box(sx + 14, sy + 256 - 10, sw - 28, 26, { fill: tint(c, "c1", 35), stroke: "transparent", r: 6 });
    cv.text("256 KB L1 / shared memory: the slices of x and W this block is working on", sx + sw / 2, sy + 259, { size: 11, align: "center" });
    cv.text("from L2 and HBM:", x + 610, y + 30, { size: 11, color: c["ink-3"] });
    cv.box(x + 610, y + 44, 64, 128, { fill: tint(c, "c1", 35), stroke: c.c1, r: 4 }); cv.text(`x ${TM}×${BK}`, x + 642, y + 108, { size: 11, align: "center" });
    cv.box(x + 690, y + 44, 140, 36, { fill: tint(c, "c2", 35), stroke: c.c2, r: 4 }); cv.text(`W ${BK}×${TN}`, x + 760, y + 62, { size: 11, align: "center" });
    cv.text("48 KB in,", x + 690, y + 112, { size: 12 }); cv.text(`${big(TM * TN * BK)} multiply-adds`, x + 690, y + 132, { size: 12 }); cv.text("each number used 128–256×", x + 690, y + 152, { size: 11, color: c["ink-2"] });
    cv.line(x + 610, y + 186, sx + sw, sy + 259, { color: c.c1, lw: 1.5, dash: [4, 4] });
  },
  // 4. warp + mma
  (cv, x, y) => {
    const c = cv.colors;
    cv.text("32 threads of one warp (each holds a few elements of A, B and C in its registers)", x + 20, y + 12, { size: 11.5, color: c["ink-2"] });
    for (let t = 0; t < 32; t++) { const tx = x + 20 + (t % 16) * 50, ty = y + 26 + Math.floor(t / 16) * 34; cv.box(tx, ty, 44, 28, { fill: tint(c, "c5", 30), stroke: "transparent", r: 5 }); cv.text(`t${t}`, tx + 22, ty + 14, { size: 10, align: "center", mono: true }); }
    cv.text("one instruction: mma.sync.m16n8k16   D = A · B + C", x + 20, y + 110, { size: 13, mono: true });
    const gy = y + 128, cell = 8.5;
    grid(cv, x + 40, gy, 16, 16, cell, cell, tint(c, "c1", 50)); cv.text("A 16×16", x + 40 + 68, gy + 146, { size: 11, align: "center" });
    cv.text("×", x + 200, gy + 68, { size: 20, align: "center", color: c["ink-2"] });
    grid(cv, x + 220, gy, 8, 16, cell, cell, tint(c, "c2", 50)); cv.text("B 16×8", x + 254, gy + 146, { size: 11, align: "center" });
    cv.text("+", x + 310, gy + 68, { size: 20, align: "center", color: c["ink-2"] });
    grid(cv, x + 330, gy, 8, 16, cell, cell, tint(c, "c3", 50)); cv.text("C 16×8", x + 364, gy + 146, { size: 11, align: "center" });
    cv.text("→", x + 425, gy + 68, { size: 20, align: "center", color: c["ink-2"] });
    cv.box(x + 460, gy + 20, 180, 100, { fill: tint(c, "accent", 30), stroke: c.accent, lw: 2, r: 10 });
    cv.text("tensor core", x + 550, gy + 50, { size: 13, align: "center" });
    cv.text("2,048 multiply-adds", x + 550, gy + 72, { size: 11.5, align: "center", color: c["ink-2"] });
    cv.text("≈ 4 clocks", x + 550, gy + 92, { size: 11.5, align: "center", color: c["ink-2"] });
    cv.text("one instruction fetched and decoded,", x + 670, gy + 50, { size: 11 }); cv.text("32 threads' worth of work", x + 670, gy + 68, { size: 11 }); cv.text("and 2,048 multiply-adds", x + 670, gy + 86, { size: 11 });
  },
  // 5. tensor core
  (cv, x, y) => {
    const c = cv.colors, cw = 13, ch = 15, gx = x + 30, gy = y + 24;
    cv.text("512 multipliers (illustrative layout: 16 dot-product rows of 32)", gx, y + 8, { size: 11.5, color: c["ink-2"] });
    grid(cv, gx, gy, 32, 16, cw, ch, (r, q) => (r === 3 && q === 5 ? c.accent : tint(c, "c3", 45)));
    for (let r = 0; r < 16; r++) {
      const ry = gy + r * ch + ch / 2 - 1;
      cv.line(gx + 32 * cw + 2, ry, gx + 32 * cw + 40, ry, { color: c["ink-3"], lw: 1 });
      cv.box(gx + 32 * cw + 40, ry - 6, 40, 12, { fill: tint(c, "c2", 40), stroke: "transparent", r: 3 });
      cv.text("Σ", gx + 32 * cw + 60, ry, { size: 9.5, align: "center" });
      cv.line(gx + 32 * cw + 80, ry, gx + 32 * cw + 100, ry, { color: c["ink-3"], lw: 1 });
      cv.box(gx + 32 * cw + 100, ry - 6, 26, 12, { fill: tint(c, "c1", 40), stroke: "transparent", r: 3 });
    }
    cv.text("adder trees", gx + 32 * cw + 60, gy + 16 * ch + 12, { size: 10, align: "center", color: c["ink-3"] });
    cv.text("acc", gx + 32 * cw + 113, gy + 16 * ch + 12, { size: 10, align: "center", color: c["ink-3"] });
    const tx = x + 630;
    cv.text("Each row multiplies 32 pairs", tx, y + 50, { size: 11.5 });
    cv.text("and sums them in a tree, every clock.", tx, y + 68, { size: 11.5 });
    cv.text("A and B arrive once and are shared", tx, y + 104, { size: 11.5, color: c["ink-2"] });
    cv.text("across rows and columns, so energy", tx, y + 122, { size: 11.5, color: c["ink-2"] });
    cv.text("goes on multiplying, not on moving.", tx, y + 140, { size: 11.5, color: c["ink-2"] });
    cv.text("highlighted: one multiplier → next level", tx, y + 180, { size: 11, color: c.accent });
  },
  // 6. one FMA lane
  (cv, x, y) => {
    const c = cv.colors, cell = 13;
    bits(cv, x + 70, y + 8, [[1, "c5", "sign"], [8, "c2", "exponent"], [7, "c1", "fraction"]], cell, "a");
    bits(cv, x + 70, y + 46, [[1, "c5", ""], [8, "c2", ""], [7, "c1", ""]], cell, "b");
    bits(cv, x + 380, y + 8, [[1, "c5", "sign"], [8, "c2", "exponent"], [23, "c1", "fraction (fp32)"]], 11, "c");
    const st = [["add exponents", "8-bit adder", "c2"], ["multiply significands", "8 × 8 bits", "accent"], ["line up c", "shift", "c4"], ["add", "~24-bit adder", "c3"], ["normalise, round", "→ fp32 result", "c5"]];
    st.forEach(([a, b, k], i) => {
      const sx = x + 10 + i * 166, sy = y + 104;
      cv.box(sx, sy, 150, 56, { fill: tint(c, k, k === "accent" ? 28 : 22), stroke: k === "accent" ? c.accent : c.line, lw: k === "accent" ? 2.5 : 1.2, r: 10 });
      cv.text(a, sx + 75, sy + 21, { size: 12, align: "center" }); cv.text(b, sx + 75, sy + 40, { size: 10.5, align: "center", color: c["ink-2"] });
      if (i < st.length - 1) cv.text("→", sx + 158, sy + 28, { size: 14, align: "center", color: c["ink-3"] });
    });
    const mx = x + 206, my = y + 186;
    for (let r = 0; r < 8; r++) for (let q = 0; q < 8; q++) cv.dot(mx + (7 - q) * 12 + r * 12, my + r * 12 + 6, 3.6, r === 2 && q === 3 ? c.accent : tint(c, "accent", 50));
    cv.text("the 8×8 multiply: 64 AND gates make partial products (dots),", x + 400, my + 30, { size: 11.5 });
    cv.text("and a few dozen full adders sum each column", x + 400, my + 48, { size: 11.5 });
    cv.text("highlighted: one full adder → next level", x + 400, my + 74, { size: 11, color: c.accent });
  },
  // 7. full adder
  (cv, x, y) => {
    const c = cv.colors, ix = x + 96, a = 1, b = 1, ci = 0, x1 = a ^ b, s = x1 ^ ci, a1 = a & b, a2 = x1 & ci, co = a1 | a2;
    const ins = [["a", a, y + 40], ["b", b, y + 100], ["carry in", ci, y + 190]];
    ins.forEach(([n, v, yy]) => { cv.text(`${n} = ${v}`, ix - 8, yy, { size: 13, align: "right", mono: true }); });
    gate(cv, x + 160, y + 50, "XOR", x1); gate(cv, x + 360, y + 110, "XOR", s);
    gate(cv, x + 160, y + 160, "AND", a1); gate(cv, x + 360, y + 210, "AND", a2); gate(cv, x + 540, y + 180, "OR", co);
    wire(cv, [[ix, y + 40], [x + 120, y + 40], [x + 120, y + 62], [x + 160, y + 62]], a); wire(cv, [[ix, y + 100], [x + 130, y + 100], [x + 130, y + 80], [x + 160, y + 80]], b);
    wire(cv, [[x + 120, y + 62], [x + 120, y + 172], [x + 160, y + 172]], a); wire(cv, [[x + 130, y + 100], [x + 130, y + 190], [x + 160, y + 190]], b);
    wire(cv, [[x + 230, y + 70], [x + 300, y + 70], [x + 300, y + 122], [x + 360, y + 122]], x1);
    wire(cv, [[x + 300, y + 122], [x + 300, y + 222], [x + 360, y + 222]], x1);
    wire(cv, [[ix, y + 190], [x + 104, y + 190], [x + 104, y + 260], [x + 330, y + 260], [x + 330, y + 240], [x + 360, y + 240]], ci);
    wire(cv, [[x + 330, y + 240], [x + 330, y + 140], [x + 360, y + 140]], ci);
    wire(cv, [[x + 230, y + 180], [x + 500, y + 180], [x + 500, y + 192], [x + 540, y + 192]], a1);
    wire(cv, [[x + 430, y + 230], [x + 500, y + 230], [x + 500, y + 208], [x + 540, y + 208]], a2);
    wire(cv, [[x + 430, y + 130], [x + 700, y + 130]], s); wire(cv, [[x + 610, y + 200], [x + 700, y + 200]], co);
    cv.text(`sum = ${s}`, x + 710, y + 130, { size: 13, mono: true }); cv.text(`carry out = ${co}`, x + 710, y + 200, { size: 13, mono: true });
    cv.text("1 + 1 + 0 = 10 in binary. Lit wires carry a 1.", x + 160, y + 20, { size: 11.5, color: c["ink-2"] });
    cv.box(x + 156, y + 156, 78, 48, { fill: "transparent", stroke: c.accent, lw: 1.5, r: 12 });
    cv.text("AND = NAND then NOT → next level", x + 160, y + 288, { size: 11, color: c.accent });
  },
  // 8. NAND gate
  (cv, x, y) => {
    const c = cv.colors, cx = x + 300, top = y + 20, bot = y + 280, out = y + 130;
    cv.line(cx - 140, top, cx + 140, top, { color: c.ink, lw: 3 }); cv.text("supply 0.75 V", cx + 150, top, { size: 11.5 });
    cv.line(cx - 140, bot, cx + 140, bot, { color: c.ink, lw: 3 }); cv.text("ground 0 V", cx + 150, bot, { size: 11.5 });
    function fet(fx, fy, kind, on, label) {
      cv.box(fx - 16, fy - 22, 32, 44, { fill: on ? tint(c, "accent", 35) : c.surface, stroke: on ? c.accent : c.line, lw: 2, r: 6 });
      cv.text(kind, fx, fy, { size: 11, align: "center" });
      cv.line(fx - 34, fy, fx - 18, fy, { color: c.ink, lw: 2 }); cv.text(label, fx - 40, fy, { size: 12, align: "right", mono: true });
      cv.text(on ? "on" : "off", fx + 24, fy, { size: 10.5, color: on ? c.accent : c["ink-3"] });
    }
    cv.line(cx - 60, top, cx - 60, y + 50, { color: c["ink-3"], lw: 2 }); cv.line(cx + 60, top, cx + 60, y + 50, { color: c["ink-3"], lw: 2 });
    fet(cx - 60, y + 72, "p", false, "A=1"); fet(cx + 60, y + 72, "p", false, "B=1");
    cv.line(cx - 60, y + 94, cx - 60, out, { color: c["ink-3"], lw: 2 }); cv.line(cx + 60, y + 94, cx + 60, out, { color: c["ink-3"], lw: 2 });
    cv.line(cx - 60, out, cx + 60, out, { color: c.accent, lw: 2.5 });
    cv.line(cx, out, cx, y + 150, { color: c.accent, lw: 2.5 }); fet(cx, y + 172, "n", true, "A=1");
    cv.line(cx, y + 194, cx, y + 214, { color: c.accent, lw: 2.5 }); fet(cx, y + 236, "n", true, "B=1");
    cv.line(cx, y + 258, cx, bot, { color: c.accent, lw: 2.5 });
    cv.line(cx + 60, out, cx + 200, out, { color: c.accent, lw: 2.5 }); cv.text("out = 0", cx + 210, out, { size: 13, mono: true });
    cv.line(cx + 180, out, cx + 180, out + 40, { color: c["ink-3"] }); cv.line(cx + 166, out + 40, cx + 194, out + 40, { color: c.ink, lw: 2 }); cv.line(cx + 166, out + 48, cx + 194, out + 48, { color: c.ink, lw: 2 });
    cv.text("C ≈ 0.2 fF (wire + next gates)", cx + 204, out + 44, { size: 11, color: c["ink-2"] });
    cv.text("both n-types on: a path to ground drains the output's ~940 electrons' worth of charge", x + 20, y + 300 - 6, { size: 11, color: c["ink-3"] });
    cv.box(cx - 20, y + 210, 40, 52, { fill: "transparent", stroke: c.accent, lw: 1.5, r: 8 });
    cv.text("one transistor → next level", cx - 40, y + 252, { size: 10.5, align: "right", color: c.accent });
  },
  // 9. FinFET cross-section
  (cv, x, y) => {
    const c = cv.colors, bx = x + 120, by = y + 240;
    cv.box(bx - 80, by, 460, 40, { fill: tint(c, "ink-3", 25), stroke: "transparent", r: 4 }); cv.text("silicon substrate", bx - 64, by + 20, { size: 11 });
    cv.box(bx + 40, by - 14, 220, 14, { fill: tint(c, "ink-3", 12), stroke: "transparent", r: 2 }); cv.text("isolation oxide", bx + 300, by - 7, { size: 10, color: c["ink-3"] });
    cv.box(bx + 70, y + 30, 160, 200, { fill: tint(c, "warn", 40), stroke: c.warn, lw: 2, r: 12 }); cv.text("gate metal", bx + 150, y + 48, { size: 11, align: "center" });
    cv.box(bx + 126, y + 70, 48, 172, { fill: c.surface, stroke: c.ink, lw: 1, r: 3 });
    cv.box(bx + 132, y + 76, 36, 166, { fill: tint(c, "c1", 40), stroke: "transparent", r: 2 }); cv.text("fin", bx + 150, y + 200, { size: 11, align: "center" });
    for (let i = 0; i < 14; i++) { cv.dot(bx + 135, y + 84 + i * 10, 2.4, c.accent); cv.dot(bx + 165, y + 84 + i * 10, 2.4, c.accent); }
    for (let i = 0; i < 3; i++) cv.dot(bx + 141 + i * 9, y + 81, 2.4, c.accent);
    cv.text("electrons gather on the fin's three faces (gate on)", bx + 150, y + 14, { size: 11, align: "center", color: c.accent });
    const dim = (x1, y1, x2, y2, t, tx, ty) => { cv.line(x1, y1, x2, y2, { color: c.ink, lw: 1 }); cv.text(t, tx, ty, { size: 11, mono: true }); };
    dim(bx + 132, by + 50, bx + 168, by + 50, "", 0, 0); cv.line(bx + 132, by + 44, bx + 132, by + 56, { color: c.ink, lw: 1 }); cv.line(bx + 168, by + 44, bx + 168, by + 56, { color: c.ink, lw: 1 });
    cv.text("fin ≈ 6 nm ≈ 25 atoms wide (drawn wider)", bx + 178, by + 50, { size: 11 });
    dim(bx + 300, y + 76, bx + 300, by, "≈ 50 nm tall", bx + 310, y + 160);
    cv.text("gate oxide ~1 nm (hafnium oxide)", bx + 250, y + 100, { size: 10.5, color: c["ink-2"] });
    cv.line(bx + 172, y + 100, bx + 246, y + 100, { color: c["ink-3"], lw: 1 });
    cv.text("Looking along the channel: current flows into the page,", x + 520, y + 60, { size: 11.5 });
    cv.text("from source to drain, under the gate.", x + 520, y + 78, { size: 11.5 });
    cv.text("80 billion of these on 814 mm²:", x + 520, y + 120, { size: 11.5, color: c["ink-2"] });
    cv.text("one per ~100 × 100 nm on average.", x + 520, y + 138, { size: 11.5, color: c["ink-2"] });
    cv.text("highlighted dots: the electrons → next level", x + 520, y + 180, { size: 11, color: c.accent });
  },
  // 10. electrons
  (cv, x, y) => {
    const c = cv.colors, cols = 47, rows = 20, gx = x + 20, gy = y + 30, step = 8.6;
    cv.text(`Each dot is one electron: ${cols * rows} of them, the charge that makes one wire read 1 instead of 0`, gx, y + 10, { size: 11.5, color: c["ink-2"] });
    cv.box(gx - 8, gy - 8, cols * step + 14, rows * step + 14, { fill: c.sunken, stroke: c.line, r: 8 });
    let seed = 7; const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
    for (let r = 0; r < rows; r++) for (let q = 0; q < cols; q++) cv.dot(gx + q * step + (rnd() - 0.5) * 3, gy + r * step + (rnd() - 0.5) * 3, 2.2, c.accent);
    const ex = x + 450, ey = y + 40, lo = -21, hi = -15, X = (e) => ex + ((Math.log10(e) - lo) / (hi - lo)) * 360;
    cv.text("energy, log scale (each tick 10×)", ex, ey, { size: 11, color: c["ink-3"] });
    for (let d = lo; d <= hi; d++) cv.line(X(10 ** d), ey + 14, X(10 ** d), ey + 200, { color: c.line, lw: 1, dash: [2, 4] });
    [["Landauer floor kT ln 2", LANDAUER, "good"], ["kT: one jiggle of heat", 4.14e-21, "c5"], ["one electron through 0.75 V", VDD * QE, "c2"], ["one wire toggled: C·V²", E_TOGGLE, "accent"], ["chip-wide, per multiply-add", E_FMA, "bad"]].forEach(([n, e, k], i) => {
      const yy = ey + 30 + i * 36;
      cv.bar(ex, yy, Math.min(400, X(Math.min(e, 10 ** hi)) - ex + 8), 12, 1, c[k]);
      cv.text(`${n}: ${fmtJ(e)}`, ex, yy + 24, { size: 11 });
    });
    cv.text(`the bottom of the stack: ${(Math.round(E_TOGGLE / LANDAUER / 1000) * 1000).toLocaleString("en")}× above the floor physics allows`, x + 20, y + 246, { size: 12.5 });
    cv.text("(the per-FMA bar runs off the end: 1.4 pJ is another ~3 ticks right)", ex, y + 280, { size: 10.5, color: c["ink-3"] });
  },
];
