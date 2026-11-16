import { h, loop, reducedMotion } from "../lib/core.js";
import { launchLab } from "../lib/labs.js";
import { slider, stats, predict, code, callout, section, prose, table, segmented, stepper, toggle, button } from "../lib/widgets.js";
import { flowCanvas } from "../lib/flow.js";
import { goal, whyChain, stackMap, sticker } from "../lib/why.js";
import { walk } from "../lib/walks.js";

function canvas(width, height, label) {
  const fc = flowCanvas({ width, height, label });
  fc.paint = (fn) => { fc.redraw = fn; fn(); };
  return fc;
}
const sup = (e) => String(e).split("").map((d) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[+d]).join("");
const sciS = (v) => { const x = 10 ** v, e = Math.floor(v + 1e-9), m = x / 10 ** e; return `${m < 1.05 ? "" : `${m.toFixed(1)} × `}10${sup(e)}`; };
const fmtT = (s) => (s >= 1 ? `${s.toFixed(2)} s` : s >= 1e-3 ? `${(s * 1e3).toFixed(2)} ms` : `${(s * 1e6).toFixed(1)} µs`);

export default function (root, { id }) {
  // ---------- 0a. the life of a CUDA program, to scale ----------
  let lstep = 0;
  const lf = canvas(760, 250, "A timeline of one CUDA program: CPU, PCIe link and GPU, drawn to scale");
  // n = 1M floats: copy a and b in (8 MB at ~25 GB/s), launch (~5 µs), kernel (12 MB at ~3 TB/s), copy c out (4 MB)
  const PH = [
    { lane: 0, t0: 0, t1: 10, txt: "cudaMalloc", st: 1 },
    { lane: 1, t0: 10, t1: 330, txt: "a, b → GPU: 8 MB at ~25 GB/s = 320 µs", st: 1 },
    { lane: 0, t0: 10, t1: 330, txt: "waits for the copy", st: 1, wait: true },
    { lane: 0, t0: 330, t1: 335, txt: "launch", st: 2 },
    { lane: 0, t0: 335, t1: 380, txt: "free to do other work", st: 2, idle: true },
    { lane: 2, t0: 338, t1: 342, txt: "kernel: 4 µs", st: 3 },
    { lane: 0, t0: 380, t1: 502, txt: "cudaMemcpy waits", st: 4, wait: true },
    { lane: 1, t0: 342, t1: 502, txt: "c → CPU: 4 MB = 160 µs", st: 4 },
  ];
  lf.paint(() => {
    lf.clear(); const c = lf.colors, s = lstep, ac = c.accent;
    const L = ["CPU (host)", "PCIe link", "GPU (device)"], x0 = 272, x1 = 744, T = 510, X = (t) => x0 + (t / T) * (x1 - x0), Y = (l) => 44 + l * 52;
    // build time
    lf.box(12, 20, 160, 172, { fill: s === 0 ? `color-mix(in srgb, ${ac} 10%, ${c.surface})` : c.surface, stroke: s === 0 ? ac : c.line, lw: s === 0 ? 2 : 1, r: 8 });
    lf.text("before it runs: nvcc", 92, 34, { size: 10, color: c["ink-3"], align: "center", mono: true });
    lf.box(52, 46, 80, 24, { fill: c.sunken, stroke: c.line, r: 4 }); lf.text("add.cu", 92, 58, { size: 10.5, align: "center", mono: true });
    lf.line(76, 70, 52, 96, { color: c["ink-3"] }); lf.line(108, 70, 132, 96, { color: c["ink-3"] });
    lf.box(18, 96, 70, 24, { fill: `color-mix(in srgb, ${c.c1} 25%, ${c.surface})`, stroke: "transparent", r: 4 }); lf.text("host C++", 53, 108, { size: 10, align: "center" });
    lf.box(96, 96, 70, 24, { fill: `color-mix(in srgb, ${c.c3} 30%, ${c.surface})`, stroke: "transparent", r: 4 }); lf.text("kernel", 131, 108, { size: 10, align: "center" });
    lf.line(131, 120, 131, 136, { color: c["ink-3"] }); lf.text("PTX → SASS", 131, 146, { size: 9.5, align: "center", mono: true, color: c["ink-2"] });
    lf.text("one binary", 92, 176, { size: 10, align: "center", color: c["ink-3"] });
    // lanes
    for (let l = 0; l < 3; l++) { lf.text(L[l], x0 - 8, Y(l) + 14, { size: 11, align: "right" }); lf.box(x0, Y(l), x1 - x0, 28, { fill: c.sunken, stroke: "transparent", r: 5 }); }
    for (const p of PH) {
      const cur = p.st === s, past = p.st < s || cur, col = p.lane === 2 ? c.c3 : p.lane === 1 ? c.c4 : c.c1;
      if (!past) continue;
      const x = X(p.t0), w = Math.max(3, X(p.t1) - X(p.t0));
      const fill = p.wait ? `color-mix(in srgb, ${c["ink-3"]} 18%, ${c.surface})` : p.idle ? c.surface : `color-mix(in srgb, ${col} ${cur ? 60 : 30}%, ${c.surface})`;
      lf.box(x, Y(p.lane) + 2, w, 24, { fill, stroke: cur ? ac : p.idle ? c.line : "transparent", lw: cur ? 2 : 1, r: 4 });
      if (w > 60) lf.text(p.txt, x + w / 2, Y(p.lane) + 14, { size: 10, align: "center", color: p.wait || p.idle ? c["ink-3"] : c.ink });
      else if (p.lane === 2) { lf.line(x + 2, Y(2) + 28, x + 2, Y(2) + 36, { color: cur ? ac : c["ink-3"] }); lf.text(p.txt + " (the actual work)", x + 6, Y(2) + 42, { size: 10.5, align: "right", color: cur ? ac : c.ink }); }
    }
    for (let t = 0; t <= 500; t += 100) { lf.line(X(t), 206, X(t), 211, { color: c["ink-3"] }); lf.text(`${t} µs`, X(t), 220, { size: 9.5, color: c["ink-3"], align: "center", mono: true }); }
    const cap = ["nvcc splits one file into CPU code and GPU code, ahead of time", "the copy in dominates: PCIe is ~100× slower than HBM", "the launch returns at once; the CPU carries on", "the kernel itself: a few microseconds, a sliver of the total", "copy back: the CPU blocks until the kernel and the copy finish"][s];
    lf.text(cap, 470, 240, { size: 11.5, align: "center" });
  });

  // ---------- 0b. why blocks can't wait for each other: a live deadlock ----------
  const dl = { mode: "wait" };
  const dk = canvas(760, 250, "Blocks queued for 4 SMs with 2 slots each, under three rules");
  const dst = stats([{ label: "Blocks finished" }, { label: "Still queued" }, { label: "State" }]);
  let D;
  function resetD() {
    an.stop(); const n = dl.mode === "coop" ? 8 : 24;
    D = { n, queue: [...Array(n).keys()], slots: Array(8).fill(null), done: [], arrived: 0, t: 0, state: "ready" };
    statD(); dk.redraw?.();
  }
  const work = (b) => 0.8 + ((b * 37) % 11) / 20;            // blocks take slightly different times
  function tickD(dt) {
    D.t += dt;
    for (let k = 0; k < 8; k++) if (!D.slots[k] && D.queue.length) D.slots[k] = { b: D.queue.shift(), p: 0, wait: false };
    for (const sl of D.slots) {
      if (!sl || sl.wait) continue;
      sl.p += dt * 1.6 / work(sl.b);
      if (sl.p >= 1) { sl.p = 1; if (dl.mode === "free") sl.fin = true; else { sl.wait = true; D.arrived++; } }
    }
    if (dl.mode !== "free" && D.arrived === D.n) for (const sl of D.slots) if (sl) sl.fin = true;   // barrier releases
    D.slots = D.slots.map((sl) => { if (sl?.fin) { D.done.push(sl.b); return null; } return sl; });
    if (D.done.length === D.n) D.state = "done";
    else if (D.slots.every((sl) => !sl || sl.wait) && D.slots.some(Boolean) && D.queue.length) D.state = "deadlock";
    else D.state = "running";
  }
  function statD() {
    dst.set(0, `${D.done.length} of ${D.n}`, D.done.length === D.n ? "good" : "");
    dst.set(1, String(D.queue.length));
    dst.set(2, { ready: "press Run", running: "running", done: "finished ✓", deadlock: "stuck forever ✕" }[D.state], { done: "good", deadlock: "bad" }[D.state] || "");
  }
  const an = loop((dt) => { for (let k = 0; k < 3; k++) tickD(dt / 3); statD(); dk.redraw(); if (D.state === "done" || D.state === "deadlock") an.stop(); }, dk.el);
  function runD() {
    if (D.state === "done" || D.state === "deadlock") resetD();
    if (reducedMotion()) { D.state = "running"; let g = 0; while (D.state === "running" && g++ < 20000) tickD(0.02); statD(); dk.redraw(); return; }
    D.state = "running"; an.start();
  }
  dk.paint(() => {
    if (!D) return;
    dk.clear(); const c = dk.colors;
    const blk = (x, y, b, fill, txt = c.ink) => { dk.box(x, y, 26, 22, { fill, stroke: "transparent", r: 4 }); dk.text(String(b), x + 13, y + 11, { size: 10, align: "center", color: txt, mono: true }); };
    // queue
    dk.box(10, 10, 200, 214, { fill: c.surface, stroke: c.line, r: 10 }); dk.text("waiting to start", 20, 24, { size: 10.5, color: c["ink-3"], mono: true });
    D.queue.forEach((b, i) => blk(20 + (i % 6) * 30, 38 + Math.floor(i / 6) * 28, b, `color-mix(in srgb, ${c.c1} 30%, ${c.surface})`));
    if (!D.queue.length) dk.text("(empty)", 110, 120, { size: 11, color: c["ink-3"], align: "center" });
    // SMs
    for (let m = 0; m < 4; m++) {
      const x = 222 + m * 106, y = 10;
      dk.box(x, y, 98, 214, { fill: c.surface, stroke: c.line, r: 10 }); dk.text(`SM ${m}`, x + 49, y + 16, { size: 11, align: "center" });
      for (let k = 0; k < 2; k++) {
        const sl = D.slots[m * 2 + k], sy = y + 34 + k * 88;
        dk.box(x + 8, sy, 82, 78, { fill: c.sunken, stroke: sl?.wait ? (D.state === "deadlock" ? c.bad : c.warn) : c.line, lw: sl?.wait ? 2 : 1, r: 7 });
        if (!sl) { dk.text("free slot", x + 49, sy + 39, { size: 9.5, color: c["ink-3"], align: "center" }); continue; }
        blk(x + 36, sy + 8, sl.b, `color-mix(in srgb, ${c.c3} 45%, ${c.surface})`);
        dk.bar(x + 16, sy + 40, 66, 7, sl.p, sl.wait ? c.warn : c.c3);
        dk.text(sl.wait ? (D.state === "deadlock" ? "waiting forever" : "at the barrier") : "working", x + 49, sy + 62, { size: 9.5, align: "center", color: sl.wait ? (D.state === "deadlock" ? c.bad : c.warn) : c["ink-2"] });
      }
    }
    // finished
    dk.box(650, 10, 100, 214, { fill: c.surface, stroke: c.line, r: 10 }); dk.text("finished", 660, 24, { size: 10.5, color: c["ink-3"], mono: true });
    D.done.forEach((b, i) => blk(660 + (i % 3) * 29, 38 + Math.floor(i / 3) * 24, b, `color-mix(in srgb, ${c.good} 35%, ${c.surface})`));
    const msg = { ready: dl.mode === "coop" ? "8 blocks: no more than fit at once, so all are running together" : `${D.n} blocks for 8 slots: they must run in waves`, running: "", done: dl.mode === "free" ? "each block finished on its own and freed its slot for the next" : "every block was running at once, so the barrier could release", deadlock: "the 8 running blocks wait for blocks that can only start when a slot frees up: deadlock" }[D.state];
    dk.text(msg, 380, 240, { size: 11.5, align: "center", color: D.state === "deadlock" ? c.bad : c.ink });
  });
  resetD();

  // ---------- 1. grid, blocks, threads, SMs ----------
  const gm = { n: 300, bd: 64, sms: 4, perSM: 2 };
  const gc = canvas(880, 300, "An array split into blocks, and blocks scheduled onto SMs in waves");
  const gst = stats([{ label: "Blocks in the grid" }, { label: "Threads launched (idle ones)" }, { label: "Waves of blocks" }, { label: "SM slots used in the last wave" }]);
  const COLS = ["c1", "c2", "c3", "c4", "c5", "c6"];
  function drawG() {
    const blocks = Math.ceil(gm.n / gm.bd), slots = gm.sms * gm.perSM, waves = Math.ceil(blocks / slots), last = blocks - (waves - 1) * slots;
    gst.set(0, `${blocks} = ⌈${gm.n} ÷ ${gm.bd}⌉`); gst.set(1, `${blocks * gm.bd} (${blocks * gm.bd - gm.n} idle)`, blocks * gm.bd - gm.n > gm.bd / 2 ? "warn" : "good");
    gst.set(2, String(waves)); gst.set(3, `${last} of ${slots} (${Math.round((last / slots) * 100)}%)`, last / slots > 0.75 ? "good" : last / slots > 0.4 ? "warn" : "bad");
    gc.paint(() => {
      gc.clear(); const c = gc.colors, total = blocks * gm.bd, per = 64, cw = 800 / per, ch = 9;
      gc.text("THE ARRAY: one thread per element, coloured by block (grey = past the end)", 40, 12, { size: 10.5, color: c["ink-3"], mono: true });
      for (let t = 0; t < total; t++) {
        const r = Math.floor(t / per), q = t % per;
        if (r > 8) break;
        gc.box(40 + q * cw + 0.5, 24 + r * (ch + 2), cw - 1, ch, { fill: t < gm.n ? c[COLS[Math.floor(t / gm.bd) % COLS.length]] : c.line, stroke: "transparent", r: 1.5 });
      }
      if (total > 9 * per) gc.text(`… ${total - 9 * per} more threads`, 40, 24 + 9 * (ch + 2) + 6, { size: 10.5, color: c["ink-3"] });
      const y0 = 150;
      gc.text(`THE GPU: ${gm.sms} SMs × ${gm.perSM} resident blocks each. Blocks are handed out in waves, in no promised order.`, 40, y0 - 12, { size: 10.5, color: c["ink-3"], mono: true });
      const sw = Math.min(190, 800 / gm.sms - 10);
      for (let s = 0; s < gm.sms; s++) {
        const x = 40 + s * (sw + 10);
        gc.box(x, y0, sw, 120, { fill: c.sunken, stroke: c.line, r: 8 });
        gc.text(`SM ${s}`, x + 8, y0 + 12, { size: 11, color: c["ink-2"] });
        for (let k = 0; k < gm.perSM; k++) {
          const lane = s * gm.perSM + k, mine = []; for (let b = lane; b < blocks; b += slots) mine.push(b);
          const bw = (sw - 16) / gm.perSM;
          mine.slice(0, 6).forEach((b, wv) => {
            gc.box(x + 8 + k * bw, y0 + 24 + wv * 15, bw - 4, 12, { fill: c[COLS[b % COLS.length]], stroke: "transparent", r: 3 });
            gc.text(`block ${b}`, x + 8 + k * bw + (bw - 4) / 2, y0 + 30 + wv * 15, { size: 9, color: "#fff", align: "center", mono: true });
          });
          if (mine.length > 6) gc.text(`+${mine.length - 6}`, x + 8 + k * bw + 4, y0 + 24 + 6 * 15 + 2, { size: 9.5, color: c["ink-3"] });
        }
      }
      gc.text("↓ wave 1, wave 2, … each slot takes its next block as soon as it's free", 40, 288, { size: 10.5, color: c["ink-3"] });
    });
  }

  // ---------- 2. is the GPU worth the trip? ----------
  const wt = { n: 7, flops: 10, resident: false };
  const wst = stats([{ label: "CPU (16 cores, ~1 TFLOP/s, 100 GB/s)" }, { label: "GPU total" }, { label: "…of which PCIe copies" }, { label: "…of which launch overhead" }, { label: "Winner" }]);
  function drawW() {
    const N = 10 ** wt.n, F = wt.flops, B = 12;                                      // two float inputs and one output per element
    const cpu = Math.max((N * F) / 1e12, (N * B) / 100e9);
    const copy = wt.resident ? 0 : (N * 8) / 25e9 + (N * 4) / 25e9, launch = 5e-6;
    const kern = Math.max((N * F) / 50e12, (N * B) / 3e12), gpu = launch + copy + kern;
    wst.set(0, fmtT(cpu)); wst.set(1, fmtT(gpu)); wst.set(2, fmtT(copy), copy > kern * 3 ? "bad" : "");
    wst.set(3, fmtT(launch), launch > kern ? "warn" : "");
    wst.set(4, gpu < cpu ? `GPU, ${(cpu / gpu).toFixed(1)}× faster` : `CPU, ${(gpu / cpu).toFixed(1)}× faster`, gpu < cpu ? "good" : "bad");
    const times = (n) => {
      const c1 = Math.max((n * F) / 1e12, (n * B) / 100e9), cp = wt.resident ? 0 : (n * 12) / 25e9;
      return [c1, launch + cp + Math.max((n * F) / 50e12, (n * B) / 3e12)];
    };
    wc.paint(() => {
      wc.clear(); const c = wc.colors;
      // left: one run, drawn to scale
      wc.box(10, 8, 420, 264, { fill: c.surface, stroke: c.line, r: 10 });
      wc.text("One run, to scale", 24, 26, { size: 12 });
      const T = Math.max(cpu, gpu) * 1.02, x0 = 70, x1 = 412, X = (t) => x0 + (t / T) * (x1 - x0);
      wc.text("CPU", 24, 76, { size: 11 }); wc.box(x0, 66, Math.max(2, X(cpu) - x0), 20, { fill: `color-mix(in srgb, ${c.c1} 70%, transparent)`, stroke: "transparent", r: 3 });
      wc.text(fmtT(cpu), Math.min(X(cpu) + 6, 412), 98, { size: 10, mono: true, align: X(cpu) > 340 ? "right" : "left" });
      wc.text("GPU", 24, 140, { size: 11 });
      const cin = wt.resident ? 0 : (N * 8) / 25e9, cout = wt.resident ? 0 : (N * 4) / 25e9;
      const segs = [["copy in", cin, c.c6], ["launch", launch, c.warn], ["kernel", kern, c.good], ["copy out", cout, c.c6]];
      let t = 0;
      for (const [, d, col] of segs) { if (d > 0) wc.box(X(t), 130, Math.max(1.5, X(t + d) - X(t) - 1), 20, { fill: `color-mix(in srgb, ${col} 70%, transparent)`, stroke: "transparent", r: 3 }); t += d; }
      wc.text(fmtT(gpu), Math.min(X(gpu) + 6, 412), 162, { size: 10, mono: true, align: X(gpu) > 340 ? "right" : "left" });
      let kx = 24;
      for (const [nm, d, col] of segs.slice(0, 3)) { wc.box(kx, 196, 10, 10, { fill: `color-mix(in srgb, ${col} 70%, transparent)`, stroke: "transparent", r: 2 }); wc.text(nm === "copy in" ? "PCIe copies" : nm, kx + 15, 201, { size: 10, color: c["ink-2"] }); kx += nm === "copy in" ? 100 : 80; }
      const share = (cin + cout) / gpu;
      wc.text(wt.resident ? "data already on the GPU: no copies" : `copies are ${(share * 100).toFixed(0)}% of the GPU's time`, 24, 236, { size: 11, color: share > 0.5 ? c.bad : c["ink-2"] });
      wc.text(gpu < cpu ? `GPU wins by ${(cpu / gpu).toFixed(1)}×` : `CPU wins by ${(gpu / cpu).toFixed(1)}×`, 24, 256, { size: 12, color: gpu < cpu ? c.good : c.bad });
      // right: time against problem size, log-log, so the crossover shows
      wc.box(442, 8, 348, 264, { fill: c.surface, stroke: c.line, r: 10 });
      wc.text("Time against problem size", 456, 26, { size: 12 });
      const lx0 = 490, lx1 = 776, ly0 = 236, ly1 = 44, ns = [];
      for (let e = 2; e <= 9; e += 0.1) ns.push(10 ** e);
      const all = ns.flatMap(times), lo = Math.log10(Math.min(...all)), hi = Math.log10(Math.max(...all));
      const LX = (n) => lx0 + ((Math.log10(n) - 2) / 7) * (lx1 - lx0), LY = (v) => ly0 - ((Math.log10(v) - lo) / (hi - lo)) * (ly0 - ly1);
      for (const e of [2, 4, 6, 8]) { wc.line(LX(10 ** e), ly1, LX(10 ** e), ly0, { color: c.line, lw: 1 }); wc.text(`10${"⁰¹²³⁴⁵⁶⁷⁸⁹"[e]}`, LX(10 ** e), ly0 + 12, { size: 9.5, color: c["ink-3"], align: "center", mono: true }); }
      wc.text("elements →", lx1, ly0 + 26, { size: 9.5, color: c["ink-3"], align: "right" });
      wc.text("time ↑ (log)", lx0 - 4, ly1 - 8, { size: 9.5, color: c["ink-3"] });
      [[0, c.c1, "CPU"], [1, c.good, "GPU"]].forEach(([k, col, nm]) => {
        const ctx = wc.ctx; ctx.beginPath(); ns.forEach((n, i) => { const y = LY(times(n)[k]); i ? ctx.lineTo(LX(n), y) : ctx.moveTo(LX(n), y); });
        ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.stroke();
        const yl = LY(times(100)[k]); const below = yl > LY(times(100)[1 - k]), yt = below && yl + 12 > ly0 - 10 ? yl - 12 : yl + (below ? 12 : -12); wc.text(nm, lx0 + (below && yt < yl ? 40 : 4), yt, { size: 10.5, color: col });
      });
      wc.line(LX(N), ly1, LX(N), ly0, { color: c.accent, lw: 1.2, dash: [3, 3] });
      wc.dot(LX(N), LY(cpu), 4.5, c.c1); wc.dot(LX(N), LY(gpu), 4.5, c.good);
      wc.text("you", LX(N) + 5, ly1 + 6, { size: 10, color: c.accent });
    });
  }
  const wc = canvas(800, 280, "CPU against GPU time for one run, split into copies, launch and kernel, and both times against problem size");

  // ---------- ways to program it: pick a goal, see which layer fits ----------
  const WAYS = [
    ["PyTorch / JAX", 1, 2, "a line of Python"],
    ["torch.compile", 1.2, 3, "one decorator"],
    ["Triton", 4, 6, "a block-level kernel in Python"],
    ["CUDA C++", 7, 8.5, "one thread's code, shared memory, sync"],
    ["CUTLASS / CuTe", 9, 10, "tile templates for tensor cores"],
  ];
  const GOALS = [["model", "Train a standard model", 0], ["fuse", "Speed up a chain of small ops", 1], ["custom", "A new fused op (e.g. attention)", 2], ["learn", "Learn how the GPU works", 3], ["peak", "The fastest matmul possible", 4]];
  const wy = { goal: "custom" };
  const ycv = canvas(880, 280, "Five ways to program a GPU, by how much code you write and how much of the hardware you control");
  function drawWays() {
    const pick = GOALS.find((g) => g[0] === wy.goal)[2];
    ycv.paint(() => {
      ycv.clear(); const c = ycv.colors, x0 = 170, bw = 300, row = 42;
      ycv.text("code you write", x0, 18, { size: 11, color: c["ink-3"] });
      ycv.text("control over the hardware", x0 + bw + 40, 18, { size: 11, color: c["ink-3"] });
      WAYS.forEach(([name, effort, ctrl, what], k) => {
        const y = 30 + k * row, on = k === pick;
        if (on) ycv.box(8, y - 4, 864, row - 4, { fill: `color-mix(in srgb, ${c.accent} 12%, ${c.surface})`, stroke: c.accent, lw: 1.5, r: 10 });
        ycv.text(name, x0 - 14, y + 20, { size: 13, align: "right", color: on ? c.ink : c["ink-2"] });
        ycv.box(x0, y + 8, bw, 14, { fill: c.sunken, stroke: "transparent", r: 7 });
        ycv.box(x0, y + 8, (bw * effort) / 10, 14, { fill: on ? c.c2 : `color-mix(in srgb, ${c.c2} 45%, ${c.surface})`, stroke: "transparent", r: 7 });
        ycv.box(x0 + bw + 40, y + 8, bw, 14, { fill: c.sunken, stroke: "transparent", r: 7 });
        ycv.box(x0 + bw + 40, y + 8, (bw * ctrl) / 10, 14, { fill: on ? c.accent : `color-mix(in srgb, ${c.accent} 40%, ${c.surface})`, stroke: "transparent", r: 7 });
      });
      ycv.text(`${WAYS[pick][0]}: you write ${WAYS[pick][3]}`, 440, 252, { size: 13, align: "center" });
      ycv.text("Start at the top and move down only when profiling says the layer above is the bottleneck.", 440, 272, { size: 11.5, align: "center", color: c["ink-3"] });
    });
  }

  root.append(
    section(null,
      goal("<p>We have a machine with 132 SMs, each wanting 64 warps of 32 threads in flight. We need a way to <b>describe a million small, independent pieces of work</b> so the hardware can spread them over whatever SMs it has, without the programmer managing any of it. CUDA's answer: write the code for <b>one thread</b>, and launch it as a grid.</p>",
        ["Say what one piece of work is", "Map each piece to its data", "Run on any size of GPU, unchanged"]),
      stackMap(id),
      h("div.vx-row", sticker("chip", { title: "The device", caption: "the GPU, with its own memory: your data must be copied here first", px: 4 }), sticker("package", { title: "Its memory", caption: "HBM, 80 GB at 3.35 TB/s, but ~25–60 GB/s from the CPU over PCIe", px: 3 }), sticker("grid", { title: "Your launch", caption: "a grid of blocks, each a patch of threads: you describe it, the GPU schedules it", px: 4 })),
    ),
    section("Grid, blocks, threads",
      prose("<p>A kernel launch says: run this function in a <b>grid</b> of <b>blocks</b>, each with the same number of <b>threads</b>. Every thread gets its block number (<code>blockIdx</code>) and its number inside the block (<code>threadIdx</code>), and uses them to pick its piece of the data. Each block is placed whole on one SM; the hardware hands out blocks as SMs free up. Change the sizes.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Elements n", min: 1, max: 600, value: gm.n, onInput: (v) => { gm.n = v; drawG(); } }),
          segmented({ label: "Threads per block", options: [32, 64, 128, 256].map((v) => ({ value: v, label: String(v) })), value: gm.bd, onChange: (v) => { gm.bd = v; drawG(); } }),
          slider({ label: "SMs on this GPU", min: 1, max: 8, value: gm.sms, onInput: (v) => { gm.sms = v; drawG(); } }),
          slider({ label: "Blocks resident per SM", min: 1, max: 4, value: gm.perSM, onInput: (v) => { gm.perSM = v; drawG(); } })),
        h("div.stage-body", gst.el, h("div.wide-canvas", gc.el))),
      callout("tip", "The tail effect", "<p>Watch the last-wave number. If the final wave only fills a few SM slots, the rest of the GPU idles while it finishes. With 132 SMs this is real: a grid of 140 blocks that each take the same time runs in two waves, the second one 6% full. Kernel authors size grids as multiples of the SM count, or use “persistent” kernels that loop over work.</p>"),
      predict({
        chapter: id,
        question: "You add two arrays of n = 1,000,000 floats with 256 threads per block. How many blocks do you launch, and how many threads in the last block do useful work?",
        options: ["3,906 blocks; all 256", "3,907 blocks; 64", "1,000,000 blocks; 1", "4,000 blocks; 256"],
        answer: 1,
        explain: "<p>⌈1,000,000 ÷ 256⌉ = 3,907. The first 3,906 blocks cover 999,936 elements, leaving 64 for the last block; its other 192 threads fail the <code>if (i &lt; n)</code> check and exit. Those 3,907 blocks then run in waves across the SMs.</p>",
      }),
    ),
    walk("cuda"),
    section("Launch a grid, and watch it land",
      prose("<p>Pick how many elements and how many threads per block. The launch becomes blocks; the GPU hands blocks to SMs until each is full, then runs the rest in later waves. This toy GPU has 4 SMs with H100 limits each; hover any cube to see whose threads it holds.</p>"),
      launchLab()),
    section("Why blocks can't wait for each other",
      prose("<p>A GPU with 4 SMs, each with room for 2 blocks, runs a grid of blocks. Pick a rule and press Run. In the middle option every block, after its work, waits for all the others: a grid-wide barrier that plain CUDA doesn't allow.</p>"),
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Rule", options: [{ value: "free", label: "Independent blocks" }, { value: "wait", label: "Every block waits for all" }, { value: "coop", label: "Cooperative launch" }], value: dl.mode, onChange: (v) => { dl.mode = v; resetD(); } }),
          h("div.btn-row", button("Run", () => runD(), "primary"), button("Reset", () => resetD()))),
        h("div.stage-body", dst.el, h("div.wide-canvas", dk.el))),
      whyChain({
        claim: "Threads in a block can share memory and synchronise; separate blocks can't (inside one kernel).",
        steps: [
          { q: "Why can threads in a block cooperate?", a: "<p>A block runs entirely on one SM. Its threads share that SM's shared memory, physical SRAM a few hundred micrometres away, and the SM has a hardware barrier, <code>__syncthreads()</code>, that holds every warp of the block until all arrive.</p>" },
          { q: "Why not the same across blocks?", a: "<p>Different blocks may be on different SMs. The only memory they share is L2 and HBM, hundreds of cycles away (<a href='/learn?c=memory-physics'>why memory is far</a>). Worse, they may not even be running at the same time.</p>" },
          { q: "Why might they not run at the same time?", a: "<p>A grid can have millions of blocks; an H100 holds at most a few thousand at once (132 SMs × up to 32 blocks). The rest wait in a queue for a free slot. If block 0 waited for block 900,000, and block 900,000 couldn't start until block 0 finished and freed its slot, both would wait forever.</p>" },
          { q: "Why design it so blocks queue?", a: "<p>So the same compiled program runs on any GPU: a laptop chip with 16 SMs runs the 3,907 blocks in more waves, an H100 in fewer. The program never says how many processors exist, so it scales from 2006's G80 to today without changes. Where cross-block sync is truly needed, you end the kernel and launch another, or use a cooperative launch that guarantees all blocks are resident.</p>" },
        ],
        axiom: { name: "Circular waiting never ends (deadlock)", html: "<p>If A can't finish until B starts, and B can't start until A finishes, neither ever happens. That's logic, not engineering. Any system with more tasks than slots must forbid tasks from waiting on tasks that may not have started. CUDA's rule, “blocks are independent”, is exactly that prohibition.</p>" },
      }),
      table(["Memory space", "Declared as", "Who can see it", "Lives in", "Speed"], [
        ["Registers", "ordinary local variables", "one thread", "the SM's register file", "~1 cycle"],
        ["Shared memory", "<code>__shared__ float tile[32][33];</code>", "the threads of one block", "SRAM in the SM", "~30 cycles"],
        ["Global memory", "<code>cudaMalloc</code>, pointers you pass in", "every thread, every kernel, and the CPU via copies", "HBM (cached in L2 and L1)", "~600 cycles"],
        ["Local memory", "arrays the compiler can't keep in registers", "one thread", "HBM, cached", "slow: a sign to restructure"],
        ["Constant memory", "<code>__constant__</code>", "every thread, read-only", "HBM with a small broadcast cache", "fast when a warp reads one address"],
      ].map((r) => r.map((x) => (typeof x === "string" && x.includes("<code>") ? h("span", { html: x }) : x)))),
    ),
    section("The life of a CUDA program",
      prose("<p>Adding two arrays of a million floats, drawn to scale in time. Step through and watch which of the CPU, the PCIe link and the GPU is busy.</p>"),
      h("div.wide-canvas", lf.el),
      stepper([
        { title: "1. Compile: two programs in one file", html: "<p><code>nvcc</code> splits your <code>.cu</code> file: ordinary C++ for the CPU (the <b>host</b>), and kernels marked <code>__global__</code> for the GPU (the <b>device</b>). Kernels compile to <b>PTX</b>, a virtual instruction set, and then to <b>SASS</b>, the machine code of a specific GPU generation. Triton and <code>torch.compile</code> generate the same PTX by other routes.</p>" },
        { title: "2. Allocate and copy in", html: "<p>The GPU has its own memory. <code>cudaMalloc</code> reserves space in HBM, and <code>cudaMemcpy</code> copies your arrays across <b>PCIe</b>, at roughly 25 GB/s (PCIe 4.0 ×16) to 50 GB/s (5.0): about 100× slower than HBM itself. This is why PyTorch keeps tensors on the GPU between operations.</p>" },
        { title: "3. Launch: <<<blocks, threads>>>", html: "<p><code>add&lt;&lt;&lt;3907, 256&gt;&gt;&gt;(a, b, c, n)</code> queues the kernel and returns immediately: launches are <b>asynchronous</b>. Each launch costs a few microseconds of CPU and driver work, which is why thousands of tiny kernels are slow and why CUDA Graphs and kernel fusion exist.</p>" },
        { title: "4. The GPU schedules it", html: "<p>A hardware scheduler gives blocks to SMs with room for them. Each SM splits its blocks into warps of 32 and its warp schedulers issue an instruction from a ready warp every cycle, hiding memory latency behind other warps (<a href='/learn?c=gpu-design'>design a GPU</a>). You never see this, but every performance rule comes from it.</p>" },
        { title: "5. Copy out, and synchronise", html: "<p><code>cudaMemcpy</code> back to the host waits for the kernel to finish first. Errors inside a kernel (like an out-of-bounds write) only surface at the next synchronising call, which is why CUDA code checks <code>cudaGetLastError()</code> and why debugging uses <code>CUDA_LAUNCH_BLOCKING=1</code>.</p>" },
      ], { onStep: (k) => { lstep = k; lf.redraw?.(); } }),
    ),
    section("Is the GPU worth the trip?",
      prose("<p>A GPU only wins if the work outweighs getting the data there and starting the kernel. Here, an elementwise operation reads two floats and writes one per element, with a chosen number of operations per element. Compare a good CPU with an H100 (practical rates, not peaks).</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Elements", min: 2, max: 9, step: 0.5, value: wt.n, fmt: sciS, onInput: (v) => { wt.n = v; drawW(); } }),
          slider({ label: "Operations per element", min: 1, max: 2000, value: wt.flops, onInput: (v) => { wt.flops = v; drawW(); } }),
          toggle({ label: "Data already on the GPU", checked: wt.resident, onChange: (v) => { wt.resident = v; drawW(); } })),
        h("div.stage-body", h("div.wide-canvas", wc.el), wst.el)),
      callout("note", "Three lessons in one widget", "<p><b>Small problems stay on the CPU</b>: below about 10⁴ elements, the 5 µs launch alone loses. <b>Copying kills cheap work</b>: with the data on the CPU, an elementwise add is limited by PCIe, and the CPU wins. <b>Keep data resident</b>: tick the box and the GPU wins by 30× even for simple operations, because HBM is 30× faster than CPU memory. This is why models are loaded onto the GPU once and stay there, and why AI frameworks fuse many small operations into one kernel.</p>"),
    ),
    section("Ways to program it",
      prose("<p>You rarely write raw CUDA. Each layer trades code you have to write for control over the machine. Pick what you're trying to do.</p>"),
      h("div.stage",
        h("div.stage-controls", segmented({ label: "Your goal", options: GOALS.map(([v, l]) => ({ value: v, label: l })), value: wy.goal, onChange: (v) => { wy.goal = v; drawWays(); } })),
        h("div.stage-body", h("div.wide-canvas", ycv.el))),
      table(["Layer", "You write", "Good for"], [
        ["PyTorch / JAX", "<code>y = x @ W</code>", "almost everything: calls NVIDIA's tuned libraries (cuBLAS, cuDNN) and generated kernels"],
        ["torch.compile", "the same Python, plus one decorator", "fusing chains of small operations into fewer kernels automatically"],
        ["Triton", "Python-like kernels over blocks of data", "custom fused kernels (attention, norms) without managing threads one by one"],
        ["CUDA C++", "one thread's code, plus shared memory and sync", "full control; learning how the machine works"],
        ["CUTLASS / CuTe, ThunderKittens", "templated tile-level C++", "matrix multiply and attention kernels at near peak speed on tensor cores"],
      ].map((r) => r.map((x) => (x.includes("<code>") ? h("span", { html: x }) : x)))),
      callout("tip", "Where to go from here", "<p>You now have the whole stack, from electrons to a kernel launch. To see it as one picture, zoom through all ten levels with real numbers in <a href='/learn?c=all-the-way-down'>one multiply, all the way down</a>. The next steps are in the systems chapters. Start with <a href='/learn?c=gpu-from-scratch'>GPU kernels from first principles</a> (coalescing, tiling, bank conflicts, occupancy), then <a href='/learn?c=kernels'>the roofline and FlashAttention</a> and <a href='/learn?c=precision'>number formats</a>. Last comes <a href='/learn?c=distributed'>spreading a model over many GPUs</a>, which is where AlexNet's two-GPU split grew up.</p>"),
    ),
    section("In code",
      code({
        "A complete CUDA program": `#include <cstdio>
#include <cuda_runtime.h>

__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    int n = 1 << 20; size_t bytes = n * sizeof(float);
    float *ha = (float*)malloc(bytes), *hb = (float*)malloc(bytes), *hc = (float*)malloc(bytes);
    for (int i = 0; i < n; ++i) { ha[i] = i; hb[i] = 2 * i; }

    float *a, *b, *c;
    cudaMalloc(&a, bytes); cudaMalloc(&b, bytes); cudaMalloc(&c, bytes);
    cudaMemcpy(a, ha, bytes, cudaMemcpyHostToDevice);        // across PCIe
    cudaMemcpy(b, hb, bytes, cudaMemcpyHostToDevice);

    int threads = 256, blocks = (n + threads - 1) / threads;  // round up
    add<<<blocks, threads>>>(a, b, c, n);                      // asynchronous
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("%s\\n", cudaGetErrorString(err)); return 1; }

    cudaMemcpy(hc, c, bytes, cudaMemcpyDeviceToHost);         // waits for the kernel
    printf("c[123] = %f (expect %f)\\n", hc[123], 3.0f * 123);
    cudaFree(a); cudaFree(b); cudaFree(c);
}
// nvcc add.cu -o add && ./add`,
        "From Python, on Colab": `import torch
from torch.utils.cpp_extension import load_inline

src = r"""
__global__ void add_k(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
torch::Tensor add(torch::Tensor a, torch::Tensor b) {
    auto c = torch::empty_like(a);
    int n = a.numel(), t = 256;
    add_k<<<(n + t - 1) / t, t>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n);
    return c;
}"""
ext = load_inline("add_ext", cpp_sources="torch::Tensor add(torch::Tensor a, torch::Tensor b);",
                  cuda_sources=src, functions=["add"])
a = torch.randn(10**7, device="cuda"); b = torch.randn_like(a)
assert torch.allclose(ext.add(a, b), a + b)`,
        "The same in Triton": `import triton, triton.language as tl

@triton.jit
def add_kernel(a, b, c, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)                       # like blockIdx.x
    offs = pid * BLOCK + tl.arange(0, BLOCK)      # a whole block of indices at once
    mask = offs < n                               # the guard
    tl.store(c + offs, tl.load(a + offs, mask=mask) + tl.load(b + offs, mask=mask), mask=mask)

grid = (triton.cdiv(n, 1024),)
add_kernel[grid](a, b, c, n, BLOCK=1024)`,
      }, "Project 60 compiles and times this kernel on a free Colab T4, measures PCIe against on-GPU bandwidth, and checks Little's law."),
    ),
  );
  drawG(); drawW(); drawWays();
}
