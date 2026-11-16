import { h } from "../lib/core.js";
import { slider, stats, predict, code, callout, section, prose, table, segmented, stepper } from "../lib/widgets.js";
import { flowCanvas } from "../lib/flow.js";
import { goal, whyChain, stackMap, sticker } from "../lib/why.js";
import { walk } from "../lib/walks.js";

function canvas(width, height, label) {
  const fc = flowCanvas({ width, height, label });
  fc.paint = (fn) => { fc.redraw = fn; fn(); };
  return fc;
}

export default function (root, { id }) {
  // ---------- 0. the design, decision by decision: one SM fills up ----------
  let bstep = 0;
  const bd = canvas(760, 290, "One GPU chip of 132 SMs, and one SM gaining a part with each design decision");
  const CAP = [
    "132 identical SMs, each a block of simple lanes: no branch predictor, no out-of-order engine",
    "one instruction is fetched once and drives 32 lanes: a warp",
    "64 warps wait in each SM; every cycle the scheduler issues from one that's ready",
    "a 256 KB register file: every resident thread keeps its own registers, so switching costs nothing",
    "256 KB of SRAM the program fills itself: load a tile once, reuse it many times",
    "a tensor core per quarter: a small matrix multiply in one instruction",
    "a 50 MB L2 shared by every SM, and HBM stacks right beside the die",
  ];
  bd.paint(() => {
    bd.clear(); const c = bd.colors, s = bstep, ac = c.accent;
    const hl = (k) => (k === s ? { stroke: ac, lw: 2.5 } : {});
    // chip panel
    bd.box(10, 10, 236, 244, { fill: c.surface, stroke: c.line, r: 10 }); bd.text("the chip", 20, 24, { size: 10.5, color: c["ink-3"], mono: true });
    const gx = 50, gy = 42, cs = 13.2;
    for (let r = 0; r < 11; r++) for (let q = 0; q < 12; q++) {
      const x = gx + q * cs, y = gy + r * cs + (s >= 6 && r >= 6 ? 16 : 0);
      bd.box(x, y, cs - 2.5, cs - 2.5, { fill: r === 2 && q === 9 ? ac : `color-mix(in srgb, ${c.c3} 45%, ${c.surface})`, stroke: "transparent", r: 2 });
    }
    if (s >= 6) {
      bd.box(gx, gy + 6 * cs - 1, 12 * cs - 2.5, 14, { fill: `color-mix(in srgb, ${c.c6} 35%, ${c.surface})`, ...hl(6), r: 3 });
      bd.text("L2  50 MB", gx + 6 * cs, gy + 6 * cs + 6, { size: 9.5, align: "center" });
      for (const xx of [18, 214]) for (let k = 0; k < 3; k++) bd.box(xx, 44 + k * 58, 22, 50, { fill: `color-mix(in srgb, ${c.c4} ${k === 2 && xx > 100 ? 15 : 45}%, ${c.surface})`, ...hl(6), r: 3 });
      bd.text("HBM", 29, 228, { size: 9, color: c["ink-2"], align: "center" }); bd.text("HBM", 225, 228, { size: 9, color: c["ink-2"], align: "center" });
    }
    bd.text("132 SMs", 128, 240, { size: 11, align: "center", color: s === 0 ? ac : c.ink });
    bd.line(gx + 9 * cs + 11, gy + 2 * cs + 5, 268, 40, { color: ac, lw: 1.2, dash: [3, 3] });
    // SM panel
    const sx = 268, sy = 10, sw = 482, sh = 244;
    bd.box(sx, sy, sw, sh, { fill: c.surface, stroke: c.line, r: 10 }); bd.text("one SM", sx + 10, sy + 14, { size: 10.5, color: c["ink-3"], mono: true });
    if (s >= 2) { bd.box(sx + sw - 128, sy + 9, 9, 9, { fill: c.good, stroke: "transparent", r: 2 }); bd.text("warp ready to issue", sx + sw - 114, sy + 14, { size: 9.5, color: c["ink-3"] }); }
    const qw = 230, qh = 84;
    for (let qi = 0; qi < 4; qi++) {
      const x = sx + 8 + (qi % 2) * (qw + 6), y = sy + 26 + Math.floor(qi / 2) * (qh + 6);
      bd.box(x, y, qw, qh, { fill: c.sunken, stroke: c.line, r: 7 });
      // lanes: 2 rows of 16
      const lx = x + 58, ly = y + 8;
      for (let k = 0; k < 32; k++) bd.box(lx + (k % 16) * 9, ly + Math.floor(k / 16) * 9, 7, 7, { fill: `color-mix(in srgb, ${c.c3} ${s === 0 ? 75 : 50}%, ${c.surface})`, stroke: "transparent", r: 1.5 });
      if (qi === 0) bd.text("32 lanes", lx + 72, ly + 25, { size: 9, color: c["ink-3"], align: "center" });
      if (s >= 1) {   // one instruction fans out to the lanes
        bd.box(x + 6, y + 6, 46, 18, { fill: `color-mix(in srgb, ${c.c2} 30%, ${c.surface})`, ...hl(1), r: 4 });
        bd.text("instr", x + 29, y + 15, { size: 9, align: "center", mono: true });
        bd.line(x + 52, y + 15, lx - 2, y + 15, { color: s === 1 ? ac : c["ink-3"], lw: s === 1 ? 2 : 1 });
      }
      if (s >= 2) {   // 16 warp slots per quarter
        for (let k = 0; k < 16; k++) { const rdy = (k * 7 + qi * 3) % 5 === 0; bd.box(x + 6 + (k % 4) * 11.5, y + 30 + Math.floor(k / 4) * 12, 9.5, 9, { fill: rdy ? c.good : `color-mix(in srgb, ${c["ink-3"]} 35%, ${c.surface})`, stroke: "transparent", r: 2 }); }
        if (s === 2) bd.box(x + 3, y + 27, 50, 52, { fill: "transparent", stroke: ac, lw: 2, r: 4 });
      }
      if (s >= 3) { bd.box(lx, y + 30, 142, 22, { fill: `color-mix(in srgb, ${c.c1} 30%, ${c.surface})`, ...hl(3), r: 4 }); bd.text("registers 64 KB", lx + 71, y + 41, { size: 9.5, align: "center" }); }
      if (s >= 5) { bd.box(x + 204, y + 8, 22, 70, { fill: `color-mix(in srgb, ${c.c5} 40%, ${c.surface})`, ...hl(5), r: 4 }); bd.text("TC", x + 215, y + 43, { size: 9.5, align: "center" }); }
    }
    if (s >= 4) { bd.box(sx + 8, sy + 208, sw - 16, 28, { fill: `color-mix(in srgb, ${c.c4} 30%, ${c.surface})`, ...hl(4), r: 6 }); bd.text("shared memory / L1: 256 KB", sx + sw / 2, sy + 222, { size: 11, align: "center" }); }
    bd.text(`${s + 1}. ${CAP[s]}`, 380, 274, { size: 11.5, align: "center", color: c.ink });
  });

  // ---------- 0b. the H100 floor plan ----------
  const PARTS = [["sm", "SMs"], ["tc", "Tensor cores"], ["rf", "Registers"], ["smem", "Shared memory"], ["l2", "L2"], ["hbm", "HBM"]];
  const NOTE = {
    sm: "144 SMs on the die, 132 switched on: a defect in any one just turns it off, so more chips pass (yield)",
    tc: "4 tensor cores per SM × 132 = 528: they do nearly all of the 989 TFLOP/s",
    rf: "64 KB per quarter, 256 KB per SM, about 34 MB across the chip: two-thirds the size of the L2",
    smem: "256 KB per SM of SRAM, up to 228 KB usable as shared memory, about 34 MB across the chip",
    l2: "50 MB in two halves across the middle, shared by every SM",
    hbm: "5 active stacks of DRAM beside the die: 80 GB, 5,120 wires wide, 3.35 TB/s",
  };
  const hp = { part: "sm" };
  const hc = canvas(880, 400, "Floor plan of an H100: 8 GPCs of SMs, the L2 across the middle, HBM stacks on both sides, and one SM in detail");
  const OFF = new Set([3, 16, 22, 37, 49, 58, 71, 85, 96, 104, 121, 139]);   // 12 of 144: an illustrative pattern, real chips differ
  hc.paint(() => {
    hc.clear(); const c = hc.colors, P = hp.part, ac = c.accent, on = (k) => P === k;
    const dim = (col, k) => (P === k ? `color-mix(in srgb, ${col} 70%, ${c.surface})` : `color-mix(in srgb, ${col} 45%, ${c.surface})`);
    // package
    hc.box(10, 10, 540, 352, { fill: c.sunken, stroke: c.line, r: 12 });
    for (const xx of [20, 482]) for (let k = 0; k < 3; k++) {
      const spare = xx > 100 && k === 2;
      hc.box(xx, 30 + k * 108, 58, 96, { fill: spare ? c.surface : dim(c.c4, "hbm"), stroke: on("hbm") && !spare ? ac : c.line, lw: on("hbm") && !spare ? 2.5 : 1, r: 5 });
      hc.text(spare ? "unused" : "HBM3", xx + 29, 78 + k * 108, { size: 10, color: spare ? c["ink-3"] : c.ink, align: "center" });
      if (!spare) hc.text("16 GB", xx + 29, 92 + k * 108, { size: 9, color: c["ink-2"], align: "center", mono: true });
    }
    // die
    const dx = 90, dy = 26, dw = 380, dh = 320;
    hc.box(dx, dy, dw, dh, { fill: c.surface, stroke: c["ink-3"], lw: 1.5, r: 6 });
    hc.text("NVLink · PCIe", dx + dw / 2, dy + 9, { size: 9, color: c["ink-3"], align: "center", mono: true });
    hc.box(dx + 8, dy + 151, dw - 16, 22, { fill: dim(c.c6, "l2"), stroke: on("l2") ? ac : "transparent", lw: 2.5, r: 4 });
    hc.text("L2 cache  25 MB  |  25 MB", dx + dw / 2, dy + 162, { size: 10, align: "center" });
    let n = 0;
    for (let g = 0; g < 8; g++) {
      const gx0 = dx + 8 + (g % 4) * 92, gy0 = dy + 20 + Math.floor(g / 4) * 160;
      hc.box(gx0, gy0, 88, 126, { fill: "transparent", stroke: c.line, r: 5 });
      if (g === 0) hc.text("GPC", gx0 + 44, gy0 + 132, { size: 8.5, color: c["ink-3"], align: "center", mono: true });
      for (let k = 0; k < 18; k++, n++) {
        const x = gx0 + 5 + (k % 3) * 27.5, y = gy0 + 5 + Math.floor(k / 3) * 20, off = OFF.has(n);
        const f = off ? c.sunken : P === "sm" ? `color-mix(in srgb, ${c.c3} 65%, ${c.surface})` : `color-mix(in srgb, ${c.c3} 25%, ${c.surface})`;
        hc.box(x, y, 24, 17, { fill: f, stroke: n === 56 ? ac : "transparent", lw: 2, r: 2.5 });
        if (off) hc.text("×", x + 12, y + 8.5, { size: 10, color: c["ink-3"], align: "center" });
        else if (P === "tc") for (let t = 0; t < 4; t++) hc.box(x + 3 + (t % 2) * 10, y + 2.5 + Math.floor(t / 2) * 6.5, 8, 5, { fill: c.c5, stroke: "transparent", r: 1 });
        else if (P === "rf" || P === "smem") hc.box(x + 3, P === "rf" ? y + 3 : y + 10, 18, P === "rf" ? 6 : 5, { fill: P === "rf" ? c.c1 : c.c4, stroke: "transparent", r: 1 });
      }
    }
    hc.line(446, 51, 446, 18, { color: ac, lw: 1.2, dash: [3, 3] }); hc.line(446, 18, 566, 18, { color: ac, lw: 1.2, dash: [3, 3] });
    // one SM in detail
    const sx = 566, sy = 10, sw = 304, sh = 352;
    hc.box(sx, sy, sw, sh, { fill: c.surface, stroke: c.line, r: 10 }); hc.text("one SM", sx + 10, sy + 14, { size: 10.5, color: c["ink-3"], mono: true });
    for (let q = 0; q < 4; q++) {
      const x = sx + 8, y = sy + 26 + q * 66, w = sw - 16;
      hc.box(x, y, w, 60, { fill: c.sunken, stroke: c.line, r: 6 });
      hc.box(x + 6, y + 6, 70, 14, { fill: `color-mix(in srgb, ${c.c2} 25%, ${c.surface})`, stroke: "transparent", r: 3 }); hc.text("warp sched", x + 41, y + 13, { size: 8.5, align: "center" });
      hc.box(x + 6, y + 24, 70, 30, { fill: dim(c.c1, "rf"), stroke: on("rf") ? ac : "transparent", lw: 2, r: 3 }); hc.text("regs 64 KB", x + 41, y + 39, { size: 8.5, align: "center" });
      for (let k = 0; k < 32; k++) hc.box(x + 84 + (k % 16) * 9.5, y + 8 + Math.floor(k / 16) * 10, 7.5, 7.5, { fill: `color-mix(in srgb, ${c.c3} ${on("sm") ? 70 : 40}%, ${c.surface})`, stroke: "transparent", r: 1.5 });
      hc.text("32 FP32 lanes", x + 84 + 76, y + 37, { size: 8.5, color: c["ink-3"], align: "center" });
      hc.box(x + 242, y + 6, 40, 48, { fill: dim(c.c5, "tc"), stroke: on("tc") ? ac : "transparent", lw: 2, r: 4 }); hc.text("tensor", x + 262, y + 25, { size: 8.5, align: "center" }); hc.text("core", x + 262, y + 36, { size: 8.5, align: "center" });
    }
    hc.box(sx + 8, sy + 292, sw - 16, 52, { fill: dim(c.c4, "smem"), stroke: on("smem") ? ac : "transparent", lw: 2.5, r: 6 });
    hc.text("shared memory / L1", sx + sw / 2, sy + 312, { size: 10.5, align: "center" }); hc.text("256 KB SRAM", sx + sw / 2, sy + 328, { size: 9.5, color: c["ink-2"], align: "center", mono: true });
    hc.text(NOTE[P], 440, 384, { size: 11.5, align: "center" });
  });

  // ---------- 1. SIMT and divergence ----------
  const dv = { pattern: "half", frac: 0.5, len: 6 };
  const dc = canvas(880, 250, "One warp of 32 lanes running an if/else, cycle by cycle");
  const dst = stats([{ label: "Cycles for the if/else" }, { label: "Lanes doing useful work" }, { label: "Instruction fetches shared by" }]);
  function lanesTaking() {
    if (dv.pattern === "all") return Array(32).fill(true);
    if (dv.pattern === "alt") return Array.from({ length: 32 }, (_, i) => i % 2 === 0);
    if (dv.pattern === "half") return Array.from({ length: 32 }, (_, i) => i < 16);
    let s = 3; return Array.from({ length: 32 }, () => ((s = (s * 1103515245 + 12345) % 2147483648) / 2147483648) < dv.frac);
  }
  function drawD() {
    const take = lanesTaking(), nA = take.filter(Boolean).length, nB = 32 - nA, L = dv.len;
    const passes = (nA ? 1 : 0) + (nB ? 1 : 0), cycles = passes * L, util = (nA * L + nB * L) / (32 * cycles);
    dst.set(0, `${cycles} (${passes === 2 ? "both paths, one after the other" : "one path"})`, passes === 1 ? "good" : "warn");
    dst.set(1, `${Math.round(util * 100)}%`, util > 0.9 ? "good" : util > 0.6 ? "warn" : "bad");
    dst.set(2, "32 lanes: 1 fetch + decode per 32 operations", "good");
    dc.paint(() => {
      dc.clear(); const c = dc.colors, lw = 24, x0 = 70, y0 = 30, ch = 14;
      for (let i = 0; i < 32; i++) dc.text(String(i), x0 + i * lw + lw / 2, 16, { size: 9.5, color: c["ink-3"], align: "center", mono: true });
      let row = 0;
      const paint = (mask, label, col) => {
        for (let k = 0; k < L; k++, row++) {
          dc.text(k === 0 ? label : "", 8, y0 + row * ch + ch / 2, { size: 10.5, color: c["ink-2"] });
          for (let i = 0; i < 32; i++) dc.box(x0 + i * lw + 1, y0 + row * ch + 1, lw - 2, ch - 2, { fill: mask[i] ? col : c.sunken, stroke: "transparent", r: 2 });
        }
      };
      if (nA) paint(take, "if", c.accent);
      if (nB) paint(take.map((t) => !t), "else", c.c2);
      dc.text(`time ↓ · ${cycles} cycles · coloured = lane working, grey = lane masked off and idle`, x0, y0 + row * ch + 18, { size: 11, color: c["ink-3"] });
    });
  }

  // ---------- 2. latency hiding ----------
  const lh = { warps: 4, C: 20, L: 400 };
  const lc = canvas(880, 300, "Warps taking turns on one scheduler while others wait for memory");
  const lst = stats([{ label: "Scheduler busy" }, { label: "Warps needed to stay busy" }, { label: "Data in flight to feed an H100 (Little's law)" }]);
  function simulate(nw, C, L, T) {
    const ready = Array(nw).fill(0), left = Array(nw).fill(C), rows = Array.from({ length: nw }, () => new Uint8Array(T));
    let busy = 0;
    for (let t = 0; t < T; t++) {
      let pick = -1;
      for (let w = 0; w < nw; w++) if (ready[w] <= t) { pick = w; break; }
      for (let w = 0; w < nw; w++) rows[w][t] = ready[w] > t ? 2 : 0;          // 2 = waiting on memory, 0 = ready but not picked
      if (pick >= 0) {
        rows[pick][t] = 1; busy++;
        if (--left[pick] === 0) { left[pick] = C; ready[pick] = t + 1 + L; }
      }
    }
    return { rows, util: busy / T };
  }
  function drawL() {
    const T = 900, sim = simulate(lh.warps, lh.C, lh.L, T), need = Math.ceil((lh.C + lh.L) / lh.C);
    const ns = lh.L / 1.755, inflight = 3.35e12 * ns * 1e-9;
    lst.set(0, `${Math.round(sim.util * 100)}%`, sim.util > 0.9 ? "good" : sim.util > 0.5 ? "warn" : "bad");
    lst.set(1, `${need} (= (compute + wait) ÷ compute)`);
    lst.set(2, `${(inflight / 1e6).toFixed(2)} MB ≈ ${Math.round(inflight / 132 / 1024)} KB per SM`);
    lc.paint(() => {
      lc.clear(); const c = lc.colors, x0 = 70, show = Math.min(lh.warps, 16), rh = Math.min(36, 250 / show), cw = (880 - x0 - 10) / T;
      for (let w = 0; w < show; w++) {
        lc.text(`warp ${w}`, 8, 14 + w * rh + rh / 2, { size: 10, color: c["ink-3"], mono: true });
        let t = 0;
        while (t < T) {
          const v = sim.rows[w][t]; let e = t; while (e + 1 < T && sim.rows[w][e + 1] === v) e++;
          if (v) lc.box(x0 + t * cw, 14 + w * rh + 1, (e - t + 1) * cw, rh - 2, { fill: v === 1 ? c.accent : `color-mix(in srgb, ${c.c4} 25%, ${c.surface})`, stroke: "transparent", r: 2 });
          t = e + 1;
        }
      }
      if (lh.warps > show) lc.text(`… and ${lh.warps - show} more warps`, x0, 14 + show * rh + 8, { size: 10.5, color: c["ink-3"] });
      lc.text("purple = this warp is computing · yellow = waiting for memory · time →", x0, 290, { size: 11, color: c["ink-3"] });
    });
  }

  // ---------- 3. tensor cores ----------
  const tc = { n: 16 };
  const tst = stats([{ label: "Multiply-adds per instruction" }, { label: "Operands read per multiply-add" }, { label: "Instructions for a 4096³ matmul" }, { label: "Energy per MAC on operand delivery (vs n = 1)" }]);
  function drawT() {
    const n = tc.n, macs = n ** 3, reads = 3 * n * n;
    tst.set(0, macs.toLocaleString()); tst.set(1, (reads / macs).toFixed(n > 4 ? 3 : 2), n >= 8 ? "good" : "warn");
    const q = 4096 ** 3 / macs, e = Math.floor(Math.log10(q)); tst.set(2, `${(q / 10 ** e).toFixed(1)} × 10${String(e).split("").map((d) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[+d]).join("")}`);
    tst.set(3, `${((reads / macs) / 3 * 100).toFixed(n > 8 ? 1 : 0)}%`, n >= 8 ? "good" : "warn");
    tcv.paint(() => {
      tcv.clear(); const c = tcv.colors;
      tcv.box(10, 8, 500, 264, { fill: c.surface, stroke: c.line, r: 10 });
      tcv.text(`One instruction: C (${n}×${n}) += A (${n}×${n}) × B (${n}×${n})`, 24, 26, { size: 12 });
      const sz = Math.min(30, 120 / n), side = sz * n, y0 = 70 + (150 - side) / 2, r = Math.min(n - 1, Math.floor(n / 3)), k = Math.min(n - 1, Math.floor((2 * n) / 3));
      const grid = (x, label, hl) => {
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
          const on = hl(i, j);
          tcv.ctx.fillStyle = on ? on : `color-mix(in srgb, ${c.c1} 18%, transparent)`;
          tcv.ctx.fillRect(x + j * sz + 0.5, y0 + i * sz + 0.5, Math.max(0.6, sz - 1), Math.max(0.6, sz - 1));
        }
        tcv.box(x - 1, y0 - 1, side + 2, side + 2, { fill: "transparent", stroke: c.line, lw: 1, r: 2 });
        tcv.text(label, x + side / 2, y0 + side + 16, { size: 11, align: "center" });
      };
      const gx = (i) => 40 + i * 160;
      grid(gx(0), "A", (i) => (i === r ? c.c3 : null));
      tcv.text("×", gx(0) + side + (160 - side) / 2, y0 + side / 2, { size: 18, align: "center", color: c["ink-3"] });
      grid(gx(1), "B", (i, j) => (j === k ? c.c4 : null));
      tcv.text("→", gx(1) + side + (160 - side) / 2, y0 + side / 2, { size: 18, align: "center", color: c["ink-3"] });
      grid(gx(2), "C", (i, j) => (i === r && j === k ? c.accent : i === r || j === k ? `color-mix(in srgb, ${c.accent} 30%, transparent)` : null));
      tcv.text(`one C value = a row of A · a column of B (${n} multiply-adds)`, 24, 238, { size: 10.5, color: c["ink-2"] });
      tcv.text(`the same row of A is reused for all ${n} values in that row of C`, 24, 256, { size: 10.5, color: c["ink-2"] });
      // right: reads vs work, log scale
      tcv.box(522, 8, 268, 264, { fill: c.surface, stroke: c.line, r: 10 });
      tcv.text("Numbers read vs work done", 536, 26, { size: 12 });
      const L = (v) => 170 * (Math.log10(Math.max(1, v)) / 5);
      [["read (A, B, C)", reads, c.c6], ["multiply-adds", macs, c.good]].forEach(([nm, v, col], i) => {
        const y = 62 + i * 46;
        tcv.text(nm, 536, y, { size: 10.5, color: c["ink-2"] });
        tcv.box(536, y + 10, Math.max(4, L(v)), 14, { fill: `color-mix(in srgb, ${col} 70%, transparent)`, stroke: "transparent", r: 3 });
        tcv.text(v.toLocaleString(), 542 + Math.max(4, L(v)), y + 17, { size: 10.5, mono: true });
      });
      tcv.text("bar length on a log scale", 536, 160, { size: 9.5, color: c["ink-3"] });
      tcv.text("work per number fetched (n / 3)", 536, 196, { size: 11, color: c["ink-2"] });
      tcv.text(`${(macs / reads).toFixed(macs / reads < 10 ? 1 : 0)} multiply-adds`, 536, 222, { size: 16, color: n >= 8 ? c.good : c.warn });
    });
  }
  const tcv = canvas(800, 280, "A tensor core instruction: tiles of A and B multiplied into C, with how many numbers are read against how many multiply-adds are done");

  root.append(
    section(null,
      goal("<p>Now design the machine. The job (chapter 1): trillions of <b>independent multiply-adds</b>. The constraints we derived: <b>heat</b> caps the clock, so go wide (chapter 6); <b>memory is far</b>, ~600 cycles and ~170× the energy of the math (chapter 5); and multipliers grow with the <b>square of their bits</b> (chapter 4). Each feature of a GPU is the cheapest answer to one of these. Build it one decision at a time.</p>",
        ["Thousands of MACs per clock within ~700 W", "Never let them wait on memory", "Spend as little energy as possible on anything that isn't arithmetic"]),
      stackMap(id),
      h("div.vx-row", sticker("package", { title: "What we're designing", caption: "a die of many identical SMs, with HBM stacks millimetres away", px: 3 }), sticker("sm", { title: "One SM", caption: "four tensor cores (purple) beside their register files, over shared memory (blue)", px: 3 }), sticker("tensor", { title: "A tensor core's job", caption: "a small matrix product is a cube of multiply-adds: A on one side, B on the other", px: 4 })),
    ),
    section("The design, decision by decision",
      prose("<p>Seven decisions, each forced by a constraint from an earlier chapter. Watch one SM fill up as you step: the newest part is outlined in purple.</p>"),
      h("div.wide-canvas", bd.el),
      stepper([
        { title: "1. Many simple cores, not a few clever ones", html: "<p><b>Constraint:</b> power (chapter 6). <b>Decision:</b> spend transistors on arithmetic, not on making one thread fast: no branch prediction, no out-of-order execution, modest clock (~1.8 GHz). Group the units into identical blocks called <b>streaming multiprocessors (SMs)</b> and copy them: an H100 has 132.</p>" },
        { title: "2. One instruction drives 32 lanes (a warp)", html: "<p><b>Constraint:</b> fetching and decoding an instruction costs tens of picojoules, far more than the add it describes. <b>Decision:</b> 32 threads share one instruction stream and run in lockstep: a <b>warp</b>. The control cost is split 32 ways. Price: when threads in a warp branch differently, both paths run one after the other (try it below).</p>" },
        { title: "3. Hide memory latency with other warps", html: "<p><b>Constraint:</b> a load from HBM takes ~600 cycles. A CPU hides this with big caches and prediction. <b>Decision:</b> keep many warps resident (up to 64 per SM) and, every cycle, issue from any warp that's ready. While one waits for memory, others compute. Latency isn't reduced; it's <b>overlapped</b>.</p>" },
        { title: "4. A giant register file so switching is free", html: "<p><b>Constraint:</b> switching warps every cycle must cost nothing; a CPU context switch saves and restores registers, taking microseconds. <b>Decision:</b> give every resident thread its own registers, all at once: 65,536 32-bit registers per SM (256 KB, bigger than many CPUs' L1 cache). Switching warps is just reading a different part of it.</p>" },
        { title: "5. A scratchpad the programmer controls", html: "<p><b>Constraint:</b> data must be reused many times once fetched (chapter 5: ~300 uses per byte from HBM to break even). <b>Decision:</b> up to 228 KB per SM of <b>shared memory</b>, fast SRAM that the threads of a block share and manage by hand. Tiling algorithms load a tile once and reuse it from here.</p>" },
        { title: "6. Matrix units: tensor cores", html: "<p><b>Constraint:</b> even from registers, feeding a multiplier its two operands costs more than the multiply. <b>Decision:</b> a unit that multiplies small matrices in one instruction. An n×n×n matrix multiply does n³ MACs from only 3n² numbers, so each operand is reused n times inside the unit. Most of an H100's 989 TFLOP/s come from these, at low precision (bf16, fp8) because multiplier size grows with bits².</p>" },
        { title: "7. Wide, close memory and a big L2", html: "<p><b>Constraint:</b> 132 SMs all reading at once. <b>Decision:</b> HBM stacks on an interposer (5,120 bits wide, 3.35 TB/s), a 50 MB L2 cache shared by all SMs, and a network on the chip connecting them. Then copy SMs until the die is full: 80 billion transistors, 814 mm², 700 W.</p>" },
      ], { onStep: (k) => { bstep = k; bd.redraw?.(); } }),
    ),
    section("Warps: one instruction, 32 lanes",
      prose("<p>A warp runs one instruction at a time for all 32 of its threads. If they disagree at an <code>if</code>, the hardware runs the <code>if</code> path with the disagreeing lanes switched off, then the <code>else</code> path with the others off. Nothing breaks, but lanes sit idle. Pick how the 32 lanes split.</p>"),
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Which lanes take the if", options: [{ value: "all", label: "All of them" }, { value: "half", label: "Lanes 0–15" }, { value: "alt", label: "Even lanes" }, { value: "rand", label: "Depends on the data" }], value: dv.pattern, onChange: (v) => { dv.pattern = v; drawD(); } }),
          slider({ label: "Share taking the if (data case)", min: 0, max: 1, step: 0.05, value: dv.frac, fmt: (v) => `${Math.round(v * 100)}%`, onInput: (v) => { dv.frac = v; if (dv.pattern === "rand") drawD(); } }),
          slider({ label: "Instructions on each path", min: 1, max: 12, value: dv.len, onInput: (v) => { dv.len = v; drawD(); } })),
        h("div.stage-body", dst.el, h("div.wide-canvas", dc.el))),
      callout("tip", "Why 32 and not 1,000", "<p>Wider warps split the control cost more ways but waste more when threads diverge, and need more wiring between lanes. 32 has been NVIDIA's warp size since its first CUDA GPU (G80, 2006); AMD uses 32 or 64. Matrix math never diverges, which is part of why AI runs so well: every lane always takes the same path.</p>"),
      predict({
        chapter: id,
        question: "In a warp, even-numbered threads take a 10-instruction if-branch and odd threads take a 10-instruction else-branch. How long does the warp take, and what share of lane-cycles do useful work?",
        options: ["10 instructions, 100%", "20 instructions, 50%", "10 instructions, 50%", "320 instructions, one lane at a time"],
        answer: 1,
        explain: "<p>The warp issues all 10 instructions of the if with odd lanes masked off, then all 10 of the else with even lanes masked off: 20 issue slots, each with half the lanes working. The fix is to arrange data so that whole warps take the same branch, e.g. sort or partition the work by branch first.</p>",
      }),
    ),
    section("Hiding latency: the trick that defines a GPU",
      prose("<p>Each warp computes for a while, then needs data from memory and waits hundreds of cycles. With one warp, the units sit idle most of the time. With enough warps, there's always one ready to go. One scheduler below issues one instruction per cycle from the first ready warp.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Resident warps", min: 1, max: 64, value: lh.warps, onInput: (v) => { lh.warps = v; drawL(); } }),
          slider({ label: "Compute cycles between loads", min: 2, max: 100, value: lh.C, onInput: (v) => { lh.C = v; drawL(); } }),
          slider({ label: "Memory latency (cycles)", min: 50, max: 900, step: 10, value: lh.L, onInput: (v) => { lh.L = v; drawL(); } })),
        h("div.stage-body", lst.el, h("div.wide-canvas", lc.el))),
      whyChain({
        claim: "A GPU needs tens of thousands of threads in flight to run at full speed.",
        steps: [
          { q: "Why so many?", a: "<p>To keep memory busy. To pull 3.35 TB/s from HBM when each request takes ~350 ns to come back, enough requests must be outstanding to cover that whole time: about a megabyte in flight, every moment.</p>" },
          { q: "Why must that much be in flight?", a: "<p>Think of a pipe: if each byte spends 350 ns in transit and you want 3.35 TB to come out per second, then at any instant 3.35 × 10¹² × 350 × 10⁻⁹ ≈ 1.2 million bytes are inside the pipe. Fewer outstanding requests, less bandwidth.</p>" },
          { q: "Why do threads provide the requests?", a: "<p>Each warp can only have so many loads outstanding before it needs their results. 1.2 MB across 132 SMs is ~9 KB per SM; at 512 bytes per warp-wide load, that's about 18 warps' worth per SM, before counting the warps needed to keep the arithmetic busy too. Hence 64 warp slots per SM, 2,048 threads, ~270,000 per GPU.</p>" },
          { q: "Why not make memory faster instead?", a: "<p>Latency is set by the physics and protocol of DRAM (chapter 5), and has barely improved in decades: it's ~100 ns on CPUs too. Bandwidth can be bought with more pins and stacks; latency can't. So designs tolerate latency instead of fighting it.</p>" },
        ],
        axiom: { name: "Little's law: items in flight = throughput × time in flight", html: "<p>Proved by John Little in 1961, and true for any stable system: a queue at a shop, cars on a road, bytes in a memory pipe. It isn't physics; it's arithmetic about averages, so no engineering can escape it. To get high throughput through a slow pipe, you must have a lot in the pipe at once. A GPU is a machine built around that fact.</p>" },
      }),
      predict({
        chapter: id,
        question: "To keep 3.35 TB/s flowing from memory with ~350 ns latency, how much data must be in flight at once?",
        options: ["About 3 KB", "About 1 MB", "About 3 GB", "None: bandwidth and latency are unrelated"],
        answer: 1,
        explain: "<p>3.35 × 10¹² B/s × 350 × 10⁻⁹ s ≈ 1.2 × 10⁶ B. That's why occupancy matters for memory-bound kernels, and why kernels issue wide (16-byte) loads and several independent loads per thread: each trick puts more bytes in flight per warp.</p>",
      }),
    ),
    walk("gpu-design"),
    section("Tensor cores: reuse inside the unit",
      prose("<p>A tensor core takes a small tile of A and of B and computes their product in one instruction. An n×n tile product does n³ multiply-adds but reads only 3n² numbers (A, B and the accumulator C), so the operand traffic per multiply-add falls as 3/n. Slide the tile size.</p>"),
      h("div.stage",
        h("div.stage-controls", slider({ label: "Tile size n (an n×n×n multiply per instruction)", min: 0, max: 5, value: 4, fmt: (v) => String(2 ** v), onInput: (v) => { tc.n = 2 ** v; drawT(); } })),
        h("div.stage-body", h("div.wide-canvas", tcv.el), tst.el)),
      callout("note", "The same idea at every scale", "<p>Reuse is the theme of the whole machine: tensor cores reuse operands inside the unit, shared memory reuses tiles inside an SM, L2 reuses data across SMs, and Google's TPU takes it furthest with a <b>systolic array</b>, a 128×128 grid of multiply-add cells where numbers flow from cell to cell and each one is used 128 times without returning to a register file. On an H100 each SM's four tensor cores do about 2,048 bf16 multiply-adds per clock.</p>"),
    ),
    section("Decode an H100",
      prose("<p>The whole chip at once. Pick a part to see where it sits on the die and inside one SM, and how many there are.</p>"),
      h("div.stage",
        h("div.stage-controls", segmented({ label: "Show", options: PARTS.map(([v, l]) => ({ value: v, label: l })), value: hp.part, onChange: (v) => { hp.part = v; hc.redraw(); } })),
        h("div.stage-body", h("div.wide-canvas", hc.el))),
      table(["Spec", "Value", "Which decision explains it"], [
        ["SMs", "132 (of 144 on the die; the rest are disabled to improve yield)", "1: copy a simple block many times"],
        ["Clock", "~1.6–1.98 GHz", "1: slow and wide beats fast and narrow under a power cap"],
        ["Threads per SM", "up to 2,048 (64 warps of 32)", "2 and 3: warps, and enough of them to hide latency"],
        ["Registers per SM", "65,536 × 32-bit = 256 KB", "4: every resident thread keeps its own"],
        ["Shared memory / L1 per SM", "256 KB (up to 228 KB as shared)", "5: a scratchpad for reuse"],
        ["Tensor cores", "528 (4 per SM); 989 TFLOP/s dense bf16, 1,979 fp8", "6: matrix units at low precision"],
        ["FP32 cores", "16,896 (128 per SM); 67 TFLOP/s", "the general-purpose lanes"],
        ["L2", "50 MB", "7: shared reuse across SMs"],
        ["HBM3", "80 GB, 3.35 TB/s, 5,120-bit bus", "7: wide and close"],
        ["Power", "700 W", "the budget everything above is fitted into"],
      ]),
      callout("tip", "The one ratio to remember", "<p>989 TFLOP/s ÷ 3.35 TB/s ≈ <b>295 operations per byte</b>. Any kernel that does fewer operations per byte it loads from HBM can't use the tensor cores fully, no matter how it's written. That's the roofline. Next: how you actually program this machine, in <a href='/learn?c=cuda'>CUDA</a>.</p>"),
    ),
    section("In code",
      code({
        "Divergence": `__global__ void diverge(float* x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x % 2 == 0) x[i] = sinf(x[i]);   // even lanes...
    else                      x[i] = cosf(x[i]);   // ...then odd lanes: 2 passes
}
__global__ void no_diverge(float* x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ((threadIdx.x / 32) % 2 == 0) x[i] = sinf(x[i]);   // whole warps agree
    else                             x[i] = cosf(x[i]);
}`,
        "Warp reduction": `__inline__ __device__ float warp_sum(float v) {
    for (int off = 16; off > 0; off /= 2)
        v += __shfl_down_sync(0xffffffff, v, off);   // 5 steps, registers only
    return v;                                         // lane 0 holds the total
}`,
        "A tensor core instruction (WMMA)": `#include <mma.h>
using namespace nvcuda;
__global__ void tile16(const half* A, const half* B, float* C) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::fill_fragment(c, 0.0f);
    wmma::load_matrix_sync(a, A, 16);
    wmma::load_matrix_sync(b, B, 16);
    wmma::mma_sync(c, a, b, c);                 // 4,096 multiply-adds, one warp, one call
    wmma::store_matrix_sync(C, c, 16, wmma::mem_row_major);
}`,
      }, "These run on any NVIDIA GPU from Volta (2017) on, including the free Colab T4."),
    ),
  );
  drawD(); drawL(); drawT();
}
