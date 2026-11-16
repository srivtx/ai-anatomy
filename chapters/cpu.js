import { h, loop, reducedMotion } from "../lib/core.js";
import { slider, stats, predict, code, callout, section, prose, table, segmented, stepper, button, toggle } from "../lib/widgets.js";
import { flowCanvas } from "../lib/flow.js";
import { goal, whyChain, stackMap, sticker } from "../lib/why.js";
import { walk } from "../lib/walks.js";

function canvas(width, height, label) {
  const fc = flowCanvas({ width, height, label });
  fc.paint = (fn) => { fc.redraw = fn; fn(); };
  return fc;
}

// ---------- a tiny CPU: sum four numbers from memory ----------
const PROG = [
  { op: "LOAD", txt: "LOAD r1, [r0]", note: "r1 = memory[r0]" },
  { op: "ADD", txt: "ADD  r2, r2, r1", note: "sum = sum + r1" },
  { op: "ADDI", txt: "ADDI r0, r0, 1", note: "point at the next number" },
  { op: "SUBI", txt: "SUBI r3, r3, 1", note: "one fewer to go" },
  { op: "BNZ", txt: "BNZ  r3, 0", note: "if r3 ≠ 0, jump back to 0" },
  { op: "HALT", txt: "HALT", note: "stop" },
];
const DMEM = [5, 3, 8, 2];
const PHASES = ["fetch", "decode", "execute", "write back"];

// the dynamic trace of that loop: what actually runs, in order
const TRACE = [];
for (let it = 0; it < 4; it++) {
  TRACE.push({ txt: "LOAD r1, [r0]", dst: "r1", src: ["r0"], load: true });
  TRACE.push({ txt: "ADD  r2, r2, r1", dst: "r2", src: ["r2", "r1"] });
  TRACE.push({ txt: "ADDI r0, r0, 1", dst: "r0", src: ["r0"] });
  TRACE.push({ txt: "SUBI r3, r3, 1", dst: "r3", src: ["r3"] });
  TRACE.push({ txt: "BNZ  r3, 0", dst: null, src: ["r3"], branch: true, taken: it < 3 });
}
TRACE.push({ txt: "HALT", dst: null, src: [] });

/** When each instruction enters each stage, for three kinds of machine. */
function schedule(mode) {
  const S = [];
  TRACE.forEach((ins, i) => {
    const p = S[i - 1];
    let IF, ID;
    if (mode === "seq") { IF = p ? p.WB + 1 : 0; ID = IF + 1; }
    else {
      IF = p ? Math.max(p.IF + 1, p.ID) : 0;
      if (p && TRACE[i - 1].branch) {
        const miss = mode === "stall" || !TRACE[i - 1].taken;   // "fwd" predicts the backward branch taken
        if (miss) IF = Math.max(IF, p.EX + 1);
      }
      ID = Math.max(IF + 1, p ? p.ID + 1 : 0);
      for (const r of ins.src) {
        let k = i - 1; while (k >= 0 && TRACE[k].dst !== r) k--;
        if (k < 0) continue;
        const q = S[k];
        if (mode === "stall") ID = Math.max(ID, q.WB);
        else ID = Math.max(ID, TRACE[k].load ? q.MEM : q.EX);
      }
    }
    S.push({ IF, ID, EX: ID + 1, MEM: ID + 2, WB: ID + 3 });
  });
  return S;
}

// ---------- out-of-order: a block of ten instructions ----------
const OOO = [
  { t: "LOAD a ← [x]", dep: [], lat: 4, load: true },
  { t: "ADD  b ← a + 1", dep: [0], lat: 1 },
  { t: "MUL  c ← b × b", dep: [1], lat: 3 },
  { t: "LOAD d ← [y]", dep: [], lat: 4 },
  { t: "ADD  e ← d + 2", dep: [3], lat: 1 },
  { t: "MUL  f ← e × 3", dep: [4], lat: 3 },
  { t: "ADD  g ← i + 1", dep: [], lat: 1 },
  { t: "ADD  h ← g + 1", dep: [6], lat: 1 },
  { t: "ADD  k ← c + f", dep: [2, 5], lat: 1 },
  { t: "STORE [z] ← k", dep: [8], lat: 1 },
];
function runOOO({ inOrder, width, miss }) {
  const lat = OOO.map((o, i) => (i === 0 && miss ? 200 : o.lat));
  const start = Array(OOO.length).fill(-1), fin = Array(OOO.length).fill(Infinity);
  let cyc = 0, done = 0, next = 0;
  const ready = (i) => OOO[i].dep.every((d) => fin[d] <= cyc);
  while (done < OOO.length && cyc < 2000) {
    let issued = 0;
    if (inOrder) { while (issued < width && next < OOO.length && ready(next)) { start[next] = cyc; fin[next] = cyc + lat[next]; next++; issued++; done++; } }
    else for (let i = 0; i < OOO.length && issued < width; i++) if (start[i] < 0 && ready(i)) { start[i] = cyc; fin[i] = cyc + lat[i]; issued++; done++; }
    cyc++;
  }
  return { start, fin, lat, total: Math.max(...fin) };
}

// ---------- branch patterns ----------
let sd = 3; const rnd = () => ((sd = (sd * 16807) % 2147483647) / 2147483647);
const PAT = {
  loop: Array.from({ length: 48 }, (_, i) => (i % 8 === 7 ? 0 : 1)),
  random: Array.from({ length: 48 }, () => (rnd() < 0.5 ? 1 : 0)),
  sorted: Array.from({ length: 48 }, (_, i) => (i < 24 ? 0 : 1)),
};
function predictRun(seq, kind) {
  let st = 2, last = 1; const out = [];
  for (const o of seq) {
    const p = kind === "always" ? 1 : kind === "one" ? last : st >= 2 ? 1 : 0;
    out.push({ o, p, st }); last = o; st = o ? Math.min(3, st + 1) : Math.max(0, st - 1);
  }
  return out;
}

const MILESTONES = [
  [1945, "EDVAC report", "stored program", "John von Neumann's report describes a machine that keeps its program in the same memory as its data and runs it in a loop: fetch an instruction, decode it, execute it. Every CPU since runs that loop."],
  [1971, "Intel 4004", "a CPU on one chip", "2,300 transistors, 740 kHz, 4-bit. About 92,000 instructions per second, one at a time. Built for a calculator."],
  [1978, "Intel 8086", "x86 begins", "29,000 transistors. Its instruction set, extended many times, still runs in most laptops and servers."],
  [1985, "RISC: MIPS, ARM", "simple instructions", "Berkeley RISC, Stanford MIPS and Acorn's ARM (1985) bet on few, simple, fixed-size instructions that are easy to pipeline. ARM now runs every phone."],
  [1989, "Intel 486", "pipeline + cache", "A 5-stage pipeline and an 8 KB cache on the chip: close to one instruction finished per clock."],
  [1993, "Pentium", "superscalar", "Two pipelines side by side: up to two instructions per clock when they don't depend on each other."],
  [1995, "Pentium Pro", "out-of-order", "Translates x86 into simple micro-ops, runs them out of order and speculatively past branches, then retires them in order. Still the template."],
  [2000, "Pentium 4", "chasing GHz", "A 20-stage pipeline (31 in 2004) to push the clock to 3.8 GHz. Power and mispredicted branches made it a dead end."],
  [2005, "Multicore", "two cores per chip", "Athlon 64 X2 and Pentium D, then Core 2 (2006): the extra transistors went into more cores, because a faster clock was no longer affordable."],
  [2018, "Spectre & Meltdown", "the cost of guessing", "Speculative execution was shown to leak secrets through cache timing. Guessing ahead is how CPUs are fast, and it is now also a security problem."],
  [2020, "Apple M1", "very wide", "8 instructions decoded per clock, a reorder window of ~600, unified memory: a wide, deep core at a modest 3.2 GHz."],
];

export default function (root, { id }) {
  // ---------- 1. the fetch-decode-execute loop ----------
  let C;
  const reset = () => { C = { pc: 0, r: { r0: 0, r1: 0, r2: 0, r3: 4 }, ph: 0, ir: null, npc: 0, res: null, addr: null, n: 0, steps: 0, done: false }; };
  reset();
  const cc = canvas(880, 300, "A tiny CPU running a loop: program counter, instruction memory, registers, ALU and data memory");
  const cst = stats([{ label: "Instructions finished" }, { label: "Phases (clock ticks)" }, { label: "Sum so far (r2)" }]);
  function stepC() {
    if (C.done) return;
    const ph = PHASES[C.ph];
    if (ph === "fetch") { C.ir = PROG[C.pc]; C.npc = C.pc + 1; }
    else if (ph === "execute") {
      const o = C.ir.op, r = C.r; C.addr = null;
      if (o === "LOAD") { C.addr = r.r0; C.res = ["r1", DMEM[r.r0]]; }
      else if (o === "ADD") C.res = ["r2", r.r2 + r.r1];
      else if (o === "ADDI") C.res = ["r0", r.r0 + 1];
      else if (o === "SUBI") C.res = ["r3", r.r3 - 1];
      else if (o === "BNZ") { C.res = null; if (r.r3 !== 0) C.npc = 0; }
      else { C.res = null; }
    } else if (ph === "write back") {
      if (C.res) C.r[C.res[0]] = C.res[1];
      C.n++; if (C.ir.op === "HALT") C.done = true; else C.pc = C.npc;
    }
    C.steps++; C.ph = (C.ph + 1) % 4;
    drawC();
  }
  const anim = loop((dt) => { anim.t = (anim.t || 0) + dt; if (anim.t > 0.35) { anim.t = 0; stepC(); if (C.done) anim.stop(); } }, cc.el);
  function runC() { if (C.done) reset(); if (reducedMotion()) { while (!C.done) stepC(); return; } anim.start(); }
  function drawC() {
    cst.set(0, String(C.n)); cst.set(1, String(C.steps)); cst.set(2, String(C.r.r2), C.done ? "good" : "");
    cc.paint(() => {
      cc.clear(); const c = cc.colors, ac = c.accent, last = PHASES[(C.ph + 3) % 4], fresh = C.steps > 0;
      const on = (...ps) => fresh && ps.includes(last);
      const unit = (x, y, w, hh, title, act) => { cc.box(x, y, w, hh, { fill: act ? `color-mix(in srgb, ${ac} 12%, ${c.surface})` : c.surface, stroke: act ? ac : c.line, lw: act ? 2.5 : 1.2, r: 8 }); cc.text(title, x + 8, y + 12, { size: 10, color: act ? ac : c["ink-3"], mono: true }); };
      const wire = (pts, act) => { for (let k = 1; k < pts.length; k++) cc.line(pts[k - 1][0], pts[k - 1][1], pts[k][0], pts[k][1], { color: act ? ac : c.line, lw: act ? 2.5 : 1.5 }); const [x, y] = pts[pts.length - 1]; cc.dot(x, y, act ? 4 : 3, act ? ac : c["ink-3"]); };
      // program counter
      unit(16, 110, 84, 56, "PC", on("fetch", "write back")); cc.text(String(C.pc), 58, 145, { size: 20, align: "center", mono: true });
      // instruction memory
      unit(126, 22, 214, 200, "instruction memory", on("fetch"));
      PROG.forEach((p, k) => { const y = 58 + k * 27, cur = k === C.pc; if (cur) cc.box(132, y - 11, 202, 22, { fill: `color-mix(in srgb, ${c.c4} 30%, ${c.surface})`, stroke: "transparent", r: 4 }); cc.text(`${k}  ${p.txt}`, 140, y, { size: 11, mono: true, color: cur ? c.ink : c["ink-2"] }); });
      // instruction register + decode
      unit(366, 22, 220, 58, "instruction register", on("fetch", "decode"));
      cc.text(C.ir ? C.ir.txt : "(empty)", 378, 54, { size: 12.5, mono: true, color: C.ir ? c.ink : c["ink-3"] });
      if (C.ir) cc.text(C.ir.note, 476, 70, { size: 9.5, color: c["ink-3"], align: "center" });
      // registers
      unit(366, 98, 220, 124, "registers", on("decode", "write back"));
      Object.entries(C.r).forEach(([k, v], i) => {
        const x = 378 + (i % 2) * 104, y = 128 + Math.floor(i / 2) * 46, hit = fresh && last === "write back" && C.res && C.res[0] === k;
        cc.box(x, y - 14, 94, 34, { fill: hit ? `color-mix(in srgb, ${c.good} 25%, ${c.surface})` : c.sunken, stroke: hit ? c.good : "transparent", lw: 2, r: 6 });
        cc.text(k, x + 10, y + 3, { size: 11, color: c["ink-3"], mono: true }); cc.text(String(v), x + 84, y + 3, { size: 15, align: "right", mono: true });
      });
      // ALU
      unit(616, 98, 100, 124, "ALU", on("execute"));
      const res = C.res && fresh && (last === "execute" || last === "write back") ? `${C.res[0]} ← ${C.res[1]}` : C.ir && C.ir.op === "BNZ" && last === "execute" ? (C.npc === 0 ? "jump to 0" : "fall through") : "";
      cc.text(res, 666, 168, { size: 12, align: "center", mono: true });
      // data memory
      unit(744, 22, 120, 200, "data memory", on("execute") && C.ir && C.ir.op === "LOAD");
      DMEM.forEach((v, k) => { const y = 54 + k * 40, hit = C.addr === k && fresh && last === "execute"; cc.box(756, y - 14, 96, 30, { fill: hit ? `color-mix(in srgb, ${c.c4} 35%, ${c.surface})` : c.sunken, stroke: hit ? c.c4 : "transparent", lw: 2, r: 5 }); cc.text(`[${k}]`, 766, y + 1, { size: 10.5, color: c["ink-3"], mono: true }); cc.text(String(v), 842, y + 1, { size: 14, align: "right", mono: true }); });
      // wires
      wire([[100, 138], [126, 138]], on("fetch"));
      wire([[340, 50], [366, 50]], on("fetch"));
      wire([[476, 80], [476, 98]], on("decode"));
      wire([[586, 160], [616, 160]], on("execute"));
      wire([[716, 120], [744, 120]], on("execute") && C.ir?.op === "LOAD");
      wire([[666, 222], [666, 246], [476, 246], [476, 222]], on("write back") && !!C.res);
      wire([[640, 222], [640, 262], [58, 262], [58, 166]], on("write back"));
      // phase strip
      PHASES.forEach((p, k) => { const x = 140 + k * 150, act = fresh && p === last; cc.box(x, 274, 140, 20, { fill: act ? ac : c.sunken, stroke: "transparent", r: 10 }); cc.text(`${k + 1}. ${p}`, x + 70, 284, { size: 10.5, align: "center", color: act ? c.surface : c["ink-2"] }); });
      if (C.done) cc.text(`HALT: 5 + 3 + 8 + 2 = ${C.r.r2}, after ${C.n} instructions and ${C.steps} phases`, 440, 233, { size: 11.5, align: "center", color: c.good });
    });
  }

  // ---------- 2. pipelining ----------
  const pp = { mode: "fwd" };
  const pcv = canvas(880, 320, "A pipeline diagram: which stage each instruction is in, clock by clock");
  const pst = stats([{ label: "Clocks for all 21 instructions" }, { label: "Clocks per instruction" }, { label: "Speed-up over one at a time" }]);
  const SEQ = schedule("seq"), seqT = SEQ[SEQ.length - 1].WB + 1;
  function drawP() {
    const S = schedule(pp.mode), T = S[S.length - 1].WB + 1;
    pst.set(0, String(T)); pst.set(1, (T / TRACE.length).toFixed(2), T / TRACE.length < 1.5 ? "good" : T / TRACE.length < 3 ? "warn" : "bad"); pst.set(2, `${(seqT / T).toFixed(1)}×`);
    pcv.paint(() => {
      pcv.clear(); const c = pcv.colors, N = 10, x0 = 160, cw = 19.4, y0 = 34, rh = 24, maxC = 37;
      const col = { IF: c.c1, ID: c.c6, EX: c.c4, MEM: c.c3, WB: c.c5 };
      for (let k = 0; k < maxC; k++) if (k % 5 === 0) pcv.text(String(k), x0 + k * cw + cw / 2, 18, { size: 9.5, color: c["ink-3"], align: "center", mono: true });
      pcv.text("clock →", x0 + maxC * cw, 18, { size: 9.5, color: c["ink-3"], align: "right", mono: true });
      for (let i = 0; i < N; i++) {
        const s = S[i], y = y0 + i * rh;
        pcv.text(TRACE[i].txt, 10, y + rh / 2 - 1, { size: 10.5, mono: true, color: TRACE[i].branch ? c.c2 : c.ink });
        const cells = [["IF", s.IF], ...Array.from({ length: s.ID - s.IF - 1 }, (_, k) => ["stall", s.IF + 1 + k]), ["ID", s.ID], ["EX", s.EX], ["MEM", s.MEM], ["WB", s.WB]];
        for (const [st, t] of cells) {
          if (t >= maxC) continue;
          const x = x0 + t * cw;
          if (st === "stall") { pcv.box(x + 1, y + 2, cw - 2, rh - 5, { fill: c.sunken, stroke: c.line, lw: 1, r: 3 }); pcv.text("·", x + cw / 2, y + rh / 2 - 1, { size: 12, color: c["ink-3"], align: "center" }); continue; }
          pcv.box(x + 1, y + 2, cw - 2, rh - 5, { fill: `color-mix(in srgb, ${col[st]} 55%, ${c.surface})`, stroke: "transparent", r: 3 });
          pcv.text({ IF: "F", ID: "D", EX: "E", MEM: "M", WB: "W" }[st], x + cw / 2, y + rh / 2 - 1, { size: 9, align: "center", mono: true });
        }
      }
      const lx = 160; [["F fetch", c.c1], ["D decode", c.c6], ["E execute", c.c4], ["M memory", c.c3], ["W write back", c.c5]].forEach(([n, cl], k) => { pcv.box(lx + k * 118, 302, 10, 10, { fill: `color-mix(in srgb, ${cl} 55%, ${c.surface})`, stroke: "transparent", r: 2 }); pcv.text(n, lx + k * 118 + 15, 307, { size: 10, color: c["ink-2"] }); });
      pcv.box(lx + 5 * 118, 302, 10, 10, { fill: c.sunken, stroke: c.line, lw: 1, r: 2 }); pcv.text("stall", lx + 5 * 118 + 15, 307, { size: 10, color: c["ink-2"] });
      pcv.text(`first 10 of 21 instructions shown · ${pp.mode === "seq" ? "each waits for the one before to finish" : pp.mode === "stall" ? "waits for results to be written, and for every branch to resolve" : "results forwarded straight to the next ALU; the loop branch predicted taken"}`, 10, y0 + N * rh + 12, { size: 10.5, color: c["ink-3"] });
    });
  }

  // ---------- 3. branch prediction ----------
  const bp = { pat: "loop", kind: "two" };
  const bcv = canvas(880, 200, "A stream of branch outcomes and whether the predictor guessed each one right");
  const bst = stats([{ label: "Guessed right" }, { label: "Wrong guesses" }, { label: "Clocks thrown away (15 each)" }]);
  function drawB() {
    const R = predictRun(PAT[bp.pat], bp.kind), ok = R.filter((x) => x.o === x.p).length, bad = R.length - ok;
    bst.set(0, `${Math.round((100 * ok) / R.length)}%`, ok / R.length > 0.9 ? "good" : ok / R.length > 0.7 ? "warn" : "bad");
    bst.set(1, `${bad} of ${R.length}`); bst.set(2, String(bad * 15), bad * 15 < 100 ? "good" : bad * 15 < 300 ? "warn" : "bad");
    bcv.paint(() => {
      bcv.clear(); const c = bcv.colors, cw = 34;
      for (let row = 0; row < 2; row++) {
        const y = 28 + row * 84;
        R.slice(row * 24, row * 24 + 24).forEach((r, k) => {
          const x = 30 + k * cw, hit = r.o === r.p;
          bcv.box(x, y, cw - 4, 22, { fill: `color-mix(in srgb, ${r.o ? c.c3 : c.c2} 45%, ${c.surface})`, stroke: "transparent", r: 4 });
          bcv.text(r.o ? "T" : "N", x + (cw - 4) / 2, y + 11, { size: 10.5, align: "center", mono: true });
          if (bp.kind === "two") for (let s = 0; s < 4; s++) bcv.box(x + 3 + s * 6.5, y + 27, 5, 5, { fill: s <= r.st ? c.accent : c.sunken, stroke: "transparent", r: 1 });
          bcv.text(hit ? "✓" : "✗", x + (cw - 4) / 2, y + 44, { size: 12, align: "center", color: hit ? c.good : c.bad });
        });
      }
      bcv.text("T = taken (jump), N = not taken", 30, 14, { size: 10, color: c["ink-3"], mono: true });
      if (bp.kind === "two") bcv.text("■■■■ the 2-bit counter before each branch: 2 or more predicts taken", 860, 14, { size: 10, color: c["ink-3"], mono: true, align: "right" });
      bcv.text({ loop: "an inner loop of 8: taken 7 times, then falls through", random: "branching on random data: nothing to learn", sorted: "the same test on sorted data: one switch from N to T" }[bp.pat], 440, 192, { size: 11.5, align: "center" });
    });
  }

  // ---------- 4. out of order ----------
  const oo = { inOrder: false, width: 2, miss: false };
  const ocv = canvas(880, 300, "Ten instructions on a timeline: when each starts and finishes, in order or out of order");
  const ost = stats([{ label: "Clocks to finish" }, { label: "Instructions per clock" }, { label: "Longest dependency chain" }]);
  function drawO() {
    const R = runOOO(oo);
    const chain = (() => { const f = []; OOO.forEach((o, i) => { f[i] = R.lat[i] + Math.max(0, ...o.dep.map((d) => f[d])); }); return Math.max(...f); })();
    ost.set(0, String(R.total)); ost.set(1, (OOO.length / R.total).toFixed(2), OOO.length / R.total > 0.8 ? "good" : OOO.length / R.total > 0.3 ? "warn" : "bad"); ost.set(2, `${chain} clocks`);
    ocv.paint(() => {
      ocv.clear(); const c = ocv.colors, x0 = 170, x1 = 860, X = (t) => x0 + (t / Math.max(12, R.total)) * (x1 - x0), rh = 24;
      OOO.forEach((o, i) => {
        const y = 22 + i * rh, s = R.start[i], f = R.fin[i];
        ocv.text(o.t, 10, y + 10, { size: 10.5, mono: true, color: o.dep.length ? c.ink : c.good });
        ocv.box(X(s), y + 1, Math.max(4, X(f) - X(s) - 1), rh - 6, { fill: `color-mix(in srgb, ${o.load ? c.c4 : o.t.startsWith("MUL") ? c.c6 : c.c1} 50%, ${c.surface})`, stroke: i === 0 && oo.miss ? c.bad : "transparent", lw: 2, r: 4 });
        if (X(f) - X(s) > 40) ocv.text(i === 0 && oo.miss ? "cache miss: 200 clocks" : `${R.lat[i]}`, X(s) + 6, y + 10, { size: 9.5, mono: true });
        for (const d of o.dep) ocv.line(X(R.fin[d]) - 1, 22 + d * rh + rh - 6, X(s) + 1, y + 2, { color: `color-mix(in srgb, ${c["ink-3"]} 50%, transparent)`, lw: 1 });
      });
      const T = Math.max(12, R.total), stepT = T > 100 ? 50 : T > 30 ? 10 : 2;
      for (let t = 0; t <= T; t += stepT) { ocv.line(X(t), 262, X(t), 266, { color: c["ink-3"] }); ocv.text(String(t), X(t), 274, { size: 9.5, align: "center", color: c["ink-3"], mono: true }); }
      ocv.text("clock →", x1, 288, { size: 9.5, align: "right", color: c["ink-3"], mono: true });
      ocv.text("green: needs nothing earlier · lines: waits for a result", 10, 288, { size: 10, color: c["ink-3"] });
    });
  }

  // ---------- 5. inside a modern core ----------
  const CORE = [["front", "Front end"], ["ooo", "Out-of-order engine"], ["exec", "Execution units"], ["mem", "Caches"]];
  const CNOTE = {
    front: "predicts branches, fetches and decodes 4–8 instructions per clock, keeps decoded micro-ops in a cache",
    ooo: "renames registers and tracks 300–600 instructions in flight, so it can start any whose inputs are ready",
    exec: "the part that does arithmetic: a handful of integer, vector and load/store units, a small slice of the area",
    mem: "L1 (~32–64 KB) beside the core, L2 (1–2 MB) per core, L3 (tens of MB) shared by all: most of the chip",
  };
  const kc = { part: "exec" };
  const kcv = canvas(880, 330, "A CPU chip of 8 cores around a shared L3 cache, and one core in detail");
  kcv.paint(() => {
    kcv.clear(); const c = kcv.colors, P = kc.part, ac = c.accent;
    const tone = (col, k) => `color-mix(in srgb, ${col} ${P === k ? 60 : 22}%, ${c.surface})`, edge = (k) => (P === k ? { stroke: ac, lw: 2.5 } : { stroke: "transparent" });
    // chip
    kcv.box(10, 10, 360, 290, { fill: c.sunken, stroke: c.line, r: 10 }); kcv.text("the chip: 8 cores", 20, 24, { size: 10.5, color: c["ink-3"], mono: true });
    for (let k = 0; k < 8; k++) {
      const x = 22 + (k % 4) * 86, y = k < 4 ? 36 : 210;
      kcv.box(x, y, 78, 78, { fill: c.surface, stroke: k === 0 ? ac : c.line, lw: k === 0 ? 2.5 : 1, r: 6 });
      kcv.box(x + 4, y + 4, 70, 16, { fill: tone(c.c1, "front"), stroke: "transparent", r: 3 });
      kcv.box(x + 4, y + 23, 44, 26, { fill: tone(c.c6, "ooo"), stroke: "transparent", r: 3 });
      kcv.box(x + 51, y + 23, 23, 26, { fill: tone(c.c4, "exec"), stroke: "transparent", r: 3 });
      kcv.box(x + 4, y + 52, 70, 22, { fill: tone(c.c3, "mem"), stroke: "transparent", r: 3 });
    }
    kcv.box(22, 122, 336, 80, { fill: tone(c.c3, "mem"), ...edge("mem"), r: 6 }); kcv.text("L3 cache, shared: 32–96 MB", 190, 162, { size: 11.5, align: "center" });
    kcv.line(100, 36, 400, 24, { color: ac, lw: 1.2, dash: [3, 3] });
    // one core
    const x = 400, y = 14; kcv.box(x, y, 470, 300, { fill: c.surface, stroke: c.line, r: 10 }); kcv.text("one core", x + 10, y + 14, { size: 10.5, color: c["ink-3"], mono: true });
    const blk = (bx, by, bw, bh, t, sub, col, k) => { kcv.box(bx, by, bw, bh, { fill: tone(col, k), ...edge(k), r: 6 }); kcv.text(t, bx + bw / 2, by + bh / 2 - (sub ? 7 : 0), { size: 11, align: "center" }); if (sub) kcv.text(sub, bx + bw / 2, by + bh / 2 + 9, { size: 9.5, align: "center", color: c["ink-2"], mono: true }); };
    blk(x + 12, y + 28, 140, 52, "branch predictor", "guesses ahead", c.c1, "front");
    blk(x + 162, y + 28, 140, 52, "fetch + decode", "4–8 per clock", c.c1, "front");
    blk(x + 312, y + 28, 146, 52, "micro-op cache", "decoded already", c.c1, "front");
    blk(x + 12, y + 90, 210, 64, "rename + reorder buffer", "300–600 in flight", c.c6, "ooo");
    blk(x + 232, y + 90, 226, 64, "schedulers", "start whatever is ready", c.c6, "ooo");
    ["ALU", "ALU", "ALU", "ALU", "vec", "vec", "ld", "ld", "st"].forEach((t, k) => blk(x + 12 + k * 50, y + 164, 44, 40, t, null, c.c4, "exec"));
    blk(x + 12, y + 214, 220, 74, "L1 data + L1 instr", "~32–64 KB each", c.c3, "mem");
    blk(x + 242, y + 214, 216, 74, "L2", "1–2 MB", c.c3, "mem");
    kcv.text(CNOTE[P], 440, 322, { size: 11.5, align: "center" });
  });

  // ---------- 6. history ----------
  let hstep = 0;
  const hcv = canvas(880, 190, "A timeline of CPU milestones from 1945 to 2020");
  hcv.paint(() => {
    hcv.clear(); const c = hcv.colors, X = (yr) => 40 + ((yr - 1940) / 85) * 800, y = 96;
    hcv.line(30, y, 850, y, { color: c.line, lw: 2 });
    for (let yr = 1940; yr <= 2020; yr += 10) { hcv.line(X(yr), y - 4, X(yr), y + 4, { color: c["ink-3"] }); hcv.text(String(yr), X(yr), y + 16, { size: 9.5, align: "center", color: c["ink-3"], mono: true }); }
    MILESTONES.forEach(([yr, name, idea], k) => {
      const cur = k === hstep, up = k % 2 === 0, ly = up ? y - 34 - (k % 4 === 0 ? 0 : 22) : y + 40 + (k % 4 === 1 ? 0 : 22);
      hcv.line(X(yr), y, X(yr), ly + (up ? 8 : -8), { color: cur ? c.accent : c.line, lw: cur ? 2 : 1 });
      hcv.dot(X(yr), y, cur ? 7 : 4.5, cur ? c.accent : k <= hstep ? c.c1 : c["ink-3"]);
      hcv.text(name, X(yr), ly, { size: cur ? 11.5 : 10, align: "center", color: cur ? c.accent : c["ink-2"], weight: cur ? 700 : 600 });
    });
    const [yr, name, idea] = MILESTONES[hstep]; hcv.text(`${yr} · ${name}: ${idea}`, 440, 182, { size: 12, align: "center" });
  });

  // ---------- caches: a tiny fully associative LRU cache over 256 lines of memory ----------
  const ca = { pat: "seq", lines: 16 };
  const LINES = 256, PER = 16, HIT = 4, MISS = 250, NACC = 1024;
  const PATS = [["seq", "Walk in order"], ["stride", "Skip 16 (one per line)"], ["col", "2D array by column"], ["reuse", "Small loop, reused"], ["rand", "Random"]];
  function accesses(pat) {
    let x = 12345; const rnd = () => ((x = (x * 1103515245 + 12345) % 2147483648) / 2147483648);
    const out = [];
    for (let k = 0; k < NACC; k++) {
      if (pat === "seq") out.push(k % (LINES * PER));
      else if (pat === "stride") out.push((k * PER) % (LINES * PER));
      else if (pat === "col") { const n = 64, r = k % n, cc = Math.floor(k / n) % n; out.push(r * n + cc); }   // 64 × 64 ints, row-major, walked down columns
      else if (pat === "reuse") out.push(k % (6 * PER));
      else out.push(Math.floor(rnd() * LINES * PER));
    }
    return out;
  }
  const cst2 = stats([{ label: "Hit rate" }, { label: "Average time per access" }, { label: "Slower than all hits by" }]);
  const ccv = canvas(880, 300, "Memory as 256 cache lines, coloured by hits and misses, and the first accesses as a strip");
  function drawCache() {
    const acc = accesses(ca.pat), lru = new Map(), state = new Uint8Array(LINES), seq = [];
    let hits = 0;
    for (const a of acc) {
      const line = Math.floor(a / PER);
      if (lru.has(line)) { hits++; lru.delete(line); seq.push(1); if (state[line] < 2) state[line] = 2; }
      else { seq.push(0); state[line] = state[line] === 2 ? 3 : Math.max(state[line], 1); if (lru.size >= ca.lines) lru.delete(lru.keys().next().value); }
      lru.set(line, true);
    }
    const hr = hits / acc.length, amat = HIT + (1 - hr) * MISS;
    cst2.set(0, `${Math.round(hr * 100)}%`, hr > 0.9 ? "good" : hr > 0.6 ? "warn" : "bad");
    cst2.set(1, `${amat.toFixed(0)} cycles`, amat < 30 ? "good" : amat < 100 ? "warn" : "bad");
    cst2.set(2, `${(amat / HIT).toFixed(1)}×`, amat < 30 ? "good" : amat < 100 ? "warn" : "bad");
    ccv.paint(() => {
      ccv.clear(); const c = ccv.colors, cs = 13, x0 = 20, y0 = 30;
      ccv.text("memory: 256 lines of 64 bytes (16 numbers each)", x0, 18, { size: 11, color: c["ink-3"] });
      for (let l = 0; l < LINES; l++) {
        const i = l % 16, j = Math.floor(l / 16), st = state[l];
        const fill = st === 0 ? c.sunken : st === 1 ? `color-mix(in srgb, ${c.bad} 55%, ${c.surface})` : st === 2 ? `color-mix(in srgb, ${c.good} 55%, ${c.surface})` : `color-mix(in srgb, ${c.warn} 60%, ${c.surface})`;
        ccv.box(x0 + i * cs, y0 + j * cs, cs - 2, cs - 2, { fill, stroke: lru.has(l) ? c.accent : "transparent", lw: 2, r: 3 });
      }
      const ky = y0 + 16 * cs + 16;
      [[c.good, "hit"], [c.bad, "only missed"], [c.warn, "hit, pushed out, missed again"]].forEach(([col, t], k) => { ccv.box(x0 + [0, 44, 134][k], ky - 9, 10, 10, { fill: col, stroke: "transparent", r: 2 }); ccv.text(t, x0 + [0, 44, 134][k] + 14, ky, { size: 10.5, color: c["ink-2"] }); });
      ccv.box(x0 + 0, ky + 12, 10, 10, { fill: c.surface, stroke: c.accent, lw: 2, r: 2 }); ccv.text(`in the cache at the end (${ca.lines} lines = ${ca.lines * 64 / 1024} KB)`, x0 + 14, ky + 21, { size: 10.5, color: c["ink-2"] });
      // the strip: first 192 accesses
      const sx = 300, sw = 560, n = 192, w = sw / n;
      ccv.text("the first 192 accesses, in order (each tick: a hit or a miss)", sx, 18, { size: 11, color: c["ink-3"] });
      for (let k = 0; k < n; k++) ccv.box(sx + k * w, 30, Math.max(1, w - 0.6), 34, { fill: seq[k] ? c.good : c.bad, stroke: "transparent", r: 1 });
      // time bar: hits cost 4 cycles, misses 250
      const tot = seq.slice(0, n).reduce((t, s) => t + (s ? HIT : MISS), 0), best = n * HIT;
      ccv.text("time those 192 accesses took", sx, 92, { size: 11, color: c["ink-3"] });
      ccv.box(sx, 100, sw, 18, { fill: c.sunken, stroke: "transparent", r: 9 });
      ccv.box(sx, 100, Math.max(10, (sw * tot) / (n * MISS)), 18, { fill: tot / best < 8 ? c.good : tot / best < 25 ? c.warn : c.bad, stroke: "transparent", r: 9 });
      ccv.text(`${tot.toLocaleString("en-US")} cycles (all hits: ${best}; all misses: ${(n * MISS).toLocaleString("en-US")})`, sx, 136, { size: 11.5 });
      const why = { seq: "Each miss brings in 64 bytes: the next 15 numbers are free. That's spatial locality.", stride: "One number per line: every access is a new line, so the free neighbours are wasted.", col: "Down a column, each step jumps a whole row (256 bytes): a new line every time, just like the stride.", reuse: "The same 6 lines again and again: once they're in, everything hits. That's temporal locality.", rand: "No pattern to exploit: only the share of memory that fits in the cache hits." }[ca.pat];
      ccv.text(why, sx, 170, { size: 12.5, color: c.ink });
      ccv.text(`A hit costs ~${HIT} cycles; a trip to DRAM ~${MISS} (about 60 ns at 4 GHz).`, sx, 196, { size: 11.5, color: c["ink-2"] });
      ccv.text("Real CPUs add hardware prefetchers that spot simple strides and fetch ahead,", sx, 226, { size: 11, color: c["ink-3"] });
      ccv.text("so the stride pattern does better on real hardware than here. Random access can't be helped.", sx, 242, { size: 11, color: c["ink-3"] });
    });
  }

  // ---------- SIMD: one instruction, many numbers ----------
  const sm = { bits: 512, dt: 32 };
  const sst = stats([{ label: "Numbers per instruction (lanes)" }, { label: "Peak per core per clock (2 FMA units)" }, { label: "A GPU warp, for comparison" }]);
  const scv = canvas(880, 220, "A vector register split into lanes: one instruction adds every lane at once");
  function drawSimd() {
    const lanes = sm.bits / sm.dt, peak = 2 * lanes * 2;
    sst.set(0, String(lanes)); sst.set(1, `${peak} ${sm.dt === 8 ? "int8 ops" : sm.dt === 16 ? "bf16 FLOPs" : "fp32 FLOPs"}`, peak >= 64 ? "good" : "warn"); sst.set(2, "32 lanes per instruction");
    scv.paint(() => {
      scv.clear(); const c = scv.colors, x0 = 110, W = 740, lw = W / lanes;
      const row = (y, name, col, val) => {
        scv.text(name, x0 - 10, y + 19, { size: 12, align: "right", color: c["ink-2"] });
        for (let k = 0; k < lanes; k++) {
          scv.box(x0 + k * lw + 1, y, lw - 2, 28, { fill: `color-mix(in srgb, ${col} ${40 + ((k * 37) % 30)}%, ${c.surface})`, stroke: "transparent", r: Math.min(5, lw / 4) });
          if (lw > 22) scv.text(String(val(k)), x0 + k * lw + lw / 2, y + 19, { size: lw > 40 ? 12 : 9.5, align: "center", mono: true });
        }
      };
      row(20, "a", c.c1, (k) => k + 1); row(62, "b", c.c4, (k) => 10 * ((k % 3) + 1));
      scv.text("+", 60, 116, { size: 20, align: "center" });
      row(110, "a + b", c.accent, (k) => k + 1 + 10 * ((k % 3) + 1));
      scv.text(`one instruction (e.g. ${sm.bits === 512 ? "AVX-512 VADDPS" : sm.bits === 256 ? "AVX2 VADDPS" : "SSE / NEON add"}) works on all ${lanes} lanes of a ${sm.bits}-bit register at once`, x0, 172, { size: 12.5 });
      scv.text("it only helps when the same operation applies to many numbers side by side: exactly what a matrix multiply is", x0, 194, { size: 11.5, color: c["ink-2"] });
    });
  }

  root.append(
    section(null,
      goal("<p>A CPU has one job: run <b>any</b> program, one instruction after another, and finish it as soon as possible. Programs branch, depend on their own earlier results and reach into memory unpredictably, so the CPU spends most of its transistors guessing what comes next. Seeing why it works this way, and where it stops, explains why AI needed a different machine.</p>",
        ["Run any program, in order", "Finish one thread as fast as possible", "Hide that memory is slow"]),
      stackMap(id),
      h("div.vx-row", sticker("cpudie", { title: "A CPU chip", caption: "8 big cores (green) around a big shared cache (blue)", px: 4 }), sticker("pipeline", { title: "A pipeline", caption: "five stages, each working on a different instruction at once", px: 4 }), sticker("chip", { title: "Where it started", caption: "1971: one 4-bit core, 2,300 transistors", px: 4 })),
    ),
    section("Fetch, decode, execute",
      prose("<p>Every CPU runs the same loop, described by John von Neumann in 1945. The <b>program counter</b> (PC) holds the address of the next instruction. <b>Fetch</b> it from memory, <b>decode</b> it into control signals, <b>execute</b> it in the arithmetic unit (ALU), <b>write back</b> the result, move the PC on. Here a 6-instruction program adds up four numbers. Step it one phase at a time.</p>"),
      h("div.stage",
        h("div.stage-controls", h("div.btn-row", button("Next phase", () => { if (C.done) reset(); stepC(); }, "primary"), button("Run", runC), button("Reset", () => { anim.stop(); reset(); drawC(); }))),
        h("div.stage-body", cst.el, h("div.wide-canvas", cc.el))),
      callout("note", "Instructions are just numbers", "<p>The program sits in memory as bit patterns, exactly like the data. That's the <b>stored-program</b> idea: change the numbers and the same hardware does something else. This simple CPU takes 4 ticks per instruction, so 21 instructions take 84. The next three sections are how real CPUs got that down to well under one tick per instruction.</p>"),
    ),
    section("Pipelining: an assembly line",
      prose("<p>While one instruction executes, the next can already be decoding and the one after that fetching, like a laundry with a washer and a dryer. With five stages, five instructions are in flight at once. The catch is <b>hazards</b>: an instruction that needs a result not yet written must wait (a <b>stall</b>), and after a branch the CPU doesn't know which instruction comes next. Compare three machines on the loop above, all 21 instructions it runs.</p>"),
      h("div.stage",
        h("div.stage-controls", segmented({ label: "Machine", options: [{ value: "seq", label: "One at a time" }, { value: "stall", label: "Pipelined, waits" }, { value: "fwd", label: "Forwarding + prediction" }], value: pp.mode, onChange: (v) => { pp.mode = v; drawP(); } })),
        h("div.stage-body", pst.el, h("div.wide-canvas", pcv.el))),
      predict({ chapter: id, question: "A pipeline has 5 stages. In the best case, how much faster can it finish a long program than doing one instruction at a time?", options: ["5×", "25×", "No faster: each instruction still takes 5 steps", "Unlimited, if the clock is fast enough"], answer: 0, explain: "<p>Each instruction still takes 5 steps (latency doesn't improve), but one finishes every clock instead of every 5: throughput up to 5×. Stalls and branches eat into it, which is why deeper pipelines (the Pentium 4's 31 stages) paid off less than hoped: more stages means more in flight to throw away on every wrong guess.</p>" }),
    ),
    section("Guessing the future: branch prediction",
      prose("<p>About one instruction in five is a branch. A modern pipeline is 15–20 stages deep, so waiting for every branch to resolve would waste most of the time. Instead the CPU <b>guesses</b> and runs ahead; if it guessed wrong it throws the work away (~15 clocks). A <b>2-bit counter</b> per branch remembers which way it usually goes and needs two surprises in a row to change its mind. Try it on three kinds of branch.</p>"),
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Branch", options: [{ value: "loop", label: "Loop" }, { value: "random", label: "Random data" }, { value: "sorted", label: "Sorted data" }], value: bp.pat, onChange: (v) => { bp.pat = v; drawB(); } }),
          segmented({ label: "Predictor", options: [{ value: "always", label: "Always taken" }, { value: "one", label: "Same as last time" }, { value: "two", label: "2-bit counter" }], value: bp.kind, onChange: (v) => { bp.kind = v; drawB(); } })),
        h("div.stage-body", bst.el, h("div.wide-canvas", bcv.el))),
      predict({ chapter: id, question: "A loop adds up only the bytes ≥ 128 in a big array of random bytes. Sorting the array first (not counting the sort) makes the loop several times faster. Why?", options: ["Sorted data fits in the cache better", "The branch becomes predictable: all the small values, then all the large", "The compiler removes the loop", "Sorted numbers are faster to add"], answer: 1, explain: "<p>Same data, same cache behaviour, but on random data the <code>if</code> is a coin flip and half the guesses are wrong, ~15 clocks each. Sorted, the branch goes one way for the first half and the other way for the second: almost every guess is right. (A compiler that turns the <code>if</code> into a branch-free conditional move erases the difference, which is a good trick to know.)</p>" }),
      prose("<p>Step through a 2-bit predictor, branch by branch:</p>"),
      walk("cpu"),
    ),
    section("Out of order: start whatever is ready",
      prose("<p>Programs are written in order, but much of the work doesn't depend on what came just before. A modern core reads far ahead (hundreds of instructions), starts any whose inputs are ready, and puts the results back in program order at the end. Here ten instructions form two chains plus a little independent work. Widen the machine, let it go out of order, and make the first load miss the cache.</p>"),
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Order", options: [{ value: "in", label: "In order" }, { value: "ooo", label: "Out of order" }], value: "ooo", onChange: (v) => { oo.inOrder = v === "in"; drawO(); } }),
          slider({ label: "Instructions started per clock", min: 1, max: 4, value: oo.width, onInput: (v) => { oo.width = v; drawO(); } }),
          toggle({ label: "First load misses the cache", checked: false, onChange: (v) => { oo.miss = v; drawO(); } })),
        h("div.stage-body", ost.el, h("div.wide-canvas", ocv.el))),
      callout("tip", "The wall it hits", "<p>However wide the machine, it can't beat the <b>longest dependency chain</b>: c needs b needs a. Out of order hides small waits by doing other work, but a 200-clock cache miss on the chain still stalls the finish. A real core has about 2–5 independent instructions to find per clock in typical code. That's the ceiling on one thread, and it's why the GPU took a different road: thousands of threads with no dependencies between them.</p>"),
    ),
    section("Caches: keep the hot data close",
      prose("<p>A trip to DRAM takes ~60 ns, about 250 clock ticks, during which a core could have run ~1,000 instructions. So a CPU spends a large share of its area on <b>caches</b>: small, fast SRAM copies of recently used memory. They work because programs are predictable in two ways. <b>Temporal locality</b>: what you used, you'll use again soon. <b>Spatial locality</b>: what's next to it, you'll use soon too, so memory moves in 64-byte <b>lines</b>, not single numbers. Pick an access pattern and a cache size.</p>"),
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Access pattern", options: PATS.map(([v, l]) => ({ value: v, label: l })), value: ca.pat, onChange: (v) => { ca.pat = v; drawCache(); } }),
          slider({ label: "Cache size", min: 2, max: 7, step: 1, value: 4, fmt: (v) => `${2 ** v} lines (${(2 ** v * 64) / 1024} KB)`, onInput: (v) => { ca.lines = 2 ** v; drawCache(); } })),
        h("div.stage-body", h("div.wide-canvas", ccv.el), cst2.el)),
      predict({ chapter: id, question: "A 4,000 × 4,000 array of floats is stored row by row (as in C and NumPy). You sum it twice: once row by row, once column by column. Same number of additions. What happens?",
        options: ["Same speed: same work", "Row by row is several times faster", "Column by column is faster", "Depends only on the clock speed"], answer: 1,
        explain: "<p>Row by row walks memory in order: each 64-byte line brings 16 floats, and the prefetcher streams the next lines ahead. Column by column jumps 16 KB each step, a new line every time, and the row is long enough that lines are pushed out before they're reused. Typically 3–10× slower for the same arithmetic. GPUs care even more: <a href='/learn?c=gpu-from-scratch'>coalescing</a> is the same idea for 32 threads at once.</p>" }),
    ),
    section("SIMD: one instruction, many numbers",
      prose("<p>Decoding an instruction, tracking it and retiring it costs far more energy than the addition itself. So CPUs added <b>vector</b> (SIMD: single instruction, multiple data) units: one instruction that adds 4, 8 or 16 numbers side by side. This is the CPU's step towards the GPU's idea: pay for control once, do lots of arithmetic. Pick a register width and a number format.</p>"),
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Register width", options: [128, 256, 512].map((b) => ({ value: b, label: `${b}-bit` })), value: sm.bits, onChange: (v) => { sm.bits = v; drawSimd(); } }),
          segmented({ label: "Number format", options: [[32, "fp32"], [16, "bf16"], [8, "int8"]].map(([v, l]) => ({ value: v, label: l })), value: sm.dt, onChange: (v) => { sm.dt = v; drawSimd(); } })),
        h("div.stage-body", h("div.wide-canvas", scv.el), sst.el)),
      callout("note", "SIMD on a CPU, SIMT on a GPU", "<p>A GPU warp is the same trick with 32 lanes, but the hardware lets each lane look like its own thread with its own registers (<b>SIMT</b>, single instruction, multiple threads), so you write ordinary scalar code and the GPU groups it. On a CPU you (or the compiler, or NumPy) must arrange data into vectors yourself. That's a big part of <a href='/learn?c=languages'>why NumPy beats a Python loop</a>: its inner loops use these instructions.</p>"),
    ),
    section("Inside a modern core",
      prose("<p>Where the transistors of a 2020s laptop or server core actually go. Pick a part.</p>"),
      h("div.stage",
        h("div.stage-controls", segmented({ label: "Show", options: CORE.map(([v, l]) => ({ value: v, label: l })), value: kc.part, onChange: (v) => { kc.part = v; kcv.redraw(); } })),
        h("div.stage-body", h("div.wide-canvas", kcv.el))),
      table(["", "CPU core", "GPU SM"], [
        ["Built to", "finish one thread fast", "finish many threads per second"],
        ["Hides memory latency with", "big caches, prediction, out-of-order", "dozens of warps to switch between"],
        ["Instructions in flight", "300–600 from one thread", "one per warp, from 64 warps"],
        ["Arithmetic per clock", "~32–64 fp32 ops (two 512-bit vector units)", "~256 fp32 ops, ~2,000+ in tensor cores"],
        ["Good at", "branchy code, pointers, the operating system", "the same operation on huge arrays"],
      ]),
    ),
    section("How the CPU got here",
      h("div.wide-canvas", hcv.el),
      stepper(MILESTONES.map(([yr, name, idea, txt]) => ({ title: `${yr}: ${name}`, html: `<p><b>${idea[0].toUpperCase() + idea.slice(1)}.</b> ${txt}</p>` })), { onStep: (k) => { hstep = k; hcv.redraw?.(); } }),
    ),
    section("Dig down: why can't one thread go 100× faster?",
      whyChain({
        claim: "One thread on a CPU is only a few times faster than in 2005, while chips have 100× more transistors.",
        steps: [
          { q: "Why not raise the clock?", a: "<p>Power rises with voltage squared times frequency, and a faster clock needs more voltage. Clocks stalled near 4–6 GHz in 2005: see <a href='/learn?c=power-wall'>the power wall</a>.</p>" },
          { q: "Why not finish more instructions per clock?", a: "<p>Real code has only a few independent instructions to find at any moment: one in five is a branch, many need the result just computed, and loads that miss the cache stall the chain. Cores already run 4–8 wide; going wider finds little more.</p>" },
          { q: "Why not look further ahead?", a: "<p>Tracking more instructions in flight costs area and energy that grow faster than the gain (Pollack's rule: ~√area), and every wrong guess throws away more work. Speculation also leaks secrets (Spectre), so it now carries a security cost too.</p>" },
          { q: "Why can't hardware break the dependencies?", a: "<p>Because a result can't be used before it exists. Guessing values (value prediction) is rarely right. A program that is a chain of steps, each needing the last, takes as long as its chain, however many units you add.</p>" },
        ],
        axiom: { name: "Causality: an effect can't come before its cause", html: "<p>If step B needs the result of step A, B waits for A. That's not engineering, it's the order of events. Amdahl's law follows: if a fraction s of the work is a chain, no number of processors makes it more than 1/s times faster. So the extra transistors went to more threads: a few on a CPU, tens of thousands on a GPU, for work like AI that is not a chain.</p>" },
      }),
      predict({ chapter: id, question: "90% of a program can run in parallel; 10% is a chain of dependent steps. With 1,000 cores, what's the best speed-up?", options: ["About 900×", "About 100×", "Just under 10×", "1,000×"], answer: 2, explain: "<p>Amdahl's law: 1 / (0.1 + 0.9/1000) ≈ 9.9×. The serial 10% takes the same time however many cores you add. This is why GPUs only pay off for work that is almost entirely parallel, like the multiply-adds of a neural network.</p>" }),
    ),
    section("In code",
      code({
        "Dependency chains (C)": `// Same work, different dependency chains. Compile: gcc -O2 ilp.c
// (-O2 keeps the floating-point adds in the order written)
double sum1(const double* x, long n) {        // one chain: each add waits ~4 clocks
    double s = 0;
    for (long i = 0; i < n; i++) s += x[i];
    return s;
}
double sum4(const double* x, long n) {        // four chains: up to 4 adds in flight
    double a = 0, b = 0, c = 0, d = 0;
    for (long i = 0; i < n; i += 4) { a += x[i]; b += x[i+1]; c += x[i+2]; d += x[i+3]; }
    return (a + b) + (c + d);
}
// typically sum4 runs 2-4x faster on a modern core: same instructions, shorter chain`,
        "Branch prediction (C)": `#include <stdlib.h>
// The famous test: the same loop on random bytes, then on the same bytes sorted.
long count_big(const unsigned char* x, long n) {
    long s = 0;
    for (long i = 0; i < n; i++)
        if (x[i] >= 128) s += x[i];      // random data: a coin flip each time
    return s;
}
// qsort the array first and time again. With gcc -O1 the sorted run is several times
// faster; at -O3 the compiler may use a branch-free conditional move and erase the gap.`,
        "Measure it": `# Linux: count mispredicted branches and instructions per clock
perf stat -e cycles,instructions,branches,branch-misses ./a.out
# instructions / cycles = IPC; a modern core manages ~1-4 on typical code`,
        "A 2-bit predictor (Python)": `def two_bit(outcomes, state=2):
    hits = 0
    for taken in outcomes:
        guess = state >= 2              # 2, 3 = predict taken; 0, 1 = not taken
        hits += guess == taken
        state = min(3, state + 1) if taken else max(0, state - 1)
    return hits / len(outcomes)

loop = ([True] * 7 + [False]) * 6       # inner loop of 8
print(two_bit(loop))                    # 0.875: only the exit is missed
import random; print(two_bit([random.random() < .5 for _ in range(10000)]))   # ~0.5`,
      }),
    ),
  );
  drawC(); drawP(); drawB(); drawO(); drawCache(); drawSimd();
}
