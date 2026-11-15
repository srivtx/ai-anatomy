import { h } from "../lib/core.js";
import { stats, predict, code, callout, section, prose, table, segmented, stepper, toggle } from "../lib/widgets.js";
import { flowCanvas } from "../lib/flow.js";
import { goal, whyChain, stackMap, sticker } from "../lib/why.js";
import { walk } from "../lib/walks.js";

function canvas(width, height, label) {
  const fc = flowCanvas({ width, height, label });
  fc.paint = (fn) => { fc.redraw = fn; fn(); };
  return fc;
}

// ---------- one function, assembled by hand: int sum(int n), the total of 1..n (x86-64, n arrives in edi) ----------
export const ASM = [
  { asm: "xor  eax, eax", bytes: [0x31, 0xc0], what: "s = 0", how: "31 = XOR (register with register). C0 = 11 000 000: both operands are eax (register 0). Anything XOR itself is 0." },
  { asm: "mov  ecx, 1", bytes: [0xb9, 1, 0, 0, 0], what: "i = 1", how: "B8 + 1 = MOV a constant into register 1 (ecx). Then the constant 1 as four bytes, least significant first." },
  { asm: "cmp  ecx, edi", bytes: [0x39, 0xf9], what: "compare i with n", label: ".loop", how: "39 = CMP. F9 = 11 111 001: register 7 (edi, holding n) against register 1 (ecx). Sets flags, stores nothing." },
  { asm: "jg   .done", bytes: [0x7f, 0x06], what: "if i > n, leave", how: "7F = jump if greater. 06 = skip forward 6 bytes, to the ret." },
  { asm: "add  eax, ecx", bytes: [0x01, 0xc8], what: "s = s + i", how: "01 = ADD. C8 = 11 001 000: add register 1 (ecx) into register 0 (eax)." },
  { asm: "inc  ecx", bytes: [0xff, 0xc1], what: "i = i + 1", how: "FF with reg field 000 = INC. C1 = 11 000 001: register 1 (ecx)." },
  { asm: "jmp  .loop", bytes: [0xeb, 0xf6], what: "go round again", how: "EB = short jump. F6 = −10 in two's complement: back 10 bytes, to the cmp." },
  { asm: "ret", bytes: [0xc3], what: "return s (in eax)", label: ".done", how: "C3 = return. By convention the result is whatever is in eax." },
];
const hex = (b) => b.toString(16).toUpperCase().padStart(2, "0");
const bin = (b) => b.toString(2).padStart(8, "0");

const LEVELS = [
  { k: "bits", name: "Voltages on wires", who: "the transistors", txt: ["00110001 11000000 10111001 00000001", "00000000 00000000 00000000 00111001", "11111001 01111111 00000110 00000001", "11001000 11111111 11000001 11101011", "11110110 11000011"], note: "what the hardware actually holds: 144 high or low voltages" },
  { k: "hex", name: "Machine code", who: "decoded by the CPU", txt: ["31 C0", "B9 01 00 00 00", "39 F9", "7F 06", "01 C8", "FF C1", "EB F6", "C3"], note: "the same bits written 4 at a time: 18 bytes" },
  { k: "asm", name: "Assembly", who: "assembler → machine code", txt: ASM.map((a) => (a.label ? `${a.label}:` : "      ") + "  " + a.asm), note: "one line per instruction, with names instead of numbers" },
  { k: "c", name: "C (1972)", who: "compiler → assembly", txt: ["int sum(int n) {", "    int s = 0;", "    for (int i = 1; i <= n; i++)", "        s += i;", "    return s;", "}"], note: "variables and loops; the compiler picks registers and jumps" },
  { k: "py", name: "Python (1991)", who: "interpreter runs bytecode", txt: ["def total(n):", "    return sum(range(1, n + 1))"], note: "no types, no registers, no memory to manage" },
  { k: "torch", name: "PyTorch (2016)", who: "library calls into C++ and CUDA", txt: ["import torch", "torch.arange(1, n + 1, device='cuda').sum()"], note: "say what, not how: one call runs on 10,000 GPU threads" },
];

const HIST = [
  [1843, 0.5, "Ada Lovelace", "Note G: a program before computers", "Her notes on Babbage's Analytical Engine include a step-by-step method to compute Bernoulli numbers, and the idea that the machine could act on symbols, not just numbers."],
  [1945, 0, "ENIAC", "programmed with cables", "Programs were set by plugging cables and turning switches. Changing the program could take days."],
  [1949, 1, "EDSAC", "instructions as letters", "Cambridge's EDSAC loaded programs from paper tape with “initial orders” that turned letter codes into instructions: the first assembler."],
  [1952, 1.6, "A-0", "the first compiler", "Grace Hopper's A-0 stitched routines together from a library. She argued, against resistance, that computers could write programs."],
  [1957, 2.5, "FORTRAN", "formulas, not instructions", "John Backus's team at IBM delivered a compiler whose output ran nearly as fast as hand-written code. Scientists started writing X = A + B * C."],
  [1958, 3.2, "LISP", "programs as data", "John McCarthy's language for AI research: recursion, garbage collection, code that writes code."],
  [1972, 2, "C", "portable, close to the metal", "Dennis Ritchie's language for rewriting Unix. Close enough to the hardware for operating systems, portable across machines. Still underneath almost everything."],
  [1985, 2.5, "C++", "C with abstractions", "Bjarne Stroustrup added classes and templates to C without giving up its speed. CUDA, PyTorch's core and most game engines are C++."],
  [1991, 4, "Python", "readable above all", "Guido van Rossum's language: slow to run, quick to write. It became the glue of science because its slow parts can call fast C."],
  [2006, 4.4, "NumPy", "whole arrays at once", "One line operates on a million numbers, in C underneath. The model for every AI framework since."],
  [2007, 2, "CUDA", "C for 10,000 threads", "NVIDIA's extension of C++ to the GPU. A step back down in abstraction, to reach a new kind of machine."],
  [2016, 4.8, "PyTorch", "the model is the program", "Write the model as ordinary Python; the library runs it on GPUs and works out the gradients."],
  [2021, 3.4, "Triton", "GPU kernels in Python", "OpenAI's Triton lets you write fast GPU kernels in a few lines of Python; torch.compile generates it for you."],
];

/** A tiny compiler: arithmetic expressions → a parse tree → stack-machine code. */
function parse(src) {
  const toks = src.match(/\d+|[a-z]+|[-+*()]/g) || []; let i = 0;
  const peek = () => toks[i], eat = () => toks[i++];
  const prim = () => { const t = eat(); if (t === "(") { const e = expr(); eat(); return e; } return /\d/.test(t) ? { k: "num", v: +t } : { k: "var", v: t }; };
  const term = () => { let a = prim(); while (peek() === "*") { eat(); a = { k: "op", v: "*", l: a, r: prim() }; } return a; };
  const expr = () => { let a = term(); while (peek() === "+" || peek() === "-") { const o = eat(); a = { k: "op", v: o, l: a, r: term() }; } return a; };
  return expr();
}
const same = (a, b) => a.k === b.k && a.v === b.v && (a.k !== "op" || (same(a.l, b.l) && same(a.r, b.r)));
function fold(n) {
  if (n.k !== "op") return n;
  const l = fold(n.l), r = fold(n.r), num = (x, v) => x.k === "num" && x.v === v;
  if (l.k === "num" && r.k === "num") return { k: "num", v: n.v === "+" ? l.v + r.v : n.v === "-" ? l.v - r.v : l.v * r.v, opt: true };
  if (n.v === "*" && (num(l, 0) || num(r, 0))) return { k: "num", v: 0, opt: true };
  if (n.v === "*" && num(r, 1)) return { ...l, opt: true };
  if (n.v === "*" && num(l, 1)) return { ...r, opt: true };
  if ((n.v === "+" || n.v === "-") && num(r, 0)) return { ...l, opt: true };
  if (n.v === "+" && num(l, 0)) return { ...r, opt: true };
  return { ...n, l, r };
}
function gen(n, out, opt) {
  if (n.k === "num") out.push(`PUSH ${n.v}`);
  else if (n.k === "var") out.push(`LOAD ${n.v}`);
  else if (opt && same(n.l, n.r)) { gen(n.l, out, opt); out.push("DUP"); out.push({ "+": "ADD", "-": "SUB", "*": "MUL" }[n.v]); }
  else { gen(n.l, out, opt); gen(n.r, out, opt); out.push({ "+": "ADD", "-": "SUB", "*": "MUL" }[n.v]); }
  return out;
}
const ENV = { a: 2, b: 3, c: 4, d: 1, x: 5 };
function runStack(prog) {
  const st = [];
  for (const ins of prog) {
    const [op, arg] = ins.split(" ");
    if (op === "PUSH") st.push(+arg); else if (op === "LOAD") st.push(ENV[arg]); else if (op === "DUP") st.push(st[st.length - 1]);
    else { const b = st.pop(), a = st.pop(); st.push(op === "ADD" ? a + b : op === "SUB" ? a - b : a * b); }
  }
  return st.pop();
}
const PRESETS = ["a + b * c", "(a + b) * (c - d)", "(x + 1) * (x + 1)", "2 * 3 + x * 1 + 0"];

export default function (root, { id }) {
  // ---------- 1. wires, then numbers ----------
  const wn = { mode: "wires" };
  const wc = canvas(880, 250, "Programming ENIAC by cables, against a stored program held as numbers in memory");
  wc.paint(() => {
    wc.clear(); const c = wc.colors, W = wn.mode === "wires";
    if (W) {
      const U = [["accumulator 1", 40, 40], ["accumulator 2", 40, 150], ["multiplier", 250, 40], ["function table", 250, 150], ["divider", 460, 40], ["printer", 460, 150], ["master programmer", 670, 95]];
      const cab = [[0, 2], [1, 2], [2, 4], [3, 1], [4, 5], [6, 0], [6, 3], [0, 5], [3, 4]];
      cab.forEach(([a, b], k) => { const A = U[a], B = U[b]; wc.ctx.beginPath(); wc.ctx.moveTo(A[1] + 80, A[2] + 30); wc.ctx.bezierCurveTo(A[1] + 140, A[2] + 30 + (k % 2 ? 60 : -40), B[1] + 20, B[2] - (k % 2 ? 30 : -50), B[1] + 80, B[2] + 30); wc.ctx.strokeStyle = [c.c2, c.c1, c.c3, c.c4, c.c6][k % 5]; wc.ctx.lineWidth = 2.5; wc.ctx.stroke(); });
      U.forEach(([n, x, y]) => { wc.box(x, y, 160, 60, { fill: c.sunken, stroke: c.line, r: 6 }); for (let k = 0; k < 10; k++) wc.dot(x + 14 + k * 14, y + 46, 3, c["ink-3"]); wc.text(n, x + 80, y + 20, { size: 11.5, align: "center" }); });
      wc.text("1945: the program is the wiring. A new problem means replugging hundreds of cables: days of work.", 440, 236, { size: 12, align: "center" });
    } else {
      wc.box(40, 30, 260, 180, { fill: c.surface, stroke: c.line, r: 8 }); wc.text("CPU", 170, 50, { size: 12, align: "center" });
      wc.box(70, 70, 200, 40, { fill: `color-mix(in srgb, ${c.c6} 20%, ${c.surface})`, stroke: "transparent", r: 6 }); wc.text("fetch · decode · execute", 170, 90, { size: 11, align: "center" });
      wc.box(70, 130, 200, 50, { fill: c.sunken, stroke: "transparent", r: 6 }); wc.text("the same fixed wiring for every program", 170, 155, { size: 10.5, align: "center", color: c["ink-2"] });
      wc.line(300, 120, 380, 120, { color: c.accent, lw: 2.5 }); wc.text("reads", 340, 108, { size: 10, align: "center", color: c["ink-3"] });
      wc.box(380, 20, 470, 200, { fill: c.surface, stroke: c.line, r: 8 }); wc.text("memory: one row of numbers", 392, 34, { size: 10.5, color: c["ink-3"], mono: true });
      const cells = [...ASM.flatMap((a) => a.bytes)];
      cells.forEach((b, k) => { const x = 392 + (k % 9) * 50, y = 50 + Math.floor(k / 9) * 44; wc.box(x, y, 44, 34, { fill: `color-mix(in srgb, ${c.c1} 22%, ${c.surface})`, stroke: "transparent", r: 4 }); wc.text(hex(b), x + 22, y + 17, { size: 12, align: "center", mono: true }); });
      for (let k = 0; k < 9; k++) { const x = 392 + k * 50, y = 50 + 2 * 44; wc.box(x, y, 44, 34, { fill: `color-mix(in srgb, ${c.c4} 22%, ${c.surface})`, stroke: "transparent", r: 4 }); wc.text(String([5, 3, 8, 2, 0, 0, 0, 0, 0][k]), x + 22, y + 17, { size: 12, align: "center", mono: true }); }
      wc.text("instructions (blue) and data (amber) side by side", 615, 208, { size: 10.5, align: "center", color: c["ink-3"] });
      wc.text("1948 on: the program is numbers in memory. A new problem means loading new numbers: seconds.", 440, 236, { size: 12, align: "center" });
    }
  });

  // ---------- 2. assemble it yourself ----------
  const as = { line: 0 };
  const acv = canvas(880, 330, "Eight lines of x86 assembly, the bytes each becomes, and one byte's bits");
  acv.paint(() => {
    acv.clear(); const c = acv.colors, L = as.line, a = ASM[L];
    let off = 0; const offs = ASM.map((x) => { const o = off; off += x.bytes.length; return o; });
    ASM.forEach((x, k) => {
      const y = 36 + k * 28, cur = k === L;
      if (cur) acv.box(10, y - 12, 860, 26, { fill: `color-mix(in srgb, ${c.accent} 12%, ${c.surface})`, stroke: c.accent, lw: 1.5, r: 6 });
      acv.text(x.label ? `${x.label}:` : "", 20, y, { size: 11, mono: true, color: c["ink-3"] });
      acv.text(x.asm, 80, y, { size: 12.5, mono: true, color: cur ? c.ink : c["ink-2"] });
      acv.text(`; ${x.what}`, 220, y, { size: 11, color: c["ink-3"] });
      acv.text(String(offs[k]).padStart(2, "0"), 420, y, { size: 10, mono: true, color: c["ink-3"] });
      x.bytes.forEach((b, j) => { const bx = 448 + j * 40; acv.box(bx, y - 10, 34, 22, { fill: `color-mix(in srgb, ${j === 0 ? c.c2 : c.c1} ${cur ? 45 : 18}%, ${c.surface})`, stroke: "transparent", r: 4 }); acv.text(hex(b), bx + 17, y + 1, { size: 12, align: "center", mono: true }); });
    });
    acv.text("offset", 412, 14, { size: 9, color: c["ink-3"], mono: true }); acv.text("bytes (first byte = what to do)", 448, 14, { size: 9, color: c["ink-3"], mono: true });
    // the bits of the selected line
    const y0 = 270; acv.text(a.how, 440, y0 - 14, { size: 11.5, align: "center" });
    a.bytes.slice(0, 2).forEach((b, j) => {
      const bits = bin(b), x0 = 220 + j * 240;
      bits.split("").forEach((bit, k) => { const x = x0 + k * 26; acv.box(x, y0, 22, 30, { fill: bit === "1" ? c.accent : c.sunken, stroke: c.line, lw: 1, r: 3 }); acv.text(bit, x + 11, y0 + 15, { size: 11, align: "center", mono: true, color: bit === "1" ? c.surface : c["ink-3"] }); });
      acv.text(`${hex(b)} =`, x0 - 8, y0 + 15, { size: 11, align: "right", mono: true });
    });
    acv.text("each 1 is a wire held at the supply voltage, each 0 a wire held at ground", 440, y0 + 46, { size: 10.5, align: "center", color: c["ink-3"] });
  });

  // ---------- 3. one program, every level ----------
  const lv = { k: "asm" };
  const lc = canvas(880, 290, "The same function at six levels, from voltages to PyTorch, and what translates each level down");
  lc.paint(() => {
    lc.clear(); const c = lc.colors, cur = LEVELS.findIndex((l) => l.k === lv.k);
    LEVELS.forEach((l, k) => {
      const y = 250 - k * 42, on = k === cur, w = 130 + k * 12;
      lc.box(20, y - 16, w, 32, { fill: on ? `color-mix(in srgb, ${c.accent} 22%, ${c.surface})` : `color-mix(in srgb, ${c.c1} ${8 + k * 5}%, ${c.surface})`, stroke: on ? c.accent : "transparent", lw: 2, r: 6 });
      lc.text(l.name, 32, y, { size: 11.5, color: on ? c.ink : c["ink-2"] });
      if (k) { lc.text("↓", 20 + w + 12, y + 20, { size: 13, color: c["ink-3"] }); lc.text(l.who, 20 + w + 26, y + 20, { size: 9.5, color: c["ink-3"], mono: true }); }
    });
    const l = LEVELS[cur], x = 470, bh = 22 + l.txt.length * 22;
    lc.box(x, 20, 396, Math.max(bh, 120), { fill: c.sunken, stroke: c.line, r: 8 });
    l.txt.forEach((t, k) => lc.text(t, x + 16, 42 + k * 22, { size: l.k === "bits" ? 12.5 : 13, mono: true }));
    lc.text(l.note, x + 198, Math.max(bh, 120) + 40, { size: 11.5, align: "center", color: c["ink-2"] });
  });

  // ---------- 4. a tiny compiler ----------
  const cp = { src: PRESETS[1], opt: false };
  const ccv = canvas(880, 300, "An expression parsed into a tree, and the stack-machine code generated from it");
  const cst = stats([{ label: "Instructions" }, { label: "Result (a=2, b=3, c=4, d=1, x=5)" }, { label: "Without optimising" }]);
  function drawCp() {
    const raw = parse(cp.src), tree = cp.opt ? fold(raw) : raw, prog = gen(tree, [], cp.opt), base = gen(raw, [], false);
    cst.set(0, String(prog.length), prog.length < base.length ? "good" : ""); cst.set(1, String(runStack(prog))); cst.set(2, `${base.length} instructions, result ${runStack(base)}`);
    ccv.paint(() => {
      ccv.clear(); const c = ccv.colors;
      // lay the tree out: leaves left to right, depth down
      let leaf = 0; const pos = new Map(), depth = (n) => (n.k === "op" ? 1 + Math.max(depth(n.l), depth(n.r)) : 1), D = depth(tree);
      const place = (n, d) => { if (n.k === "op") { place(n.l, d + 1); place(n.r, d + 1); pos.set(n, [(pos.get(n.l)[0] + pos.get(n.r)[0]) / 2, d]); } else pos.set(n, [leaf++, d]); };
      place(tree, 0);
      const X = (i) => 40 + (leaf > 1 ? (i / (leaf - 1)) * 420 : 210), Y = (d) => 40 + d * Math.min(60, 220 / Math.max(1, D - 1));
      ccv.text(`source:  ${cp.src}`, 20, 14, { size: 12, mono: true });
      for (const [n, [i, d]] of pos) if (n.k === "op") for (const ch of [n.l, n.r]) { const [ci, cd] = pos.get(ch); ccv.line(X(i), Y(d), X(ci), Y(cd), { color: c.line, lw: 1.5 }); }
      for (const [n, [i, d]] of pos) {
        const op = n.k === "op", col = op ? c.c6 : n.k === "num" ? c.c4 : c.c1;
        ccv.ctx.beginPath(); ccv.ctx.arc(X(i), Y(d), 17, 0, 7); ccv.ctx.fillStyle = `color-mix(in srgb, ${col} 30%, ${c.surface})`; ccv.ctx.fill(); ccv.ctx.lineWidth = n.opt ? 2.5 : 1; ccv.ctx.strokeStyle = n.opt ? c.good : c.line; ccv.ctx.stroke();
        ccv.text(op ? (n.v === "*" ? "×" : n.v) : String(n.v), X(i), Y(d), { size: 13, align: "center", mono: true });
      }
      if (cp.opt && same(tree.l || {}, tree.r || {}) && tree.k === "op") ccv.text("both sides identical: compute once, DUP", 250, 286, { size: 10.5, align: "center", color: c.good });
      // the code
      const x = 520; ccv.box(x, 20, 346, 266, { fill: c.sunken, stroke: c.line, r: 8 }); ccv.text("stack-machine code", x + 12, 34, { size: 10, color: c["ink-3"], mono: true });
      const step = Math.min(21, 230 / prog.length);
      prog.forEach((ins, k) => ccv.text(`${String(k + 1).padStart(2)}  ${ins}`, x + 16, 54 + k * step, { size: Math.min(12.5, step * 0.62), mono: true, color: /PUSH|LOAD|DUP/.test(ins) ? c.ink : c.accent }));
    });
  }

  // ---------- 5. the price of abstraction ----------
  const COST = [["Python for-loop", 0.6, c => c.c2], ["Python sum()", 0.06, c => c.c2], ["NumPy .sum()", 0.005, c => c.c4], ["C loop, -O2", 0.004, c => c.c1], ["C, vectorised", 0.0015, c => c.c1], ["GPU, data already there", 0.00003, c => c.c3]];
  const pc = canvas(880, 250, "Time to add up ten million numbers at different levels of abstraction, on a log scale");
  pc.paint(() => {
    pc.clear(); const c = pc.colors, x0 = 220, x1 = 850, X = (t) => x0 + ((Math.log10(t) + 5) / 5.3) * (x1 - x0);
    for (let e = -5; e <= 0; e++) { pc.line(X(10 ** e), 14, X(10 ** e), 226, { color: c.line, lw: 1 }); pc.text(["10 µs", "100 µs", "1 ms", "10 ms", "100 ms", "1 s"][e + 5], X(10 ** e), 238, { size: 9.5, align: "center", color: c["ink-3"], mono: true }); }
    COST.forEach(([n, t, col], k) => {
      const y = 22 + k * 34; pc.text(n, x0 - 10, y + 10, { size: 11.5, align: "right" });
      pc.box(x0, y, X(t) - x0, 20, { fill: `color-mix(in srgb, ${col(c)} 60%, ${c.surface})`, stroke: "transparent", r: 4 });
      pc.text(t >= 0.01 ? `~${Math.round(t * 1000)} ms` : t >= 0.001 ? `~${(t * 1000).toFixed(1)} ms` : `~${Math.round(t * 1e6)} µs`, X(t) + 8, y + 10, { size: 10.5, mono: true, color: c["ink-2"] });
    });
  });

  // ---------- 6. history ----------
  let hs = 0;
  const hc = canvas(880, 280, "Programming languages by year and level of abstraction, 1843 to 2021");
  hc.paint(() => {
    hc.clear(); const c = hc.colors, X = (y) => (y < 1940 ? 64 + ((y - 1840) / 100) * 80 : 170 + ((y - 1940) / 85) * 670), Y = (a) => 240 - a * 42;
    ["wiring", "assembly", "", "", "high-level", "", "array / model"].forEach((t, a) => { if (t) { hc.line(56, Y(a), 860, Y(a), { color: c.line, lw: 1, dash: [2, 4] }); hc.text(t, 52, Y(a), { size: 9.5, align: "right", color: c["ink-3"] }); } });
    hc.text("⋯", 152, 262, { size: 11, align: "center", color: c["ink-3"] });
    for (const y of [1840, 1940, 1960, 1980, 2000, 2020]) hc.text(String(y), X(y), 262, { size: 9.5, align: "center", color: c["ink-3"], mono: true });
    HIST.forEach(([y, a, n], k) => {
      const cur = k === hs; hc.dot(X(y), Y(a), cur ? 8 : 5, cur ? c.accent : k < hs ? c.c1 : c["ink-3"]);
      hc.text(n, X(y) + (k % 2 ? 10 : -10), Y(a) - 12, { size: cur ? 12 : 10, align: k % 2 ? "left" : "right", color: cur ? c.accent : c["ink-2"], weight: cur ? 700 : 600 });
    });
    hc.text("↑ further from the wires", 860, 14, { size: 10, color: c["ink-3"], mono: true, align: "right" });
  });

  root.append(
    section(null,
      goal("<p>Hardware only understands patterns of high and low voltages. People think in formulas and steps. A <b>programming language</b> is the bridge: something a person can write that a program can translate, exactly, into those patterns. Each new language moved the bridge one step further from the wires, and each step traded some speed for a lot of human time. AI code today sits at the top of a tower built over 80 years.</p>",
        ["Something people can write and read", "Translates exactly into machine code", "Fast enough that the translation is worth it"]),
      stackMap(id),
      h("div.vx-row", sticker("punchcard", { title: "A punched card", caption: "how programs were fed to machines into the 1970s: a hole is a 1", px: 4 }), sticker("bits", { title: "Machine code", caption: "in the end, every program is bits on wires", px: 4 }), sticker("softstack", { title: "The tower", caption: "each language stands on the ones below it", px: 4 })),
    ),
    section("Before languages: wires, then numbers",
      prose("<p>The first electronic computers had no programs to write. You built the computation out of cables. The breakthrough of 1945–48 was to store the program as <b>numbers in memory</b>, in the same memory as the data, read by fixed hardware. Programming became writing numbers, and that made it possible to write programs that write programs.</p>"),
      h("div.stage",
        h("div.stage-controls", segmented({ label: "Era", options: [{ value: "wires", label: "1945: cables" }, { value: "numbers", label: "1948: stored program" }], value: wn.mode, onChange: (v) => { wn.mode = v; wc.redraw(); } })),
        h("div.stage-body", h("div.wide-canvas", wc.el))),
      callout("note", "The first run", "<p>On 21 June 1948 the Manchester Baby ran the first program held in electronic memory: 17 instructions that found the largest factor of 2¹⁸. ENIAC itself was converted to run stored programs the same year.</p>"),
    ),
    section("Assemble a function by hand",
      prose("<p>Here is a real x86-64 function, the one your laptop would run, adding up 1 + 2 + … + n. Each line of <b>assembly</b> is one instruction, and an <b>assembler</b> turns it into bytes by table lookup: the first byte says what to do, the next says which registers. Pick a line to see its encoding, down to the bits.</p>"),
      h("div.stage",
        h("div.stage-controls", segmented({ label: "Line", options: ASM.map((a, k) => ({ value: String(k), label: a.asm.split(/\s+/)[0] })), value: "0", onChange: (v) => { as.line = +v; acv.redraw(); } })),
        h("div.stage-body", h("div.wide-canvas", acv.el))),
      prose("<p>Step through the whole program, byte by byte, down to the 144 wires:</p>"),
      walk("languages"),
    ),
    section("One program, six levels",
      prose("<p>The same function written at every level the industry has climbed. Each level is shorter and closer to how you think, and each needs a translator to get back down: an assembler, a compiler, an interpreter, a library. Pick a level.</p>"),
      h("div.stage",
        h("div.stage-controls", segmented({ label: "Level", options: LEVELS.map((l) => ({ value: l.k, label: l.name.split(" (")[0] })), value: lv.k, onChange: (v) => { lv.k = v; lc.redraw(); } })),
        h("div.stage-body", h("div.wide-canvas", lc.el))),
    ),
    section("What a compiler does",
      prose("<p>A compiler reads text, builds a <b>tree</b> of what it means, then walks the tree to emit instructions. Here's a working one in 40 lines, for arithmetic, targeting a <b>stack machine</b> (push values, then each operation takes the top two). Turn on the optimiser: it folds constants, drops ×1 and +0, and computes repeated work once. Real compilers run hundreds of passes like these.</p>"),
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Expression", options: PRESETS.map((p) => ({ value: p, label: p })), value: cp.src, onChange: (v) => { cp.src = v; drawCp(); } }),
          toggle({ label: "Optimise", checked: false, onChange: (v) => { cp.opt = v; drawCp(); } })),
        h("div.stage-body", cst.el, h("div.wide-canvas", ccv.el))),
      callout("tip", "Why FORTRAN was a revolution", "<p>In 1957 most programmers believed no compiler could match hand-written assembly. Backus's team spent most of their effort on the optimiser, and FORTRAN's output came close. Once compiled code was nearly as fast, nobody went back. The same argument is happening now with GPU kernels: <a href='/learn?c=software-stack'>torch.compile and Triton</a> against hand-written CUDA.</p>"),
    ),
    section("The price of abstraction",
      prose("<p>Each level up costs speed, unless the level is a <b>thin layer over fast code</b>. Adding up ten million numbers: a Python loop runs every addition through the interpreter (~60 ns each); NumPy makes one call into a tight C loop; the GPU spreads it over thousands of threads. Rough times on a laptop and an H100; measure your own with the code below.</p>"),
      h("div.stage", h("div.stage-body", h("div.wide-canvas", pc.el))),
      predict({ chapter: id, question: "Why is NumPy's sum about 100× faster than a Python for-loop over the same numbers?", options: ["NumPy uses the GPU", "Python's loop handles every number as a full object, checking its type and dispatching each +; NumPy runs one compiled C loop over raw numbers", "NumPy skips some of the numbers", "Python stores numbers in decimal"], answer: 1, explain: "<p>Each Python <code>+</code> looks up the types of both operands, finds the right add function, allocates a new number object and counts references: dozens of instructions for one addition. NumPy keeps the numbers as a raw array of doubles and runs one compiled loop the C compiler has vectorised. Same answer, the overhead paid once instead of ten million times. This is the trick behind all of AI: write Python, run C++ and CUDA.</p>" }),
    ),
    section("80 years of climbing",
      h("div.wide-canvas", hc.el),
      stepper(HIST.map(([y, , n, idea, txt]) => ({ title: `${y}: ${n}`, html: `<p><b>${idea[0].toUpperCase() + idea.slice(1)}.</b> ${txt}</p>` })), { onStep: (k) => { hs = k; hc.redraw?.(); } }),
      callout("note", "The pattern", "<p>Each new kind of machine starts low, close to the wires, because nobody knows yet what to hide. Then languages climb. CUDA in 2007 was a step back down to reach the GPU; Triton and torch.compile are the climb back up.</p>"),
    ),
    section("Dig down: why does every language end up as the same instructions?",
      whyChain({
        claim: "Python, C, CUDA and Lisp all end up as sequences from the same few hundred machine instructions, and any of them could compute what the others can.",
        steps: [
          { q: "Why translate at all?", a: "<p>The hardware is fixed wiring: a decoder that recognises bit patterns and routes them to adders and memory. Only patterns it was built for mean anything to it. Everything else must be turned into those patterns first.</p>" },
          { q: "Why not build hardware that runs Python directly?", a: "<p>It's been tried: Lisp machines (1970s–80s), Java processors (1990s). General-purpose hardware plus a compiler kept winning, because simple hardware runs faster and every improvement in the compiler helps every program. Special hardware pays off only for one huge workload: graphics, and now matrix multiplies.</p>" },
          { q: "Why can a small instruction set express every program?", a: "<p>Because very little is needed: read and write memory, do arithmetic, and jump depending on a value. With those, you can build loops, functions, data structures and interpreters for any other language. A language with these powers is called <b>Turing-complete</b>.</p>" },
          { q: "Why does that small set suffice?", a: "<p>In 1936 Alan Turing described a machine with only a tape, a head that reads and writes one symbol, and a table of rules, and showed that one such machine can simulate any other given its description. Alonzo Church reached the same class of functions with a completely different formalism.</p>" },
        ],
        axiom: { name: "Universality (the Church–Turing thesis)", html: "<p>Every reasonable model of step-by-step computation computes exactly the same set of functions. So languages differ in convenience and speed, never in what they can compute. That's why a compiler can always exist from one to another, and why the whole tower can stand on one kind of hardware. (It's a thesis, not a theorem: it can't be proved, but no counterexample has ever been found.)</p>" },
      }),
    ),
    section("In code",
      code({
        "Dump the machine code": `# compile the C function, then look at what the assembler produced
cat > sum.c << 'EOF'
int sum(int n) { int s = 0; for (int i = 1; i <= n; i++) s += i; return s; }
EOF
gcc -O1 -c sum.c && objdump -d -M intel sum.o
# at -O2 the compiler may replace the loop with arithmetic or vectorise it: try both`,
        "Python's own machine code": `import dis
def total(n):
    s = 0
    for i in range(1, n + 1):
        s += i
    return s
dis.dis(total)        # the bytecode CPython's interpreter loop executes, one op at a time`,
        "The price, measured": `import time, numpy as np
x = np.random.rand(10_000_000); xs = x.tolist()

t = time.perf_counter(); s = 0.0
for v in xs: s += v
print("python loop", time.perf_counter() - t)

t = time.perf_counter(); sum(xs); print("sum()      ", time.perf_counter() - t)
t = time.perf_counter(); x.sum();  print("numpy      ", time.perf_counter() - t)`,
        "A tiny compiler": `import re
def parse(s):
    toks = re.findall(r"\\d+|[a-z]+|[-+*()]", s); i = 0
    def prim():
        nonlocal i; t = toks[i]; i += 1
        if t == "(": e = expr(); i += 1; return e
        return ("num", int(t)) if t.isdigit() else ("var", t)
    def term():
        nonlocal i; a = prim()
        while i < len(toks) and toks[i] == "*": i += 1; a = ("*", a, prim())
        return a
    def expr():
        nonlocal i; a = term()
        while i < len(toks) and toks[i] in "+-": o = toks[i]; i += 1; a = (o, a, term())
        return a
    return expr()

def gen(n):
    if n[0] == "num": return [f"PUSH {n[1]}"]
    if n[0] == "var": return [f"LOAD {n[1]}"]
    return gen(n[1]) + gen(n[2]) + [{"+": "ADD", "-": "SUB", "*": "MUL"}[n[0]]]

print(gen(parse("(a + b) * (c - d)")))`,
      }),
    ),
  );
  drawCp();
}
