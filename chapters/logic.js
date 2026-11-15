import { h } from "../lib/core.js";
import { slider, stats, predict, code, callout, section, prose, table, segmented, stepper } from "../lib/widgets.js";
import { flowCanvas } from "../lib/flow.js";
import { goal, whyChain, stackMap } from "../lib/why.js";
import { walk } from "../lib/walks.js";

function canvas(width, height, label) {
  const fc = flowCanvas({ width, height, label });
  fc.paint = (fn) => { fc.redraw = fn; fn(); };
  return fc;
}

// CMOS gates: transistors with their position, and which network (up = pull-up to VDD, down = pull-down to ground)
const GATES = {
  not: { name: "Inverter (NOT)", f: (a) => +!a, ins: ["A"], tr: [{ t: "P", in: "A", x: 300, y: 92, net: "up" }, { t: "N", in: "A", x: 300, y: 238, net: "down" }],
    wires: { up: [[300, 30, 300, 72], [300, 112, 300, 165]], down: [[300, 165, 300, 218], [300, 258, 300, 300]] }, series: { up: false, down: false } },
  nand: { name: "NAND", f: (a, b) => +!(a && b), ins: ["A", "B"], tr: [{ t: "P", in: "A", x: 230, y: 92, net: "up" }, { t: "P", in: "B", x: 370, y: 92, net: "up" }, { t: "N", in: "A", x: 300, y: 208, net: "down" }, { t: "N", in: "B", x: 300, y: 262, net: "down" }],
    wires: { up: [[230, 30, 230, 72], [370, 30, 370, 72], [230, 112, 230, 165], [370, 112, 370, 165], [230, 165, 370, 165]], down: [[300, 165, 300, 188], [300, 228, 300, 242], [300, 282, 300, 300]] }, series: { up: false, down: true } },
  nor: { name: "NOR", f: (a, b) => +!(a || b), ins: ["A", "B"], tr: [{ t: "P", in: "A", x: 300, y: 64, net: "up" }, { t: "P", in: "B", x: 300, y: 122, net: "up" }, { t: "N", in: "A", x: 230, y: 238, net: "down" }, { t: "N", in: "B", x: 370, y: 238, net: "down" }],
    wires: { up: [[300, 30, 300, 44], [300, 84, 300, 102], [300, 142, 300, 165]], down: [[230, 165, 370, 165], [230, 165, 230, 218], [370, 165, 370, 218], [230, 258, 230, 300], [370, 258, 370, 300]] }, series: { up: true, down: false } },
};

export default function (root, { id }) {
  // ---------- 1. CMOS gates ----------
  const g = { kind: "nand", A: 1, B: 0 };
  const gc = canvas(880, 330, "A CMOS logic gate: which transistors conduct, and the output");
  const gst = stats([{ label: "Output" }, { label: "Path that conducts" }, { label: "Current from supply to ground" }]);
  const truth = h("div");
  const inBtn = (n) => h("button.btn", { type: "button", onclick: () => { g[n] ^= 1; drawG(); } });
  const btnA = inBtn("A"), btnB = inBtn("B");
  function drawG() {
    const G = GATES[g.kind], on = (tr) => (tr.t === "P" ? g[tr.in] === 0 : g[tr.in] === 1);
    const net = (n) => { const ts = G.tr.filter((t) => t.net === n); return G.series[n] ? ts.every(on) : ts.some(on); };
    const up = net("up"), down = net("down"), out = G.f(g.A, g.B);
    btnA.textContent = `Input A = ${g.A}`; btnB.textContent = `Input B = ${g.B}`; btnB.hidden = G.ins.length < 2;
    gst.set(0, `${out} (${out ? "high, ~0.8 V" : "low, 0 V"})`, out ? "good" : "");
    gst.set(1, up ? "pull-up: supply → output" : "pull-down: output → ground");
    gst.set(2, up && down ? "short circuit!" : "none: one network is always off", up && down ? "bad" : "good");
    const rows = G.ins.length === 1 ? [[0], [1]] : [[0, 0], [0, 1], [1, 0], [1, 1]];
    truth.replaceChildren(table([...G.ins, "out"], rows.map((r) => [...r.map(String), String(G.f(...r))].map((v, i, arr) => (r[0] === g.A && (r.length < 2 || r[1] === g.B) ? h("mark", v) : v)))));
    gc.paint(() => {
      gc.clear(); const c = gc.colors, live = c.good, dead = c.line;
      gc.line(140, 30, 460, 30, { color: up ? live : c["ink-3"], lw: 4 }); gc.text("supply, VDD (1)", 470, 30, { size: 12, color: c["ink-2"] });
      gc.line(140, 300, 460, 300, { color: down ? live : c["ink-3"], lw: 4 }); gc.text("ground (0)", 470, 300, { size: 12, color: c["ink-2"] });
      for (const n of ["up", "down"]) for (const [x1, y1, x2, y2] of G.wires[n]) {
        // a vertical wire in a parallel branch belongs to that branch's transistor: lit only if it conducts
        const br = !G.series[n] && x1 === x2 && G.tr.filter((t) => t.net === n).length > 1 ? G.tr.find((t) => t.net === n && t.x === x1) : null;
        gc.line(x1, y1, x2, y2, { color: (n === "up" ? up : down) && (!br || on(br)) ? live : dead, lw: 3 });
      }
      for (const tr of G.tr) {
        const o = on(tr);
        gc.box(tr.x - 46, tr.y - 20, 92, 40, { fill: o ? `color-mix(in srgb, ${c.good} 22%, ${c.surface})` : c.sunken, stroke: o ? c.good : c.line, lw: 2, r: 8 });
        gc.text(`${tr.t === "P" ? "p-type" : "n-type"} · ${tr.in}`, tr.x, tr.y - 6, { size: 11.5, align: "center" });
        gc.text(o ? "on" : "off", tr.x, tr.y + 9, { size: 10.5, color: o ? c.good : c["ink-3"], align: "center", mono: true });
        gc.text(tr.in, tr.x - 70, tr.y, { size: 12, color: g[tr.in] ? c.accent : c["ink-3"], align: "center", mono: true });
        gc.line(tr.x - 62, tr.y, tr.x - 46, tr.y, { color: g[tr.in] ? c.accent : c["ink-3"], lw: 2 });
      }
      gc.line(G.tr.length > 2 ? 370 : 300, 165, 600, 165, { color: out ? c.c2 : c["ink-3"], lw: 4 });
      gc.dot(600, 165, 9, out ? c.c2 : c["ink-3"]);
      gc.text(`output = ${out}`, 620, 165, { size: 16 });
      gc.text("p-type transistors: on when their input is 0", 620, 92, { size: 11, color: c["ink-3"] });
      gc.text("n-type transistors: on when their input is 1", 620, 238, { size: 11, color: c["ink-3"] });
    });
  }

  // ---------- 2. an 8-bit adder ----------
  const ad = { A: 87, B: 45 };
  const ac = canvas(880, 220, "An 8-bit ripple-carry adder: bits, carries and the sum");
  const ast = stats([{ label: "Sum" }, { label: "Longest carry chain" }, { label: "Time to settle (2 gate delays per bit, ~10 ps each)" }]);
  function drawA() {
    const s = ad.A + ad.B, carries = [];
    let c = 0, chain = 0, best = 0;
    for (let i = 0; i < 8; i++) { const a = (ad.A >> i) & 1, b = (ad.B >> i) & 1, t = a + b + c; c = t >> 1; carries.push(c); chain = c ? chain + 1 : 0; best = Math.max(best, chain); }
    ast.set(0, `${ad.A} + ${ad.B} = ${s}${s > 255 ? " (9 bits: overflow into the carry out)" : ""}`);
    ast.set(1, `${best} bit${best === 1 ? "" : "s"}`); ast.set(2, `${Math.max(1, best) * 20} ps this time, 160 ps at worst`);
    ac.paint(() => {
      ac.clear(); const col = ac.colors, x = (i) => 760 - i * 92;
      for (let i = 0; i < 8; i++) {
        const a = (ad.A >> i) & 1, b = (ad.B >> i) & 1, sum = (s >> i) & 1, cin = i ? carries[i - 1] : 0;
        ac.text(`A${i}=${a}`, x(i), 22, { size: 12, color: a ? col.accent : col["ink-3"], align: "center", mono: true });
        ac.text(`B${i}=${b}`, x(i), 40, { size: 12, color: b ? col.accent : col["ink-3"], align: "center", mono: true });
        ac.box(x(i) - 34, 56, 68, 70, { fill: col.surface, stroke: col.line, r: 8 });
        ac.text("full", x(i), 80, { size: 11, align: "center", color: col["ink-2"] }); ac.text("adder", x(i), 96, { size: 11, align: "center", color: col["ink-2"] });
        ac.text(`${2 ** i}s`, x(i), 114, { size: 10, align: "center", color: col["ink-3"], mono: true });
        ac.line(x(i), 126, x(i), 160, { color: sum ? col.c2 : col.line, lw: 3 });
        ac.dot(x(i), 168, 10, sum ? col.c2 : col.sunken); ac.text(String(sum), x(i), 168, { size: 12, align: "center", color: sum ? "#fff" : col["ink-3"], mono: true });
        if (i < 7) { ac.line(x(i) - 34, 91, x(i + 1) + 34, 91, { color: carries[i] ? col.c4 : col.line, lw: carries[i] ? 4 : 2 }); }
        void cin;
      }
      ac.text("carries ripple right to left (orange = a 1 is being carried)", 440, 205, { size: 11.5, color: col["ink-3"], align: "center" });
      ac.text(carries[7] ? "carry out = 1" : "carry out = 0", 40, 91, { size: 11, color: carries[7] ? col.c4 : col["ink-3"] });
    });
  }

  // ---------- 3. clock speed from gate delays ----------
  const ck = { depth: 30, delay: 12, stages: 1 };
  const kst = stats([{ label: "Slowest path" }, { label: "Fastest safe clock" }, { label: "Results per second" }, { label: "Latency of one result" }]);
  function drawK() {
    const flop = 25, path = (ck.depth / ck.stages) * ck.delay + flop, f = 1e3 / path;
    kst.set(0, `${(ck.depth / ck.stages).toFixed(0)} gates × ${ck.delay} ps + ${flop} ps register = ${path.toFixed(0)} ps`);
    kst.set(1, `${f.toFixed(2)} GHz`, f > 3 ? "good" : f > 1.5 ? "warn" : "bad");
    kst.set(2, `${f.toFixed(2)} billion`);
    kst.set(3, `${(path * ck.stages).toFixed(0)} ps (${ck.stages} stage${ck.stages > 1 ? "s" : ""})`);
  }

  // ---------- 4. multiplier size ----------
  const mu = { bits: 24 };
  const mst = stats([{ label: "AND gates (partial-product bits)" }, { label: "Full adders, roughly" }, { label: "Size vs a 24-bit (fp32) multiplier" }]);
  function drawMu() {
    const n = mu.bits, rel = (n * n) / (24 * 24);
    mst.set(0, String(n * n)); mst.set(1, String(n * (n - 1)));
    mst.set(2, rel >= 1 ? `${rel.toFixed(1)}× bigger` : `${(1 / rel).toFixed(1)}× smaller`, rel < 0.2 ? "good" : rel <= 1 ? "warn" : "bad");
  }

  root.append(
    section(null,
      goal("<p>We have a voltage-controlled switch. We need <b>arithmetic</b>: circuits that add and multiply numbers, reliably, a billion times a second, and that keep millions of results in step. This chapter climbs from two transistors to a multiplier, and shows why the size of a multiplier grows with the square of its bits, which is the root of every low-precision trick in AI.</p>",
        ["A signal that survives noise", "One building block that makes every function", "A beat that keeps results in step"]),
      stackMap(id),
    ),
    section("Why everything is 0 or 1",
      whyChain({
        claim: "Computers represent numbers with two voltage levels, not ten.",
        steps: [
          { q: "Why not use ten levels and store a decimal digit per wire?", a: "<p>Every voltage on a chip wobbles: from heat, from neighbouring wires switching, from tiny differences between transistors. With ten levels 0.08 V apart, a small wobble turns a 4 into a 5. With two levels, a wobble has to be half the supply voltage before a 0 reads as a 1.</p>" },
          { q: "Why don't errors build up through millions of gates?", a: "<p>Because each gate <b>restores</b> the signal. A gate's output swings fully to the supply or to ground even if its input is a little off: its gain near the switching point is much more than 1, so small errors shrink at every stage instead of adding up. An analog circuit can't do that: every stage adds its own error.</p>" },
          { q: "Why is a transistor good at exactly two states?", a: "<p>Fully on and fully off are the two states where a CMOS gate wastes no power: in either, one of its two networks is off, so no current flows from supply to ground. In-between levels would mean both are partly on, burning power all the time.</p>" },
          { q: "Why is there noise at all?", a: "<p>Electrons jiggle with thermal energy, so the voltage on any wire fluctuates by about √(kT/C), and manufacturing makes no two transistors identical. Smaller transistors mean smaller C, so more noise per node.</p>" },
        ],
        axiom: { name: "Information needs states that noise can't confuse", html: "<p>Shannon showed that the information a signal can carry is limited by its power over the noise. With thermal noise always present, the most robust and cheapest choice in CMOS is the fewest levels: two. Everything below this point is analog physics; everything above is digital, exact, and copyable without loss.</p>" },
      }),
    ),
    section("Two transistors make a NOT; four make a NAND",
      prose("<p>CMOS (complementary MOS) gates come in matched halves. A <b>pull-up</b> network of p-type transistors connects the output to the supply; a <b>pull-down</b> network of n-type transistors connects it to ground. They are built so exactly one is on for every input. In a NAND, the pull-downs are in series (both inputs must be 1 to reach ground) and the pull-ups in parallel (either 0 connects the supply). Click the inputs.</p>"),
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Gate", options: [{ value: "not", label: "NOT" }, { value: "nand", label: "NAND" }, { value: "nor", label: "NOR" }], value: g.kind, onChange: (v) => { g.kind = v; drawG(); } }),
          h("div.btn-row", btnA, btnB)),
        h("div.stage-body", gst.el, h("div.wide-canvas", gc.el), truth)),
      callout("tip", "Why CMOS won", "<p>Notice the third number: with the inputs still, <b>no current flows</b> from supply to ground. A CMOS gate only uses energy when it switches, to charge or drain the capacitance of whatever it drives. Older logic families burned power all the time. This is why a phone with billions of transistors can idle for a day.</p>"),
    ),
    section("NAND makes everything",
      stepper([
        { title: "NOT from NAND", html: "<p>Tie both inputs together: NAND(A, A) = NOT A. One gate.</p>" },
        { title: "AND and OR", html: "<p>AND = NOT(NAND(A, B)): two gates. OR = NAND(NOT A, NOT B): three gates (De Morgan's law). So NAND alone is <b>universal</b>: any truth table at all can be built from it.</p>" },
        { title: "XOR: “exactly one”", html: "<p>XOR(A, B) is 1 when the inputs differ. It's the sum bit of a 1-bit addition: 0 + 1 = 1, 1 + 1 = 0 carry 1. Four NAND gates.</p>" },
        { title: "A full adder: one column of addition", html: "<p>A <b>full adder</b> adds three bits (A, B and a carry in) and outputs a sum bit and a carry out: sum = A XOR B XOR C, carry = majority(A, B, C). Nine NAND gates, or about 28 transistors in a tuned CMOS design.</p>" },
        { title: "A row of them adds numbers", html: "<p>Chain eight full adders, each passing its carry to the next, and you add two 8-bit numbers. Chain 24 and you add the fractions of two fp32 numbers. Below, try it.</p>" },
      ]),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "A", min: 0, max: 255, value: ad.A, onInput: (v) => { ad.A = v; drawA(); } }),
          slider({ label: "B", min: 0, max: 255, value: ad.B, onInput: (v) => { ad.B = v; drawA(); } })),
        h("div.stage-body", ast.el, h("div.wide-canvas", ac.el))),
      callout("note", "Carries are the slow part", "<p>Try 255 + 1: the carry ripples through all eight adders before the top bit is right. Real adders use <b>carry-lookahead</b> trees that compute all carries in about log₂(n) steps instead of n, spending more gates to save time. That trade (more transistors for less waiting) is the oldest move in chip design.</p>"),
    ),
    walk("logic"),
    section("Multipliers grow with the square of the bits",
      prose("<p>The walkthrough above multiplied two 4-bit numbers with 16 AND gates and a block of adders. An n-bit multiplier needs n² AND gates and about n² full adders. A floating-point number multiplies its fractions (24 bits in fp32, 8 in bf16, 4 in fp8 E4M3, counting the hidden leading 1) and just adds its exponents, which is cheap.</p>"),
      h("div.stage",
        h("div.stage-controls", slider({ label: "Bits in the fraction being multiplied", min: 2, max: 53, value: mu.bits, onInput: (v) => { mu.bits = v; drawMu(); } })),
        h("div.stage-body", mst.el)),
      table(["Format", "Fraction bits multiplied", "Multiplier size vs fp32", "Where it's used"], [
        ["fp64", "53", "≈ 4.9× bigger", "science; barely used in AI"],
        ["fp32", "24", "1×", "master weights, accumulators"],
        ["bf16", "8", "≈ 9× smaller", "training since ~2020"],
        ["fp8 (E4M3)", "4", "≈ 36× smaller", "training and inference on H100 and later"],
        ["fp4 (E2M1)", "2", "≈ 144× smaller", "inference on Blackwell"],
      ]),
      callout("note", "Why chips don't get the full square", "<p>A tensor core typically does 2× the operations per halving of precision, not 4×. Each operation still needs its inputs fetched and its outputs added into an fp32 accumulator, and that wiring and data movement doesn't shrink with the multiplier. The square law is why low precision is cheap; the cost of moving bits is why it's not free. More on that in <a href='/learn?c=memory-physics'>the memory chapter</a>.</p>"),
      predict({
        chapter: id,
        question: "Roughly how much smaller is an 8-bit multiplier than a 32-bit one?",
        options: ["4× (a quarter of the bits)", "About 16× (the square of 4)", "The same: a multiplier is a multiplier", "64×"],
        answer: 1,
        explain: "<p>Partial products grow as n × n: 64 for 8 bits, 1,024 for 32. Ratio 16. That quadratic is why int8 and fp8 inference is so much cheaper per operation than fp32, and why mixed-precision training (multiply in bf16, accumulate in fp32) became standard.</p>",
      }),
    ),
    section("The clock: keeping billions of results in step",
      prose("<p>After the inputs change, a circuit's outputs take time to <b>settle</b>: each gate takes some picoseconds, and the answer is only right once the slowest path has finished. So chips put <b>registers</b> (flip-flops, ~20 transistors per bit) between blocks of logic, and a <b>clock</b> tells every register to capture its input at the same instant. The clock period must be longer than the slowest path. Split the logic into <b>pipeline stages</b> and each stage is shorter, so the clock can run faster: more results per second, though each one takes a little longer start to finish.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Gates on the slowest path", min: 4, max: 80, value: ck.depth, onInput: (v) => { ck.depth = v; drawK(); } }),
          slider({ label: "Delay per gate", min: 4, max: 40, value: ck.delay, fmt: (v) => `${v} ps`, onInput: (v) => { ck.delay = v; drawK(); } }),
          slider({ label: "Pipeline stages", min: 1, max: 8, value: ck.stages, onInput: (v) => { ck.stages = v; drawK(); } })),
        h("div.stage-body", kst.el)),
      callout("note", "Why clocks stopped at ~4–5 GHz", "<p>Deeper pipelines and faster gates pushed clocks from 1 MHz to 3.8 GHz between 1975 and 2004. Then they stopped. Gates kept getting faster; the problem was <b>heat</b>, because every clock tick charges and drains millions of capacitors. That's <a href='/learn?c=power-wall'>the power wall</a>, and it's why the answer became more units side by side instead of a faster clock.</p>"),
    ),
    section("In code",
      code({
        "Everything from NAND": `def NAND(a, b): return 1 - (a & b)
def NOT(a):     return NAND(a, a)
def AND(a, b):  return NOT(NAND(a, b))
def OR(a, b):   return NAND(NOT(a), NOT(b))
def XOR(a, b):
    n = NAND(a, b)
    return NAND(NAND(a, n), NAND(b, n))       # 4 NANDs

def full_adder(a, b, c):
    s1 = XOR(a, b)
    return XOR(s1, c), OR(AND(a, b), AND(s1, c))   # sum, carry

def add(x, y, n=8):
    out, c = 0, 0
    for i in range(n):                         # carries ripple upward
        s, c = full_adder((x >> i) & 1, (y >> i) & 1, c)
        out |= s << i
    return out | (c << n)

assert add(87, 45) == 132`,
        "A multiplier from AND + adders": `def multiply(x, y, n=4):
    total = 0
    for i in range(n):
        bit = (y >> i) & 1
        row = sum(AND((x >> k) & 1, bit) << (k + i) for k in range(n))  # n AND gates
        total = add(total, row, 2 * n)
    return total

assert multiply(11, 13) == 143
# gates used grows as n*n: count calls to NAND to see it`,
        "Same thing in Verilog": `module mac8 (input clk, input [7:0] a, b, output reg [19:0] acc);
  always @(posedge clk)            // a register: captures on each clock tick
    acc <= acc + a * b;            // synthesis builds the multiplier and adder from gates
endmodule`,
      }, "Project 60 builds the NAND adder and multiplier and counts their gates as the bit width grows."),
    ),
  );
  drawG(); drawA(); drawK(); drawMu();
}
