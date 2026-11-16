// Step-throughs for the electrons-to-CUDA level: a multiplier built from AND gates and adders,
// a warp summing 32 numbers in 5 steps, and a CUDA grid mapping threads onto an array.
import { tensorWalk, op } from "./tensorviz.js";

const BIT = (v) => (v == null ? "#f6f4fa" : v ? "#5b3fd6" : "#e6e1f3");
const bitsOf = (x, n) => Array.from({ length: n }, (_, j) => (x >> (n - 1 - j)) & 1);   // most significant bit first

export const SILICON = {
  // -------------------------------------------------------------------------------------------
  logic() {
    const A = 11, B = 13, N = 4, W = 8, P = A * B;
    const a = bitsOf(A, N), b = bitsOf(B, N);
    // partial product i (for bit i of B, i = 0 is the least significant) is A shifted left by i
    const PP = Array.from({ length: N }, (_, i) => Array.from({ length: W }, (_, j) => { const pos = W - 1 - j, k = pos - i; return k >= 0 && k < N && (B >> i) & 1 ? (A >> k) & 1 : k >= 0 && k < N ? 0 : null; }));
    const colSum = [], carryIn = [], out = Array(W).fill(null), cIn = Array(W).fill(null);
    let carry = 0;
    for (let pos = 0; pos < W; pos++) {
      const j = W - 1 - pos, ones = PP.reduce((s, r) => s + (r[j] || 0), 0), total = ones + carry;
      colSum[j] = ones; carryIn[j] = carry; carry = total >> 1;
    }
    const prod = bitsOf(P, W);
    const T = {
      a: { data: [a], label: `A = ${A}`, dims: "bit (8 4 2 1)", colors: BIT, kind: "ids", px: 7 },
      b: { data: b.slice().reverse().map((v) => [v]), label: `B = ${B}`, dims: "bit 1, 2, 4, 8 (top to bottom)", colors: BIT, kind: "ids", px: 7 },
      PP: { data: PP, label: "partial products", dims: "row = bit of B, column = place value", colors: BIT, kind: "ids", empty: "nothing in this place", rowNames: ["× 1", "× 2", "× 4", "× 8"] },
      S: { data: [Array(W).fill(null)], label: "column sums", dims: "place value", kind: "prob", scale: 4, bars: 3, empty: "not added yet" },
      P: { data: [Array(W).fill(null)], label: "product bits", dims: "place value 128 … 1", colors: BIT, kind: "ids" },
      F: { data: [prod], label: `A × B = ${P}`, dims: "place value 128 … 1", colors: BIT, kind: "ids" },
    };
    const S = T.S.data[0], Pr = T.P.data[0];
    const place = (j) => 2 ** (W - 1 - j);
    return tensorWalk({
      title: "A multiplier, built from AND gates and adders",
      lede: "Multiplying binary numbers is the long multiplication you learned at school, except every digit is 0 or 1, so “times a digit” is just an AND gate. The cost grows with the square of the number of bits.",
      code: `A, B = 0b1011, 0b1101              # 11 and 13, four bits each
pp = []
for i in range(4):                    # one row per bit of B
    bit = (B >> i) & 1
    pp.append((A if bit else 0) << i) # AND gates: copy A or write zeros, shifted
product = 0
for row in pp:                        # adders sum the columns, carrying left
    product += row
print(product)                        # 143 = 0b10001111`,
      tensors: T,
      key: [["#5b3fd6", "1"], ["#e6e1f3", "0"], ["#f6f4fa", "empty"]],
      steps: [
        { title: "Two numbers as bits", lines: [1], show: ["a", "b"], text: `<p>A = 11 is 1011 (8 + 2 + 1), B = 13 is 1101 (8 + 4 + 1). Each cube is one wire in the circuit: dark means high voltage (1), pale means low (0).</p>` },
        { title: "AND gates make the partial products", lines: [3, 4, 5], show: ["b", "→", "PP"], speed: 1500, op: op.custom(N, (i) => ({ hl: { b: [`${i},0`], PP: Array.from({ length: W }, (_, j) => `${i},${j}`).filter((_, j) => PP[i][j] != null) }, read: `bit ${i} of B is <b>${(B >> i) & 1}</b>, so row ${i} is ${(B >> i) & 1 ? `A shifted ${i} place${i === 1 ? "" : "s"} left (A × ${2 ** i})` : "all zeros"}: each cube is A's bit AND this bit of B` }), (name, key) => (name === "PP" || name === "b" ? +key.split(",")[0] : null)), text: "<p>For each bit of B, 4 AND gates either copy A or output zeros, shifted one place further left each row. A 4-bit multiplier needs 4 × 4 = 16 AND gates; a 24-bit one (the fraction of an fp32 number) needs 576.</p>" },
        { title: "Adders sum each column, carrying left", lines: [7, 8, 9], show: ["PP", "→", "S", "→", "P"], speed: 1300, op: op.custom(W, (k) => {
          const j = W - 1 - k;
          for (let q = 0; q < W; q++) { S[q] = q >= j ? colSum[q] + carryIn[q] : null; Pr[q] = q >= j ? prod[q] : null; }
          return { hl: { PP: Array.from({ length: N }, (_, i) => `${i},${j}`).filter((key) => PP[+key[0]][j] != null), S: [`0,${j}`], P: [`0,${j}`] }, read: `the ${place(j)}s column: ${colSum[j]} one${colSum[j] === 1 ? "" : "s"} + ${carryIn[j]} carried in = <b>${colSum[j] + carryIn[j]}</b> → write <b>${prod[j]}</b>${colSum[j] + carryIn[j] > 1 ? `, carry ${(colSum[j] + carryIn[j]) >> 1} to the ${place(j) * 2}s column` : ""}` };
        }, (name, key) => (name === "P" || name === "S" ? W - 1 - +key.split(",")[1] : null)), text: "<p>Full adders (each about 28 transistors in CMOS) add the bits in each column plus the carry from the column to the right. Hardware doesn't add the rows one at a time like this: it uses a tree of adders so the whole sum settles in a few gate delays. But the work is the same.</p>" },
        { title: "The product", lines: [10], show: ["a", "×", "b", "=", "F"], text: `<p>10001111 = 128 + 8 + 4 + 2 + 1 = <b>${P}</b>, which is 11 × 13. Every multiply-add in a neural network is this, in silicon, with wider numbers. Halving the bits roughly quarters the multiplier, which is why AI hardware moved from fp32 to bf16 to fp8.</p>` },
      ],
    });
  },

  // -------------------------------------------------------------------------------------------
  "gpu-design"() {
    // 32 lanes of one warp each hold a partial dot product; shuffles halve the count each step
    const v = Array.from({ length: 32 }, (_, i) => +(((i * 37) % 19) / 10 - 0.6).toFixed(1));
    const levels = [v];
    for (let off = 16; off >= 1; off /= 2) { const p = levels[levels.length - 1]; levels.push(Array.from({ length: off }, (_, i) => +(p[i] + p[i + off]).toFixed(1))); }
    const grid = (a) => { const w = Math.min(8, a.length); return Array.from({ length: Math.ceil(a.length / w) }, (_, r) => a.slice(r * w, r * w + w)); };
    const names = ["x", "s16", "s8", "s4", "s2", "s1"];
    const T = {};
    levels.forEach((a, k) => { T[names[k]] = { data: grid(a), label: k ? `after offset ${32 >> k}` : "32 lanes", dims: k ? `${a.length} live lanes` : "lane = row × 8 + column", scale: 3 }; });
    const lane = (i, w) => `${Math.floor(i / w)},${i % w}`;
    const stepOp = (k) => {
      const off = 32 >> (k + 1), src = names[k], dst = names[k + 1], ws = Math.min(8, levels[k].length), wd = Math.min(8, off);
      return op.custom(off, (i) => ({ hl: { [src]: [lane(i, ws), lane(i + off, ws)], [dst]: [lane(i, wd)] }, read: `lane ${i} adds lane ${i + off}'s value, fetched straight from its registers: ${levels[k][i]} + ${levels[k][i + off] < 0 ? `(${levels[k][i + off]})` : levels[k][i + off]} = <b>${levels[k + 1][i]}</b>` }),
        (name, key) => { if (name !== dst) return null; const [r, c] = key.split(",").map(Number); return r * wd + c; });
    };
    return tensorWalk({
      title: "A warp adds 32 numbers in 5 steps",
      lede: "The 32 threads of a warp sit side by side in one SM and can read each other's registers directly with a shuffle. Summing across them takes log₂ 32 = 5 steps instead of 31, with no trip to memory.",
      code: `float v = partial;                         // each of 32 lanes holds one number
for (int off = 16; off > 0; off /= 2)
    v += __shfl_down_sync(0xffffffff, v, off);   // read lane (me + off)'s v
if (lane == 0) out[warp] = v;             // lane 0 now holds the total`,
      tensors: T,
      steps: [
        { title: "32 lanes, 32 numbers", lines: [1], show: ["x"], text: "<p>Each lane (thread) has computed a partial sum, say its share of a dot product. They're in registers: the fastest memory on the chip, private to each lane.</p>" },
        ...[0, 1, 2, 3, 4].map((k) => ({ title: k === 0 ? "Fold in half: offset 16" : k === 4 ? "Last fold: offset 1" : `Fold again: offset ${32 >> (k + 1)}`, lines: [2, 3], show: [names[k], "→", names[k + 1]], op: stepOp(k), speed: k < 2 ? 500 : 1000,
          text: k === 0 ? "<p>Every lane i reads the value of lane i + 16 and adds it. All 16 additions happen in the same instruction, in one cycle-ish, because the warp executes in lockstep.</p>" : k === 4 ? `<p>Lane 0 adds lane 1. The total is <b>${levels[5][0]}</b>, after 5 instructions. A CPU would need 31 dependent adds; a naive GPU version would write all 32 values to memory and read them back.</p>` : "<p>Half as many lanes do useful work each step: the other lanes still execute the instruction, but their results are ignored.</p>" })),
        { title: "Why this is fast", lines: [4], show: ["x", "→", "s1"], text: "<p>Nothing left the SM. The shuffle uses the wiring between lanes of a warp, which exists because the warp shares one instruction stream. Reductions like this sit inside every softmax, every layer norm and every loss.</p>" },
      ],
    });
  },

  // -------------------------------------------------------------------------------------------
  cuda() {
    const N = 20, BD = 8, G = Math.ceil(N / BD), NT = G * BD;
    const BLK = ["#4f8ef7", "#ff7a45", "#22b8a0"], OFF = "#c9ccd6";
    const a = Array.from({ length: N }, (_, i) => ((i * 7) % 10) - 3), b = Array.from({ length: N }, (_, i) => ((i * 3) % 7) - 2), c = a.map((x, i) => x + b[i]);
    const blk = Array.from({ length: NT }, (_, t) => Math.floor(t / BD)), tid = Array.from({ length: NT }, (_, t) => t % BD), gi = Array.from({ length: NT }, (_, t) => t);
    const T = {
      blk: { data: [blk], label: "blockIdx.x", dims: "one per thread", colors: (v) => BLK[v], kind: "ids" },
      tid: { data: [tid], label: "threadIdx.x", dims: "one per thread", kind: "prob", scale: 7 },
      i: { data: [gi], label: "i = blockIdx.x × 8 + threadIdx.x", dims: "one per thread", colors: (v) => (v >= N ? OFF : BLK[Math.floor(v / BD)]), kind: "ids" },
      a: { data: [a], label: "a", dims: "20 elements" }, b: { data: [b], label: "b", dims: "20 elements" }, c: { data: [c], label: "c = a + b", dims: "20 elements" },
    };
    return tensorWalk({
      title: "Your first kernel: a million threads, one line each",
      lede: "A CUDA kernel is the code for one thread. You launch a grid of blocks of threads, and each thread works out which element is its own. Here: 20 elements, blocks of 8 threads, so 3 blocks.",
      code: `__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // which element is mine?
    if (i < n)                                         // the last block has spare threads
        c[i] = a[i] + b[i];
}
int blocks = (n + 7) / 8;                              // round up: 3 blocks for n = 20
add<<<blocks, 8>>>(a, b, c, n);                        // launch 24 threads`,
      tensors: T,
      key: [[BLK[0], "block 0"], [BLK[1], "block 1"], [BLK[2], "block 2"], [OFF, "past the end"]],
      steps: [
        { title: "Launch: 3 blocks of 8 threads", lines: [6, 7], show: ["blk", "tid"], text: "<p>The launch <code>&lt;&lt;&lt;3, 8&gt;&gt;&gt;</code> creates 24 threads. Each knows two numbers: which block it's in (colour) and its position inside the block (0–7, light to dark). Real launches use blocks of 128–1,024 threads and up to millions of blocks.</p>" },
        { title: "Each thread finds its element", lines: [2], show: ["blk", "tid", "→", "i"], op: op.custom(NT, (t) => ({ hl: { blk: [`0,${t}`], tid: [`0,${t}`], i: [`0,${t}`] }, read: `block ${blk[t]}, thread ${tid[t]}: i = ${blk[t]} × 8 + ${tid[t]} = <b>${t}</b>${t >= N ? " (past the end of the array)" : ""}` }), (name, key) => (["blk", "tid", "i"].includes(name) ? +key.split(",")[1] : null)), speed: 450, text: "<p>The same line runs in every thread, with different built-in values, so each one lands on a different element. Neighbouring threads get neighbouring elements, which is exactly what memory likes (coalescing, in <a href='/learn?c=gpu-from-scratch'>the kernels chapter</a>).</p>" },
        { title: "The guard: spare threads do nothing", lines: [3], show: ["i"], op: op.custom(NT - N, (k) => ({ hl: { i: [`0,${N + k}`] }, read: `i = ${N + k} ≥ n = ${N}: this thread skips the add. Without the <code>if</code> it would write past the end of c.` }), (name, key) => (name === "i" && +key.split(",")[1] >= N ? +key.split(",")[1] - N : null)), speed: 900, text: "<p>n isn't a multiple of 8, so the last block has 4 threads with nothing to do. Grids are rounded up, and every kernel checks its bounds. Forgetting this is the most common CUDA bug.</p>" },
        { title: "c[i] = a[i] + b[i], all at once", lines: [4], show: ["a", "+", "b", "=", "c"], op: op.cells(["a", "b", "c"], 1, N, (_, j) => `thread ${j} (block ${Math.floor(j / BD)}, lane ${j % BD}): c[${j}] = ${a[j]} + ${b[j] < 0 ? `(${b[j]})` : b[j]} = <b>${c[j]}</b>`), speed: 350, text: "<p>Every thread does one add. On a GPU these run as warps of 32 on many SMs at the same time; the order between blocks is not defined and must not matter. That rule, <b>blocks are independent</b>, is what lets the same program run on a GPU with 16 SMs or 132.</p>" },
      ],
    });
  },

  // -------------------------------------------------------------------------------------------
  "all-the-way-down"() {
    // 1.5 × 2.75 = 4.125 in bf16, the way a multiplier lane does it
    const wA = 0b0011111111000000, wB = 0b0100000000110000, wR = 0b0100000010000100;
    const eA = 127, eB = 128, eP = eA + eB - 127, mA = 0b11000000, mB = 0b10110000, mP = mA * mB;   // 192 × 176 = 33792
    const PP = Array.from({ length: 8 }, (_, i) => Array.from({ length: 16 }, (_, j) => { const pos = 15 - j, k = pos - i; return k >= 0 && k < 8 ? ((mB >> i) & 1 ? (mA >> k) & 1 : 0) : null; }));
    const T = {
      a: { data: [bitsOf(wA, 16)], label: "a = 1.5", dims: "sign · exponent (8) · fraction (7)", colors: BIT, kind: "ids", px: 6 },
      b: { data: [bitsOf(wB, 16)], label: "b = 2.75", dims: "sign · exponent (8) · fraction (7)", colors: BIT, kind: "ids", px: 6 },
      ea: { data: [bitsOf(eA, 8)], label: `exponent of a = ${eA}`, dims: "means 2^(127 − 127) = 1", colors: BIT, kind: "ids" },
      eb: { data: [bitsOf(eB, 8)], label: `exponent of b = ${eB}`, dims: "means 2^(128 − 127) = 2", colors: BIT, kind: "ids" },
      ep: { data: [bitsOf(eP, 8)], label: `${eA} + ${eB} − 127 = ${eP}`, dims: "8-bit adder, minus the bias", colors: BIT, kind: "ids" },
      ma: { data: [bitsOf(mA, 8)], label: "1.1000000 (1.5)", dims: "hidden 1, then a's 7 fraction bits", colors: BIT, kind: "ids" },
      mb: { data: bitsOf(mB, 8).reverse().map((v) => [v]), label: "1.0110000 (1.375)", dims: "bits of b, lowest at the top", colors: BIT, kind: "ids", px: 7 },
      PP: { data: PP, label: "partial products", dims: "row = bit of b, column = place", colors: BIT, kind: "ids", empty: "nothing in this place" },
      mp: { data: [bitsOf(mP, 16)], label: `${mA} × ${mB} = ${mP}`, dims: "16-bit product: 10.00010000… = 2.0625", colors: BIT, kind: "ids" },
      en: { data: [bitsOf(eP + 1, 8)], label: `exponent ${eP + 1}`, dims: "+1 because the product was ≥ 2", colors: BIT, kind: "ids" },
      r: { data: [bitsOf(wR, 16)], label: "a × b = 4.125", dims: "0 · 10000001 · 0000100", colors: BIT, kind: "ids", px: 6 },
    };
    const row = (n, i) => Array.from({ length: 16 }, (_, j) => `${i},${j}`).filter((_, j) => PP[i][j] != null);
    return tensorWalk({
      title: "One bf16 multiply, bit by bit",
      lede: "Zoom level 6 of this chapter, in slow motion. A tensor core does 512 of these every tick: split each number into sign, exponent and fraction, add the exponents, multiply the fractions with AND gates and adders, then line the result back up.",
      code: `a, b = 1.5, 2.75                 # bf16: 1 sign, 8 exponent, 7 fraction bits
sign = 0 ^ 0                      # XOR the signs: positive
exp  = 127 + 128 - 127            # add exponents, remove one bias: 128
man  = 0b11000000 * 0b10110000    # 1.5 × 1.375 as 8-bit integers = 33792
if man >= 2 << 14:                # product is 2.0625, not in [1, 2)
    man >>= 1; exp += 1           # shift right, exponent 129
frac = (man >> 7) & 0x7F          # keep 7 bits after the leading 1: 0000100
print((1 + frac / 128) * 2 ** (exp - 127))   # 4.125`,
      tensors: T,
      key: [["#5b3fd6", "1"], ["#e6e1f3", "0"], ["#f6f4fa", "empty"]],
      steps: [
        { title: "Two numbers as 16 wires each", lines: [1], show: ["a", "b"], text: "<p>bf16 is the top half of an fp32 number. 1.5 is 1.1 in binary × 2⁰; 2.75 is 1.011 × 2¹. The exponent is stored with a bias of 127, and the leading 1 of the fraction isn't stored at all: every normal number has one.</p>" },
        { title: "Signs: one XOR gate", lines: [2], show: ["a", "b"], op: op.custom(1, () => ({ hl: { a: ["0,0"], b: ["0,0"] }, read: "sign of a = 0, sign of b = 0, 0 XOR 0 = <b>0</b>: the product is positive" })), text: "<p>The first bit of each is the sign. Positive × positive is positive: one XOR gate decides it.</p>" },
        { title: "Exponents: an 8-bit adder", lines: [3], show: ["ea", "+", "eb", "→", "ep"], text: `<p>Multiplying powers of two adds their exponents. Both are stored +127, so add them and subtract one 127: 127 + 128 − 127 = <b>128</b>, meaning 2¹. That's an 8-bit adder, the same circuit as <a href="/learn?c=logic">the logic chapter</a>'s, with a carry chain.</p>` },
        { title: "Fractions: AND gates make the rows", lines: [4], show: ["ma", "×", "mb", "→", "PP"], speed: 1100, op: op.custom(8, (i) => ({ hl: { mb: [`${i},0`], PP: row(PP, i) }, read: `bit ${i} of b's fraction is <b>${(mB >> i) & 1}</b>, so row ${i} is ${(mB >> i) & 1 ? `1.1000000 shifted ${i} place${i === 1 ? "" : "s"}` : "all zeros"}` }), (name, key) => (name === "PP" || name === "mb" ? +key.split(",")[0] : null)), text: "<p>Put the hidden 1 back and multiply the two 8-bit fractions: 8 × 8 = 64 AND gates make 8 shifted rows. This is why bf16 is cheap. fp32 has 24-bit fractions: 576 AND gates and 9× the adders.</p>" },
        { title: "Adders sum the columns", lines: [4], show: ["PP", "→", "mp"], text: `<p>A tree of full adders sums the columns: 192 × 176 = <b>33,792</b>. Read as a fixed-point number with 14 bits after the point, that's 10.00010000000000 in binary = 2.0625, which is 1.5 × 1.375.</p>` },
        { title: "Normalise: shift and bump the exponent", lines: [5, 6, 7], show: ["mp", "→", "en"], op: op.custom(1, () => ({ hl: { mp: ["0,0", "0,1"], en: ["0,7"] }, read: "the product starts with <b>10.</b>, so it's ≥ 2: shift right one place and add 1 to the exponent (128 → <b>129</b>)" })), text: "<p>A normal number must look like 1.xxx. Ours is 10.0001, so shift it right one place and add 1 to the exponent. Then keep the 7 bits after the leading 1 (<b>0000100</b>) and round: every bit dropped here is 0, so this product is exact.</p>" },
        { title: "The result, and then + c", lines: [8], show: ["a", "×", "b", "=", "r"], text: `<p>0 · 10000001 · 0000100 = 1.03125 × 2² = <b>4.125</b>. In a tensor core this product goes straight into an fp32 adder that adds it to c, the running sum: the “add” in multiply-add. Repeat 512 times per tick, 528 tensor cores, 1.83 billion ticks a second, and that's the H100's 989 TFLOP/s.</p>` },
      ],
    });
  },

  // -------------------------------------------------------------------------------------------
  electrons() {
    const Z = [[14, 14, 14, 14, 14], [14, 14, 14, 15, 14], [14, 14, 14, 14, 14], [14, 5, 14, 14, 14], [14, 14, 14, 14, 14]];
    const VAL = { 14: 4, 15: 5, 5: 3 }, NAME = { 14: "silicon", 15: "phosphorus", 5: "boron" };
    const val = Z.map((r) => r.map((z) => VAL[z])), bond = Z.map((r) => r.map(() => 4)), left = val.map((r) => r.map((v) => v - 4));
    const atomC = (v) => (v === 15 ? "#ff8a3d" : v === 5 ? "#e76fa8" : "#8ea3c7");
    const valC = (v) => (v === 5 ? "#ff8a3d" : v === 3 ? "#e76fa8" : "#b9c4dc");
    const leftC = (v) => (v > 0 ? "#ffd23f" : v < 0 ? "#e76fa8" : "#eef1f8");
    const k = 8.617e-5, gap = (T) => 1.17 - 4.73e-4 * T * T / (T + 636);
    const ni = (T) => Math.sqrt(2.86e19 * 3.10e19 * (T / 300) ** 3) * Math.exp(-gap(T) / (2 * k * T));
    const TS = [250, 300, 350, 400, 500], lg = TS.map((T) => +Math.log10(ni(T)).toFixed(2));
    const T = {
      Z: { data: Z, label: "the crystal", dims: "atomic number: 14 Si, 15 P, 5 B", colors: atomC, kind: "ids", px: 6 },
      val: { data: val, label: "outer electrons", dims: "per atom", colors: valC, kind: "ids", px: 6 },
      bond: { data: bond, label: "used in bonds", dims: "4 neighbours, one each", colors: () => "#b9c4dc", kind: "ids", px: 6 },
      left: { data: left, label: "left over", dims: "+1 free electron, −1 hole", colors: leftC, kind: "ids", px: 6 },
      temp: { data: [TS], label: "temperature (K)", dims: "250 … 500 K", colors: () => "#ffcf9e", kind: "ids" },
      lg: { data: [lg], label: "log₁₀ free electrons per cm³", dims: "pure silicon, from heat alone", kind: "prob", scale: 16 },
    };
    return tensorWalk({
      title: "Doping, atom by atom",
      lede: "Count electrons on a patch of silicon crystal. Every atom shares one electron with each of its 4 neighbours. Swap in an atom with 5 or 3, and one electron is left over, or one is missing.",
      code: `Z   = lattice()                         # 14 = Si, 15 = P, 5 = B
val = {14: 4, 15: 5, 5: 3}[Z]           # outer (valence) electrons
used = 4                                # one per bond, four bonds each
left = val - used                       # +1 free electron, -1 hole
# pure silicon: only heat frees electrons
n_i = sqrt(Nc * Nv) * exp(-Eg / (2 * k * T))`,
      tensors: T,
      key: [["#8ea3c7", "silicon"], ["#ff8a3d", "phosphorus"], ["#e76fa8", "boron / hole"], ["#ffd23f", "free electron"]],
      steps: [
        { title: "A patch of crystal", lines: [1], show: ["Z"], text: "<p>A 5 × 5 patch of a silicon crystal (really 3D, drawn flat). One phosphorus atom and one boron atom have been put in on purpose. Real doping is about 1 atom in 10⁵–10⁸, not 2 in 25.</p>" },
        { title: "Count the outer electrons", lines: [2], show: ["Z", "→", "val"], speed: 260, op: op.cells(["Z", "val"], 5, 5, (i, j) => `${NAME[Z[i][j]]} (Z = ${Z[i][j]}) has <b>${val[i][j]}</b> outer electrons`), text: "<p>Silicon is in column 14 of the periodic table: 4 outer electrons. Phosphorus (column 15) has 5, boron (column 13) has 3. Only the outer electrons take part in bonding.</p>" },
        { title: "Bonds use four each", lines: [3], show: ["val", "−", "bond"], text: "<p>In the crystal every atom sits between 4 neighbours and shares one electron with each: a covalent bond. That uses exactly 4 outer electrons per atom, whatever the atom is.</p>" },
        { title: "What's left over", lines: [4], show: ["val", "−", "bond", "=", "left"], speed: 260, op: op.cells(["val", "bond", "left"], 5, 5, (i, j) => `${val[i][j]} − 4 = <b>${left[i][j] > 0 ? "+1: a spare electron, free to wander" : left[i][j] < 0 ? "−1: a missing electron, a hole" : "0: nothing free"}</b>`), text: "<p>Phosphorus has one electron with no bond to sit in, so it wanders freely: <b>n-type</b>. Boron is one short, leaving a hole that neighbouring electrons can hop into: <b>p-type</b>. That's the whole knob. A gate voltage then decides where these carriers gather.</p>" },
        { title: "Without dopants: only heat", lines: [5, 6], show: ["temp", "→", "lg"], speed: 700, op: op.cells(["temp", "lg"], 1, 5, (_, j) => `at ${TS[j]} K: about <b>10^${lg[j]}</b> free electrons per cm³ in pure silicon, out of 5 × 10²² atoms`), text: "<p>In pure silicon, an electron only gets free if heat kicks it across the 1.12 eV gap. The odds fall as e^(−Eg/2kT): at room temperature only ~10¹⁰ per cm³, about one atom in five trillion. Heating from 300 K to 400 K multiplies it by hundreds, which is why chips must be kept cool and why dopants, not heat, set the carriers.</p>" },
      ],
    });
  },

  // -------------------------------------------------------------------------------------------
  transistor() {
    const kT = 0.025852, dec = kT * Math.LN10, VG = [0, 0.06, 0.12, 0.18, 0.24, 0.3];
    const phi = VG.map((v) => +(0.5 - v).toFixed(2)), lg = phi.map((p) => +(-p / dec).toFixed(2));
    const ratio = lg.map((v, i) => (i ? +(10 ** (v - lg[i - 1])).toFixed(1) : null)), real = lg.map((v) => +(v / 1.3).toFixed(2));
    const T = {
      vg: { data: [VG], label: "gate voltage (V)", dims: "steps of 60 mV", kind: "prob", scale: 0.4 },
      phi: { data: [phi], label: "barrier (eV)", dims: "0.5 − gate voltage", kind: "prob", scale: 0.6 },
      lg: { data: [lg], label: "log₁₀ (fraction over the barrier)", dims: "−barrier ÷ (kT ln 10)", scale: 9 },
      ratio: { data: [ratio], label: "× more current than the step before", dims: "10^(difference)", kind: "prob", scale: 12, empty: "first step" },
      real: { data: [real], label: "a real gate (n = 1.3)", dims: "log₁₀, 78 mV per decade", scale: 9 },
    };
    return tensorWalk({
      title: "Why 60 mV buys 10× current",
      lede: "Below threshold, current is electrons hopping a barrier. The gate lowers the barrier; the Boltzmann factor says what fraction of electrons have enough energy to get over. Step through six gate voltages.",
      code: `kT = 0.02585                          # eV at 300 K
for Vg in [0, 0.06, 0.12, 0.18, 0.24, 0.30]:
    barrier = 0.5 - Vg                # the gate pulls the barrier down
    frac = exp(-barrier / kT)         # Boltzmann: fraction with enough energy
    print(Vg, log10(frac))            # rises by 1 every ~60 mV`,
      tensors: T,
      steps: [
        { title: "Six gate voltages", lines: [2], show: ["vg"], text: "<p>Raise the gate from 0 to 0.3 V in steps of 60 mV. All well below the ~0.3–0.4 V threshold of a modern transistor, so it's still “off”, but how off?</p>" },
        { title: "The gate pulls the barrier down", lines: [3], show: ["vg", "→", "phi"], speed: 600, op: op.cells(["vg", "phi"], 1, 6, (_, j) => `gate ${VG[j].toFixed(2)} V → barrier 0.5 − ${VG[j].toFixed(2)} = <b>${phi[j].toFixed(2)} eV</b>`), text: "<p>Between source and channel sits an energy barrier. In an ideal transistor the gate lowers it volt for volt (real gates lose a little to the oxide: the factor n ≈ 1.2–1.5).</p>" },
        { title: "Boltzmann: who gets over", lines: [4, 5], show: ["phi", "→", "lg"], speed: 600, op: op.cells(["phi", "lg"], 1, 6, (_, j) => `e^(−${phi[j].toFixed(2)} / 0.02585) = 10^<b>${lg[j].toFixed(2)}</b> of electrons have enough energy`), text: "<p>Electron energies follow the Boltzmann tail e^(−E/kT). The fraction that can climb a barrier of height E is e^(−E/kT). At 0.5 eV that's 10⁻⁸·⁴: tiny but not zero, which is why “off” transistors still leak.</p>" },
        { title: "Every 60 mV: × 10", lines: [5], show: ["lg", "→", "ratio"], speed: 600, op: op.cells(["lg", "ratio"], 1, 6, (_, j) => (j ? `10^(${lg[j].toFixed(2)} − (${lg[j - 1].toFixed(2)})) = <b>${ratio[j]}×</b>` : "the first step has nothing before it")), text: "<p>Each 60 mV step multiplies the current by about 10. Not because of anything about silicon: kT ln 10 = 59.5 mV at room temperature, for any material. To switch from off to on by 10⁶ you need at least 6 × 60 = 360 mV of swing, which is why supply voltages can't go much below ~0.7 V.</p>" },
        { title: "A real gate: 78 mV", lines: [3], show: ["lg", "→", "real"], text: "<p>Real transistors lose some gate voltage to the oxide, so the barrier moves by only 1/n of the gate voltage, n ≈ 1.3: 78 mV per decade. FinFETs and gate-all-around wrap the gate around the channel to push n back towards 1, but nobody beats 1 with this physics.</p>" },
      ],
    });
  },

  // -------------------------------------------------------------------------------------------
  "memory-physics"() {
    const N = 4, TL = 2;
    const A = [[1, 2, 0, 1], [0, 1, 3, 2], [2, 0, 1, 1], [1, 3, 0, 2]], B = [[2, 1, 0, 1], [0, 2, 1, 3], [1, 0, 2, 1], [3, 1, 1, 0]];
    const C = A.map((r, i) => B[0].map((_, j) => r.reduce((s, a, q) => s + a * B[q][j], 0)));
    const z = () => Array.from({ length: N }, () => Array(N).fill(0));
    const cA = z(), cB = z(), cC = Array.from({ length: N }, () => Array(N).fill(null)), tA = z(), tB = z();
    const CNT = ["#f3f1f8", "#d9d0f7", "#b7a6f0", "#9479e6", "#6b4bd8"], cnt = (v) => CNT[Math.min(4, v)];
    const T = {
      A: { data: A, label: "A in DRAM", dims: "4 × 4", kind: "ids", colors: () => "#8fd7ff", px: 6 },
      B: { data: B, label: "B in DRAM", dims: "4 × 4", kind: "ids", colors: () => "#ffc29a", px: 6 },
      C: { data: cC, label: "C = A @ B", dims: "filled as computed", kind: "ids", colors: (v) => (v == null ? "#f6f4fa" : "#5fd176"), empty: "not computed yet", px: 6 },
      cA: { data: cA, label: "loads of each A number", dims: "naive", kind: "ids", colors: cnt, px: 6 },
      cB: { data: cB, label: "loads of each B number", dims: "naive", kind: "ids", colors: cnt, px: 6 },
      tA: { data: tA, label: "loads of each A number", dims: "2 × 2 tiles", kind: "ids", colors: cnt, px: 6 },
      tB: { data: tB, label: "loads of each B number", dims: "2 × 2 tiles", kind: "ids", colors: cnt, px: 6 },
    };
    const naive = op.custom(N * N, (k) => {
      const i = Math.floor(k / N), j = k % N;
      for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
        const done = (rr, cc) => rr * N + cc <= k;
        cA[r][c] = [0, 1, 2, 3].filter((cc) => done(r, cc)).length;
        cB[r][c] = [0, 1, 2, 3].filter((rr) => done(rr, c)).length;
        cC[r][c] = done(r, c) ? C[r][c] : null;
      }
      return { hl: { A: [0, 1, 2, 3].map((q) => `${i},${q}`), B: [0, 1, 2, 3].map((q) => `${q},${j}`), C: [`${i},${j}`], cA: [0, 1, 2, 3].map((q) => `${i},${q}`), cB: [0, 1, 2, 3].map((q) => `${q},${j}`) },
        read: `C[${i},${j}] = row ${i} of A · column ${j} of B = <b>${C[i][j]}</b>: 8 numbers fetched from DRAM. Total so far: <b>${8 * (k + 1)}</b> loads for ${k + 1} output${k ? "s" : ""}` };
    }, (name, key) => { const [i, j] = key.split(",").map(Number); return name === "C" ? i * N + j : null; });
    const order = []; for (let bi = 0; bi < 2; bi++) for (let bj = 0; bj < 2; bj++) for (let kt = 0; kt < 2; kt++) order.push([bi, bj, kt]);
    const tile = (r0, c0) => [0, 1].flatMap((a) => [0, 1].map((b) => `${r0 + a},${c0 + b}`));
    const tiled = op.custom(order.length, (s) => {
      for (const m of [tA, tB]) for (const r of m) r.fill(0);
      order.slice(0, s + 1).forEach(([bi, bj, kt]) => { for (const a of [0, 1]) for (const b of [0, 1]) { tA[bi * TL + a][kt * TL + b]++; tB[kt * TL + a][bj * TL + b]++; } });
      const [bi, bj, kt] = order[s];
      return { hl: { A: tile(bi * TL, kt * TL), B: tile(kt * TL, bj * TL), C: tile(bi * TL, bj * TL), tA: tile(bi * TL, kt * TL), tB: tile(kt * TL, bj * TL) },
        read: `block for C tile (${bi}, ${bj}), step ${kt + 1} of 2: copy a 2 × 2 tile of A and of B into shared memory (8 loads), then do 2 × 2 × 2 = 8 multiply-adds from there. Total: <b>${8 * (s + 1)}</b> loads` };
    });
    return tensorWalk({
      title: "Count the trips to DRAM",
      lede: "A DRAM read costs ~170× a multiply. Multiply two 4 × 4 matrices twice: once fetching every number each time it's needed, once in 2 × 2 tiles kept in on-chip memory. Same answer, half the trips. Real tiles of 128 cut them 128×.",
      code: `# naive: every output fetches its own row and column
for i in range(4):
    for j in range(4):
        C[i][j] = sum(A[i][k] * B[k][j] for k in range(4))   # 8 DRAM loads

# tiled: a block loads 2x2 tiles once into shared memory, then reuses them
for bi, bj in tiles:
    for kt in range(2):
        a, b = load_tile(A, bi, kt), load_tile(B, kt, bj)     # 8 loads
        C_tile[bi][bj] += a @ b                              # 8 multiply-adds, no DRAM`,
      tensors: T,
      key: [["#f3f1f8", "0 loads"], ["#b7a6f0", "2"], ["#6b4bd8", "4"]],
      steps: [
        { title: "Two matrices far away", lines: [1], show: ["A", "B"], text: "<p>A and B live in DRAM: big, cheap and far. Every number fetched costs ~640 pJ at 45 nm, against ~3.7 pJ to multiply it (Horowitz). So count the fetches.</p>" },
        { title: "Naive: fetch for every output", lines: [2, 3, 4], show: ["A", "B", "→", "C", "cA", "cB"], speed: 500, op: naive, text: "<p>Each output needs a row of A and a column of B: 8 fetches. 16 outputs × 8 = <b>128 fetches</b>, and every number is fetched 4 times, once for each output that uses it. In general 2N³ fetches for N³ multiply-adds: one fetch per multiply, so the chip waits on memory.</p>" },
        { title: "Tiled: fetch once, reuse from on-chip memory", lines: [7, 8, 9, 10], show: ["A", "B", "→", "C", "tA", "tB"], speed: 900, op: tiled, text: "<p>A block of threads copies a 2 × 2 tile of A and of B into shared memory (on the SM, ~100× cheaper to read) and does all the multiply-adds that need them before moving on. Each number is now fetched only twice: <b>64 fetches</b>, half. With tile size T it's 2N³/T: real GPU kernels use T = 128–256, which is how a matmul gets from memory-bound to compute-bound.</p>" },
      ],
    });
  },

  // -------------------------------------------------------------------------------------------
  cpu() {
    const out = [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0], st = [], pred = [], hit = [], after = [];
    let s0 = 1;
    for (const o of out) { st.push(s0); const p = s0 >= 2 ? 1 : 0; pred.push(p); hit.push(p === o ? 1 : 0); s0 = o ? Math.min(3, s0 + 1) : Math.max(0, s0 - 1); after.push(s0); }
    const TN = (v) => (v ? "#22b8a0" : "#ff8a3d"), ST = (v) => ["#e9e4fb", "#c9bdf5", "#9c87ec", "#6a4fdc"][v], OK = (v) => (v ? "#5fd176" : "#ff5d5d");
    const row = (d, label, dims, colors) => ({ data: [d], label, dims, colors, kind: "ids" });
    const T = {
      out: row(out, "what the branch did", "1 = taken, 0 = not: a loop of 4", TN),
      st: row(st, "counter before", "0 … 3", ST),
      pred: row(pred, "the guess", "counter ≥ 2 → taken", TN),
      hit: row(hit, "right?", "1 = yes", OK),
      after: row(after, "counter after", "+1 if taken, −1 if not", ST),
    };
    const words = (v) => (v ? "taken" : "not taken");
    return tensorWalk({
      title: "A 2-bit branch predictor, branch by branch",
      lede: "A loop of 4 runs three times: its closing branch is taken, taken, taken, then not. One small counter per branch decides each guess. Watch it learn the pattern after one miss and then miss only the loop exits.",
      code: `state = 1                          # 0,1 = guess not taken; 2,3 = guess taken
for taken in outcomes:
    guess = state >= 2
    right = guess == taken             # wrong: ~15 clocks thrown away
    if taken: state = min(3, state + 1)
    else:     state = max(0, state - 1)`,
      tensors: T,
      key: [["#22b8a0", "taken"], ["#ff8a3d", "not taken"], ["#5fd176", "right"], ["#ff5d5d", "wrong"]],
      steps: [
        { title: "The branch's history", lines: [2], show: ["out"], text: "<p>A loop that runs 4 times ends with a branch back to its start: taken 3 times, then not taken when the loop is done. The whole loop runs 3 times, so 12 branches.</p>" },
        { title: "Guess from the counter", lines: [1, 3], show: ["st", "→", "pred"], speed: 420, op: op.cells(["st", "pred"], 1, 12, (_, j) => `counter ${st[j]} → guess <b>${words(pred[j])}</b>`), text: "<p>Each branch has a 2-bit counter: 0 and 1 mean “probably not taken”, 2 and 3 “probably taken”. It starts at 1, so the first guess is “not taken”.</p>" },
        { title: "Right or wrong?", lines: [4], show: ["pred", "out", "→", "hit"], speed: 420, op: op.cells(["pred", "out", "hit"], 1, 12, (_, j) => `guessed ${words(pred[j])}, it was ${words(out[j])}: <b>${hit[j] ? "right" : "wrong, ~15 clocks lost"}</b>`), text: "<p>The CPU ran ahead on its guess. If wrong, everything fetched after the branch is thrown away and the pipeline refills.</p>" },
        { title: "Learn from it", lines: [5, 6], show: ["st", "out", "→", "after"], speed: 420, op: op.cells(["st", "out", "after"], 1, 12, (_, j) => `${st[j]} ${out[j] ? "+ 1" : "− 1"} → <b>${after[j]}</b>`), text: "<p>Taken nudges the counter up, not taken nudges it down, stopping at 0 and 3. Because it takes two surprises in a row to flip the guess, one loop exit doesn't make it forget that the branch is usually taken.</p>" },
        { title: "The score", show: ["out", "hit"], text: `<p>${hit.filter(Boolean).length} of 12 right. After the first branch it only misses the loop exits: 1 in 4 here, 1 in 1,000 for a loop of 1,000. Real predictors use the history of recent branches too, and get ~95–99% right on typical code. Random data still defeats them: nothing to learn.</p>` },
      ],
    });
  },

  // -------------------------------------------------------------------------------------------
  languages() {
    const A = [["xor eax, eax", [0x31, 0xc0]], ["mov ecx, 1", [0xb9, 1, 0, 0, 0]], ["cmp ecx, edi", [0x39, 0xf9]], ["jg .done", [0x7f, 0x06]], ["add eax, ecx", [0x01, 0xc8]], ["inc ecx", [0xff, 0xc1]], ["jmp .loop", [0xeb, 0xf6]], ["ret", [0xc3]]];
    const owner = A.flatMap(([, b], k) => b.map(() => k)), bytes = A.flatMap(([, b]) => b);
    const hx = (b) => b.toString(16).toUpperCase().padStart(2, "0");
    const bits = bytes.map((b) => bitsOf(b, 8));
    const T = {
      asm: { data: [A.map((_, k) => k)], label: "assembly", dims: "8 instructions", kind: "ids" },
      own: { data: [owner], label: "which instruction", dims: "18 bytes", kind: "ids" },
      bytes: { data: [bytes], label: "machine code", dims: "18 bytes, 0 … 255", scale: 255, kind: "prob" },
      bits: { data: bits, label: "the bits", dims: "18 bytes × 8 wires", colors: BIT, kind: "ids", px: 5 },
    };
    return tensorWalk({
      title: "From assembly to wires, byte by byte",
      lede: "Eight lines of x86-64 that add up 1 + 2 + … + n. An assembler turns each into one to five bytes; the bytes are stored as 144 bits; each bit is a voltage on a wire. The CPU's decoder reads the same bits back as instructions.",
      code: `sum:  xor  eax, eax      ; s = 0           31 C0
      mov  ecx, 1        ; i = 1           B9 01 00 00 00
.loop:cmp  ecx, edi      ; i vs n          39 F9
      jg   .done         ; i > n: leave    7F 06
      add  eax, ecx      ; s += i          01 C8
      inc  ecx           ; i += 1          FF C1
      jmp  .loop         ;                 EB F6
.done:ret                ; result in eax   C3`,
      tensors: T,
      key: [["#5b3fd6", "1: wire at the supply voltage"], ["#e6e1f3", "0: wire at ground"]],
      steps: [
        { title: "Eight instructions", lines: [1, 8], show: ["asm"], speed: 500, op: op.cells(["asm"], 1, 8, (_, j) => `instruction ${j + 1}: <b>${A[j][0]}</b>`), text: "<p>Each line names one thing the CPU can do, and which registers to do it to. Names like <code>eax</code> are just labels for register numbers: eax is 0, ecx is 1, edi is 7.</p>" },
        { title: "Assemble: look each one up", lines: [1, 8], show: ["asm", "→", "own", "bytes"], speed: 320, op: op.cells(["own", "bytes"], 1, 18, (_, j) => `byte ${j} = <b>${hx(bytes[j])}</b>, part of <code>${A[owner[j]][0]}</code>`), text: "<p>The assembler looks up each instruction in a table: the first byte (the opcode) says what to do, the next packs which registers into fields of 2, 3 and 3 bits. Constants and jump distances follow. The whole function is 18 bytes.</p>" },
        { title: "Bytes are bits", show: ["bytes", "→", "bits"], speed: 60, op: op.cells(["bits"], 18, 8, (i, j) => `byte ${i} (${hx(bytes[i])}), bit ${7 - j}: <b>${bits[i][j]}</b>, a wire at ${bits[i][j] ? "the supply voltage" : "ground"}`), text: "<p>Stored in memory, each byte is 8 cells, each holding a high or low voltage (a charged capacitor in DRAM, a flipped latch in SRAM). 144 of them hold this program. Nothing in memory marks them as code: the same cells could hold numbers.</p>" },
        { title: "The CPU reads them back", show: ["bits", "→", "asm"], text: "<p>When the program counter points here, the decoder's logic gates look at the first byte's bits and route the rest to the right registers and the adder. That's the bottom of every language: a pattern of voltages that switches gates. Every Python program you run ends as patterns like these.</p>" },
      ],
    });
  },

  // -------------------------------------------------------------------------------------------
  "gpu-history"() {
    const A = [[1, 2, 0], [0, 1, 3], [2, 0, 1], [1, 1, 1]], B = [[2, 0, 1, 1], [1, 3, 0, 2], [0, 1, 2, 1]];
    const C = A.map((r) => B[0].map((_, j) => r.reduce((s, v, k) => s + v * B[k][j], 0)));
    const T = {
      A: { data: A, label: "texture A", dims: "4 × 3, numbers stored as pixels", scale: 3 },
      B: { data: B, label: "texture B", dims: "3 × 4", scale: 3 },
      C: { data: C, label: "the drawn image = C", dims: "4 × 4 pixels, one shader run each", scale: 12 },
    };
    return tensorWalk({
      title: "A pixel shader doing a matrix multiply (2004 style)",
      lede: "Before CUDA, the only way to run your own code on a GPU was to draw something. So store A and B as images, draw a 4 × 4 rectangle, and let the pixel shader compute each pixel's “colour” as a row of A times a column of B.",
      code: `// runs once per pixel of the output image
uniform sampler2D A, B;
void main() {
    float s = 0.0;
    for (int k = 0; k < 3; k++)
        s += A[row][k] * B[k][col];    // two texture reads, one multiply-add
    gl_FragColor = vec4(s);            // the "colour" is C[row][col]
}`,
      tensors: T,
      steps: [
        { title: "Numbers as images", lines: [2], show: ["A", "B"], text: "<p>Each number is stored in the red channel of one texel (texture pixel). Upload two images to the graphics card: that's how you got data onto a GPU in 2004.</p>" },
        { title: "Draw a rectangle; each pixel is one dot product", lines: [3, 7], show: ["A", "×", "B", "=", "C"], speed: 600, op: op.matmul(T, "A", "B", "C"), text: "<p>Draw a quad covering 4 × 4 pixels. The rasteriser launches the shader once per pixel, in parallel, and each run reads one row of A and one column of B. No pixel needs any other: the same independence that made graphics fast.</p>" },
        { title: "Read the picture back", lines: [8], show: ["C"], text: "<p>The rendered image is the product. Copy it back to the CPU (slowly, in 2004), or feed it as a texture into the next pass. CUDA (2007) kept exactly this model, one thread per output element, and dropped the pretence of drawing.</p>" },
      ],
    });
  },

  // -------------------------------------------------------------------------------------------
  circuits() {
    const t = [0, 1, 2, 3, 4, 5], r3 = (x) => +x.toFixed(3);
    const ex = t.map((k) => r3(Math.exp(-k))), V = ex.map((e) => r3(0.75 * (1 - e)));
    const sup = V.map((v) => r3(0.2 * 0.75 * v)), st = V.map((v) => r3(0.5 * 0.2 * v * v)), heat = sup.map((x, i) => r3(x - st[i]));
    const row = (a, label, dims, extra = {}) => ({ data: [a], label, dims, ...extra });
    const T = {
      t: row(t, "time, in τ", "τ = R·C = 2 ps here", { kind: "ids" }),
      ex: row(ex, "e^(−t/τ)", "how far from full", { kind: "prob", scale: 1 }),
      V: row(V, "voltage on the wire (V)", "rising to 0.75", { kind: "prob", scale: 0.75 }),
      sup: row(sup, "energy from the supply (fJ)", "C · Vdd · V(t)", { kind: "prob", scale: 0.12 }),
      st: row(st, "stored on the wire (fJ)", "½ · C · V(t)²", { kind: "prob", scale: 0.12 }),
      heat: row(heat, "turned to heat (fJ)", "the difference", { kind: "prob", scale: 0.12 }),
    };
    return tensorWalk({
      title: "Charging a wire, one time constant at a time",
      lede: "A 0.2 fF wire charged to 0.75 V through 10 kΩ. Follow the voltage and the energy books every τ.",
      code: `R, C, Vdd = 10e3, 0.2e-15, 0.75          # tau = R*C = 2 ps
for k in range(6):                         # t = 0 .. 5 tau
    V = Vdd * (1 - exp(-k))
    supplied = C * Vdd * V                 # charge delivered x supply voltage
    stored = 0.5 * C * V**2
    heat = supplied - stored               # burnt in the resistor`,
      tensors: T,
      steps: [
        { title: "Count in time constants", lines: [2], show: ["t"], text: "<p>Measure time in units of τ = R × C. Here that's 10 kΩ × 0.2 fF = 2 ps, but the shape is the same for any R and C.</p>" },
        { title: "How far from full", lines: [3], show: ["t", "→", "ex"], speed: 500, op: op.cells(["t", "ex"], 1, 6, (_, j) => `e^(−${t[j]}) = <b>${ex[j]}</b>`), text: "<p>Each τ, the gap to full shrinks to 37% of what it was. It never quite reaches zero, but after 5τ it's under 1%.</p>" },
        { title: "The voltage", lines: [3], show: ["ex", "→", "V"], speed: 500, op: op.cells(["ex", "V"], 1, 6, (_, j) => `0.75 × (1 − ${ex[j]}) = <b>${V[j]} V</b>`), text: "<p>Fast at first, then slower: 63% after 1τ, 86% after 2τ. A gate reading this wire sees a 1 somewhere around 2τ.</p>" },
        { title: "What the supply paid", lines: [4], show: ["V", "→", "sup"], speed: 500, op: op.cells(["V", "sup"], 1, 6, (_, j) => `0.2 fF × 0.75 × ${V[j]} = <b>${sup[j]} fJ</b>`), text: "<p>All the charge on the wire came from the supply at the full 0.75 V, so the supply paid C × Vdd × V(t): 0.11 fJ once the wire is full.</p>" },
        { title: "What was stored", lines: [5], show: ["V", "→", "st"], speed: 500, op: op.cells(["V", "st"], 1, 6, (_, j) => `½ × 0.2 × ${V[j]}² = <b>${st[j]} fJ</b>`), text: "<p>The wire stores only ½·C·V²: it filled up gradually, so the early charge went on at a low voltage.</p>" },
        { title: "The rest is heat", lines: [6], show: ["sup", "st", "→", "heat"], speed: 500, op: op.cells(["sup", "heat"], 1, 6, (_, j) => `${sup[j]} − ${st[j]} = <b>${heat[j]} fJ</b>`), text: "<p>At the end, half (0.056 fJ) was burnt in the resistor, whatever the resistance was. Emptying the wire later burns the other half. That's C·V² per 0 → 1 → 0, the root of P = α·C·V²·f.</p>" },
      ],
    });
  },

  // -------------------------------------------------------------------------------------------
  fab() {
    const A = [50, 100, 200, 400, 800], D = 0.1, r2 = (x) => +x.toFixed(2);
    const dpw = A.map((a) => Math.round((Math.PI * 300 ** 2) / (4 * a) - (Math.PI * 300) / Math.sqrt(2 * a)));
    const Y = A.map((a) => r2(Math.exp((-a / 100) * D))), good = dpw.map((n, i) => Math.round(n * Y[i])), cost = good.map((g) => Math.round(17000 / g));
    const row = (a, label, dims, extra = {}) => ({ data: [a], label, dims, ...extra });
    const T = {
      A: row(A, "die area (mm²)", "50 … 800", { kind: "ids" }),
      dpw: row(dpw, "dies per 300 mm wafer", "area, minus the round edge", { kind: "prob", scale: 1400 }),
      Y: row(Y, "yield", "e^(−area × 0.1 per cm²)", { kind: "prob", scale: 1 }),
      good: row(good, "working dies", "dies × yield", { kind: "prob", scale: 1400 }),
      cost: row(cost, "$ per working die", "$17,000 ÷ working dies", { kind: "prob", scale: 600 }),
    };
    return tensorWalk({
      title: "What a bigger die costs, step by step",
      lede: "The same $17,000 wafer cut into dies of five sizes, with 0.1 killer defects per cm².",
      code: `for area in [50, 100, 200, 400, 800]:                       # mm²
    dies = pi * 300**2 / (4 * area) - pi * 300 / sqrt(2 * area)
    yield_ = exp(-area / 100 * 0.1)                           # Poisson
    good = dies * yield_
    cost = 17000 / good`,
      tensors: T,
      steps: [
        { title: "Five die sizes", lines: [1], show: ["A"], text: "<p>From a small phone-chip-sized die (50 mm²) to a big GPU (800 mm², near the 858 mm² reticle limit).</p>" },
        { title: "How many fit", lines: [2], show: ["A", "→", "dpw"], speed: 500, op: op.cells(["A", "dpw"], 1, 5, (_, j) => `${A[j]} mm² → <b>${dpw[j]}</b> dies`), text: "<p>The wafer's area divided by the die's, minus the partial dies lost around the round edge. Bigger squares waste more of the edge.</p>" },
        { title: "How many survive", lines: [3], show: ["A", "→", "Y"], speed: 500, op: op.cells(["A", "Y"], 1, 5, (_, j) => `e^(−${A[j] / 100} × 0.1) = <b>${Math.round(Y[j] * 100)}%</b>`), text: "<p>If defects land at random, the chance a die has none falls exponentially with its area.</p>" },
        { title: "Working dies", lines: [4], show: ["dpw", "Y", "→", "good"], speed: 500, op: op.cells(["dpw", "good"], 1, 5, (_, j) => `${dpw[j]} × ${Y[j]} = <b>${good[j]}</b>`), text: `<p>Both effects multiply: 16 times the area gives ~${Math.round(good[0] / good[4])}× fewer working dies.</p>` },
        { title: "Cost of each", lines: [5], show: ["good", "→", "cost"], speed: 500, op: op.cells(["good", "cost"], 1, 5, (_, j) => `$17,000 ÷ ${good[j]} = <b>$${cost[j]}</b>`), text: `<p>From ~$${cost[0]} to ~$${cost[4]} for 16× the silicon. That curve is why a big AI chip is expensive even before HBM and packaging, and why designs split into chiplets.</p>` },
      ],
    });
  },

  // -------------------------------------------------------------------------------------------
  edges() {
    const kap2 = 12.77, L = [1, 2, 3, 4, 5, 6, 7, 8], d = L.map((n) => +(n * 0.3).toFixed(1)), ex = d.map((x) => +(kap2 * x).toFixed(1)), lg = ex.map((x) => +(-x / Math.LN10).toFixed(1));
    const row = (a, label, dims, extra = {}) => ({ data: [a], label, dims, ...extra });
    const T = {
      L: row(L, "atomic layers", "1 … 8", { kind: "ids" }),
      d: row(d, "thickness (nm)", "≈ 0.3 nm a layer", { scale: 2.4 }),
      ex: row(ex, "2κd", "κ ≈ 6.4 per nm for SiO₂", { scale: 31 }),
      lg: row(lg, "log₁₀ of the leak", "how many factors of 10 fewer get through", { scale: 13.5 }),
    };
    return tensorWalk({
      title: "Tunnelling through a gate oxide, one atomic layer at a time",
      lede: "Why a single layer of atoms decides whether a transistor leaks. An electron's wave dies off inside the oxide as e^(−κx); what's left at the far side is the chance it gets through.",
      code: `kappa = sqrt(2 * m_eff * barrier) / hbar   # SiO2: 3.1 eV, m_eff = 0.5 m_e → 6.4 per nm
for layers in range(1, 9):
    d = 0.3 * layers                        # nm
    leak = exp(-2 * kappa * d)              # chance an electron tunnels through
    print(layers, d, log10(leak))`,
      tensors: T,
      steps: [
        { title: "Count the layers", lines: [2], show: ["L"], text: "<p>A gate oxide is a few atomic layers of silicon dioxide, each roughly 0.3 nm. In 2005 it was about four layers.</p>" },
        { title: "Layers to nanometres", lines: [3], show: ["L", "→", "d"], speed: 500, op: op.cells(["L", "d"], 1, 8, (_, j) => `${L[j]} layer${j ? "s" : ""} ≈ <b>${d[j]} nm</b>`), text: "<p>Thickness is just layers × 0.3 nm. There is nothing in between: you can't add half a layer.</p>" },
        { title: "How fast the wave dies", lines: [1, 4], show: ["d", "→", "ex"], speed: 500, op: op.cells(["d", "ex"], 1, 8, (_, j) => `2κd = 12.8 × ${d[j]} = <b>${ex[j]}</b>`), text: "<p>κ comes from the barrier height (3.1 eV) and the electron's effective mass: about 6.4 per nm. The leak is e<sup>−2κd</sup>, so what matters is 2κd.</p>" },
        { title: "The leak", lines: [4, 5], show: ["ex", "→", "lg"], speed: 500, op: op.cells(["ex", "lg"], 1, 8, (_, j) => `e^(−${ex[j]}) = <b>10^${lg[j]}</b>: ${lg[j] > -6 ? "a serious leak" : lg[j] > -12 ? "tolerable" : "negligible"}`), text: "<p>Each extra layer cuts the leak by about 50×. Four layers (1.2 nm) leak about one electron in a million per attempt, and electrons attempt ~10¹⁴ times a second: that was the gate leakage crisis of the mid-2000s, solved by switching to thicker hafnium oxide in 2007.</p>" },
      ],
    });
  },

  // -------------------------------------------------------------------------------------------
  "power-wall"() {
    const k = 1.4, G = [0, 1, 2, 3, 4], r2 = (x) => +x.toFixed(2);
    const L = G.map((g) => r2(k ** -g)), V = L.slice(), f = G.map((g) => r2(k ** g)), ppt = G.map((g) => r2(k ** (-2 * g)));
    const dens = G.map((g) => r2(k ** (2 * g))), pd = G.map(() => 1), pd2 = dens.slice(), pd3 = G.map((g) => r2(k ** g));
    const row = (a, label, dims, sc) => ({ data: [a], label, dims, kind: "prob", scale: sc });
    const T = {
      L: row(L, "size of a transistor", "generations 0 … 4, each × 0.71", 1),
      V: row(V, "voltage (Dennard)", "falls with size", 1),
      f: row(f, "clock (Dennard)", "rises 1.4× a generation", 4),
      ppt: row(ppt, "power per transistor", "C · V² · f", 1),
      dens: row(dens, "transistors per mm²", "2× a generation", 15),
      pd: row(pd, "power per mm² (Dennard)", "constant!", 15),
      pd2: row(pd2, "power per mm², voltage stuck", "if clocks kept rising", 15),
      pd3: row(pd3, "power per mm², clocks frozen too", "still rising", 15),
    };
    const cell = (names, txt) => op.cells(names, 1, 5, (_, j) => txt(j));
    return tensorWalk({
      title: "Dennard scaling, generation by generation",
      lede: "From 1974 to ~2005, every shrink made transistors faster and cooler at the same time. Then the voltage got stuck. Step through five generations and watch power per mm² go from flat to climbing.",
      code: `k = 1.4                                    # one generation: every length × 1/1.4
for g in range(5):
    L, V, C = k**-g, k**-g, k**-g          # Dennard: shrink everything, voltage too
    f = k**g                               # smaller switches are faster
    power = C * V**2 * f                   # per transistor: k**-2g
    density = k**(2*g)                     # transistors per mm²
    print(g, power * density)              # 1.0 every generation
# after ~2005 the voltage stopped falling: V = 1`,
      tensors: T,
      steps: [
        { title: "Shrink everything", lines: [1, 3], show: ["L"], text: "<p>Each process generation scales every length by ~0.7 (1/1.4), so a transistor's area halves. Five generations is about 10 years of Moore's law.</p>" },
        { title: "Dennard's rules: lower voltage, faster clock", lines: [3, 4], show: ["L", "V", "f"], speed: 700, op: cell(["L", "V", "f"], (j) => `generation ${j}: size ${L[j]}, voltage ${V[j]}, clock <b>${f[j]}×</b>`), text: "<p>Robert Dennard's 1974 rule: scale the voltage down with the size, and the electric fields inside stay the same, so the transistor works the same, only smaller and faster.</p>" },
        { title: "Power per transistor = C · V² · f", lines: [5], show: ["V", "f", "→", "ppt"], speed: 700, op: cell(["V", "f", "ppt"], (j) => `C ${L[j]} × V² ${r2(V[j] ** 2)} × f ${f[j]} = <b>${ppt[j]}</b>`), text: "<p>Smaller capacitance, lower voltage squared, faster clock: each transistor uses half the power each generation.</p>" },
        { title: "Twice the transistors, same heat per mm²", lines: [6, 7], show: ["ppt", "×", "dens", "=", "pd"], speed: 700, op: cell(["ppt", "dens", "pd"], (j) => `${ppt[j]} × ${dens[j]} = <b>${r2(ppt[j] * dens[j])}</b> per mm²`), text: "<p>Half the power each, twice as many: power per mm² stays flat. Faster chips with more transistors, at the same heat, every two years. That was the free lunch.</p>" },
        { title: "After 2005: the voltage gets stuck", lines: [8], show: ["dens", "→", "pd2", "pd3"], speed: 700, op: cell(["dens", "pd2", "pd3"], (j) => `generation ${j}: ${pd2[j]}× the heat per mm² if clocks keep rising, <b>${pd3[j]}×</b> even with clocks frozen`), text: "<p>Leakage (the Boltzmann tail from <a href='/learn?c=transistor'>the transistor chapter</a>) stopped the threshold, and so the voltage, from falling. With V fixed, power per mm² climbs with density: 15× in four generations if clocks kept rising, still ~4× with clocks frozen. So clocks froze, and the extra transistors went into more cores, specialised units, or stayed switched off. The GPU is the answer that turned this into throughput.</p>" },
      ],
    });
  },

  // -------------------------------------------------------------------------------------------
  "software-stack"() {
    const x = [[-1, 0.5, -0.25, 1], [0.75, -0.75, 0.25, -1.25], [1.5, -0.5, 0, 0.5], [-2, 1, -0.5, 0.25]];
    const y = x.map((r) => r.map((v) => v * 2)), z = y.map((r) => r.map((v) => v + 1)), o = z.map((r) => r.map((v) => Math.max(0, v)));
    const T = {
      x: { data: x, label: "x in HBM", dims: "4 × 4", scale: 3 },
      y: { data: y, label: "y = x × 2 (in HBM)", dims: "kernel 1 writes it", scale: 3 },
      z: { data: z, label: "z = y + 1 (in HBM)", dims: "kernel 2 writes it", scale: 3 },
      o: { data: o, label: "out = relu(z)", dims: "kernel 3 writes it", scale: 3 },
      f: { data: o, label: "out, fused", dims: "one kernel, registers in between", scale: 3 },
    };
    const f2 = (v) => (v < 0 ? `(${v})` : String(v));
    const k = (a, b, what) => op.cells([a, b], 4, 4, (i, j) => `${what(i, j)}: <b>2 HBM trips</b> (load one, store one) for this element`);
    return tensorWalk({
      title: "Fusion: three kernels become one",
      lede: "Eager PyTorch runs relu(x * 2 + 1) as three kernels, each reading its input from HBM and writing its output back. A compiler that sees all three writes one kernel that keeps the in-between values in registers.",
      code: `# eager: three kernels, three round trips through HBM
y = x * 2
z = y + 1
out = torch.relu(z)

# torch.compile: one generated Triton kernel
@triton.jit
def fused(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i, mask=i < n)                        # 1 read
    tl.store(out_ptr + i, tl.maximum(x * 2 + 1, 0), mask=i < n)   # 1 write`,
      tensors: T,
      steps: [
        { title: "The input sits in HBM", lines: [1], show: ["x"], text: "<p>16 numbers in GPU memory. On a real model this would be millions, and HBM bandwidth, not arithmetic, would set the time.</p>" },
        { title: "Kernel 1: × 2", lines: [2], show: ["x", "→", "y"], speed: 220, op: k("x", "y", (i, j) => `load x = ${f2(x[i][j])}, × 2 = ${f2(y[i][j])}, store y`), text: "<p>Launch one kernel. Every thread loads one x from HBM, multiplies by 2, stores y back to HBM. 32 HBM accesses for 16 multiplies.</p>" },
        { title: "Kernel 2: + 1", lines: [3], show: ["y", "→", "z"], speed: 220, op: k("y", "z", (i, j) => `load y = ${f2(y[i][j])}, + 1 = ${f2(z[i][j])}, store z`), text: "<p>Another launch. It reads back the y that kernel 1 just wrote: the data goes out to HBM and straight back in.</p>" },
        { title: "Kernel 3: ReLU", lines: [4], show: ["z", "→", "o"], speed: 220, op: k("z", "o", (i, j) => `load z = ${f2(z[i][j])}, relu = ${o[i][j]}, store out`), text: "<p>Third launch. Total: <b>3 launches and 96 HBM accesses</b> for 48 trivial operations. Each one is cheap; the traffic is the cost.</p>" },
        { title: "Fused: one kernel, values stay in registers", lines: [6, 7, 8, 9, 10, 11], show: ["x", "→", "f"], speed: 220, op: op.cells(["x", "f"], 4, 4, (i, j) => `load x = ${f2(x[i][j])} → × 2 = ${f2(y[i][j])} → + 1 = ${f2(z[i][j])} → relu = ${o[i][j]} (all in one register) → store: <b>2 HBM trips</b> instead of 6`), text: "<p>Same answer, <b>1 launch and 32 HBM accesses</b>: 3× less traffic. With 8 operations in the chain it would be 8×. This is most of what torch.compile buys on elementwise code, and it's the same idea as tiling: keep data close and use it as much as possible before it goes back.</p>" },
      ],
    });
  },
};

