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
};
