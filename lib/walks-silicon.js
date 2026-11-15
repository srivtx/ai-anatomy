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
};
