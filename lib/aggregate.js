// How a reward model aggregates disagreeing people. Bradley–Terry on pooled comparisons ≈ Borda count.
export const RESPONSES = ["Direct answer", "Answer + caveats", "Asks to clarify", "Politely declines"];
export const GROUPS = [
  { name: "Experts", u: [9, 6, 3, 0], color: "var(--c1)" },
  { name: "Cautious", u: [2, 8, 6, 7], color: "var(--c3)" },
  { name: "Casual", u: [8, 7, 1, 2], color: "var(--c4)" },
];
const sig = (x) => 1 / (1 + Math.exp(-x));
export function aggregate(shares, T = 1, groups = GROUPS) {
  const tot = shares.reduce((a, b) => a + b, 0) || 1, w = shares.map((s) => s / tot), n = groups[0].u.length;
  const P = Array.from({ length: n }, (_, a) => Array.from({ length: n }, (_, b) => a === b ? 0.5 : groups.reduce((s, g, k) => s + w[k] * sig((g.u[a] - g.u[b]) / T), 0)));
  // Bradley–Terry maximum likelihood on the pooled win rates
  const r = new Array(n).fill(0);
  for (let it = 0; it < 2000; it++) {
    const g = new Array(n).fill(0);
    for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) if (a !== b) { const d = P[a][b] - sig(r[a] - r[b]); g[a] += d; g[b] -= d; }
    for (let a = 0; a < n; a++) r[a] += 0.1 * g[a];
  }
  const mu = r.reduce((a, b) => a + b, 0) / n; for (let a = 0; a < n; a++) r[a] -= mu;
  const borda = P.map((row, a) => row.reduce((s, v, b) => s + (a === b ? 0 : v), 0) / (n - 1));
  const util = Array.from({ length: n }, (_, a) => groups.reduce((s, g, k) => s + w[k] * g.u[a], 0));
  const plural = new Array(n).fill(0); groups.forEach((g, k) => { plural[g.u.indexOf(Math.max(...g.u))] += w[k]; });
  const condorcet = P.findIndex((row, a) => row.every((v, b) => a === b || v > 0.5));
  const best = (v) => v.indexOf(Math.max(...v));
  return { P, r, borda, util, plural, condorcet, w, win: { bt: best(r), borda: best(borda), util: best(util), plural: best(plural), condorcet } };
}
