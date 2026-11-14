// A tiny ARC-style world: grids of colours 0..6, a handful of primitives, and a
// brute-force search for a program consistent with every training pair.
const clone = (g) => g.map((r) => [...r]);
export const PRIMS = {
  identity: (g) => clone(g),
  flipH: (g) => g.map((r) => [...r].reverse()),
  flipV: (g) => clone(g).reverse(),
  rot90: (g) => g[0].map((_, j) => g.map((r) => r[j]).reverse()),
  rot180: (g) => clone(g).reverse().map((r) => r.reverse()),
  transpose: (g) => g[0].map((_, j) => g.map((r) => r[j])),
  gravity: (g) => {
    const H = g.length, out = g.map((r) => r.map(() => 0));
    for (let j = 0; j < g[0].length; j++) { const col = g.map((r) => r[j]).filter((v) => v); col.forEach((v, i) => (out[H - col.length + i][j] = v)); }
    return out;
  },
  crop: (g) => {
    let r0 = g.length, r1 = -1, c0 = g[0].length, c1 = -1;
    g.forEach((r, i) => r.forEach((v, j) => { if (v) { r0 = Math.min(r0, i); r1 = Math.max(r1, i); c0 = Math.min(c0, j); c1 = Math.max(c1, j); } }));
    return r1 < 0 ? clone(g) : g.slice(r0, r1 + 1).map((r) => r.slice(c0, c1 + 1));
  },
  scale2: (g) => g.flatMap((r) => { const rr = r.flatMap((v) => [v, v]); return [rr, [...rr]]; }),
  tile2: (g) => { const w = g.map((r) => [...r, ...r]); return [...w, ...clone(w)]; },
  outline: (g) => g.map((r, i) => r.map((v, j) => (v && [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]].every(([a, b]) => g[a]?.[b]) ? 0 : v))),
};
const same = (a, b) => a.length === b.length && a.every((r, i) => r.length === b[i].length && r.every((v, j) => v === b[i][j]));
/** A colour map m such that m(a) == b cell by cell for every pair, if one exists. */
function colourMap(pairs) {
  const m = new Map();
  for (const [a, b] of pairs) {
    if (a.length !== b.length || a[0].length !== b[0].length) return null;
    for (let i = 0; i < a.length; i++) for (let j = 0; j < a[0].length; j++) {
      const k = a[i][j];
      if (m.has(k) && m.get(k) !== b[i][j]) return null;
      m.set(k, b[i][j]);
    }
  }
  return m;
}
/** Search programs of up to `depth` primitives, each optionally followed by a learned recolouring. */
export function search(train, depth = 2) {
  const names = Object.keys(PRIMS);
  let tried = 0;
  const progs = [[]];
  for (let d = 1; d <= depth; d++) for (const p of progs.filter((q) => q.length === d - 1)) for (const n of names) if (n !== "identity" || d === 1) progs.push([...p, n]);
  for (const prog of progs.slice(1)) {
    tried++;
    const run = (g) => prog.reduce((x, n) => PRIMS[n](x), g);
    const outs = train.map(([i]) => run(i));
    if (outs.every((o, k) => same(o, train[k][1]))) return { prog, tried, apply: run };
    const m = colourMap(outs.map((o, k) => [o, train[k][1]]));
    if (m && [...m].some(([a, b]) => a !== b)) {
      const recol = (g) => g.map((r) => r.map((v) => (m.has(v) ? m.get(v) : v)));
      return { prog: [...prog, `recolour(${[...m].filter(([a, b]) => a !== b).map(([a, b]) => `${a}→${b}`).join(", ")})`], tried, apply: (g) => recol(run(g)) };
    }
  }
  return { prog: null, tried };
}
export { same };

const G = (s) => s.trim().split("/").map((r) => [...r].map(Number));
export const PUZZLES = [
  { name: "Mirror", train: [[G("120/300/000"), G("021/003/000")], [G("4400/0050/0000"), G("0044/0500/0000")]], test: [G("1023/0040/5000"), G("3201/0400/0005")] },
  { name: "Things fall", train: [[G("110/000/202"), G("000/100/212")], [G("0302/0000/1000/0000"), G("0000/0000/0000/1302")]], test: [G("2020/0101/0000/3000"), G("0000/0000/2000/3121")] },
  { name: "Crop the shape", train: [[G("00000/02200/02000/00000"), G("22/20")], [G("000000/000330/000030/000330/000000"), G("33/03/33")]], test: [G("00000/00000/04440/00400/00000"), G("444/040")] },
  { name: "Recolour and flip", train: [[G("100/120/000"), G("000/340/300")], [G("011/002/000"), G("000/004/033")]], test: [G("110/200/100"), G("300/400/330")] },
  { name: "Zoom in", train: [[G("12/30"), G("1122/1122/3300/3300")], [G("05/50"), G("0055/0055/5500/5500")]], test: [G("40/26"), G("4400/4400/2266/2266")] },
  { name: "Count the dots", train: [[G("10000/00010/00000/01000"), G("111")], [G("00000/01000/00000/00000"), G("1")], [G("10001/00000/10010/00000"), G("1111")]], test: [G("01000/00000/00010"), G("11")], note: "Needs counting, which isn't in the toolbox." },
];
