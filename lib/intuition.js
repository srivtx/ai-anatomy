// Random hyperparameter search on a small network, then: which knob explains the variance in the result?
import { MLP, dataset, bceLogit } from "./nn.js";
import { rng } from "./core.js";

export const KNOBS = [
  { key: "lr", label: "Learning rate", lo: -4, hi: -0.5, log: true, fmt: (v) => v.toExponential(0) },
  { key: "width", label: "Width", lo: 4, hi: 48, int: true, fmt: (v) => v },
  { key: "batch", label: "Batch size", lo: 3, hi: 7, pow2: true, fmt: (v) => v },
  { key: "init", label: "Init scale", lo: 0.5, hi: 2, fmt: (v) => v.toFixed(2) },
  { key: "wd", label: "Weight decay", lo: -6, hi: -2, log: true, fmt: (v) => v.toExponential(0) },
];
export function sample(r) {
  const c = {};
  for (const k of KNOBS) {
    const u = k.lo + (k.hi - k.lo) * r();
    c[k.key] = k.log ? 10 ** u : k.pow2 ? 2 ** Math.round(u) : k.int ? Math.round(u) : u;
  }
  return c;
}
const SETS = {};
const get = (kind) => (SETS[kind] ||= [dataset(kind, 240, { noise: 0.08, seed: 5 }), dataset(kind, 240, { noise: 0.08, seed: 6 })]);
/** Train for a fixed number of examples seen (so batch size changes the number of steps, as in practice). */
export function trial(c, seed, kind = "moons", examples = 12000) {
  const [DATA, TEST] = get(kind);
  const m = new MLP([2, c.width, c.width, 1], { act: "tanh", seed, initScale: c.init }), r = rng(seed);
  const steps = Math.round(examples / c.batch);
  for (let s = 0; s < steps; s++) {
    for (let i = 0; i < c.batch; i++) { const p = DATA[Math.floor(r() * DATA.length)]; m.backward([bceLogit(m.forward(p.x)[0], p.y).d]); }
    m.step(c.lr, c.batch, { wd: c.wd });
  }
  let L = 0; for (const p of TEST) L += bceLogit(m.forward(p.x)[0], p.y).loss;
  return Math.min(L / TEST.length, 1.2);
}
/** Share of variance in the outcome explained by each knob: variance of bin means / total (a one-way ANOVA). */
export function importance(trials, bins = 5) {
  const ys = trials.map((t) => t.loss), mu = ys.reduce((a, b) => a + b, 0) / ys.length;
  const tot = ys.reduce((a, b) => a + (b - mu) ** 2, 0) / ys.length || 1;
  return KNOBS.map((k) => {
    const xs = trials.map((t) => (k.log ? Math.log10(t.c[k.key]) : t.c[k.key]));
    const order = xs.map((x, i) => [x, i]).sort((a, b) => a[0] - b[0]);
    let between = 0;
    for (let b = 0; b < bins; b++) {
      const sl = order.slice(Math.floor((b * order.length) / bins), Math.floor(((b + 1) * order.length) / bins));
      if (!sl.length) continue;
      const m = sl.reduce((a, [, i]) => a + ys[i], 0) / sl.length; between += (sl.length / ys.length) * (m - mu) ** 2;
    }
    return between / tot;
  });
}
