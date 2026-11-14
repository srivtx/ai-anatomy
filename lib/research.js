// Small, real experiments for the research-method chapter: learning-rate sweeps
// and ablation grids on a toy 2D task. Pure maths so it runs in node too.
import { MLP, bceLogit, dataset, gauss } from "./nn.js";
import { rng } from "./core.js";

export const DATA = { train: dataset("spiral", 240, { noise: 0.08, seed: 21 }), test: dataset("spiral", 800, { noise: 0.08, seed: 121 }) };

/**
 * One training run. cfg: { width, act, lr, wd, aug, steps, seed }
 * aug: std of Gaussian jitter added to inputs each step (a data augmentation)
 */
export function run(cfg, data = DATA) {
  const { width = 16, act = "tanh", lr = 0.01, wd = 0, aug = 0, steps = 500, seed = 1, batch = 24 } = cfg;
  const m = new MLP([2, width, width, 1], { act, seed });
  const r = rng(seed * 7919 + 3);
  const n = data.train.length;
  for (let s = 0; s < steps; s++) {
    for (let b = 0; b < batch; b++) {
      const p = data.train[Math.floor(r() * n)];
      const x = aug ? [p.x[0] + gauss(r) * aug, p.x[1] + gauss(r) * aug] : p.x;
      m.backward([bceLogit(m.forward(x)[0], p.y).d]);
    }
    m.step(lr, batch, { wd });
  }
  return { model: m, acc: accuracy(m, data.test) };
}
export const accuracy = (m, set) => set.filter((p) => (m.forward(p.x)[0] > 0) === !!p.y).length / set.length;

export const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
export const sd = (a) => { const m = mean(a); return Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / Math.max(1, a.length - 1)); };
