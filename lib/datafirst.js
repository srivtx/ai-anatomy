// The "data vs model" experiment: noisy labels, then compare a bigger model against cleaner data.
// Pure maths so it can be tested in node.
import { MLP, bceLogit, dataset } from "./nn.js";
import { rng } from "./core.js";

export function makeData(noiseFrac, seed = 3) {
  const train = dataset("spiral", 400, { noise: 0.06, seed }).map((p) => ({ ...p, clean: p.y }));
  const test = dataset("spiral", 600, { noise: 0.06, seed: seed + 100 });
  const r = rng(seed + 7);
  // flip a fraction of training labels (annotator mistakes)
  for (const p of train) if (r() < noiseFrac) p.y = 1 - p.y;
  return { train, test };
}

export const ARCH = { small: [2, 24, 24, 1], big: [2, 64, 64, 1] };

/** A trainer that can be advanced in chunks. */
export function trainer(data, arch, { seed = 1, lr = 0.01, epochs = 250 } = {}) {
  const m = new MLP(arch, { act: "tanh", seed });
  const r = rng(seed + 11);
  let ep = 0;
  return {
    model: m,
    get epoch() { return ep; },
    get done() { return ep >= epochs; },
    run(k = 5) {
      for (let e = 0; e < k && ep < epochs; e++, ep++) {
        const idx = data.map((_, i) => i).sort(() => r() - 0.5);
        for (let j = 0; j < idx.length; j++) { const p = data[idx[j]], z = m.forward(p.x)[0]; m.backward([bceLogit(z, p.y).d]); if (j % 16 === 15) m.step(lr, 16); }
      }
    },
  };
}
export const acc = (m, set) => set.filter((p) => (m.forward(p.x)[0] > 0) === !!p.y).length / set.length;

/** Cross-fitted probability that each training label is right (low = suspicious). Returns points sorted most suspicious first. */
export function suspicion(train, arch, { epochs = 150 } = {}) {
  const halves = [train.filter((_, i) => i % 4 < 2), train.filter((_, i) => i % 4 >= 2)];
  const scored = [];
  halves.forEach((held, h) => {
    const t = trainer(halves[1 - h], arch, { seed: 5 + h, epochs });
    t.run(epochs);
    for (const p of held) { const pz = 1 / (1 + Math.exp(-t.model.forward(p.x)[0])); scored.push([p.y ? pz : 1 - pz, p]); }
  });
  return scored.sort((a, b) => a[0] - b[0]).map(([s, p]) => ({ p, s }));
}
