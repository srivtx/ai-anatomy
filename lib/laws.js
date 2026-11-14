// Training dynamics you can measure: sharpness (top Hessian eigenvalue) and the gradient noise scale.
import { MLP, dataset } from "./nn.js";

export function flat(m) {
  const n = m.W.reduce((a, w, i) => a + w.length + m.b[i].length, 0), v = new Float64Array(n);
  let k = 0; for (let l = 0; l < m.W.length; l++) { v.set(m.W[l], k); k += m.W[l].length; v.set(m.b[l], k); k += m.b[l].length; }
  return v;
}
export function setFlat(m, v) {
  let k = 0; for (let l = 0; l < m.W.length; l++) { m.W[l].set(v.subarray(k, k + m.W[l].length)); k += m.W[l].length; m.b[l].set(v.subarray(k, k + m.b[l].length)); k += m.b[l].length; }
}
function gradFlat(m) {
  const n = m.gW.reduce((a, w, i) => a + w.length + m.gb[i].length, 0), v = new Float64Array(n);
  let k = 0; for (let l = 0; l < m.W.length; l++) { v.set(m.gW[l], k); k += m.gW[l].length; v.set(m.gb[l], k); k += m.gb[l].length; }
  m.zeroGrad(); return v;
}
/** Full-batch MSE loss and gradient (flat). Targets are ±1. */
export function lossGrad(m, data) {
  let L = 0;
  for (const p of data) { const y = m.forward(p.x)[0], e = y - (p.y ? 1 : -1); L += e * e; m.backward([(2 * e) / data.length]); }
  return { loss: L / data.length, g: gradFlat(m) };
}
/** Per-example gradients, for the noise scale. */
function exampleGrads(m, data) {
  return data.map((p) => { const y = m.forward(p.x)[0]; m.backward([2 * (y - (p.y ? 1 : -1))]); return gradFlat(m); });
}
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const norm = (a) => Math.sqrt(dot(a, a));

/** Train a small tanh MLP with full-batch gradient descent and watch its sharpness. */
export function eos({ lr = 0.05, seed = 1, kind = "moons", n = 80, width = 24, warmup = 0 } = {}) {
  const data = dataset(kind, n, { noise: 0.12, seed: 5 });
  const m = new MLP([2, width, width, 1], { act: "tanh", seed });
  let v = null, step = 0;
  /** Power iteration on finite-difference Hessian-vector products, warm-started. */
  function sharpness(iters = 8) {
    const w = flat(m), eps = 1e-3;
    if (!v) { v = new Float64Array(w.length).map(() => Math.random() - 0.5); }
    let lam = 0;
    for (let t = 0; t < iters; t++) {
      const nv = norm(v); for (let i = 0; i < v.length; i++) v[i] /= nv;
      const wp = w.map((x, i) => x + eps * v[i]), wm = w.map((x, i) => x - eps * v[i]);
      setFlat(m, wp); const gp = lossGrad(m, data).g; setFlat(m, wm); const gm = lossGrad(m, data).g;
      const Hv = gp.map((x, i) => (x - gm[i]) / (2 * eps));
      lam = dot(Hv, v); v = Hv;
    }
    setFlat(m, w); return lam;
  }
  function noiseScale() {
    const gs = exampleGrads(m, data), G = new Float64Array(gs[0].length);
    for (const g of gs) for (let i = 0; i < G.length; i++) G[i] += g[i] / gs.length;
    let tr = 0; for (const g of gs) for (let i = 0; i < G.length; i++) tr += (g[i] - G[i]) ** 2;
    tr /= gs.length - 1;
    return tr / dot(G, G);           // B_simple = tr(Σ) / |G|²  (McCandlish et al., 2018)
  }
  return {
    data, model: m, get step() { return step; }, lr, eta: () => (warmup ? lr * Math.min(1, step / warmup) : lr),
    train(k = 1) {
      let loss;
      for (let t = 0; t < k; t++) {
        const r = lossGrad(m, data); loss = r.loss;
        const eta = warmup ? lr * Math.min(1, (step + 1) / warmup) : lr;
        const w = flat(m); for (let i = 0; i < w.length; i++) w[i] -= eta * r.g[i]; setFlat(m, w); step++;
      }
      return loss;
    },
    loss() { return lossGrad(m, data).loss; },
    sharpness, noiseScale,
    acc() { return data.filter((p) => (m.forward(p.x)[0] > 0) === !!p.y).length / data.length; },
  };
}
