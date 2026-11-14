// In-context learning = gradient descent (von Oswald et al., 2023), in its simplest exact form.
// Task: y = aᵀx with a ~ N(0, I), x ∈ R^D. Tokens e = (x, y); the query token is (x_q, 0).
// One linear self-attention layer predicts ŷ = (1/N) Σᵢ (wᵀeᵢ)(eᵢᵀ A e_q).
// One GD step from zero on the in-context points predicts ŷ = (η/N) Σᵢ yᵢ xᵢᵀx_q,
// which is the layer with w = (0, …, 0, 1) and A = η·[[I, 0], [0, 0]].
import { rng } from "./core.js";
import { gauss } from "./nn.js";

export const D = 2, E = D + 1;
export function task(r, N = 16) {
  const a = Array.from({ length: D }, () => gauss(r)), pts = [];
  const mk = () => Array.from({ length: D }, () => gauss(r));
  for (let i = 0; i < N; i++) { const x = mk(); pts.push([...x, x.reduce((s, v, k) => s + v * a[k], 0)]); }
  const xq = mk();
  return { a, pts, q: [...xq, 0], yq: xq.reduce((s, v, k) => s + v * a[k], 0) };
}
export function predict(P, pts, q) {
  let s = 0;
  for (const e of pts) {
    let v = 0; for (let i = 0; i < E; i++) v += P.w[i] * e[i];
    let k = 0; for (let i = 0; i < E; i++) for (let j = 0; j < E; j++) k += e[i] * P.A[i * E + j] * q[j];
    s += v * k;
  }
  return s / pts.length;
}
export function gdPredict(pts, q, eta, steps = 1) {
  let w = new Array(D).fill(0);
  for (let t = 0; t < steps; t++) {
    const g = new Array(D).fill(0);
    for (const e of pts) { let err = -e[D]; for (let k = 0; k < D; k++) err += w[k] * e[k]; for (let k = 0; k < D; k++) g[k] += err * e[k]; }
    w = w.map((v, k) => v - (eta * g[k]) / pts.length);
  }
  return w.reduce((s, v, k) => s + v * q[k], 0);
}
export function learner(seed = 1) {
  const r = rng(seed);
  const P = { w: Array.from({ length: E }, () => 0.01 * gauss(r)), A: Array.from({ length: E * E }, () => 0.01 * gauss(r)) };
  let step = 0; const m = { w: new Array(E).fill(0), A: new Array(E * E).fill(0) }, v = { w: new Array(E).fill(0), A: new Array(E * E).fill(0) };
  function train(k = 1, bs = 16, lr = 0.003) {
    let L = 0;
    for (let t = 0; t < k; t++) {
      const gw = new Array(E).fill(0), gA = new Array(E * E).fill(0);
      for (let b = 0; b < bs; b++) {
        const T = task(r), yh = predict(P, T.pts, T.q), d = (2 * (yh - T.yq)) / bs; L += (yh - T.yq) ** 2 / bs;
        for (const e of T.pts) {
          let val = 0; for (let i = 0; i < E; i++) val += P.w[i] * e[i];
          let kk = 0; for (let i = 0; i < E; i++) for (let j = 0; j < E; j++) kk += e[i] * P.A[i * E + j] * T.q[j];
          for (let i = 0; i < E; i++) gw[i] += (d * e[i] * kk) / T.pts.length;
          for (let i = 0; i < E; i++) for (let j = 0; j < E; j++) gA[i * E + j] += (d * val * e[i] * T.q[j]) / T.pts.length;
        }
      }
      step++;
      for (const [p, g, mm, vv] of [[P.w, gw, m.w, v.w], [P.A, gA, m.A, v.A]]) for (let i = 0; i < p.length; i++) {
        mm[i] = 0.9 * mm[i] + 0.1 * g[i]; vv[i] = 0.999 * vv[i] + 0.001 * g[i] * g[i];
        p[i] -= (lr * mm[i] / (1 - 0.9 ** step)) / (Math.sqrt(vv[i] / (1 - 0.999 ** step)) + 1e-8);
      }
    }
    return L / k;
  }
  /** The combined matrix M[i][j] = w_y · A[i][j] (what the layer effectively computes, scale-free), the
   *  learning rate η it implements, and how much of its output 1-step GD at that η explains. */
  function asGD(trials = 300) {
    const rr = rng(99), tasks = Array.from({ length: trials }, () => task(rr));
    let num = 0, den = 0; for (const T of tasks) { const a = predict(P, T.pts, T.q), g = gdPredict(T.pts, T.q, 1); num += a * g; den += g * g; }
    const eta = num / den; let err = 0, base = 0; const pairs = [];
    for (const T of tasks) { const a = predict(P, T.pts, T.q), g = gdPredict(T.pts, T.q, eta); err += (a - g) ** 2; base += a * a; pairs.push([g, a, T.yq]); }
    return { eta, match: 1 - err / (base || 1), pairs };
  }
  return { P, train, asGD, get step() { return step; } };
}
