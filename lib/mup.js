// A μP coordinate check: train a 3-layer MLP of several widths for a few Adam steps
// under standard parametrisation (SP) or μP, and measure how much the hidden
// activations and outputs move. In SP the output update grows with width; in μP it doesn't.
import { rng } from "./core.js";
import { gauss } from "./nn.js";

const D_IN = 16, BASE = 64;

/** param: "sp" | "mup". Returns RMS of output logits and RMS change of last hidden activations after `steps`. */
export function coordCheck(width, param, { steps = 4, lr = 0.01, batch = 16, seed = 1 } = {}) {
  const r = rng(seed), n = width, m = width / BASE;
  const Win = Float64Array.from({ length: D_IN * n }, () => gauss(r) / Math.sqrt(D_IN));
  const Wh = Float64Array.from({ length: n * n }, () => gauss(r) / Math.sqrt(n));
  // μP: output layer initialised with variance 1/n² (std 1/n) instead of 1/n
  const Wout = Float64Array.from({ length: n }, () => gauss(r) / (param === "mup" ? n / Math.sqrt(BASE) : Math.sqrt(n)));
  // μP with Adam: hidden and output learning rates scale as 1/m; input layer keeps the base LR
  const LR = param === "mup" ? [lr, lr / m, lr / m] : [lr, lr, lr];
  const P = [Win, Wh, Wout], M = P.map((w) => new Float64Array(w.length)), V = P.map((w) => new Float64Array(w.length)), G = P.map((w) => new Float64Array(w.length));
  const xs = Array.from({ length: batch }, () => Array.from({ length: D_IN }, () => gauss(r)));
  const ys = xs.map((x) => Math.sin(x[0]) + 0.5 * x[1]);
  const h1 = new Float64Array(n), a1 = new Float64Array(n), h2 = new Float64Array(n), a2 = new Float64Array(n);
  function fwd(x) {
    for (let j = 0; j < n; j++) { let s = 0; for (let i = 0; i < D_IN; i++) s += x[i] * Win[i * n + j]; h1[j] = s; a1[j] = Math.tanh(s); }
    for (let j = 0; j < n; j++) { let s = 0; for (let i = 0; i < n; i++) s += a1[i] * Wh[i * n + j]; h2[j] = s; a2[j] = Math.tanh(s); }
    let o = 0; for (let j = 0; j < n; j++) o += a2[j] * Wout[j];
    return o;
  }
  const snapshot = () => xs.map((x) => { fwd(x); return Float64Array.from(a2); });
  const before = snapshot();
  const d1 = new Float64Array(n), d2 = new Float64Array(n);
  for (let t = 1; t <= steps; t++) {
    G.forEach((g) => g.fill(0));
    xs.forEach((x, b) => {
      const o = fwd(x), dO = (2 * (o - ys[b])) / batch;
      for (let j = 0; j < n; j++) { G[2][j] += dO * a2[j]; d2[j] = dO * Wout[j] * (1 - a2[j] * a2[j]); }
      for (let i = 0; i < n; i++) { let s = 0; const ai = a1[i]; for (let j = 0; j < n; j++) { G[1][i * n + j] += ai * d2[j]; s += Wh[i * n + j] * d2[j]; } d1[i] = s * (1 - ai * ai); }
      for (let i = 0; i < D_IN; i++) for (let j = 0; j < n; j++) G[0][i * n + j] += x[i] * d1[j];
    });
    P.forEach((W, l) => { for (let k = 0; k < W.length; k++) { const g = G[l][k]; M[l][k] = 0.9 * M[l][k] + 0.1 * g; V[l][k] = 0.999 * V[l][k] + 0.001 * g * g; W[k] -= (LR[l] * M[l][k] / (1 - 0.9 ** t)) / (Math.sqrt(V[l][k] / (1 - 0.999 ** t)) + 1e-8); } });
  }
  const after = snapshot();
  let dh = 0, out = 0;
  xs.forEach((x, b) => { for (let j = 0; j < n; j++) dh += (after[b][j] - before[b][j]) ** 2; out += fwd(x) ** 2; });
  return { dHidden: Math.sqrt(dh / (batch * n)), out: Math.sqrt(out / batch) };
}
