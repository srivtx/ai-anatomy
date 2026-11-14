// Grokking modular addition in the browser: (a + b) mod p with a one-hidden-layer
// network on one-hot inputs, full-batch AdamW. Written for speed: the first layer
// of a one-hot input is just two row lookups.
import { rng } from "./core.js";
import { gauss } from "./nn.js";

export function grokker({ p = 23, H = 96, frac = 0.45, wd = 1, lr = 3e-3, seed = 1, act = "sq", init = 1 } = {}) {
  const r = rng(seed);
  const pairs = [];
  for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) pairs.push([a, b, (a + b) % p]);
  for (let i = pairs.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [pairs[i], pairs[j]] = [pairs[j], pairs[i]]; }
  const nTr = Math.round(pairs.length * frac), train = pairs.slice(0, nTr), test = pairs.slice(nTr);
  const W1 = Float64Array.from({ length: 2 * p * H }, () => gauss(r) * init / Math.sqrt(2));
  const W2 = Float64Array.from({ length: H * p }, () => gauss(r) * init / Math.sqrt(H));
  const P = [W1, W2], G = P.map((w) => new Float64Array(w.length)), M = P.map((w) => new Float64Array(w.length)), V = P.map((w) => new Float64Array(w.length));
  const pre = new Float64Array(H), hid = new Float64Array(H), z = new Float64Array(p), dh = new Float64Array(H);
  const f = act === "sq" ? (x) => x * x : (x) => (x > 0 ? x : 0), df = act === "sq" ? (x) => 2 * x : (x) => (x > 0 ? 1 : 0);
  let t = 0;
  function fwd(a, b) {
    for (let j = 0; j < H; j++) { pre[j] = W1[a * H + j] + W1[(p + b) * H + j]; hid[j] = f(pre[j]); }
    let mx = -Infinity;
    for (let k = 0; k < p; k++) { let s = 0; for (let j = 0; j < H; j++) s += hid[j] * W2[j * p + k]; z[k] = s; if (s > mx) mx = s; }
    let se = 0; for (let k = 0; k < p; k++) { z[k] = Math.exp(z[k] - mx); se += z[k]; }
    for (let k = 0; k < p; k++) z[k] /= se;
    return z;
  }
  function evalSet(set) {
    let ok = 0, loss = 0;
    for (const [a, b, c] of set) { const q = fwd(a, b); let am = 0; for (let k = 1; k < p; k++) if (q[k] > q[am]) am = k; ok += am === c; loss -= Math.log(Math.max(q[c], 1e-12)); }
    return { acc: ok / set.length, loss: loss / set.length };
  }
  function step() {
    G[0].fill(0); G[1].fill(0);
    for (const [a, b, c] of train) {
      const q = fwd(a, b);
      q[c] -= 1;                                   // dLoss/dlogits
      dh.fill(0);
      for (let j = 0; j < H; j++) { let s = 0; const hj = hid[j]; for (let k = 0; k < p; k++) { G[1][j * p + k] += hj * q[k]; s += W2[j * p + k] * q[k]; } dh[j] = s * df(pre[j]); }
      for (let j = 0; j < H; j++) { G[0][a * H + j] += dh[j]; G[0][(p + b) * H + j] += dh[j]; }
    }
    t++;
    const n = train.length;
    P.forEach((W, i) => {
      for (let k = 0; k < W.length; k++) {
        const g = G[i][k] / n;
        M[i][k] = 0.9 * M[i][k] + 0.1 * g; V[i][k] = 0.98 * V[i][k] + 0.02 * g * g;
        W[k] -= lr * ((M[i][k] / (1 - 0.9 ** t)) / (Math.sqrt(V[i][k] / (1 - 0.98 ** t)) + 1e-8) + wd * W[k]);   // AdamW: decoupled decay
      }
    });
  }
  const norm = () => Math.sqrt(P.reduce((a, W) => a + W.reduce((s, v) => s + v * v, 0), 0));
  /**
   * Fourier view of the input embedding. For each hidden unit, how much of its weight pattern over a = 0..p-1
   * sits in its single strongest frequency (1 = a pure wave), and which frequency that is.
   */
  function fourier() {
    const K = (p - 1) / 2, units = [];
    for (let j = 0; j < H; j++) {
      const e = [];
      for (let k = 1; k <= K; k++) {
        let c = 0, s = 0;
        for (let a = 0; a < p; a++) { const w = W1[a * H + j] + W1[(p + a) * H + j]; c += w * Math.cos((2 * Math.PI * k * a) / p); s += w * Math.sin((2 * Math.PI * k * a) / p); }
        e.push(c * c + s * s);
      }
      const tot = e.reduce((x, y) => x + y, 0) || 1, best = e.indexOf(Math.max(...e));
      units.push({ k: best + 1, purity: e[best] / tot });
    }
    return { purity: units.reduce((x, u) => x + u.purity, 0) / H, units };
  }
  /** weights of hidden unit j over the input value a = 0..p-1 (both operand embeddings summed) */
  const unit = (j) => Array.from({ length: p }, (_, a) => W1[a * H + j] + W1[(p + a) * H + j]);
  return { p, H, train, test, step, evalSet, norm, fourier, unit, get t() { return t; } };
}
