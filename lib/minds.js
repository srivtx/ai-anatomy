// Do different networks learn the same thing? Linear CKA between hidden layers, and model stitching.
import { MLP, softmaxCE, gauss } from "./nn.js";
import { rng } from "./core.js";

/** Three interleaved spiral arms: a task with some structure to discover. */
export function arms(n = 300, seed = 4, noise = 0.06) {
  const r = rng(seed), out = [];
  for (let i = 0; i < n; i++) {
    const c = i % 3, t = r() , th = t * 2.6 * Math.PI + (c * 2 * Math.PI) / 3, rad = 0.12 + t * 0.85;
    out.push({ x: [rad * Math.cos(th) + gauss(r) * noise, rad * Math.sin(th) + gauss(r) * noise], y: c });
  }
  return out;
}

export function net(kind, seed) {
  const spec = { A: { sizes: [2, 32, 32, 3], act: "tanh" }, B: { sizes: [2, 24, 48, 3], act: "relu" }, C: { sizes: [2, 64, 16, 3], act: "sigmoid" } }[kind];
  return new MLP(spec.sizes, { act: spec.act, seed });
}

export function trainEpoch(m, data, lr = 0.01, bs = 30) {
  let L = 0;
  for (let i = 0; i < data.length; i += bs) {
    const batch = data.slice(i, i + bs);
    for (const p of batch) { const { loss, d } = softmaxCE(m.forward(p.x), p.y); L += loss; m.backward(d); }
    m.step(lr, batch.length);
  }
  return L / data.length;
}
export const acc = (m, data) => data.filter((p) => argmax(m.forward(p.x)) === p.y).length / data.length;
export const argmax = (z) => { let b = 0; for (let i = 1; i < z.length; i++) if (z[i] > z[b]) b = i; return b; };

/** Hidden activations at every layer for a probe set: reps[l] is n × width. */
export function reps(m, xs) {
  const L = m.W.length - 1, out = Array.from({ length: L }, () => []);
  for (const x of xs) { m.forward(x); for (let l = 0; l < L; l++) out[l].push(Array.from(m.cache.ys[l + 1])); }
  return out;
}
function center(X) {
  const n = X.length, p = X[0].length, mu = new Float64Array(p);
  for (const r of X) for (let j = 0; j < p; j++) mu[j] += r[j] / n;
  return X.map((r) => r.map((v, j) => v - mu[j]));
}
/** ||XᵀY||²_F of centred matrices, via the n×n Gram matrices (n is small). */
function hsic(X, Y) {
  const p = X[0].length, q = Y[0].length; let s = 0;
  for (let a = 0; a < p; a++) for (let b = 0; b < q; b++) { let d = 0; for (let i = 0; i < X.length; i++) d += X[i][a] * Y[i][b]; s += d * d; }
  return s;
}
/** Linear CKA (Kornblith et al., 2019): 1 = same representation up to rotation and scale. */
export function cka(X, Y) {
  X = center(X); Y = center(Y);
  const xy = hsic(X, Y), xx = hsic(X, X), yy = hsic(Y, Y);
  return xy / Math.sqrt(xx * yy + 1e-12);
}

/** Least-squares linear map from A's layer-l activations to B's (with bias). Ridge for stability. */
function lstsq(X, Y, lam = 1e-3) {
  const n = X.length, p = X[0].length + 1, q = Y[0].length;
  const Xa = X.map((r) => [...r, 1]);
  const A = Array.from({ length: p }, (_, i) => Array.from({ length: p }, (_, j) => { let s = 0; for (let k = 0; k < n; k++) s += Xa[k][i] * Xa[k][j]; return s + (i === j ? lam * n : 0); }));
  const B = Array.from({ length: p }, (_, i) => Array.from({ length: q }, (_, j) => { let s = 0; for (let k = 0; k < n; k++) s += Xa[k][i] * Y[k][j]; return s; }));
  for (let i = 0; i < p; i++) {                    // Gauss–Jordan
    let piv = i; for (let r = i + 1; r < p; r++) if (Math.abs(A[r][i]) > Math.abs(A[piv][i])) piv = r;
    [A[i], A[piv]] = [A[piv], A[i]]; [B[i], B[piv]] = [B[piv], B[i]];
    const d = A[i][i];
    for (let j = 0; j < p; j++) A[i][j] /= d; for (let j = 0; j < q; j++) B[i][j] /= d;
    for (let r = 0; r < p; r++) if (r !== i) { const f = A[r][i]; if (!f) continue; for (let j = 0; j < p; j++) A[r][j] -= f * A[i][j]; for (let j = 0; j < q; j++) B[r][j] -= f * B[i][j]; }
  }
  return (x) => { const xa = [...x, 1]; return Array.from({ length: q }, (_, j) => { let s = 0; for (let i = 0; i < p; i++) s += xa[i] * B[i][j]; return s; }); };
}
/** Run B's layers after `l` from a given hidden vector. */
function forwardFrom(m, hvec, l) {
  let a = hvec;
  for (let k = l + 1; k < m.W.length; k++) {
    const nIn = m.sizes[k], nOut = m.sizes[k + 1], W = m.W[k], f = { tanh: Math.tanh, relu: (z) => Math.max(0, z), sigmoid: (z) => 1 / (1 + Math.exp(-z)), linear: (z) => z }[m.acts[k]];
    const y = new Array(nOut);
    for (let j = 0; j < nOut; j++) { let s = m.b[k][j]; for (let i = 0; i < nIn; i++) s += a[i] * W[i * nOut + j]; y[j] = f(s); }
    a = y;
  }
  return a;
}
/** Stitch: A's first l+1 layers → linear map → B's remaining layers. Accuracy of the hybrid. */
export function stitch(A, B, fitXs, testData, l = 0) {
  const RA = reps(A, fitXs)[l], RB = reps(B, fitXs)[l], map = lstsq(RA, RB);
  return testData.filter((p) => { A.forward(p.x); const ha = Array.from(A.cache.ys[l + 1]); return argmax(forwardFrom(B, map(ha), l)) === p.y; }).length / testData.length;
}

/** Hungarian algorithm (min-cost assignment) for a square cost matrix. Returns col index per row. */
export function hungarian(C) {
  const n = C.length, INF = 1e18, u = new Float64Array(n + 1), v = new Float64Array(n + 1), p = new Int32Array(n + 1), way = new Int32Array(n + 1);
  for (let i = 1; i <= n; i++) {
    p[0] = i; let j0 = 0; const minv = new Float64Array(n + 1).fill(INF), used = new Uint8Array(n + 1);
    do {
      used[j0] = 1; const i0 = p[j0]; let delta = INF, j1 = 0;
      for (let j = 1; j <= n; j++) if (!used[j]) {
        const cur = C[i0 - 1][j - 1] - u[i0] - v[j];
        if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
        if (minv[j] < delta) { delta = minv[j]; j1 = j; }
      }
      for (let j = 0; j <= n; j++) if (used[j]) { u[p[j]] += delta; v[j] -= delta; } else minv[j] -= delta;
      j0 = j1;
    } while (p[j0] !== 0);
    do { const j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
  }
  const ans = new Int32Array(n); for (let j = 1; j <= n; j++) ans[p[j] - 1] = j - 1; return ans;
}

export function clone(m) {
  const c = new MLP(m.sizes, { act: m.acts[0], out: m.acts[m.acts.length - 1] });
  m.W.forEach((w, l) => { c.W[l].set(w); c.b[l].set(m.b[l]); }); return c;
}
/** Re-order B's hidden units to best match A's, layer by layer, by activation correlation. */
export function align(A, B, xs) {
  const P = clone(B);
  for (let l = 0; l < A.W.length - 1; l++) {
    const ra = reps(A, xs)[l], rb = reps(P, xs)[l], n = ra[0].length;
    const z = (R) => { const N = R.length; return Array.from({ length: n }, (_, j) => { let mu = 0, s = 0; for (const r of R) mu += r[j] / N; for (const r of R) s += (r[j] - mu) ** 2; s = Math.sqrt(s / N) + 1e-9; return R.map((r) => (r[j] - mu) / s); }); };
    const za = z(ra), zb = z(rb);
    const cost = za.map((a) => zb.map((b) => { let c = 0; for (let k = 0; k < a.length; k++) c += a[k] * b[k]; return -c; }));
    const perm = hungarian(cost);                     // A's unit i ↔ B's unit perm[i]
    const nIn = P.sizes[l], nOut = P.sizes[l + 1], nNext = P.sizes[l + 2];
    const W = P.W[l].slice(), b = P.b[l].slice(), W2 = P.W[l + 1].slice();
    for (let i = 0; i < nOut; i++) {
      const j = perm[i]; P.b[l][i] = b[j];
      for (let k = 0; k < nIn; k++) P.W[l][k * nOut + i] = W[k * nOut + j];
      for (let k = 0; k < nNext; k++) P.W[l + 1][i * nNext + k] = W2[j * nNext + k];
    }
  }
  return P;
}
/** The network at (1 − t)·A + t·B, weight by weight. */
export function lerpNet(A, B, t) {
  const c = clone(A);
  c.W.forEach((w, l) => { for (let i = 0; i < w.length; i++) w[i] = (1 - t) * A.W[l][i] + t * B.W[l][i]; for (let i = 0; i < c.b[l].length; i++) c.b[l][i] = (1 - t) * A.b[l][i] + t * B.b[l][i]; });
  return c;
}
export function meanLoss(m, data) { let L = 0; for (const p of data) L += softmaxCE(m.forward(p.x), p.y).loss; return L / data.length; }
