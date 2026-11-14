// A tiny neural-network engine for the browser: dense layers, manual backprop
// and Adam. Small enough to read in one sitting, fast enough to train live.
// Everything is plain Float64Arrays; one sample at a time, gradients summed
// over a batch, then one optimizer step.
import { rng as makeRng } from "./core.js";

export const ACTS = {
  tanh: { f: Math.tanh, d: (y) => 1 - y * y },
  relu: { f: (x) => (x > 0 ? x : 0), d: (y) => (y > 0 ? 1 : 0) },
  sigmoid: { f: (x) => 1 / (1 + Math.exp(-x)), d: (y) => y * (1 - y) },
  linear: { f: (x) => x, d: () => 1 },
  sin: { f: Math.sin, d: (y, x) => Math.cos(x) },
};

/** Box-Muller normal sample from a uniform RNG. */
export function gauss(r) {
  let u = 0;
  while (u === 0) u = r();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r());
}

export class MLP {
  /**
   * sizes: [in, hidden..., out]; act: hidden activation; out: output activation.
   * initScale multiplies the standard Xavier/He scale so chapters can show bad init.
   */
  constructor(sizes, { act = "tanh", out = "linear", seed = 1, initScale = 1 } = {}) {
    this.sizes = sizes;
    this.acts = sizes.slice(1).map((_, i) => (i === sizes.length - 2 ? out : act));
    const r = makeRng(seed);
    this.W = []; this.b = []; this.gW = []; this.gb = [];
    this.mW = []; this.vW = []; this.mb = []; this.vb = [];
    for (let l = 0; l < sizes.length - 1; l++) {
      const nIn = sizes[l], nOut = sizes[l + 1];
      const he = this.acts[l] === "relu" ? Math.sqrt(2 / nIn) : Math.sqrt(1 / nIn);
      const W = new Float64Array(nIn * nOut).map(() => gauss(r) * he * initScale);
      this.W.push(W); this.b.push(new Float64Array(nOut));
      for (const arr of [this.gW, this.mW, this.vW]) arr.push(new Float64Array(nIn * nOut));
      for (const arr of [this.gb, this.mb, this.vb]) arr.push(new Float64Array(nOut));
    }
    this.t = 0;
    this.cache = null;
  }
  get params() { return this.W.reduce((a, w, i) => a + w.length + this.b[i].length, 0); }

  /** Forward one sample. Keeps pre-activations and outputs for backward(). */
  forward(x) {
    const zs = [], ys = [Float64Array.from(x)];
    let a = ys[0];
    for (let l = 0; l < this.W.length; l++) {
      const nIn = this.sizes[l], nOut = this.sizes[l + 1], W = this.W[l], f = ACTS[this.acts[l]].f;
      const z = new Float64Array(nOut), y = new Float64Array(nOut);
      for (let j = 0; j < nOut; j++) {
        let s = this.b[l][j];
        for (let i = 0; i < nIn; i++) s += a[i] * W[i * nOut + j];
        z[j] = s; y[j] = f(s);
      }
      zs.push(z); ys.push(y); a = y;
    }
    this.cache = { zs, ys };
    return a;
  }

  /** Backward from dLoss/dOutput. Adds into the gradient buffers; returns dLoss/dInput. */
  backward(dy) {
    const { zs, ys } = this.cache;
    let g = Float64Array.from(dy);
    for (let l = this.W.length - 1; l >= 0; l--) {
      const nIn = this.sizes[l], nOut = this.sizes[l + 1], W = this.W[l], d = ACTS[this.acts[l]].d;
      const y = ys[l + 1], z = zs[l], a = ys[l];
      for (let j = 0; j < nOut; j++) g[j] *= d(y[j], z[j]);
      const gin = new Float64Array(nIn);
      for (let i = 0; i < nIn; i++) {
        let s = 0;
        for (let j = 0; j < nOut; j++) {
          this.gW[l][i * nOut + j] += a[i] * g[j];
          s += W[i * nOut + j] * g[j];
        }
        gin[i] = s;
      }
      for (let j = 0; j < nOut; j++) this.gb[l][j] += g[j];
      g = gin;
    }
    return g;
  }

  /** One optimizer step on gradients averaged over n samples, then zero them. */
  step(lr, n = 1, { opt = "adam", wd = 0, b1 = 0.9, b2 = 0.999, momentum = 0.9 } = {}) {
    this.t++;
    for (let l = 0; l < this.W.length; l++) {
      for (const [P, G, M, V, decay] of [[this.W[l], this.gW[l], this.mW[l], this.vW[l], wd], [this.b[l], this.gb[l], this.mb[l], this.vb[l], 0]]) {
        for (let k = 0; k < P.length; k++) {
          const g = G[k] / n + decay * P[k];
          if (opt === "adam") {
            M[k] = b1 * M[k] + (1 - b1) * g;
            V[k] = b2 * V[k] + (1 - b2) * g * g;
            const mh = M[k] / (1 - b1 ** this.t), vh = V[k] / (1 - b2 ** this.t);
            P[k] -= (lr * mh) / (Math.sqrt(vh) + 1e-8);
          } else if (opt === "momentum") {
            M[k] = momentum * M[k] + g;
            P[k] -= lr * M[k];
          } else P[k] -= lr * g;
          G[k] = 0;
        }
      }
    }
  }
  zeroGrad() { for (const G of [...this.gW, ...this.gb]) G.fill(0); }
}

// ---- losses: each returns { loss, d } where d is dLoss/dOutput for one sample
export function mse(y, t) {
  const d = new Float64Array(y.length);
  let loss = 0;
  for (let i = 0; i < y.length; i++) { const e = y[i] - t[i]; loss += e * e; d[i] = (2 * e) / y.length; }
  return { loss: loss / y.length, d };
}
export function softmax(z, T = 1) {
  const m = Math.max(...z);
  const e = Array.from(z, (v) => Math.exp((v - m) / T));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}
/** Softmax cross-entropy on logits z with integer class c. */
export function softmaxCE(z, c) {
  const p = softmax(z);
  const d = Float64Array.from(p);
  d[c] -= 1;
  return { loss: -Math.log(Math.max(p[c], 1e-12)), d, p };
}
/** Binary cross-entropy on a logit z (linear output) with target t in {0,1}. */
export function bceLogit(z, t) {
  const p = 1 / (1 + Math.exp(-z));
  const q = Math.min(1 - 1e-12, Math.max(1e-12, p));
  return { loss: -(t * Math.log(q) + (1 - t) * Math.log(1 - q)), d: p - t, p };
}
export const sigmoid = (x) => 1 / (1 + Math.exp(-x));

// ---- toy datasets in [-1, 1]^2: [{x:[a,b], y:class}]
export function dataset(kind, n = 200, { noise = 0.1, seed = 3 } = {}) {
  const r = makeRng(seed), out = [];
  const nz = () => gauss(r) * noise;
  for (let i = 0; i < n; i++) {
    const c = i % 2;
    if (kind === "xor") {
      const a = r() * 2 - 1, b = r() * 2 - 1;
      out.push({ x: [a + nz() * 0.5, b + nz() * 0.5], y: a * b > 0 ? 1 : 0 });
    } else if (kind === "circle") {
      const rad = c ? r() * 0.45 : 0.6 + r() * 0.35, th = r() * Math.PI * 2;
      out.push({ x: [rad * Math.cos(th) + nz(), rad * Math.sin(th) + nz()], y: c });
    } else if (kind === "spiral") {
      const t = (i >> 1) / (n / 2), th = t * 3.2 * Math.PI + c * Math.PI, rad = 0.1 + t * 0.85;
      out.push({ x: [rad * Math.cos(th) + nz() * 0.6, rad * Math.sin(th) + nz() * 0.6], y: c });
    } else if (kind === "moons") {
      const th = r() * Math.PI;
      const p = c ? [0.5 - Math.cos(th) * 0.8, 0.25 - Math.sin(th) * 0.8 + 0.25] : [-0.5 + Math.cos(th) * 0.8, Math.sin(th) * 0.8 - 0.25];
      out.push({ x: [p[0] * 0.75 + nz(), p[1] * 0.75 + nz()], y: c });
    } else {
      // linear: two blobs
      out.push({ x: [(c ? 0.45 : -0.45) + gauss(r) * 0.25, (c ? 0.35 : -0.35) + gauss(r) * 0.25], y: c });
    }
  }
  return out;
}
