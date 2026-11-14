// A tiny CLIP: an MLP image encoder and a bag-of-words text encoder trained with the
// symmetric contrastive (InfoNCE) loss on coloured shapes. Pure maths, no DOM, so it
// also runs in node for testing.
import { MLP, gauss } from "./nn.js";
import { rng } from "./core.js";

export const N = 10;
export const COLORS = { red: [0.95, 0.2, 0.15], green: [0.15, 0.75, 0.25], blue: [0.2, 0.35, 0.95], yellow: [0.95, 0.85, 0.1] };
export const SHAPES = ["circle", "square", "triangle", "cross"];
export const CNAMES = Object.keys(COLORS);
export const WORDS = [...CNAMES, ...SHAPES];
// combinations never shown during training: zero-shot tests compositional generalisation
export const HELD = new Set(["red triangle", "green cross", "blue circle", "yellow square"]);
export const ALL_PAIRS = CNAMES.flatMap((c) => SHAPES.map((s) => `${c} ${s}`));

function inside(shape, x, y, cx, cy, r) {
  const dx = x - cx, dy = y - cy;
  if (shape === "circle") return dx * dx + dy * dy <= r * r;
  if (shape === "square") return Math.abs(dx) <= r * 0.8 && Math.abs(dy) <= r * 0.8;
  if (shape === "triangle") return dy <= r * 0.8 && dy >= -r && Math.abs(dx) <= (dy + r) * 0.55;
  return (Math.abs(dx) <= r * 0.3 && Math.abs(dy) <= r) || (Math.abs(dy) <= r * 0.3 && Math.abs(dx) <= r);
}

/** Render a caption to an N×N×3 image (flat array), with random position, size and noise. */
export function render(caption, r) {
  const [c, s] = caption.split(" "), col = COLORS[c];
  const cx = 4.5 + (r() - 0.5) * 2.4, cy = 4.5 + (r() - 0.5) * 2.4, rad = 2.6 + r() * 1.2;
  const img = new Float64Array(N * N * 3);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const on = inside(s, x, y, cx, cy, rad);
    for (let k = 0; k < 3; k++) img[(y * N + x) * 3 + k] = (on ? col[k] : 0.06) + gauss(r) * 0.04;
  }
  return img;
}

const norm = (v) => { const n = Math.hypot(...v) || 1; return [v.map((x) => x / n), n]; };

export class TinyClip {
  constructor(seed = 1, dim = 8) {
    this.dim = dim; this.r = rng(seed);
    this.img = new MLP([N * N * 3, 32, dim], { act: "tanh", seed: seed + 3 });
    this.E = WORDS.map(() => Array.from({ length: dim }, () => gauss(this.r) * 0.5));
    this.mE = this.E.map((e) => e.map(() => 0)); this.vE = this.E.map((e) => e.map(() => 0)); this.t = 0;
    this.logT = Math.log(1 / 0.1); // learned inverse temperature, starts at 1/τ = 10
    this.steps = 0;
  }
  textRaw(caption) { const [c, s] = caption.split(" "); const a = this.E[WORDS.indexOf(c)], b = this.E[WORDS.indexOf(s)]; return a.map((v, i) => v + b[i]); }
  embedText(caption) { return norm(this.textRaw(caption))[0]; }
  embedImage(img) { return norm([...this.img.forward(img)])[0]; }

  /** One contrastive step on a batch of captions (each rendered once). Returns the loss. */
  step(batch, lr = 0.01) {
    const B = batch.length, scale = Math.exp(this.logT);
    const imgs = batch.map((c) => render(c, this.r));
    const iRaw = imgs.map((x) => [...this.img.forward(x)]), tRaw = batch.map((c) => this.textRaw(c));
    const I = iRaw.map((v) => norm(v)), T = tRaw.map((v) => norm(v));
    const S = I.map(([a]) => T.map(([b]) => scale * a.reduce((s, x, k) => s + x * b[k], 0)));
    // symmetric cross-entropy: rows (image→text) and columns (text→image)
    const dS = S.map((row) => row.map(() => 0));
    let loss = 0;
    for (let i = 0; i < B; i++) {
      const m = Math.max(...S[i]), e = S[i].map((v) => Math.exp(v - m)), z = e.reduce((a, b) => a + b, 0);
      loss -= Math.log(e[i] / z) / (2 * B);
      for (let j = 0; j < B; j++) dS[i][j] += (e[j] / z - (i === j)) / (2 * B);
      const col = S.map((row) => row[i]), mc = Math.max(...col), ec = col.map((v) => Math.exp(v - mc)), zc = ec.reduce((a, b) => a + b, 0);
      loss -= Math.log(ec[i] / zc) / (2 * B);
      for (let j = 0; j < B; j++) dS[j][i] += (ec[j] / zc - (i === j)) / (2 * B);
    }
    // gradients wrt normalised embeddings, then through the L2 normalisation
    const dI = I.map(() => new Array(this.dim).fill(0)), dT = T.map(() => new Array(this.dim).fill(0));
    let dLogT = 0;
    for (let i = 0; i < B; i++) for (let j = 0; j < B; j++) {
      dLogT += dS[i][j] * S[i][j];
      for (let k = 0; k < this.dim; k++) { dI[i][k] += dS[i][j] * scale * T[j][0][k]; dT[j][k] += dS[i][j] * scale * I[i][0][k]; }
    }
    const throughNorm = ([u, n], g) => { const d = u.reduce((s, x, k) => s + x * g[k], 0); return g.map((gk, k) => (gk - u[k] * d) / n); };
    for (let i = 0; i < B; i++) { this.img.forward(imgs[i]); this.img.backward(throughNorm(I[i], dI[i])); }
    this.img.step(lr, 1);
    this.t++;
    const gE = this.E.map((e) => e.map(() => 0));
    batch.forEach((c, j) => { const g = throughNorm(T[j], dT[j]); for (const w of c.split(" ")) { const wi = WORDS.indexOf(w); for (let k = 0; k < this.dim; k++) gE[wi][k] += g[k]; } });
    this.E.forEach((e, wi) => e.forEach((_, k) => {
      const g = gE[wi][k]; this.mE[wi][k] = 0.9 * this.mE[wi][k] + 0.1 * g; this.vE[wi][k] = 0.999 * this.vE[wi][k] + 0.001 * g * g;
      e[k] -= (lr * this.mE[wi][k] / (1 - 0.9 ** this.t)) / (Math.sqrt(this.vE[wi][k] / (1 - 0.999 ** this.t)) + 1e-8);
    }));
    this.logT = Math.min(Math.log(100), this.logT - lr * 0.5 * dLogT);
    this.steps++;
    return loss;
  }

  /** Zero-shot: classify a fresh image of `caption` against all 16 captions. */
  classify(img) {
    const e = this.embedImage(img);
    const scores = ALL_PAIRS.map((c) => e.reduce((s, x, k) => s + x * this.embedText(c)[k], 0));
    return ALL_PAIRS[scores.indexOf(Math.max(...scores))];
  }
  accuracy(pairs, n = 6) {
    const r = rng(999); let ok = 0, tot = 0;
    for (const c of pairs) for (let k = 0; k < n; k++) { ok += this.classify(render(c, r)) === c; tot++; }
    return ok / tot;
  }
}
