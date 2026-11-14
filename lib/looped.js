// A looped (weight-tied) cellular network: the same tiny rule is applied to every cell of a
// grid, over and over. Trained end to end by backpropagation through the iterations.
// Pure maths, no DOM, so it runs in node for testing.
import { rng } from "./core.js";
import { gauss } from "./nn.js";

const F = 7, Hd = 10; // features: open, source, self, up, down, left, right
const sig = (x) => 1 / (1 + Math.exp(-x));

/** Random maze: open cells with probability `density`, source at the top-left corner. */
export function maze(n, r, density = 0.68, connected = false) {
  // connected: keep drawing until the far corner is reachable, so the flood has a long way to travel
  for (let tries = 0; tries < 200; tries++) { const m = drawMaze(n, r, density); if (!connected || m.reach[n * n - 1]) return m; }
  return drawMaze(n, r, density);
}
function drawMaze(n, r, density) {
  const open = Array.from({ length: n * n }, () => (r() < density ? 1 : 0));
  open[0] = 1; open[n * n - 1] = 1;
  // breadth-first search from the source: reachability and shortest-path distance (= iterations a perfect flood fill needs)
  const reach = new Array(n * n).fill(0), dist = new Array(n * n).fill(Infinity), q = [0]; reach[0] = 1; dist[0] = 0;
  while (q.length) { const c = q.shift(), x = c % n, y = (c / n) | 0; for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) { const nx = x + dx, ny = y + dy, k = ny * n + nx; if (nx >= 0 && ny >= 0 && nx < n && ny < n && open[k] && !reach[k]) { reach[k] = 1; dist[k] = dist[c] + 1; q.push(k); } } }
  const depth = Math.max(...dist.filter(Number.isFinite));
  return { n, open, reach, dist, depth, src: 0 };
}

export class LoopNet {
  constructor(seed = 1) {
    const r = rng(seed);
    this.W1 = Float64Array.from({ length: F * Hd }, () => gauss(r) * 0.6);
    this.b1 = new Float64Array(Hd);
    this.W2 = Float64Array.from({ length: Hd }, () => gauss(r) * 0.6);
    this.b2 = 0;
    this.m = new Float64Array(F * Hd + Hd + Hd + 1); this.v = new Float64Array(this.m.length); this.t = 0;
  }
  feats(m, R, c) {
    const n = m.n, x = c % n, y = (c / n) | 0, g = (xx, yy) => (xx >= 0 && yy >= 0 && xx < n && yy < n ? R[yy * n + xx] : 0);
    return [m.open[c] * 2 - 1, c === m.src ? 1 : 0, R[c], g(x, y - 1), g(x, y + 1), g(x - 1, y), g(x + 1, y)];
  }
  /** Run T iterations. Returns the list of states R_0..R_T (and caches for backward if keep). */
  run(m, T, keep = false) {
    const N = m.n * m.n; let R = new Float64Array(N);
    const states = [R], cache = [];
    for (let t = 0; t < T; t++) {
      const Rn = new Float64Array(N), cs = keep ? [] : null;
      for (let c = 0; c < N; c++) {
        const f = this.feats(m, R, c), hh = new Float64Array(Hd);
        let z = this.b2;
        for (let j = 0; j < Hd; j++) { let s = this.b1[j]; for (let i = 0; i < F; i++) s += f[i] * this.W1[i * Hd + j]; hh[j] = Math.tanh(s); z += hh[j] * this.W2[j]; }
        Rn[c] = sig(z);
        if (keep) cs.push([f, hh]);
      }
      R = Rn; states.push(R); if (keep) cache.push(cs);
    }
    return { states, cache };
  }
  /**
   * One training step on a batch of mazes with T iterations.
   * mode "final": loss only on the final state (pure end-to-end).
   * mode "steps": after iteration t, cells within distance t should be on (supervise every iteration).
   */
  train(batch, T, lr = 0.01, mode = "steps") {
    const gW1 = new Float64Array(F * Hd), gb1 = new Float64Array(Hd), gW2 = new Float64Array(Hd);
    let gb2 = 0, loss = 0, cnt = 0;
    for (const m of batch) {
      const n = m.n, N = n * n, { states, cache } = this.run(m, T, true);
      const lossAt = (t, dR) => {
        const R = states[t];
        for (let c = 0; c < N; c++) { const p = Math.min(1 - 1e-7, Math.max(1e-7, R[c])), y = mode === "final" ? m.reach[c] : m.dist[c] <= t - 1 ? 1 : 0; loss -= y * Math.log(p) + (1 - y) * Math.log(1 - p); dR[c] += (p - y) / (p * (1 - p)); cnt++; }
      };
      let dR = new Float64Array(N);
      lossAt(T, dR);
      for (let t = T - 1; t >= 0; t--) {
        const Rout = states[t + 1], dPrev = new Float64Array(N);
        for (let c = 0; c < N; c++) {
          const dz = dR[c] * Rout[c] * (1 - Rout[c]);
          if (dz === 0) continue;
          const [f, hh] = cache[t][c];
          gb2 += dz;
          const df = new Float64Array(F);
          for (let j = 0; j < Hd; j++) {
            gW2[j] += dz * hh[j];
            const dh = dz * this.W2[j] * (1 - hh[j] * hh[j]);
            gb1[j] += dh;
            for (let i = 0; i < F; i++) { gW1[i * Hd + j] += dh * f[i]; df[i] += dh * this.W1[i * Hd + j]; }
          }
          const x = c % n, y = (c / n) | 0;
          dPrev[c] += df[2];
          if (y > 0) dPrev[c - n] += df[3];
          if (y < n - 1) dPrev[c + n] += df[4];
          if (x > 0) dPrev[c - 1] += df[5];
          if (x < n - 1) dPrev[c + 1] += df[6];
        }
        dR = dPrev;
        if (mode !== "final" && t > 0) lossAt(t, dR);
      }
    }
    // Adam over all parameters, with gradient clipping (backprop through many iterations can explode)
    const g = [...gW1, ...gb1, ...gW2, gb2].map((v) => v / cnt);
    const norm = Math.hypot(...g), clip = norm > 1 ? 1 / norm : 1;
    const P = [this.W1, this.b1, this.W2];
    this.t++;
    let k = 0;
    const upd = (val, gi) => { const gg = g[gi] * clip; this.m[gi] = 0.9 * this.m[gi] + 0.1 * gg; this.v[gi] = 0.999 * this.v[gi] + 0.001 * gg * gg; return val - (lr * this.m[gi] / (1 - 0.9 ** this.t)) / (Math.sqrt(this.v[gi] / (1 - 0.999 ** this.t)) + 1e-8); };
    for (const arr of P) for (let i = 0; i < arr.length; i++) { arr[i] = upd(arr[i], k); k++; }
    this.b2 = upd(this.b2, k);
    return loss / cnt;
  }
  /** Fraction of mazes where every cell is classified correctly after T iterations. */
  solved(mazes, T) {
    let ok = 0;
    for (const m of mazes) { const R = this.run(m, T).states[T]; ok += m.reach.every((y, c) => (R[c] > 0.5) === !!y); }
    return ok / mazes.length;
  }
}
