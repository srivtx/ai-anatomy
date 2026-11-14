// Shortcut learning on tiny images: shape (the real signal) vs background colour (the shortcut).
import { MLP, bceLogit, gauss } from "./nn.js";
import { rng } from "./core.js";

export const S = 8;
/** One image: label 1 = vertical bar, 0 = horizontal bar. hue ∈ {0, 1} is the background colour. */
export function image(label, hue, r, noise = 0.35) {
  const px = new Float64Array(S * S), k = Math.floor(r() * S);
  for (let i = 0; i < S; i++) for (let j = 0; j < S; j++) {
    const on = label ? j === k : i === k;
    px[i * S + j] = (on ? 1 : 0) + gauss(r) * noise;
  }
  return { px, hue, y: label };
}
/** corr: probability the background colour agrees with the label (0.5 = no shortcut, 1 = perfect shortcut). */
export function set(n, corr, seed, noise) {
  const r = rng(seed), out = [];
  for (let i = 0; i < n; i++) { const y = i % 2, agree = r() < corr; out.push(image(y, agree ? y : 1 - y, r, noise)); }
  return out;
}
export const input = (d) => [...d.px, d.hue * 2 - 1];
export function model(seed = 1) { return new MLP([S * S + 1, 12, 1], { act: "tanh", seed }); }
export function epoch(m, data, lr = 0.01, bs = 20) {
  let L = 0;
  for (let i = 0; i < data.length; i += bs) {
    const b = data.slice(i, i + bs);
    for (const d of b) { const { loss, d: g } = bceLogit(m.forward(input(d))[0], d.y); L += loss; m.backward([g]); }
    m.step(lr, b.length);
  }
  return L / data.length;
}
export const accuracy = (m, data) => data.filter((d) => (m.forward(input(d))[0] > 0) === !!d.y).length / data.length;
/** How much the decision depends on colour vs shape: |output change when hue flips| averaged. */
export function reliance(m, data) {
  let dh = 0;
  for (const d of data) { const a = m.forward(input(d))[0], b = m.forward(input({ ...d, hue: 1 - d.hue }))[0]; dh += Math.abs(Math.tanh(a) - Math.tanh(b)) / 2; }
  return dh / data.length;
}
