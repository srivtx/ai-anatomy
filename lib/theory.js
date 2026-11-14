// Simulations behind "Theories I'd test with a cluster".
import { rng } from "./core.js";
import { gauss } from "./nn.js";

/** Problem success probabilities drawn from a difficulty distribution. */
export function problems(kind, n = 3000, seed = 3) {
  const r = rng(seed), ps = [];
  for (let i = 0; i < n; i++) {
    let lp;
    if (kind === "loguniform") lp = -4 * r();
    else if (kind === "lognormal") lp = Math.min(0, -1.6 + 0.55 * gauss(r));
    else lp = r() < 0.45 ? -0.4 * r() : r() < 0.55 ? -8 : -1 - 2 * r();       // easy, impossible, mid
    ps.push(10 ** lp);
  }
  return ps;
}
/** Accuracy at B samples when a verifier accepts wrong answers with probability f (f = 0: perfect checker). */
export function accAt(ps, B, f = 0) {
  let s = 0;
  for (const p of ps) {
    const acc = p + (1 - p) * f;                               // P(a sample is accepted)
    s += f === 0 ? 1 - (1 - p) ** B : (p / acc) * (1 - (1 - acc) ** B);
  }
  return s / ps.length;
}
/** Student-t with ν degrees of freedom, scaled to unit variance (ν = Infinity → Gaussian). */
function noise(r, nu) {
  if (!isFinite(nu)) return gauss(r);
  let chi = 0; const k = Math.round(nu); for (let i = 0; i < k; i++) chi += gauss(r) ** 2;
  return (gauss(r) / Math.sqrt(chi / k)) * Math.sqrt((k - 2) / k);
}
/** Best-of-n on a proxy = gold + error. Returns mean gold reward of the selected sample. */
export function bestOfN(n, nu, trials, seed = 1, sigma = 1) {
  const r = rng(seed); let tot = 0;
  for (let t = 0; t < trials; t++) {
    let best = -Infinity, g = 0;
    for (let i = 0; i < n; i++) { const gi = gauss(r), pi = gi + sigma * noise(r, nu); if (pi > best) { best = pi; g = gi; } }
    tot += g;
  }
  return tot / trials;
}
