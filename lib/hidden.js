// Simulations for the "hidden curriculum" chapters. Pure functions, no DOM,
// so they can be tested in Node.
import { rng } from "./core.js";

/* ---------- 1. What RL does to a model: sharpening a strategy distribution ---------- */
export const STRATEGIES = ["Standard algebra", "Case split", "Guess and check", "Work backwards", "Draw a diagram", "Find an invariant", "Parity trick", "Generating function"];
const COVER = [0.7, 0.4, 0.25, 0.2, 0.15, 0.1, 0.08, 0.06];
const PRIOR = [0.4, 0.18, 0.12, 0.1, 0.08, 0.06, 0.04, 0.02];

/** P problems; q[p][s] = chance strategy s solves problem p. */
export function rlWorld(P = 300, seed = 4) {
  const r = rng(seed);
  const q = Array.from({ length: P }, () => COVER.map((c) => (r() < c ? 0.25 + 0.65 * r() : 0.01)));
  return { q, logits0: PRIOR.map(Math.log), S: STRATEGIES.length, P };
}
export const softmax = (z) => { const m = Math.max(...z), e = z.map((v) => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0); return e.map((v) => v / s); };
const solveP = (w, pi) => w.q.map((row) => row.reduce((a, qs, s) => a + pi[s] * qs, 0));
/** Unbiased-equivalent pass@k averaged over problems, for a policy pi. */
export const passAtK = (w, pi, k) => { const ps = solveP(w, pi); return ps.reduce((a, p) => a + 1 - Math.pow(1 - p, k), 0) / ps.length; };
export const entropy = (pi) => -pi.reduce((a, p) => a + (p > 0 ? p * Math.log(p) : 0), 0);

/** One policy-gradient step on logits: expected reward + entropy bonus - KL to the base policy. Sampling noise from a finite batch. */
export function rlStep(w, logits, { lr = 0.5, ent = 0, kl = 0, batch = 64 }, r) {
  const pi = softmax(logits), pi0 = softmax(w.logits0);
  // finite-batch estimate of each strategy's mean reward
  const R = pi.map((_, s) => { let t = 0; for (let b = 0; b < 4; b++) t += w.q[Math.floor(r() * w.P)][s]; return t / 4; });
  const J = pi.reduce((a, p, s) => a + p * R[s], 0);
  const H = entropy(pi);
  const KL = pi.reduce((a, p, s) => a + p * Math.log(p / pi0[s]), 0);
  return logits.map((z, s) => {
    const g = pi[s] * (R[s] - J)                                  // ∇ expected reward
      - ent * pi[s] * (Math.log(pi[s]) + H)                       // ∇ entropy
      - kl * pi[s] * (Math.log(pi[s] / pi0[s]) - KL);             // −∇ KL(pi || pi0)
    return z + lr * g * (1 + 0.15 * (r() - 0.5) * Math.sqrt(64 / batch));
  });
}

/* ---------- 2. Models trained on their own outputs ---------- */
/** Zipf vocabulary of V tokens. Returns true probabilities. */
export function zipf(V = 400, s = 1.1) { const w = Array.from({ length: V }, (_, i) => 1 / Math.pow(i + 1, s)), t = w.reduce((a, b) => a + b, 0); return w.map((x) => x / t); }
function sampleCounts(p, n, r, temp = 1) {
  let q = p;
  if (temp !== 1) { const z = p.map((x) => (x > 0 ? Math.pow(x, 1 / temp) : 0)), t = z.reduce((a, b) => a + b, 0); q = z.map((x) => x / t); }
  const cdf = []; let c = 0; for (const x of q) cdf.push((c += x));
  const counts = new Array(p.length).fill(0);
  for (let i = 0; i < n; i++) { const u = r() * c; let lo = 0, hi = cdf.length - 1; while (lo < hi) { const m = (lo + hi) >> 1; cdf[m] < u ? (lo = m + 1) : (hi = m); } counts[lo]++; }
  return counts;
}
/**
 * Run G generations. mode: "replace" (train only on the last generation's samples),
 * "accumulate" (keep all data so far), or "mix" (each generation adds `fresh` real data).
 * Each "model" is the maximum-likelihood unigram fit of its training data.
 */
export function selfConsume({ G = 30, n = 2000, mode = "replace", fresh = 0.2, temp = 1, seed = 1, V = 400 } = {}) {
  const r = rng(seed), truth = zipf(V);
  let data = sampleCounts(truth, n, r), model = data.map((x) => x / n);
  const hist = [stat(model, truth, 0)];
  for (let g = 1; g <= G; g++) {
    const synth = sampleCounts(model, n, r, temp);
    if (mode === "replace") data = synth;
    else if (mode === "accumulate") data = data.map((x, i) => x + synth[i]);
    else { const real = sampleCounts(truth, Math.round(n * fresh), r); data = synth.map((x, i) => x * (1 - fresh) + real[i]); }
    const tot = data.reduce((a, b) => a + b, 0);
    model = data.map((x) => x / tot);
    hist.push(stat(model, truth, g));
  }
  return { hist, model, truth };
}
function stat(model, truth, g) {
  let kl = 0, alive = 0, tail = 0;
  truth.forEach((t, i) => { const m = Math.max(model[i], 1e-6); kl += t * Math.log(t / m); if (model[i] > 0) alive++; if (i >= truth.length / 2) tail += model[i]; });
  return { g, kl, alive, tail };
}

/* ---------- 3. Error bars on evals ---------- */
/**
 * Two models on an N-question benchmark. Each question has a latent difficulty;
 * model A's chance of solving it is sigmoid(skillA - d), B's is sigmoid(skillA + delta' - d),
 * with delta' chosen so B's true accuracy is acc + delta. Each run draws k samples per
 * question (k = 1 means one sampled answer). Optionally the B team tries F prompt formats,
 * each adding a random offset, and reports the best one.
 */
const sig = (x) => 1 / (1 + Math.exp(-x));
function solveSkill(diffs, target) { let lo = -10, hi = 10; for (let i = 0; i < 50; i++) { const m = (lo + hi) / 2; const a = diffs.reduce((s, d) => s + sig(m - d), 0) / diffs.length; a < target ? (lo = m) : (hi = m); } return (lo + hi) / 2; }
export function evalNoise({ N = 500, acc = 0.6, delta = 0.02, k = 1, formats = 1, formatSd = 0.15, reruns = 400, seed = 3 } = {}) {
  const r = rng(seed), gauss = () => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
  const out = [], pop = Array.from({ length: 4000 }, () => 1.6 * gauss());
  const sa = solveSkill(pop, acc), sb = solveSkill(pop, acc + delta);   // true accuracies on the whole population
  for (let run = 0; run < reruns; run++) {
    const diffs = Array.from({ length: N }, () => 1.6 * gauss());       // a fresh benchmark draw from the same population
    const score = (skill) => { let c = 0; const diff = []; for (const d of diffs) { let s = 0; for (let j = 0; j < k; j++) s += r() < sig(skill - d) ? 1 : 0; diff.push(s / k); c += s / k; } return [c / N, diff]; };
    const [a, pa] = score(sa);
    let best = -1, pb = null;
    for (let f = 0; f < formats; f++) { const off = formats > 1 ? formatSd * gauss() : 0; const [b, p] = score(sb + off); if (b > best) { best = b; pb = p; } }
    const d = pa.map((x, i) => pb[i] - x), md = d.reduce((s, x) => s + x, 0) / N;
    const sePaired = Math.sqrt(d.reduce((s, x) => s + (x - md) ** 2, 0) / (N - 1) / N);
    const seUnpaired = Math.sqrt((a * (1 - a) + best * (1 - best)) / N);
    out.push({ a, b: best, diff: best - a, sePaired, seUnpaired });
  }
  return out;
}

/* ---------- 4. How long should a model think? ---------- */
/**
 * A problem needs d serial steps (d ~ geometric around `meanSteps`). Each step is wrong with
 * probability eps * (1 + t / rot) (context rot: later steps are sloppier). After each step the
 * model re-checks the last step with probability check and fixes a wrong one (costs one step).
 * After finishing, a model that keeps thinking second-guesses itself: each extra step flips a
 * correct answer with probability flip and a wrong one with probability flip / 3.
 */
export function thinkCurve({ eps = 0.03, check = 0.3, rot = 400, flip = 0.004, stopWhenDone = false, meanSteps = 30, budgets, trials = 1500, seed = 2 } = {}) {
  const r = rng(seed);
  const problems = Array.from({ length: trials }, () => Math.max(2, Math.round(-Math.log(1 - r()) * meanSteps)));
  return budgets.map((B) => {
    const r = rng(seed + 101);          // common random numbers: every budget sees the same luck
    let ok = 0;
    for (const d of problems) {
      let t = 0, done = 0, wrong = false, pendingBad = false;
      while (t < B && done < d) {
        t++;
        if (pendingBad && r() < check) { pendingBad = false; continue; }        // caught and fixed
        if (pendingBad) wrong = true;                                            // error now baked in
        pendingBad = r() < eps * (1 + t / rot);
        done++;
      }
      if (done < d) continue;                                                    // ran out of budget: no answer
      let correct = !(wrong || pendingBad);
      if (!stopWhenDone) for (; t < B; t++) { if (correct ? r() < flip : r() < flip / 3) correct = !correct; }
      if (correct) ok++;
    }
    return ok / problems.length;
  });
}

/* ---------- 5. How models teach themselves: expert iteration / STaR ---------- */
/** Chance a model of skill s solves a problem of difficulty d in one try. */
export const solveProb = (s, d) => sig(1.5 * (s - d));
/** Test accuracy on a fixed spread of difficulties 0..14. */
export const testAcc = (s) => { let a = 0; for (let i = 0; i < 141; i++) a += solveProb(s, i / 10); return a / 141; };
/**
 * Rounds of: sample K tries per problem, keep what the verifier accepts, fine-tune on it.
 * pool "fixed": 400 problems spread over difficulty 0..10. pool "frontier": a generator
 * writes fresh problems around the model's current skill every round.
 * leak: chance the verifier accepts a wrong answer. hint: STaR-style rationalisation
 * (show the answer, ask for a derivation) for problems the model failed.
 */
export function selfImprove({ rounds = 30, K = 4, leak = 0, pool = "fixed", hint = false, lr = 0.2, s0 = 2, M = 400, seed = 5 } = {}) {
  const r = rng(seed);
  const fixed = Array.from({ length: M }, (_, i) => (10 * (i + 0.5)) / M);
  let s = s0;
  const skill = [s], acc = [testAcc(s)], solvedFrac = [], dataSize = [];
  for (let t = 0; t < rounds; t++) {
    const probs = pool === "fixed" ? fixed : Array.from({ length: M }, () => Math.min(14, s - 1 + 4 * r()));
    let gain = 0, harm = 0, kept = 0, solved = 0;
    for (const d of probs) {
      const p = solveProb(s, d);
      const teach = Math.exp(-((d - s - 1.2) ** 2) / 1.5); // examples near the edge of ability teach the most
      if (r() < 1 - Math.pow(1 - p, K)) { gain += teach; kept++; solved++; continue; }
      if (hint && r() < 1 - Math.pow(1 - solveProb(s + 2, d), K)) { gain += 0.5 * teach; kept++; continue; } // off-policy, worth less
      if (leak > 0 && r() < 1 - Math.pow(1 - leak, K)) { harm += 1; kept++; } // a wrong answer slips through
    }
    s += (lr * (gain - 0.5 * harm)) / M * 10;
    s = Math.max(-2, s);
    skill.push(s); acc.push(testAcc(s)); solvedFrac.push(solved / probs.length); dataSize.push(kept);
  }
  return { skill, acc, solvedFrac, dataSize };
}

/* ---------- 6. The cooldown: SGD on a noisy quadratic, in expectation ---------- */
/** Learning-rate multiplier at step t of T. */
export function lrAt(kind, t, T, { warm = 0.02, decay = 0.2, branch = 1 } = {}) {
  const w = Math.max(1, Math.round(warm * T));
  if (t < w) return (t + 1) / w;
  if (kind === "constant") return 1;
  if (kind === "cosine") return 0.5 * (1 + Math.cos(Math.PI * (t - w) / (T - w)));
  if (kind === "linear") return 1 - (t - w) / (T - w);
  if (kind === "invsqrt") return Math.sqrt(w / (t + 1));
  if (kind === "wsd") { // stable, then a linear cooldown to zero over the last `decay` of the run (ending at `branch` · T)
    const end = branch * T, start = end - decay * end;
    if (t < start) return 1;
    return Math.max(0, 1 - (t - start) / (end - start));
  }
  return 1;
}
/**
 * Expected loss of SGD on L = ½ Σ λᵢ θᵢ² with gradient noise ∝ λᵢ (like real nets: noise
 * follows curvature). Exact in expectation: mᵢ ← (1 − ηλᵢ)² mᵢ + η² σ² λᵢ.
 */
export function quadRun({ T = 2000, peak = 0.5, noise = 0.05, kind = "cosine", decay = 0.2, branch = 1, D = 40, every = 10 } = {}) {
  const lam = Array.from({ length: D }, (_, i) => Math.pow(10, -3 + (3 * i) / (D - 1)));
  const m = lam.map(() => 1);
  const loss = (mm) => 0.5 * mm.reduce((a, v, i) => a + lam[i] * v, 0);
  const pts = [[0, loss(m)]];
  const stop = Math.round(branch * T);
  for (let t = 0; t < stop; t++) {
    const eta = peak * lrAt(kind, t, T, { decay, branch });
    for (let i = 0; i < D; i++) { const a = 1 - eta * lam[i]; m[i] = Math.min(1e6, a * a * m[i] + eta * eta * noise * lam[i]); }
    if ((t + 1) % every === 0 || t + 1 === stop) pts.push([t + 1, loss(m)]);
  }
  return pts;
}

/* ---------- 7. Emergence: a smooth skill seen through a sharp metric ---------- */
/** Per-token chance of being right at log10-compute x. sharp > 0 adds a genuine phase change at x0. */
export function tokenAcc(x, { mid = 21.5, slope = 1.1, sharp = 0, x0 = 22.5 } = {}) {
  const smooth = sig(slope * (x - mid));
  if (!sharp) return smooth;
  return smooth * (0.35 + 0.65 * sig(sharp * (x - x0)));
}
/** Metrics for a task whose answer is L tokens long. */
export function emergeMetrics(x, L, opts) {
  const p = tokenAcc(x, opts);
  return { token: p, exact: Math.pow(p, L), logp: L * Math.log(Math.max(p, 1e-12)), partial: p };
}
