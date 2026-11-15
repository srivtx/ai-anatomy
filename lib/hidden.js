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

// ---------------------------------------------------------------------------
// Hallucination from statistics: how many questions are about facts seen 0, 1, or < k times.
// F facts with Zipf-like popularity; a corpus mentions fact i Poisson(M · w_i) times.
// ---------------------------------------------------------------------------
const poissonCdf = (k, lam) => { if (k < 0) return 0; let term = Math.exp(-lam), s = term; for (let i = 1; i <= k; i++) { term *= lam / i; s += term; } return Math.min(1, s); };
export function factStats({ M = 1e6, F = 1e6, alpha = 1.1, k = 3, queries = "popular" } = {}) {
  // facts grouped into log-spaced rank bins so a million facts costs a few hundred bins
  const bins = [], B = 400; let prev = 0, Z = 0;
  for (let b = 1; b <= B; b++) { const hi = Math.max(prev + 1, Math.round(Math.pow(F, b / B))); if (hi <= prev) continue; const mid = Math.sqrt((prev + 1) * hi), n = hi - prev; bins.push({ n, w: Math.pow(mid, -alpha) }); Z += n * Math.pow(mid, -alpha); prev = hi; }
  let unseen = 0, notKnown = 0, n1 = 0, tot = 0, factsUnseen = 0;
  for (const b of bins) {
    const w = b.w / Z, lam = M * w, q = queries === "popular" ? w : 1 / F;
    const p0 = Math.exp(-lam), p1 = lam * p0, pk = poissonCdf(k - 1, lam);
    unseen += b.n * q * p0; notKnown += b.n * q * pk; n1 += b.n * p1; tot += b.n * lam; factsUnseen += b.n * p0;
  }
  return { unseen, notKnown, goodTuring: n1 / tot, factsUnseen: factsUnseen / F };
}

/** Expected score of answering only when confidence ≥ t, with +1 right, −pen wrong, 0 for "I don't know". */
export function abstainScore({ t = 0, pen = 0, known = 0.6 } = {}) {
  // confidence grid: a share `known` sits near certainty, the rest spread over 0–0.6
  let score = 0, wrong = 0, idk = 0, right = 0; const G = 400;
  for (let i = 0; i < G; i++) {
    const u = (i + 0.5) / G, p = u < known ? 0.85 + 0.15 * ((u / known) ** 0.5) : 0.6 * ((u - known) / (1 - known));
    if (p >= t) { score += p - (1 - p) * pen; right += p; wrong += 1 - p; } else idk++;
  }
  return { score: score / G, wrong: wrong / G, right: right / G, idk: idk / G };
}

// ---------------------------------------------------------------------------
// The reversal curse, trained for real: a tiny network learns "A_i r → B_i".
// ---------------------------------------------------------------------------
export function reversalNet({ P = 40, d = 32, revFrac = 0.5, extra = false, tied = false, seed = 7 } = {}) {
  const r = rng(seed), V = 2 * P + (extra ? 8 : 0), R = 3;              // entities (+ 8 attribute tokens), relations: forward, backward, "is known for"
  const rnd = (s) => (r() * 2 - 1) * s;
  const E = Array.from({ length: V }, () => Array.from({ length: d }, () => rnd(0.5)));
  const Rl = Array.from({ length: R }, () => Array.from({ length: d }, () => rnd(0.5)));
  const U = tied ? E : Array.from({ length: V }, () => Array.from({ length: d }, () => rnd(0.3)));
  const ent = 2 * P, nRev = Math.round(P * revFrac);
  const train = [];
  for (let i = 0; i < P; i++) train.push([i, 0, P + i]);                  // "A_i is the parent of B_i"
  for (let i = 0; i < nRev; i++) train.push([P + i, 1, i]);              // "B_i is the child of A_i", for some pairs
  if (extra) for (let i = 0; i < P; i++) train.push([P + i, 2, ent + (i % 8)]);  // B_i also appears as a subject elsewhere
  const tests = { forward: [...Array(P).keys()].map((i) => [i, 0, P + i]), revTrained: [...Array(nRev).keys()].map((i) => [P + i, 1, i]), revHeld: [...Array(P - nRev).keys()].map((j) => [P + nRev + j, 1, nRev + j]) };
  const m = new Map(), v = new Map(); let step = 0;
  const forward = (a, rel) => { const z = E[a].map((x, j) => Math.tanh(x + Rl[rel][j])); const lg = []; for (let o = 0; o < ent; o++) { let s = 0; for (let j = 0; j < d; j++) s += U[o][j] * z[j]; lg.push(s); } return { z, lg }; };
  const adam = (key, arr, j, g, lr) => { const k = key + j; const mm = 0.9 * (m.get(k) || 0) + 0.1 * g, vv = 0.999 * (v.get(k) || 0) + 0.001 * g * g; m.set(k, mm); v.set(k, vv); arr[j] -= (lr * (mm / (1 - 0.9 ** step))) / (Math.sqrt(vv / (1 - 0.999 ** step)) + 1e-8); };
  function trainSteps(n, lr = 0.02) {
    for (let s = 0; s < n; s++) {
      step++;
      const gE = new Map(), gR = Array.from({ length: R }, () => new Float64Array(d)), gU = new Map();
      for (const [a, rel, b] of train) {
        const nOut = rel === 2 ? V : ent;
        const z = E[a].map((x, j) => Math.tanh(x + Rl[rel][j]));
        const lg = []; for (let o = 0; o < nOut; o++) { let sum = 0; for (let j = 0; j < d; j++) sum += U[o][j] * z[j]; lg.push(sum); }
        const mx = Math.max(...lg), ex = lg.map((x) => Math.exp(x - mx)), Zs = ex.reduce((p, q) => p + q, 0);
        const dz = new Float64Array(d);
        for (let o = 0; o < nOut; o++) { const g = ex[o] / Zs - (o === b ? 1 : 0); if (Math.abs(g) < 1e-6) continue; const gu = gU.get(o) || new Float64Array(d); for (let j = 0; j < d; j++) { gu[j] += g * z[j]; dz[j] += g * U[o][j]; } gU.set(o, gu); }
        const ge = gE.get(a) || new Float64Array(d);
        for (let j = 0; j < d; j++) { const pre = dz[j] * (1 - z[j] * z[j]); ge[j] += pre; gR[rel][j] += pre; }
        gE.set(a, ge);
      }
      const sc = 1 / train.length;
      for (const [o, g] of gU) for (let j = 0; j < d; j++) adam(`u${o}:`, U[o], j, g[j] * sc + (tied ? (gE.get(o)?.[j] || 0) * sc : 0), lr);
      for (const [a, g] of gE) if (!(tied && gU.has(a))) for (let j = 0; j < d; j++) adam(`e${a}:`, E[a], j, g[j] * sc, lr);
      for (let rel = 0; rel < R; rel++) for (let j = 0; j < d; j++) adam(`r${rel}:`, Rl[rel], j, gR[rel][j] * sc, lr);
    }
  }
  const acc = (set) => { if (!set.length) return null; let ok = 0; for (const [a, rel, b] of set) { const { lg } = forward(a, rel); if (lg.indexOf(Math.max(...lg)) === b) ok++; } return ok / set.length; };
  const rankOf = (set) => { if (!set.length) return null; let s = 0; for (const [a, rel, b] of set) { const { lg } = forward(a, rel); s += lg.filter((x) => x > lg[b]).length + 1; } return s / set.length; };
  return { trainSteps, measure: () => ({ step, forward: acc(tests.forward), revTrained: acc(tests.revTrained), revHeld: acc(tests.revHeld), heldRank: rankOf(tests.revHeld) }), chance: 1 / ent, ent };
}
