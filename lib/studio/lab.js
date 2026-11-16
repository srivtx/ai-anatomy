// The lab: turn a Studio design into a small real training run (notebook 61 on Colab), predict how it will
// compare with the model you started from, and read the measured result back.
//
// A lab run keeps the design's structure (which blocks have attention or an MLP, residuals, norms and where they
// sit, activation, positions, head size, K/V sharing, windows) and shrinks its size, so it fits a free T4.
import { full, blk } from "./arch.js";

export const SIZES = {
  S: { label: "Quick", mins: "about 2 min", d: 128, cap: 6, ctx: 128, steps: 1500, batch: 64, lr: 1e-3, every: 100, seeds: 2 },
  M: { label: "Standard", mins: "about 8 min", d: 256, cap: 8, ctx: 256, steps: 3000, batch: 32, lr: 6e-4, every: 150, seeds: 2 },
  L: { label: "Long", mins: "about 25 min", d: 384, cap: 12, ctx: 256, steps: 5000, batch: 48, lr: 5e-4, every: 250, seeds: 1 },
};
// Loss the GPT-2 recipe reaches at each size on Shakespeare characters (measured; see notebook 61).
export const BASE_LOSS = { S: 1.62, M: 1.47, L: 1.38 };
export const NOTEBOOK = "61_studio_lab";
/** Our own runs of notebook 61 at the Quick length (one seed, CPU): what each rule predicted, and what happened. */
export const MEASURED = [
  { name: "GPT-2 recipe (the baseline)", c: 0, loss: 1.622 },
  { name: "No residual stream", c: 0.54, loss: 3.342, note: "Stalled at letter frequencies. The rule was raised to match." },
  { name: "No attention anywhere", c: 0.85, loss: 2.482 },
  { name: "No MLP anywhere", c: 0.2, loss: 1.984, note: "Worse than we guessed. The rule was raised to match." },
  { name: "Llama recipe (SwiGLU, RoPE, RMSNorm, one wide head)", c: -0.025, loss: 1.502, note: "Better than we guessed: SwiGLU and RoPE help more on characters. The rules were raised to match." },
  { name: "Post-norm (the 2017 original)", c: 0, loss: 1.835, note: "We expected a tie at 6 blocks. In a short run it trailed by 0.21. The rule was raised to match." },
  { name: "2 blocks instead of 6", c: 0.24, loss: 1.751, note: "Cheaper than we guessed: in a short run, extra depth isn't fully used yet. The rule was lowered to match." },
  { name: "No normalisation", c: 0.08, loss: 2.102, note: "Much worse than we guessed: norms matter even at 6 blocks. The rule was raised to match." },
];

const divisors = (n) => Array.from({ length: n }, (_, i) => i + 1).filter((i) => n % i === 0);
const nearest = (list, v) => list.reduce((b, x) => (Math.abs(Math.log(x / v)) < Math.abs(Math.log(b / v)) ? x : b), list[0]);

/** Which original block each lab block copies. Repeating patterns (every other block) keep their rhythm. */
export function blockMap(c, cap) {
  const L = c.L, n = Math.min(L, cap);
  if (n === L) return Array.from({ length: L }, (_, i) => i + 1);
  const same = (a, b) => JSON.stringify(blk(c, a)) === JSON.stringify(blk(c, b));
  for (let p = 1; p <= 4; p++) {
    let ok = true;
    for (let i = 1; i + p <= L && ok; i++) ok = same(i, i + p);
    if (ok) return Array.from({ length: n }, (_, j) => (j % p) + 1);
  }
  return Array.from({ length: n }, (_, j) => Math.min(L, Math.floor((j * L) / n) + 1));
}

/** The small model a lab run trains for design c. */
export function labOf(cfg, size = "S") {
  const c = full(cfg), S = SIZES[size];
  const hd = c.d / c.heads, hdLab = Math.max(16, Math.min(S.d, hd));
  const heads = nearest(divisors(S.d).filter((x) => S.d / x >= 16), S.d / hdLab);
  const kv = c.kv === c.heads ? heads : c.kv === 1 ? 1 : nearest(divisors(heads), heads / (c.heads / c.kv));
  const map = blockMap(c, S.cap);
  const window = c.window >= c.ctx ? S.ctx : Math.max(4, Math.round((c.window / c.ctx) * S.ctx));
  return {
    d: S.d, heads, kv, ff: Math.max(8, Math.round((S.d * (c.ff / c.d)) / 8) * 8), act: c.act, norm: c.norm, normPos: c.normPos,
    residual: c.residual, pos: c.pos, tied: c.tied, bias: !!c.bias, window, experts: Math.min(8, c.experts), topk: Math.min(c.topk, c.experts, 8),
    blocks: map.map((i) => blk(c, i)), map,
  };
}

export const sameShape = (a, b) => JSON.stringify({ ...a, map: 0 }) === JSON.stringify({ ...b, map: 0 });

/** A short id for a design at a lab size: results are stored under it. */
export function labId(cfg, size) {
  const s = JSON.stringify([size, labOf(cfg, size)]);
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return (h >>> 0).toString(36);
}

const b64 = (s) => btoa(unescape(encodeURIComponent(s))).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");

/** The line to paste into notebook 61. */
export function labSpec(cfg, base, size, names) {
  const S = SIZES[size], strip = ({ map, ...m }) => m;
  const spec = {
    v: 1, id: labId(cfg, size), name: names.design, size,
    train: { steps: S.steps, batch: S.batch, ctx: S.ctx, lr: S.lr, every: S.every, seeds: S.seeds },
    runs: [{ key: "baseline", label: names.base, model: strip(labOf(base, size)) }, { key: "design", label: names.design, model: strip(labOf(cfg, size)) }],
  };
  return { spec, line: b64(JSON.stringify(spec)) };
}

// ---------------------------------------------------------------------------------------------
// What we expect the lab run to show. Each structural choice costs (or saves) some loss compared with the
// GPT-2 recipe at the same size: c is our best guess, s how unsure we are. Calibrated on the Quick size.
// ---------------------------------------------------------------------------------------------
function costs(m, size) {
  const S = SIZES[size], out = [], add = (key, what, c, s, why) => out.push({ key, what, c, s, why });
  const L = m.blocks.length, attnL = m.blocks.filter((b) => b.attn !== "none").length, mlpL = m.blocks.filter((b) => b.mlp).length;
  const parts = attnL + mlpL, hd = m.d / m.heads, ffr = m.ff / m.d;
  if (L < S.cap) { const c = 0.12 * Math.log(S.cap / L); add("depth", `${L} block${L > 1 ? "s" : ""} instead of ${S.cap}`, c, 0.3 * c + 0.02, "Fewer rounds of mixing and processing. Each block lets every character build on what the previous blocks worked out. Measured: 2 blocks instead of 6 cost 0.13, less than we guessed, because a short run can't make full use of depth anyway."); }
  if (!m.residual) { const c = parts >= 6 ? 1.7 : 0.28 * parts; add("residual", "No residual stream", c, parts >= 6 ? 0.15 : 0.5 * c + 0.05, parts >= 6 ? "Measured: with 6 blocks and no residuals, the model stalled at 3.34, no better than counting how common each letter is. Stacked parts without skips squash every position towards the same vector (rank collapse), so the input never reaches the output." : "Each part must pass the signal on through its own weights; with only a few parts some of it still gets through."); }
  if (m.norm === "None") add("norm", "No normalisation", m.residual ? 0.48 : 0.5, m.residual ? 0.12 : 0.25, "Vectors grow as blocks add to them, and nothing rescales them, so each layer sees inputs of a different size every step. Clipping keeps it alive, but in our run it learned far slower (measured +0.48).");
  if (m.norm !== "None" && m.normPos === "post") add("post", "Norm after the add (post-norm)", 0.21 + 0.03 * Math.max(0, L - 6), 0.08, "The norm sits on the highway itself, so every block rescales the whole stream and gradients shrink on the way down. Measured: even at 6 blocks it cost 0.21 in the same short run (1.84 against 1.62). Long warmups close some of the gap.");
  if (m.norm === "RMSNorm") add("rms", "RMSNorm instead of LayerNorm", 0, 0.015, "Skips subtracting the mean. Almost always a tie on loss; it's chosen for speed.");
  if (m.act === "SwiGLU") add("act", "SwiGLU MLP", -0.05, 0.03, "A gate decides which features pass. It also has a third matrix, so the MLP is 1.5× bigger here. Part of why our Llama-recipe run beat GPT-2's by 0.12.");
  if (m.act === "ReLU") add("act", "ReLU instead of GELU", 0.02, 0.02, "Hard zero below 0: some units stop learning. Usually a small loss.");
  if (m.pos === "RoPE") add("pos", "RoPE positions", -0.05, 0.03, "Positions as rotations: attention sees how far apart two characters are, not just where each one is. On characters, where distance matters more than place, this helped more than we guessed.");
  if (m.pos === "none") add("pos", "No position signal", 0.04, 0.04, "The causal mask still lets the model count, so it learns order, but more slowly.");
  if (attnL === 0) add("attn", "No attention anywhere", 0.85, 0.15, "Each character is predicted from itself alone: a table of what usually follows each letter. Shakespeare letters after letters gives about 2.45.");
  else if (attnL < L) { const k = L - attnL; add("attn", `${k} block${k > 1 ? "s" : ""} without attention`, (attnL === 1 ? 0.1 : 0) + 0.03 * k, 0.03 * k + 0.02, attnL === 1 ? "One attention layer can look things up but can't do two-step tricks like copying what followed last time (that needs two)." : "Fewer chances to mix information between characters."); }
  if (mlpL === 0) add("mlp", "No MLP anywhere", 0.36, 0.08, "Attention mixes, but nothing processes each position on its own: much less room to store patterns. Measured: 1.98 against 1.62 with MLPs.");
  else if (mlpL < L) { const k = L - mlpL; add("mlp", `${k} block${k > 1 ? "s" : ""} without an MLP`, 0.03 * k, 0.025 * k + 0.01, "Less room to store patterns."); }
  const win = m.blocks.filter((b) => b.attn === "window").length;
  if (win && m.window < S.ctx) { const per = m.window < 16 ? 0.05 : m.window < 48 ? 0.015 : 0.005; add("window", `${win} window layer${win > 1 ? "s" : ""} of ${m.window} characters`, per * win, per * win * 0.8 + 0.01, m.window < 16 ? "That's shorter than a line of verse: the model loses the context that predicts the next word." : "Most of what predicts the next character is nearby, so a window this size costs little."); }
  if (hd <= 16) add("heads", `Small heads (${hd} numbers each)`, 0.03, 0.03, "Each head compares only a few numbers. Many small heads ask many questions, each less precisely.");
  if (m.heads === 1) add("heads", "A single head", hd >= 96 ? 0.01 : 0.04, 0.03, hd >= 96 ? "One attention pattern per layer, but a wide one. In our Llama-recipe run (one 128-number head) it barely hurt." : "One attention pattern per layer: it can't look at two places at once.");
  if (m.kv < m.heads) add("kv", m.kv === 1 ? "One K/V for all heads" : "Grouped K/V", m.kv === 1 ? 0.02 : 0.01, 0.02, "Heads share what they can be found by. Saves memory; small cost in quality.");
  if (ffr < 2) add("ff", `Narrow MLP (${ffr.toFixed(1)}×)`, 0.05 * (2 - ffr) + 0.02, 0.03, "Fewer features per position.");
  if (ffr > 4.5) add("ff", `Wide MLP (${ffr.toFixed(1)}×)`, -Math.min(0.04, 0.01 * (ffr - 4)), 0.03, "More features per position, more parameters.");
  const moe = m.blocks.filter((b) => b.mlp === "moe").length;
  if (moe) add("moe", `Experts in ${moe} block${moe > 1 ? "s" : ""} (${m.topk} of ${m.experts})`, -0.015 * Math.log2(m.experts / m.topk) * (moe / L), 0.03, "More stored parameters at the same compute per character. At this small size and short run the gain is modest, and an unlucky router can waste experts.");
  if (!m.tied) add("tied", "Separate readout matrix", -0.005, 0.015, "With only 65 characters, the readout is tiny either way.");
  return out;
}

/** Expected gap (design minus baseline) in lab loss, with the reasons. */
export function expectLab(cfg, base, size = "S") {
  const d = costs(labOf(cfg, size), size), b = costs(labOf(base, size), size);
  const keyOf = (x) => `${x.key}|${x.what}`, bk = new Set(b.map(keyOf)), dk = new Set(d.map(keyOf));
  const items = [...d.filter((x) => !bk.has(keyOf(x))), ...b.filter((x) => !dk.has(keyOf(x))).map((x) => ({ ...x, c: -x.c, what: `Not: ${x.what.charAt(0).toLowerCase()}${x.what.slice(1)}`, undo: true }))];
  const c = items.reduce((a, x) => a + x.c, 0), noise = 0.02;
  const s = Math.sqrt(items.reduce((a, x) => a + x.s * x.s, 0) + noise * noise);
  const baseLoss = BASE_LOSS[size] + b.reduce((a, x) => a + x.c, 0);
  return { items, c, s, baseLoss, designLoss: baseLoss + c, verdict: verdictOf(c, s) };
}

export function verdictOf(c, s = 0) {
  if (Math.abs(c) <= Math.max(0.02, s / 2) && Math.abs(c) < 0.05) return { word: "About the same", tone: "n" };
  if (c > 0.3) return { word: "Much worse", tone: "bad" };
  if (c > 0) return { word: "A little worse", tone: "bad" };
  if (c < -0.3) return { word: "Much better", tone: "good" };
  return { word: "A little better", tone: "good" };
}

/** Read a STUDIO_RESULT line. Returns null if it isn't one. */
export function parseResult(text) {
  const m = String(text || "").match(/STUDIO_RESULT:([A-Za-z0-9_-]+)/);
  if (!m) return null;
  try {
    const s = m[1].replace(/-/g, "+").replace(/_/g, "/");
    const r = JSON.parse(decodeURIComponent(escape(atob(s + "=".repeat((4 - (s.length % 4)) % 4)))));
    return r && r.v === 1 && r.runs ? r : null;
  } catch { return null; }
}

const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
/** How the measured gap compares with the prediction. */
export function score(result, expect) {
  const B = result.runs.baseline, D = result.runs.design;
  if (!B || !D) return null;
  const bl = mean(B.finals), dl = mean(D.finals);
  const fin = (x) => Number.isFinite(x);
  const spread = Math.max(0.01, ...[B, D].map((r) => (r.finals.length > 1 ? Math.max(...r.finals) - Math.min(...r.finals) : 0.02)));
  if (!fin(dl) || D.diverged) return { bl, dl, gap: Infinity, spread, held: expect.c > 0.25, word: "It diverged", say: expect.c > 0.25 ? "The loss blew up, as the prediction feared." : "The loss blew up. The prediction didn't see that coming: look at the gradient and stream sizes for where it started." };
  const gap = dl - bl, lo = expect.c - expect.s - spread / 2, hi = expect.c + expect.s + spread / 2;
  const held = gap >= lo && gap <= hi, sign = Math.sign(gap) === Math.sign(expect.c) || Math.abs(gap) < spread;
  return { bl, dl, gap, spread, lo, hi, held, word: held ? "Prediction held" : sign ? "Right direction, wrong size" : "Surprise",
    say: held ? "The measured gap landed inside the predicted range." : sign ? `The direction was right but the size was off by ${Math.abs(gap - expect.c).toFixed(2)}.` : "The opposite of what we expected. That's what experiments are for: check the gradient and stream sizes below for why." };
}

/** Saved experiments, newest first. */
const KEY = "studio-lab-v1";
export const log = {
  all() { try { return JSON.parse(localStorage.getItem(KEY) || "[]"); } catch { return []; } },
  add(e) { try { const a = log.all().filter((x) => !(x.id === e.id && x.at === e.at)); a.unshift(e); localStorage.setItem(KEY, JSON.stringify(a.slice(0, 40))); } catch { /* private mode */ } },
  attach(id, sc) { try { const a = log.all(), e = a.find((x) => x.id === id); if (!e || e.gap === sc.gap) return; Object.assign(e, { gap: sc.gap, held: sc.held }); localStorage.setItem(KEY, JSON.stringify(a)); } catch { /* private mode */ } },
  clear() { try { localStorage.removeItem(KEY); } catch { /* private mode */ } },
};
