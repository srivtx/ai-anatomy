// The Speed tab: what makes this design faster, what doesn't, and why. Every number comes from the same
// physics as the Predict tab: decoding is limited by how many bytes must be read from memory per token,
// training by how many FLOPs each token needs.
import { h } from "../core.js";
import { full, count, blk, bytes } from "./arch.js";
import { H100, MFU } from "./predict.js";
import { WIDTHS } from "./edit.js";

/** Speed numbers for a design. mods: serving tricks that don't change the architecture. */
export function metrics(cfg, mods = {}) {
  const c = full(cfg), k = count(c), wB = (mods.wBits || 16) / 8, kvB = (mods.kvBits || 16) / 8;
  const N = k.total, weights = N * wB, used = k.active * wB;   // all experts sit in memory; one token reads only its own
  let kvTok = 0;
  const kvAt = (len) => { let s = 0; for (let i = 1; i <= c.L; i++) { const a = blk(c, i).attn; if (a !== "none") s += 2 * k.kvd * kvB * Math.min(len, a === "window" ? c.window : len); } return s; };
  for (let i = 1; i <= c.L; i++) if (blk(c, i).attn !== "none") kvTok += 2 * k.kvd * kvB;
  const attnAt = (len) => { let s = 0; for (let i = 1; i <= c.L; i++) { const a = blk(c, i).attn; if (a !== "none") s += 4 * c.d * Math.min(len, a === "window" ? c.window : len); } return s; };
  const mat = 2 * (k.nonEmbActive + c.d * c.vocab), mid = c.ctx / 2;
  const kvFull = kvAt(c.ctx), kvMid = kvAt(mid), flopTok = mat + attnAt(mid);
  const fits = weights + kvFull < H100.mem;
  const decode1 = fits ? (H100.bw / (used + kvMid)) * (mods.spec || 1) : 0;
  const users = fits ? Math.floor((H100.mem - weights) / kvFull) : 0;
  const B = Math.max(1, users), tMem = (weights + B * kvMid) / H100.bw, tCmp = (B * flopTok) / (H100.flops * 0.6);
  const all = fits ? B / Math.max(tMem, tCmp) : 0;
  const train = (H100.flops * MFU) / (3 * flopTok);
  // without FlashAttention, every head writes and reads its T×T score matrix: about 8 bytes per score, forward and back
  let naiveBytes = 0;
  for (let i = 1; i <= c.L; i++) { const a = blk(c, i).attn; if (a !== "none") naiveBytes += c.heads * Math.min(c.ctx, a === "window" ? c.window : c.ctx) * 8; }
  const trainNaive = 1 / (1 / train + naiveBytes / H100.bw);
  return { N, weights, used, kvTok, kvFull, kvMid, mat, attn: attnAt(mid), flopTok, fits, decode1, users, all, train, trainNaive, memBound: tMem >= tCmp };
}

const near = (list, v) => list.reduce((b, x) => (Math.abs(x - v) < Math.abs(b - v) ? x : b), list[0]);
const divs = (n) => Array.from({ length: n }, (_, i) => i + 1).filter((i) => n % i === 0);

/** Every lever, with the change it makes. patch: an architecture change you can apply. mods: a serving trick. */
function levers(c) {
  const L = [], k = count(c);
  if (c.kv > 1 && c.heads >= 4) {
    const kv = c.kv === c.heads ? near(divs(c.heads), c.heads / 4) : near(divs(c.heads), c.kv / 2);
    if (kv < c.kv) L.push({ id: "gqa", name: c.kv === c.heads ? "Share keys and values between heads" : "Share them more", patch: { kv },
      quality: "Tiny loss", what: `${c.kv} → ${kv} K/V groups. Each token stores ${c.kv / kv}× less in the cache.`,
      why: "Decoding re-reads the whole KV cache for every new token. Fewer K/V groups means fewer bytes to read and more users fit in memory. Queries stay separate, so heads still ask different questions. Llama 2 70B and Llama 3 all do this." });
  }
  if (c.ctx >= 4096 && !k.winL && k.attnL > 1) {
    const w = Math.min(4096, c.ctx / 4), blocks = { ...c.blocks };
    for (let i = 1; i <= c.L; i += 2) blocks[i] = { ...blk(c, i), attn: "window" };
    L.push({ id: "win", name: "Sliding window on every other block", patch: { blocks, window: w }, quality: "Small loss far away",
      what: `Half the blocks only look back ${w.toLocaleString("en-US")} tokens. Their cache stops growing there.`,
      why: "Most of what predicts the next word is nearby. Windowed blocks handle that cheaply; the full blocks in between still connect distant words. Gemma 2 alternates like this; Mistral 7B windowed every layer." });
  }
  if (!k.moeL && c.L >= 4) L.push({ id: "moe", name: "Mixture of experts (8, top 2)", patch: { experts: 8, topk: 2, blocks: Object.fromEntries(Array.from({ length: c.L }, (_, i) => [i + 1, { ...blk(c, i + 1), mlp: blk(c, i + 1).mlp ? "moe" : false }])) },
    quality: "Better per FLOP", what: "Every MLP becomes 8 experts; each token uses 2. About 4× the MLP parameters at about the same compute per token.",
    why: "It's a speed-up in disguise: you get the quality of a much bigger model while each token only pays for 2 experts. One user's decoding reads only the chosen experts, but all 8 must fit in memory, so fewer users fit and batches touch every expert. Mixtral 8×7B matches models with 5× its active size." });
  L.push({ id: "w8", name: "Store weights in 8 bits", mods: { wBits: 8 }, quality: "≈ same", what: "Two bytes per weight become one.",
    why: "One user's decoding reads every weight once per token, so half the bytes is nearly twice the speed. Matmuls still run in 16-bit after unpacking. Well-tested methods (LLM.int8, FP8) lose almost nothing." });
  L.push({ id: "w4", name: "Store weights in 4 bits", mods: { wBits: 4.5 }, quality: k.total < 3e9 ? "Noticeable loss at this size" : "Small loss",
    what: "About 4.5 bits per weight, counting the scale numbers.", why: "Even fewer bytes per token, so faster still. The rounding error is bigger, and small models feel it more than big ones (GPTQ, AWQ)." });
  L.push({ id: "kv8", name: "Store the KV cache in 8 bits", mods: { kvBits: 8 }, quality: "≈ same", what: "Half the cache per user.",
    why: "Doesn't change one user's speed much when weights dominate, but twice as many conversations fit in the memory that's left." });
  L.push({ id: "spec", name: "Speculative decoding", mods: { spec: 2.5 }, quality: "Identical outputs", what: "A small model drafts ~4 tokens; this one checks them all in one pass.",
    why: "Checking 4 tokens costs about the same as producing 1, because the weights are read once either way. When most drafts are right you get 2–3× per user, and the maths guarantees the same output distribution." });
  if (c.L >= 8) {
    const d = near(WIDTHS, c.d * Math.SQRT2), L2 = Math.max(1, Math.round(c.L / 2)), heads = near(divs(d).filter((x) => d / x >= 32 && d / x <= 256), c.heads * (d / c.d));
    const kv = c.kv === c.heads ? heads : near(divs(heads), heads / (c.heads / c.kv));
    L.push({ id: "wide", name: "Half the blocks, wider", patch: { L: L2, d, ff: Math.round(d * (c.ff / c.d)), heads, kv, blocks: {} }, quality: "Law: about the same",
      what: `${c.L} → ${L2} blocks, width ${c.d.toLocaleString("en-US")} → ${d.toLocaleString("en-US")}. About the same parameters.`,
      why: "Blocks run one after another, so fewer of them means fewer sequential steps per token: lower latency when the GPU isn't busy anyway. Memory bytes stay the same, so one user's ceiling barely moves. Kaplan et al. found loss depends little on this shape within 20–200 width per layer." });
  }
  if (!c.tied && c.vocab > 30000 && k.head / k.total > 0.15) L.push({ id: "tie", name: "Tie the readout to the embedding", patch: { tied: true }, quality: "Small loss, or none",
    what: `Saves ${Math.round((100 * k.head) / k.total)}% of the parameters.`, why: "The same table maps token → vector and vector → token scores. Fewer bytes to read per token. Common in small models (GPT-2, Gemma)." });
  return L;
}

const DONT = [
  ["RMSNorm instead of LayerNorm", "Saves one subtraction per number. Norms are under 1% of the FLOPs; the gain is a few percent at most, from fewer memory passes."],
  ["More heads at the same width", "Same matrices, cut into more pieces: identical FLOPs and bytes. It changes what the model can express, not its speed."],
  ["ReLU or GELU instead of SwiGLU", "Activations are a rounding error next to the matmuls. Speed comes from the matrix sizes."],
  ["A chip with more FLOPs but the same memory", "One user's decoding waits on memory, not maths. Twice the FLOPs gives the same tokens per second; only bandwidth or fewer bytes helps."],
  ["Removing biases", "Biases are a tiny fraction of the weights. Llama drops them for simplicity and stability, not speed."],
];

const x = (a, b) => {
  if (!b && !a) return { t: "—", g: null };
  if (!b) return { t: "now fits", g: true };
  if (!a) return { t: "won't fit", g: false };
  const r = a / b;
  if (Math.abs(r - 1) < 0.03) return { t: "=", g: null };
  return { t: r >= 10 ? `×${Math.round(r)}` : r >= 1 ? `×${r.toFixed(r < 1.1 ? 2 : 1)}` : `÷${(1 / r).toFixed(1 / r < 1.1 ? 2 : 1)}`, g: r > 1 };
};
const flop = (v) => (v >= 1e12 ? `${(v / 1e12).toFixed(1)} TFLOP` : v >= 1e9 ? `${(v / 1e9).toFixed(1)} GFLOP` : v >= 1e6 ? `${(v / 1e6).toFixed(1)} MFLOP` : `${(v / 1e3).toFixed(1)} kFLOP`);
const tps = (v) => (v >= 1e6 ? `${(v / 1e6).toFixed(1)}M` : v >= 1e4 ? `${Math.round(v / 1e3)}k` : Math.round(v).toLocaleString("en-US"));

/** One H100's 80 GB as 80 pixel squares: the weights, then one block of cache per full-length chat, then what's left. */
function memMap(m) {
  const GB = 1e9, cells = 80, w = Math.min(cells, Math.ceil(m.weights / GB)), over = Math.max(0, Math.ceil((m.weights + m.kvFull) / GB) - cells);
  const per = m.kvFull / GB, users = m.fits ? m.users : 0, out = [];
  for (let i = 0; i < cells; i++) {
    if (i < w) { out.push(["w", "Weights"]); continue; }
    const j = i - w;
    if (!m.fits) { out.push(["x", "Not enough room"]); continue; }
    if (per >= 1) { const u = Math.floor(j / Math.ceil(per)); out.push(u < users ? [u % 2 ? "k2" : "k", `Chat ${u + 1}'s KV cache`] : ["f", "Free"]); }
    else { const filled = Math.floor(users * per); out.push(j < filled ? [j % 2 ? "k2" : "k", `KV caches, about ${Math.round(1 / per)} chats per square`] : ["f", "Free"]); }
  }
  const say = !m.fits ? `The weights alone need ${bytes(m.weights)}${over ? `: ${over} GB more than the card has` : ", leaving no room for a chat"}.`
    : per >= 1 ? `Each square is 1 GB. Each full chat needs ${bytes(m.kvFull)} of cache, so ${users} fit beside the weights.`
    : `Each square is 1 GB. A full chat needs only ${bytes(m.kvFull)} of cache, so about ${Math.round(1 / per)} share each square: ${users.toLocaleString("en-US")} at once.`;
  return h("div.sp-mem",
    h("div.st-sub", "One H100's 80 GB"),
    h("div.sp-mem-g", { role: "img", "aria-label": say }, out.map(([k, t], i) => h("i", { "data-k": k, title: t, style: { "--i": i } }))),
    h("p.sp-leg", h("span", h("i", { style: "background:#7b61ff" }), "Weights"), h("span", h("i", { style: "background:#22b8a0" }), "KV caches"), h("span", h("i.sp-free"), "Free")),
    h("p.st-small.muted", say));
}

/** The Speed tab. apply(patch) changes the design. */
export function speedPane(cfg, apply) {
  const c = full(cfg), m = metrics(c);
  const bar = (parts) => {
    const tot = parts.reduce((a, p) => a + p[1], 0) || 1;
    return h("div.sp-bar", parts.map(([label, v, col]) => h("span", { style: `flex:${v / tot};background:${col}`, title: `${label}: ${Math.round((100 * v) / tot)}%` })));
  };
  const chip = (label, r) => h("span.sp-x", { "data-g": r.g == null ? "n" : r.g ? "1" : "" }, `${label} ${r.t}`);
  const rows = levers(c).map((lv) => {
    const after = lv.patch ? metrics({ ...c, ...lv.patch }) : metrics(c, lv.mods);
    return h("li.sp-lever",
      h("div.sp-top", h("b", lv.name), lv.patch ? h("button.btn.small", { type: "button", onclick: () => apply(lv.patch) }, "Try it") : h("span.sp-kind", "serving")),
      h("p.sp-what", lv.what),
      h("div.sp-xs", chip("1 user", x(after.decode1, m.decode1)), chip("Users", x(after.users, m.users)), chip("Training", x(after.train, m.train)), h("span.sp-x", { "data-g": "q" }, lv.quality)),
      h("details", h("summary", "Why"), h("p", lv.why)));
  });
  return h("div.st-pane-in",
    h("div.st-card",
      h("div.eyebrow", "How fast, on one H100"),
      h("div.sp-tiles",
        h("div", h("span", "One user"), h("b", m.fits ? `${tps(m.decode1)}` : "—"), h("small", "tokens/s, best case")),
        h("div", h("span", "Users at once"), h("b", m.fits ? tps(m.users) : "0"), h("small", `full ${c.ctx.toLocaleString("en-US")}-token chats`)),
        h("div", h("span", "Training"), h("b", tps(m.train)), h("small", "tokens/s"))),
      m.fits ? null : h("p.st-small", { style: "color:var(--bad)" }, `Doesn't fit: ${bytes(m.weights + m.kvFull)} needed, 80 GB available.`),
      memMap(m),
      h("div.st-sub", "Each new token reads"),
      bar([["Weights", m.used, "#7b61ff"], ["KV cache", m.kvMid, "#22b8a0"]]),
      h("p.sp-leg", h("span", h("i", { style: "background:#7b61ff" }), m.used < m.weights ? `Weights used ${bytes(m.used)} of ${bytes(m.weights)}` : `Weights ${bytes(m.weights)}`), h("span", h("i", { style: "background:#22b8a0" }), `KV cache ${bytes(m.kvMid)}, half-way through a chat`)),
      h("div.st-sub", "Each token's maths"),
      bar([["Matmuls", m.mat, "#ff8a3d"], ["Attention scores", m.attn, "#4f8ef7"]]),
      h("p.sp-leg", h("span", h("i", { style: "background:#ff8a3d" }), `Matmuls ${Math.round((100 * m.mat) / m.flopTok)}%`), h("span", h("i", { style: "background:#4f8ef7" }), `Attention scores ${Math.round((100 * m.attn) / m.flopTok)}%`)),
      h("p.st-small.muted", `One user is ${m.memBound ? "memory-bound: the chip waits for bytes, so fewer bytes is the only real speed-up" : "compute-bound"}. Training is compute-bound: it needs ${flop(3 * m.flopTok)} per token, forward and backward.${m.train / m.trainNaive > 1.15 ? ` FlashAttention is assumed: without it, writing out every attention score would make training ${(m.train / m.trainNaive).toFixed(1)}× slower at this context.` : ""}`)),
    h("div.st-card",
      h("div.eyebrow", "What speeds it up"),
      h("p.st-small.muted", "Each lever against this design. “Try it” changes the model; serving tricks don't."),
      h("ul.sp-levers", rows)),
    h("div.st-card",
      h("div.eyebrow", "What doesn't"),
      h("ul.sp-dont", DONT.map(([t, w]) => h("li", h("b", t), h("p", w))))));
}
