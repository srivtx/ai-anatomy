// The 3D labs: each is a voxel scene you configure, with the numbers computed from real formulas.
import { vxLab } from "./vxlab.js";
import { shade } from "./voxel.js";

const f1 = (x) => (x >= 100 ? Math.round(x).toLocaleString("en-US") : x >= 10 ? x.toFixed(1) : x.toFixed(2));
const big = (n) => n >= 1e12 ? `${f1(n / 1e12)} T` : n >= 1e9 ? `${f1(n / 1e9)} B` : n >= 1e6 ? `${f1(n / 1e6)} M` : n >= 1e3 ? `${f1(n / 1e3)} k` : String(Math.round(n));
const gb = (b) => `${f1(b / 1e9)} GB`;

// ---------------------------------------------------------------------------------------------
// Inference: will this model, at this context and this many users, fit on these GPUs?
// ---------------------------------------------------------------------------------------------
const LLAMA = { "8B": { p: 8.03e9, L: 32, kv: 8 }, "70B": { p: 70.6e9, L: 80, kv: 8 }, "405B": { p: 405.9e9, L: 126, kv: 8 } };
function fitNums(c) {
  const m = LLAMA[c.model], ctx = 1024 * 2 ** c.ctx;
  const w = (m.p * c.bits) / 8, perTok = 2 * m.L * m.kv * 128 * 2, user = perTok * ctx, kv = user * c.users;
  const cap = 80e9 * c.gpus, total = w + kv, free = cap - w;
  return { m, ctx, w, perTok, user, kv, cap, total, maxUsers: free > 0 ? Math.floor(free / user) : 0 };
}
export function fitLab() {
  return vxLab({
    title: "Will it fit?",
    controls: [
      { key: "model", label: "Model (Llama 3)", options: Object.keys(LLAMA).map((k) => ({ value: k, label: k })), value: "70B" },
      { key: "bits", label: "Weights", options: [{ value: 16, label: "16-bit" }, { value: 8, label: "8-bit" }, { value: 4, label: "4-bit" }], value: 16 },
      { key: "ctx", label: "Context per user", min: 0, max: 7, value: 3, fmt: (v) => `${2 ** v}k tokens` },
      { key: "users", label: "Users at once", min: 1, max: 64, value: 8 },
      { key: "gpus", label: "80 GB GPUs", options: [1, 2, 4, 8].map((g) => ({ value: g, label: String(g) })), value: 2 },
    ],
    presets: [
      { label: "Laptop-ish: 8B, 4-bit", set: { model: "8B", bits: 4, ctx: 2, users: 1, gpus: 1 } },
      { label: "70B on 1 GPU", set: { model: "70B", bits: 16, ctx: 3, users: 1, gpus: 1 } },
      { label: "Busy server", set: { model: "70B", bits: 8, ctx: 5, users: 48, gpus: 4 } },
    ],
    draw(s, c) {
      const n = fitNums(c), G = c.gpus, CELL = 2e9;
      const wCells = Math.round(n.w / G / CELL), kvCells = Math.max(0, Math.ceil(n.total / G / CELL) - wCells), perUser = Math.max(1e-9, n.user / G / CELL);
      for (let g = 0; g < G; g++) {
        const ox = (g % 4) * 10, oy = Math.floor(g / 4) * 8;
        s.box(ox, oy, 0, 8, 5, 1, "#1e5a3c", (i, j) => ({ faces: { top: (i + j) % 2 ? "#23684a" : "#1f5f42" } }));
        for (let cI = 0; cI < 80; cI++) {
          const isW = cI < wCells, isKV = !isW && cI < wCells + kvCells;
          if (cI >= 40 && !isW && !isKV) break;
          const x = ox + (cI % 40) % 8, y = oy + Math.floor((cI % 40) / 8), z = cI < 40 ? 1 : 2;
          const over = cI >= 40;
          if (isW) s.put(x, y, z, over ? "#e23b2e" : "#7b61ff", { hit: s.tip(over ? `<b>GPU ${g + 1}: doesn't fit.</b> These 2 GB of weights have nowhere to go.` : `<b>GPU ${g + 1}: weights.</b> Each GPU holds 1/${G} of every layer (tensor parallel). This block is 2 GB.`) });
          else if (isKV) {
            const u = Math.floor((cI - wCells) / perUser);
            s.put(x, y, z, over ? "#e23b2e" : u % 2 ? "#ffb070" : "#ff8a3d", { hit: s.tip(over ? `<b>GPU ${g + 1}: doesn't fit.</b> KV cache with nowhere to go.` : `<b>GPU ${g + 1}: KV cache${perUser >= 1 ? `, user ${Math.min(c.users, u + 1)}` : ""}.</b> Keys and values for every past token, every layer.`) });
          } else s.put(x, y, 1, "#2a2d36", { top: "#3a3e4a", hit: s.tip(`<b>GPU ${g + 1}: free.</b> 2 GB unused.`) });
        }
      }
    },
    read(c) {
      const n = fitNums(c), ok = n.total <= n.cap;
      return {
        stats: [["Weights", gb(n.w)], ["KV cache per user", gb(n.user)], [`Needed (you have ${n.cap / 1e9} GB)`, gb(n.total), ok ? "good" : "bad"], ["Most users that fit", n.w > n.cap ? "0" : n.maxUsers.toLocaleString("en-US"), ok ? "" : "bad"]],
        say: `<p>Each token of context costs <b>${f1(n.perTok / 1e6 * 1000)} KB</b> of cache: 2 (key and value) × ${n.m.L} layers × ${n.m.kv} KV heads × 128 numbers × 2 bytes. ${n.w > n.cap ? "The weights alone don't fit: add GPUs or use fewer bits." : ok ? "It fits. Real servers keep ~10% spare for activations and fragmentation." : "The weights fit but the caches don't: fewer users, shorter context, or more GPUs."} Purple = weights, orange = caches (alternating shades per user), red = spill-over.</p>`,
      };
    },
  });
}

// ---------------------------------------------------------------------------------------------
// MoE: route one token to the top-k of E experts
// ---------------------------------------------------------------------------------------------
const TOK = [["the", "#4f8ef7"], ["Paris", "#22b8a0"], ["def", "#ff8a3d"], ["= 3", "#e76fa8"]];
const hash = (a, b) => { let x = (a * 374761393 + b * 668265263) ^ 0x5bd1e995; x = Math.imul(x ^ (x >>> 13), 1274126177); return ((x ^ (x >>> 16)) >>> 0) / 4294967296; };
function route(c) {
  const sc = Array.from({ length: c.E }, (_, e) => (e % 4 === c.tok ? 1.4 : 0) + hash(e, c.tok) * 1.2);
  const top = sc.map((v, e) => [v, e]).sort((a, b) => b[0] - a[0]).slice(0, c.k);
  const z = top.reduce((a, [v]) => a + Math.exp(v), 0);
  return new Map(top.map(([v, e]) => [e, Math.exp(v) / z]));
}
export function moeLab() {
  const SHARED = 1.6e9, EXPERT = 5.64e9;   // Mixtral-like: attention and embeddings shared; one expert's FFNs across 32 layers
  return vxLab({
    title: "Route a token",
    controls: [
      { key: "tok", label: "Token", options: TOK.map(([t], i) => ({ value: i, label: t })), value: 1 },
      { key: "E", label: "Experts", options: [8, 16, 32, 64].map((e) => ({ value: e, label: String(e) })), value: 8 },
      { key: "k", label: "Top-k (experts per token)", min: 1, max: 8, value: 2 },
    ],
    presets: [{ label: "Mixtral 8 × 7B", set: { E: 8, k: 2 } }, { label: "Many small: 64, top 8", set: { E: 64, k: 8 } }, { label: "Dense-ish: all 8", set: { E: 8, k: 8 } }],
    draw(s, c) {
      const sel = route(c), cols = 8, rows = c.E / cols, col = TOK[c.tok][1];
      s.box(-1, -1, 0, cols * 2 + 1, rows * 2 + 5, 1, "#2a2d36", (i, j) => ({ faces: { top: (i + j) % 2 ? "#3a3e4a" : "#353944" } }));
      for (let e = 0; e < c.E; e++) {
        const x = (e % cols) * 2, y = Math.floor(e / cols) * 2, g = sel.get(e), topic = TOK[e % 4][1];
        if (g != null) {
          const hh = 2 + Math.round(g * 5), k = s.tip(`<b>Expert ${e + 1}: chosen.</b> Gate weight ${g.toFixed(2)}: its output counts for ${Math.round(g * 100)}% of this layer's answer.`);
          s.box(x, y, 1, 1, 1, hh, topic, () => ({ faces: { hit: k } }));
        } else {
          const k = s.tip(`<b>Expert ${e + 1}: idle</b> for this token. Its weights still sit in memory.`);
          s.box(x, y, 1, 1, 1, 1, "#5b6172", () => ({ faces: { top: shade(topic, 0.55), hit: k } }));
        }
      }
      const ry = rows * 2 + 1, rk = s.tip("<b>The router:</b> a small linear layer scores every expert for this token; the top-k win.");
      s.box(cols - 3, ry, 1, 4, 2, 1, "#7b61ff", () => ({ faces: { hit: rk } }));
      s.put(cols - 2, ry, 2, col, { top: shade(col, 1.4), hit: s.tip(`<b>The token “${TOK[c.tok][0]}”</b>, on its way through one MoE layer.`) });
    },
    read(c) {
      const total = SHARED + c.E * EXPERT, act = SHARED + c.k * EXPERT;
      return {
        stats: [["Total parameters", big(total)], ["Used per token", big(act)], ["Share used", `${Math.round((act / total) * 100)}%`, act / total < 0.4 ? "good" : ""], ["Weights in memory (16-bit)", gb(total * 2)]],
        say: `<p>Compute per token follows the <b>used</b> parameters; memory follows the <b>total</b>. Experts are coloured by what they've learned to like: try different tokens and watch the router pick mostly from one colour. (Each expert here is ~5.6 B parameters, Mixtral's size, so 64 experts is a 360 B model.)</p>`,
      };
    },
  });
}

// ---------------------------------------------------------------------------------------------
// Attention: shape the mask, see the softmax weights as heights
// ---------------------------------------------------------------------------------------------
const MASKS = { full: "Full", causal: "Causal", window: "Sliding window", prefix: "Prefix-LM" };
const allowed = (c, i, j) => c.mask === "full" ? true : c.mask === "causal" ? j <= i : c.mask === "window" ? j <= i && i - j < c.w : j <= i || j < c.w;
function attnW(c) {
  const W = [];
  for (let i = 0; i < c.n; i++) {
    const row = [];
    for (let j = 0; j < c.n; j++) row.push(allowed(c, i, j) ? Math.exp(-0.45 * Math.abs(i - j) + (j === 0 ? 0.6 : 0)) : 0);
    const z = row.reduce((a, b) => a + b, 0); W.push(row.map((v) => v / z));
  }
  return W;
}
export function maskLab() {
  return vxLab({
    title: "Shape the attention mask",
    controls: [
      { key: "mask", label: "Pattern", options: Object.entries(MASKS).map(([value, label]) => ({ value, label })), value: "causal" },
      { key: "n", label: "Sequence length", min: 4, max: 12, value: 8 },
      { key: "w", label: "Window / prefix size", min: 1, max: 8, value: 3 },
    ],
    draw(s, c) {
      const W = attnW(c);
      for (let i = 0; i < c.n; i++) for (let j = 0; j < c.n; j++) {
        if (!W[i][j]) { s.put(i, j, 0, "#3a3e4a", { top: "#4a4f5d", hit: s.tip(`<b>Masked:</b> token ${i + 1} may not look at token ${j + 1}${j > i ? " (it's in the future)" : " (outside the window)"}. Never computed.`) }); continue; }
        const hh = 1 + Math.round(W[i][j] * 6), cc = W[i][j] > 0.3 ? "#7b61ff" : W[i][j] > 0.12 ? "#9c88ff" : "#c3b8ff";
        const k = s.tip(`<b>Token ${i + 1} → token ${j + 1}:</b> weight ${W[i][j].toFixed(2)}${j === 0 ? ". The first token soaks up extra attention: an <i>attention sink</i>." : i === j ? ". Looking at itself." : "."}`);
        s.box(i, j, 0, 1, 1, hh, cc, () => ({ faces: { hit: k } }));
      }
    },
    read(c) {
      let on = 0; for (let i = 0; i < c.n; i++) for (let j = 0; j < c.n; j++) on += allowed(c, i, j) ? 1 : 0;
      const N = 131072, P = Math.round((N * c.w) / c.n);
      const atN = c.mask === "full" ? N * N : c.mask === "causal" ? (N * (N + 1)) / 2 : c.mask === "window" ? N * 4096 : (N * (N + 1)) / 2 + (P * (P - 1)) / 2;
      const why = { full: "Every token sees every token. Used by encoders like BERT, which read a whole text at once.", causal: "Each token sees itself and the past only, so the model can't cheat by reading the answer. Every GPT.", window: "Each token sees only the last few. Memory stays flat however long the text (Mistral, and some layers of Gemma).", prefix: "The prompt (first block) is read both ways, like an encoder; the answer after it is causal (T5-style prefix LMs)." }[c.mask];
      return {
        stats: [["Scores computed", `${on} of ${c.n * c.n}`], ["Share", `${Math.round((on / (c.n * c.n)) * 100)}%`], [`At 128k tokens${c.mask === "window" ? " (window 4,096)" : ""}`, big(atN)], ["As a 16-bit matrix", gb(atN * 2), atN * 2 > 80e9 ? "bad" : ""]],
        say: `<p>${why} Heights are real softmax weights over the allowed keys, each row summing to 1. (That last number is why FlashAttention never stores the matrix: it works through it in tiles.)</p>`,
      };
    },
  });
}

// ---------------------------------------------------------------------------------------------
// CNN: how far can one unit see after L layers?
// ---------------------------------------------------------------------------------------------
function rfSizes(c) {   // footprint of one top unit, at each layer's input, top first
  const out = [1]; let sz = 1;
  for (let l = 0; l < c.L; l++) { sz = (sz - 1) * (c.s === 2 ? 2 : 1) + c.k; out.push(sz); }
  return out;
}
export function rfLab() {
  return vxLab({
    title: "Grow a receptive field",
    controls: [
      { key: "L", label: "Conv layers", min: 1, max: 6, value: 3 },
      { key: "k", label: "Kernel", options: [3, 5, 7].map((k) => ({ value: k, label: `${k} × ${k}` })), value: 3 },
      { key: "s", label: "Stride", options: [{ value: 1, label: "1" }, { value: 2, label: "2 (downsample)" }], value: 1 },
    ],
    draw(s, c) {
      const F = rfSizes(c), N = 15, mid = Math.floor(N / 2), rf = F[F.length - 1];
      for (let x = 0; x < N; x++) for (let y = 0; y < N; y++) {
        const inR = Math.abs(x - mid) <= (rf - 1) / 2 && Math.abs(y - mid) <= (rf - 1) / 2;
        s.put(x, y, 0, inR ? "#ff8a3d" : "#c9d1e2", { top: inR ? ((x + y) % 2 ? "#ffb070" : "#ffa05a") : (x + y) % 2 ? "#eef1f7" : "#e3e8f1", hit: s.tip(inR ? `<b>Pixel inside the receptive field.</b> Changing it can change the top unit.` : `<b>Pixel outside.</b> The top unit can't see it at all.`) });
      }
      const pal = ["#22b8a0", "#4f8ef7", "#7b61ff", "#e76fa8", "#ffd23f", "#5fd176"];
      for (let l = 1; l < F.length - 1; l++) {
        const sz = Math.min(N, F[F.length - 1 - l]), half = (sz - 1) / 2, cc = pal[(l - 1) % pal.length], z = l;
        const k = s.tip(`<b>After layer ${l}:</b> the top unit depends on a ${F[F.length - 1 - l]} × ${F[F.length - 1 - l]} patch of this map.`);
        s.box(Math.ceil(mid - half), Math.ceil(mid - half), z, sz, sz, 1, cc, (i, j) => ({ faces: { top: shade(cc, (i + j) % 2 ? 1.25 : 1.1), hit: k } }));
      }
      s.put(mid, mid, F.length - 1, "#5fd176", { top: "#b6f5c4", hit: s.tip("<b>One unit at the top.</b> Everything it knows comes from the orange patch.") });
    },
    read(c) {
      const F = rfSizes(c), rf = F[F.length - 1], per = c.k * c.k * 64 * 64;
      return {
        stats: [["Receptive field", `${rf} × ${rf} px`, rf > 15 ? "good" : ""], ["Weights per layer (64→64 ch)", big(per)], ["All layers", big(per * c.L)], ["Dense layer, 224 × 224 × 64 → same", "10 T"]],
        say: `<p>${c.s === 2 ? "With stride 2 each layer doubles the step between samples, so the field grows <b>exponentially</b> with depth." : `With stride 1 each ${c.k} × ${c.k} layer adds ${c.k - 1} pixels: the field grows <b>linearly</b>, ${c.k - 1}·L + 1.`} ${rf > 15 ? "It's now wider than this 15 × 15 picture." : ""} That's how a network of tiny windows ends up seeing whole objects.</p>`,
      };
    },
  });
}

// ---------------------------------------------------------------------------------------------
// Gradient descent: roll a ball over a quadratic loss with your learning rate
// ---------------------------------------------------------------------------------------------
const VALLEY = { round: [1, 1], narrow: [1.8, 0.1] };
function gdPath(c) {
  const [a, b] = VALLEY[c.v], beta = c.m === "mom" ? 0.8 : 0;
  let x = -4.5, y = 5, vx = 0, vy = 0; const P = [[x, y]];
  for (let t = 0; t < c.steps; t++) {
    vx = beta * vx - c.lr * a * x; vy = beta * vy - c.lr * b * y; x += vx; y += vy; P.push([x, y]);
    if (!isFinite(x) || Math.abs(x) > 1e6) break;
  }
  return P;
}
const lossAt = (c, x, y) => { const [a, b] = VALLEY[c.v]; return 0.5 * (a * x * x + b * y * y); };
export function gdLab() {
  return vxLab({
    title: "Roll the ball",
    controls: [
      { key: "lr", label: "Learning rate", min: 0.05, max: 2.2, step: 0.05, value: 0.3, fmt: (v) => v.toFixed(2) },
      { key: "v", label: "Loss surface", options: [{ value: "round", label: "Round bowl" }, { value: "narrow", label: "Narrow valley" }], value: "narrow" },
      { key: "m", label: "Optimizer", options: [{ value: "gd", label: "Plain" }, { value: "mom", label: "Momentum" }], value: "gd" },
      { key: "steps", label: "Steps", min: 0, max: 30, value: 12 },
    ],
    presets: [{ label: "Too timid", set: { lr: 0.1, v: "narrow", m: "gd", steps: 30 } }, { label: "Zig-zag", set: { lr: 1.05, v: "narrow", m: "gd", steps: 20 } }, { label: "Blows up", set: { lr: 1.2, v: "narrow", m: "gd", steps: 10 } }, { label: "Momentum", set: { lr: 0.3, v: "narrow", m: "mom", steps: 20 } }],
    draw(s, c) {
      const N = 13, at = (i) => i - 6, P = gdPath(c);
      const H = (x, y) => 1 + Math.min(4, Math.floor(Math.sqrt(2 * lossAt(c, x, y)) / 1.9));
      const col = ["#2f6fd6", "#22b8a0", "#5fd176", "#ffd23f", "#ff8a3d"];
      const visit = new Map();
      P.forEach(([x, y], t) => { const i = Math.round(x) + 6, j = Math.round(y) + 6; if (i >= 0 && i < N && j >= 0 && j < N) visit.set(`${i},${j}`, t); });
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
        const hh = H(at(i), at(j)), t = visit.get(`${i},${j}`), cc = col[hh - 1];
        const k = s.tip(t != null ? `<b>Step ${t}:</b> loss ${lossAt(c, P[t][0], P[t][1]).toFixed(3)}` : `<b>Loss here:</b> ${lossAt(c, at(i), at(j)).toFixed(2)}`);
        s.box(i, j, 0, 1, 1, hh, cc, (a, b, z) => ({ faces: z === hh - 1 && t != null ? { top: t === 0 ? "#ffffff" : "#ffd0e0", hit: k } : { hit: k } }));
      }
      const [lx, ly] = P[P.length - 1], li = Math.round(lx) + 6, lj = Math.round(ly) + 6;
      if (li >= 0 && li < N && lj >= 0 && lj < N) s.put(li, lj, H(at(li), at(lj)), "#ff4f8b", { top: "#ffb3cc", hit: s.tip(`<b>The ball after ${P.length - 1} steps.</b>`) });
    },
    read(c) {
      const P = gdPath(c), [x, y] = P[P.length - 1], L = lossAt(c, x, y), L0 = lossAt(c, P[0][0], P[0][1]), [a] = VALLEY[c.v], lim = 2 / a;
      const blown = !isFinite(L) || L > L0;
      return {
        stats: [["Loss at start", L0.toFixed(2)], [`Loss after ${P.length - 1} steps`, isFinite(L) ? (L < 0.01 ? L.toExponential(1) : L.toFixed(3)) : "∞", blown ? "bad" : L < 0.05 * L0 ? "good" : ""], ["Steepest curvature λ", a.toFixed(1)], ["Plain GD breaks above 2/λ", lim.toFixed(2), c.lr > lim && c.m === "gd" ? "bad" : ""]],
        say: `<p>${blown ? "<b>It diverged.</b> Each step overshoots the steep direction by more than it corrects." : c.lr > 0.8 * lim && c.m === "gd" ? "Close to the limit: the ball bounces across the steep direction." : c.v === "narrow" && c.m === "gd" ? "Safe for the steep direction, but the flat floor of the valley barely moves: the step that suits one direction is wrong for the other." : "The ball settles toward the bottom."} White marks the start, pink the path; turn the picture to see behind the walls.</p>`,
      };
    },
  });
}

// ---------------------------------------------------------------------------------------------
// Transformer: size a model and see the tower
// ---------------------------------------------------------------------------------------------
export function sizeLab() {
  const D = [768, 2048, 4096, 8192, 12288, 16384];
  return vxLab({
    title: "Size a model",
    controls: [
      { key: "L", label: "Layers (blocks)", min: 2, max: 128, step: 2, value: 32 },
      { key: "d", label: "Width d_model", options: D.map((d) => ({ value: d, label: d.toLocaleString("en-US") })), value: 4096 },
    ],
    presets: [{ label: "GPT-2 small", set: { L: 12, d: 768 } }, { label: "Llama 3 8B", set: { L: 32, d: 4096 } }, { label: "Llama 3 70B", set: { L: 80, d: 8192 } }, { label: "GPT-3", set: { L: 96, d: 12288 } }],
    height: 380, fixed: 1.6,
    draw(s, c) {
      const w = 2 + Math.round(Math.log2(c.d / 768) * 1.2), per = 4, plates = Math.ceil(c.L / per), o = (w0) => Math.floor((12 - w0) / 2);
      const k0 = s.tip(`<b>Embedding:</b> one ${c.d.toLocaleString("en-US")}-number vector per token in the vocabulary.`);
      s.box(o(w + 2), o(w + 2), 0, w + 2, w + 2, 1, "#22b8a0", (i, j) => ({ faces: { top: (i + j) % 2 ? "#5fd1b8" : "#46c2a6", hit: k0 } }));
      for (let p = 0; p < plates; p++) {
        const a = p * per + 1, b = Math.min(c.L, (p + 1) * per), cc = p % 2 ? "#ff8a3d" : "#7b61ff";
        const k = s.tip(`<b>Block${a === b ? ` ${a}` : `s ${a}–${b}`}:</b> attention + MLP, ~${big(12 * c.d * c.d)} parameters each.`);
        s.box(o(w), o(w), 1 + p, w, w, 1, cc, (i, j) => ({ faces: { top: shade(cc, (i + j) % 2 ? 1.3 : 1.15), hit: k } }));
      }
      const kt = s.tip("<b>Readout:</b> turns the final vector into a score for every token in the vocabulary.");
      s.box(o(w + 2), o(w + 2), plates + 1, w + 2, w + 2, 1, "#5fd176", () => ({ faces: { hit: kt } }));
    },
    read(c) {
      const N = 12 * c.L * c.d * c.d + 50257 * c.d, per = 4;
      return {
        stats: [["Parameters ≈ 12·L·d²", big(N)], ["FLOPs per token ≈ 2N", big(2 * N)], ["Weights in 16-bit", gb(2 * N)], ["Train at 20 tokens/param", `${(6 * N * 20 * N).toExponential(1)} FLOPs`]],
        say: `<p>Width counts <b>squared</b>, depth only linearly: doubling d quadruples the parameters, doubling L doubles them. Each slab is ${per === 1 ? "one block" : `${per} blocks`}; purple and orange just alternate so you can count them. (The rule of thumb leaves out details like grouped-query attention and SwiGLU, so real counts differ by ~10–20%.)</p>`,
      };
    },
  });
}

// ---------------------------------------------------------------------------------------------
// Logic: watch a carry ripple through an adder, against a lookahead tree
// ---------------------------------------------------------------------------------------------
const adderDelay = (c) => (c.kind === "ripple" ? 2 * c.n : 2 * Math.ceil(Math.log2(c.n)) + 2);
export function adderLab() {
  return vxLab({
    title: "Watch a carry ripple",
    controls: [
      { key: "n", label: "Bits", options: [4, 8, 16, 32, 64].map((n) => ({ value: n, label: String(n) })), value: 16 },
      { key: "kind", label: "Adder", options: [{ value: "ripple", label: "Ripple carry" }, { value: "look", label: "Carry lookahead" }], value: "ripple" },
      { key: "t", label: "Time (gate delays)", min: 0, max: 130, value: 12 },
    ],
    draw(s, c) {
      const show = Math.min(c.n, 16), T = adderDelay(c), per = c.n / show;
      for (let b = 0; b < show; b++) {
        const x = (show - 1 - b) * 3, bit = Math.round(b * per);   // bit 0 at the right
        const ready = c.kind === "ripple" ? 2 * (bit + per) : T, done = c.t >= ready;
        const k = s.tip(`<b>Bit ${bit}${per > 1 ? `–${bit + per - 1}` : ""}:</b> ${done ? "settled, correct" : "still waiting for its carry; the output shown may be wrong"}. Ready at t = ${ready}.`);
        s.box(x, 0, 0, 2, 2, 2, done ? "#22b8a0" : "#5b6172", (i, j, z) => ({ faces: { top: z === 1 ? (done ? "#7fe0cc" : "#ff9a9a") : undefined, hit: k } }));
        if (b < show - 1) {
          const lit = c.t >= ready;
          s.put(x - 1, 1, 0, lit ? "#ffd23f" : "#3a3e4a", { top: lit ? "#fff0a8" : "#4a4f5d", hit: s.tip(`<b>Carry wire</b> from bit ${bit} to bit ${bit + 1}${lit ? ": carry has arrived." : ": nothing yet."}`) });
        }
      }
      if (c.kind === "look") {
        const lv = Math.ceil(Math.log2(c.n));
        for (let l = 0; l < lv; l++) {
          const lit = c.t >= 2 * (l + 1), k = s.tip(`<b>Lookahead level ${l + 1} of ${lv}:</b> combines carries over spans of ${2 ** (l + 1)} bits at once.`);
          s.box(0, 3 + l, 0, show * 3 - 1, 1, 1, lit ? "#7b61ff" : "#3a3e4a", (i) => ({ faces: { top: lit ? (i % 2 ? "#a996ff" : "#9c88ff") : "#4a4f5d", hit: k } }));
        }
      }
    },
    read(c) {
      const T = adderDelay(c), ok = c.t >= T, ps = T * 10;
      return {
        stats: [["Worst-case delay", `${T} gate delays`], ["At ~10 ps per gate", `${ps} ps`], ["Fastest clock if this were the whole cycle", `${f1(1000 / ps)} GHz`, ps > 300 ? "bad" : "good"], ["Result now", ok ? "correct" : "not ready", ok ? "good" : "bad"]],
        say: `<p>Worst case: <code>0111…1 + 1</code>, where the carry must travel the whole way. ${c.kind === "ripple" ? `Each bit waits for the one before: delay grows with n. 64 bits would take ${2 * 64} gate delays.` : `The lookahead tree works out carries for spans of 2, 4, 8 … bits in parallel: delay grows with log₂ n, at the cost of more gates.`} Drag the time slider to watch the bits settle. ${c.n > 16 ? `(Each block stands for ${c.n / 16} bits.)` : ""}</p>`,
      };
    },
  });
}

// ---------------------------------------------------------------------------------------------
// CUDA: launch a grid on a toy 4-SM GPU and watch the blocks land, wave by wave
// ---------------------------------------------------------------------------------------------
const BCOL = ["#7b61ff", "#22b8a0", "#ff8a3d", "#4f8ef7", "#e76fa8", "#5fd176", "#ffd23f", "#9c88ff"];
function launchNums(c) {
  const G = Math.ceil(c.N / c.B), perSM = Math.min(32, Math.floor(2048 / c.B)), wave = 4 * perSM;
  return { G, perSM, wave, waves: Math.ceil(G / wave), idle: G * c.B - c.N, occ: (perSM * c.B) / 2048, h100: Math.ceil(G / (132 * perSM)) };
}
export function launchLab() {
  return vxLab({
    title: "Launch a grid",
    controls: [
      { key: "N", label: "Elements (one thread each)", min: 100, max: 40000, step: 100, value: 10000, fmt: (v) => v.toLocaleString("en-US") },
      { key: "B", label: "Threads per block", options: [32, 64, 128, 256, 512, 1024].map((b) => ({ value: b, label: String(b) })), value: 256 },
    ],
    presets: [{ label: "Tiny blocks", set: { N: 10000, B: 32 } }, { label: "Just over a wave", set: { N: 8400, B: 256 } }, { label: "Big job", set: { N: 40000, B: 512 } }],
    draw(s, c) {
      const n = launchNums(c), wpb = c.B / 32;
      for (let sm = 0; sm < 4; sm++) {
        const ox = (sm % 2) * 10, oy = Math.floor(sm / 2) * 10, k = s.tip(`<b>SM ${sm + 1}</b> of this toy GPU: room for 64 warps (2,048 threads) and at most 32 blocks at once.`);
        s.box(ox, oy, 0, 8, 8, 1, "#2a2d36", (i, j) => ({ faces: { top: (i + j) % 2 ? "#3a3e4a" : "#353944", hit: k } }));
      }
      for (let b = 0; b < n.G; b++) {
        const wv = Math.floor(b / n.wave), inWave = b % n.wave, sm = inWave % 4, slot = Math.floor(inWave / 4);
        if (wv > 7) break;
        const ox = (sm % 2) * 10, oy = Math.floor(sm / 2) * 10, cc = BCOL[b % BCOL.length];
        for (let w = 0; w < wpb; w++) {
          const cell = slot * wpb + w, first = b * c.B + w * 32, used = Math.max(0, Math.min(32, c.N - first));
          const col = used === 0 ? "#e23b2e" : cc, top = used === 0 ? "#ff9a9a" : used < 32 ? "#ffd0d0" : shade(cc, 1.25);
          s.put(ox + (cell % 8), oy + Math.floor(cell / 8), 1 + wv, col, { top, hit: s.tip(`<b>Block ${b}, warp ${w}</b> on SM ${sm + 1}, wave ${wv + 1}. Threads ${first.toLocaleString("en-US")}–${(first + 31).toLocaleString("en-US")}: ${used === 32 ? "all 32 busy" : used === 0 ? "<b>all 32 idle</b> (past the end of the data)" : `${used} busy, ${32 - used} idle`}.`) });
        }
      }
    },
    read(c) {
      const n = launchNums(c);
      return {
        stats: [["Blocks", n.G.toLocaleString("en-US")], ["Idle threads", n.idle.toLocaleString("en-US"), n.idle >= 32 ? "bad" : "good"], ["Waves (toy 4 SMs)", String(n.waves), ""], ["Occupancy per SM", `${Math.round(n.occ * 100)}%`, n.occ < 1 ? "bad" : "good"]],
        say: `<p>Each block is one colour; each cube is a warp of 32 threads. An SM takes up to ${n.perSM} of these blocks at once${c.B <= 32 ? " (the 32-blocks-per-SM limit bites: half the warp slots stay empty)" : ""}; the rest wait and run in later <b>waves</b>, stacked upward. Pale tops are warps with some idle threads, red ones have none busy. On a real H100 (132 SMs) this launch is ${n.h100} wave${n.h100 > 1 ? "s" : ""}.</p>`,
      };
    },
  });
}

// ---------------------------------------------------------------------------------------------
// CUDA: a matrix multiply as a cube of multiply-adds, cut into tiles
// ---------------------------------------------------------------------------------------------
export function tileLab() {
  const n = 8, REAL = 4096, PEAK = 989e12, BW = 3.35e12;
  return vxLab({
    title: "Tile a matrix multiply",
    controls: [
      { key: "T", label: "Tile (toy / real)", options: [1, 2, 4, 8].map((t) => ({ value: t, label: `${t} / ${t * 16}` })), value: 4 },
      { key: "k", label: "Progress through k", min: 1, max: n, value: 5 },
    ],
    draw(s, c) {
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) for (let k = 0; k < c.k; k++) {
        const ti = Math.floor(i / c.T), tj = Math.floor(j / c.T), b = ti * (n / c.T) + tj, cc = BCOL[(ti * 3 + tj) % BCOL.length];
        const edge = k === c.k - 1;
        const tipK = edge ? s.tip(`<b>C[${i}][${j}]</b>, owned by block ${b} (tile ${ti}, ${tj}). After ${c.k} of ${n} steps along k it holds a partial sum: Σ A[${i}][k] · B[k][${j}] for k < ${c.k}.`) : undefined;
        s.put(i, j, k, cc, { top: edge ? shade(cc, 1.35) : undefined, hit: tipK });
      }
    },
    read(c) {
      const T = c.T * 16, flops = 2 * REAL ** 3, naive = 2 * REAL ** 3 * 2, tiled = naive / T, tC = flops / PEAK, tM = tiled / BW;
      return {
        stats: [[`Multiply-adds, ${REAL}³`, big(REAL ** 3)], ["Bytes from HBM, no reuse", gb(naive)], [`With ${T} × ${T} tiles`, gb(tiled), T >= 64 ? "good" : ""], [`Moving bytes vs maths (${tM > tC ? "memory-bound" : "compute-bound"})`, `${f1(tM * 1e3)} / ${f1(tC * 1e3)} ms`, tM > tC ? "bad" : "good"]],
        say: `<p>Every cube is one multiply-add; the whole cube is the matrix multiply (rows × columns × the shared k). Each colour is one block, which owns a T × T tile of the output and walks down k. A block loads its row strip of A and column strip of B <b>once</b> into shared memory and uses each number T times, so traffic falls by T. Even at 128 the bytes still take longer than the maths on an H100, which is why real kernels add register tiles, warp tiles and L2 reuse on top.</p>`,
      };
    },
  });
}

// ---------------------------------------------------------------------------------------------
// RL: value iteration on a maze; tower height = how much reward the agent expects from there
// ---------------------------------------------------------------------------------------------
const MAZE = ["#########", "#S..#...#", "#.#.#.#.#", "#.#...#.#", "#.#####.#", "#.......#", "#.###.#.#", "#...#.#G#", "#########"];
function valueIter(c) {
  const H = MAZE.length, W = MAZE[0].length, open = (x, y) => MAZE[y][x] !== "#";
  let V = MAZE.map((r) => [...r].map(() => 0));
  for (let k = 0; k < c.k; k++) {
    const N = V.map((r) => r.slice());
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      if (!open(x, y) || MAZE[y][x] === "G") continue;
      let best = 0;
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const nx = x + dx, ny = y + dy; if (!open(nx, ny)) continue;
        best = Math.max(best, MAZE[ny][nx] === "G" ? 1 : c.g * V[ny][nx]);
      }
      N[y][x] = best;
    }
    V = N;
  }
  // greedy path from the start
  const path = new Set(); let x = 1, y = 1;
  for (let t = 0; t < 40 && MAZE[y][x] !== "G"; t++) {
    let bx = -1, by = -1, bv = 0;
    for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      const nx = x + dx, ny = y + dy; if (!open(nx, ny)) continue;
      const v = MAZE[ny][nx] === "G" ? 1 : c.g * V[ny][nx]; if (v > bv) { bv = v; bx = nx; by = ny; }
    }
    if (bx < 0) break; x = bx; y = by; path.add(`${x},${y}`);
  }
  return { V, path, reached: MAZE[y][x] === "G" };
}
export function bellmanLab() {
  const ramp = ["#2f6fd6", "#22b8a0", "#5fd176", "#ffd23f", "#ff8a3d"];
  return vxLab({
    title: "Watch value flow back",
    rot0: 2,   // goal at the back, so the tall towers near it don't hide the rest
    controls: [
      { key: "k", label: "Sweeps of the Bellman update", min: 0, max: 16, value: 5 },
      { key: "g", label: "Discount γ", min: 0.5, max: 0.99, step: 0.01, value: 0.9, fmt: (v) => v.toFixed(2) },
    ],
    presets: [{ label: "Not yet", set: { k: 6, g: 0.9 } }, { label: "Just reaches", set: { k: 12, g: 0.9 } }, { label: "Short-sighted γ", set: { k: 16, g: 0.6 } }],
    draw(s, c) {
      const { V, path } = valueIter(c);
      MAZE.forEach((row, y) => [...row].forEach((ch, x) => {
        if (ch === "#") { s.put(x, y, 0, "#3a3e4a", { top: "#4a4f5d" }); return; }
        if (ch === "G") { const k = s.tip("<b>The goal.</b> Stepping in pays a reward of 1 and ends the episode."); s.put(x, y, 0, "#9aa3b5", { hit: k }).put(x, y, 1, "#9aa3b5", { hit: k }).put(x, y, 2, "#5fd176", { top: "#b6f5c4", hit: k }); return; }
        const v = V[y][x], hh = 1 + Math.round(v * 6), cc = v <= 0 ? "#c9d1e2" : ramp[Math.min(4, Math.floor(v * 5))];
        const k = s.tip(`<b>V = ${v.toFixed(3)}</b>${v > 0 ? `: the reward is ${Math.round(Math.log(v) / Math.log(c.g)) + 1} steps away, worth γ^${Math.round(Math.log(v) / Math.log(c.g))} of it from here.` : ": no reward has reached this square yet."}${ch === "S" ? " This is the start." : ""}`);
        s.box(x, y, 0, 1, 1, hh, cc, (i, j, z) => ({ faces: { top: z === hh - 1 ? (path.has(`${x},${y}`) ? "#ffffff" : v <= 0 ? "#e3e8f1" : undefined) : undefined, hit: k } }));
        if (ch === "S") s.put(x, y, hh, "#7b61ff", { top: "#c3b8ff", hit: k });
      }));
    },
    read(c) {
      const { V, reached } = valueIter(c), v0 = V[1][1];
      return {
        stats: [["Value at the start", v0.toFixed(3), v0 > 0 ? "good" : ""], ["Shortest path", "12 steps"], ["Sweeps needed to reach start", "12"], ["Greedy agent", reached ? "finds the goal" : "is lost", reached ? "good" : "bad"]],
        say: `<p>Each sweep, every square takes the best of its neighbours, times γ. Value spreads back from the goal one square per sweep, so the start (purple) only learns anything after 12 sweeps, the length of the shortest path. Then an agent that just walks uphill (white path) finds the goal. With a small γ the start's value is tiny (γ¹¹ = ${c.g.toFixed(2)}¹¹ ≈ ${(c.g ** 11).toFixed(3)}): far-off rewards barely count.</p>`,
      };
    },
  });
}
