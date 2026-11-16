// What the Studio shows: real model configs, their exact parameter counts, and a 3D view for every level
// you can open. A path names where you are: "" is the whole model, "b3" is block 3, "b3/attn" its
// attention, "b3/attn/h5" one head, "b3/mlp", "b3/ln1", "embed", "pos", "lnf", "head".

export const C = {
  purple: "#7b61ff", lilac: "#a996ff", teal: "#22b8a0", blue: "#4f8ef7", green: "#5fd176", orange: "#ff8a3d",
  red: "#f0524f", yellow: "#ffd23f", pink: "#e76fa8", grey: "#8ea3c7", slate: "#5b6172", dark: "#3a3e4a", plate: "#2a2d36", paper: "#d5dceb",
};

export const PRESETS = {
  tiny: { name: "Tiny GPT", vocab: 65, ctx: 256, d: 128, L: 4, heads: 4, kv: 4, ff: 512, norm: "LayerNorm", act: "GELU", pos: "learned", tied: true, bias: true,
    note: "Character-level, the kind you can train in a browser tab." },
  gpt2: { name: "GPT-2 small", vocab: 50257, ctx: 1024, d: 768, L: 12, heads: 12, kv: 12, ff: 3072, norm: "LayerNorm", act: "GELU", pos: "learned", tied: true, bias: true,
    note: "OpenAI, 2019. The model most courses build from scratch." },
  llama8: { name: "Llama 3 8B", vocab: 128256, ctx: 8192, d: 4096, L: 32, heads: 32, kv: 8, ff: 14336, norm: "RMSNorm", act: "SwiGLU", pos: "RoPE", tied: false, bias: false,
    note: "Meta, 2024. The modern recipe: RMSNorm, SwiGLU, RoPE, grouped-query attention." },
  mistral7: { name: "Mistral 7B", vocab: 32000, ctx: 32768, d: 4096, L: 32, heads: 32, kv: 8, ff: 14336, norm: "RMSNorm", act: "SwiGLU", pos: "RoPE", tied: false, bias: false, window: 4096,
    blocks: Object.fromEntries(Array.from({ length: 32 }, (_, i) => [i + 1, { attn: "window", mlp: true }])),
    note: "Mistral AI, 2023. Llama's recipe plus a 4,096-token sliding window in every layer." },
};

/** Settings a preset doesn't spell out. `blocks` holds per-block changes: { 5: { attn: "window", mlp: false } }. */
export const DEFAULTS = { residual: true, normPos: "pre", window: 1024, blocks: {} };
export const full = (c) => ({ ...DEFAULTS, ...c, blocks: { ...(c.blocks || {}) } });
/** One block's own settings: attention "full" | "window" | "none", and whether it has an MLP. */
export const blk = (c, b) => ({ attn: "full", mlp: true, ...(c.blocks?.[b] || {}) });
export const isStandard = (c) => c.residual && c.norm !== "None" && c.normPos === "pre" && Array.from({ length: c.L }, (_, i) => blk(c, i + 1)).every((s) => s.attn !== "none" && s.mlp);

/** Shapes that can't work, said plainly. An empty list means the model can be built. */
export function check(c) {
  const e = [];
  if (c.d % c.heads) e.push({ part: "attn", say: `Width ${n(c.d)} can't be split into ${c.heads} equal heads: that's ${(c.d / c.heads).toFixed(2)} numbers each. Pick a head count that divides ${n(c.d)}.` });
  if (c.kv > c.heads || c.heads % c.kv) e.push({ part: "attn", say: `${c.heads} heads can't share ${c.kv} K/V groups evenly. The number of K/V groups must divide the number of heads.` });
  if (c.L < 1) e.push({ part: "model", say: "A model needs at least one block." });
  return e;
}

/** Exact parameter counts, part by part. */
export function count(c) {
  c = full(c);
  const hd = c.d / c.heads, kvd = c.kv * hd, b = c.bias ? 1 : 0, glu = c.act === "SwiGLU";
  const normP = c.norm === "LayerNorm" ? 2 * c.d : c.norm === "None" ? 0 : c.d;
  const q = c.d * c.d + b * c.d, k = c.d * kvd + b * kvd, v = k, o = c.d * c.d + b * c.d;
  const up = c.d * c.ff + b * c.ff, gate = glu ? c.d * c.ff + b * c.ff : 0, down = c.ff * c.d + b * c.d;
  const attn = q + k + v + o, mlp = up + gate + down, block = attn + mlp + 2 * normP;
  const blockOf = (i) => { const s = blk(c, i); return (s.attn !== "none" ? attn + normP : 0) + (s.mlp ? mlp + normP : 0); };
  let blocks = 0, attnL = 0, winL = 0, mlpL = 0;
  for (let i = 1; i <= c.L; i++) { const s = blk(c, i); blocks += blockOf(i); if (s.attn !== "none") attnL++; if (s.attn === "window") winL++; if (s.mlp) mlpL++; }
  const embed = c.vocab * c.d, pos = c.pos === "learned" ? c.ctx * c.d : 0, head = c.tied ? 0 : c.vocab * c.d;
  const total = embed + pos + blocks + normP + head;
  return { hd, kvd, q, k, v, o, up, gate, down, attn, mlp, normP, block, blockOf, blocks, attnL, winL, mlpL, embed, pos, head, total, nonEmb: blocks + normP };
}

export const fmt = (n) => (n >= 1e9 ? `${(n / 1e9).toFixed(2)}B` : n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e4 ? `${(n / 1e3).toFixed(1)}k` : n.toLocaleString("en-US"));
export const bytes = (n) => (n >= 1e9 ? `${(n / 1e9).toFixed(1)} GB` : n >= 1e6 ? `${(n / 1e6).toFixed(1)} MB` : `${(n / 1e3).toFixed(1)} kB`);
export const pct = (a, b) => { if (!a) return "0%"; const p = (100 * a) / b; return p >= 10 ? `${p.toFixed(0)}%` : p >= 0.1 ? `${p.toFixed(1)}%` : "under 0.1%"; };
export const flops = (p) => { const f = 2 * p; return f >= 1e9 ? `${(f / 1e9).toFixed(2)} GFLOP` : f >= 1e6 ? `${(f / 1e6).toFixed(1)} MFLOP` : `${(f / 1e3).toFixed(1)} kFLOP`; };
const n = (x) => (Number.isInteger(x) ? x.toLocaleString("en-US") : x.toFixed(1));

// the sentence every view uses
const TOK = ["The", " cat", " sat", " on", " the"], T = TOK.length;
const NEXT = [[" mat", 0.41], [" floor", 0.17], [" sofa", 0.12], [" bed", 0.09], [" roof", 0.05], [" table", 0.04], [" ground", 0.03], [" couch", 0.03], [" grass", 0.02], [" dog", 0.01]];
const TCOL = [C.blue, C.teal, C.orange, C.pink, C.green];
const q = (t) => `“${t.trim()}”`;

/** How many cells wide a vector of width d is drawn: grows with log d so big models stay visible. */
const wOf = (d) => Math.max(4, Math.min(16, Math.round(2 * Math.log2(d / 32))));
const wFf = (c) => wOf(c.d) + Math.max(2, Math.round(Math.log2(c.ff / c.d) * 3));

function rand(seed) { let a = seed >>> 0; return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

// attention head personalities (an illustration: real heads learn patterns like these)
export const HEADS = [
  { name: "previous-token head", col: C.blue, say: "Each word looks at the word just before it.", w: (i, j) => (j === i - 1 ? 6 : j === i ? 1 : 0.25) },
  { name: "sink head", col: C.pink, say: "Almost everything piles onto the first token, a safe place to put attention that has nowhere useful to go.", w: (i, j) => (j === 0 ? 6 : 0.35) },
  { name: "same-word head", col: C.green, say: "A word looks back at earlier copies of itself, like “the” finding “The”.", w: (i, j) => (TOK[j].trim().toLowerCase() === TOK[i].trim().toLowerCase() && j !== i ? 6 : j === i ? 1.5 : 0.3) },
  { name: "broad head", col: C.grey, say: "Attention spread evenly: an average of everything so far.", w: () => 1 },
];
export const headType = (layer, h) => HEADS[(h + layer * 3 + Math.floor(h / 4)) % 4];
function pattern(ht) {
  return Array.from({ length: T }, (_, i) => { const r = Array.from({ length: T }, (_, j) => (j <= i ? ht.w(i, j) : 0)); const s = r.reduce((a, b) => a + b, 0); return r.map((v) => v / s); });
}

// ---------------------------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------------------------
export function parse(c, path) {
  const s = (path || "").split("/").filter(Boolean);
  const bi = s[0] && /^b(\d+)$/.exec(s[0]);
  if (!s.length) return { kind: "model" };
  if (["embed", "head"].includes(s[0]) && s.length === 1) return { kind: s[0] };
  if (s[0] === "lnf" && c.norm !== "None" && s.length === 1) return { kind: "lnf" };
  if (s[0] === "pos" && c.pos === "learned" && s.length === 1) return { kind: "pos" };
  if (bi) {
    const b = +bi[1];
    if (b < 1 || b > c.L) return { kind: "model" };
    if (s.length === 1) return { kind: "block", b };
    const own = blk(c, b);
    if ((s[1] === "ln1" || s[1] === "ln2") && c.norm !== "None" && (s[1] === "ln1" ? own.attn !== "none" : own.mlp)) return s.length === 2 ? { kind: "norm", b, which: s[1] } : { kind: "block", b };
    if (s[1] === "mlp" && own.mlp) return s.length === 2 ? { kind: "mlp", b } : { kind: "block", b };
    if (s[1] === "attn" && own.attn !== "none") {
      if (s.length === 2) return { kind: "attn", b };
      const hm = /^h(\d+)$/.exec(s[2] || ""), h = hm ? +hm[1] : 0;
      return h >= 1 && h <= c.heads && s.length === 3 ? { kind: "headview", b, h } : { kind: "attn", b };
    }
  }
  return { kind: "model" };
}
export const pathOf = (p) => ({ model: "", embed: "embed", pos: "pos", lnf: "lnf", head: "head", block: `b${p.b}`, norm: `b${p.b}/${p.which}`, mlp: `b${p.b}/mlp`, attn: `b${p.b}/attn`, headview: `b${p.b}/attn/h${p.h}` })[p.kind];
export const parentOf = (path) => (path || "").split("/").filter(Boolean).slice(0, -1).join("/");

/** Breadcrumbs: [{ label, path }]. */
export function crumbs(c, path) {
  const p = parse(c, path), out = [{ label: c.name, path: "" }];
  const NAMES = { embed: "Embedding", pos: "Positions", lnf: "Final norm", head: "Unembedding" };
  if (NAMES[p.kind]) out.push({ label: NAMES[p.kind], path: p.kind });
  if (p.b) out.push({ label: `Block ${p.b}`, path: `b${p.b}` });
  if (p.kind === "norm") out.push({ label: p.which === "ln1" ? "Norm 1" : "Norm 2", path: `b${p.b}/${p.which}` });
  if (p.kind === "mlp") out.push({ label: "MLP", path: `b${p.b}/mlp` });
  if (p.kind === "attn" || p.kind === "headview") out.push({ label: "Attention", path: `b${p.b}/attn` });
  if (p.kind === "headview") out.push({ label: `Head ${p.h}`, path: `b${p.b}/attn/h${p.h}` });
  return out;
}

// ---------------------------------------------------------------------------------------------
// Views: each fills a Build and returns { title, blurb }
// ---------------------------------------------------------------------------------------------
export function draw(c, path, B, opts = {}) {
  c = full(c);
  const p = parse(c, path), k = count(c), errs = check(c);
  return VIEWS[p.kind](c, k, p, B, errs, opts);
}

/** A measured number per block (from a lab run) as a colour from dark (tiny) to bright (big), on a log scale. */
const mix = (a, b, t) => "#" + [0, 2, 4].map((i) => Math.round(parseInt(a.slice(1 + i, 3 + i), 16) * (1 - t) + parseInt(b.slice(1 + i, 3 + i), 16) * t).toString(16).padStart(2, "0")).join("");
export const heatColor = (t) => (t < 0.5 ? mix("#2f3340", "#c2571f", t * 2) : mix("#c2571f", "#ffe066", t * 2 - 1));
function heatOf(heat, L) {
  if (!heat?.values?.length) return null;
  const v = heat.values.map((x) => Math.log10(Math.max(x, 1e-12))), lo = Math.min(...v), hi = Math.max(...v), n = v.length;
  return (b) => { const j = Math.min(n - 1, Math.floor(((b - 1) * n) / L)); return { t: hi > lo ? (v[j] - lo) / (hi - lo) : 1, x: heat.values[j] }; };
}

const plate = (B, x, y, z, w, d) => B.cells(x, y, z - 0.5, w, d, 0.5, C.plate, B.obj({ decor: true }), { top: (i, j) => ((i + j) % 2 ? "#3a3e4a" : "#353944") });
const normName = (c) => (c.norm === "RMSNorm" ? "RMSNorm" : "LayerNorm");
const ATTN_SAY = { full: "full attention (every earlier word)", window: "sliding-window attention (only the last few words)", none: "no attention" };

const VIEWS = {
  model(c, k, p, B, errs, opts) {
    const W = wOf(c.d), big = c.L > 16, hb = big ? 1 : 2, gap = big ? 0.4 : 1, heat = heatOf(opts.heat, c.L);
    // tokens in
    TOK.forEach((t, j) => B.box(W / 2 - 0.5, j, 0, 1, 1, 1, TCOL[j], B.obj({ name: `Token ${q(t)}`, tag: j === 0 ? "Your words" : null,
      what: `Text is cut into tokens (words or pieces of words). Each one is just a number: its row in the vocabulary of ${n(c.vocab)}.`, rows: [["Position", String(j)]] })));
    let z = 2;
    const embed = B.obj({ name: "Embedding", tag: "Embedding", open: "embed", edit: "embed", what: "A lookup table. Each token's number picks one row: a list of numbers the rest of the model can work with.",
      rows: [["Table", `${n(c.vocab)} × ${n(c.d)}`], ["Parameters", fmt(k.embed)], ["Share", pct(k.embed, k.total)]] });
    B.cells(0, 0, z, W, T, 1, C.blue, embed);
    if (c.pos === "learned") {
      const pos = B.obj({ name: "Positions", tag: "+ position", open: "pos", edit: "pos", what: "A second table, one row per position, added on so the model knows word order.",
        rows: [["Table", `${n(c.ctx)} × ${n(c.d)}`], ["Parameters", fmt(k.pos)], ["Longest input", `${n(c.ctx)} tokens`]] });
      B.cells(W + 1, 0, z, 2, T, 1, C.pink, pos);
    }
    z += 2;
    const stream = B.obj({ decor: true });
    const bad = errs.some((e) => e.part === "attn");
    for (let b = 1; b <= c.L; b++) {
      const s = blk(c, b), bp = k.blockOf(b);
      const ht = heat && heat(b);
      const id = B.obj({ name: `Block ${b}`, tag: ht ? (b === 1 || b === c.L || b === Math.ceil(c.L / 2) ? `Block ${b} · ${ht.x.toPrecision(2)}` : null) : b === 1 || b === c.L ? `Block ${b}` : null, open: `b${b}`, edit: { block: b },
        what: s.attn === "none" && !s.mlp ? "An empty block: no attention and no MLP, so the stream passes straight through. It costs nothing and does nothing." : `One round of thinking: ${s.attn === "none" ? "no attention (words can't look at each other here)" : `${ATTN_SAY[s.attn]} on the bottom layer, where words look at each other`}, then ${s.mlp ? "the MLP on top, where each word is processed alone" : "no MLP"}.`,
        rows: [["Parameters", fmt(bp)], ["Share", pct(bp, k.total)], ["Attention", s.attn], ["MLP", s.mlp ? "yes" : "none"]] });
      if (ht) { B.cells(0, 0, z, W, T, hb, heatColor(ht.t), id); z += hb + gap; if (b < c.L) for (let j = 0; j < T; j++) B.box(W / 2 - 0.15, j + 0.35, z - gap, 0.3, 0.3, gap, C.paper, stream); continue; }
      if (s.attn !== "none") B.cells(0, 0, z, W, T, hb / 2, bad ? C.red : s.attn === "window" ? C.lilac : C.purple, id, s.attn === "window" ? { top: (i) => (i % 2 ? "#cfc4ff" : "#8f7ae8") } : {});
      if (s.mlp) B.cells(0, 0, z + hb / 2, W, T, hb / 2, C.orange, id);
      if (s.attn === "none" && !s.mlp) B.cells(0, 0, z + hb / 2 - 0.1, W, T, 0.2, C.dark, id);
      if (b < c.L) for (let j = 0; j < T; j++) B.box(W / 2 - 0.15, j + 0.35, z + hb, 0.3, 0.3, gap, C.paper, stream);
      z += hb + gap;
    }
    if (gap < 1) z += 1 - gap;
    if (c.norm !== "None") {
      const lnf = B.obj({ name: "Final norm", tag: "Final norm", open: "lnf", edit: "norm", what: `One last ${normName(c)}: rescale each token's vector before reading out a word.`, rows: [["Parameters", n(k.normP)]] });
      B.cells(0, 0, z, W, T, 0.5, C.yellow, lnf);
    }
    z += 1.5;
    const head = B.obj({ name: "Unembedding", tag: "Unembedding", open: "head", edit: "embed",
      what: `Turns the last token's vector into a score for every one of the ${n(c.vocab)} tokens.${c.tied ? " It reuses the embedding table (tied weights), so it adds no parameters." : ""}`,
      rows: [["Matrix", `${n(c.d)} × ${n(c.vocab)}`], ["Parameters", c.tied ? "0 (shared)" : fmt(k.head)], ["Share", pct(k.head, k.total)]] });
    B.cells(0, 0, z, W, T, 1, C.teal, head); z += 1;
    NEXT.slice(0, 5).forEach(([t, pr], i) => {
      const id = B.obj({ name: `Next token ${q(t)}`, tag: i === 0 ? `${q(t)} ${Math.round(pr * 100)}%` : null, what: `The model's guess for what comes after “The cat sat on the”: ${Math.round(pr * 100)}% (an illustration).`, rows: [["Probability", `${Math.round(pr * 100)}%`]] });
      B.box(Math.round((i * (W - 1)) / 4), T - 1, z, 1, 1, 0.4 + pr * 14, i ? C.green : C.yellow, id);
    });
    plate(B, -1, -1, 0, W + (c.pos === "learned" ? 5 : 2), T + 2);
    return { title: c.name, blurb: `${c.note} Click any part to see what it is; double-click (or press Open) to go inside.` };
  },

  block(c, k, p, B, errs) {
    const W = wOf(c.d), F = wFf(c), fx = (W - F) / 2, s = blk(c, p.b), bp = k.blockOf(p.b), nm = c.norm !== "None", pre = c.normPos === "pre";
    const bad = errs.some((e) => e.part === "attn");
    const N = (w) => ["Norm " + w, 0.5, C.yellow, W, 0, "ln" + w, `${normName(c)}: rescale each vector so the numbers stay a sensible size.${pre ? "" : " Post-norm: it sits after the add, on the stream itself, like the original 2017 Transformer."}`, [["Parameters", n(k.normP)]], "norm"];
    const ADD = (what) => ["+ add", 0.6, C.yellow, W, 0, null, what, [["Parameters", "0"]], "residual"];
    const A = ["Attention", 3, bad ? C.red : s.attn === "window" ? C.lilac : C.purple, W, 0, "attn", bad ? errs.find((e) => e.part === "attn").say : s.attn === "window" ? `Sliding window: each word looks back at most ${n(c.window)} tokens. ${c.heads} heads.` : `Words look at each other. ${c.heads} heads each ask a different question and pull in what they find.`, [["Parameters", fmt(k.attn)], ["Share of block", pct(k.attn, bp)], ["Heads", `${c.heads}${c.kv < c.heads ? ` (sharing ${c.kv} K/V)` : ""}`]], "attn"];
    const M = ["MLP", 3, C.orange, F, fx, "mlp", `Each token alone goes through a wider layer (${n(c.ff)} wide) and back. Most of the model's knowledge is stored here.`, [["Parameters", fmt(k.mlp)], ["Share of block", pct(k.mlp, bp)], ["Width", `${n(c.d)} → ${n(c.ff)} → ${n(c.d)}`]], "mlp"];
    const lay = [["Input", 0.6, C.grey, W, 0, null, `This block's input: one vector of ${n(c.d)} numbers for each token.`, [], null]];
    const part = (P, w, addSay) => {
      if (nm && pre) lay.push(N(w));
      lay.push(P);
      if (c.residual) lay.push(ADD(addSay));
      if (nm && !pre) lay.push(N(w));
    };
    if (s.attn !== "none") part(A, 1, "Attention's output is added to the stream, not swapped in.");
    if (s.mlp) part(M, 2, "The MLP's output is added to the stream too.");
    lay.push(["Output", 0.6, C.grey, W, 0, null, c.residual ? "The same shape as the input, a little more informed. It goes on to the next block." : "With no residual, this is all that's left: each part replaced the stream instead of adding to it.", [], null]);
    let z = 0; const adds = [];
    for (const [name, h, col, w, x, open, what, rows, edit] of lay) {
      const id = B.obj({ name: open || name.startsWith("+") ? name : `Block ${p.b} ${name.toLowerCase()}`, tag: name, open: open ? `b${p.b}/${open}` : null, edit, what, rows: [...rows, ["Shape", `T × ${n(c.d)}`]] });
      B.cells(x, 0, z, w, T, h, col, id);
      if (name === "+ add") adds.push(z);
      z += h + 1;
    }
    // the residual highway: one thin rail per token on the side, with bridges where it splits and merges
    if (c.residual) {
      const res = B.obj({ name: "Residual stream", tag: "Residual stream", edit: "residual", what: "The highway. Each token's vector rides up the side, and attention and the MLP only add their changes to it. Because the original is always kept, a deep stack can still learn: gradients flow straight down this road." });
      for (let j = 0; j < T; j++) {
        B.box(-3, j + 0.3, 0, 0.4, 0.4, z - 0.6, C.paper, res);
        for (const az of [0.1, ...adds.map((a) => a + 0.1)]) B.box(-2.6, j + 0.3, az, 2.6, 0.4, 0.4, C.paper, res);
      }
    }
    plate(B, -4, -1, 0, Math.max(W, s.mlp ? F + fx : W) + 5, T + 2);
    const order = lay.slice(1, -1).map((l) => (l[0] === "MLP" ? "MLP" : l[0].replace(/ \d$/, "").toLowerCase())).join(", ") || "nothing";
    return { title: `Block ${p.b}`, blurb: `${order[0].toUpperCase() + order.slice(1)}.${c.residual ? " The side rails are the residual stream: the input passes around each part and the part's output is added back." : " No residual stream: every part replaces what came before."} ${pct(bp, k.total)} of the whole model is this block.` };
  },

  attn(c, k, p, B, errs) {
    const W = wOf(c.d), wk = Math.max(1, Math.min(W, Math.round((W * c.kv) / c.heads))), gqa = c.kv < c.heads, bad = errs.filter((e) => e.part === "attn");
    const win = blk(c, p.b).attn === "window";
    const mk = (name, tag, what, rows, open) => B.obj({ name, tag, what, rows, open, edit: "attn" });
    const cx = (W + 2 * wk + 2) / 2;
    B.cells(cx - W / 2, 0, 0, W, T, 0.6, C.grey, mk("Input", "Input", "The normalised token vectors.", [["Shape", `T × ${n(c.d)}`]]));
    const Q = mk("Queries (Q)", "Q", "Each token's question: “what am I looking for?”", [["Weights", `${n(c.d)} × ${n(c.d)}`], ["Parameters", fmt(k.q)], ["Compute / token", flops(k.q)]]);
    const K = mk("Keys (K)", "K", `Each token's label: “here is what I contain.”${gqa ? ` Only ${c.kv} sets, shared across heads.` : ""}`, [["Weights", `${n(c.d)} × ${n(k.kvd)}`], ["Parameters", fmt(k.k)]]);
    const V = mk("Values (V)", "V", `What each token hands over if it is looked at.${gqa ? ` Also ${c.kv} sets, shared.` : ""}`, [["Weights", `${n(c.d)} × ${n(k.kvd)}`], ["Parameters", fmt(k.v)]]);
    B.cells(0, 0, 1.8, W, T, 1, C.blue, Q);
    B.cells(W + 1, 0, 1.8, wk, T, 1, C.green, K);
    B.cells(W + wk + 2, 0, 1.8, wk, T, 1, C.orange, V);
    // heads: one small tower each, coloured by the pattern it learned
    const nh = Math.min(c.heads, 128), cols = Math.min(nh, 8), rows = Math.ceil(nh / cols), hx = (W + 2 * wk + 2 - cols * 1.5) / 2;
    plate(B, hx - 0.5, -0.25 + (T - rows * 1.5) / 2, 5, cols * 1.5 + 0.5, rows * 1.5 + 0.5);
    for (let h = 1; h <= nh; h++) {
      const ht = headType(p.b, h), ci = (h - 1) % cols, ri = Math.floor((h - 1) / cols);
      if (bad.length) { B.cells(hx + ci * 1.5, (T - rows * 1.5) / 2 + ri * 1.5, 5, 1, 1, 1.2, C.red, mk(`Head ${h}`, h === 1 ? "Shape error" : null, bad.map((e) => e.say).join(" "), [["Numbers per head", n(k.hd)]])); continue; }
      const id = mk(`Head ${h}`, h === 1 ? "Heads" : null, `A ${ht.name}. ${ht.say}`, [["Size", `${k.hd} numbers per token`], ["Shares K/V with", gqa ? `heads ${Math.floor((h - 1) / (c.heads / c.kv)) * (c.heads / c.kv) + 1}–${(Math.floor((h - 1) / (c.heads / c.kv)) + 1) * (c.heads / c.kv)}` : "no one"]], `b${p.b}/attn/h${h}`);
      B.cells(hx + ci * 1.5, (T - rows * 1.5) / 2 + ri * 1.5, 5, 1, 1, 1.2 + (h % 3) * 0.5, ht.col, id);
    }
    const O = mk("Output projection", "Mix heads", "The heads' results are laid side by side and mixed back into one vector per token.", [["Weights", `${n(c.d)} × ${n(c.d)}`], ["Parameters", fmt(k.o)]]);
    B.cells(cx - W / 2, 0, 8.6, W, T, 1, C.purple, O);
    B.cells(cx - W / 2, 0, 10.4, W, T, 0.6, C.grey, mk("Output", "Output", "Added back onto the residual stream.", [["Shape", `T × ${n(c.d)}`]]));
    if (bad.length) return { title: "Attention", blurb: `Shape error. ${bad.map((e) => e.say).join(" ")}` };
    return { title: win ? "Sliding-window attention" : "Attention", blurb: `${win ? `Each word may only look back ${n(c.window)} tokens. ` : ""}Make Q, K, V from each token, let ${c.heads} heads compare queries with keys${c.pos === "RoPE" ? " (RoPE rotates Q and K so position is part of the comparison)" : ""}, then mix the heads back together. Double-click a head to see who looks at whom.${gqa ? ` Grouped-query attention: ${c.heads} heads share ${c.kv} K/V sets, which shrinks the KV cache ${c.heads / c.kv}×.` : ""}` };
  },

  headview(c, k, p, B) {
    const ht = headType(p.b, p.h), win = blk(c, p.b).attn === "window", vw = win ? Math.max(1, Math.min(T - 1, c.window < c.ctx ? 2 : T)) : T;
    const P = pattern(ht).map((row, i) => { const r = row.map((v, j) => (i - j >= vw ? 0 : v)); const s = r.reduce((a, b) => a + b, 0) || 1; return r.map((v) => v / s); });
    TOK.forEach((t, i) => {
      B.box(i, -2, 0, 1, 1, 1, C.blue, B.obj({ name: `Query from ${q(t)}`, tag: t.trim(), what: `${q(t)} asks its question. Its row of towers shows where it looks.`, rows: [] }));
      B.box(-2, i, 0, 1, 1, 1, C.green, B.obj({ name: `Key of ${q(t)}`, tag: t.trim(), what: `${q(t)} offers its key. Its column shows who looks at it.`, rows: [] }));
    });
    plate(B, -0.25, -0.25, 0, T + 0.5, T + 0.5);
    for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) {
      const masked = j > i || i - j >= vw, past = j <= i && masked;
      const id = B.obj({ name: masked ? `${q(TOK[i])} → ${q(TOK[j])}: masked` : `${q(TOK[i])} looks at ${q(TOK[j])}`, edit: "attn",
        what: past ? `Outside the window. This head may only look ${n(c.window)} tokens back (drawn here as ${vw}), so this score is never computed.` : masked ? "A future word. Masked out: the model may only look backwards, so this score is never used." : `Attention weight ${Math.round(P[i][j] * 100)}%. Each row adds up to 100%.`, rows: masked ? [] : [["Weight", `${(P[i][j] * 100).toFixed(1)}%`]] });
      B.box(i + 0.05, j + 0.05, 0, 0.9, 0.9, masked ? 0.25 : 0.25 + P[i][j] * 5, past ? C.slate : masked ? C.dark : ht.col, id);
    }
    TOK.forEach((t, i) => B.box(i, T + 1, 0, 1, 1, 1, C.orange, B.obj({ name: `Result for ${q(t)}`, tag: i === 0 ? "Results" : null, what: `A blend of the values of the words ${q(t)} looked at, weighted by its row.`, rows: [["Size", `${k.hd} numbers`]] })));
    return { title: `Head ${p.h}`, blurb: `${ht.name[0].toUpperCase() + ht.name.slice(1)}. ${ht.say} Rows are the asking word (blue), columns the word looked at (green). Tower height is the attention weight. Dark tiles are the future, masked out.${win ? " Grey tiles are outside the sliding window." : ""} (Patterns like this are found in real models; this one is drawn for illustration.)` };
  },

  mlp(c, k, p, B) {
    const W = wOf(c.d), F = wFf(c), x0 = (W - F) / 2, glu = c.act === "SwiGLU", r = rand(p.b * 97 + 13);
    const mk = (name, tag, what, rows) => B.obj({ name, tag, what, rows, edit: "mlp" });
    B.cells(0, 0, 0, W, T, 0.6, C.grey, mk("Input", "Input", "One vector per token. Each token goes through on its own.", [["Shape", `T × ${n(c.d)}`]]));
    if (glu) {
      B.cells(x0 - F / 2 - 0.5, 0, 1.8, F, T, 1, C.orange, mk("Up projection", "Up", `Widen each vector from ${n(c.d)} to ${n(c.ff)} numbers.`, [["Parameters", fmt(k.up)], ["Compute / token", flops(k.up)]]));
      B.cells(x0 + F / 2 + 0.5, 0, 1.8, F, T, 1, C.pink, mk("Gate projection", "Gate", "A second widening, used as a gate: it decides how much of each up value gets through.", [["Parameters", fmt(k.gate)], ["Compute / token", flops(k.gate)]]));
    } else B.cells(x0, 0, 1.8, F, T, 1, C.orange, mk("Up projection", "Up", `Widen each vector from ${n(c.d)} to ${n(c.ff)} numbers.`, [["Parameters", fmt(k.up)], ["Compute / token", flops(k.up)]]));
    const act = mk(glu ? "SwiGLU" : c.act, glu ? "SiLU(gate) × up" : c.act, glu ? "Each up value is multiplied by a squashed gate value. Gates near zero switch that feature off for this token." : "A bend: negative values are squeezed to about zero, positive ones pass. Without it, the two matrices would collapse into one.", [["Parameters", "0"], ["Features", n(c.ff)]]);
    // one cell per (feature, token), each its own outlined block; height is how strongly the feature fired
    let shown = 0;
    for (let i = 0; i < F; i++) for (let j = 0; j < T; j++) {
      const g = (r() + r() + r() - 1.5) * 2, v = g > 0 ? g : 0, f = Math.floor((i + 0.5) * (c.ff / F));
      const id = shown++ ? B.obj({ edit: "mlp", name: `Feature ${n(f)} for ${q(TOK[j])}`, what: v > 0.01 ? `Fired at ${v.toFixed(2)}. Features are learned detectors; this one says something about ${q(TOK[j])} that the next layer will read.` : "Stayed off for this token (the activation squashed it to about zero).", rows: [["Activation", v.toFixed(2)]] }) : act;
      B.box(x0 + i + 0.06, j + 0.06, 3.6, 0.88, 0.88, 0.2 + v * 1.4, v > 0.01 ? C.yellow : C.dark, id);
    }
    B.cells(0, 0, 7.4, W, T, 1, C.orange, mk("Down projection", "Down", `Squeeze back from ${n(c.ff)} to ${n(c.d)} numbers.`, [["Parameters", fmt(k.down)], ["Compute / token", flops(k.down)]]));
    B.cells(0, 0, 9.2, W, T, 0.6, C.grey, mk("Output", "Output", "Added back onto the residual stream.", [["Shape", `T × ${n(c.d)}`]]));
    return { title: "MLP", blurb: `Widen, bend, narrow. The yellow towers are features that switched on for each token; dark ones stayed off. ${pct(k.mlp, k.block)} of the block's parameters live here.${glu ? " SwiGLU uses a third matrix as a gate." : ""}` };
  },

  norm(c, k, p, B) { return normView(c, k, B, p.which === "ln1" ? "Norm 1" : p.which === "ln2" ? "Norm 2" : "Final norm", p.b || 0); },
  lnf(c, k, p, B) { return normView(c, k, B, "Final norm", 99); },

  embed(c, k, p, B) { return tableView(c, k, B, "embed"); },
  pos(c, k, p, B) { return tableView(c, k, B, "pos"); },

  head(c, k, p, B) {
    const W = wOf(c.d), R = NEXT.length;
    B.cells(0, R / 2 - 0.5, 0, W, 1, 0.6, C.grey, B.obj({ name: "Last token's vector", tag: q(TOK[T - 1]), what: "Only the last position predicts the next token.", rows: [["Size", n(c.d)]] }));
    const M = B.obj({ edit: "embed", name: "Unembedding matrix", tag: c.tied ? "Same table as the embedding" : "Unembedding", what: `One row per token in the vocabulary. Each row is dotted with the vector to give that token a score.${c.tied ? " Here it is literally the embedding table, reused." : ""}`, rows: [["Matrix", `${n(c.d)} × ${n(c.vocab)}`], ["Parameters", c.tied ? "0 (shared)" : fmt(k.head)], ["Compute / token", flops(c.d * c.vocab)]] });
    B.cells(0, 0, 1.6, W, R, 0.6, C.teal, M);
    NEXT.forEach(([t, pr], i) => {
      B.box(W + 1, i, 1.6, 1, 1, 0.5 + (Math.log(pr) + 5) * 0.9, C.slate, B.obj({ name: `Score for ${q(t)}`, what: "A raw score (logit). Bigger means more likely, but they don't add up to anything yet.", rows: [["Logit", (Math.log(pr) + 3).toFixed(2)]] }));
      B.box(W + 3, i, 1.6, 1, 1, 0.3 + pr * 14, i ? C.green : C.yellow, B.obj({ name: `Probability of ${q(t)}`, tag: i < 3 ? `${t.trim()} ${Math.round(pr * 100)}%` : null, what: "After softmax: scores become probabilities that add up to 100%.", rows: [["Probability", `${(pr * 100).toFixed(0)}%`]] }));
    });
    plate(B, -1, -1, 1.6, W + 6, R + 2);
    return { title: "Unembedding", blurb: `Score all ${n(c.vocab)} tokens (only 10 shown), turn scores into probabilities with softmax, then pick one. That word is appended and the whole model runs again for the next.` };
  },
};

function normView(c, k, B, label, seed) {
  const r = rand(seed * 31 + 7), D = 12, raw = Array.from({ length: D }, () => (r() - 0.5) * (r() < 0.2 ? 9 : 3) + (c.norm === "LayerNorm" ? 0.8 : 0));
  const mean = c.norm === "LayerNorm" ? raw.reduce((a, b) => a + b, 0) / D : 0;
  const rms = Math.sqrt(raw.reduce((a, b) => a + (b - mean) ** 2, 0) / D), nrm = raw.map((v) => (v - mean) / rms);
  const g = Array.from({ length: D }, () => 0.6 + r() * 0.9), out = nrm.map((v, i) => v * g[i]);
  const rows = [["Before", raw, C.slate, "One token's vector (12 of its numbers). Some are huge, some tiny: the size drifts from block to block."],
    [c.norm === "LayerNorm" ? "− mean, ÷ spread" : "÷ RMS", nrm, C.yellow, c.norm === "LayerNorm" ? "Subtract the average and divide by the spread, so the numbers average 0 with spread 1." : "Divide by the root-mean-square, so the overall size is 1. (Cheaper than LayerNorm: no mean.)"],
    ["× learned scale", out, C.orange, `Multiply each number by its own learned weight${c.norm === "LayerNorm" ? " (and add a learned shift)" : ""}, so the model can still make some features louder.`]];
  rows.forEach(([name, v, col, what], ri) => {
    const id = B.obj({ name, tag: name, what, edit: "norm", rows: [["Parameters", ri === 2 ? n(k.normP) : "0"]] });
    v.forEach((x, i) => B.box(ri * 3, i, 0, 1, 1, Math.max(0.15, Math.abs(x) * 1.2), x < 0 ? C.dark : col, id, x < 0 ? "#6b7385" : undefined));
  });
  plate(B, -1, -1, 0, 9, D + 2);
  return { title: label, blurb: `${normName(c)} keeps every token's vector at a steady size so the next layer sees familiar numbers. Tall dark bars are negative values. It has only ${n(k.normP)} parameters.` };
}

function tableView(c, k, B, which) {
  const W = wOf(c.d), emb = which === "embed", R = 14, rowsN = emb ? c.vocab : c.ctx;
  const picks = emb ? [3, 6, 9, 12, 6] : [0, 1, 2, 3, 4];
  const tab = B.obj({ edit: emb ? "embed" : "pos", name: emb ? "Embedding table" : "Position table", tag: `${n(rowsN)} rows × ${n(c.d)}`, what: emb ? "One row for every token in the vocabulary. Training nudges each row until similar tokens get similar rows." : "One row for every position. Row 0 is added to the first token, row 1 to the second, and so on.", rows: [["Parameters", fmt(emb ? k.embed : k.pos)], ["Memory (16-bit)", bytes(2 * (emb ? k.embed : k.pos))]] });
  const used = new Set(picks);
  for (let j = 0; j < R; j++) B.cells(0, j + (j >= R / 2 ? 1 : 0), 0, W, 1, 0.6, used.has(j) ? (emb ? C.lilac : C.pink) : C.paper, tab);
  TOK.forEach((t, i) => {
    const row = picks[i], y = row + (row >= R / 2 ? 1 : 0);
    const id = B.obj({ edit: emb ? "embed" : "pos", name: emb ? `Row for ${q(t)}` : `Position ${i}`, tag: t.trim(), what: emb ? `${q(t)} is looked up: this row is copied out as its vector.${t.trim().toLowerCase() === "the" ? " “The” and “the” are different tokens, so they get different rows." : ""}` : `Added to ${q(t)}, the word in position ${i}.`, rows: [["Size", `${n(c.d)} numbers`]] });
    B.cells(-W - 3, i * 1.6 + (R + 1 - T * 1.6) / 2, 2.4, W, 1, 0.6, emb ? C.blue : C.pink, id);
    B.box(-W - 5, i * 1.6 + (R + 1 - T * 1.6) / 2, 2.4, 1, 1, 1, TCOL[i], id);
    B.box(-2.4, y + 0.3, 0.6, 1.6, 0.4, 0.4, C.paper, id);
  });
  plate(B, -W - 6, -1, 0, 2 * W + 7, R + 3);
  return { title: emb ? "Embedding" : "Positions", blurb: emb ? `A token's number is just a row index. The model copies that row out (left) and uses it as the token's starting vector. The table holds ${fmt(k.embed)} numbers, ${pct(k.embed, k.total)} of the model.` : `Without this, “cat sat” and “sat cat” would look the same. Learned positions stop at ${n(c.ctx)}: the model has no row for position ${n(c.ctx)}.` };
}
