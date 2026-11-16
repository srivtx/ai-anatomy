// Recipes: the changes famous models made, as one-click edits to whatever you're building, each with why
// it works and what it costs.
import { blk } from "./arch.js";

const divs = (n) => Array.from({ length: n }, (_, i) => i + 1).filter((i) => n % i === 0);
const near = (list, v) => list.reduce((b, x) => (Math.abs(Math.log(x / v)) < Math.abs(Math.log(b / v)) ? x : b), list[0]);
const blocksWith = (c, f) => { const o = {}; for (let i = 1; i <= c.L; i++) { const s = { ...blk(c, i), ...f(i) }; if (s.attn !== "full" || !s.mlp) o[i] = s; } return o; };

export const RECIPES = [
  {
    id: "llama", name: "The Llama recipe", who: "Meta, 2023–24",
    patch: (c) => ({ norm: "RMSNorm", act: "SwiGLU", ff: Math.round((c.d * 8) / 3 / 64) * 64 || Math.round((c.d * 8) / 3), pos: "RoPE", bias: false, kv: c.heads >= 8 ? near(divs(c.heads), c.heads / 4) : c.kv }),
    why: "Four swaps that each fix one thing. RMSNorm: same stability, less work. SwiGLU: a gate lets the MLP pick which features pass, better per parameter (the MLP narrows to 8/3× so the size stays put). RoPE: attention sees distances, and long text can be stretched later. Grouped K/V: a 4× smaller cache while serving.",
    cost: "Nothing measurable in quality; SwiGLU has three matrices, so the MLP is a little slower per parameter.",
  },
  {
    id: "mistral", name: "Sliding window everywhere", who: "Mistral 7B, 2023",
    patch: (c) => ({ window: Math.min(4096, c.ctx / 2), blocks: blocksWith(c, (i) => (blk(c, i).attn !== "none" ? { attn: "window" } : {})) }),
    why: "Each layer only looks back a fixed number of tokens, so its cache and cost stop growing with length. Information still travels further: layer 2 sees what layer 1 saw, so the reach is layers × window.",
    cost: "Facts more than layers × window tokens apart can never meet. Exact long-range lookups get harder.",
  },
  {
    id: "gemma", name: "Alternate local and global", who: "Gemma 2, 2024",
    patch: (c) => ({ window: Math.min(4096, c.ctx / 2), blocks: blocksWith(c, (i) => (i % 2 && blk(c, i).attn !== "none" ? { attn: "window" } : {})) }),
    why: "Half the layers are cheap and windowed (most of what matters is nearby); the other half see everything. You keep exact long-range lookups and cut the cache nearly in half. Gemma 3 went to 5 local per global.",
    cost: "Less saving than windowing everything; the global layers still grow with length.",
  },
  {
    id: "mqa", name: "One K/V for all heads", who: "PaLM, Falcon, 2022–23",
    patch: () => ({ kv: 1 }),
    why: "Multi-query attention: every head keeps its own query, but they all share one key and one value. The cache shrinks by the number of heads, which is what limits how many users fit on a GPU.",
    cost: "A small quality loss and some training instability; grouped-query (a few groups) became the usual middle ground.",
  },
  {
    id: "hybrid", name: "Attention in 1 block of 4", who: "Jamba, Samba-style hybrids, 2024",
    patch: (c) => ({ blocks: blocksWith(c, (i) => (i % 4 ? { attn: "none" } : {})) }),
    why: "Hybrids keep a few attention layers for exact lookups and use cheaper mixers elsewhere. Our parts box doesn't have a Mamba mixer yet, so the other blocks keep only their MLP: you'll see how much attention you can drop before the predictions object.",
    cost: "Without a replacement mixer, fewer blocks can move information between words.",
  },
  {
    id: "deepthin", name: "Deep and thin", who: "MobileLLM, 2024",
    patch: (c) => { const d = Math.max(64, Math.round(c.d / Math.SQRT2 / 64) * 64), heads = near(divs(d).filter((x) => d / x >= 32), c.heads * (d / c.d)); return { L: c.L * 2, d, ff: Math.round(d * (c.ff / c.d)), heads, kv: c.kv === c.heads ? heads : near(divs(heads), heads / (c.heads / c.kv)), blocks: {} }; },
    why: "At the same size, twice the blocks at 0.7× the width. For small models (under ~1B), MobileLLM found depth beats width: more rounds of processing matter more than wider vectors.",
    cost: "More sequential steps per token, so higher latency; past the 20–200 width-per-block range, loss gets worse.",
  },
  {
    id: "orig", name: "The 2017 original", who: "Attention Is All You Need",
    patch: () => ({ normPos: "post", norm: "LayerNorm", act: "ReLU", bias: true }),
    why: "Norm after the add, ReLU MLP, biases everywhere. It trained (with warmup) at 6 layers, and at that depth can even edge out pre-norm. Worth trying to see why everyone moved away from it.",
    cost: "Deep post-norm models diverge early in training without careful warmup.",
  },
  {
    id: "tie", name: "Tie the readout", who: "GPT-2, Gemma",
    patch: (c) => ({ tied: !c.tied }),
    why: "The embedding turns a token into a vector; the readout turns a vector into token scores. Using one table for both saves vocabulary × width parameters, a big share of a small model.",
    cost: "Big models usually untie: the two jobs differ a little, and at their size the saving is small.",
  },
];
