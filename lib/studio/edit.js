// The Studio's editors: click a part, change it. Each editor takes get() for the current design and
// set(patch) to change it; set redraws the 3D view and the predictions without rebuilding the editor,
// so a slider keeps working while you drag it.
import { h } from "../core.js";
import { slider, segmented } from "../widgets.js";
import { blk } from "./arch.js";

export const WIDTHS = [64, 128, 192, 256, 384, 512, 640, 768, 1024, 1280, 1536, 2048, 2560, 3072, 4096, 5120, 6144, 8192, 12288, 16384];
export const HEADS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 96, 128];
const CTXS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072];
const VOCABS = [65, 256, 32000, 50257, 100277, 128256, 256000];
const FFR = [1, 1.5, 2, 2.667, 3, 3.5, 4, 6, 8];
const WINDOWS = [64, 128, 256, 512, 1024, 2048, 4096, 8192];
const EXPERTS = [2, 4, 8, 16, 32, 64, 128, 256];
export const TOKMUL = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000];
const n = (x) => x.toLocaleString("en-US");
const near = (list, v) => list.reduce((bi, x, i) => (Math.abs(x - v) < Math.abs(list[bi] - v) ? i : bi), 0);

/** A slider over a list of allowed values. */
const pick = (label, list, value, fmt, on, hint) => slider({ label, min: 0, max: list.length - 1, step: 1, value: near(list, value), fmt: (i) => fmt(list[i]), onInput: (i) => on(list[i]), hint });
const seg = (label, opts, value, on) => segmented({ label, options: opts.map((o) => (Array.isArray(o) ? { value: o[0], label: o[1] } : { value: o, label: String(o) })), value, onChange: on });

const heads = (get, set) => pick("Heads", HEADS, get().heads, (v) => `${v} (${get().d % v ? "doesn't divide" : `${get().d / v} each`})`, (v) => { const c = get(); set({ heads: v, kv: c.kv === c.heads ? v : c.kv }); });
const kv = (get, set) => pick("K/V groups", HEADS, get().kv, (v) => (v === get().heads ? `${v} (one per head)` : v === 1 ? "1 (multi-query)" : `${v} (grouped)`), (v) => set({ kv: v }), "Fewer groups: a smaller KV cache.");
const width = (get, set) => pick("Width", WIDTHS, get().d, (v) => n(v), (v) => { const c = get(); set({ d: v, ff: Math.round(v * (c.ff / c.d)) }); }, "Numbers per token vector. The MLP keeps its ratio.");
const layers = (get, set) => slider({ label: "Blocks", min: 1, max: 128, step: 1, value: get().L, onInput: (v) => set({ L: v }) });
const ctx = (get, set) => pick("Context", CTXS, get().ctx, (v) => `${n(v)} tokens`, (v) => set({ ctx: v }));
const vocab = (get, set) => pick("Vocabulary", VOCABS, get().vocab, (v) => (v === 65 ? "65 (characters)" : v === 256 ? "256 (bytes)" : n(v)), (v) => set({ vocab: v }));
const pos = (get, set) => seg("Positions", [["learned", "Learned table"], ["RoPE", "RoPE"], ["none", "None"]], get().pos, (v) => set({ pos: v }));

/** Editor for whatever part is selected. Returns null if the part has nothing to change. */
export function partEditor(edit, get, set) {
  if (!edit) return null;
  if (typeof edit === "object" && edit.block) return blockEditor(edit.block, get, set);
  const E = {
    attn: () => [heads(get, set), kv(get, set), pick("Window size", WINDOWS, get().window, (v) => `${n(v)} tokens`, (v) => set({ window: v }), "Used by blocks set to sliding window."), pos(get, set)],
    mlp: () => [...(get().blocks && Object.values(get().blocks).some((b) => b.mlp === "moe") ? [
      pick("Experts", EXPERTS, get().experts, (v) => `${v} per block`, (v) => set({ experts: v, topk: Math.min(get().topk, v) }), "Used by blocks set to Experts. More experts: more knowledge, same compute per token."),
      pick("Experts per token", [1, 2, 4, 8], get().topk, (v) => (v === 1 ? "1 (Switch)" : v === 2 ? "2 (Mixtral)" : String(v)), (v) => set({ topk: Math.min(v, get().experts) }))] : []),
      pick("MLP width", FFR, get().ff / get().d, (r) => `${r === 2.667 ? "8/3" : r}× = ${n(Math.round(get().d * r))}`, (r) => set({ ff: Math.round(get().d * r) })),
      seg("Activation", ["GELU", "ReLU", "SwiGLU"], get().act, (v) => set({ act: v })),
      h("small.hint", "SwiGLU adds a third matrix. Llama narrows the MLP (about 3.5× instead of 4×) so the size stays similar.")],
    norm: () => [seg("Normalisation", ["LayerNorm", "RMSNorm", "None"], get().norm, (v) => set({ norm: v })),
      seg("Placement", [["pre", "Before each part"], ["post", "After the add"]], get().normPos, (v) => set({ normPos: v }))],
    residual: () => [seg("Residual stream", [[true, "On"], [false, "Off"]], get().residual, (v) => set({ residual: v })),
      h("small.hint", "Off: each part replaces the stream instead of adding to it. Watch the predictions.")],
    embed: () => [vocab(get, set), seg("Readout", [[true, "Tied to the embedding"], [false, "Its own matrix"]], get().tied, (v) => set({ tied: v }))],
    pos: () => [pos(get, set), ctx(get, set)],
  };
  return E[edit] ? h("div.st-edit", E[edit]()) : null;
}

const SCOPES = [["one", "This block"], ["all", "All"], ["odd", "Every other"], ["lo", "First half"], ["hi", "Last half"]];
function blockEditor(b, get, set) {
  let scope = "one";
  const targets = () => { const L = get().L; return Array.from({ length: L }, (_, i) => i + 1).filter((i) => (scope === "one" ? i === b : scope === "all" ? true : scope === "odd" ? i % 2 === b % 2 : scope === "lo" ? i <= L / 2 : i > L / 2)); };
  const apply = (patch) => {
    const blocks = { ...get().blocks };
    for (const i of targets()) {
      const s = { ...blk(get(), i), ...patch };
      if (s.attn === "full" && s.mlp) delete blocks[i]; else blocks[i] = s;
    }
    set({ blocks });
  };
  const own = blk(get(), b);
  return h("div.st-edit",
    seg("Apply to", SCOPES, scope, (v) => { scope = v; }),
    seg("Attention", [["full", "Full"], ["window", "Sliding window"], ["none", "None"]], own.attn, (v) => apply({ attn: v })),
    seg("MLP", [[true, "Dense"], ["moe", "Experts"], [false, "Off"]], own.mlp, (v) => apply({ mlp: v })),
    h("button.btn.small", { type: "button", onclick: () => set({ blocks: {} }) }, "Make every block standard again"));
}

/** The whole-model knobs on the Design tab. tok: { get, set } for training tokens per parameter. */
export function designEditor(get, set, tok) {
  return h("div.st-edit",
    layers(get, set), width(get, set), heads(get, set), kv(get, set), ctx(get, set), vocab(get, set),
    pick("Training data", TOKMUL, tok.get(), (v) => `${v} tokens per parameter${v === 20 ? " (compute-optimal)" : ""}`, (v) => tok.set(v), "Used by the loss and cost predictions."));
}
