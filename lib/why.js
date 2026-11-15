// Pieces for the "electrons to CUDA" level: a goal box that says what a layer is for before how it works,
// a chain of "why?" questions you dig through one at a time until you reach a law of physics, and a map
// of the whole stack with this chapter's layer lit.
import { h } from "./core.js";
import { burst } from "./pixel.js";
import { Scene, shade } from "./voxel.js";

/** The stack, top (what we want) to bottom (what we have). Each row is a chapter. */
export const STACK = [
  ["ai-physics", "The job", "AI is a function: trillions of multiply-adds"],
  ["cuda", "The program", "CUDA: describe a million threads"],
  ["gpu-design", "The machine", "SMs, warps, registers, tensor cores"],
  ["power-wall", "The constraint", "Heat: why go wide, not fast"],
  ["memory-physics", "The bottleneck", "Bits that are far away"],
  ["logic", "The arithmetic", "Gates, adders, multipliers, clocks"],
  ["transistor", "The switch", "A field that opens a channel"],
  ["electrons", "The material", "Electrons in silicon"],
];

/** What this layer is for, before how it does it. needs: short requirement strings. */
export function goal(html, needs = []) {
  return h("aside.goal",
    h("div.goal-k", h("span.goal-ic", { "aria-hidden": "true" }), "The goal first"),
    h("div.goal-b", { html }),
    needs.length ? h("ul.goal-needs", needs.map((n, i) => h("li", h("b", `need ${i + 1}`), n))) : null);
}

/**
 * Dig down: a claim, then "Why?" one level at a time, until an axiom (a law we take as given).
 * steps: [{q, a}] where q is the question at that depth ("Why is that?") and a the answer (html).
 */
export function whyChain({ claim, steps, axiom }) {
  let shown = 0;
  const list = h("ol.why-list");
  const more = h("button.btn.small.primary.why-go", { type: "button", onclick: () => reveal(shown + 1, true) });
  const all = h("button.btn.small.why-all", { type: "button", onclick: () => reveal(steps.length + 1, false) }, "Show all");
  const depth = h("span.why-depth");
  const rows = steps.map((st, i) => h("li.why-step", { style: { "--d": String(i / Math.max(1, steps.length)) } },
    h("div.why-q", h("span.why-n", String(i + 1)), st.q || "Why?"), h("div.why-a", { html: st.a })));
  const bed = h("li.why-axiom",
    h("div.why-rock", { "aria-hidden": "true" }, Array.from({ length: 14 }, (_, k) => h("i", { style: { "--k": String(k % 4) } }))),
    h("div.why-ax", h("span.why-tag", "Bedrock"), h("b", axiom.name), h("div", { html: axiom.html })));
  function reveal(n, celebrate) {
    shown = Math.min(steps.length + 1, n);
    list.replaceChildren(...rows.slice(0, Math.min(shown, steps.length)), ...(shown > steps.length ? [bed] : []));
    const left = steps.length - shown;
    more.hidden = shown > steps.length; all.hidden = more.hidden;
    more.textContent = left > 0 ? `Why? (${left + 1} more down)` : "Why? (last one)";
    depth.textContent = shown ? `depth ${Math.min(shown, steps.length)} of ${steps.length}${shown > steps.length ? " · bedrock" : ""}` : `${steps.length} levels to bedrock`;
    if (celebrate && shown > steps.length) burst(bed);
  }
  reveal(0, false);
  return h("div.why", h("div.why-head", h("span.eyebrow", "Ask why until it stops"), depth),
    h("p.why-claim", { html: claim }), list, h("div.why-nav", more, all));
}

/** The whole stack as layers, this chapter's lit, each a link. */
export function stackMap(current) {
  return h("nav.stk", { "aria-label": "The stack from AI down to electrons" },
    h("div.stk-head", h("span.eyebrow", "Where this sits"), h("span.muted", "top: what we want · bottom: what physics gives us")),
    h("ol", STACK.map(([id, k, t], i) => h("li", { class: id === current ? "on" : "", style: { "--i": String(i / (STACK.length - 1)) } },
      h("a", { href: `/learn?c=${id}`, "aria-current": id === current ? "page" : null }, h("span.stk-k", k), h("span.stk-t", t), id === current ? h("span.stk-here", "you are here") : null)))));
}

// ---------------------------------------------------------------------------------------------
// Voxel stickers for this level: sand to chip, the crystal, three shapes of transistor, a GPU package
// ---------------------------------------------------------------------------------------------
const SI = "#8ea3c7", BOND = "#d5dceb", P_ = "#ff8a3d", B_ = "#e76fa8", E_ = "#ffd23f", METAL = "#9aa3b5", OX = "#f2f4f8";
export const SCENES = {
  sand(s) {   // a little heap of quartz sand
    for (let i = 0; i < 7; i++) for (let j = 0; j < 7; j++) {
      const hgt = Math.max(0, 3 - Math.round(Math.hypot(i - 3, j - 3) * 0.9));
      for (let k = 0; k <= hgt; k++) if (Math.hypot(i - 3, j - 3) < 3.6) s.put(i, j, k, (i * 3 + j + k) % 3 ? "#e3c27a" : "#d2ad5f", { top: (i + j) % 2 ? "#f3dca2" : "#ecd08c" });
    }
  },
  ingot(s) {  // a pulled single crystal, standing up
    for (let k = 0; k < 7; k++) for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++) {
      const r = k === 6 ? 1.3 : k === 0 ? 2.0 : 2.5;
      if (Math.hypot(i - 2, j - 2) <= r) s.put(i, j, k, "#7e8aa3", { top: "#c2cbe0", right: shade("#7e8aa3", 0.7) });
    }
  },
  wafer(s) {  // a thin disc ruled into dies
    for (let i = 0; i < 13; i++) for (let j = 0; j < 13; j++) {
      if (Math.hypot(i - 6, j - 6) > 6.3) continue;
      const edge = Math.hypot(i - 6, j - 6) > 5.4;
      s.put(i, j, 0, "#6b7a99", { top: edge ? "#9aa8c4" : (Math.floor(i / 2) + Math.floor(j / 2)) % 2 ? "#b9a3ff" : "#8fd7ff" });
    }
  },
  chip(s) {
    s.box(0, 0, 0, 8, 8, 1, "#1e7a4c", (i, j) => ({ faces: { top: (i === 0 || i === 7 || j === 0 || j === 7) && (i + j) % 2 ? "#d4af37" : "#2f9a62" } }));
    s.box(1, 1, 1, 6, 6, 1, "#2a2d36", () => ({ faces: { top: "#3a3e4a" } }));
    s.box(2, 2, 2, 4, 4, 1, "#aeb6c8", (i, j) => ({ faces: { top: (i + j) % 2 ? "#d7ddea" : "#c4ccdd" } }));
  },
  lattice(s, kind = "pure") {   // one plane of the crystal: atoms on even sites, shared bonds between
    const N = 4;
    for (let i = 0; i <= 2 * N; i++) for (let j = 0; j <= 2 * N; j++) {
      const atom = i % 2 === 0 && j % 2 === 0, bond = (i + j) % 2 === 1;
      const centre = i === N && j === N;
      if (atom) s.put(i, j, 0, centre && kind === "n" ? P_ : centre && kind === "p" ? B_ : SI, { top: centre && kind !== "pure" ? undefined : "#c3d0ea" });
      else if (bond && !(kind === "p" && i === N + 1 && j === N)) s.put(i, j, 0, BOND, { top: "#eef1f8" });
    }
    if (kind === "n") s.put(N + 1, N - 1, 2, E_, { top: "#fff1a8" });   // the fifth electron, loose
  },
  planar(s, on = true) {   // a planar MOSFET in cross-section: source, channel under the gate, drain
    s.box(0, 0, 0, 12, 4, 3, "#e8a7bd", (i, j, k) => {
      if (k === 2 && (i <= 2 || i >= 9)) return { color: "#6c8cff", faces: { top: "#9fb5ff" } };          // n+ source and drain
      if (k === 2 && on && i > 2 && i < 9) return { color: "#7fd0ff", faces: { top: "#b8e8ff" } };        // the channel, when the gate pulls electrons up
      return { faces: { top: "#f3c6d5" } };
    });
    s.box(3, 0, 3, 6, 4, 1, OX, () => ({ faces: { top: "#ffffff" } }));                                  // thin oxide
    s.box(3, 0, 4, 6, 4, 2, METAL, () => ({ faces: { top: on ? "#ffe07a" : "#c6ccd9" } }));               // gate
    s.box(0, 1, 3, 2, 2, 2, "#d4af37"); s.box(10, 1, 3, 2, 2, 2, "#d4af37");                               // contacts
  },
  finfet(s) {   // the channel stands up as a fin, and the gate wraps three sides
    s.box(0, 0, 0, 12, 6, 1, "#e8a7bd", () => ({ faces: { top: "#f3c6d5" } }));
    s.box(0, 2, 1, 12, 2, 4, "#6c8cff", (i) => (i > 3 && i < 8 ? { color: "#7fd0ff", faces: { top: "#b8e8ff" } } : { faces: { top: "#9fb5ff" } }));
    s.box(5, 1, 1, 2, 4, 5, METAL, (i, j, k) => (j > 0 && j < 3 && k < 4 ? false : { faces: { top: "#ffe07a" } }));
  },
  gaa(s) {   // gate-all-around: stacked nanosheets, gate on every side
    s.box(0, 0, 0, 12, 6, 1, "#e8a7bd", () => ({ faces: { top: "#f3c6d5" } }));
    for (const z of [1, 3, 5]) s.box(0, 2, z, 12, 2, 1, "#6c8cff", (i) => (i > 3 && i < 8 ? { color: "#7fd0ff", faces: { top: "#b8e8ff" } } : { faces: { top: "#9fb5ff" } }));
    s.box(5, 1, 1, 2, 4, 6, METAL, (i, j, k) => (j > 0 && j < 3 && k % 2 === 0 ? false : { faces: { top: "#ffe07a" } }));
  },
  package(s) {   // a GPU die on a silicon interposer, with stacks of DRAM (HBM) right beside it
    s.box(0, 0, 0, 16, 11, 1, "#c9b37e", () => ({ faces: { top: "#e2d2a4" } }));
    s.box(5, 2, 1, 6, 7, 1, "#2a2d36", (i, j) => ({ faces: { top: (i + j) % 2 ? "#4b5163" : "#3a3e4a" } }));
    for (const [x, y] of [[1, 1], [1, 4], [1, 7], [12, 1], [12, 4], [12, 7]])
      s.box(x, y, 1, 3, 2, 5, "#6d7890", (i, j, k) => ({ color: k % 2 ? "#59637a" : "#7a86a0", faces: { top: k === 4 ? "#aab4c9" : undefined } }));
  },
  dram(s, charged = true) {   // one DRAM bit: an access transistor and a tall capacitor that holds the charge
    s.box(0, 0, 0, 6, 3, 1, "#e8a7bd", () => ({ faces: { top: "#f3c6d5" } }));
    s.box(0, 0, 1, 2, 3, 1, METAL, () => ({ faces: { top: "#ffe07a" } }));
    for (let k = 1; k < 8; k++) s.box(3, 0, k, 2, 2, 1, k <= (charged ? 6 : 1) ? "#ffd23f" : "#d9dce4", () => ({ faces: { top: k === (charged ? 6 : 1) ? "#fff1a8" : undefined } }));
  },
};

/** A voxel sticker with a caption. px: pixels per voxel pixel. */
export function sticker(key, { arg, label = key, title, caption, px = 4 } = {}) {
  const sc = new Scene(); SCENES[key](sc, arg);
  const svg = sc.svg({ cls: "vx", label });
  const [, W, H] = /viewBox="0 0 (\d+) (\d+)"/.exec(svg) || [0, 60, 40];
  return h("figure.vx-fig", h("div", { html: svg.replace("<svg ", `<svg width="${W * px}" height="${H * px}" `) }),
    title || caption ? h("figcaption", title ? h("b", title) : null, caption || null) : null);
}
