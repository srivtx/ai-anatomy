// Pieces for the "electrons to CUDA" level: a goal box that says what a layer is for before how it works,
// a chain of "why?" questions you dig through one at a time until you reach a law of physics, and a map
// of the whole stack with this chapter's layer lit.
import { h } from "./core.js";
import { burst } from "./pixel.js";
import { Scene, shade } from "./voxel.js";

/** The stack, top (what we want) to bottom (what we have). Each row is a chapter. */
export const STACK = [
  ["ai-physics", "The job", "AI is a function: trillions of multiply-adds"],
  ["software-stack", "The layers", "Python, compilers, driver, machine code"],
  ["languages", "The languages", "From cables to Python, 80 years of translators"],
  ["cuda", "The program", "CUDA: describe a million threads"],
  ["gpu-design", "The machine", "SMs, warps, registers, tensor cores"],
  ["cpu", "The other machine", "CPU: one thread, as fast as it can go"],
  ["power-wall", "The constraint", "Heat: why go wide, not fast"],
  ["memory-physics", "The bottleneck", "Bits that are far away"],
  ["logic", "The arithmetic", "Gates, adders, multipliers, clocks"],
  ["fab", "The factory", "Printing 80 billion switches with light"],
  ["transistor", "The switch", "A field that opens a channel"],
  ["circuits", "The laws", "Volts, amps, ohms and farads"],
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
  bits(s, word = 0b0100000010000100) {   // a bf16 number as 16 wires: sign, 8 exponent, 7 fraction; a 1 stands up
    for (let j = 0; j < 16; j++) {
      const bit = (word >> (15 - j)) & 1, c = j === 0 ? B_ : j < 9 ? P_ : "#6c8cff";
      s.put(j, 0, 0, shade(c, 0.85), { top: shade(c, 1.25) });
      if (bit) s.box(j, 0, 1, 1, 1, 2, c, () => ({ faces: { top: E_ } }));
    }
  },
  nand(s) {   // four transistors: two p-type in parallel on top, two n-type in series below, wired in gold
    s.box(0, 0, 0, 9, 7, 1, "#e8a7bd", () => ({ faces: { top: "#f3c6d5" } }));
    for (const [x, y, c] of [[1, 1, "#ff9ec4"], [5, 1, "#ff9ec4"], [3, 4, "#6c8cff"], [3, 6, "#6c8cff"]]) {
      s.box(x, y, 1, 3, 1, 1, c);
      s.box(x + 1, y, 2, 1, 1, 1, METAL, () => ({ faces: { top: "#ffe07a" } }));
    }
    s.box(0, 0, 1, 9, 1, 1, "#d4af37"); s.box(2, 3, 1, 5, 1, 1, "#d4af37");
  },
  sm(s) {   // one SM: four sub-partitions, each a tensor core (tall) beside a register file, over shared memory
    s.box(0, 0, 0, 11, 11, 1, "#2a2d36", () => ({ faces: { top: "#3a3e4a" } }));
    s.box(1, 9, 1, 9, 1, 1, "#8fd7ff", () => ({ faces: { top: "#c6ecff" } }));
    for (const [x, y] of [[1, 1], [6, 1], [1, 5], [6, 5]]) {
      s.box(x, y, 1, 2, 2, 3, "#7b61ff", () => ({ faces: { top: "#b9a3ff" } }));
      s.box(x + 2, y, 1, 2, 3, 1, "#ff8a3d", (i, j) => ({ faces: { top: j % 2 ? "#ffc29a" : "#ffad75" } }));
      s.box(x, y + 2, 1, 2, 1, 1, "#22b8a0");
    }
  },
  tensor(s) {   // a matrix product as a cube of multiply-adds: A along one side, B along the other, C on top
    const N = 4;
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) for (let k = 0; k < N; k++) {
      const edge = i === 0 || j === N - 1;
      s.put(i, j, k, edge ? (i === 0 ? "#6c8cff" : "#ff8a3d") : "#7b61ff", { top: k === N - 1 ? "#5fd176" : undefined });
    }
  },
  grid(s) {   // a CUDA grid: 3 × 2 blocks, each a 4 × 2 patch of threads, one colour per block
    const cols = ["#7b61ff", "#22b8a0", "#ff8a3d", "#4f8ef7", "#ff7eb6", "#ffd23f"];
    for (let b = 0; b < 6; b++) {
      const bx = (b % 3) * 5, by = Math.floor(b / 3) * 3;
      s.box(bx, by, 0, 4, 2, 1, cols[b], (i, j) => ({ faces: { top: shade(cols[b], (i + j) % 2 ? 1.25 : 1.1) } }));
    }
  },
  softstack(s) {   // layers of software, widest at the bottom: silicon, machine code, driver, runtime, library, framework, Python
    const L = [["#2a2d36", 12], ["#22b8a0", 11], ["#9aa3b5", 10], ["#4f8ef7", 9], ["#7b61ff", 8], ["#ff8a3d", 7], ["#ffd23f", 5]];
    L.forEach(([c, w], k) => { const o = (12 - w) / 2; s.box(Math.floor(o), Math.floor(o), k, w, w, 1, c, (i, j) => ({ faces: { top: shade(c, (i + j) % 2 ? 1.3 : 1.15) } })); });
  },
  cpudie(s) {   // a CPU: 8 big cores, each mostly control and cache with a small block of arithmetic (orange), around a shared L3
    s.box(0, 0, 0, 14, 12, 1, "#2a2d36", () => ({ faces: { top: "#3a3e4a" } }));
    s.box(1, 5, 1, 12, 2, 1, "#4f8ef7", (i) => ({ faces: { top: i % 2 ? "#8fb8ff" : "#a8c8ff" } }));
    for (const [x, y] of [[1, 1], [4, 1], [7, 1], [10, 1], [1, 8], [4, 8], [7, 8], [10, 8]]) {
      s.box(x, y, 1, 3, 3, 1, "#1e9e86", (i, j) => ({ faces: { top: (i + j) % 2 ? "#5fd1b8" : "#46c2a6" } }));
      s.put(x + 1, y + 1, 2, "#ff8a3d", { top: "#ffc29a" });
    }
  },
  pipeline(s) {   // five stages in a row, each holding a different instruction (purple) at the same moment
    const C = ["#4f8ef7", "#7b61ff", "#ffb000", "#22b8a0", "#ff7eb6"];
    C.forEach((c, k) => {
      s.box(k * 3, 0, 0, 2, 4, 1, c, (i, j) => ({ faces: { top: shade(c, (i + j) % 2 ? 1.3 : 1.15) } }));
      s.box(k * 3, 1, 1, 2, 2, 1 + (k % 2), "#5b3fd6", () => ({ faces: { top: "#b9a3ff" } }));
    });
  },
  punchcard(s) {   // a punched card: cream card, a hole (dark) is a 1
    const holes = [[1, 1], [2, 3], [3, 0], [4, 2], [5, 4], [6, 1], [7, 3], [8, 0], [9, 2], [10, 4], [11, 1], [12, 3], [3, 4], [7, 0], [10, 1]];
    const H = new Set(holes.map(([x, y]) => `${x},${y}`));
    s.box(0, 0, 0, 14, 5, 1, "#e9d9a8", (i, j) => ({ color: H.has(`${i},${j}`) ? "#3a3e4a" : "#e9d9a8", faces: { top: H.has(`${i},${j}`) ? "#22252d" : (i + j) % 2 ? "#f5e9c4" : "#f1e2b6" } }));
  },
  raster(s) {   // a pixel grid with a triangle filled in, blended between three corner colours; covered pixels raised
    const V = [[1, 1], [10, 3], [4, 10]], E = (a, b, p) => (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]), A = E(V[0], V[1], V[2]);
    for (let i = 0; i < 12; i++) for (let j = 0; j < 12; j++) {
      const p = [i + 0.5, j + 0.5], w = [E(V[1], V[2], p) / A, E(V[2], V[0], p) / A, E(V[0], V[1], p) / A];
      if (w.every((x) => x >= 0)) { const r = Math.round(235 * w[0] + 40 * w[1] + 60 * w[2]), g = Math.round(80 * w[0] + 200 * w[1] + 110 * w[2]), b = Math.round(70 * w[0] + 90 * w[1] + 235 * w[2]); const col = `#${[r, g, b].map((x) => x.toString(16).padStart(2, "0")).join("")}`; s.put(i, j, 1, col, { top: shade(col, 1.2) }); }
      else s.put(i, j, 0, (i + j) % 2 ? "#3a3e4a" : "#2f333d", { top: (i + j) % 2 ? "#4b5163" : "#434857" });
    }
  },
  tunnel(s) {   // electrons (yellow) piled against a thin wall; a few have leaked through to the other side
    s.box(0, 0, 0, 13, 6, 1, "#c7cfdd", (i, j) => ({ faces: { top: (i + j) % 2 ? "#e3e8f1" : "#d8dee9" } }));
    s.box(6, 0, 1, 1, 6, 4, "#7b61ff", () => ({ faces: { top: "#b9a3ff" } }));
    for (const [x, y, z] of [[1, 1, 1], [2, 3, 1], [3, 0, 1], [4, 2, 1], [5, 4, 1], [5, 1, 1], [3, 4, 1], [1, 4, 1], [4, 5, 1], [2, 1, 2], [4, 3, 2], [9, 2, 1], [11, 4, 1]]) s.put(x, y, z, "#ffd23f", { top: "#fff1a8" });
  },
  resistor(s) {   // a through-hole resistor: beige body with colour bands, wire leads, on a board
    s.box(0, 0, 0, 14, 5, 1, "#1e7a4c", (i, j) => ({ faces: { top: (i + j) % 2 ? "#2f9a62" : "#2a8e59" } }));
    s.box(0, 2, 1, 14, 1, 1, METAL, () => ({ faces: { top: "#d7ddea" } }));
    const bands = { 4: "#8a4b16", 6: "#111318", 7: "#e23b2e", 9: "#d4af37" };
    s.box(3, 1, 1, 8, 3, 2, "#e6c89a", (i) => ({ color: bands[i] || "#e6c89a", faces: { top: bands[i] ? shade(bands[i], 1.3) : "#f3dfbd" } }));
  },
  capacitor(s) {   // two metal plates with an empty gap: electrons (yellow) crowd onto the near plate, the far one is left short of them (pink)
    s.box(0, 0, 0, 11, 6, 1, "#c7cfdd", (i, j) => ({ faces: { top: (i + j) % 2 ? "#e3e8f1" : "#d8dee9" } }));
    s.box(3, 0, 1, 1, 6, 6, "#9aa3b5", () => ({ faces: { top: "#d7ddea" } }));
    s.box(7, 0, 1, 1, 6, 6, "#9aa3b5", () => ({ faces: { top: "#d7ddea" } }));
    for (const [y, z] of [[0, 2], [1, 5], [2, 3], [3, 6], [4, 2], [5, 4], [1, 1], [3, 4]]) s.put(8, y, z, E_, { top: "#fff1a8" });
    for (const [y, z] of [[0, 4], [2, 6], [3, 2], [5, 5]]) s.put(2, y, z, B_, { top: "#f6b3d3" });
  },
  metalstack(s) {   // a chip in cross-section: transistors (orange) on the silicon, then copper wiring layers that cross at right angles and get thicker going up
    s.box(0, 0, 0, 12, 12, 1, "#e8a7bd", () => ({ faces: { top: "#f3c6d5" } }));
    for (let i = 1; i < 12; i += 3) for (let j = 1; j < 12; j += 3) s.put(i, j, 1, "#ff6b3d", { top: "#ffb08a" });
    let z = 2;
    [1, 1, 2, 2, 3].forEach((w, k) => {
      for (let t = 0; t + w <= 12; t += w * 2) k % 2 ? s.box(t, 0, z, w, 12, 1, "#b8692c", () => ({ faces: { top: "#e8a66a" } })) : s.box(0, t, z, 12, w, 1, "#c27a3a", () => ({ faces: { top: "#f0b27a" } }));
      z += 2;
    });
  },
  litho(s) {   // a scanner, simplified: light (purple) through a mask pattern, focused down onto a wafer
    s.box(0, 0, 0, 13, 13, 1, "#6b7a99", (i, j) => ({ faces: { top: Math.hypot(i - 6, j - 6) > 6.3 ? undefined : (Math.floor(i / 3) + Math.floor(j / 3)) % 2 ? "#b9a3ff" : "#8fd7ff" } }));
    for (let k = 1; k < 6; k++) { const r = 1 + (k - 1) * 0.6; for (let i = 0; i < 13; i++) for (let j = 0; j < 13; j++) if (Math.hypot(i - 6, j - 6) <= r) s.put(i, j, k, "#a78bfa", { top: "#d9ccff" }); }
    s.box(2, 2, 6, 9, 9, 1, "#e9eef6", (i, j) => ({ color: i % 2 ? "#2a2d36" : "#e9eef6", faces: { top: i % 2 ? "#3a3e4a" : "#ffffff" } }));
  },
  defects(s) {   // a wafer of dies with a few killer defects (red); a die with a defect is lost
    const bad = new Set(["2,4", "6,1", "8,7", "4,9", "10,5"]);
    for (let i = 0; i < 13; i++) for (let j = 0; j < 13; j++) {
      if (Math.hypot(i - 6, j - 6) > 6.3) continue;
      const die = `${Math.floor(i / 2) * 2},${Math.floor(j / 2) * 2}`, hit = [...bad].some((b) => { const [x, y] = b.split(",").map(Number); return Math.floor(x / 2) * 2 + "," + Math.floor(y / 2) * 2 === die; });
      s.put(i, j, 0, "#6b7a99", { top: bad.has(`${i},${j}`) ? "#e23b2e" : hit ? "#f2a3a0" : (Math.floor(i / 2) + Math.floor(j / 2)) % 2 ? "#5fd176" : "#7fe08f" });
    }
  },
  // ---- the AI side: stickers for the model chapters ----
  tokens(s) {   // text cut into tokens: bricks of different lengths, one colour per token
    const P = [["#7b61ff", 2], ["#22b8a0", 4], ["#ff8a3d", 3], ["#4f8ef7", 2]];
    let x = 0;
    for (const [c, w] of P) { s.box(x, 0, 0, w, 2, 1, c, (i, j) => ({ faces: { top: shade(c, (i + j) % 2 ? 1.32 : 1.14) } })); x += w + 1; }
  },
  embed(s) {   // words as points in space: three clusters floating over a floor, each with its shadow
    s.box(0, 0, 0, 10, 10, 1, "#c9d1e2", (i, j) => ({ faces: { top: (i + j) % 2 ? "#eef1f7" : "#e3e8f1" } }));
    const C = [["#7b61ff", [[1, 1, 3], [2, 1, 3], [1, 2, 3], [2, 2, 4]]], ["#22b8a0", [[7, 1, 2], [8, 2, 2], [7, 2, 3]]], ["#ff8a3d", [[4, 7, 3], [5, 7, 2], [4, 8, 2], [5, 8, 3]]]];
    for (const [c, pts] of C) for (const [x, y, z] of pts) { s.put(x, y, 0, "#c9d1e2", { top: "#a9b2c6" }); s.put(x, y, z, c); }
  },
  attn(s) {   // a causal attention map: query along one side, key along the other; height = weight; the future is masked
    const N = 6, W = [[4], [2, 3], [1, 1, 4], [1, 2, 1, 3], [1, 1, 1, 2, 3], [2, 1, 1, 1, 1, 3]];
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const x = i, y = j;   // the masked future (dark) faces you; the diagonal, each word looking at itself, is a ridge
      if (j > i) { s.put(x, y, 0, "#3a3e4a", { top: "#4a4f5d" }); continue; }
      const hgt = W[i][j], c = hgt >= 3 ? "#7b61ff" : hgt === 2 ? "#9c88ff" : "#c3b8ff";
      s.box(x, y, 0, 1, 1, hgt, c);
    }
  },
  bowl(s) {   // a loss surface as terrain, with a ball rolling down the back slope to the bottom along the gradient
    const N = 9, H = (x, y) => Math.min(5, 1 + Math.round(((x - 4) ** 2 + (y - 4) ** 2) / 6));
    const col = ["#2f6fd6", "#22b8a0", "#5fd176", "#ffd23f", "#ff8a3d"];
    const path = new Set(["1,2", "2,2", "2,3", "3,3", "4,4"]);
    for (let x = 0; x < N; x++) for (let y = 0; y < N; y++) {
      if (x > 4 && y > 4) continue;   // cut away the front quarter so you can see down to the bottom
      const hh = H(x, y), c = col[hh - 1];
      s.box(x, y, 0, 1, 1, hh, c, (i, j, k) => ({ faces: k === hh - 1 && path.has(`${x},${y}`) ? { top: "#ffffff" } : {} }));
    }
    s.put(1, 1, H(1, 1), "#ff4f8b", { top: "#ffb3cc" });
  },
  conv(s) {   // a 3 × 3 filter hovering over an image with a vertical edge, its footprint lit below
    for (let x = 0; x < 8; x++) for (let y = 0; y < 8; y++) {
      const under = x >= 2 && x <= 4 && y >= 3 && y <= 5;
      s.put(x, y, 0, x < 4 ? "#2a2d36" : "#d5dceb", { top: under ? (x < 4 ? "#a0522d" : "#ffc29a") : x < 4 ? "#3a3e4a" : "#f2f4f8" });
    }
    s.box(2, 3, 4, 3, 3, 1, "#ff8a3d", (i, j) => ({ faces: { top: i === 0 ? "#ffd0b0" : i === 2 ? "#c75a1e" : "#ffa766" } }));
  },
  noise(s) {   // diffusion: the same picture under heavy noise, half noise, none
    let seed = 7; const r = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
    const IMG = ["..#..", ".###.", "#####", ".###.", "..#.."], PAL = ["#7b61ff", "#22b8a0", "#ff8a3d", "#4f8ef7", "#ffd23f", "#e76fa8"];
    [0.9, 0.45, 0].forEach((lvl, t) => {
      for (let x = 0; x < 5; x++) for (let y = 0; y < 5; y++) {
        const on = IMG[y][x] === "#", c = r() < lvl ? PAL[Math.floor(r() * PAL.length)] : on ? "#ff4f8b" : "#e9eef6";
        s.put(t * 7 + x, y, 0, c);
      }
    });
  },
  experts(s) {   // mixture of experts: a router sends this token to 2 of 8 experts; the rest stay dark
    s.box(0, 0, 0, 15, 8, 1, "#2a2d36", (i, j) => ({ faces: { top: (i + j) % 2 ? "#3a3e4a" : "#353944" } }));
    for (let e = 0; e < 8; e++) {
      const on = e === 2 || e === 5, c = on ? (e === 2 ? "#ff8a3d" : "#22b8a0") : "#5b6172";
      s.box(e * 2, 0, 1, 1, 2, on ? 4 : 2, c);
      if (on) for (let y = 2; y < 5; y++) s.put(e * 2, y, 1, c, { top: shade(c, 1.35) });
    }
    s.box(6, 5, 1, 4, 2, 1, "#7b61ff", (i, j) => ({ faces: { top: (i + j) % 2 ? "#a996ff" : "#9c88ff" } }));
    s.put(7, 6, 2, "#ffd23f", { top: "#fff0a8" });
  },
  kvcache(s) {   // the KV cache: every token leaves a key (blue) and a value (orange) in every layer; the newest is lit
    const T = 7, L = 4;   // tokens run left to right, layers front to back; each cell is a key with its value on top
    for (let t = 0; t < T; t++) for (let l = 0; l < L; l++) {
      const nw = t === T - 1;
      s.put(t, l * 2, 0, "#4f8ef7");
      s.put(t, l * 2, 1, "#ff8a3d", nw ? { top: "#fff0a8" } : {});
    }
  },
  nexttok(s) {   // the next-token distribution: one tower per candidate word, the likeliest lit
    s.box(0, 0, 0, 13, 3, 1, "#2a2d36", () => ({ faces: { top: "#3a3e4a" } }));
    [7, 4, 3, 2, 1, 1].forEach((hh, n) => s.box(1 + n * 2, 1, 1, 1, 1, hh, n ? "#8ea3c7" : "#5fd176"));
  },
  neuron(s) {   // a neuron: three inputs on wires of different strength (weights) into a sum, then one output
    [["#4f8ef7", 0, 3], ["#7b61ff", 2, 1], ["#e76fa8", 4, 2]].forEach(([c, y, w]) => {
      s.put(0, y, 0, c);
      for (let x = 1; x < 6; x++) s.put(x, y, 0, w === 3 ? "#ffd23f" : w === 2 ? "#e5c75a" : "#a9a07a", { top: w === 3 ? "#fff0a8" : w === 2 ? "#f2e2a0" : "#d1caa8" });
    });
    s.box(6, 0, 0, 2, 5, 2, "#22b8a0", (i, j, k) => ({ faces: { top: (i + j) % 2 ? "#7fe0cc" : "#5fd1b8" } }));
    for (let x = 8; x < 11; x++) s.put(x, 2, 0, "#9aa3b5");
    s.box(11, 2, 0, 1, 1, 2, "#5fd176");
  },
  recur(s) {   // a recurrent net unrolled: the same cell at every step, inputs below-front, outputs on top, the state (yellow) carried through
    for (let t = 0; t < 4; t++) {
      const x = t * 4;
      s.box(x, 0, 0, 2, 2, 2, "#22b8a0", (i, j) => ({ faces: { top: (i + j) % 2 ? "#7fe0cc" : "#5fd1b8" } }));
      s.put(x, 3, 0, "#4f8ef7").put(x + 1, 3, 0, "#4f8ef7");
      s.put(x, 0, 2, "#5fd176");
    }
    for (let x = 0; x < 15; x++) if (x % 4 >= 2) s.put(x, 1, 1, "#ffd23f", { top: "#fff0a8" });
  },
  maze(s) {   // a grid world: walls, a few tried tiles, the agent (purple) and the flag (goal)
    const M = ["#######", "#..#..#", "#.##.##", "#.....#", "###.#.#", "#...#.#", "#######"];
    const tried = new Set(["2,5", "3,5", "3,4", "3,3", "4,3", "4,2", "4,1"]);
    for (let y = 0; y < 7; y++) for (let x = 0; x < 7; x++) {
      if (M[y][x] === "#") s.put(x, y, 0, "#3a3e4a", { top: "#4a4f5d" });
      else s.put(x, y, 0, "#d5dceb", { top: tried.has(`${x},${y}`) ? "#c3b8ff" : (x + y) % 2 ? "#eef1f7" : "#e3e8f1" });
    }
    s.put(1, 5, 1, "#7b61ff").put(1, 5, 2, "#7b61ff", { top: "#c3b8ff" });
    s.put(5, 1, 1, "#9aa3b5").put(5, 1, 2, "#9aa3b5").put(5, 1, 3, "#9aa3b5").put(5, 1, 4, "#5fd176", { top: "#b6f5c4" }).put(4, 1, 4, "#5fd176", { top: "#b6f5c4" });
  },
  judge(s) {   // two answers side by side; the preferred one wears a gold crown; the reward model (purple) in front
    s.box(0, 0, 0, 9, 5, 1, "#2a2d36", (i, j) => ({ faces: { top: (i + j) % 2 ? "#3a3e4a" : "#353944" } }));
    s.box(1, 1, 1, 2, 2, 3, "#8ea3c7");
    s.box(6, 1, 1, 2, 2, 4, "#ff8a3d");
    for (const [x, y] of [[6, 1], [7, 1], [6, 2], [7, 2]]) if ((x + y) % 2) s.put(x, y, 5, "#ffd23f", { top: "#fff0a8" });
    s.box(3, 3, 1, 3, 1, 1, "#7b61ff", (i) => ({ faces: { top: i === 1 ? "#5fd176" : "#a996ff" } }));
  },
  ssm(s) {   // a state space model: a fixed-size state rides along the tokens; tokens behind are folded in, not stored
    for (let x = 0; x < 14; x++) s.put(x, 0, 0, x < 8 ? "#5b6172" : ["#4f8ef7", "#22b8a0", "#ff8a3d", "#e76fa8"][x % 4], x < 8 ? { top: "#7a8194" } : {});
    s.box(7, -1, 1, 2, 2, 2, "#7b61ff", (i, j) => ({ faces: { top: (i + j) % 2 ? "#a996ff" : "#9c88ff" } }));
  },
  distill(s) {   // a big teacher, a small student, and soft targets stepping down between them
    s.box(0, 0, 0, 4, 4, 5, "#7b61ff", (i, j, k) => ({ faces: { top: (i + j) % 2 ? "#a996ff" : "#9c88ff" } }));
    [[4, 1, 4], [5, 1, 3], [6, 1, 2], [7, 1, 1]].forEach(([x, y, z]) => s.put(x, y + 1, z, "#ffd23f", { top: "#fff0a8" }));
    s.box(8, 1, 0, 2, 2, 2, "#22b8a0", (i, j) => ({ faces: { top: (i + j) % 2 ? "#7fe0cc" : "#5fd1b8" } }));
  },
  hourglass(s) {   // an autoencoder on its side: wide input, narrowing encoder, a 2-number code (yellow), widening decoder
    const P = [[8, "#4f8ef7"], [6, "#4f8ef7"], [4, "#4f8ef7"], [2, "#ffd23f"], [4, "#ff8a3d"], [6, "#ff8a3d"], [8, "#ff8a3d"]];
    P.forEach(([w, c], n) => { const o = (8 - w) / 2; s.box(n, 0, o, 1, 2, w, c, (i, j, k) => ({ faces: { left: shade(c, k % 2 ? 1.12 : 1) } })); });
  },
  patches(s) {   // a picture cut into four patches, and the same patches lifted into a row of tokens
    const px = (x, y) => (Math.hypot(x - 1.5, y - 1.5) < 1.2 ? "#ffd23f" : y >= 4 ? "#5fd176" : "#8fb8ff");
    for (let x = 0; x < 6; x++) for (let y = 0; y < 6; y++) s.put(x, y, 0, px(x, y), { top: shade(px(x, y), (Math.floor(x / 3) + Math.floor(y / 3)) % 2 ? 1.0 : 1.18) });
    [[0, 0], [3, 0], [0, 3], [3, 3]].forEach(([px0, py0], n) => {
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) s.put(9 + i, n * 4 + j - 2, 1, px(px0 + i, py0 + j));
    });
  },
  heatsink(s) {   // the heat has one way out: up through the lid into a finned sink
    s.box(0, 0, 0, 10, 8, 1, "#1e7a4c", () => ({ faces: { top: "#2f9a62" } }));
    s.box(2, 2, 1, 6, 4, 1, "#ff6b3d", () => ({ faces: { top: "#ffb08a" } }));
    s.box(1, 1, 2, 8, 6, 1, "#b7bfcc", () => ({ faces: { top: "#d9dee8" } }));
    for (let i = 0; i < 8; i += 2) s.box(1 + i, 1, 3, 1, 6, 4, "#a3acbb", () => ({ faces: { top: "#e3e7ee" } }));
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

// ---------------------------------------------------------------------------------------------
// Law cards: one law of physics per card, with its formula, a small picture, what it means, and where it bites
// ---------------------------------------------------------------------------------------------
const GLYPH = {
  ohm: `<path d="M4 18h8l3-7 5 14 5-14 5 14 5-14 3 7h8" class="lw-s"/>`,
  heat: `<path d="M8 18h10l2-5 3 10 3-10 3 10 2-5h10" class="lw-s"/><path d="M20 10c0-3 3-4 2-7M27 10c0-3 3-4 2-7" class="lw-a"/>`,
  charge: `<path d="M18 6v24M28 6v24M4 18h14M28 18h16" class="lw-s"/><circle cx="14" cy="10" r="2.2" class="lw-e"/><circle cx="14" cy="18" r="2.2" class="lw-e"/><circle cx="14" cy="26" r="2.2" class="lw-e"/>`,
  rc: `<path d="M4 30C12 30 14 8 26 7S40 6 44 6" class="lw-a"/><path d="M4 30h40M4 4v26" class="lw-s"/>`,
  node: `<circle cx="24" cy="18" r="3.5" class="lw-e"/><path d="M4 8l16 8M4 28l16-8M28 18h16" class="lw-s"/><path d="M38 14l6 4-6 4" class="lw-s"/>`,
  wire: `<rect x="4" y="12" width="40" height="12" rx="3" class="lw-s"/><path d="M10 18h28" class="lw-a"/>`,
  clock: `<path d="M4 26h6V10h8v16h8V10h8v16h10" class="lw-s"/>`,
  light: `<path d="M4 18c3-8 6-8 9 0s6 8 9 0 6-8 9 0 6 8 9 0" class="lw-a"/><path d="M42 14l3 4-3 4" class="lw-s"/>`,
  boltz: `<path d="M4 6c8 0 10 24 40 24" class="lw-a"/><path d="M4 30h40M4 4v26" class="lw-s"/>`,
  wafer: `<circle cx="24" cy="18" r="14" class="lw-s"/><path d="M14 12h20M12 18h24M14 24h20M18 6v24M24 4v28M30 6v24" class="lw-t"/><circle cx="29" cy="21" r="2.4" class="lw-x"/>`,
};
/** cards: [{ name, formula, glyph, html, where: [[chapterId, label]] }] */
export function lawCards(cards, { whereLabel = "bites in" } = {}) {
  return h("div.laws", cards.map((c) => h("article.law",
    h("div.law-top", h("div.law-g", { html: `<svg viewBox="0 0 48 36" aria-hidden="true">${GLYPH[c.glyph] || ""}</svg>` }), h("div", h("div.law-n", c.name), h("div.law-f", c.formula))),
    h("div.law-b", { html: c.html }),
    c.where?.length ? h("div.law-w", h("span", whereLabel), c.where.map(([id, label]) => h("a", { href: `/learn?c=${id}` }, label))) : null)));
}
