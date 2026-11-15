// Tensor walkthroughs: step through a few lines of real code and watch the tensors as stacks of
// pixel cubes. Every cube is one number, coloured by its value (blue negative, orange positive,
// purple for probabilities). Each step shows the operation as an equation of blocks, highlights the
// line of code that does it, and cycles through exactly which cubes combine into which. Hover any cube
// to read its value and the arithmetic behind it.
import { h } from "./core.js";
import { Scene } from "./voxel.js";
import { highlight } from "./widgets.js";

// ------------------------------------------------------------------------------------------------
// small maths, seeded so every reader sees the same numbers
// ------------------------------------------------------------------------------------------------
export function rng(seed = 1) {
  let a = seed >>> 0;
  const u = () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  u.n = () => { let x = 0; for (let i = 0; i < 6; i++) x += u(); return (x - 3) / Math.SQRT2; };   // roughly N(0, 1)
  return u;
}
export const M = {
  rand: (r, c, R, s = 1) => Array.from({ length: r }, () => Array.from({ length: c }, () => +(R.n() * s).toFixed(2))),
  mm: (A, B) => A.map((row) => B[0].map((_, j) => row.reduce((s, a, k) => s + a * B[k][j], 0))),
  T: (A) => A[0].map((_, j) => A.map((r) => r[j])),
  map: (A, f) => A.map((r, i) => r.map((v, j) => f(v, i, j))),
  add: (A, B) => A.map((r, i) => r.map((v, j) => v + (Array.isArray(B[0]) ? B[i][j] : B[j]))),
  softmax: (A) => A.map((r) => { const m = Math.max(...r.filter(Number.isFinite)); const e = r.map((v) => (Number.isFinite(v) ? Math.exp(v - m) : 0)); const s = e.reduce((a, b) => a + b, 0); return e.map((v) => v / s); }),
  round: (A, d = 2) => A.map((r) => r.map((v) => (Number.isFinite(v) ? +v.toFixed(d) : v))),
};
export const fmt = (v, d = 2) => (v === -Infinity ? "−∞" : v === Infinity ? "∞" : (v < 0 && Math.abs(v) >= 0.5 * 10 ** -d ? "−" : "") + Math.abs(v).toFixed(d));

// ------------------------------------------------------------------------------------------------
// colour
// ------------------------------------------------------------------------------------------------
const hex = (c) => [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16));
const toHex = (a) => `#${a.map((v) => Math.round(Math.max(0, Math.min(255, v))).toString(16).padStart(2, "0")).join("")}`;
const mix = (a, b, t) => { const A = hex(a), B = hex(b); return toHex(A.map((v, i) => v + (B[i] - v) * t)); };
const NEG = "#2f6fe0", POS = "#f26a2e", MID = "#e4e0ee", PROB0 = "#ece7ff", PROB1 = "#5b3fd6", DEAD = "#2b2933", DIM = "#d4d1de", GLOW = "#ffd84a";
const IDS = ["#4f8ef7", "#ff7a45", "#22b8a0", "#f2b705", "#e86aa6", "#7b61ff", "#5cb85c", "#9aa3b5"];
function colorOf(t, v, sc) {
  if (t.colors) return t.colors(v);
  if (!Number.isFinite(v)) return DEAD;
  if (t.kind === "ids") return IDS[Math.abs(Math.round(v)) % IDS.length];
  if (t.kind === "prob") return mix(PROB0, PROB1, Math.min(1, Math.max(0, v / (sc || 1))) ** 0.7);
  if (t.kind === "solid") return t.color || "#9aa3b5";
  const x = Math.max(-1, Math.min(1, v / (sc || 1)));
  return x < 0 ? mix(MID, NEG, (-x) ** 0.6) : mix(MID, POS, x ** 0.6);   // a gentle curve, so middling values still read
}
const maxAbs = (layers) => Math.max(1e-9, ...layers.flat(2).filter(Number.isFinite).map(Math.abs));

/** One tensor as voxels. 2D data is one slab (columns run right, rows run toward you);
 *  3D data [layer][row][col] stacks slabs upward, so heads and channels read as layers. */
function tensorSVG(t, hl) {
  const s = new Scene(), D = t.data, is3 = Array.isArray(D[0][0]), layers = is3 ? D : [D];
  const sc = t.scale ?? maxAbs(layers), gap = t.gap ?? 3, focus = hl && hl.size;
  layers.forEach((Mx, l) => Mx.forEach((row, i) => row.forEach((v, j) => {
    const key = is3 ? `${l},${i},${j}` : `${i},${j}`, on = focus && hl.has(key);
    let c = colorOf(t, v, sc);
    if (focus && !on) c = mix(c, DIM, 0.42);
    const hgt = t.bars && Number.isFinite(v) ? 1 + Math.round(Math.min(1, Math.abs(v) / sc) * (t.bars === true ? 3 : t.bars)) : 1;
    // stacked layers step back as they go up (a staircase), so the front of every slab stays visible
    const z0 = (is3 ? l * gap : 0) + (on ? 1 : 0), y0 = is3 ? -l * (t.stair ?? Math.ceil(Mx.length * 0.55)) : 0;
    for (let k = 0; k < hgt; k++) {
      const top = k === hgt - 1;
      const tc = on ? mix(c, GLOW, 0.6) : (i + j) % 2 ? mix(c, "#ffffff", 0.1) : mix(c, "#ffffff", 0.22);
      s.put(j, y0 + i, z0 + k, c, { top: top ? tc : undefined, hit: top ? key : undefined });
    }
  })));
  const svg = s.svg({ cls: "tw-vox", label: `${t.label || ""} tensor` });
  const [, W, H] = /viewBox="0 0 (\d+) (\d+)"/.exec(svg) || [0, 60, 40];
  const k = t.px ?? Math.max(2, Math.min(5, Math.round((is3 ? 250 : 190) / W)));   // aim for about 190px wide (stacks a bit more)
  return svg.replace("<svg ", `<svg width="${W * k}" height="${H * k}" `);
}
const shape = (D) => (Array.isArray(D[0][0]) ? [D.length, D[0].length, D[0][0].length] : [D.length, D[0].length]);

// ------------------------------------------------------------------------------------------------
// operations: for step k, which cubes light up and the arithmetic behind the result
// ------------------------------------------------------------------------------------------------
const rowKeys = (i, n, l) => Array.from({ length: n }, (_, j) => (l == null ? `${i},${j}` : `${l},${i},${j}`));
const colKeys = (j, n, l) => Array.from({ length: n }, (_, i) => (l == null ? `${i},${j}` : `${l},${i},${j}`));
const terms = (pairs, d = 2) => {
  const shown = pairs.slice(0, 4).map(([a, b]) => `${fmt(a, d)}×${fmt(b, d)}`).join(" + ");
  return pairs.length > 4 ? `${shown} + … <span class="tw-dim">(${pairs.length} products)</span>` : shown;
};

export const op = {
  /** out = A @ B (times scale): row i of A meets column j of B in cell (i, j). */
  matmul(T, a, b, out, { scale = 1, scaleLabel = "" } = {}) {
    const A = T[a].data, B = T[b].data, O = T[out].data, C = O[0].length, K = A[0].length;
    return {
      n: O.length * C,
      at(k) {
        const i = Math.floor(k / C), j = k % C, pairs = A[i].map((x, q) => [x, B[q][j]]);
        const dot = pairs.reduce((s, [x, y]) => s + x * y, 0);
        return {
          hl: { [a]: rowKeys(i, K), [b]: colKeys(j, K), [out]: [`${i},${j}`] },
          read: `<b>${T[out].label}[${i},${j}]</b> = row ${i} of ${T[a].label} · column ${j} of ${T[b].label}${scaleLabel ? ` ${scaleLabel}` : ""}<br>= ${scaleLabel ? "(" : ""}${terms(pairs)}${scaleLabel ? `) ${scaleLabel}` : ""} = <b>${fmt(dot * scale)}</b>`,
        };
      },
      find(name, key) { const [i, j] = key.split(",").map(Number); return name === out ? i * C + j : null; },
    };
  },
  /** Something done to each row on its own (softmax, layer norm, a mask): rows light up together. */
  rows(names, n, read, width = {}) {
    return {
      n,
      at(i) { const hl = {}; for (const nm of names) hl[nm] = rowKeys(i, width[nm] ?? 64); return { hl, read: read(i) }; },
      find(name, key) { return names.includes(name) ? +key.split(",")[0] : null; },
    };
  },
  /** Something done cell by cell (add, ReLU, scale): the same cell lights up in every tensor. */
  cells(names, R, C, read) {
    return {
      n: R * C,
      at(k) { const i = Math.floor(k / C), j = k % C, hl = {}; for (const nm of names) hl[nm] = [`${i},${j}`]; return { hl, read: read(i, j) }; },
      find(name, key) { if (!names.includes(name)) return null; const [i, j] = key.split(",").map(Number); return i * C + j; },
    };
  },
  /** out[t] = table[ids[t]]: a token id picks one row of the table. */
  gather(T, ids, table, out, words = []) {
    const I = T[ids].data[0], D = T[table].data[0].length;
    return {
      n: I.length,
      at(t) { return { hl: { [ids]: [`0,${t}`], [table]: rowKeys(I[t], D), [out]: rowKeys(t, D) }, read: `token ${t}${words[t] ? ` (“${words[t]}”)` : ""} has id <b>${I[t]}</b>, so row ${t} of ${T[out].label} is a copy of row ${I[t]} of ${T[table].label}: [${T[table].data[I[t]].slice(0, 4).map((v) => fmt(v)).join(", ")}${D > 4 ? ", …" : ""}]` }; },
      find(name, key) { const [a, b] = key.split(",").map(Number); return name === ids ? b : name === out ? a : null; },
    };
  },
  /** A 3×3 kernel over every channel: a window of the input meets the whole kernel in one output cell. */
  conv(T, inp, ker, out) {
    const X = T[inp].data, Kr = T[ker].data, O = T[out].data, C = O[0].length, ch = X.length, kh = Kr[0].length;
    return {
      n: O.length * C,
      at(k) {
        const i = Math.floor(k / C), j = k % C, win = [], kk = [], pairs = [];
        for (let c = 0; c < ch; c++) for (let di = 0; di < kh; di++) for (let dj = 0; dj < kh; dj++) { win.push(`${c},${i + di},${j + dj}`); kk.push(`${c},${di},${dj}`); pairs.push([X[c][i + di][j + dj], Kr[c][di][dj]]); }
        return { hl: { [inp]: win, [ker]: kk, [out]: [`${i},${j}`] }, read: `<b>${T[out].label}[${i},${j}]</b> = the ${kh}×${kh} window at (${i}, ${j}) in all ${ch} channels, times the kernel, summed<br>= ${terms(pairs)} = <b>${fmt(O[i][j])}</b>` };
      },
      find(name, key) { if (name !== out) return null; const [i, j] = key.split(",").map(Number); return i * C + j; },
    };
  },
  /** Fully custom: n steps, at(k) => {hl, read}. */
  custom(n, at, find) { return { n, at, find }; },
};

// ------------------------------------------------------------------------------------------------
// the widget
// ------------------------------------------------------------------------------------------------
/**
 * tensorWalk({ title, lede, code, tensors: {name: {data, label, dims, kind, bars, scale}}, steps: [{title, text, lines, show, op, read}] })
 * show: an equation such as ["Q", "@", "Kt", "=", "S"]; names are tensors, anything else is an operator.
 */
export function tensorWalk({ title = "Inside the code, step by step", lede, code, tensors, steps }) {
  let s = 0, k = 0, playing = !matchMedia("(prefers-reduced-motion: reduce)").matches && document.documentElement.dataset.motion !== "reduce", hovering = false, timer = 0;
  const stage = h("div.tw-stage", { role: "group", "aria-label": "Tensors in this step" });
  const hover = h("div.tw-hover", " ");
  const read = h("div.tw-read", { "aria-live": "polite" });
  const text = h("div.tw-text");
  const stitle = h("h3.tw-st");
  const count = h("span.tw-count");
  const lines = highlight(code.replace(/\n+$/, "")).split("\n");
  const pre = h("pre.code.tw-code", { html: lines.join("") });
  const dots = h("div.tw-dots", steps.map((st, i) => h("button", { type: "button", "aria-label": `Step ${i + 1}: ${st.title}`, onclick: () => go(i) })));
  const back = h("button.btn.small", { type: "button", onclick: () => go(s - 1) }, "← Back");
  const next = h("button.btn.small.primary", { type: "button", onclick: () => go(s + 1) }, "Next →");
  const play = h("button.tw-play", { type: "button", onclick: () => { playing = !playing; paint(); loop(); } });

  function paint() {
    const st = steps[s], o = st.op, at = o ? o.at(Math.min(k, o.n - 1)) : null;
    const hl = {}; if (at) for (const [n, keys] of Object.entries(at.hl)) hl[n] = new Set(keys);
    stage.replaceChildren(...st.show.map((item) => {
      const t = tensors[item];
      if (!t) return h("span.tw-op", item);
      const sh = shape(t.data);
      return h("figure.tw-t", { "data-t": item, class: hl[item] ? "tw-t on" : "tw-t" },
        h("div.tw-svg", { html: tensorSVG(t, hl[item]) }),
        h("figcaption", h("b", t.label || item), h("span", sh.join(" × ")), t.dims ? h("small", `(${t.dims})`) : null));
    }));
    read.innerHTML = at ? at.read : st.read || "";
    play.hidden = !o || o.n < 2;
    play.textContent = playing ? "❚❚ Pause" : "▶ Play";
    play.setAttribute("aria-pressed", String(playing));
  }
  function go(i) {
    s = Math.max(0, Math.min(steps.length - 1, i)); k = 0;
    const st = steps[s];
    stitle.textContent = st.title;
    text.innerHTML = st.text;
    count.textContent = `Step ${s + 1} of ${steps.length}`;
    back.disabled = s === 0; next.disabled = s === steps.length - 1;
    [...dots.children].forEach((d, j) => { d.classList.toggle("on", j === s); d.classList.toggle("past", j < s); });
    const on = new Set(st.lines || []);
    [...pre.children].forEach((ln, j) => ln.classList.toggle("on", on.has(j + 1)));
    pre.classList.toggle("has-on", on.size > 0);
    const first = pre.children[(st.lines || [1])[0] - 1];
    if (first) pre.scrollTo({ top: Math.max(0, first.offsetTop - 28), behavior: "smooth" });
    paint(); loop();
  }
  function loop() {
    clearTimeout(timer);
    const o = steps[s].op;
    if (!playing || hovering || !o || o.n < 2 || !el.isConnected) return;
    timer = setTimeout(() => { k = (k + 1) % o.n; paint(); loop(); }, steps[s].speed || 1100);
  }
  // hover a cube: its value, and if it's an output, the arithmetic that made it
  stage.addEventListener("pointerover", (e) => {
    const p = e.target.closest?.("[data-h]"), card = e.target.closest?.("[data-t]");
    if (!p || !card) return;
    const name = card.dataset.t, t = tensors[name], key = p.dataset.h, idx = key.split(",").map(Number);
    const v = idx.reduce((a, i) => a[i], t.data);
    const tag = [t.rowNames?.[idx[idx.length - 2]], t.names?.[idx[idx.length - 1]]].filter(Boolean).join(" → ");
    hover.innerHTML = `<b>${t.label || name}[${key}]</b> = ${typeof v === "number" ? fmt(v, t.kind === "ids" ? 0 : 3) : v}${tag ? ` · ${tag}` : ""}`;
    const o = steps[s].op, f = o?.find?.(name, key);
    if (f != null) { hovering = true; clearTimeout(timer); if (f !== k) { k = f; paint(); } }
  });
  stage.addEventListener("pointerleave", () => { hovering = false; hover.innerHTML = " "; loop(); });
  stage.addEventListener("click", (e) => { if (e.target.closest?.("[data-h]") && steps[s].op) { playing = false; paint(); } });

  const el = h("div.tw",
    h("div.tw-top", h("div.tw-head", count, stitle), h("div.tw-nav", play, back, next)),
    stage, hover,
    h("div.tw-body",
      h("div.tw-codebox", h("div.tw-cl", "The code"), pre),
      h("div.tw-explain", text, read)),
    h("div.tw-foot", dots, h("div.tw-key", h("i.n"), "negative", h("i.p"), "positive", h("i.pr"), "probability", h("i.d"), "masked")));
  queueMicrotask(() => go(0));
  // pause when scrolled away
  if ("IntersectionObserver" in window) new IntersectionObserver((es) => { for (const e of es) { if (e.isIntersecting) loop(); else clearTimeout(timer); } }).observe(el);
  el.addEventListener("keydown", (e) => { if (e.target.closest("button") && (e.key === "ArrowRight" || e.key === "ArrowLeft")) { go(s + (e.key === "ArrowRight" ? 1 : -1)); e.preventDefault(); } });
  return h("section.block.tw-sec", h("h2", title), lede ? h("p.muted", lede) : null, el);
}
