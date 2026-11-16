// Interactive quiz questions: you answer by doing, not by picking a letter.
//   estimate  drag a log-scale slider to guess a number; right if within the tolerance
//   order     tap the steps in the order they happen
//   bits      flip switches until the bits spell a number
//   neuron    tilt and shift one neuron's line until it sorts the dots (or say it can't be done)
//   mask      click the attention cells a model is allowed to look at
//   kernel    paint a 3 × 3 filter and see what it finds in two test images
//   gates     pick the gate behind each output until the truth table matches
// Each type: view(q, value, set, revealed) → element; right(q, value); said(q, value); answer(q).
import { h } from "./core.js";

const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
const sup = (n) => String(n).split("").map((c) => SUP[c] ?? c).join("");
const sig = (v, n = 2) => { const s = Number(v.toPrecision(n)); return s >= 1000 ? s.toLocaleString("en-US") : String(s); };
/** 80000000000 → "80 billion"; 3.1e23 → "3.1 × 10²³"; 0.0029 → "0.0029" */
export function fmtNum(v) {
  const a = Math.abs(v);
  if (a >= 1e15 || (a > 0 && a < 1e-3)) { const e = Math.floor(Math.log10(a)), m = v / 10 ** e; return `${sig(m)} × 10${sup(e)}`; }
  for (const [d, w] of [[1e12, "trillion"], [1e9, "billion"], [1e6, "million"]]) if (a >= d) return `${sig(v / d)} ${w}`;
  return sig(v);
}
const withUnit = (q, v) => `${fmtNum(v)}${q.unit ? " " + q.unit : ""}`;

// seeded shuffle, so a question's pieces sit in the same scrambled order every time you see it
function seeded(str) { let x = 2166136261; for (const c of str) x = Math.imul(x ^ c.charCodeAt(0), 16777619); return () => { x = Math.imul(x ^ (x >>> 15), 2246822507); x = Math.imul(x ^ (x >>> 13), 3266489909); return ((x ^= x >>> 16) >>> 0) / 4294967296; }; }
function scramble(n, key) {
  const r = seeded(key), a = [...Array(n).keys()];
  for (let t = 0; t < 8; t++) {
    for (let i = n - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
    if (a.some((v, i) => v !== i)) break;
  }
  return a;
}
const hint = (text) => h("p.qp-hint", text);
const fill = (el, ...kids) => el.replaceChildren(...kids.flat().filter((k) => k != null && k !== false));

// ---------------------------------------------------------------------------------------------
const estimate = {
  label: "Guess the number",
  view(q, value, set, revealed) {
    const [lo, hi] = q.range, W = 640, x0 = 24, x1 = W - 24, X = (v) => x0 + ((Math.log10(v) - lo) / (hi - lo)) * (x1 - x0);
    let g = value ?? 10 ** ((lo + hi) / 2);
    const read = h("div.qp-read"), svg = h("div.qp-svg");
    const tol = q.tol || 3;
    function paint() {
      read.replaceChildren(h("span.muted", value == null && !revealed ? "drag to guess" : "your guess"), h("b", withUnit(q, g)));
      const ticks = [];
      const step = hi - lo > 12 ? 3 : hi - lo > 7 ? 2 : 1;
      for (let e = Math.ceil(lo); e <= hi; e++) {
        ticks.push(`<line x1="${X(10 ** e)}" x2="${X(10 ** e)}" y1="66" y2="${e % step ? 71 : 74}" class="qp-tk"/>`);
        if (e % step === 0) ticks.push(`<text x="${X(10 ** e)}" y="90" class="qp-tl">10${sup(e)}</text>`);
      }
      const refs = (q.refs || []).map(([v, name], k) => {
        const y = k % 2 ? 34 : 18;
        return `<line x1="${X(v)}" x2="${X(v)}" y1="${y + 4}" y2="66" class="qp-ref"/><circle cx="${X(v)}" cy="66" r="3" class="qp-refd"/><text x="${Math.min(x1 - 4, Math.max(x0 + 4, X(v)))}" y="${y}" class="qp-rl" text-anchor="${X(v) < x0 + 60 ? "start" : X(v) > x1 - 60 ? "end" : "middle"}">${name}</text>`;
      }).join("");
      let rev = "";
      if (revealed) {
        const a = q.answer, ok = estimate.right(q, value);
        rev = `<rect x="${X(Math.max(10 ** lo, a / tol))}" y="58" width="${Math.max(2, X(Math.min(10 ** hi, a * tol)) - X(Math.max(10 ** lo, a / tol)))}" height="16" rx="8" class="qp-band"/>
          <line x1="${X(a)}" x2="${X(a)}" y1="50" y2="80" class="qp-ans"/><text x="${Math.min(x1, Math.max(x0, X(a)))}" y="108" class="qp-al" text-anchor="${X(a) < x0 + 80 ? "start" : X(a) > x1 - 80 ? "end" : "middle"}">actual: ${withUnit(q, a)}</text>`;
        if (!ok && value != null) rev += `<text x="${X(g)}" y="48" class="qp-yl" text-anchor="middle">you</text>`;
      }
      const top = q.refs?.length ? 0 : 34;
      svg.innerHTML = `<svg viewBox="0 ${top} ${W} ${116 - top}" role="img" aria-label="A log scale from 10${sup(lo)} to 10${sup(hi)}">
        <line x1="${x0}" x2="${x1}" y1="66" y2="66" class="qp-axis"/>${ticks.join("")}${refs}${rev}
        <circle cx="${X(g)}" cy="66" r="9" class="qp-you${revealed ? (estimate.right(q, value) ? " ok" : " no") : ""}"/></svg>`;
    }
    const input = h("input.qp-range", { type: "range", min: lo, max: hi, step: 0.01, value: Math.log10(g), disabled: revealed || null, "aria-label": "Your guess, on a log scale",
      oninput: (e) => { g = 10 ** +e.target.value; value = g; set(g); paint(); } });
    paint();
    return h("div.qp.qp-est", read, svg, input, revealed ? null : hint(`Each tick is 10× the last. Within ${tol}× of the real value counts as right.`));
  },
  right: (q, v) => v != null && Math.abs(Math.log10(v / q.answer)) <= Math.log10(q.tol || 3) + 1e-9,
  said: (q, v) => (v == null ? "no guess" : withUnit(q, v)),
  answer: (q) => `${withUnit(q, q.answer)} (anything within ${q.tol || 3}× counts)`,
};

// ---------------------------------------------------------------------------------------------
const order = {
  label: "Put in order",
  view(q, value, set, revealed) {
    const perm = scramble(q.items.length, q.id || q.q);
    let seq = value ? [...value] : [];
    const el = h("div.qp.qp-ord");
    function paint() {
      const slots = q.items.map((_, k) => {
        const it = seq[k];
        const cls = revealed ? (it === k ? ".ok" : ".no") : it != null ? ".on" : "";
        return h(`button.qp-slot${cls}`, { type: "button", disabled: revealed || it == null || null, title: it != null ? "Tap to take it back" : null,
          onclick: () => { seq = seq.slice(0, k); set(null); paint(); } },
          h("span.qp-n", String(k + 1)), h("span", it != null ? q.items[it] : "…"));
      });
      const pool = perm.filter((k) => !seq.includes(k)).map((k) => h("button.qp-chip", { type: "button", disabled: revealed || null,
        onclick: () => { seq.push(k); set(seq.length === q.items.length ? [...seq] : null); paint(); } }, q.items[k]));
      fill(el, 
        h("div.qp-l", q.first ? `Your order · ${q.first}` : "Your order"), h("div.qp-slots", slots),
        pool.length ? h("div", h("div.qp-l", seq.length ? "Tap the next one" : "Tap the one that comes first"), h("div.qp-pool", pool)) : null,
        revealed && !order.right(q, value) ? h("div.qp-truth", h("div.qp-l", "The right order"), h("ol", q.items.map((t) => h("li", t)))) : null,
        revealed || pool.length ? null : hint("Tap a placed step to take it and everything after it back."));
    }
    paint();
    return el;
  },
  right: (q, v) => Array.isArray(v) && v.length === q.items.length && v.every((x, i) => x === i),
  said: (q, v) => (v ? v.map((k) => q.items[k]).join(" → ") : "not finished"),
  answer: (q) => q.items.join(" → "),
};

// ---------------------------------------------------------------------------------------------
const bitVal = (q, b) => b.reduce((s, x, i) => s + x * (i === 0 && q.signed ? -1 : 1) * 2 ** (b.length - 1 - i), 0);
const bits = {
  label: "Build it",
  view(q, value, set, revealed) {
    const n = q.n || 8;
    let b = value ? [...value] : Array(n).fill(0);
    const el = h("div.qp.qp-bits");
    function paint() {
      const v = bitVal(q, b), ok = v === q.target;
      fill(el, 
        h("div.qp-bitrow", b.map((x, i) => h("div.qp-bitcol",
          h("small", `${i === 0 && q.signed ? "−" : ""}${2 ** (n - 1 - i)}`),
          h(`button.qp-bit${x ? ".on" : ""}`, { type: "button", disabled: revealed || null, "aria-pressed": x ? "true" : "false", "aria-label": `bit worth ${2 ** (n - 1 - i)}`,
            onclick: () => { b[i] ^= 1; set([...b]); paint(); } }, String(x))))),
        h(`div.qp-sum${revealed ? (ok ? ".ok" : ".no") : ""}`,
          h("span", b.map((x, i) => (x ? `${i === 0 && q.signed ? "−" : ""}${2 ** (n - 1 - i)}` : null)).filter(Boolean).join(" + ") || "0"),
          h("b", ` = ${v}`), h("span.muted", `  ·  target ${q.target}`)),
        revealed && !ok ? h("p.qp-hint", `One answer: ${bits.answer(q)}`) : null);
    }
    paint();
    return el;
  },
  right: (q, v) => Array.isArray(v) && bitVal(q, v) === q.target,
  said: (q, v) => (v ? `${v.join("")} = ${bitVal(q, v)}` : "no bits set"),
  answer: (q) => { const n = q.n || 8, t = q.signed && q.target < 0 ? q.target + 2 ** n : q.target; return t.toString(2).padStart(n, "0"); },
};

// ---------------------------------------------------------------------------------------------
const fires = (w, p) => w.w1 * p[0] + w.w2 * p[1] + w.b > 0 ? 1 : 0;
const neuron = {
  label: "Build it",
  view(q, value, set, revealed) {
    let w = value ? { ...value } : { w1: 1, w2: -1, b: 0, no: false };
    const plot = h("div.qp-svg.qp-plot"), stat = h("div.qp-sum"), ctrls = h("div.qp-ctrls");
    const S = 240, P = (t) => 20 + ((t + 0.5) / 2) * (S - 40);   // data -0.5..1.5 → px
    function paint() {
      const good = q.points.filter((p) => fires(w, p) === p[2]).length;
      // the half of the box where it fires: clip the square by w·x + b > 0
      const f = (p) => w.w1 * p[0] + w.w2 * p[1] + w.b, sq = [[-0.5, -0.5], [1.5, -0.5], [1.5, 1.5], [-0.5, 1.5]], poly = [];
      sq.forEach((a, k) => { const c = sq[(k + 1) % 4], fa = f(a), fc = f(c);
        if (fa > 0) poly.push(a);
        if ((fa > 0) !== (fc > 0)) { const t = fa / (fa - fc); poly.push([a[0] + t * (c[0] - a[0]), a[1] + t * (c[1] - a[1])]); } });
      const cells = poly.length > 2 ? `<polygon points="${poly.map((p) => `${P(p[0])},${S - P(p[1])}`).join(" ")}" class="qp-on"/>` : "";
      // the boundary line w1 x + w2 y + b = 0, clipped to the box
      let line = "";
      const ends = [];
      for (const x of [-0.5, 1.5]) if (Math.abs(w.w2) > 1e-6) { const y = -(w.w1 * x + w.b) / w.w2; if (y >= -0.5 && y <= 1.5) ends.push([x, y]); }
      for (const y of [-0.5, 1.5]) if (Math.abs(w.w1) > 1e-6) { const x = -(w.w2 * y + w.b) / w.w1; if (x >= -0.5 && x <= 1.5) ends.push([x, y]); }
      if (ends.length >= 2) line = `<line x1="${P(ends[0][0])}" y1="${S - P(ends[0][1])}" x2="${P(ends[1][0])}" y2="${S - P(ends[1][1])}" class="qp-bound"/>`;
      const pts = q.points.map((p) => { const bad = fires(w, p) !== p[2];
        return `<circle cx="${P(p[0])}" cy="${S - P(p[1])}" r="11" class="${p[2] ? "qp-p1" : "qp-p0"}${bad && !w.no ? " qp-miss" : ""}"/><text x="${P(p[0])}" y="${S - P(p[1]) + 4}" class="qp-pt">${p[2]}</text>`; }).join("");
      plot.innerHTML = `<svg viewBox="0 0 ${S} ${S}" role="img" aria-label="Four points and the region where the neuron fires"><rect x="20" y="20" width="${S - 40}" height="${S - 40}" class="qp-box"/>${cells}${line}${pts}
        <text x="${P(0)}" y="${S - 4}" class="qp-ax">x₁ = 0</text><text x="${P(1)}" y="${S - 4}" class="qp-ax">x₁ = 1</text></svg>`;
      stat.className = `qp-sum${revealed ? (neuron.right(q, w) ? " ok" : " no") : ""}`;
      stat.replaceChildren(w.no ? h("b", "You say: no single neuron can do this") : h("span", h("b", `${good} of ${q.points.length}`), " dots on the right side", h("span.muted", `  ·  fires when ${w.w1.toFixed(1)}·x₁ ${w.w2 < 0 ? "−" : "+"} ${Math.abs(w.w2).toFixed(1)}·x₂ ${w.b < 0 ? "−" : "+"} ${Math.abs(w.b).toFixed(1)} > 0`.replace(/-/g, "−"))));
    }
    const sl = (key, label) => h("label.qp-sl", h("span", label), h("input", { type: "range", min: -3, max: 3, step: 0.1, value: w[key], disabled: revealed || null, "aria-label": label,
      oninput: (e) => { w[key] = +e.target.value; w.no = false; noBtn.setAttribute("aria-pressed", "false"); set({ ...w }); paint(); } }));
    const noBtn = h("button.btn.small.qp-no", { type: "button", "aria-pressed": w.no ? "true" : "false", disabled: revealed || null,
      onclick: () => { w.no = !w.no; noBtn.setAttribute("aria-pressed", String(w.no)); set({ ...w }); paint(); } }, "It can't be done with one neuron");
    ctrls.append(sl("w1", "weight on x₁"), sl("w2", "weight on x₂"), sl("b", "bias"), noBtn);
    paint();
    return h("div.qp.qp-neu", h("div.qp-split", plot, h("div", ctrls, h("div.qp-key", h("span.qp-k1", "1"), " should fire  ", h("span.qp-k0", "0"), " should not  ", h("span.qp-ks"), " where it fires"))), stat,
      revealed ? null : hint("The neuron fires on one side of a straight line. Weights tilt the line; the bias slides it."));
  },
  right(q, v) { if (!v) return false; if (q.impossible) return !!v.no; return !v.no && q.points.every((p) => fires(v, p) === p[2]); },
  said: (q, v) => (!v ? "not set" : v.no ? "it can't be done" : `w = (${v.w1.toFixed(1)}, ${v.w2.toFixed(1)}), b = ${v.b.toFixed(1)}`),
  answer: (q) => q.impossible ? "It can't be done: no straight line puts the two 1s on one side and the two 0s on the other" : q.example,
};

// ---------------------------------------------------------------------------------------------
const maskRule = (q, i, j) => q.rule === "causal" ? j <= i : q.rule === "window" ? j <= i && j > i - (q.w || 3) : q.rule === "packed" ? j <= i && (i < q.split) === (j < q.split) : true;
const mask = {
  label: "Build it",
  view(q, value, set, revealed) {
    const n = q.tokens.length;
    let m = value ? value.map((r) => [...r]) : Array.from({ length: n }, () => Array(n).fill(0));
    const el = h("div.qp.qp-mask");
    function paint() {
      const grid = h("div.qp-grid", { style: { "--n": String(n + 1) } }, h("span"), q.tokens.map((t) => h("span.qp-ch", t)));
      for (let i = 0; i < n; i++) {
        grid.append(h("span.qp-rh", q.tokens[i]));
        for (let j = 0; j < n; j++) {
          const want = maskRule(q, i, j) ? 1 : 0, wrong = revealed && m[i][j] !== want;
          grid.append(h(`button.qp-cell${m[i][j] ? ".on" : ""}${wrong ? ".no" : ""}`, { type: "button", disabled: revealed || null, "aria-label": `${q.tokens[i]} looks at ${q.tokens[j]}`, "aria-pressed": m[i][j] ? "true" : "false",
            onclick: () => { m[i][j] ^= 1; set(m.map((r) => [...r])); paint(); } }));
        }
      }
      const on = m.flat().reduce((a, b) => a + b, 0);
      fill(el, h("div.qp-l", "rows: the token doing the looking · columns: the token it may look at"), grid,
        h(`div.qp-sum${revealed ? (mask.right(q, m) ? ".ok" : ".no") : ""}`, h("b", `${on} of ${n * n}`), " cells open", revealed && !mask.right(q, m) ? h("span.muted", " · red cells are wrong") : null));
    }
    paint();
    return el;
  },
  right: (q, v) => Array.isArray(v) && v.every((r, i) => r.every((x, j) => x === (maskRule(q, i, j) ? 1 : 0))),
  said: (q, v) => (v ? `${v.flat().reduce((a, b) => a + b, 0)} cells open` : "nothing clicked"),
  answer: (q) => q.answerText,
};

// ---------------------------------------------------------------------------------------------
const IMG = 8;
const testImg = (kind) => Array.from({ length: IMG }, (_, r) => Array.from({ length: IMG }, (_, c) => (kind === "v" ? (c >= IMG / 2 ? 1 : 0) : kind === "h" ? (r >= IMG / 2 ? 1 : 0) : 1)));
function conv(img, k) {
  const out = [];
  for (let r = 0; r < IMG - 2; r++) { const row = []; for (let c = 0; c < IMG - 2; c++) { let s = 0; for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) s += img[r + a][c + b] * k[a * 3 + b]; row.push(s); } out.push(row); }
  return out;
}
const peak = (o) => Math.max(...o.flat().map(Math.abs));
const kernel = {
  label: "Build it",
  view(q, value, set, revealed) {
    let k = value ? [...value] : Array(9).fill(0);
    const el = h("div.qp.qp-kern");
    const heat = (o, label) => {
      const mx = Math.max(1, ...o.flat().map(Math.abs));
      return h("figure.qp-heat", h("div.qp-hg", { style: { "--n": String(o.length) } }, o.flat().map((v) => h("i", { style: { "--a": String(Math.abs(v) / mx), "--hc": v >= 0 ? "var(--c4)" : "var(--c1)" }, title: String(v) }))), h("figcaption", label, h("b", ` peak ${peak(o)}`)));
    };
    const img = (m, label) => h("figure.qp-heat", h("div.qp-hg.img", { style: { "--n": String(IMG) } }, m.flat().map((v) => h("i", { style: { "--a": String(v ? 0.95 : 0.12), "--hc": "var(--ink-2)" } }))), h("figcaption", label));
    function paint() {
      const V = conv(testImg("v"), k), H = conv(testImg("h"), k), F = conv(testImg("f"), k);
      fill(el, 
        h("div.qp-split",
          h("div", h("div.qp-l", "Your 3 × 3 filter · tap to cycle 0 → +1 → −1"),
            h("div.qp-k3", k.map((x, i) => h(`button.qp-kc${x > 0 ? ".pos" : x < 0 ? ".neg" : ""}`, { type: "button", disabled: revealed || null, "aria-label": `weight ${i + 1}: ${x}`,
              onclick: () => { k[i] = k[i] === 0 ? 1 : k[i] === 1 ? -1 : 0; set([...k]); paint(); } }, x > 0 ? "+1" : x < 0 ? "−1" : "0")))),
          h("div.qp-imgs",
            h("div.qp-pair", img(testImg("v"), "vertical edge"), heat(V, "response")),
            h("div.qp-pair", img(testImg("h"), "horizontal edge"), heat(H, "response")),
            h("div.qp-pair", img(testImg("f"), "flat grey"), heat(F, "response")))),
        h(`div.qp-sum${revealed ? (kernel.right(q, k) ? ".ok" : ".no") : ""}`, h("span", `Wanted: a strong response to the ${q.goal === "v" ? "vertical" : "horizontal"} edge, and exactly 0 everywhere on the other two.`)));
    }
    paint();
    return el;
  },
  right(q, v) {
    if (!Array.isArray(v)) return false;
    const want = conv(testImg(q.goal), v), other = conv(testImg(q.goal === "v" ? "h" : "v"), v), flat = conv(testImg("f"), v);
    return peak(want) >= 2 && peak(other) === 0 && peak(flat) === 0;
  },
  said: (q, v) => (v ? `[${v.slice(0, 3).join(" ")} / ${v.slice(3, 6).join(" ")} / ${v.slice(6).join(" ")}]` : "empty filter"),
  answer: (q) => (q.goal === "v" ? "Columns of opposite sign, e.g. [−1 0 +1 / −1 0 +1 / −1 0 +1] (the Prewitt filter)" : "Rows of opposite sign, e.g. [−1 −1 −1 / 0 0 0 / +1 +1 +1]"),
};

// ---------------------------------------------------------------------------------------------
const GATES = { AND: [0, 0, 0, 1], OR: [0, 1, 1, 1], XOR: [0, 1, 1, 0], NAND: [1, 1, 1, 0], NOR: [1, 0, 0, 0], XNOR: [1, 0, 0, 1] };
const GN = Object.keys(GATES);
const gates = {
  label: "Build it",
  view(q, value, set, revealed) {
    let g = value ? [...value] : q.outs.map(() => null);
    const el = h("div.qp.qp-gates");
    function paint() {
      const rows = [[0, 0], [0, 1], [1, 0], [1, 1]];
      fill(el, 
        h("div.qp-circ", q.outs.map((o, k) => h("div.qp-wire",
          h("span.qp-in", "A, B"), h("span.qp-arrow"),
          h(`div.qp-gate${g[k] ? ".on" : ""}${revealed ? (g[k] && GATES[g[k]].every((x, r) => x === o.tt[r]) ? ".ok" : ".no") : ""}`,
            GN.map((n) => h("button", { type: "button", disabled: revealed || null, "aria-pressed": g[k] === n ? "true" : "false", onclick: () => { g[k] = n; set(g.every(Boolean) ? [...g] : null); paint(); } }, n))),
          h("span.qp-arrow"), h("b.qp-out", o.name)))),
        h("table.qp-tt", h("thead", h("tr", h("th", "A"), h("th", "B"), q.outs.map((o) => [h("th", `${o.name} wanted`), h("th", "yours")]))),
          h("tbody", rows.map(([a, b], r) => h("tr", h("td", String(a)), h("td", String(b)), q.outs.map((o, k) => {
            const y = g[k] ? GATES[g[k]][r] : null;
            return [h("td", String(o.tt[r])), h(`td.${y == null ? "q" : y === o.tt[r] ? "ok" : "no"}`, y == null ? "?" : String(y))];
          }))))));
    }
    paint();
    return el;
  },
  right: (q, v) => Array.isArray(v) && q.outs.every((o, k) => v[k] && GATES[v[k]].every((x, r) => x === o.tt[r])),
  said: (q, v) => (v ? q.outs.map((o, k) => `${o.name} = ${v[k]}`).join(", ") : "not every gate chosen"),
  answer: (q) => q.outs.map((o) => `${o.name} = ${GN.find((n) => GATES[n].every((x, r) => x === o.tt[r]))}`).join(", "),
};

// ---------------------------------------------------------------------------------------------
const BLOCKS = [32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1056];
const launchOk = (q, v) => v && v.block % 32 === 0 && v.block <= 1024 && v.grid * v.block >= q.n && v.grid * v.block - q.n < (q.tight ? 32 : v.block);
const launch = {
  label: "Build it",
  view(q, value, set, revealed) {
    let v = value ? { ...value } : { block: 256, grid: 1 };
    const pic = h("div.qp-svg"), sum = h("div.qp-sum");
    function paint() {
      const total = v.grid * v.block, W = 640, shown = Math.min(v.grid, 40), bw = W / Math.max(shown, Math.min(40, Math.ceil(q.n / v.block)));
      let blocks = "";
      for (let k = 0; k < shown; k++) {
        const start = k * v.block, used = Math.max(0, Math.min(v.block, q.n - start)) / v.block;
        blocks += `<rect x="${k * bw + 1}" y="20" width="${Math.max(1, bw - 2)}" height="34" rx="3" class="qp-blk"/>`;
        if (used > 0) blocks += `<rect x="${k * bw + 1}" y="20" width="${Math.max(1, (bw - 2) * used)}" height="34" rx="3" class="qp-blk-on"/>`;
      }
      const need = Math.ceil(q.n / v.block);
      pic.innerHTML = `<svg viewBox="0 0 ${W} 80" role="img" aria-label="${v.grid} blocks of ${v.block} threads">${blocks}
        <text x="0" y="12" class="qp-rl">${v.grid} block${v.grid === 1 ? "" : "s"} × ${v.block} threads${v.grid > shown ? ` (first ${shown} drawn)` : ""}</text>
        <text x="0" y="74" class="qp-rl">filled: threads with an element · empty: threads that just exit at if (i &lt; n)</text></svg>`;
      const idle = total - q.n;
      sum.className = `qp-sum${revealed ? (launchOk(q, v) ? " ok" : " no") : ""}`;
      fill(sum, h("b", `${total.toLocaleString("en-US")} threads`), ` for ${q.n.toLocaleString("en-US")} elements · `,
        idle < 0 ? h("span", { style: { color: "var(--bad)" } }, `${(-idle).toLocaleString("en-US")} elements never touched`) : h("span", `${idle.toLocaleString("en-US")} threads idle`),
        v.block % 32 ? h("span", { style: { color: "var(--bad)" } }, " · not a whole number of warps") : null,
        v.block > 1024 ? h("span", { style: { color: "var(--bad)" } }, " · over the 1,024-thread block limit") : null,
        idle >= v.block ? h("span.muted", ` · ${need} block${need === 1 ? "" : "s"} would do`) : null);
    }
    const gridMax = Math.max(8, Math.ceil(q.n / 32) + 4);
    const bl = h("label.qp-sl", h("span", "threads per block"), h("input", { type: "range", min: 0, max: BLOCKS.length - 1, step: 1, value: Math.max(0, BLOCKS.indexOf(v.block)), disabled: revealed || null, "aria-label": "threads per block",
      oninput: (e) => { v.block = BLOCKS[+e.target.value]; set({ ...v }); paint(); } }));
    const gr = h("label.qp-sl", h("span", "blocks in the grid"), h("input", { type: "range", min: 1, max: gridMax, step: 1, value: v.grid, disabled: revealed || null, "aria-label": "blocks in the grid",
      oninput: (e) => { v.grid = +e.target.value; set({ ...v }); paint(); } }));
    paint();
    return h("div.qp.qp-launch", h("div.qp-ctrls", bl, gr), pic, sum, revealed ? null : hint("Cover every element, use whole warps (multiples of 32), at most 1,024 threads a block, and no spare block."));
  },
  right: (q, v) => !!launchOk(q, v),
  said: (q, v) => (v ? `${v.grid} blocks × ${v.block} threads` : "no launch"),
  answer: (q) => (q.tight ? `${Math.ceil(q.n / 32)} blocks × 32 threads, or 1 block × ${32 * Math.ceil(q.n / 32)}: less than one warp left idle` : `e.g. ${Math.ceil(q.n / 256)} block${Math.ceil(q.n / 256) === 1 ? "" : "s"} × 256 threads (blocks = ceil(n / threads per block))`),
};

export const PLAY = { estimate, order, bits, neuron, mask, kernel, gates, launch };
export const isRight = (q, a) => (q.type ? PLAY[q.type].right(q, a) : a === q.answer);
export const saidText = (q, a) => (q.type ? PLAY[q.type].said(q, a) : q.options[a]);
export const answerText = (q) => (q.type ? PLAY[q.type].answer(q) : q.options[q.answer]);
