// Small pictures for the Ask Neuro conversations. Each takes options and { onGoal } and returns
// { el, solve? }. "Try" steps call onGoal when the learner reaches the goal; solve() jumps there.
// Everything is SVG or plain HTML in theme colours, so it works in light and dark and at phone width.
import { h, s, clamp } from "./core.js";
import { box, svg, rangeIn } from "./askkit.js";
import { VIZ2 } from "./askviz2.js";

/* 1. learn: label a few messages yourself; that's training data */
function sortMail({ onGoal }) {
  const mails = [["WIN a FREE phone now!!!", 1], ["Lunch at 1 tomorrow?", 0], ["Your prize is waiting, click here", 1], ["Notes from today's class", 0], ["Cheap pills, no doctor needed", 1], ["Mum: call me when you land", 0]];
  const lab = mails.map(() => null);
  const rows = mails.map(([t], i) => {
    const b = (v, txt) => h("button.av-tag", { type: "button", onclick: () => { lab[i] = v; draw(); } }, txt);
    return h("div.av-mail", h("span.av-mt", t), h("span.av-tags", b(1, "spam"), b(0, "fine")));
  });
  const note = h("p.av-note");
  function draw() {
    rows.forEach((r, i) => { r.dataset.v = lab[i] == null ? "" : lab[i] ? "spam" : "ok"; r.dataset.right = lab[i] == null ? "" : String(lab[i] === mails[i][1]); });
    const n = lab.filter((v) => v != null).length;
    note.textContent = n < mails.length ? `${n} of ${mails.length} labelled.` : "Every message now has a label. That table of examples is training data.";
    if (n === mails.length) onGoal?.();
  }
  draw();
  return { el: box("mail", rows, note), solve() { mails.forEach(([, v], i) => (lab[i] = v)); draw(); } };
}

/* 2. numbers: a picture is a grid of numbers. Tap cells to paint. */
function pixels({ onGoal }) {
  const N = 8, art = ["..XXXX..", ".X....X.", "......X.", "...XXX..", "......X.", "......X.", ".X....X.", "..XXXX.."];
  const v = art.flatMap((r) => [...r].map((c) => (c === "X" ? 9 : 0)));
  let showNums = false, painted = 0;
  const cells = v.map((_, i) => h("button.av-px", { type: "button", "aria-label": `pixel ${i + 1}`, onclick: () => { v[i] = v[i] ? 0 : 9; painted++; draw(); if (painted >= 3) onGoal?.(); } }));
  const grid = h("div.av-pxg", { style: { "--n": N } }, cells);
  const tog = h("button.btn.small", { type: "button", onclick: () => { showNums = !showNums; draw(); } });
  function draw() {
    cells.forEach((c, i) => { c.style.setProperty("--v", v[i] / 9); c.textContent = showNums ? String(v[i]) : ""; });
    tog.textContent = showNums ? "Show the picture" : "Show the numbers";
  }
  draw();
  return { el: box("pixels", grid, h("div.av-row", tog, h("small.muted", "Tap squares to paint them."))), solve() { painted = 3; showNums = true; draw(); } };
}

/* 3–4. line: house size → price. Move the line; optionally see each miss and the total. */
const HOUSES = [[40, 120], [55, 150], [62, 185], [75, 200], [88, 240], [95, 250], [110, 300], [125, 320]];
function line({ onGoal, errors = false, squares = false, goal = 0 }) {
  const W = 320, H = 210, px = (x) => 30 + (x / 140) * (W - 40), py = (y) => H - 24 - (y / 360) * (H - 40);
  let a = 1.2, b = 60;
  const lineEl = s("line", { class: "av-ln" }), misses = s("g"), dots = s("g");
  HOUSES.forEach(([x, y]) => dots.append(s("circle", { cx: px(x), cy: py(y), r: 5, class: "av-dot" })));
  const total = h("b.av-big");
  const out = svg(W, H,
    s("line", { x1: 30, y1: H - 24, x2: W - 6, y2: H - 24, class: "av-axis" }), s("line", { x1: 30, y1: 8, x2: 30, y2: H - 24, class: "av-axis" }),
    s("text", { x: W - 8, y: H - 8, "text-anchor": "end", class: "av-lab" }, "house size (m²)"), s("text", { x: 34, y: 16, class: "av-lab" }, "price (k)"),
    misses, lineEl, dots);
  const err = () => HOUSES.reduce((t, [x, y]) => t + (squares ? (y - (a * x + b)) ** 2 / 100 : Math.abs(y - (a * x + b))), 0);
  function draw() {
    lineEl.setAttribute("x1", px(0)); lineEl.setAttribute("y1", py(b)); lineEl.setAttribute("x2", px(140)); lineEl.setAttribute("y2", py(a * 140 + b));
    misses.replaceChildren(...(errors ? HOUSES.map(([x, y]) => s("line", { x1: px(x), x2: px(x), y1: py(y), y2: py(a * x + b), class: "av-miss" })) : []));
    const e = err();
    total.textContent = errors ? `${squares ? "total of squared misses" : "total miss"}: ${Math.round(e)}` : "";
    if (goal && e < goal) onGoal?.();
  }
  const sa = rangeIn("steepness", 0, 3, 0.05, a, (v) => { a = v; draw(); });
  const sb = rangeIn("start", -40, 160, 1, b, (v) => { b = v; draw(); });
  sa.init(); sb.init();
  return { el: box("line", out, h("div.av-ctl", sa.el, sb.el), errors ? total : null), solve() { sa.set(2.3); sb.set(30); } };
}

/* 5. downhill: a ball on a valley. Take steps of a chosen size. */
function valley({ onGoal }) {
  const W = 320, H = 190, f = (x) => (x - 1) ** 2, X = (x) => 20 + ((x + 2) / 6) * (W - 40), Y = (y) => H - 20 - (y / 10) * (H - 40);
  let x = -1.6, lr = 0.2, steps = 0;
  let d = ""; for (let i = 0; i <= 60; i++) { const xx = -2 + (6 * i) / 60; if (f(xx) <= 10.5) d += `${d ? "L" : "M"}${X(xx)} ${Y(f(xx))}`; }
  const ball = s("circle", { r: 9, class: "av-ball" }), arrow = s("line", { class: "av-arrow" }), trail = s("g");
  const out = svg(W, H, s("path", { d, class: "av-curve" }), trail, arrow, ball, s("text", { x: X(1), y: H - 4, "text-anchor": "middle", class: "av-lab" }, "lowest point"));
  const note = h("p.av-note");
  function draw() {
    ball.setAttribute("cx", X(clamp(x, -2, 4))); ball.setAttribute("cy", Y(Math.min(10, f(x))) - 9);
    const slope = 2 * (x - 1);
    arrow.setAttribute("x1", X(x)); arrow.setAttribute("y1", Y(Math.min(10, f(x))) - 22); arrow.setAttribute("x2", X(x) - clamp(slope, -3, 3) * 14); arrow.setAttribute("y2", Y(Math.min(10, f(x))) - 22);
    note.textContent = Math.abs(x) > 6 ? "The ball flew off! The steps are too big." : `Steps: ${steps}. Slope here: ${slope.toFixed(2)} (${Math.abs(slope) < 0.1 ? "flat: we're at the bottom" : slope > 0 ? "uphill to the right, so step left" : "downhill to the right, so step right"}).`;
    if (Math.abs(x - 1) < 0.05 && steps) onGoal?.();
  }
  const sl = rangeIn("step size", 0.05, 1.1, 0.05, lr, (v) => { lr = v; });
  sl.init();
  const step = () => { const old = x; x = x - lr * 2 * (x - 1); steps++; trail.append(s("circle", { cx: X(clamp(old, -2, 4)), cy: Y(Math.min(10, f(old))) - 9, r: 3, class: "av-trail" })); draw(); };
  const reset = () => { x = -1.6; steps = 0; trail.replaceChildren(); draw(); };
  draw();
  return { el: box("valley", out, h("div.av-ctl", sl.el, h("div.av-row", h("button.btn.primary.small", { type: "button", onclick: step }, "Take a step"), h("button.btn.small", { type: "button", onclick: reset }, "Start over"))), note),
    solve() { sl.set(0.5); x = 1; steps = 1; draw(); } };
}

/* 6. neuron: two inputs, two weights, a bias, and a squash. Make it act like AND. */
function neuron({ onGoal, target = "and" }) {
  let w1 = 0.5, w2 = 0.5, bias = 0, x1 = 1, x2 = 1;
  const sq = (z) => 1 / (1 + Math.exp(-z));
  const out = (a, b) => sq(w1 * a + w2 * b + bias);
  const W = 320, H = 170;
  const in1 = s("circle", { cx: 40, cy: 50, r: 16 }), in2 = s("circle", { cx: 40, cy: 120, r: 16 }), bulb = s("circle", { cx: 270, cy: 85, r: 22, class: "av-bulb" });
  const e1 = s("line", { x1: 56, y1: 50, x2: 150, y2: 85 }), e2 = s("line", { x1: 56, y1: 120, x2: 150, y2: 85 });
  const t1 = s("text", { x: 40, y: 55, "text-anchor": "middle", class: "av-nt" }), t2 = s("text", { x: 40, y: 125, "text-anchor": "middle", class: "av-nt" }), tz = s("text", { x: 170, y: 90, "text-anchor": "middle", class: "av-nt" }), to = s("text", { x: 270, y: 125, "text-anchor": "middle", class: "av-lab" });
  const pic = svg(W, H, e1, e2, s("line", { x1: 190, y1: 85, x2: 248, y2: 85, class: "av-edge" }), in1, in2, s("rect", { x: 150, y: 65, width: 40, height: 40, rx: 10, class: "av-sum" }), tz, bulb, t1, t2, to,
    s("text", { x: 100, y: 52, "text-anchor": "middle", class: "av-lab" }, "× w1"), s("text", { x: 100, y: 128, "text-anchor": "middle", class: "av-lab" }, "× w2"));
  const table = h("div.av-tt");
  const want = (a, b) => (target === "and" ? a && b : a || b);
  function draw() {
    [in1, in2].forEach((c, i) => c.setAttribute("class", `av-in ${[x1, x2][i] ? "on" : ""}`));
    t1.textContent = String(x1); t2.textContent = String(x2);
    e1.setAttribute("class", "av-edge"); e1.setAttribute("stroke-width", 1 + Math.abs(w1) * 2.2);
    e2.setAttribute("class", "av-edge"); e2.setAttribute("stroke-width", 1 + Math.abs(w2) * 2.2);
    tz.textContent = "+";
    const o = out(x1, x2);
    bulb.style.setProperty("--o", o.toFixed(2)); to.textContent = `output ${o.toFixed(2)}`;
    let ok = 0;
    table.replaceChildren(h("span.av-th", "in 1"), h("span.av-th", "in 2"), h("span.av-th", "output"), h("span.av-th", "want"),
      ...[[0, 0], [0, 1], [1, 0], [1, 1]].flatMap(([a, b]) => { const y = out(a, b), good = (y > 0.5) === !!want(a, b); ok += good; return [h("span", String(a)), h("span", String(b)), h(`span.${good ? "good" : "bad"}`, `${good ? "✓" : "✗"} ${y.toFixed(2)}`), h("span", want(a, b) ? "on" : "off")]; }));
    if (ok === 4) onGoal?.();
  }
  const s1 = rangeIn("w1", -4, 4, 0.1, w1, (v) => { w1 = v; draw(); });
  const s2 = rangeIn("w2", -4, 4, 0.1, w2, (v) => { w2 = v; draw(); });
  const sb = rangeIn("bias", -6, 4, 0.1, bias, (v) => { bias = v; draw(); });
  [s1, s2, sb].forEach((x) => x.init());
  in1.addEventListener("click", () => { x1 = 1 - x1; draw(); }); in2.addEventListener("click", () => { x2 = 1 - x2; draw(); });
  return { el: box("neuron", pic, h("div.av-ctl", s1.el, s2.el, sb.el), table), solve() { s1.set(3); s2.set(3); sb.set(-4.5); } };
}

/* 7. xor: four points; a single line can't split them, two lines can. */
function xor({ onGoal }) {
  const W = 260, H = 220, P = [[0, 0, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1]], X = (x) => 50 + x * 160, Y = (y) => 180 - y * 140;
  let ang = 30, off = 0, two = false;
  const g = s("g"), lines = s("g"), note = h("p.av-note");
  P.forEach(([x, y, c]) => g.append(s("circle", { cx: X(x), cy: Y(y), r: 13, class: `av-cls c${c}` })));
  const pic = svg(W, H, lines, g);
  function draw() {
    const t = (ang * Math.PI) / 180, nx = Math.cos(t), ny = Math.sin(t), cx = 0.5 + off * nx, cy = 0.5 + off * ny;
    const L = (o) => { const ax = cx + o * nx, ay = cy + o * ny; return s("line", { x1: X(ax - 2 * ny), y1: Y(ay + 2 * nx), x2: X(ax + 2 * ny), y2: Y(ay - 2 * nx), class: "av-ln" }); };
    lines.replaceChildren(...(two ? [L(-0.32), L(0.32)] : [L(0)]));
    if (two) { note.textContent = "Two lines make a band. Orange inside, blue outside. Solved."; return; }
    // best labelling of the two sides
    const side = P.map(([x, y]) => (x - cx) * nx + (y - cy) * ny > 0);
    let right = 0; for (const flip of [0, 1]) right = Math.max(right, P.filter(([, , c], i) => (side[i] !== !!flip) === !!c).length);
    note.textContent = `${right} of 4 on the correct side. ${right < 4 ? "Keep trying... or is it impossible?" : ""}`;
  }
  const sa = rangeIn("turn", 0, 180, 1, ang, (v) => { ang = v; draw(); });
  const so = rangeIn("move", -0.7, 0.7, 0.01, off, (v) => { off = v; draw(); });
  sa.init(); so.init();
  const btn = h("button.btn.small", { type: "button", onclick: () => { two = !two; btn.textContent = two ? "Back to one line" : "Use two lines"; ang = 45; sa.set(45); so.set(0); draw(); if (two) onGoal?.(); } }, "Use two lines");
  return { el: box("xor", pic, h("div.av-ctl", sa.el, so.el, btn), note), solve() { if (!two) btn.click(); } };
}

/* 8. blame: an error at the end flows backwards, shared by how much each part pushed. */
function blame() {
  const names = ["input", "part A", "part B", "answer"], W = 320, H = 120, X = (i) => 40 + i * 80;
  const nodes = names.map((n, i) => s("g", {}, s("circle", { cx: X(i), cy: 50, r: 22, class: "av-node" }), s("text", { x: X(i), y: 92, "text-anchor": "middle", class: "av-lab" }, n)));
  const vals = names.map((_, i) => s("text", { x: X(i), y: 55, "text-anchor": "middle", class: "av-nt" }, ""));
  const arrows = s("g");
  const pic = svg(W, H, arrows, ...nodes, ...vals);
  const note = h("p.av-note", "The answer was 9. It should have been 6. Who's to blame?");
  let k = 0;
  const blames = [null, "0.5", "1.5", "3"];
  function step() {
    if (k >= 3) return;
    const i = 3 - k;
    vals[i].textContent = blames[i];
    nodes[i].firstChild.classList.add("hot");
    arrows.append(s("line", { x1: X(i) - 24, y1: 30, x2: X(i - 1) + 24, y2: 30, class: "av-back" }));
    k++;
    note.textContent = ["Off by 3 at the answer.", "Part B pushed the answer most, so it gets most of the blame: 1.5.", "Part A pushed part B a little, so it gets a smaller share: 0.5. Now every part knows how to change."][k - 1];
    if (k === 3) btn.disabled = true;
  }
  const btn = h("button.btn.primary.small", { type: "button", onclick: step }, "Pass the blame back");
  return { el: box("blame", pic, h("div.av-row", btn), note), solve() { while (k < 3) step(); } };
}

/* 9. memorise: a curve that bends more and more. Practice error falls; test error doesn't. */
function wiggle({ onGoal }) {
  const R = (i) => Math.sin(i * 12.9898) * 43758.5453 % 1;
  const train = Array.from({ length: 9 }, (_, i) => { const x = i / 8; return [x, Math.sin(x * 3) * 0.6 + (R(i + 1) - 0.5) * 0.35]; });
  const test = Array.from({ length: 8 }, (_, i) => { const x = (i + 0.5) / 8; return [x, Math.sin(x * 3) * 0.6 + (R(i + 40) - 0.5) * 0.35]; });
  const W = 320, H = 200, X = (x) => 20 + x * (W - 40), Y = (y) => H / 2 - y * 70;
  const fit = (deg) => { // least squares on powers of (x - 0.5), with a tiny ridge so high degrees stay stable
    const m = deg + 1, A = Array.from({ length: m }, () => Array(m).fill(0)), bv = Array(m).fill(0);
    for (const [x, y] of train) { const p = Array.from({ length: m }, (_, j) => (2 * (x - 0.5)) ** j); for (let i = 0; i < m; i++) { bv[i] += p[i] * y; for (let j = 0; j < m; j++) A[i][j] += p[i] * p[j]; } }
    for (let i = 0; i < m; i++) A[i][i] += 1e-9;
    for (let i = 0; i < m; i++) { let piv = i; for (let r = i + 1; r < m; r++) if (Math.abs(A[r][i]) > Math.abs(A[piv][i])) piv = r; [A[i], A[piv]] = [A[piv], A[i]]; [bv[i], bv[piv]] = [bv[piv], bv[i]]; for (let r = i + 1; r < m; r++) { const f = A[r][i] / A[i][i]; for (let c = i; c < m; c++) A[r][c] -= f * A[i][c]; bv[r] -= f * bv[i]; } }
    const w = Array(m).fill(0); for (let i = m - 1; i >= 0; i--) { let t = bv[i]; for (let j = i + 1; j < m; j++) t -= A[i][j] * w[j]; w[i] = t / A[i][i]; }
    return (x) => w.reduce((t, c, j) => t + c * (2 * (x - 0.5)) ** j, 0);
  };
  let deg = 1, showTest = false;
  const curve = s("path", { class: "av-ln", fill: "none" }), tg = s("g"), pg = s("g");
  train.forEach(([x, y]) => pg.append(s("circle", { cx: X(x), cy: Y(y), r: 5, class: "av-dot" })));
  const pic = svg(W, H, curve, pg, tg);
  const bars = h("div.av-bars");
  function draw() {
    const f = fit(deg); let d = "";
    for (let i = 0; i <= 120; i++) { const x = i / 120; d += `${i ? "L" : "M"}${X(x)} ${Y(clamp(f(x), -1.4, 1.4))}`; }
    curve.setAttribute("d", d);
    tg.replaceChildren(...(showTest ? test.map(([x, y]) => s("rect", { x: X(x) - 5, y: Y(y) - 5, width: 10, height: 10, class: "av-test" })) : []));
    const mse = (pts) => pts.reduce((t, [x, y]) => t + (f(x) - y) ** 2, 0) / pts.length;
    const a = mse(train), b = mse(test), top = 0.25;
    bars.replaceChildren(
      h("div", h("small", "practice mistakes"), h("span.av-bar", h("i", { style: { width: `${Math.min(100, (100 * a) / top)}%` } })), h("b", a.toFixed(3))),
      h("div", h("small", "test mistakes"), h("span.av-bar.t", h("i", { style: { width: showTest ? `${Math.min(100, (100 * b) / top)}%` : "0%" } })), h("b", showTest ? b.toFixed(3) : "hidden")));
    if (showTest && deg >= 7) onGoal?.();
  }
  const sd = rangeIn("how bendy", 1, 8, 1, deg, (v) => { deg = v; draw(); });
  sd.init();
  const tb = h("button.btn.small", { type: "button", onclick: () => { showTest = !showTest; tb.textContent = showTest ? "Hide the test points" : "Show new test points"; draw(); } }, "Show new test points");
  return { el: box("wiggle", pic, h("div.av-ctl", sd.el, tb), bars), solve() { if (!showTest) tb.click(); sd.set(8); } };
}

export const VIZ = { sortMail, pixels, line, valley, neuron, xor, blame, wiggle, ...VIZ2 };
export function make(spec, ctx) {
  const [type, opts] = typeof spec === "string" ? [spec, {}] : [spec.type, spec];
  const fn = VIZ[type];
  if (!fn) return h("div.av", `missing picture: ${type}`);
  return fn({ ...opts, ...ctx });
}
