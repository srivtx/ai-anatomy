// Plotting for ML chapters: a 2D canvas plane (scatter + decision fields),
// a static SVG x/y chart with log axes and hover, and an HTML heatmap.
import { h, s } from "./core.js";
import { flowCanvas } from "./flow.js";

const hex = (c) => {
  const m = /^#?([0-9a-f]{6})$/i.exec(String(c).trim());
  if (!m) return [128, 128, 128];
  const n = parseInt(m[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
};
export const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));

/** A 2D coordinate plane on canvas. xr/yr are data ranges. */
export function plane({ width = 420, height = 420, xr = [-1.2, 1.2], yr = [-1.2, 1.2], label, onPoint } = {}) {
  const fc = flowCanvas({ width, height, label });
  const X = (x) => ((x - xr[0]) / (xr[1] - xr[0])) * width;
  const Y = (y) => height - ((y - yr[0]) / (yr[1] - yr[0])) * height;
  const inv = (px, py) => [xr[0] + (px / width) * (xr[1] - xr[0]), yr[0] + ((height - py) / height) * (yr[1] - yr[0])];
  const off = document.createElement("canvas");
  if (onPoint) {
    fc.el.style.cursor = "crosshair";
    fc.el.addEventListener("pointerdown", (e) => {
      const rect = fc.el.getBoundingClientRect();
      const px = ((e.clientX - rect.left) / rect.width) * width, py = ((e.clientY - rect.top) / rect.height) * height;
      onPoint(...inv(px, py), e);
    });
  }
  const api = {
    ...fc, X, Y, inv, xr, yr,
    /** Draw now and again whenever the canvas resizes or the theme changes. */
    paint(fn) { fc.redraw = fn; fn(); },
    /** fn(x, y) -> t in [0,1]; painted between colour a (t=0) and b (t=1) through the surface colour */
    field(fn, { res = 60, a = fc.colors.c2, b = fc.colors.c1, alpha = 0.5, mid = fc.colors.surface } = {}) {
      off.width = res; off.height = res;
      const octx = off.getContext("2d"), img = octx.createImageData(res, res);
      const A = hex(a), B = hex(b), M = hex(mid);
      for (let j = 0; j < res; j++) for (let i = 0; i < res; i++) {
        const [x, y] = inv(((i + 0.5) / res) * width, ((j + 0.5) / res) * height);
        const t = Math.max(0, Math.min(1, fn(x, y)));
        const c = t < 0.5 ? mix(A, M, t * 2) : mix(M, B, (t - 0.5) * 2);
        const k = (j * res + i) * 4;
        img.data[k] = c[0]; img.data[k + 1] = c[1]; img.data[k + 2] = c[2]; img.data[k + 3] = 255 * alpha;
      }
      octx.putImageData(img, 0, 0);
      fc.ctx.imageSmoothingEnabled = true;
      fc.ctx.drawImage(off, 0, 0, width, height);
    },
    axes({ color = fc.colors.line } = {}) {
      if (xr[0] < 0 && xr[1] > 0) fc.line(X(0), 0, X(0), height, { color, lw: 1 });
      if (yr[0] < 0 && yr[1] > 0) fc.line(0, Y(0), width, Y(0), { color, lw: 1 });
    },
    scatter(points, { color = (p) => (p.y ? fc.colors.c1 : fc.colors.c2), r = 4, ring = true } = {}) {
      for (const p of points) {
        const x = X(p.x[0]), y = Y(p.x[1]);
        if (ring) fc.dot(x, y, r + 1.5, fc.colors.surface);
        fc.dot(x, y, r, typeof color === "function" ? color(p) : color);
      }
    },
    curve(fn, { color = fc.colors.ink, lw = 2, n = 200, dash } = {}) {
      const ctx = fc.ctx;
      ctx.beginPath(); ctx.setLineDash(dash || []);
      for (let i = 0; i <= n; i++) {
        const x = xr[0] + ((xr[1] - xr[0]) * i) / n, y = fn(x);
        if (!isFinite(y)) continue;
        i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y));
      }
      ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.stroke(); ctx.setLineDash([]);
    },
  };
  return api;
}

const nice = (v) => {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e12) return (v / 1e12).toFixed(a >= 1e13 ? 0 : 1) + "T";
  if (a >= 1e9) return (v / 1e9).toFixed(a >= 1e10 ? 0 : 1) + "B";
  if (a >= 1e6) return (v / 1e6).toFixed(a >= 1e7 ? 0 : 1) + "M";
  if (a >= 1e3) return (v / 1e3).toFixed(a >= 1e4 ? 0 : 1) + "k";
  if (a >= 10) return v.toFixed(0);
  if (a >= 1) return v.toFixed(1).replace(/\.0$/, "");
  return v.toPrecision(2);
};
export const fmtSI = nice;

function ticks(lo, hi, log) {
  if (log) {
    const out = [];
    for (let e = Math.ceil(Math.log10(lo) - 1e-9); e <= Math.floor(Math.log10(hi) + 1e-9); e++) out.push(10 ** e);
    if (out.length > 7) return out.filter((_, i) => i % Math.ceil(out.length / 6) === 0);
    if (out.length < 3) {
      // a narrow range: add 2× and 5× steps so the axis still has labels
      const more = [];
      for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) for (const m of [1, 2, 5]) { const v = m * 10 ** e; if (v >= lo * 0.999 && v <= hi * 1.001) more.push(v); }
      return more;
    }
    return out;
  }
  const span = hi - lo, step0 = span / 4, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((st) => st >= step0) || step0;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) out.push(+v.toPrecision(12));
  return out;
}

/**
 * Static x/y line chart in SVG with optional log axes and a hover readout.
 * series: [{name, color, points: [[x, y]], dash?, dots?}]
 */
export function xyChart({ width = 800, height = 260, xLog = false, yLog = false, xLabel = "", yLabel = "", xDomain, yDomain, fmtX = nice, fmtY = nice, label } = {}) {
  const W = width, H = height, P = { l: 58, r: 16, t: 12, b: 40 };
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart xy", role: "img", "aria-label": label || yLabel || "chart" });
  const legend = h("div.legend");
  const tip = h("div.xy-tip", { hidden: true });
  const box = h("div.chartbox.xy-box", legend, h("div", { style: { position: "relative" } }, svg, tip));
  let cur = [], sx, sy;
  function update(series, opts = {}) {
    cur = series;
    const xs = series.flatMap((se) => se.points.map((p) => p[0])), ys = series.flatMap((se) => se.points.map((p) => p[1])).filter((v) => isFinite(v));
    const xd = opts.xDomain || xDomain || [Math.min(...xs), Math.max(...xs)];
    const yd = opts.yDomain || yDomain || [Math.min(...ys), Math.max(...ys)];
    const tx = (v) => (xLog ? Math.log10(v) : v), ty = (v) => (yLog ? Math.log10(Math.max(v, 1e-12)) : v);
    const [x0, x1] = [tx(xd[0]), tx(xd[1])], [y0, y1] = [ty(yd[0]), ty(yd[1])];
    sx = (v) => P.l + ((tx(v) - x0) / (x1 - x0 || 1)) * (W - P.l - P.r);
    sy = (v) => P.t + (1 - (ty(v) - y0) / (y1 - y0 || 1)) * (H - P.t - P.b);
    svg.replaceChildren();
    for (const t of ticks(yd[0], yd[1], yLog)) {
      svg.append(s("line", { x1: P.l, x2: W - P.r, y1: sy(t), y2: sy(t), class: "gl" }), s("text", { x: P.l - 8, y: sy(t) + 4, "text-anchor": "end", class: "tk", text: fmtY(t) }));
    }
    for (const t of ticks(xd[0], xd[1], xLog)) svg.append(s("text", { x: sx(t), y: H - P.b + 16, "text-anchor": "middle", class: "tk", text: fmtX(t) }));
    if (xLabel) svg.append(s("text", { x: (P.l + W - P.r) / 2, y: H - 6, "text-anchor": "middle", class: "ax", text: xLabel }));
    if (yLabel) svg.append(s("text", { x: 14, y: (P.t + H - P.b) / 2, "text-anchor": "middle", class: "ax", transform: `rotate(-90 14 ${(P.t + H - P.b) / 2})`, text: yLabel }));
    for (const se of series) {
      const pts = se.points.filter((p) => isFinite(p[1]) && (!yLog || p[1] > 0));
      if (se.line !== false && pts.length > 1)
        svg.append(s("path", { class: "ln", d: pts.map((p, i) => `${i ? "L" : "M"}${sx(p[0]).toFixed(1)},${sy(Math.min(Math.max(p[1], yd[0]), yd[1])).toFixed(1)}`).join(""), style: `stroke:${se.color}${se.dash ? ";stroke-dasharray:6 5" : ""}` }));
      if (se.dots) for (const p of pts) svg.append(s("circle", { cx: sx(p[0]), cy: sy(p[1]), r: se.r || 4.5, style: `fill:${se.color};stroke:var(--surface);stroke-width:2` }));
    }
    for (const m of opts.marks || []) {
      svg.append(s("line", { x1: sx(m.x), x2: sx(m.x), y1: P.t, y2: H - P.b, style: "stroke:var(--ink-3);stroke-dasharray:3 4" }));
      if (m.label) svg.append(s("text", { x: sx(m.x) + 6, y: P.t + 12, class: "tk", text: m.label }));
    }
    legend.replaceChildren(...(series.length > 1 ? series.filter((se) => se.name).map((se) => h("span", h("i", { style: { background: se.color } }), se.name)) : []));
  }
  svg.addEventListener("pointermove", (e) => {
    if (!cur.length) return;
    const r = svg.getBoundingClientRect(), px = ((e.clientX - r.left) / r.width) * W;
    let best = null;
    for (const se of cur) for (const p of se.points) {
      const d = Math.abs(sx(p[0]) - px);
      if (!best || d < best.d) best = { d, p, se };
    }
    if (!best || best.d > 40) return (tip.hidden = true);
    tip.hidden = false;
    tip.innerHTML = `<b>${best.se.name || ""}</b> ${xLabel ? xLabel + " " : ""}${fmtX(best.p[0])} → ${fmtY(best.p[1])}`;
    tip.style.left = `${Math.min(70, (sx(best.p[0]) / W) * 100)}%`;
  });
  svg.addEventListener("pointerleave", () => (tip.hidden = true));
  return { el: box, update };
}

/** HTML heatmap for small matrices (attention weights, similarity). values in [0, 1]. */
export function heatmap(M, { rows, cols, fmt = (v) => v.toFixed(2), color = "var(--accent)", cell = 34, showValues = true, onCell } = {}) {
  const grid = h("div.hm", { style: { gridTemplateColumns: `auto repeat(${M[0].length}, minmax(${cell * 0.7}px, ${cell}px))` } });
  grid.append(h("span"));
  (cols || M[0].map((_, j) => j)).forEach((c) => grid.append(h("span.hm-c", c)));
  M.forEach((row, i) => {
    grid.append(h("span.hm-r", rows ? rows[i] : i));
    row.forEach((v, j) => grid.append(h("span.hm-v", {
      title: `${rows ? rows[i] : i} → ${cols ? cols[j] : j}: ${fmt(v)}`,
      style: { background: v == null ? "transparent" : `color-mix(in srgb, ${color} ${Math.round(Math.max(0, Math.min(1, v)) * 100)}%, var(--surface))`, color: v > 0.55 ? "var(--accent-ink)" : "var(--ink-2)" },
      onclick: onCell ? () => onCell(i, j) : null,
    }, v == null ? "" : showValues ? fmt(v) : "")));
  });
  return h("div.hm-wrap", grid);
}
