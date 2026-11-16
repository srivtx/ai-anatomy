// The Test tab: predict a small real run, send it to Colab, read the result back, keep a log.
import { h } from "../core.js";
import { segmented } from "../widgets.js";
import { colabUrl } from "../catalog.js";
import { mascot } from "../pixel.js";
import { check } from "./arch.js";
import { SIZES, NOTEBOOK, MEASURED, labOf, labSpec, expectLab, parseResult, score, log, sameShape } from "./lab.js";

const sgn = (x, d = 2) => `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(d)}`;
const COL = { baseline: "#4f8ef7", design: "#ff8a3d" };

/** ctx: { cfg, base, names, url, size, setSize(s), flag(f), overlay(values|null), result, setResult(r) } */
export function testPane(ctx) {
  const { cfg, base, size } = ctx, errs = check(cfg);
  const lb = labOf(base, size), ld = labOf(cfg, size), ex = expectLab(cfg, base, size), S = SIZES[size];
  const same = sameShape(lb, ld);

  // 1. what the lab run trains
  const row = (label, f) => { const a = f(lb), b = f(ld); return h("tr", { "data-diff": a !== b ? "1" : null }, h("th", label), h("td", a), h("td", b)); };
  const pattern = (m) => { const t = m.blocks.map((b) => (b.attn === "none" ? (b.mlp ? "M" : "·") : b.attn === "window" ? (b.mlp ? "w" : "W") : b.mlp ? "■" : "A")).join(""); return t; };
  const shape = h("div.st-card",
    h("div.eyebrow", "The lab run"),
    segmented({ label: "Length", options: Object.entries(SIZES).map(([k, s]) => ({ value: k, label: `${s.label} · ${s.mins}` })), value: size, onChange: (v) => ctx.setSize(v) }),
    h("table.lab-t",
      h("thead", h("tr", h("th", ""), h("th", ctx.names.base), h("th", "Your design"))),
      h("tbody",
        row("Blocks", (m) => `${m.blocks.length} ${pattern(m)}`),
        row("Width", (m) => String(m.d)),
        row("Heads (K/V)", (m) => `${m.heads} (${m.kv})`),
        row("MLP", (m) => `${m.act} ${(m.ff / m.d).toFixed(1)}×`),
        row("Norm", (m) => (m.norm === "None" ? "none" : `${m.norm}, ${m.normPos}`)),
        row("Residual", (m) => (m.residual ? "on" : "off")),
        row("Positions", (m) => m.pos),
        row("Window", (m) => (m.blocks.some((b) => b.attn === "window") ? `${m.window} chars` : "—")))),
    h("p.st-small.muted", `Shakespeare, one character per token, ${S.steps.toLocaleString("en-US")} steps × ${S.seeds} seed${S.seeds > 1 ? "s" : ""}, context ${S.ctx}. The run keeps your structure and shrinks the size; size itself is the scaling law's job (Predict tab). ■ = attention + MLP, A = attention only, M = MLP only, w = window, · = empty.`));

  // 2. the prediction
  const pred = h("div.st-card.lab-pred", { "data-tone": ex.verdict.tone },
    h("div.st-pc-top", h("div.eyebrow", "We predict"), h("span.st-tag", { "data-tag": "rule", title: "Rules of thumb calibrated on lab runs. A real run decides." }, "Rule of thumb")),
    same ? h("div",
      h("div.lab-big", "A tie"),
      h("p.st-small", "At lab size your design has the same structure as the baseline, so the two runs should land within seed noise (±0.02). What you changed is size, which the scaling law on the Predict tab covers."))
      : h("div",
        h("div.lab-big", ex.verdict.word, h("span", `${sgn(ex.c)} ± ${ex.s.toFixed(2)} loss`)),
        rangeBar(ex),
        h("ul.lab-items", ex.items.map((it) => h("li", h("div.st-pc-top", h("b", it.what), h("span.st-delta", { "data-good": Math.abs(it.c) < 0.005 ? "n" : it.c < 0 ? "1" : "" }, sgn(it.c))), h("p", it.why))))));

  // 3. run it
  const { line } = labSpec(cfg, base, size, ctx.names);
  const status = h("span.lab-ok", { "aria-live": "polite" });
  const remember = () => { log.add({ id: labSpec(cfg, base, size, ctx.names).spec.id, at: Date.now(), name: ctx.names.design, size, url: ctx.url, c: ex.c, s: ex.s, same }); ctx.flag("copied"); };
  const copy = async () => {
    try { await navigator.clipboard.writeText(line); status.textContent = "Copied"; } catch { prompt("Copy this line:", line); }
    remember(); setTimeout(() => (status.textContent = ""), 2200);
  };
  const download = async () => {
    try {
      const nb = await (await fetch(`/notebooks/${NOTEBOOK}.ipynb`)).json();
      const cell = nb.cells.find((c) => c.cell_type === "code" && [].concat(c.source).join("").startsWith("STUDIO = "));
      cell.source = [`STUDIO = "${line}"   # your design, from the Studio`];
      const a = h("a", { href: URL.createObjectURL(new Blob([JSON.stringify(nb, null, 1)], { type: "application/x-ipynb+json" })), download: `studio_lab_${size}.ipynb` });
      document.body.append(a); a.click(); a.remove(); remember(); status.textContent = "Downloaded. In Colab: File › Upload notebook.";
    } catch { status.textContent = "Couldn't build the notebook here; use Copy instead."; }
  };
  const run = h("div.st-card",
    h("div.eyebrow", "Run it on Colab"),
    errs.length ? h("p.st-small", { style: "color:var(--bad)" }, "Fix the shape error first: there's no way to build this model.") :
      h("ol.lab-steps",
        h("li", h("b", "Copy your design"), h("div.row", h("button.btn.primary.small", { type: "button", onclick: copy }, "Copy design"), h("button.btn.small", { type: "button", onclick: download }, "Download notebook"), status)),
        h("li", h("b", "Open the lab, paste it in the first cell, pick a T4 GPU, Run all"), h("div.row", h("a.btn.small", { href: colabUrl(NOTEBOOK), target: "_blank", rel: "noopener" }, "Open the lab in Colab ↗"))),
        h("li", h("b", "Paste the last line it prints below"))));

  // 4. paste the result
  const ta = h("textarea.lab-ta", { rows: 2, placeholder: "STUDIO_RESULT:…", spellcheck: "false", "aria-label": "Result line from the notebook" });
  const read = () => {
    const r = parseResult(ta.value);
    if (!r) { msg.textContent = "That isn't a result line. It starts with STUDIO_RESULT:"; return; }
    ctx.setResult(r);
  };
  const msg = h("p.st-small", { style: "color:var(--bad)" });
  ta.addEventListener("input", () => { if (parseResult(ta.value)) read(); });
  const paste = h("div.st-card", h("div.eyebrow", "Paste the result"), ta, h("div.row", h("button.btn.small", { type: "button", onclick: read }, "Check it"), msg));

  return h("div.st-pane-in", ctx.result ? resultCard(ctx, ex) : null, pred, run, paste, shape, logCard(ctx), measuredCard());
}

/** A number line: baseline at 0, the predicted gap as a band. */
function rangeBar(ex, measured) {
  const lo = Math.min(-0.1, ex.c - ex.s - 0.05, measured ?? 0), hi = Math.max(0.1, ex.c + ex.s + 0.05, measured ?? 0);
  const X = (v) => `${(((v - lo) / (hi - lo)) * 100).toFixed(2)}%`;
  return h("div.lab-range", { role: "img", "aria-label": `Predicted gap ${sgn(ex.c)} ± ${ex.s.toFixed(2)}${measured != null ? `, measured ${sgn(measured)}` : ""}` },
    h("span.lab-band", { style: `left:${X(ex.c - ex.s)};width:calc(${X(ex.c + ex.s)} - ${X(ex.c - ex.s)})` }),
    h("span.lab-zero", { style: `left:${X(0)}` }, h("i", "baseline")),
    h("span.lab-mid", { style: `left:${X(ex.c)}` }),
    measured != null && Number.isFinite(measured) ? h("span.lab-meas", { style: `left:${X(measured)}` }, h("i", `measured ${sgn(measured)}`)) : null,
    h("span.lab-ax", { style: "left:0" }, "better"), h("span.lab-ax", { style: "right:0" }, "worse"));
}

function resultCard(ctx, ex) {
  const r = ctx.result, sc = score(r, ex);
  if (!sc) return h("div.st-card", h("p", "This result doesn't have both runs."));
  const mood = sc.held ? "happy" : sc.word === "Surprise" ? "wow" : "meh";
  const mine = r.id === labSpec(ctx.cfg, ctx.base, ctx.size, ctx.names).spec.id;
  log.attach(r.id, sc);
  const B = r.runs.baseline, D = r.runs.design;
  return h("div.st-card.lab-res", { "data-held": sc.held ? "1" : "" },
    h("div.lab-res-top",
      h("div.lab-m", { html: mascot("neuro", mood, { label: "Neuro" }) }),
      h("div", h("div.eyebrow", "Measured"), h("div.lab-big", sc.word), h("p.st-small", sc.say))),
    mine ? null : h("p.st-small.lab-warn", "This result came from a different design or lab length than the one open now, so the prediction below may not match it. Open it from Your experiments."),
    h("div.lab-nums",
      h("div", h("span", B.label), h("b", sc.bl.toFixed(3))),
      h("div", h("span", D.label), h("b", Number.isFinite(sc.dl) ? sc.dl.toFixed(3) : "diverged")),
      h("div", h("span", "Gap (predicted)"), h("b", Number.isFinite(sc.gap) ? sgn(sc.gap, 3) : "—", h("small", ` (${sgn(ex.c)} ± ${ex.s.toFixed(2)})`)))),
    rangeBar(ex, Number.isFinite(sc.gap) ? sc.gap : null),
    curves(r),
    h("div.st-sub", "Gradient reaching each block, first step"),
    gradBars(r),
    h("div.row",
      h("button.btn.small", { type: "button", onclick: () => ctx.overlay(D.grads0, "Gradient at step 1") }, "Colour the 3D blocks by it"),
      h("button.btn.small", { type: "button", onclick: () => ctx.overlay(D.acts, "Stream size at the end") }, "…by stream size"),
      h("button.btn.small", { type: "button", onclick: () => ctx.overlay(null) }, "Clear")),
    h("p.st-small.muted", `Seeds disagree by about ${sc.spread.toFixed(3)}: gaps smaller than that are a tie. Run on ${r.device}${r.quick ? " (quick test mode, not a real result)" : ""}.`),
    h("details", h("summary", "What each one writes"),
      h("p.lab-sample", h("b", `${B.label}: `), B.sample), h("p.lab-sample", h("b", `${D.label}: `), D.sample)));
}

/** Validation loss over training, both runs. */
function curves(r) {
  const W = 320, H = 150, P = { l: 34, r: 8, t: 10, b: 22 };
  const runs = ["baseline", "design"].map((k) => [k, r.runs[k]]).filter(([, v]) => v);
  const pts = runs.flatMap(([, v]) => v.curve.filter((p) => Number.isFinite(p[1])));
  const xMax = Math.max(...pts.map((p) => p[0])), ys = pts.map((p) => p[1]);
  const y0 = Math.floor(Math.min(...ys) * 10) / 10, y1 = Math.min(Math.ceil(Math.max(...ys) * 10) / 10, y0 + 3);
  const X = (s) => P.l + (s / xMax) * (W - P.l - P.r), Y = (v) => P.t + (1 - (Math.min(v, y1) - y0) / (y1 - y0)) * (H - P.t - P.b);
  const NS = "http://www.w3.org/2000/svg", el = (t, a, ...k) => { const e = document.createElementNS(NS, t); for (const [x, v] of Object.entries(a)) e.setAttribute(x, v); e.append(...k); return e; };
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "lab-chart", role: "img", "aria-label": "Validation loss during training for both runs" });
  for (let i = 0; i <= 3; i++) { const v = y0 + ((y1 - y0) * i) / 3; svg.append(el("line", { x1: P.l, x2: W - P.r, y1: Y(v), y2: Y(v), class: "g" }), el("text", { x: P.l - 5, y: Y(v) + 3, "text-anchor": "end" }, v.toFixed(1))); }
  svg.append(el("text", { x: W - P.r, y: H - 6, "text-anchor": "end" }, `step ${xMax.toLocaleString("en-US")}`), el("text", { x: P.l, y: H - 6 }, "0"));
  for (const [k, v] of runs) {
    const c = v.curve.filter((p) => Number.isFinite(p[1]));
    svg.append(el("polyline", { points: c.map((p) => `${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(" "), fill: "none", stroke: COL[k], "stroke-width": 2, "stroke-linejoin": "round" }));
    const last = c[c.length - 1];
    if (last) svg.append(el("circle", { cx: X(last[0]), cy: Y(last[1]), r: 3.5, fill: COL[k], stroke: "var(--surface)", "stroke-width": 2 }));
  }
  const tip = h("div.lab-tip", { hidden: true });
  const wrap = h("div.lab-chart-w", svg, tip, h("div.lab-leg", runs.map(([k, v]) => h("span", h("i", { style: `background:${COL[k]}` }), v.label))));
  svg.addEventListener("pointermove", (e) => {
    const b = svg.getBoundingClientRect(), s = (((e.clientX - b.left) / b.width) * W - P.l) / (W - P.l - P.r) * xMax;
    const at = runs.map(([k, v]) => { const p = v.curve.reduce((a, q) => (Math.abs(q[0] - s) < Math.abs(a[0] - s) ? q : a)); return `${v.label}: ${Number.isFinite(p[1]) ? p[1].toFixed(3) : "—"} at step ${p[0]}`; });
    tip.hidden = false; tip.textContent = at.join(" · "); tip.style.left = `${Math.min(e.clientX - b.left, b.width - 150)}px`;
  });
  svg.addEventListener("pointerleave", () => (tip.hidden = true));
  return wrap;
}

function gradBars(r) {
  const runs = ["baseline", "design"].map((k) => [k, r.runs[k]]).filter(([, v]) => v?.grads0?.length);
  const all = runs.flatMap(([, v]) => v.grads0).filter((x) => x > 0);
  if (!all.length) return h("p.st-small.muted", "No gradient numbers in this result.");
  const lo = Math.log10(Math.min(...all)) - 0.3, hi = Math.log10(Math.max(...all));
  const H = (x) => `${Math.max(3, ((Math.log10(Math.max(x, 1e-12)) - lo) / (hi - lo || 1)) * 100)}%`;
  return h("div.lab-grads", runs.map(([k, v]) => h("div.lab-gr", h("span.lab-gl", v.label),
    h("div.lab-gb", v.grads0.map((x, i) => h("span", { style: `height:${H(x)};background:${COL[k]}`, title: `Block ${i + 1}: ${x.toExponential(1)}` }))))));
}

/** How the rules did on our own runs: the predictor's track record. */
function measuredCard() {
  const base = MEASURED[0].loss;
  return h("div.st-card",
    h("div.eyebrow", "Our track record"),
    h("p.st-small.muted", "Runs we did ourselves at the Quick length: the gap each rule predicted, and the gap we measured. Where a rule missed, we changed it."),
    h("table.lab-t.lab-tr",
      h("thead", h("tr", h("th", "Design"), h("th", "Predicted"), h("th", "Measured"))),
      h("tbody", MEASURED.slice(1).map((m) => {
        const gap = m.loss - base, ok = Math.abs(gap - m.c) <= 0.15 + 0.25 * Math.abs(m.c);
        return h("tr", h("th", m.name, m.note ? h("small", m.note) : null), h("td", sgn(m.c)), h("td", { "data-ok": ok ? "1" : "" }, `${sgn(gap)} ${ok ? "✓" : "✗"}`));
      }))),
    h("p.st-small.muted", `Baseline: ${base.toFixed(2)}. Your own runs count too: paste them above.`));
}

function logCard(ctx) {
  const items = log.all();
  if (!items.length) return h("div.st-card", h("div.eyebrow", "Your experiments"), h("p.st-small.muted", "Designs you copy or download are kept here with their prediction, so you can come back and paste results in."));
  return h("div.st-card",
    h("div.eyebrow", "Your experiments"),
    h("ul.lab-log", items.slice(0, 12).map((e) => h("li",
      h("a", { href: e.url }, e.name), h("span.lab-ls", `${SIZES[e.size]?.label || e.size} · predicted ${e.same ? "tie" : sgn(e.c)}`),
      e.gap != null ? h("span.st-delta", { "data-good": e.held ? "1" : "" }, `measured ${Number.isFinite(e.gap) ? sgn(e.gap) : "diverged"}`) : h("span.lab-ls", "waiting for a result")))),
    h("button.btn.small", { type: "button", onclick: () => { log.clear(); ctx.refresh(); } }, "Clear the list"));
}
