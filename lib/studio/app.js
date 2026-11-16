// The Studio page: a real model in 3D that you open level by level, change part by part, and get
// predictions for before anything is trained.
import { h } from "../core.js";
import { topbar } from "../topbar.js";
import { segmented, stats } from "../widgets.js";
import { Engine, Build } from "./engine.js";
import { PRESETS, full, count, fmt, bytes, draw, crumbs, parse, pathOf, parentOf } from "./arch.js";
import { predict, TAGS } from "./predict.js";
import { partEditor, designEditor, TOKMUL } from "./edit.js";

topbar(document.getElementById("top"));
const root = document.getElementById("studio");
const qs = new URLSearchParams(location.search);
let key = PRESETS[qs.get("m")] ? qs.get("m") : "gpt2";
let base = full(PRESETS[key]), cfg = { ...base, ...readDiff(qs.get("x")) };
let tokMul = TOKMUL.includes(+qs.get("tok")) ? +qs.get("tok") : 20;
let path = pathOf(parse(cfg, qs.get("at") || ""));
let build = null, engine = null, sel = 0, busy = false, labels = [], tab = "inspect";
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

// A design is the preset plus what you changed, kept in the URL so it can be shared.
function readDiff(x) { try { return x ? JSON.parse(decodeURIComponent(escape(atob(x.replace(/-/g, "+").replace(/_/g, "/"))))) : {}; } catch { return {}; } }
function diff() {
  const d = {};
  for (const k of Object.keys(cfg)) if (JSON.stringify(cfg[k]) !== JSON.stringify(base[k])) d[k] = cfg[k];
  return Object.keys(d).length ? d : null;
}
const writeDiff = (d) => btoa(unescape(encodeURIComponent(JSON.stringify(d)))).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");

// ----- layout -----
const canvas = h("canvas.st-canvas", { tabindex: "0", "aria-label": "3D model. Drag to turn, scroll to zoom, click a part to select it, double-click to open it. Arrow keys turn, Enter opens, Escape goes back out." });
const labelBox = h("div.st-labels", { "aria-hidden": "true" });
const crumbBox = h("nav.st-crumbs", { "aria-label": "Where you are" });
const alertBox = h("div.st-alert", { role: "alert", hidden: true });
const tip = h("div.st-tip", { hidden: true });
const pixBtn = h("button.btn.small", { type: "button", title: "Pixel size: big or fine", "aria-label": "Pixel size: big or fine", "aria-pressed": "true", onclick: () => { const px = engine?.px === 2 ? 1 : 2; engine?.setPixel(px); pixBtn.setAttribute("aria-pressed", String(px === 2)); } }, "▦");
const tool = (label, title, fn) => h("button.btn.small", { type: "button", title, "aria-label": title, onclick: fn }, label);
const tools = h("div.st-tools",
  tool("⟲", "Turn left", () => engine?.fly({ yaw: engine.cam.yaw - Math.PI / 2 }, 380)),
  tool("⟳", "Turn right", () => engine?.fly({ yaw: engine.cam.yaw + Math.PI / 2 }, 380)),
  tool("−", "Zoom out", () => engine?.zoomAt(1.25)),
  tool("+", "Zoom in", () => engine?.zoomAt(0.8)),
  tool("Fit", "Reset the view", () => fit(380)),
  pixBtn);
const view = h("div.st-view", canvas, labelBox, crumbBox, alertBox, tip, tools,
  h("div.st-hint", "Drag to turn · right-drag to pan · scroll to zoom · double-click to open"));

const st = stats([{ label: "Parameters" }, { label: "Blocks" }, { label: "Width" }, { label: "Weights, 16-bit" }]);
const picker = segmented({ label: "Start from", options: Object.entries(PRESETS).map(([k, p]) => ({ value: k, label: p.name })), value: key, onChange: (k) => switchModel(k) });
const edited = h("div.st-edited", { hidden: true });
const hereTitle = h("h2.st-h"), hereBlurb = h("p.st-blurb"), inside = h("div.st-inside");
const infoBody = h("div"), editHost = h("div");
const panes = {
  inspect: h("div.st-pane",
    h("div.st-card", h("div.eyebrow", "You are looking at"), hereTitle, hereBlurb, h("div.st-sub", "Open"), inside),
    h("div.st-card.st-info", { "aria-live": "polite" }, infoBody, editHost)),
  predict: h("div.st-pane"),
  design: h("div.st-pane"),
};
const tabBtns = {};
const tabBar = h("div.st-tabs", { role: "tablist", "aria-label": "Panels" }, ["inspect", "predict", "design"].map((t) =>
  (tabBtns[t] = h("button", { type: "button", role: "tab", "aria-selected": String(t === tab), onclick: () => showTab(t) }, { inspect: "Inspect", predict: "Predict", design: "Design" }[t]))));
const side = h("aside.st-side", h("div.st-card", picker, edited, st.el), tabBar, ...Object.values(panes));

root.append(
  h("div.st-head", h("div.eyebrow", "Studio"), h("h1", "Open up a model, then change it"),
    h("p.lede", "A real transformer in 3D, at its true proportions. Turn it, click any part to see what it is, go inside, and change it: heads, widths, norms, residuals, block by block. The Predict tab tells you what should happen before anything is trained, and how sure that is.")),
  h("div.st-app", view, side));

function showTab(t) {
  tab = t;
  for (const [k, b] of Object.entries(tabBtns)) b.setAttribute("aria-selected", String(k === t));
  for (const [k, p] of Object.entries(panes)) p.hidden = k !== t;
  if (t === "inspect") { showInfo(); showEditor(); }
  if (t === "design") panes.design.replaceChildren(designPane());
  if (t === "predict") showPredict();
}

// ----- engine -----
try {
  engine = new Engine(canvas, {
    hover(id, x, y) {
      canvas.classList.toggle("hov", !!id);
      if (!id) { tip.hidden = true; return; }
      const m = build.meta[id];
      tip.hidden = false; tip.textContent = m.open ? `${m.name} · double-click to open` : m.name;
      const w = view.clientWidth; tip.style.left = `${Math.min(x + 14, w - tip.offsetWidth - 8)}px`; tip.style.top = `${y + 16}px`;
    },
    click(id) { select(id); },
    open(id) { const m = build.meta[id]; if (m?.open) go(m.open, { from: id }); else select(id); },
    frame: placeLabels,
  });
  const accent = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim();
  if (/^#[0-9a-f]{6}$/i.test(accent)) engine.setTheme({ sel: accent });
} catch {
  view.classList.add("no3d");
  view.append(h("div.st-no3d", h("b", "3D isn't available in this browser."), " You can still explore and edit: the parts are listed on the right."));
}

canvas.addEventListener("keydown", (e) => {
  if (!engine) return;
  const K = { ArrowLeft: () => engine.orbit(0.2, 0), ArrowRight: () => engine.orbit(-0.2, 0), ArrowUp: () => engine.orbit(0, 0.12), ArrowDown: () => engine.orbit(0, -0.12),
    "+": () => engine.zoomAt(0.85), "=": () => engine.zoomAt(0.85), "-": () => engine.zoomAt(1.18), f: () => fit(380),
    Enter: () => { const m = build.meta[sel]; if (m?.open) go(m.open, { from: sel }); }, Escape: () => up(), Backspace: () => up() };
  if (K[e.key]) { e.preventDefault(); K[e.key](); }
});

// ----- rendering a level -----
/** Rebuild the 3D scene. keep: this is an edit, so hold on to the selection and don't touch the editor. */
function render(keep = false) {
  const oldName = keep ? build?.meta[sel]?.name : null;
  path = pathOf(parse(cfg, path));
  build = new Build();
  const v = draw(cfg, path, build);
  sel = oldName ? Math.max(0, build.meta.findIndex((m) => m?.name === oldName)) : 0;
  if (engine) { engine.setScene(build); engine.sel = sel; }
  hereTitle.textContent = v.title; hereBlurb.textContent = v.blurb;
  crumbBox.replaceChildren(...crumbs(cfg, path).flatMap((c, i, a) => [
    ...(i ? [h("span.st-sep", "›")] : []),
    h("button", { type: "button", "aria-current": i === a.length - 1 ? "page" : null, onclick: () => i < a.length - 1 && go(c.path, { up: true }) }, c.label)]));
  const seen = new Set();
  inside.replaceChildren(...build.meta.map((m, id) => (m?.open && !seen.has(m.open) && seen.add(m.open) ? h("button.st-chip", { type: "button", onclick: () => go(m.open, { from: id }) }, m.name) : null)).filter(Boolean));
  if (!inside.children.length) inside.append(h("span.muted.st-small", "Nothing deeper here. Click the parts to read about them."));
  labels = build.meta.map((m, id) => (m?.tag ? { id, el: h("span.st-label", m.tag) } : null)).filter(Boolean);
  labelBox.replaceChildren(...labels.map((l) => l.el));
  lastCam = "";
  showInfo();
  if (!keep) showEditor();
  const d = diff();
  history.replaceState(null, "", `?m=${key}${path ? `&at=${path}` : ""}${d ? `&x=${writeDiff(d)}` : ""}${tokMul !== 20 ? `&tok=${tokMul}` : ""}`);
  document.title = `${crumbs(cfg, path).map((c) => c.label).reverse().join(" · ")} · Studio · AI Anatomy`;
}

/** Everything that depends on the design but not on where you are. */
function refreshNumbers() {
  const t = count(cfg), p = predict(cfg, base, tokMul), d = diff();
  st.set(0, fmt(t.total)); st.set(1, String(cfg.L)); st.set(2, cfg.d.toLocaleString("en-US")); st.set(3, bytes(2 * t.total));
  edited.hidden = !d;
  edited.replaceChildren(h("span", `Edited · ${Object.keys(d || {}).length} change${Object.keys(d || {}).length === 1 ? "" : "s"}`), h("button.btn.small", { type: "button", onclick: () => switchModel(key) }, "Undo all"));
  alertBox.hidden = !p.errors.length;
  alertBox.textContent = p.errors.length ? `Shape error: ${p.errors[0].say}` : "";
  const worry = p.rules.filter((r) => r.level !== "info").length + p.errors.length;
  tabBtns.predict.replaceChildren(...["Predict", worry ? h("span.st-badge", { "data-bad": p.errors.length || p.rules.some((r) => r.level === "bad") ? "1" : "" }, String(worry)) : null].filter(Boolean));
  if (tab === "predict") showPredict(p);
}

/** Change the design. Redraws the scene and numbers; editors stay put so sliders keep working. */
function set(patch) {
  const before = build ? JSON.stringify(build.bounds().map((v) => Math.round(v))) : "";
  cfg = { ...cfg, ...patch };
  if (patch.L != null) cfg.blocks = Object.fromEntries(Object.entries(cfg.blocks).filter(([b]) => +b <= cfg.L));
  render(true); refreshNumbers();
  if (engine && JSON.stringify(build.bounds().map((v) => Math.round(v))) !== before) engine.fly(engine.framing(build.bounds()), 260);
}
const get = () => cfg;

// Labels sit on the part of their object you can actually see. While the camera moves they fade out;
// once it rests, one read of the id buffer tells us where every labelled object is visible.
let lastCam = "", idle = 0;
function placeLabels() {
  if (!engine) return;
  const c = engine.cam, sig = `${c.t.map((v) => v.toFixed(3))}|${c.yaw.toFixed(4)}|${c.pitch.toFixed(4)}|${c.zoom.toFixed(3)}|${engine.cssW}x${engine.cssH}`;
  if (sig !== lastCam) { lastCam = sig; labelBox.classList.add("moving"); clearTimeout(idle); idle = setTimeout(settleLabels, 140); }
  else for (const l of labels) l.el.classList.toggle("sel", l.id === sel);
}
function settleLabels() {
  if (!engine || !build) return;
  const { w, h, data } = engine.ids(), n = build.meta.length;
  const cnt = new Float64Array(n), sx = new Float64Array(n), sy = new Float64Array(n);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = (y * w + x) * 4; if (!data[i + 3]) continue;
    const id = data[i] | (data[i + 1] << 8) | (data[i + 2] << 16); if (id >= n) continue;
    cnt[id]++; sx[id] += x; sy[id] += h - 1 - y;
  }
  const at = (x, y) => { const lx = Math.floor((x / engine.cssW) * w), ly = h - 1 - Math.floor((y / engine.cssH) * h); if (lx < 0 || ly < 0 || lx >= w || ly >= h) return 0; const i = (ly * w + lx) * 4; return data[i + 3] ? data[i] | (data[i + 1] << 8) | (data[i + 2] << 16) : 0; };
  const placed = [], order = [...labels].sort((a, b) => (b.id === sel) - (a.id === sel));
  for (const l of order) {
    const b = build.bounds(l.id);
    let [x, y] = engine.project([(b[0] + b[3]) / 2, (b[1] + b[4]) / 2, b[5]]), above = true;
    if (at(x, y + 3) !== l.id) {          // the top is hidden: put the label on the middle of what is visible
      above = false;
      if (cnt[l.id] < 12) { l.el.classList.add("hide"); continue; }
      x = ((sx[l.id] / cnt[l.id] + 0.5) / w) * engine.cssW; y = ((sy[l.id] / cnt[l.id] + 0.5) / h) * engine.cssH;
    }
    const lw = l.w || (l.w = l.el.offsetWidth || 60), top = above ? y - 26 : y - 11, r = [x - lw / 2 - 2, top, x + lw / 2 + 2, top + 22];
    const clash = r[0] < 0 || r[2] > engine.cssW || r[1] < 40 || placed.some((o) => r[0] < o[2] && r[2] > o[0] && r[1] < o[3] && r[3] > o[1]);
    l.el.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px) translate(-50%, ${above ? "-100%" : "-50%"})`;
    l.el.style.top = above ? "-6px" : "0px";
    l.el.classList.toggle("sel", l.id === sel);
    l.el.classList.toggle("hide", clash);
    if (!clash) placed.push(r);
  }
  labelBox.classList.remove("moving");
}

function select(id) {
  sel = id; if (engine) { engine.sel = id; engine.draw(); }
  showInfo(); showEditor();
  if (id && tab !== "inspect") showTab("inspect");
}

function showInfo() {
  const m = build.meta[sel];
  if (!m) {
    const k = count(cfg);
    infoBody.replaceChildren(h("div.eyebrow", "Selected"), h("p.muted", "Nothing yet. Click a part of the model to read about it and change it."),
      h("p.st-small", `${cfg.name}: ${cfg.L} blocks, ${cfg.heads} heads, ${cfg.norm === "None" ? "no norm" : cfg.norm}, ${cfg.act}, ${cfg.pos === "RoPE" ? "RoPE" : cfg.pos === "none" ? "no" : "learned"} positions${cfg.kv < cfg.heads ? `, ${cfg.kv} K/V groups` : ""}. ${fmt(k.total)} parameters.`));
    return;
  }
  infoBody.replaceChildren(
    h("div.eyebrow", "Selected"), h("h3.st-h", m.name), h("p", m.what),
    m.rows?.length ? h("dl.st-rows", m.rows.flatMap(([a, b]) => [h("dt", a), h("dd", b)])) : null,
    m.open ? h("button.btn.primary.small", { type: "button", onclick: () => go(m.open, { from: sel }) }, `Open ${m.name} ›`) : null);
}
function showEditor() {
  const ed = partEditor(build.meta[sel]?.edit, get, set);
  editHost.replaceChildren(...(ed ? [h("div.st-sub", "Change it"), ed] : []));
}

function showPredict(p = predict(cfg, base, tokMul)) {
  const tagChip = (t) => h("span.st-tag", { "data-tag": t, title: TAGS[t].say }, TAGS[t].label);
  panes.predict.replaceChildren(...[
    p.errors.length ? h("div.st-card.st-errs", h("div.eyebrow", "Can't be built"), p.errors.map((e) => h("p", e.say))) : null,
    h("div.st-card",
      h("div.eyebrow", "What will happen"),
      h("p.st-small.muted", p.errors.length ? "Fix the shape error first: these numbers use shapes that can't exist." : `Compared with ${base.name} where it differs. Hover a tag to see how sure each line is.`),
      h("div.st-cards", p.cards.map((c) => h("div.st-pc",
        h("div.st-pc-top", h("span.st-pc-t", c.title), tagChip(c.tag)),
        h("div.st-pc-v", c.value, c.d ? h("span.st-delta", { "data-good": c.d.good == null ? "n" : c.d.good ? "1" : "" }, c.d.text) : null),
        h("p", c.say))))),
    h("div.st-card",
      h("div.eyebrow", "Watch out for"),
      p.rules.length ? h("ul.st-rules", p.rules.map((r) => h("li", { "data-level": r.level }, h("div.st-pc-top", h("b", r.title), tagChip(r.tag)), h("p", r.why)))) : h("p.st-small.muted", "Nothing unusual. This is a well-trodden design.")),
    h("div.st-card",
      h("div.eyebrow", "Only a real run can tell"), h("ul.st-tests", p.tests.map((t) => h("li", t))),
      h("p.st-small.muted", "Next up: send this design to Colab, train it small, and bring the measured loss back here next to the prediction.")),
    h("div.st-card.st-legend", h("div.eyebrow", "How sure"), Object.entries(TAGS).map(([k, t]) => h("p", tagChip(k), " ", t.say)))].filter(Boolean));
}

function designPane() {
  return h("div.st-card",
    h("div.eyebrow", "The whole model"),
    h("p.st-small.muted", "Click parts in 3D to change them one at a time. These change the whole model."),
    designEditor(get, set, { get: () => tokMul, set: (v) => { tokMul = v; refreshNumbers(); render(true); } }),
    h("button.btn.small", { type: "button", onclick: () => { switchModel(key); showTab("design"); } }, `Back to ${base.name}`));
}

function fit(ms = 0) { if (engine && build) return engine.fly(engine.framing(build.bounds()), ms); }

/** Go to another level: dive into the clicked part, or pull back out to the parent. */
async function go(next, { from = 0, up: isUp = false } = {}) {
  if (busy || next === path) return;
  busy = true; tip.hidden = true;
  const prev = path;
  if (engine && from && !isUp) { await engine.fly(engine.framing(build.bounds(from), { margin: 0.7 }), 320); }
  canvas.classList.add("fading"); if (engine) await wait(110);
  path = next; render();
  if (engine) {
    const target = engine.framing(build.bounds());
    const back = isUp && build.meta.findIndex((m) => m?.open && prev.startsWith(m.open) && m.open !== "");
    if (back > 0) engine.cam = engine.framing(build.bounds(back), { margin: 0.7 });
    else engine.cam = { ...target, zoom: target.zoom * 0.45 };
    canvas.classList.remove("fading");
    await engine.fly(target, 420);
  } else canvas.classList.remove("fading");
  busy = false;
}
function up() { if (path) go(parentOf(path), { up: true }); }

function switchModel(k) {
  if (k !== key || diff()) { key = k; base = full(PRESETS[k]); cfg = { ...base }; }
  picker.value = k;
  render(); refreshNumbers(); fit(0);
  if (tab === "design") panes.design.replaceChildren(designPane());
}

render(); refreshNumbers(); showTab("inspect");
requestAnimationFrame(() => fit(0));
