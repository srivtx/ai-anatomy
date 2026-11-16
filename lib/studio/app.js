// The Studio page: a real model in 3D that you open level by level.
import { h } from "../core.js";
import { topbar } from "../topbar.js";
import { segmented, stats } from "../widgets.js";
import { Engine, Build } from "./engine.js";
import { PRESETS, count, fmt, bytes, draw, crumbs, parse, pathOf, parentOf } from "./arch.js";

topbar(document.getElementById("top"));
const root = document.getElementById("studio");
const qs = new URLSearchParams(location.search);
let key = PRESETS[qs.get("m")] ? qs.get("m") : "gpt2";
let cfg = PRESETS[key];
let path = pathOf(parse(cfg, qs.get("at") || ""));
let build = null, engine = null, sel = 0, busy = false, labels = [];
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

// ----- layout -----
const canvas = h("canvas.st-canvas", { tabindex: "0", "aria-label": "3D model. Drag to turn, scroll to zoom, click a part to select it, double-click to open it. Arrow keys turn, Enter opens, Escape goes back out." });
const labelBox = h("div.st-labels", { "aria-hidden": "true" });
const crumbBox = h("nav.st-crumbs", { "aria-label": "Where you are" });
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
const view = h("div.st-view", canvas, labelBox, crumbBox, tip, tools,
  h("div.st-hint", "Drag to turn · right-drag to pan · scroll to zoom · double-click to open"));

const st = stats([{ label: "Parameters" }, { label: "Blocks" }, { label: "Width" }, { label: "Weights, 16-bit" }]);
const picker = segmented({ label: "Model", options: Object.entries(PRESETS).map(([k, p]) => ({ value: k, label: p.name })), value: key, onChange: (k) => switchModel(k) });
const hereTitle = h("h2.st-h"), hereBlurb = h("p.st-blurb"), inside = h("div.st-inside");
const info = h("div.st-card.st-info", { "aria-live": "polite" });
const side = h("aside.st-side",
  h("div.st-card", picker, st.el),
  h("div.st-card", h("div.eyebrow", "You are looking at"), hereTitle, hereBlurb, h("div.st-sub", "Open"), inside),
  info);

root.append(
  h("div.st-head", h("div.eyebrow", "Studio"), h("h1", "Open up a model"), h("p.lede", "A real transformer, drawn to its true proportions in our sticker style. Turn it, click any part to see what it is and how big it is, and go inside: model, block, attention, a single head. Every number is the real one.")),
  h("div.st-app", view, side));

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
  view.append(h("div.st-no3d", h("b", "3D isn't available in this browser."), " You can still explore: the parts are listed on the right."));
}

canvas.addEventListener("keydown", (e) => {
  if (!engine) return;
  const K = { ArrowLeft: () => engine.orbit(0.2, 0), ArrowRight: () => engine.orbit(-0.2, 0), ArrowUp: () => engine.orbit(0, 0.12), ArrowDown: () => engine.orbit(0, -0.12),
    "+": () => engine.zoomAt(0.85), "=": () => engine.zoomAt(0.85), "-": () => engine.zoomAt(1.18), f: () => fit(380),
    Enter: () => { const m = build.meta[sel]; if (m?.open) go(m.open, { from: sel }); }, Escape: () => up(), Backspace: () => up() };
  if (K[e.key]) { e.preventDefault(); K[e.key](); }
});

// ----- rendering a level -----
function render() {
  build = new Build();
  const v = draw(cfg, path, build);
  sel = 0;
  engine?.setScene(build);
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
  history.replaceState(null, "", `?m=${key}${path ? `&at=${path}` : ""}`);
  document.title = `${crumbs(cfg, path).map((c) => c.label).reverse().join(" · ")} · Studio · AI Anatomy`;
}

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
  showInfo();
}

function showInfo() {
  const m = build.meta[sel];
  if (!m) {
    const k = count(cfg);
    info.replaceChildren(h("div.eyebrow", "Selected"), h("p.muted", "Nothing yet. Click a part of the model."),
      h("p.st-small", `${cfg.name}: ${cfg.L} blocks, ${cfg.heads} heads, ${cfg.norm}, ${cfg.act}, ${cfg.pos === "RoPE" ? "RoPE positions" : "learned positions"}${cfg.kv < cfg.heads ? `, ${cfg.kv} K/V groups` : ""}. ${fmt(k.total)} parameters.`));
    return;
  }
  info.replaceChildren(
    h("div.eyebrow", "Selected"), h("h3.st-h", m.name), h("p", m.what),
    m.rows?.length ? h("dl.st-rows", m.rows.flatMap(([a, b]) => [h("dt", a), h("dd", b)])) : null,
    m.open ? h("button.btn.primary.small", { type: "button", onclick: () => go(m.open, { from: sel }) }, `Open ${m.name} ›`) : null);
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
  key = k; cfg = PRESETS[k];
  path = pathOf(parse(cfg, path)) || "";
  const t = count(cfg);
  st.set(0, fmt(t.total)); st.set(1, String(cfg.L)); st.set(2, cfg.d.toLocaleString("en-US")); st.set(3, bytes(2 * t.total));
  render(); fit(0);
}

switchModel(key);
requestAnimationFrame(() => fit(0));
