// The Studio: a real model in 3D that you open level by level, change part by part, and get predictions for
// before anything is trained. Three modes share one 3D view and one set of panels:
//   Learn       build a transformer one part at a time, with Neuro and Servo explaining why
//   Challenges  reach a goal (a smaller cache, a phone-sized model) checked live by the predictor
//   Build       start from a real model or a recipe and experiment freely, then test it on Colab
import { h } from "../core.js";
import { topbar } from "../topbar.js";
import { mascot, trophy, burst } from "../pixel.js";
import { Engine, Build } from "./engine.js";
import { PRESETS, full, count, fmt, bytes, draw, crumbs, parse, pathOf, parentOf, heatColor } from "./arch.js";
import { predict, TAGS } from "./predict.js";
import { partEditor, designEditor, TOKMUL } from "./edit.js";
import { speedPane } from "./speed.js";
import { testPane } from "./testpane.js";
import { whyCard, whyKey } from "./why.js";
import { RECIPES } from "./recipes.js";
import { LEVELS, CHALLENGES, progress } from "./learn.js";
import { SIZES } from "./lab.js";

topbar(document.getElementById("top"));
const root = document.getElementById("studio");
const qs = new URLSearchParams(location.search);
const store = { get: (k, d) => { try { return localStorage.getItem(k) ?? d; } catch { return d; } }, set: (k, v) => { try { localStorage.setItem(k, v); } catch { /* private mode */ } } };

// ----- state -----
let mode = ["learn", "ch", "build"].includes(qs.get("mode")) ? qs.get("mode") : qs.get("m") || qs.get("x") ? "build" : store.get("studio-mode", "learn");
let key = PRESETS[qs.get("m")] ? qs.get("m") : "gpt2";
let lv = Math.max(0, LEVELS.findIndex((l) => l.id === qs.get("lv")));
if (!qs.get("lv")) { const done = progress.get(); const next = LEVELS.findIndex((l) => !done[`lv:${l.id}`]); lv = next < 0 ? 0 : next; }
let chId = CHALLENGES.some((c) => c.id === qs.get("c")) ? qs.get("c") : null;
let tokMul = TOKMUL.includes(+qs.get("tok")) ? +qs.get("tok") : 20;
let base = baseFor(), cfg = { ...base, ...readDiff(qs.get("x")) };
let path = pathOf(parse(cfg, qs.get("at") || ""));
let build = null, engine = null, sel = 0, busy = false, labels = [], tab = "inspect";
let labSize = SIZES[qs.get("lab")] ? qs.get("lab") : "S", result = null, heat = null;
let flags = new Set(), stepsDone = 0, solved = false;
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

function baseFor() {
  if (mode === "learn") return full(LEVELS[lv].start);
  if (mode === "ch" && chId) return full(CHALLENGES.find((c) => c.id === chId).start);
  return full(PRESETS[key]);
}

// A design is the starting model plus what you changed, kept in the URL so it can be shared.
function readDiff(x) { try { return x ? JSON.parse(decodeURIComponent(escape(atob(x.replace(/-/g, "+").replace(/_/g, "/"))))) : {}; } catch { return {}; } }
function diff() {
  const d = {};
  for (const k of Object.keys(cfg)) if (JSON.stringify(cfg[k]) !== JSON.stringify(base[k])) d[k] = cfg[k];
  return Object.keys(d).length ? d : null;
}
const writeDiff = (d) => btoa(unescape(encodeURIComponent(JSON.stringify(d)))).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
function urlFor(p = path) {
  const d = diff(), q = new URLSearchParams();
  if (mode === "build") q.set("m", key); else q.set("mode", mode);
  if (mode === "learn") q.set("lv", LEVELS[lv].id);
  if (mode === "ch" && chId) q.set("c", chId);
  if (p) q.set("at", p);
  if (d) q.set("x", writeDiff(d));
  if (tokMul !== 20) q.set("tok", tokMul);
  if (labSize !== "S") q.set("lab", labSize);
  return `${location.pathname}?${q}`;
}

// ----- layout -----
const canvas = h("canvas.st-canvas", { tabindex: "0", "aria-label": "3D model. Drag to turn, scroll to zoom, click a part to select it, double-click to open it. Arrow keys turn, Enter opens, Escape goes back out." });
const labelBox = h("div.st-labels", { "aria-hidden": "true" });
const crumbBox = h("nav.st-crumbs", { "aria-label": "Where you are" });
const alertBox = h("div.st-alert", { role: "alert", hidden: true });
const heatBox = h("div.st-heat", { hidden: true });
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
const hud = h("div.st-hud", { "aria-live": "polite" });
const blockBtns = h("div.st-blocks",
  h("button.btn.small", { type: "button", title: "Add a block on top", onclick: () => set({ L: Math.min(128, cfg.L + 1) }) }, "+ Block"),
  h("button.btn.small", { type: "button", title: "Remove the top block", onclick: () => cfg.L > 1 && set({ L: cfg.L - 1 }) }, "− Block"));
const view = h("div.st-view", canvas, labelBox, crumbBox, alertBox, heatBox, tip, tools, hud, blockBtns,
  h("div.st-hint", "Drag to turn · right-drag to pan · scroll to zoom · double-click to open"));

const modeCard = h("div.st-mode");
const hereTitle = h("h2.st-h"), hereBlurb = h("p.st-blurb"), inside = h("div.st-inside");
const infoBody = h("div"), editHost = h("div"), whyHost = h("div");
const TABS = { inspect: "Inspect", predict: "Predict", speed: "Speed", test: "Test", design: "Design" };
const panes = {
  inspect: h("div.st-pane",
    h("div.st-card.st-info", { "aria-live": "polite" }, infoBody, editHost, whyHost),
    h("div.st-card", h("div.eyebrow", "You are looking at"), hereTitle, hereBlurb, h("div.st-sub", "Open"), inside)),
  predict: h("div.st-pane"), speed: h("div.st-pane"), test: h("div.st-pane"), design: h("div.st-pane"),
};
const tabBtns = {};
const tabBar = h("div.st-tabs", { role: "tablist", "aria-label": "Panels" }, Object.entries(TABS).map(([t, label]) =>
  (tabBtns[t] = h("button", { type: "button", role: "tab", "aria-selected": String(t === tab), onclick: () => showTab(t) }, label))));
const side = h("aside.st-side", modeCard, tabBar, ...Object.values(panes));

const modeBtns = {};
const MODES = [
  ["learn", "Learn", "Build one part at a time", () => mascot("neuro", "happy", { label: "" })],
  ["ch", "Challenges", "Reach a goal", () => trophy({ label: "" })],
  ["build", "Build", "Free design + Colab", () => mascot("servo", "happy", { label: "" })],
];
const modeBar = h("nav.st-modes", { "aria-label": "Studio mode" }, MODES.map(([m, label, sub, art]) =>
  (modeBtns[m] = h("button", { type: "button", "aria-pressed": String(m === mode), onclick: () => setMode(m) }, h("span.st-mart", { html: art() }), h("span", h("b", label), h("small", sub))))));

root.append(
  h("div.st-head",
    h("div.st-headl", h("div.eyebrow", "Studio"), h("h1", "Architecture Studio"),
      h("p.lede", "A real transformer in 3D, at its true proportions. Turn it, click any part to see what it is, go inside, and change it: heads, widths, norms, residuals, block by block. The Predict tab tells you what should happen before anything is trained, and how sure that is; the Test tab checks it on Colab.")),
    modeBar),
  h("div.st-app", view, side));

// The app fills the rest of the screen: the 3D view and the side panel scroll on their own, not the page.
const appEl = root.querySelector(".st-app");
const fitApp = () => appEl.style.setProperty("--st-off", `${Math.round(appEl.getBoundingClientRect().top + scrollY) + 14}px`);
fitApp(); addEventListener("resize", fitApp);

// ----- tabs -----
function showTab(t) {
  tab = t; flags.add(`tab:${t}`);
  for (const [k, b] of Object.entries(tabBtns)) b.setAttribute("aria-selected", String(k === t));
  for (const [k, p] of Object.entries(panes)) p.hidden = k !== t;
  if (t === "inspect") { showInfo(); showEditor(); }
  if (t === "design") panes.design.replaceChildren(designPane());
  if (t === "predict") showPredict();
  if (t === "speed") showSpeed();
  if (t === "test") showTest();
  side.scrollTop = 0;
  checkMode();
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

canvas.addEventListener("pointerdown", () => view.classList.add("touched"), { once: true });
canvas.addEventListener("keydown", (e) => {
  if (!engine) return;
  const K = { ArrowLeft: () => engine.orbit(0.2, 0), ArrowRight: () => engine.orbit(-0.2, 0), ArrowUp: () => engine.orbit(0, 0.12), ArrowDown: () => engine.orbit(0, -0.12),
    "+": () => engine.zoomAt(0.85), "=": () => engine.zoomAt(0.85), "-": () => engine.zoomAt(1.18), f: () => fit(380),
    Enter: () => { const m = build.meta[sel]; if (m?.open) go(m.open, { from: sel }); }, Escape: () => up(), Backspace: () => up() };
  if (K[e.key]) { e.preventDefault(); K[e.key](); }
});

// ----- rendering a level of the model -----
/** Rebuild the 3D scene. keep: this is an edit, so hold on to the selection and don't touch the editor. */
function render(keep = false) {
  const oldName = keep ? build?.meta[sel]?.name : null;
  path = pathOf(parse(cfg, path));
  flags.add(`at:${path}`);
  build = new Build();
  const v = draw(cfg, path, build, { heat: path === "" ? heat : null });
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
  heatBox.hidden = !heat || path !== "";
  blockBtns.hidden = path !== "";
  lastCam = "";
  showInfo();
  if (!keep) showEditor();
  history.replaceState(null, "", urlFor());
  document.title = `${crumbs(cfg, path).map((c) => c.label).reverse().join(" · ")} · Studio · AI Anatomy`;
}

const P = () => predict(cfg, base, tokMul);
/** Everything that depends on the design but not on where you are. */
function refreshNumbers() {
  const t = count(cfg), p = P();
  hud.replaceChildren(
    h("span", h("small", "Parameters"), h("b", fmt(t.total))), h("span", h("small", "Blocks"), h("b", String(cfg.L))),
    h("span", h("small", "Width"), h("b", cfg.d.toLocaleString("en-US"))), h("span", h("small", "Weights"), h("b", bytes(2 * t.total))),
    Number.isFinite(p.num.loss) ? h("span", h("small", "Loss (law)"), h("b", p.num.loss.toFixed(2))) : null);
  alertBox.hidden = !p.errors.length;
  alertBox.textContent = p.errors.length ? `Shape error: ${p.errors[0].say}` : "";
  const worry = p.rules.filter((r) => r.level !== "info").length + p.errors.length;
  tabBtns.predict.replaceChildren(...["Predict", worry ? h("span.st-badge", { "data-bad": p.errors.length || p.rules.some((r) => r.level === "bad") ? "1" : "" }, String(worry)) : null].filter(Boolean));
  if (tab === "predict") showPredict(p);
  if (tab === "speed") showSpeed();
  if (tab === "test") showTest();
  if (mode === "build") buildCard();
  checkMode(p);
}

/** Change the design. Redraws the scene and numbers; editors stay put so sliders keep working. */
function set(patch) {
  const before = build ? JSON.stringify(build.bounds().map((v) => Math.round(v))) : "";
  cfg = { ...cfg, ...patch };
  if (patch.L != null) cfg.blocks = Object.fromEntries(Object.entries(cfg.blocks || {}).filter(([b]) => +b <= cfg.L));
  if (patch.L != null || patch.blocks) heat = null;
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
  const { w, h: hh, data } = engine.ids(), n = build.meta.length;
  const cnt = new Float64Array(n), sx = new Float64Array(n), sy = new Float64Array(n);
  for (let y = 0; y < hh; y++) for (let x = 0; x < w; x++) {
    const i = (y * w + x) * 4; if (!data[i + 3]) continue;
    const id = data[i] | (data[i + 1] << 8) | (data[i + 2] << 16); if (id >= n) continue;
    cnt[id]++; sx[id] += x; sy[id] += hh - 1 - y;
  }
  const at = (x, y) => { const lx = Math.floor((x / engine.cssW) * w), ly = hh - 1 - Math.floor((y / engine.cssH) * hh); if (lx < 0 || ly < 0 || lx >= w || ly >= hh) return 0; const i = (ly * w + lx) * 4; return data[i + 3] ? data[i] | (data[i + 1] << 8) | (data[i + 2] << 16) : 0; };
  const placed = [], order = [...labels].sort((a, b) => (b.id === sel) - (a.id === sel));
  for (const l of order) {
    const b = build.bounds(l.id);
    let [x, y] = engine.project([(b[0] + b[3]) / 2, (b[1] + b[4]) / 2, b[5]]), above = true;
    if (at(x, y + 3) !== l.id) {          // the top is hidden: put the label on the middle of what is visible
      above = false;
      if (cnt[l.id] < 12) { l.el.classList.add("hide"); continue; }
      x = ((sx[l.id] / cnt[l.id] + 0.5) / w) * engine.cssW; y = ((sy[l.id] / cnt[l.id] + 0.5) / hh) * engine.cssH;
    }
    const lw = l.w || (l.w = l.el.offsetWidth || 60), top = above ? y - 26 : y - 11, r = [x - lw / 2 - 2, top, x + lw / 2 + 2, top + 22];
    const clash = r[0] < 0 || r[2] > engine.cssW - 50 || r[1] < 40 || r[3] > engine.cssH - 58 || placed.some((o) => r[0] < o[2] && r[2] > o[0] && r[1] < o[3] && r[3] > o[1]);
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
  checkMode();
}

function showInfo() {
  const m = build.meta[sel], kind = parse(cfg, path).kind;
  if (!m) {
    const k = count(cfg);
    infoBody.replaceChildren(h("div.eyebrow", "Selected"), h("p.muted", "Nothing yet. Click a part of the model to read about it and change it."),
      h("p.st-small", `${cfg.name}: ${cfg.L} blocks, ${cfg.heads} heads, ${cfg.norm === "None" ? "no norm" : cfg.norm}, ${cfg.act}, ${cfg.pos === "RoPE" ? "RoPE" : cfg.pos === "none" ? "no" : "learned"} positions${cfg.kv < cfg.heads ? `, ${cfg.kv} K/V groups` : ""}. ${fmt(k.total)} parameters.`));
    whyHost.replaceChildren(...[whyCard(whyKey(null, kind))].filter(Boolean));
    return;
  }
  infoBody.replaceChildren(
    h("div.eyebrow", "Selected"), h("h3.st-h", m.name), h("p", m.what),
    m.rows?.length ? h("dl.st-rows", m.rows.flatMap(([a, b]) => [h("dt", a), h("dd", b)])) : null,
    m.open ? h("button.btn.primary.small", { type: "button", onclick: () => go(m.open, { from: sel }) }, `Open ${m.name} ›`) : null);
  whyHost.replaceChildren(...[whyCard(whyKey(m, kind))].filter(Boolean));
}
function showEditor() {
  const ed = partEditor(build.meta[sel]?.edit, get, set);
  editHost.replaceChildren(...(ed ? [h("div.st-sub", "Change it"), ed] : []));
}

function showPredict(p = P()) {
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
      h("button.btn.small", { type: "button", onclick: () => showTab("test") }, "Test it on Colab ›")),
    h("div.st-card.st-legend", h("div.eyebrow", "How sure"), Object.entries(TAGS).map(([k, t]) => h("p", tagChip(k), " ", t.say)))].filter(Boolean));
}

function showSpeed() { panes.speed.replaceChildren(speedPane(cfg, (patch) => { set(patch); showSpeed(); })); }

function showTest() {
  panes.test.replaceChildren(testPane({
    cfg, base, size: labSize, url: urlFor(), result,
    names: { base: base.name === "Your model" ? "Level start" : base.name, design: diff() ? "Your design" : `${base.name} again` },
    setSize: (s) => { labSize = s; history.replaceState(null, "", urlFor()); showTest(); },
    setResult: (r) => { result = r; showTest(); side.scrollTop = 0; burst(panes.test.querySelector(".lab-res")); },
    overlay: (values, label) => { heat = values ? { values, label } : null; setHeatLegend(label); if (path !== "") go("", { up: true }); else render(true); },
    flag: (f) => { flags.add(f); checkMode(); },
    refresh: () => showTest(),
  }));
}
function setHeatLegend(label) {
  heatBox.replaceChildren(...(heat ? [h("div.st-heat-t", h("b", label), h("button", { type: "button", "aria-label": "Clear the colours", onclick: () => { heat = null; render(true); } }, "×")),
    h("div.st-heat-k", h("small", "small"), h("span.st-heat-bar", { style: `background:linear-gradient(90deg, ${heatColor(0)}, ${heatColor(0.5)}, ${heatColor(1)})` }), h("small", "big")),
    h("small.muted", "Measured in your lab run, log scale")] : []));
}

function designPane() {
  return h("div.st-pane-in",
    h("div.st-card",
      h("div.eyebrow", "The whole model"),
      h("p.st-small.muted", "Click parts in 3D to change them one at a time. These change the whole model."),
      designEditor(get, set, { get: () => tokMul, set: (v) => { tokMul = v; refreshNumbers(); render(true); } }),
      h("button.btn.small", { type: "button", onclick: () => { reset(); showTab("design"); } }, `Back to ${base.name}`)),
    h("div.st-card",
      h("div.eyebrow", "Recipes"),
      h("p.st-small.muted", "What famous models changed, applied to yours. Each says why it works and what it costs."),
      h("ul.st-recipes", RECIPES.map((r) => h("li",
        h("div.st-pc-top", h("span", h("b", r.name), h("small", r.who)), h("button.btn.small", { type: "button", onclick: () => { set(r.patch(full(cfg))); flags.add(`recipe:${r.id}`); panes.design.replaceChildren(designPane()); } }, "Apply")),
        h("details", h("summary", "Why it works"), h("p", r.why), h("p.muted", h("b", "Cost: "), r.cost)))))));
}

// ----- modes -----
function buildCard() {
  const d = diff(), n = Object.keys(d || {}).length;
  modeCard.replaceChildren(h("div.st-card.st-buildc",
    h("label.st-pick", h("span", "Start from"),
      h("select", { onchange: (e) => switchModel(e.target.value) }, Object.entries(PRESETS).map(([k, p]) => h("option", { value: k, selected: k === key }, p.name)))),
    h("p.st-small.muted", PRESETS[key].note),
    d ? h("div.st-edited", h("span", `Edited · ${n} change${n === 1 ? "" : "s"}`), h("button.btn.small", { type: "button", onclick: () => switchModel(key) }, "Undo all")) : null));
}

function levelCard() {
  const L = LEVELS[lv], done = progress.get(), p = P(), s = session(p);
  while (stepsDone < L.steps.length && L.steps[stepsDone].ok(cfg, p, s)) stepsDone++;
  const finished = stepsDone >= L.steps.length;
  modeCard.replaceChildren(h("div.st-card.st-level", { "data-done": finished ? "1" : "" },
    h("div.st-lvmap", { role: "list", "aria-label": "Levels" }, LEVELS.map((l, i) => h("button", { type: "button", role: "listitem", title: l.title, "aria-current": i === lv ? "step" : null, "data-done": done[`lv:${l.id}`] ? "1" : "", onclick: () => loadLevel(i) }, String(i + 1)))),
    h("div.st-lvhead", h("span.eyebrow", `Level ${lv + 1} of ${LEVELS.length}`), h("h2.st-h", L.title)),
    h("div.st-say", h("div.st-say-m", { html: mascot("neuro", finished ? "wow" : "happy", { label: "Neuro" }) }), h("p", L.neuro)),
    h("ol.st-steps", L.steps.map((st, i) => h("li", { "data-state": i < stepsDone ? "done" : i === stepsDone ? "now" : "" }, st.do))),
    finished ? h("div.st-can", h("b", "What it can do now"), h("p", L.can), L.sample ? h("pre.st-sample", L.sample) : null, L.sample ? h("small.muted", "Typical text from a model like this, trained on Shakespeare.") : null) :
      h("div.st-say.servo", h("div.st-say-m", { html: mascot("servo", "happy", { label: "Servo" }) }), h("p", h("b", "Tip: "), L.servo)),
    h("div.row.st-lvnav",
      lv > 0 ? h("button.btn.small", { type: "button", onclick: () => loadLevel(lv - 1) }, "‹ Back") : null,
      h("button.btn.small", { type: "button", onclick: () => loadLevel(lv) }, "Restart"),
      lv < LEVELS.length - 1 ? h(`button.btn.small${finished ? ".primary" : ""}`, { type: "button", onclick: () => loadLevel(lv + 1) }, finished ? "Next level ›" : "Skip ›") :
        finished ? h("button.btn.small.primary", { type: "button", onclick: () => setMode("ch") }, "Try the challenges ›") : null)));
  if (finished && !done[`lv:${L.id}`]) { progress.mark(`lv:${L.id}`); levelCard(); burst(modeCard.querySelector(".st-can")); }
}

function chCard() {
  const done = progress.get();
  if (!chId) {
    modeCard.replaceChildren(h("div.st-card",
      h("div.eyebrow", "Challenges"), h("p.st-small.muted", "Each one is a real design problem. The goals are checked live by the predictor as you edit."),
      h("div.st-chs", CHALLENGES.map((c) => h("button.st-ch", { type: "button", "data-done": done[`ch:${c.id}`] ? "1" : "", onclick: () => loadChallenge(c.id) },
        h("span.st-ch-l", c.level), h("b", c.title), h("small", c.brief.split(". ")[0] + "."), done[`ch:${c.id}`] ? h("span.st-ch-ok", "✓ solved") : null)))));
    return;
  }
  const C = CHALLENGES.find((c) => c.id === chId), p = P(), s = session(p);
  const res = C.goals.map((g) => ({ g, ok: g.ok(cfg, p, s), now: g.now(cfg, p, s) }));
  const all = res.every((r) => r.ok);
  modeCard.replaceChildren(h("div.st-card.st-chal", { "data-done": all ? "1" : "" },
    h("div.st-pc-top", h("span.eyebrow", `Challenge · ${C.level}`), h("button.btn.small", { type: "button", onclick: () => { chId = null; switchTo(); } }, "All challenges")),
    h("h2.st-h", C.title), h("p.st-small", C.brief),
    h("ul.st-goals", res.map((r) => h("li", { "data-ok": r.ok ? "1" : "" }, h("span.st-gk", r.ok ? "✓" : "○"), h("span", r.g.say), h("b", r.now)))),
    all ? h("div.st-can", h("div.st-say", h("div.st-say-m", { html: trophy({ label: "Solved" }) }), h("p", h("b", "Solved. "), C.solved))) :
      h("details.st-hint2", h("summary", "Hint from Servo"), h("div.st-say.servo", h("div.st-say-m", { html: mascot("servo", "happy", { label: "Servo" }) }), h("p", C.hint))),
    h("div.row", h("button.btn.small", { type: "button", onclick: () => loadChallenge(chId) }, "Start over"))));
  if (all && !solved) { solved = true; if (!done[`ch:${C.id}`]) progress.mark(`ch:${C.id}`); burst(modeCard.querySelector(".st-can")); }
  if (!all) solved = false;
}

function session(p) { return { sel: build?.meta[sel]?.name || "", path, tab, flags, tok: tokMul, start: predict(base, base, startTok()).num }; }
const startTok = () => (mode === "ch" && chId ? CHALLENGES.find((c) => c.id === chId).tok || 20 : 20);
function checkMode() { if (mode === "learn") levelCard(); else if (mode === "ch") chCard(); else buildCard(); }

function setMode(m) {
  if (m === mode && m !== "ch") return;
  mode = m; store.set("studio-mode", m);
  if (m === "ch" && modeBtns.ch.getAttribute("aria-pressed") === "true") chId = null;
  for (const [k, b] of Object.entries(modeBtns)) b.setAttribute("aria-pressed", String(k === m));
  switchTo();
}
function switchTo() {
  base = baseFor(); cfg = { ...base }; path = ""; heat = null; result = null; flags = new Set(); stepsDone = 0; solved = false;
  tokMul = mode === "ch" && chId ? startTok() : 20;
  render(); refreshNumbers(); fit(0); showTab(tab === "design" || tab === "inspect" ? tab : "inspect");
}
function loadLevel(i) { lv = i; switchTo(); }
function loadChallenge(id) { chId = id; switchTo(); }
function reset() { if (mode === "build") switchModel(key); else switchTo(); }

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
  checkMode();
}
function up() { if (path) go(parentOf(path), { up: true }); }

function switchModel(k) {
  key = k; base = full(PRESETS[k]); cfg = { ...base }; heat = null; result = null;
  render(); refreshNumbers(); fit(0);
  if (tab === "design") panes.design.replaceChildren(designPane());
}

render(); refreshNumbers(); showTab("inspect");
requestAnimationFrame(() => fit(0));
