// The Studio: a real model in 3D that you open level by level, change part by part, and get predictions for
// before anything is trained. Three modes share one 3D view and one set of panels:
//   Learn       build a transformer one part at a time, with Neuro and Servo explaining why
//   Challenges  reach a goal (a smaller cache, a phone-sized model) checked live by the predictor
//   Build       start from a real model or a recipe and experiment freely, then test it on Colab
import { h } from "../core.js";
import { slider } from "../widgets.js";
import { topbar } from "../topbar.js";
import { mascot, trophy, burst } from "../pixel.js";
import { Engine, Build } from "./engine.js";
import { PRESETS, full, count, fmt, bytes, draw, crumbs, parse, pathOf, parentOf, heatColor, blk, nextAt, C as COL } from "./arch.js";
import { predict, TAGS } from "./predict.js";
import { partEditor, designEditor, TOKMUL } from "./edit.js";
import { speedPane } from "./speed.js";
import { testPane } from "./testpane.js";
import { whyCard, whyKey } from "./why.js";
import { RECIPES } from "./recipes.js";
import { LEVELS, CHALLENGES, progress } from "./learn.js";
import { SIZES } from "./lab.js";
import { STICKERS, award, toast, book, earned, icon } from "./stickers.js";

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
let edits = 0, pinned = null, jump = null, histTimer = 0, scan = null;
let temp = 1, picked = -1, samples = [], twin = true;
const GUESS_KEY = "studio-guesses-v1";
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
  pixBtn,
  tool("⤓", "Save a picture of this model", () => savePicture()));
const hud = h("div.st-hud", { "aria-live": "polite" });
const blockBtns = h("div.st-blocks",
  h("button.btn.small", { type: "button", title: "Add a block on top", onclick: () => set({ L: Math.min(128, cfg.L + 1) }) }, "+ Block"),
  h("button.btn.small", { type: "button", title: "Remove the top block", onclick: () => cfg.L > 1 && set({ L: cfg.L - 1 }) }, "− Block"));
const flowBtn = h("button.btn.small.st-play", { type: "button", title: "Watch one token travel through, stop by stop (Space to pause)", onclick: () => flow() }, h("span.st-play-i", "▶"), h("span", "Watch a token"));
const flowCap = h("div.st-flowcap", { hidden: true, "aria-live": "polite" });
const KEYS = [["Drag", "turn"], ["Right-drag", "pan"], ["Scroll / pinch", "zoom"], ["Click", "select"], ["Double-click / Enter", "go inside"], ["Esc", "back out"], ["← → ↑ ↓", "turn"], ["+ −", "zoom"], ["F", "fit"], ["Space", "watch a token / pause"], [", .", "step the token back / on"], ["1 – 5", "switch tabs"], ["?", "this help"]];
const keyHelp = h("div.st-keys", { hidden: true, role: "dialog", "aria-label": "Controls" },
  h("div.st-keys-top", h("b", "Controls"), h("button.btn.small", { type: "button", "aria-label": "Close", onclick: () => (keyHelp.hidden = true) }, "×")),
  h("dl", KEYS.flatMap(([k, v]) => [h("dt", h("kbd", k)), h("dd", v)])));
const keyBtn = h("button.btn.small.st-keybtn", { type: "button", title: "Controls (?)", "aria-label": "Controls", onclick: () => (keyHelp.hidden = !keyHelp.hidden) }, "?");
const view = h("div.st-view", canvas, labelBox, crumbBox, alertBox, heatBox, tip, tools, hud, blockBtns, flowBtn, flowCap, keyHelp, keyBtn,
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

const bookBtn = h("button.st-bookbtn", { type: "button", onclick: () => openBook() });
function paintBook() {
  const n = earned().length;
  bookBtn.replaceChildren(h("span.st-bookbtn-k", "Sticker book"), h("b", `${n}/${STICKERS.length}`), h("span.st-bookbtn-go", "Open ›"));
}
function openBook() { const d = book(() => d.remove()); document.body.append(d); d.querySelector("button")?.focus(); addEventListener("keydown", function esc(e) { if (e.key === "Escape") { d.remove(); removeEventListener("keydown", esc); } }); }
paintBook();
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
    h("div.st-headl", h("div.st-eyerow", h("div.eyebrow", "Studio"), bookBtn), h("h1", "Architecture Studio"),
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

addEventListener("keydown", (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey || /input|select|textarea/i.test(e.target.tagName)) return;
  const t = Object.keys(TABS)[+e.key - 1];
  if (t && e.key >= "1" && e.key <= "5") { e.preventDefault(); showTab(t); tabBtns[t].focus(); }
});

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
    click(id) { if (fl) stopFlow(); select(id); },
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
setTimeout(() => view.classList.add("touched"), 6000);   // the hint has said its piece
canvas.addEventListener("keydown", (e) => {
  if (!engine) return;
  const K = { ArrowLeft: () => engine.orbit(0.2, 0), ArrowRight: () => engine.orbit(-0.2, 0), ArrowUp: () => engine.orbit(0, 0.12), ArrowDown: () => engine.orbit(0, -0.12),
    "+": () => engine.zoomAt(0.85), "=": () => engine.zoomAt(0.85), "-": () => engine.zoomAt(1.18), f: () => fit(380),
    Enter: () => { const m = build.meta[sel]; if (m?.open) go(m.open, { from: sel }); }, Escape: () => (!keyHelp.hidden ? (keyHelp.hidden = true) : fl ? (stopFlow(), showInfo(), showEditor()) : up()), Backspace: () => up(),
    ",": () => flowStep(-1), ".": () => flowStep(1),
    " ": () => flow(), "?": () => (keyHelp.hidden = !keyHelp.hidden) };
  if (K[e.key]) { e.preventDefault(); K[e.key](); }
});

// ----- rendering a level of the model -----
/** Rebuild the 3D scene. keep: this is an edit, so hold on to the selection and don't touch the editor. */
function render(keep = false) {
  stopFlow();
  const oldName = keep ? build?.meta[sel]?.name : null;
  path = pathOf(parse(cfg, path));
  flags.add(`at:${path}`);
  build = new Build();
  const v = draw(cfg, path, build, { heat: path === "" ? heat : null, temp, picked });
  scan = v.scan || null;
  if (path === "" && pinned && twin && mode === "build") drawTwin(build);
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
  flowBtn.hidden = !flowStops().length;
  view.classList.toggle("deep", path !== "");
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
  hud.replaceChildren(...[
    ["N", "Parameters", t.total, fmt], ["L", "Blocks", cfg.L, (v) => String(Math.round(v))], ["d", "Width", cfg.d, (v) => Math.round(v).toLocaleString("en-US")],
    ["w", "Weights", 2 * t.total, bytes], ...(Number.isFinite(p.num.loss) ? [["loss", "Loss (law)", p.num.loss, (v) => v.toFixed(2), true]] : []),
  ].map(([k, label, v, f, lowGood]) => hudTile(k, label, v, f, lowGood)));
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

/** One HUD number. When it changes it counts to its new value and a chip says by how much. */
const hudPrev = {};
function hudTile(k, label, v, f, lowGood) {
  const b = h("b", f(v)), old = hudPrev[k]; hudPrev[k] = v;
  const el = h("span", h("small", label), b);
  if (old == null || old === v || !Number.isFinite(old) || (k === "loss" && Math.abs(v - old) < 0.005)) return el;
  const r = v / old, abs = k === "loss";
  const say = abs ? `${v > old ? "+" : "−"}${Math.abs(v - old).toFixed(2)}` : r >= 1.5 ? `×${r < 10 ? r.toFixed(1) : Math.round(r)}` : r <= 1 / 1.5 ? `÷${1 / r < 10 ? (1 / r).toFixed(1) : Math.round(1 / r)}` : `${r > 1 ? "+" : "−"}${Math.abs(100 * (r - 1)).toFixed(r > 0.99 && r < 1.01 ? 1 : 0)}%`;
  el.append(h("i.st-dlt", { "data-good": lowGood ? (v < old ? "1" : "0") : "" }, say));
  el.classList.add("chg");
  if (!matchMedia("(prefers-reduced-motion: reduce)").matches) {
    const t0 = performance.now(), log = old > 0 && v > 0 && !abs;
    const step = (now) => {
      const e = Math.min(1, (now - t0) / 480), q = 1 - (1 - e) ** 3;
      b.textContent = f(log ? Math.exp(Math.log(old) + (Math.log(v) - Math.log(old)) * q) : old + (v - old) * q);
      if (e < 1 && b.isConnected) requestAnimationFrame(step);
    };
    b.textContent = f(old); requestAnimationFrame(step);
  }
  return el;
}

// ----- a token's trip: light up each stop in order, with a caption for what happens there -----
// The player: play / pause, step back and forward, speed, replay. Paused, the right panel shows that stop in full.
let fl = null;               // { stops, i, playing, speed, keep, t }
const SPEEDS = [0.5, 1, 2];
let flowSpeed = 1;
const firstSentence = (x) => (x || "").split(/(?<=[.!?])\s/)[0];
const ATT = { full: "looks back at every earlier word", window: "looks back a few words", none: "can't look at other words here" };
function flowStops() {
  if (!build) return [];
  const M = build.meta, at = (re) => M.findIndex((m) => m && re.test(m.name)), all = (re) => M.map((m, i) => (m && re.test(m.name) ? i : -1)).filter((i) => i > 0);
  if (path === "") {
    const toks = all(/^Token /), d = cfg.d.toLocaleString("en-US");
    const stops = [[toks.at(-1), `${M[toks.at(-1)].name.slice(6)} is just a number: its row in a vocabulary of ${cfg.vocab.toLocaleString("en-US")}.`, 1100],
      [at(/^Embedding$/), `Embedding: that row becomes a vector of ${d} numbers.`, 1100]];
    if (at(/^Positions$/) > 0) stops.push([at(/^Positions$/), "+ position: where it sits in the sentence is added on.", 900]);
    const per = Math.max(170, Math.min(650, 5200 / cfg.L));
    for (let b = 1; b <= cfg.L; b++) {
      const s = blk(cfg, b), mlp = s.mlp === "moe" ? `then ${cfg.topk} of ${cfg.experts} experts work on it` : s.mlp ? "then the MLP works on it alone" : "and there's no MLP";
      stops.push([at(new RegExp(`^Block ${b}$`)), `Block ${b} of ${cfg.L}: it ${ATT[s.attn]}, ${mlp}.`, per]);
    }
    if (at(/^Final norm$/) > 0) stops.push([at(/^Final norm$/), "Final norm: rescale the vector before reading out a word.", 900]);
    stops.push([at(/^Unembedding$/), `Unembedding: a score for each of ${cfg.vocab.toLocaleString("en-US")} possible next tokens.`, 1100]);
    const nx = at(/^Next token /); if (nx > 0) stops.push([nx, `Out comes the guess: ${M[nx].name.slice(11)}. Then it all runs again for the next token.`, 1600]);
    return stops.filter((x) => x[0] > 0);
  }
  if (scan) return scanStops();
  return M.map((m, i) => (m?.tag && i > 0 ? [i, `${m.name}: ${firstSentence(m.what)}`, 1600] : null)).filter(Boolean).slice(0, 16);
}
/** Inside one head: each word in turn asks its question and we watch where its attention lands. */
function scanStops() {
  const { tok, rows, head } = scan, w = (t) => `“${t.trim()}”`;
  const stops = rows.map((r, i) => {
    const seen = r.w.map((v, j) => [j, v]).filter(([, v]) => v != null).sort((a, b) => b[1] - a[1]), [top, tw] = seen[0];
    const say = i === 0 ? `${w(tok[0])} comes first: there is nothing before it, so it can only look at itself (100%).`
      : `${w(tok[i])} asks its question and compares it with ${seen.length === i + 1 ? `the ${i + 1} words so far` : `the last ${seen.length} words`}. ${seen.length > 1 && tw - seen[1][1] < 0.02 ? `Its attention is spread evenly: about ${Math.round(tw * 100)}% each.` : `Most of its attention lands on ${w(tok[top])}: ${Math.round(tw * 100)}%.`}${i < tok.length - 1 ? " Later words are hidden." : ""}`;
    return [r.q, say, 2300, { range: [r.lo, r.hi], bars: r.w.map((v, j) => [tok[j], v]) }];
  });
  const res = build.meta.findIndex((m) => m?.name?.startsWith("Result for"));
  if (res > 0) stops.push([res, `Each word walks away with a blend of the values it looked at, weighted by its row. This is a ${head.name}: ${head.say.toLowerCase()}`, 2400]);
  return stops;
}
const fbtn = (label, title, fn, cls = "") => h(`button.st-fb${cls}`, { type: "button", title, "aria-label": title, onclick: (e) => { e.stopPropagation(); fn(); } }, label);
function flowShow() {
  const n = fl.stops.length, [id, say] = fl.stops[fl.i], m = build.meta[id], end = fl.i === n - 1 && !fl.playing;
  const ex = fl.stops[fl.i][3];
  engine.sel = id; engine.glow = id; engine.glowr = ex?.range || [0, -1]; sel = id; engine.draw();
  const facts = ex?.bars ? [] : (m?.rows || []).filter(([, v]) => v && v !== "0").slice(0, 3);
  flowCap.replaceChildren(...[
    h("div.st-flowbar",
      fbtn("⏮", "Back one stop (,)", () => flowStep(-1)),
      end ? fbtn("↺", "Replay", () => { fl.i = 0; fl.playing = true; flowShow(); flowTick(); }, ".main") :
        fbtn(fl.playing ? "❚❚" : "▶", fl.playing ? "Pause (Space)" : "Play (Space)", () => flowToggle(), ".main"),
      fbtn("⏭", "Forward one stop (.)", () => flowStep(1)),
      h("span.st-flowk", `${fl.i + 1}/${n}`),
      h("span.st-flowsp", fbtn(`${flowSpeed}×`, "Speed", () => { flowSpeed = SPEEDS[(SPEEDS.indexOf(flowSpeed) + 1) % SPEEDS.length]; flowShow(); flowTick(); })),
      fbtn("×", "Close (Esc)", () => { stopFlow(); showInfo(); showEditor(); }, ".x")),
    h("p.st-flowt", say),
    facts.length ? h("div.st-flowf", facts.map(([a, b]) => h("span", h("small", a), " ", h("b", b)))) : null,
    ex?.bars ? h("div.st-scan", { role: "img", "aria-label": ex.bars.map(([t, v]) => `${t.trim()} ${v == null ? "hidden" : `${Math.round(v * 100)}%`}`).join(", ") },
      ex.bars.map(([t, v]) => h("span", { "data-off": v == null ? "1" : null }, h("i", { style: { "--h": `${v == null ? 0 : Math.max(4, v * 100)}%` } }), h("b", v == null ? "–" : `${Math.round(v * 100)}%`), h("small", t.trim())))) : null,
    !fl.playing && !end ? h("small.st-flowp", "Paused · this stop is open in the panel →") : null,
    end ? h("small.st-flowp", "That's one token. Writing a reply repeats this for every token.") : null].filter(Boolean));
  flowCap.style.setProperty("--p", `${(100 * (fl.i + 1)) / n}%`);
  view.classList.toggle("paused", !fl.playing);
  if (!fl.playing) { showInfo(); showEditor(); if (tab !== "inspect") showTab("inspect"); }
}
function flowTick() {
  clearTimeout(fl.t);
  if (!fl.playing) return;
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches, ms = fl.stops[fl.i][2] / flowSpeed;
  fl.t = setTimeout(() => {
    if (!fl) return;
    if (fl.i < fl.stops.length - 1) { fl.i++; flowShow(); flowTick(); }
    else { fl.playing = false; flags.add("flow"); flowShow(); checkMode(); }
  }, reduce ? Math.max(ms, 900) : ms);
}
function flowToggle() { if (!fl) return flow(); fl.playing = !fl.playing; if (!fl.playing) flags.add("flow-pause"); flowShow(); flowTick(); checkMode(); }
function flowStep(d) { if (!fl) return; fl.playing = false; clearTimeout(fl.t); fl.i = Math.max(0, Math.min(fl.stops.length - 1, fl.i + d)); flags.add("flow-pause"); flowShow(); checkMode(); }
function stopFlow() {
  if (!fl) return;
  clearTimeout(fl.t); const keep = fl.keep; fl = null;
  flowCap.hidden = true; view.classList.remove("flowing", "paused");
  sel = keep; if (engine) { engine.sel = keep; engine.glow = 0; engine.glowr = [0, -1]; engine.draw(); }
}
function flow() {
  if (fl) return flowToggle();
  const stops = flowStops(); if (!stops.length || !engine) return;
  fl = { stops, i: 0, playing: true, keep: sel, t: 0 };
  if (scan) { flags.add("scan"); checkMode(); }
  view.classList.add("flowing", "touched"); flowCap.hidden = false;
  flowShow(); flowTick();
}

/** Change the design. Redraws the scene and numbers; editors stay put so sliders keep working. */
function set(patch) {
  const before = build ? JSON.stringify(build.bounds().map((v) => Math.round(v))) : "";
  cfg = { ...cfg, ...patch }; edits++; jump = null; stopHistory();
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
      h("p.st-small", `${cfg.name}: ${cfg.L} block${cfg.L === 1 ? "" : "s"}, ${cfg.heads} heads, ${cfg.norm === "None" ? "no norm" : cfg.norm}, ${cfg.act}, ${cfg.pos === "RoPE" ? "RoPE" : cfg.pos === "none" ? "no" : "learned"} positions${cfg.kv < cfg.heads ? `, ${cfg.kv} K/V groups` : ""}. ${fmt(k.total)} parameters.`),
      paramBar());
    whyHost.replaceChildren(...[whyCard(whyKey(null, kind))].filter(Boolean));
    return;
  }
  infoBody.replaceChildren(...[
    h("div.eyebrow", "Selected"), h("h3.st-h", m.name), h("p", m.what),
    m.rows?.length ? h("dl.st-rows", m.rows.flatMap(([a, b]) => [h("dt", a), h("dd", b)])) : null,
    m.open ? h("button.btn.primary.small", { type: "button", onclick: () => go(m.open, { from: sel }) }, `Open ${m.name} ›`) : null].filter(Boolean));
  whyHost.replaceChildren(...[whyCard(whyKey(m, kind))].filter(Boolean));
}
/** Where the parameters live, coloured like the parts in 3D. With experts, a second bar for what one token uses. */
function paramBar() {
  const k = count(cfg), c = full(cfg);
  let attn = 0, dense = 0, experts = 0, used = 0, norms = c.norm === "None" ? 0 : k.normP;
  for (let i = 1; i <= c.L; i++) {
    const s = blk(c, i);
    if (s.attn !== "none") { attn += k.attn; norms += k.normP; }
    if (s.mlp) norms += k.normP;
    if (s.mlp === "moe") { experts += k.moe; used += k.K * k.mlp + k.router; } else if (s.mlp) dense += k.mlp;
  }
  const parts = [["Embedding", k.embed, COL.blue], ["Positions", k.pos, COL.pink], ["Attention", attn, COL.purple], ["MLP", dense, COL.orange], ["Experts", experts, "#ffb46b"], ["Norms", norms, COL.yellow], ["Readout", k.head, COL.teal]].filter(([, v]) => v > 0);
  const bar = (rows, tot) => h("div.st-pb-bar", rows.map(([l, v, col]) => h("span", { style: { flex: String(v / tot), background: col }, title: `${l}: ${fmt(v)} (${Math.round((100 * v) / tot)}%)` })));
  const pct = (v) => { const x = (100 * v) / k.total; return x >= 1 ? `${Math.round(x)}%` : x >= 0.1 ? `${x.toFixed(1)}%` : "<0.1%"; };
  return h("div.st-pb",
    h("div.st-pb-h", h("span.eyebrow", "Where the parameters live"), h("small.muted", fmt(k.total))),
    bar(parts, k.total),
    experts ? h("div.st-pb-used", h("small", `Used by one token: ${fmt(k.active)} (${Math.round((100 * k.active) / k.total)}%)`),
      h("div.st-pb-bar.thin", { style: { width: `${Math.max(4, (100 * k.active) / k.total)}%` } }, parts.map(([l, v, col]) => [l, l === "Experts" ? used : v, col]).filter(([, v]) => v > 0).map(([l, v, col]) => h("span", { style: { flex: String(v), background: col }, title: `${l}: ${fmt(v)}` })))) : null,
    h("ul.st-pb-k", parts.map(([l, v, col]) => h("li", h("i", { style: { background: col } }), h("span", l), h("b", pct(v))))));
}

function showEditor() {
  const ed = partEditor(build.meta[sel]?.edit, get, set);
  editHost.replaceChildren(...[...(ed ? [h("div.st-sub", "Change it"), ed] : []), parse(cfg, path).kind === "head" ? samplerCard() : null].filter(Boolean));
}

/** On the readout: turn the temperature and roll the dice, to see how a word is actually picked. */
function samplerCard() {
  const out = h("div.st-smp-out"), pct = (x) => `${Math.round(x * 100)}%`;
  const paint = (fresh) => {
    const pr = nextAt(temp), last = samples.at(-1), many = pr.filter(([, x]) => x >= 0.05).length;
    out.replaceChildren(
      h("p.st-smp-s", "The cat sat on the", last ? h(`b${fresh ? ".pop" : ""}`, last.trim()) : h("i", "…")),
      samples.length ? h("div.st-smp-hist", { "aria-label": "Words picked so far" }, samples.slice(-14).map((t, i, a) => h("span", { "data-last": i === a.length - 1 ? "1" : null, "data-top": t === pr[0][0] ? "1" : null }, t.trim()))) : null,
      h("small.st-smp-k", `“mat” ${pct(pr[0][1])} · “dog” ${pr[9][1] < 0.001 ? "under 0.1%" : pct(pr[9][1])} · ${many} word${many === 1 ? "" : "s"} with a real chance (5%+)`));
  };
  const say = (v) => `${v.toFixed(1)}${v <= 0.3 ? " · almost greedy" : v === 1 ? " · as trained" : v >= 1.5 ? " · wild" : ""}`;
  const roll = (k = 1) => {
    for (let n = 0; n < k; n++) {
      const pr = nextAt(temp); let r = Math.random(), i = 0;
      for (; i < pr.length - 1; i++) { r -= pr[i][1]; if (r <= 0) break; }
      picked = i; samples.push(pr[i][0]);
    }
    flags.add("sample"); render(true); paint(true); checkMode();
  };
  paint(false);
  return h("div.st-smp",
    h("div.st-sub", "Pick the next word"),
    h("p.st-small", "The model only gives odds. Something still has to pick. Temperature divides the scores before softmax: below 1 the favourite wins more often, above 1 the odds flatten. Chat models usually run around 0.7 to 1."),
    slider({ label: "Temperature", min: 0.1, max: 2, step: 0.1, value: temp, fmt: say, onInput: (v) => { temp = Math.round(v * 10) / 10; picked = -1; render(true); paint(false); } }),
    h("div.row.st-smp-b",
      h("button.btn.primary.small", { type: "button", onclick: () => roll(1) }, "🎲 Pick a word"),
      h("button.btn.small", { type: "button", onclick: () => roll(10) }, "Pick 10"),
      h("button.btn.small", { type: "button", onclick: () => { samples = []; picked = -1; render(true); paint(false); } }, "Clear")),
    out,
    h("details.st-myth", h("summary", h("span.st-myth-k", "Common mix-up"), h("span", "“Temperature 0 makes it more accurate.”")),
      h("p", "It makes it repeatable: always the top word. That's good for a quiz answer and bad for long writing, where greedy picking falls into loops (“the the the”). It doesn't make the model know more; the odds are the same, only the picking changes.")));
}

function showPredict(p = P()) {
  const tagChip = (t) => h("span.st-tag", { "data-tag": t, title: TAGS[t].say }, TAGS[t].label);
  panes.predict.replaceChildren(...[
    p.errors.length ? h("div.st-card.st-errs", { style: "order:-2" }, h("div.eyebrow", "Can't be built"), p.errors.map((e) => h("p", e.say))) : null,
    h("div.st-card",
      h("div.eyebrow", "What will happen"),
      h("p.st-small.muted", p.errors.length ? "Fix the shape error first: these numbers use shapes that can't exist." : `Compared with ${base.name} where it differs. Hover a tag to see how sure each line is.`),
      h("div.st-cards.st-grid", p.cards.map((c) => h("details.st-pc",
        h("summary",
          h("div.st-pc-top", h("span.st-pc-t", c.title), tagChip(c.tag)),
          h("div.st-pc-v", c.value, c.d ? h("span.st-delta", { "data-good": c.d.good == null ? "n" : c.d.good ? "1" : "" }, c.d.text) : null)),
        h("p", c.say)))),
      h("p.st-small.muted", "Tap a number for where it comes from.")),
    h("div.st-card", { style: p.rules.some((r) => r.level === "bad") ? "order:-1" : null },
      h("div.eyebrow", "Watch out for"),
      p.rules.length ? h("ul.st-rules", p.rules.map((r) => h("li", { "data-level": r.level }, h("details", { open: r.level === "bad" }, h("summary", h("div.st-pc-top", h("b", r.title), tagChip(r.tag))), h("p", r.why))))) : h("p.st-small.muted", "Nothing unusual. This is a well-trodden design.")),
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
  const d = diff(), n = Object.keys(d || {}).length, ch = d ? changes(base, cfg) : [];
  modeCard.replaceChildren(h("div.st-card.st-buildc",
    h("label.st-pick", h("span", "Start from"),
      h("select", { onchange: (e) => switchModel(e.target.value) }, Object.entries(PRESETS).map(([k, p]) => h("option", { value: k, selected: k === key }, p.name)))),
    h("p.st-small.muted", PRESETS[key].note),
    timeline(),
    fitsOn(),
    jump && !d ? jumpCard() : null,
    d ? h("div.st-edited", h("span", `Edited · ${n} change${n === 1 ? "" : "s"}`), h("span.row", h("button.btn.small", { type: "button", title: "Copy a link to this design", onclick: (e) => share(e.currentTarget) }, "Copy link"), h("button.btn.small", { type: "button", onclick: () => switchModel(key) }, "Undo all"))) : null,
    ch.length ? h("details.st-log", { open: ch.length <= 4 }, h("summary", `What you changed (${ch.length})`), changeList(ch), overall(base, cfg)) : null,
    pinCard()));
}

// ----- where it fits: the weights at 16, 8 and 4 bits against real devices -----
const DEVICES = [["phone", "Phone", 4e9, "about 4 GB free"], ["laptop", "Laptop", 16e9, "16 GB"], ["gpu", "Gaming GPU", 24e9, "24 GB"], ["gpu", "H100", 80e9, "80 GB"], ["rack", "8 × H100", 640e9, "640 GB"]];
function fitsOn() {
  const N = count(cfg).total, need = (bits) => (N * bits) / 8 * 1.15;   // 15% for activations and a small cache
  const fit = (mem) => [16, 8, 4].find((b) => need(b) <= mem);
  return h("div.st-fits",
    h("div.st-tlh", h("span.eyebrow", "Where it fits"), h("small.muted", `${bytes(2 * N)} at 16-bit`)),
    h("ul", DEVICES.map(([ic, name, mem, say]) => { const b = fit(mem); return h("li", { "data-fit": b ? String(b) : "0", title: `${name} (${say}): ${b ? `fits at ${b}-bit` : "doesn't fit, even at 4-bit"}` },
      h("span.st-fits-i", { html: icon(ic) }), h("b", name), h("small", b ? `${b}-bit` : "no")); })),
    h("small.st-fits-k", "Fewer bits per weight (quantising) shrinks the model: 8-bit is usually close to the original, 4-bit loses a little more. Long chats need room for the KV cache on top."));
}

// ----- the time machine: real models in order, and what changed at each step -----
const HISTORY = [["gpt2", "2019", "GPT-2"], ["gpt3", "2020", "GPT-3"], ["llama2", "2023", "Llama 2"], ["mistral7", "2023", "Mistral"], ["mixtral", "2023", "Mixtral"], ["llama8", "2024", "Llama 3"]];
function timeline() {
  const at = HISTORY.findIndex(([k]) => k === key), p = progress.get();
  return h("div.st-tlw",
    h("div.st-tlh", h("span.eyebrow", "Time machine"), h("button.btn.small", { type: "button", onclick: () => (histTimer ? stopHistory() : playHistory()) }, histTimer ? "❚❚ Stop" : "▶ Play history")),
    h("ol.st-tl", HISTORY.map(([k, y, short], i) => h("li", { "data-on": i === at ? "1" : "", "data-seen": p[`tl:${k}`] ? "1" : "", "data-past": at >= 0 && i < at ? "1" : "" },
      h("button", { type: "button", title: PRESETS[k].name, onclick: () => { stopHistory(); jumpTo(k); } }, h("small", y), h("b", short))))));
}
function jumpTo(k) {
  const from = key; if (k === from) return;
  progress.mark(`tl:${k}`);
  jump = { from, to: k };
  switchModel(k, true);
  jump = { from, to: k };
  buildCard(); stickers();
}
function playHistory() {
  stopHistory();
  let i = Math.max(0, HISTORY.findIndex(([k]) => k === key));
  if (i === HISTORY.length - 1) { i = 0; progress.mark(`tl:${HISTORY[0][0]}`); switchModel(HISTORY[0][0], true); }
  const step = () => { if (i >= HISTORY.length - 1) { stopHistory(); return; } i++; jumpTo(HISTORY[i][0]); histTimer = setTimeout(step, 5200); buildCard(); };
  histTimer = setTimeout(step, 1200); buildCard();
}
function stopHistory() { if (histTimer) { clearTimeout(histTimer); histTimer = 0; if (mode === "build") buildCard(); } }
function jumpCard() {
  const a = full(PRESETS[jump.from]), b = full(PRESETS[jump.to]), ch = changes(a, b);
  return h("div.st-jump",
    h("div.st-jump-h", h("span", PRESETS[jump.from].name), h("i", "→"), h("b", PRESETS[jump.to].name)),
    ch.length ? changeList(ch) : h("p.st-small.muted", "Same architecture."), overall(a, b));
}

/** Every difference between two designs, in words, with why it matters. */
const nn = (v) => (typeof v === "number" ? v.toLocaleString("en-US") : String(v));
function changes(a, b) {
  const out = [], add = (label, x, y, why) => out.push({ label, from: x, to: y, why });
  const scale = "Scale: more parameters, lower loss, higher cost.";
  if (a.d !== b.d) add("Width", nn(a.d), nn(b.d), b.d > a.d ? "Each token carries more numbers. Parameters grow with width squared." : "Fewer numbers per token: smaller and faster, less capacity.");
  if (a.L !== b.L) add("Blocks", nn(a.L), nn(b.L), b.L > a.L ? "More rounds of thinking, one after another." : "Fewer rounds: faster, shallower.");
  if (a.heads !== b.heads) add("Heads", nn(a.heads), nn(b.heads), "More heads ask more questions in parallel, each with fewer numbers.");
  if (a.kv !== b.kv || a.heads !== b.heads) { if (b.kv < b.heads && !(a.kv < a.heads)) add("K/V groups", nn(a.kv), nn(b.kv), "Grouped K/V: heads share keys and values. The cache shrinks, so more users fit on a GPU."); else if (a.kv !== b.kv && b.kv !== b.heads) add("K/V groups", nn(a.kv), nn(b.kv), "Fewer groups, smaller cache."); else if (a.kv !== b.kv && a.kv < a.heads && b.kv === b.heads) add("K/V groups", nn(a.kv), nn(b.kv), "Back to one K/V per head: a bigger cache."); }
  if (a.ff / a.d !== b.ff / b.d) add("MLP width", `${(a.ff / a.d).toFixed(1)}× wide`, `${(b.ff / b.d).toFixed(1)}× wide`, b.act === "SwiGLU" && a.act !== "SwiGLU" ? "Narrower, because SwiGLU's gate adds a third matrix: the same parameters as 4×." : "More or less room to store patterns.");
  if (a.act !== b.act) add("MLP", a.act, b.act, b.act === "SwiGLU" ? "A gate decides which features pass: better quality for the same parameters." : "A different bend in the MLP.");
  if (a.norm !== b.norm) add("Norm", a.norm, b.norm, b.norm === "RMSNorm" ? "Skips subtracting the mean: a little faster, the same quality." : b.norm === "None" ? "No rescaling: unstable, and in our lab run 0.48 worse." : "Back to LayerNorm.");
  if (a.normPos !== b.normPos) add("Norm position", a.normPos === "pre" ? "before each part" : "after the add", b.normPos === "pre" ? "before each part" : "after the add", b.normPos === "post" ? "The 2017 original. Slower to train: 0.21 worse in our 6-block run." : "Pre-norm: stable at any depth.");
  if (a.pos !== b.pos) add("Positions", a.pos, b.pos, b.pos === "RoPE" ? "Rotations encode how far apart tokens are, and stretch to longer text." : b.pos === "none" ? "No position signal: the causal mask alone lets it count." : "A learned table: one row per position, no further than the table.");
  if (a.vocab !== b.vocab) add("Vocabulary", nn(a.vocab), nn(b.vocab), b.vocab > a.vocab ? "More tokens to pick from: each sentence takes fewer of them, especially code and other languages." : "Fewer tokens: a smaller table, longer token sequences.");
  if (a.ctx !== b.ctx) add("Context", `${nn(a.ctx)} tokens`, `${nn(b.ctx)} tokens`, b.ctx > a.ctx ? "Reads more at once. The KV cache grows with it." : "Reads less at once.");
  if (a.tied !== b.tied) add("Readout", a.tied ? "tied" : "separate", b.tied ? "tied" : "separate", b.tied ? "Reuse the embedding to read out: fewer parameters." : "Its own readout matrix: cheap at this size, and it can learn on its own.");
  if (!!a.bias !== !!b.bias) add("Biases", a.bias ? "yes" : "no", b.bias ? "yes" : "no", b.bias ? "An extra number added after each matrix." : "Dropped: barely helps, and steadier at scale.");
  if (a.residual !== b.residual) add("Residual stream", a.residual ? "on" : "off", b.residual ? "on" : "off", b.residual ? "The highway is back." : "No highway: our 6-block run stalled at 3.34.");
  const kinds = (c) => { let w = 0, none = 0, moe = 0, nomlp = 0; for (let i = 1; i <= c.L; i++) { const x = blk(c, i); if (x.attn === "window") w++; if (x.attn === "none") none++; if (x.mlp === "moe") moe++; if (!x.mlp) nomlp++; } return { w, none, moe, nomlp }; };
  const ka = kinds(a), kb = kinds(b);
  if (ka.w !== kb.w || (kb.w && a.window !== b.window)) add("Sliding window", ka.w ? `${ka.w} blocks` : "none", kb.w ? `${kb.w} blocks, ${nn(b.window)} back` : "none", kb.w ? "Those blocks only look back a fixed distance. Stacked, they still reach far, and the cache stops growing." : "Every block sees everything again.");
  if (ka.moe !== kb.moe || (kb.moe && (a.experts !== b.experts || a.topk !== b.topk))) add("Experts", ka.moe ? `${ka.moe} blocks` : "none", kb.moe ? `${kb.moe} blocks, ${b.topk} of ${b.experts}` : "none", kb.moe ? "Many MLPs per block, a few used per token: more knowledge at the same speed." : "Back to one dense MLP per block.");
  if (ka.none !== kb.none) add("Blocks without attention", nn(ka.none), nn(kb.none), "Those blocks can't look at other tokens.");
  if (ka.nomlp !== kb.nomlp) add("Blocks without an MLP", nn(ka.nomlp), nn(kb.nomlp), "Less room to store patterns.");
  void scale;
  return out;
}
const changeList = (ch) => h("ul.st-chg", ch.map((c) => h("li", h("div", h("b", c.label), h("span.st-chg-ft", h("s", c.from), " → ", h("em", c.to))), h("small", c.why))));
/** The net effect of a change, from the predictor. */
function overall(a, b) {
  const A = predict(a, a, tokMul).num, B = predict(b, a, tokMul).num;
  const r = (x, y) => (y / x >= 1.5 ? `×${(y / x).toFixed(y / x < 10 ? 1 : 0)}` : x / y >= 1.5 ? `÷${(x / y).toFixed(x / y < 10 ? 1 : 0)}` : `${y >= x ? "+" : "−"}${Math.abs(100 * (y / x - 1)).toFixed(0)}%`);
  const chip = (label, x, y, lowGood) => h("span", { "data-good": x === y || lowGood == null ? "" : (y < x) === lowGood ? "1" : "0" }, h("small", label), r(x, y));
  return h("div.st-net", h("span.st-net-k", "Net effect"),
    chip("Parameters", A.N, B.N, null), chip("KV cache", A.kvFull, B.kvFull, true), chip("Speed", A.tps, B.tps, false),
    Number.isFinite(A.loss) && Number.isFinite(B.loss) ? h("span", { "data-good": Math.abs(B.loss - A.loss) < 0.005 ? "" : B.loss < A.loss ? "1" : "0" }, h("small", "Loss (law)"), `${B.loss <= A.loss ? "−" : "+"}${Math.abs(B.loss - A.loss).toFixed(2)}`) : null);
}

// ----- pin a design, keep editing, compare -----
function pinCard() {
  if (!pinned) return h("button.btn.small.st-pinbtn", { type: "button", title: "Keep a copy of this design to compare against", onclick: () => { pinned = { cfg: { ...cfg }, name: diff() ? `${cfg.name} (edited)` : cfg.name }; flags.add("pin"); buildCard(); stickers(); retwin(); } }, "📌 Pin this design to compare");
  const A = predict(pinned.cfg, pinned.cfg, tokMul).num, B = P().num, same = JSON.stringify(pinned.cfg) === JSON.stringify(cfg);
  const rows = [["Parameters", A.N, B.N, fmt, null], ["Weights", A.weights, B.weights, bytes, true], ["KV cache, one full chat", A.kvFull, B.kvFull, bytes, true],
    ["Tokens/s, one user", A.tps, B.tps, (v) => Math.round(v).toLocaleString("en-US"), false], ["Users per GPU", A.users, B.users, (v) => (v >= 1 ? Math.floor(v).toLocaleString("en-US") : "0"), false],
    ["Loss (law)", A.loss, B.loss, (v) => (Number.isFinite(v) ? v.toFixed(2) : "—"), true], ["Training cost", A.usd, B.usd, (v) => (v >= 1e6 ? `$${(v / 1e6).toFixed(1)}M` : v >= 1e3 ? `$${(v / 1e3).toFixed(0)}k` : `$${Math.round(v)}`), true]];
  return h("div.st-cmp",
    h("div.st-pc-top", h("span.eyebrow", "Pinned vs now"), h("span.row", h("button.btn.small", { type: "button", title: "Go back to the pinned design", onclick: () => { const c = pinned.cfg; key = Object.keys(PRESETS).find((k) => PRESETS[k].name === c.name) || key; base = full(PRESETS[key]); set({ ...c }); } }, "Restore"), h("button.btn.small", { type: "button", onclick: () => { pinned = null; buildCard(); retwin(); } }, "Unpin"))),
    h("label.st-twin", h("input", { type: "checkbox", checked: twin, onchange: (e) => { twin = e.target.checked; retwin(); } }), h("span", "Stand them side by side in 3D"), h("small", path === "" ? "" : "(on the whole-model view)")),
    same ? h("p.st-small.muted", "Now change something: the right column follows your edits.") : null,
    h("table.st-cmpt", h("thead", h("tr", h("th", ""), h("th", pinned.name), h("th", "Now"))),
      h("tbody", rows.filter(([, a, b]) => Number.isFinite(a) || Number.isFinite(b)).map(([label, a, b, f, lowGood]) => {
        const good = lowGood == null || !Number.isFinite(a) || !Number.isFinite(b) || Math.abs(b - a) <= Math.abs(a) * 0.005 ? "" : (b < a) === lowGood ? "1" : "0";
        return h("tr", h("th", label), h("td", f(a)), h("td", { "data-good": good }, f(b)));
      }))));
}

// ----- the pinned design, standing next to yours in 3D -----
const GHOST = "#b9bdc9";
const fadeCol = (c) => "#" + [1, 3, 5].map((i) => Math.round(parseInt(c.slice(i, i + 2), 16) * 0.45 + parseInt(GHOST.slice(i, i + 2), 16) * 0.55).toString(16).padStart(2, "0")).join("");
/** A Build that draws into another Build: shifted along x, washed out, and not clickable. */
class Twin {
  constructor(B, dx, label) { this.B = B; this.dx = dx; this.label = label; this.meta = B.meta; }
  obj(m) { return this.B.obj({ decor: true, name: `Pinned · ${m.name || ""}`, tag: m.name === "Embedding" ? `Pinned: ${this.label}` : null }); }
  box(x, y, z, w, d, hh, col, id, top) { this.B.box(x + this.dx, y - this.dx, z, w, d, hh, fadeCol(col), id, top && fadeCol(top)); return this; }
}
Twin.prototype.cells = Build.prototype.cells;
function drawTwin(B) {
  const b = B.bounds(), cur = B.meta.find((m) => m?.name === "Embedding");
  if (cur) cur.tag = "Now";
  draw(pinned.cfg, "", new Twin(B, b[3] + 4, pinned.name), {});
}

// ----- save a picture of the model as a sticker card -----
function retwin() { if (path !== "") return; render(true); fit(450); }
function savePicture() {
  if (!engine) return;
  engine.renderNow();
  const src = canvas, W = 1200, H = 1500, pad = 60, cv = document.createElement("canvas"); cv.width = W; cv.height = H;
  const g = cv.getContext("2d"), css = getComputedStyle(document.documentElement), col = (v, d) => css.getPropertyValue(v).trim() || d;
  const ink = col("--ink", "#16131c"), paper = col("--surface", "#ffffff"), bg = col("--bg", "#f4f2ee"), accent = col("--accent", "#6a4cf0");
  g.fillStyle = bg; g.fillRect(0, 0, W, H);
  // the card: thick ink border, offset shadow, like the stickers
  g.fillStyle = ink; g.fillRect(pad + 10, pad + 14, W - 2 * pad, H - 2 * pad);
  g.fillStyle = paper; g.fillRect(pad, pad, W - 2 * pad, H - 2 * pad);
  g.lineWidth = 8; g.strokeStyle = ink; g.strokeRect(pad, pad, W - 2 * pad, H - 2 * pad);
  const box = [pad + 40, pad + 170, W - 2 * pad - 80, 900], k = Math.min(box[2] / src.width, box[3] / src.height);
  g.imageSmoothingEnabled = false;
  g.drawImage(src, box[0] + (box[2] - src.width * k) / 2, box[1] + (box[3] - src.height * k) / 2, src.width * k, src.height * k);
  g.fillStyle = accent; g.font = "700 26px ui-monospace, monospace"; g.fillText("AI ANATOMY · STUDIO", pad + 40, pad + 70);
  g.fillStyle = ink; g.font = "800 64px system-ui, sans-serif"; g.fillText(cfg.name + (diff() ? " (edited)" : ""), pad + 40, pad + 140, W - 2 * pad - 80);
  const t = count(cfg), p = P().num, stats = [["PARAMETERS", fmt(t.total)], ["BLOCKS", String(cfg.L)], ["WIDTH", cfg.d.toLocaleString("en-US")], ["WEIGHTS", bytes(2 * t.total)], ["LOSS (LAW)", Number.isFinite(p.loss) ? p.loss.toFixed(2) : "—"]];
  const sw = (W - 2 * pad - 80 - 4 * 16) / 5;
  stats.forEach(([a, b], i) => {
    const x = pad + 40 + i * (sw + 16), y = H - pad - 230;
    g.fillStyle = ink; g.fillRect(x + 4, y + 6, sw, 130); g.fillStyle = bg; g.fillRect(x, y, sw, 130); g.lineWidth = 4; g.strokeRect(x, y, sw, 130);
    g.fillStyle = ink; g.font = "600 18px ui-monospace, monospace"; g.fillText(a, x + 16, y + 40, sw - 24);
    g.font = "800 40px system-ui, sans-serif"; g.fillText(b, x + 16, y + 100, sw - 24);
  });
  g.fillStyle = ink; g.font = "500 22px system-ui, sans-serif";
  g.fillText(`${cfg.heads} heads · ${cfg.norm} · ${cfg.act} · ${cfg.pos === "RoPE" ? "RoPE" : cfg.pos === "none" ? "no" : "learned"} positions${cfg.kv < cfg.heads ? ` · ${cfg.kv} K/V groups` : ""}`, pad + 40, H - pad - 40, W - 2 * pad - 80);
  const a = document.createElement("a");
  a.download = `${cfg.name.replace(/[^a-z0-9]+/gi, "-").toLowerCase()}-studio.png`;
  a.href = cv.toDataURL("image/png"); document.body.append(a); a.click(); a.remove();
  flags.add("photo"); stickers();
}
async function share(btn) {
  const url = location.origin + urlFor("");
  try { await navigator.clipboard.writeText(url); btn.textContent = "Copied"; } catch { prompt("Copy this link:", url); }
  setTimeout(() => (btn.textContent = "Copy link"), 1800);
}

function levelCard() {
  const L = LEVELS[lv], done = progress.get(), p = P(), s = session(p);
  while (stepsDone < L.steps.length && L.steps[stepsDone].ok(cfg, p, s)) stepsDone++;
  const finished = stepsDone >= L.steps.length;
  modeCard.replaceChildren(h("div.st-card.st-level", { "data-done": finished ? "1" : "" },
    h("div.st-lvmap", { role: "list", "aria-label": "Levels" }, LEVELS.map((l, i) => h("button", { type: "button", role: "listitem", title: l.title, "aria-current": i === lv ? "step" : null, "data-done": done[`lv:${l.id}`] ? "1" : "", onclick: () => loadLevel(i) }, done[`lv:${l.id}`] && i !== lv ? "✓" : String(i + 1))),
      h("small.st-lvn", `${LEVELS.filter((l) => done[`lv:${l.id}`]).length}/${LEVELS.length} done`)),
    h("div.st-lvhead", h("span.eyebrow", `Level ${lv + 1} of ${LEVELS.length}`), h("h2.st-h", L.title)),
    h("div.st-say", h("div.st-say-m", { html: mascot("neuro", finished ? "wow" : "happy", { label: "Neuro" }) }), h("p", L.neuro)),
    L.guess ? guessCard(L) : null,
    h("ol.st-steps", L.steps.map((st, i) => h("li", { "data-state": i < stepsDone ? "done" : i === stepsDone ? "now" : "" }, st.do))),
    finished ? h("div.st-can", h("b", "What it can do now"), h("p", L.can), L.sample ? h("pre.st-sample", L.sample) : null, L.sample ? h("small.muted", L.real || "Typical text from a model like this, trained on Shakespeare.") : null) :
      h("div.st-say.servo", h("div.st-say-m", { html: mascot("servo", "happy", { label: "Servo" }) }), h("p", h("b", "Tip: "), L.servo)),
    L.myth ? h("details.st-myth", h("summary", h("span.st-myth-k", "Common mix-up"), h("span", `“${L.myth[0]}”`)), h("p", L.myth[1])) : null,
    h("div.row.st-lvnav",
      lv > 0 ? h("button.btn.small", { type: "button", onclick: () => loadLevel(lv - 1) }, "‹ Back") : null,
      h("button.btn.small", { type: "button", onclick: () => loadLevel(lv) }, "Restart"),
      lv < LEVELS.length - 1 ? h(`button.btn.small${finished ? ".primary" : ""}`, { type: "button", onclick: () => loadLevel(lv + 1) }, finished ? "Next level ›" : "Skip ›") :
        finished ? h("button.btn.small.primary", { type: "button", onclick: () => setMode("ch") }, "Try the challenges ›") : null)));
  if (finished && !done[`lv:${L.id}`]) { progress.mark(`lv:${L.id}`); levelCard(); burst(modeCard.querySelector(".st-can")); stamp(mascot("neuro", "wow", { label: "" }), "Level clear!", L.title); }
}

/** Guess first: commit to an answer before building, then see what really happens (with our measured numbers). */
const guessStore = () => { try { return JSON.parse(store.get(GUESS_KEY, "{}")); } catch { return {}; } };
function guessCard(L) {
  const g = L.guess, pick = guessStore()[L.id];
  return h("div.st-guess", { "data-state": pick == null ? "" : pick === g.a ? "right" : "wrong" },
    h("div.st-guess-top", h("span.st-guess-k", "Guess first"), pick != null ? h("span.st-guess-r", pick === g.a ? "✓ Right" : "✗ Not quite") : null),
    h("p.st-guess-q", g.q),
    h("div.st-guess-o", g.opts.map((o, i) => h("button", { type: "button", disabled: pick != null, "data-pick": pick === i ? "1" : "", "data-ans": pick != null && i === g.a ? "1" : "",
      onclick: (e) => { store.set(GUESS_KEY, JSON.stringify({ ...guessStore(), [L.id]: i })); flags.add("guess"); levelCard(); if (i === g.a) burst(modeCard.querySelector(".st-guess")); } }, o))),
    pick != null ? h("p.st-guess-why", g.why) : null,
    pick != null ? h("button.st-guess-again", { type: "button", onclick: () => { const x = guessStore(); delete x[L.id]; store.set(GUESS_KEY, JSON.stringify(x)); levelCard(); } }, "Ask me again") : null);
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
  const all = res.every((r) => r.ok), was = goalWas[C.id] || [];
  goalWas[C.id] = res.map((r) => r.ok);
  modeCard.replaceChildren(h("div.st-card.st-chal", { "data-done": all ? "1" : "" },
    h("div.st-pc-top", h("span.eyebrow", `Challenge · ${C.level}`), h("button.btn.small", { type: "button", onclick: () => { chId = null; switchTo(); } }, "All challenges")),
    h("h2.st-h", C.title), h("p.st-small", C.brief),
    h("div.st-gbar", { "aria-label": `${res.filter((r) => r.ok).length} of ${res.length} goals met` }, res.map((r) => h("i", { "data-ok": r.ok ? "1" : "" })), h("small", `${res.filter((r) => r.ok).length}/${res.length}`)),
    h("ul.st-goals", res.map((r, i) => h(`li${r.ok && was.length && !was[i] ? ".pop" : ""}`, { "data-ok": r.ok ? "1" : "" }, h("span.st-gk", r.ok ? "✓" : "○"), h("span", r.g.say), h("b", r.now)))),
    all ? h("div.st-can", h("div.st-say", h("div.st-say-m", { html: trophy({ label: "Solved" }) }), h("p", h("b", "Solved. "), C.solved))) :
      h("details.st-hint2", h("summary", "Hint from Servo"), h("div.st-say.servo", h("div.st-say-m", { html: mascot("servo", "happy", { label: "Servo" }) }), h("p", C.hint))),
    h("div.row", h("button.btn.small", { type: "button", onclick: () => loadChallenge(chId) }, "Start over"))));
  if (all && !solved) { solved = true; if (!done[`ch:${C.id}`]) progress.mark(`ch:${C.id}`); burst(modeCard.querySelector(".st-can")); stamp(trophy({ label: "" }), "Solved!", C.title); }
  if (!all) solved = false;
}

const goalWas = {};
/** A sticker stamped on the 3D view: you cleared a level or solved a challenge. */
function stamp(art, big, small) {
  view.querySelector(".st-stamp")?.remove();
  const el = h("div.st-stamp", { "aria-hidden": "true" }, h("span.st-stamp-m", { html: art }), h("b", big), h("small", small));
  view.append(el); burst(el);
  setTimeout(() => el.classList.add("out"), 2300); setTimeout(() => el.remove(), 2800);
}
function session(p) { return { sel: build?.meta[sel]?.name || "", path, tab, flags, tok: tokMul, start: predict(base, base, startTok()).num }; }
const startTok = () => (mode === "ch" && chId ? CHALLENGES.find((c) => c.id === chId).tok || 20 : 20);
function checkMode() { if (mode === "learn") levelCard(); else if (mode === "ch") chCard(); else buildCard(); stickers(); }
function stickers() {
  const g = guessStore(), right = LEVELS.filter((l) => l.guess && g[l.id] === l.guess.a).length;
  const p = progress.get(), timeline = HISTORY.every(([k]) => p[`tl:${k}`]);
  const fresh = award({ flags, right, edits, samples: samples.length, result: !!result, timeline, moeSeen: /\/mlp$/.test(path) && build?.meta.some((m) => m?.name === "Router") });
  fresh.forEach((x, i) => setTimeout(() => toast(x), 400 * i));
  if (fresh.length) paintBook();
}

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

function switchModel(k, flyIn = false) {
  key = k; base = full(PRESETS[k]); cfg = { ...base }; heat = null; result = null; if (!flyIn) jump = null;
  render(); refreshNumbers();
  if (flyIn && engine && !matchMedia("(prefers-reduced-motion: reduce)").matches) { const t = engine.framing(build.bounds()); engine.cam = { ...t, yaw: t.yaw - 0.9, zoom: t.zoom * 1.6 }; engine.fly(t, 700); } else fit(0);
  if (tab === "design") panes.design.replaceChildren(designPane());
}

render(); refreshNumbers(); showTab("inspect");
requestAnimationFrame(() => fit(0));
