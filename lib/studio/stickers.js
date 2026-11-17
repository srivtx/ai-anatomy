// The sticker book: small pixel badges you earn by doing things in the Studio. Earned ones are kept in the same
// localStorage record as level progress (keys "st:<id>"), so they survive reloads.
import { h } from "../core.js";
import { LEVELS, CHALLENGES, progress } from "./learn.js";

// 7×7 pixel icons. X = ink, o = light.
const ICON = {
  door: ["XXXXXXX", "X.....X", "X..o..X", "X.ooo.X", "X..o..X", "X.....X", "XXXXXXX"],
  eye: [".......", "..XXX..", ".X...X.", "X..X..X", ".X...X.", "..XXX..", "......."],
  play: ["X......", "XXX....", "XXXXX..", "XXXXXXX", "XXXXX..", "XXX....", "X......"],
  pause: [".......", ".XX.XX.", ".XX.XX.", ".XX.XX.", ".XX.XX.", ".XX.XX.", "......."],
  star: ["...X...", "...X...", "XXXXXXX", ".XXXXX.", "..XXX..", ".XX.XX.", "X.....X"],
  half: ["..XXX..", ".XX..X.", "XXX...X", "XXX...X", "XXX...X", ".XX..X.", "..XXX.."],
  cap: ["...X...", ".XXXXX.", "XXXXXXX", ".XXXXX.", ".X...X.", ".X...X.", ".XXXXX."],
  cup: ["XXXXXXX", "XXXXXXX", ".XXXXX.", "..XXX..", "...X...", "..XXX..", ".XXXXX."],
  crown: ["X..X..X", "XX.X.XX", "XXXXXXX", "XXXXXXX", "XXXXXXX", ".......", "XXXXXXX"],
  bulb: ["..XXX..", ".X...X.", ".X.X.X.", ".X...X.", "..XXX..", "..XXX..", "...X..."],
  pot: ["..X.X..", "...X...", "XXXXXXX", ".X...X.", ".X...X.", ".X...X.", "..XXX.."],
  flask: ["..XXX..", "...X...", "...X...", "..X.X..", ".X...X.", "XoooooX", "XXXXXXX"],
  clock: ["..XXX..", ".X.X.X.", "X..X..X", "X..XX.X", "X.....X", ".X...X.", "..XXX.."],
  grid: ["XXXXXXX", "X.X.X.X", "XXXXXXX", "X.X.X.X", "XXXXXXX", "X.X.X.X", "XXXXXXX"],
  pin: ["..XXX..", ".XXXXX.", ".XXXXX.", "..XXX..", "...X...", "...X...", "...X..."],
  cam: [".......", "..XX...", "XXXXXXX", "X..X..X", "X.XXX.X", "X..X..X", "XXXXXXX"],
  dice: ["XXXXXXX", "X.....X", "X.X.X.X", "X..X..X", "X.X.X.X", "X.....X", "XXXXXXX"],
  scan: ["XXXXXXX", "X.....X", "XoX...X", "X.oX..X", "X..oX.X", "X...oXX", "XXXXXXX"],
  cards: ["XXXXX..", "X...XX.", "X.o.X.X", "X...X.X", "XXXXX.X", ".X....X", ".XXXXXX"],
  phone: ["..XXX..", "..X.X..", "..X.X..", "..X.X..", "..X.X..", "..XoX..", "..XXX.."],
  laptop: [".......", ".XXXXX.", ".X...X.", ".X...X.", ".XXXXX.", "XXXXXXX", "......."],
  gpu: [".......", "XXXXXXX", "X.XXX.X", "X.XoX.X", "X.XXX.X", "XXXXXXX", ".X.X.X."],
  rack: ["XXXXXXX", "X.o.o.X", "XXXXXXX", "X.o.o.X", "XXXXXXX", "X.o.o.X", "XXXXXXX"],
  wrench: ["X...X..", "XX.XX..", ".XXX...", "..XX...", "...XX..", "....XX.", ".....XX"],
};
export const icon = (k) => {
  const g = ICON[k]; let r = "";
  g.forEach((row, y) => [...row].forEach((c, x) => { if (c !== ".") r += `<rect x="${x}" y="${y}" width="1" height="1"${c === "o" ? ' opacity=".45"' : ""}/>`; }));
  return `<svg viewBox="0 0 7 7" width="21" height="21" fill="currentColor" shape-rendering="crispEdges" aria-hidden="true">${r}</svg>`;
};

const lvDone = (p) => LEVELS.filter((l) => p[`lv:${l.id}`]).length;
const chDone = (p) => CHALLENGES.filter((c) => p[`ch:${c.id}`]).length;
const visited = (s, re) => [...s.flags].some((f) => f.startsWith("at:") && re.test(f.slice(3)));

/** Each sticker: what it's for, how to get it, and a check on the current session s and saved progress p. */
export const STICKERS = [
  { id: "inside", name: "Went inside", how: "Open any part of the model.", icon: "door", col: "#4f8ef7", ok: (s) => visited(s, /^./) },
  { id: "head", name: "Head hunter", how: "Open a single attention head.", icon: "eye", col: "#7b61ff", ok: (s) => visited(s, /attn\/h\d/) },
  { id: "flow", name: "Token tourist", how: "Watch a token travel all the way through.", icon: "play", col: "#22b8a0", ok: (s) => s.flags.has("flow") },
  { id: "pause", name: "Freeze frame", how: "Pause the token on its way.", icon: "pause", col: "#2a9d8f", ok: (s) => s.flags.has("flow-pause") },
  { id: "guess", name: "Good instincts", how: "Get three guess-first questions right.", icon: "bulb", col: "#f4b400", ok: (s) => s.right >= 3 },
  { id: "lv1", name: "First level", how: "Finish any level.", icon: "star", col: "#ff8a3d", ok: (s, p) => lvDone(p) >= 1 },
  { id: "lv5", name: "Halfway", how: "Finish five levels.", icon: "half", col: "#e76f51", ok: (s, p) => lvDone(p) >= 5 },
  { id: "lv9", name: "Graduate", how: `Finish all ${LEVELS.length} levels.`, icon: "cap", col: "#d1495b", ok: (s, p) => lvDone(p) >= LEVELS.length },
  { id: "ch1", name: "Problem solver", how: "Solve a challenge.", icon: "cup", col: "#c9a227", ok: (s, p) => chDone(p) >= 1 },
  { id: "ch8", name: "Architect", how: "Solve every challenge.", icon: "crown", col: "#b5891b", ok: (s, p) => chDone(p) >= CHALLENGES.length },
  { id: "scan", name: "Mind reader", how: "Watch a head scan, word by word.", icon: "scan", col: "#3f51b5", ok: (s) => s.flags.has("scan") },
  { id: "dice", name: "Dice roller", how: "Pick ten words at the readout.", icon: "dice", col: "#ef5350", ok: (s) => s.samples >= 10 },
  { id: "review", name: "Flashcard ace", how: "Know every card in a review of 9 or more.", icon: "cards", col: "#00897b", ok: (s) => s.flags.has("review-all") && [...s.flags].some((f) => /^review:(\d+)$/.test(f) && +f.slice(7) >= 9) },
    { id: "recipe", name: "Chef", how: "Apply a recipe from the Design tab.", icon: "pot", col: "#8d6e63", ok: (s) => [...s.flags].some((f) => f.startsWith("recipe:")) },
  { id: "moe", name: "Expert witness", how: "Look inside a mixture of experts.", icon: "grid", col: "#ff6fb5", ok: (s) => s.moeSeen },
  { id: "time", name: "Time traveller", how: "Visit every model on the timeline.", icon: "clock", col: "#5c6bc0", ok: (s) => s.timeline },
  { id: "pin", name: "Side by side", how: "Pin a design and compare.", icon: "pin", col: "#26a69a", ok: (s) => s.flags.has("pin") },
  { id: "tinker", name: "Tinkerer", how: "Make ten changes to a design.", icon: "wrench", col: "#78909c", ok: (s) => s.edits >= 10 },
  { id: "photo", name: "Say cheese", how: "Save a picture of your model.", icon: "cam", col: "#ab47bc", ok: (s) => s.flags.has("photo") },
  { id: "lab", name: "Scientist", how: "Paste a Colab result into the Test tab.", icon: "flask", col: "#43a047", ok: (s) => s.result },
];

export const earned = (p = progress.get()) => STICKERS.filter((x) => p[`st:${x.id}`]);

/** Award whatever the session has earned. Returns the stickers that are new. */
export function award(s) {
  const p = progress.get(), fresh = STICKERS.filter((x) => !p[`st:${x.id}`] && x.ok(s, p));
  for (const x of fresh) progress.mark(`st:${x.id}`);
  return fresh;
}

const badge = (x, got) => h("span.stk-b", { style: { "--c": got ? x.col : "var(--line)" }, "data-got": got ? "1" : "", html: icon(x.icon) });

/** A toast that slides in for a new sticker. */
export function toast(x) {
  let box = document.querySelector(".stk-toasts");
  if (!box) document.body.append((box = h("div.stk-toasts", { "aria-live": "polite" })));
  const t = h("div.stk-toast", badge(x, true), h("span", h("small", "New sticker"), h("b", x.name)));
  box.append(t);
  setTimeout(() => t.classList.add("out"), 3200); setTimeout(() => t.remove(), 3700);
}

/** The whole book, as a dialog. */
export function book(onClose) {
  const p = progress.get(), n = earned(p).length;
  const dlg = h("div.stk-book", { role: "dialog", "aria-label": "Sticker book", onclick: (e) => e.target === dlg && onClose() },
    h("div.stk-in",
      h("div.stk-top", h("div", h("div.eyebrow", "Sticker book"), h("b.stk-n", `${n} of ${STICKERS.length}`)), h("button.btn.small", { type: "button", "aria-label": "Close", onclick: onClose }, "×")),
      h("div.stk-bar", h("i", { style: { width: `${(100 * n) / STICKERS.length}%` } })),
      h("ul.stk-grid", STICKERS.map((x) => { const got = !!p[`st:${x.id}`]; return h("li", { "data-got": got ? "1" : "" }, badge(x, got), h("b", x.name), h("small", x.how)); }))));
  return dlg;
}
