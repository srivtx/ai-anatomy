import { h, loop, reducedMotion, progress } from "./core.js";
import { LEVELS, PROJECTS } from "./catalog.js";
import { topbar } from "./topbar.js";
import { MLP, dataset, bceLogit, sigmoid } from "./nn.js";
import { plane } from "./plot.js";

topbar(document.getElementById("top"));
const done = progress.read();

// Hero: a small network learning a two-armed spiral, live.
const viz = document.getElementById("hero-viz");
const pl = plane({ width: 420, height: 320, xr: [-1.25, 1.25], yr: [-0.95, 0.95], label: "A neural network learning to separate two spirals" });
const data = dataset("spiral", 240, { noise: 0.06, seed: 5 });
let net = new MLP([2, 16, 16, 1], { act: "tanh", seed: 2 });
let epoch = 0, loss = 0;
const cap = h("span"), capR = h("span", "2 → 16 → 16 → 1, tanh, Adam");
viz.replaceChildren(pl.el, h("p.canvas-cap", cap, capR));
function trainEpoch() {
  loss = 0;
  for (const p of data) {
    const r = bceLogit(net.forward(p.x)[0], p.y);
    loss += r.loss;
    net.backward([r.d]);
  }
  net.step(0.02, data.length);
  loss /= data.length;
  epoch++;
  if (epoch > 1400 || (loss < 0.02 && epoch > 400)) { epoch = 0; net = new MLP([2, 16, 16, 1], { act: "tanh", seed: 1 + ((Math.random() * 1e6) | 0) }); }
}
function draw() {
  pl.clear();
  pl.field((x, y) => sigmoid(net.forward([x, y])[0]), { res: 84, alpha: 0.55 });
  pl.scatter(data, { r: 3 });
  cap.textContent = `epoch ${epoch} · loss ${loss.toFixed(3)}`;
}
if (reducedMotion()) { for (let i = 0; i < 600; i++) trainEpoch(); draw(); }
else { const lp = loop(() => { for (let i = 0; i < 4; i++) trainEpoch(); draw(); }, pl.el); lp.start(); }

const live = LEVELS.flatMap((lv) => lv.chapters).filter((c) => c.live).length;
const seen = Object.values(done).filter((d) => d.visited).length;
document.getElementById("hero-stats").replaceChildren(
  ...[[live, "live chapters"], [PROJECTS.length, "Colab projects"], [LEVELS.length, "levels"], [seen, seen ? "opened by you" : "sign-ups needed"]]
    .map(([n, l]) => h("div.hs", h("b", String(n)), h("span", l))));
const lc = (i) => `var(--c${(i % 6) + 1})`;
// the course map: levels grouped into acts, one level's chapters at a time
import("./explorer.js").then(({ explorer }) => explorer(document.getElementById("ladder"), {
  levels: LEVELS, done,
  acts: [
    { name: "Act I · How learning works", blurb: "maths, learning, classic ML, deep learning", levels: [0, 1, 2, 3] },
    { name: "Act II · Language models", blurb: "from tokens to assistants that think", levels: [4, 5, 6, 7, 8] },
    { name: "Act III · Building with them", blurb: "systems, agents, research", levels: [9, 10, 11] },
    { name: "Act IV · Beyond the textbook", blurb: "reality, lab notes, the hidden curriculum", levels: [12, 13, 14, 15] },
  ].map((a) => ({ ...a, levels: a.levels.filter((i) => i < LEVELS.length) })),
}));
document.getElementById("nproj").textContent = String(PROJECTS.length);

// ---------- inside the code: pick any step-through and it runs right here ----------
import("./walklist.js").then(({ WALK_LIST }) => {
  const box = document.getElementById("inside"), stage = h("div.in-stage"), link = h("a.btn.small", { href: "#" });
  let cur = "attention", mod = null;
  const chips = WALK_LIST.map(([id, name]) => h("button.in-chip", { type: "button", "data-id": id, "aria-pressed": String(id === cur), onclick: () => show(id) }, name));
  async function show(id) {
    cur = id;
    chips.forEach((c) => c.setAttribute("aria-pressed", String(c.dataset.id === id)));
    const ch = LEVELS.flatMap((lv) => lv.chapters).find((c) => c.id === id);
    link.href = `/learn?c=${id}`; link.textContent = `Read “${ch ? ch.title : id}” →`;
    mod ||= await import("./walks.js");
    if (cur === id) stage.replaceChildren(mod.walk(id));
  }
  box.replaceChildren(
    h("div.in-head",
      h("div", h("div.eyebrow", "Inside the code"), h("h2", "Watch the numbers move"),
        h("p.muted", `${WALK_LIST.length} chapters take a few lines of real code and step through them with real numbers. Every cube is one value, blue for negative and orange for positive. Press Play, or hover a cube to see the arithmetic that made it.`)),
      link),
    h("div.in-chips", { role: "group", "aria-label": "Choose a step-through" }, chips), stage);
  // load the walkthrough code only when the section comes near the screen
  const go = () => show(cur);
  if ("IntersectionObserver" in window) { const io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) { io.disconnect(); go(); } }, { rootMargin: "300px" }); io.observe(box); } else go();
});

// ---------- pixel layer: the mascot, and your bench (a garden per level, cans per quiz, compute earned) ----------
import("./pixel.js").then(async ({ mascot, flower, can }) => {
  const { quizState } = await import("./prefs.js");
  const { desk, cashPile } = await import("./voxel.js");
  const { stickerRow } = await import("./badges.js");
  const qs = quizState.get(), due = quizState.due(qs).length, hist = qs.hist || [];
  const liveCh = LEVELS.flatMap((lv) => lv.chapters.filter((c) => c.live));
  const nextCh = liveCh.find((c) => !done[c.id]?.visited);
  const opened = liveCh.filter((c) => done[c.id]?.visited).length, predicted = liveCh.filter((c) => done[c.id]?.predicted).length;
  const rightAnswers = hist.reduce((a, r) => a + r.right, 0);
  const flops = opened * 1e15 + predicted * 5e15 + rightAnswers * 2e14;
  const fmtF = (f) => { if (!f) return "0 FLOPs"; const u = [["E", 1e18], ["P", 1e15], ["T", 1e12]].find(([, v]) => f >= v) || ["T", 1e12]; return `${(f / u[1]).toFixed(f / u[1] < 10 ? 1 : 0)} ${u[0]}FLOPs`; };
  const say = due ? h("span", `${due} question${due === 1 ? " is" : "s are"} due for review. `, h("a", { href: "/quiz" }, "Let's go →"))
    : nextCh ? h("span", opened ? "Next up: " : "Start here: ", h("a", { href: `/learn?c=${nextCh.id}` }, nextCh.title), " →")
    : h("span", "You've opened every chapter. ", h("a", { href: "/quiz" }, "Sit a mock exam →"));

  // returning visitors: the main button picks up where they left off
  const cta = document.querySelector(".hero-ctas .btn.primary");
  if (cta?.dataset.path) { /* the learning mode already set it */ }
  else if (cta && opened && nextCh) { cta.textContent = `Continue: ${nextCh.title} →`; cta.href = `/learn?c=${nextCh.id}`; }
  else if (cta && opened && !nextCh) { cta.textContent = "Sit a mock exam →"; cta.href = "/quiz"; }
  // the mascot peeks over the live demo
  const fig = document.getElementById("hero-viz");
  if (fig) {
    const wrap = h("div.viz-wrap"); fig.replaceWith(wrap); wrap.append(fig,
      h("div.mascot-wrap.hero-mascot", h("div.bubble", opened ? "Training on spirals. Loss is going down!" : h("span", "Hi! I'm Neuro. This net is learning live. ", h("a", { href: "/ask" }, "Learn it with me →"))), h("div", { html: mascot("neuro", "happy", { label: "Neuro, the course mascot" }) })));
  }

  const garden = LEVELS.map((lv, i) => {
    const live = lv.chapters.filter((c) => c.live), o = live.filter((c) => done[c.id]?.visited).length, pr = live.filter((c) => done[c.id]?.predicted).length;
    const stage = !o ? 0 : o === live.length && pr ? 3 : o / live.length >= 0.5 ? 2 : 1;
    const word = ["a seed", "a sprout", "a bud", "in bloom"][stage];
    return h("a.plot", { href: `#${lv.id}`, style: { "--lc": lc(i) }, title: `Level ${i} · ${lv.name}: ${word} (${o}/${live.length} opened)` },
      h("div", { html: flower(stage, lv.id, { label: `Level ${i}: ${word}` }) }), h("span", String(i)));
  });
  const cans = hist.slice(-6).map((r, k) => h("div.can-i", { title: `${r.right}/${r.n} on ${new Date(r.at).toLocaleDateString()}`, html: can(k % 2 ? "#7cc4ff" : "#ff7eb6") }));
  document.getElementById("ladder")?.after(h("section.bench",
    h("div.bench-h", h("div.bench-t", h("div.eyebrow", "Your bench"), h("h3", "A desk that fills up as you learn"), h("p.muted", "It grows as you learn: a flower for every level, a can for every quiz, and compute for everything you do. Saved on this device.")),
      h("div.mascot-wrap.bench-mascot", h("div.bubble.left", say), h("div", { html: mascot("neuro", due ? "wow" : "happy") }))),
    h("div.bench-body",
      h("div.scene", h("div.desk-wrap", { html: desk({ cash: Math.min(6, Math.ceil(opened / 4)), cans: Math.min(4, hist.length), books: LEVELS.filter((lv) => lv.chapters.every((c) => !c.live || done[c.id]?.visited)).length,
        plant: !opened ? 0 : opened / liveCh.length >= 0.6 ? 3 : opened / liveCh.length >= 0.2 ? 2 : 1, rack: opened >= 20 }, "Your desk: it fills up as you learn") }),),
      h("div.garden-wrap", h("div.garden", { "aria-label": "Your garden: one flower per level", style: { "--n": String(LEVELS.length) } }, garden), opened ? null : h("p.garden-hint", "Open a chapter to plant its level. Make a prediction to help it bloom.")),
      h("div.bench-side",
        h("div.bench-item", h("div.cans", cans.length ? cans : h("div.can-i.ghost", { html: can("#c9ccd6") })), h("b", String(hist.length)), h("span", hist.length === 1 ? "quiz taken" : "quizzes taken")),
        h("div.bench-item", h("div.cash-i", { html: cashPile(1) }), h("b", fmtF(flops)), h("span", "compute earned")),
        h("div.bench-item", h("div.fl-i", { html: flower(opened ? 3 : 1, "bench") }), h("b", `${opened}/${liveCh.length}`), h("span", "chapters opened"))),
      stickerRow())));
});

// ---------- two ways in: questions with Neuro, or a learning mode ----------
Promise.all([import("./paths.js"), import("./ask/index.js"), import("./pixel.js")]).then(([{ MODES, getMode, modeProgress }, { THREADS, askState }, { badge, mascot }]) => {
  const box = document.getElementById("start");
  if (!box) return;
  const asked = askState.get(), mine = getMode();
  const firstOpen = THREADS.find((t) => !asked[t.id]?.done) || THREADS[0];
  const doneN = THREADS.filter((t) => asked[t.id]?.done).length;
  // a tiny chat that plays itself, to show what the questions feel like
  const lines = [["n", "You want a computer to spot spam. Rules or examples?"], ["me", "Examples!"], ["n", "Yes. A pile of examples with answers is called training data."], ["n", "Now: a miss of 10, squared, is…?"], ["me", "100"], ["n", "Right. Big misses get punished much more."]];
  const chat = h("div.st-chat", { "aria-hidden": "true" });
  let k = 0;
  const tick = () => {
    if (k >= lines.length) { k = 0; chat.replaceChildren(); }
    const [who, t] = lines[k++];
    chat.append(h(`div.st-b.${who}`, t));
    while (chat.children.length > 4) chat.firstChild.remove();
  };
  for (let i = 0; i < 3; i++) tick();
  if (!reducedMotion()) { const io = new IntersectionObserver((es) => es.forEach((e) => { clearInterval(box._t); if (e.isIntersecting) box._t = setInterval(tick, 1800); })); io.observe(box); }
  const modeCards = MODES.map((m) => {
    const pr = modeProgress(m, LEVELS, done, asked);
    return h("a.st-mode", { href: `/paths?m=${m.id}`, style: { "--mc": m.color }, "aria-current": mine?.id === m.id ? "true" : null },
      h("span.st-mi", { html: badge(m.glyph, m.color) }), h("b", m.name), h("small", mine?.id === m.id ? `your path · ${pr.n}/${pr.total}` : `~${m.hours} h · ${pr.total} stops`));
  });
  box.replaceChildren(
    h("div.st-h", h("div.eyebrow", "Two ways in"), h("h2", "Start where you are")),
    h("div.st-grid",
      h("a.st-card.ask", { href: `/ask?t=${firstOpen.id}` },
        h("div.st-top", h("span.st-face", { html: mascot("neuro", "happy") }), h("div", h("small", "New to all of this?"), h("h3", "Ask Neuro"))),
        h("p", "Learn from zero by answering easy questions, one small step at a time. 18 short conversations, from “what is learning?” to agents."),
        chat,
        h("span.st-go", doneN ? `Continue: ${firstOpen.title} →` : "Answer the first question →")),
      h("div.st-card.modes",
        h("div.st-top", h("div", h("small", "Know why you're here?"), h("h3", "Pick a learning mode"))),
        h("p", "Nobody learns 104 chapters at once. Each mode is a route through the course for one goal, with the Colab projects that fit it."),
        h("div.st-modes", modeCards),
        h("a.st-go", { href: "/paths" }, "Compare all the paths →"))));
  // the main button continues your path, if you picked one
  const cta = document.querySelector(".hero-ctas .btn.primary"), pr = mine && modeProgress(mine, LEVELS, done, asked);
  if (cta && pr?.next) {
    cta.dataset.path = "1";
    cta.textContent = `Continue your ${mine.name} path →`;
    cta.href = pr.next.ask ? `/ask?t=${pr.next.ask}` : `/learn?c=${pr.next.id}`;
  }
});
