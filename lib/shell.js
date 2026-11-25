// Loads /chapters/<id>.js for /learn?c=<id> and frames it as a lesson:
// a course sidebar (this level's chapters), a chapter cover, numbered steps,
// an "at a glance" strip and a sticky outline rail.
import { h, progress, rng } from "./core.js";
import { find, neighbours, projectsFor, LEVELS } from "./catalog.js";
import { topbar } from "./topbar.js";
import { flower, mascot, stageOf, glyph, badge } from "./pixel.js";
import { chapterCheck } from "./check.js";
import { emblem } from "./voxel.js";
import { getMode, inMode, chaptersOf, modeProgress } from "./paths.js";

// Chapters with a matching level or challenge in the 3D Architecture Studio.
const STUDIO = {
  attention: ["/studio?mode=learn&lv=attend", "Level 2: let characters look back", "Give a one-block model attention in 3D, then open a head and see who looks at whom."],
  transformer: ["/studio?mode=learn&lv=lookup", "Build a transformer part by part", "Nine levels in 3D: a lookup table, then attention, MLP, more blocks, the residual stream, norms, GPT-2 size and the Llama recipe."],
  "next-token": ["/studio?mode=learn&lv=lookup", "Level 1: a lookup table", "The smallest model that predicts the next token, in 3D. Then add parts and see what each one buys."],
  "long-context": ["/studio?mode=ch&c=long", "Challenge: read 128k tokens cheaply", "Shrink one chat's KV cache from 17 GB to under 4 GB with sliding windows and shared K/V, checked live."],
  "scaling-laws": ["/studio?mode=ch&c=budget", "Challenge: the best model for $500", "Pick parameters and tokens on a budget; the Chinchilla law scores your design."],
  inference: ["/studio?mode=ch&c=users", "Challenge: 100 users on one GPU", "Make room for 100 conversations' KV caches next to the weights. The Speed tab shows every lever."],
  "stability-at-scale": ["/studio?mode=ch&c=fix", "Challenge: make it trainable", "A 24-block model with no residuals or norms. Fix it, and read why each fix works."],
};


topbar(document.getElementById("top"));
const main = document.getElementById("chapter");
document.body.classList.add("is-lesson");
const id = new URLSearchParams(location.search).get("c") || "";
const meta = find(id);
const lcOf = (i) => `var(--c${(i % 6) + 1})`;

if (!meta || !meta.live || !/^[a-z0-9-]+$/.test(id)) {
  // a friendly dead end: the closest chapter by name, the map and the quiz
  const all = LEVELS.flatMap((lv) => lv.chapters.filter((c) => c.live));
  const dist = (a, b) => { const d = Array.from({ length: b.length + 1 }, (_, j) => j); for (let i = 1; i <= a.length; i++) { let p = d[0]; d[0] = i; for (let j = 1; j <= b.length; j++) { const t = d[j]; d[j] = Math.min(d[j] + 1, d[j - 1] + 1, p + (a[i - 1] === b[j - 1] ? 0 : 1)); p = t; } } return d[b.length]; };
  const guess = id ? all.map((c) => [c, dist(id, c.id)]).sort((a, b) => a[1] - b[1])[0] : null;
  main.replaceChildren(h("section.nf",
    h("div.nf-m", { html: mascot("neuro", "sad", { label: "The mascot, a little lost" }) }),
    h("div",
      h("div.eyebrow", meta && !meta.live ? "Coming soon" : "404 · no such chapter"),
      h("h1", meta && !meta.live ? "Not written yet." : "This page wandered off."),
      h("p.lede", meta && !meta.live ? `“${meta.title}” isn't written yet. It's on the course map as coming soon.` : id ? `There's no chapter called “${id}”.` : "That link doesn't point to a chapter."),
      h("div.nf-btns",
        guess && guess[1] <= Math.max(4, id.length / 2) ? h("a.btn.primary", { href: `/learn?c=${guess[0].id}` }, `Did you mean ${guess[0].title}? →`) : null,
        h("a.btn", { href: "/#ladder" }, "Course map"), h("a.btn", { href: "/quiz" }, "Quiz yourself")))));
} else {
  document.title = `${meta.title} · AI Anatomy`;
  const done = progress.read();
  const lc = lcOf(meta.levelIndex);
  main.classList.replace("wrap", "lesson");
  main.style.setProperty("--lc", lc);
  const root = h("div.lesson-col");
  const rail = h("aside.lesson-rail", { "aria-label": "Chapter outline" });
  main.replaceChildren(sidebar(meta, done), root, rail);

  const chIndex = meta.level.chapters.findIndex((c) => c.id === id);
  root.append(
    h("header.ch-head.ch-cover",
      h("div.cover-text",
        h("div.crumbs", h("a", { href: `/#${meta.level.id}` }, `Level ${meta.levelIndex} · ${meta.level.name.split(":")[0]}`), h("span", `Chapter ${meta.n}`)),
        h("h1", meta.title),
        h("p.lede", meta.blurb),
        h("div.cover-chips", { id: "cover-chips" },
          h("span.cchip", `${chIndex + 1} of ${meta.level.chapters.length} in this level`),
          done[id]?.predicted ? h("span.cchip.ok", "✓ done before") : done[id]?.visited ? h("span.cchip", "opened before") : h("span.cchip.new", "new to you"),
          pathChip(id))),
      h("div.cover-art", { "aria-hidden": "true", html: emblem(meta.level.id) }), h("div.cover-fl", { "aria-hidden": "true", html: flower(Math.max(1, stageOf(done[id])), id) }), h("i.px-deco", { style: { right: "26px", top: "22px" } })),
  );
  const loading = h("div.ch-loading", { role: "status" }, h("span.ch-load-m", { html: mascot("neuro", "think") }), h("span", "Loading the simulation…"));
  root.append(loading);
  let failed = false;
  try {
    // the chapter first; the extras below it (field notes, deep cuts, projects) load in parallel
    const extras = Promise.all([import("./practice.js"), import("./known.js"), import("./deepcuts.js"), import("./cuts.js"), import("./colab.js")]);
    const mod = await import(`../chapters/${id}.js`);
    // let the header and spinner paint before a chapter does heavy setup work
    await new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));
    const body = h("div.ch-body");
    root.append(body);
    mod.default(body, { id });
    loading.remove();
    if (meta.level.id === "silicon") import("./prefs.js").then(({ prefs }) => prefs.get().gloss !== "off" && import("./glossary.js").then(({ glossify }) => glossify(body)));
    progress.mark(id, "visited");
    const [{ PRACTICE }, { KNOWN, OPEN }, { DEEP_CUTS }, { cutCard }, { colabCard }] = await extras;
    const fn = PRACTICE[id];
    // everything after the chapter sits in one block of tabs: one part open at a time instead of a long stack
    const parts = [];
    if (fn) parts.push(["field", "Field notes", fn.works.length + fn.traps.length + (fn.numbers?.length || 0), [
      h("p.muted", "What practitioners actually do, what bites them, and numbers worth remembering."),
      h("div.field-grid",
        h("div.field-card.works", h("div.eyebrow", "What works"), h("ul", fn.works.map((t) => h("li", t)))),
        h("div.field-card.traps", h("div.eyebrow", "Traps"), h("ul", fn.traps.map((t) => h("li", t)))),
        fn.numbers?.length ? h("div.field-card.nums", h("div.eyebrow", "Numbers to remember"), h("ul", fn.numbers.map((t) => h("li", t)))) : null)]]);
    if (KNOWN[id] || OPEN[id]) parts.push(["known", "Found & unknown", (KNOWN[id]?.length || 0) + (OPEN[id]?.length || 0), [
      h("p.muted", "What large-scale practice has established, and what nobody knows yet."),
      h("div.field-grid.two",
        KNOWN[id] ? h("div.field-card.known", h("div.eyebrow", "What the big runs found"), h("ul", KNOWN[id].map((t) => h("li", t)))) : null,
        OPEN[id] ? h("div.field-card.open", h("div.eyebrow", "Still unknown"), h("ul", OPEN[id].map((t) => h("li", t)))) : null)]]);
    const cuts = DEEP_CUTS.filter((c) => c[0] === id);
    if (cuts.length) parts.push(["cuts", "Deeper cuts", cuts.length, [
      h("p.muted", h("span", "Lesser-known ideas around this chapter, with code. Where a result needs more compute than a laptop, it's quoted from the paper. "), h("a", { href: "/learn?c=deep-cuts" }, "See all of them in the atlas.")),
      h("div.cuts", cuts.map(cutCard))]]);
    const st = STUDIO[id];
    if (st) parts.push(["studio", "In the Studio", 1, [h("a.st-link", { href: st[0] }, h("span.st-link-m", { html: mascot("neuro", "happy", { label: "" }) }), h("span", h("b", st[1]), h("small", st[2])), h("span.st-link-go", "Open ›"))]]);
    const projects = projectsFor(id);
    if (projects.length) parts.push(["build", "Build it", projects.length, [
      h("p.muted", "Every project is a notebook you can run top to bottom on Google Colab. Read the code and change it, then try the stretch goals."), projects.map(colabCard)]]);
    if (parts.length) root.append(extrasTabs(parts));
  } catch (e) {
    console.error(e);
    loading.remove();
    failed = true;
    root.append(h("p.callout.warn", "This chapter failed to load. Try reloading the page."));
  }
  if (!failed) root.append(chapterCheck(id, { title: meta.title, kind: "neuro" }));
  const { prev } = neighbours(id);
  const next = pathNext(id) || neighbours(id).next;
  structure(root, rail, next);
  root.append(finishCard(id, next));
  root.append(h("nav.ch-foot", { "aria-label": "Chapters" },
    prev ? h("a", { href: `/learn?c=${prev.id}` }, h("small", "Previous"), prev.title) : h("span"),
    next ? h("a.next", { href: `/learn?c=${next.id}` }, h("small", pathNext(id) ? `Next on your ${getMode().name} path` : "Next"), next.title) : h("a.next", { href: "/" }, h("small", "Finished"), "Back to the course map")));
}

/** The end of a chapter: your flower for it, and what to do next. */
function finishCard(id, next) {
  const card = h("section.finish");
  const draw = () => {
    const rec = progress.read()[id], stage = Math.max(2, stageOf(rec));
    card.replaceChildren(
      h("div.finish-fl", { html: flower(stage, id, { label: stage === 3 ? "This chapter's flower, in bloom" : "This chapter's flower, a bud" }) }),
      h("div.finish-t",
        h("div.eyebrow", stage === 3 ? "Chapter done · in bloom" : "Chapter opened · a bud"),
        h("h3", stage === 3 ? "Nice. This chapter's flower bloomed." : "Make a prediction above and this flower blooms."),
        h("p.muted", "Every chapter grows a flower in your garden on the course map. Questions from this chapter also turn up in the quiz."),
        h("div.finish-btns",
          next ? h("a.btn.primary", { href: `/learn?c=${next.id}` }, `Next: ${next.title} →`) : h("a.btn.primary", { href: "/" }, "Back to the course map →"),
          h("a.btn", { href: "/quiz" }, "Quiz yourself"))),
      h("div.finish-m.mascot-wrap", { html: mascot("neuro", stage === 3 ? "happy" : "think") }));
  };
  draw();
  document.addEventListener("click", (e) => { if (e.target.closest?.(".predict .opt")) setTimeout(draw, 250); });
  return card;
}

/** The course sidebar: a level switcher and this level's chapters with your progress. */
function sidebar(meta, done) {
  const lv = meta.level;
  const first = (l) => l.chapters.find((c) => c.live)?.id;
  // level header with previous / next level, instead of a dropdown that a sticky sidebar would clip
  const i = meta.levelIndex, near = (k) => (LEVELS[k] && first(LEVELS[k]) ? k : null);
  const prev = near(i - 1), nxt = near(i + 1);
  const menu = h("div.side-head", { style: { "--lc": lcOf(i) } },
    h("div.side-head-t", h("span.badge", String(i)), h("div", h("small", `Level ${i}`), h("b", lv.name.split(":")[0]))),
    h("div.side-nav",
      prev !== null ? h("a", { href: `/learn?c=${first(LEVELS[prev])}`, title: LEVELS[prev].name }, `‹ Level ${prev}`) : h("span"),
      h("a", { href: `/#${lv.id}` }, "map"),
      nxt !== null ? h("a", { href: `/learn?c=${first(LEVELS[nxt])}`, title: LEVELS[nxt].name }, `Level ${nxt} ›`) : h("span")));
  const seen = lv.chapters.filter((c) => done[c.id]?.visited || c.id === meta.id).length;
  return h("aside.lesson-side", { "aria-label": "Chapters in this level" },
    menu,
    h("div.side-l", `Chapters · ${seen} of ${lv.chapters.length} opened`),
    h("ol.side-list", lv.chapters.map((c, j) => {
      const st = done[c.id]?.predicted ? "done" : done[c.id]?.visited || c.id === meta.id ? "seen" : "";
      const stage = c.id === meta.id ? Math.max(1, stageOf(done[c.id])) : stageOf(done[c.id]);
      const inner = [h("span.side-n", `${meta.levelIndex}.${j + 1}`), h("span.side-t", c.title), stage ? h("span.side-fl", { title: ["", "here", "opened", "done: prediction made"][stage], html: flower(stage, c.id) }) : h("span.side-dot", { title: "not started" })];
      return h("li", c.live ? h("a", { href: `/learn?c=${c.id}`, class: c.id === meta.id ? "on" : "", "aria-current": c.id === meta.id ? "page" : null }, ...inner) : h("span.soon", ...inner));
    })),
    pathBox(meta.id, done),
    h("div.side-links", h("a", { href: "/#ladder" }, "All levels"), h("a", { href: "/roadmap" }, "Colab roadmap"), h("a", { href: "/quiz" }, "Quiz yourself")));
}

/** Learning modes: is this chapter on the reader's chosen path, and what comes after it there? */
function pathChip(id) {
  const m = getMode();
  if (!m || !m.stations) return null;
  return inMode(m, id) ? h("a.cchip.path", { href: `/paths?m=${m.id}`, style: { "--mc": m.color } }, `On your ${m.name} path`)
    : h("a.cchip.off", { href: `/paths?m=${m.id}` }, `Not on your ${m.name} path: fine to skip for now`);
}
function pathNext(id) {
  const m = getMode();
  if (!m || !m.stations || !inMode(m, id)) return null;
  const list = chaptersOf(m, LEVELS), i = list.findIndex((c) => c.id === id);
  return list[i + 1] || null;
}
function pathBox(id, done) {
  const m = getMode();
  if (!m) return h("a.side-path.none", { href: "/paths" }, h("b", "New here?"), h("span", "Pick a learning mode: Essentials, Builder, From zero…"));
  let asked = {};
  try { asked = JSON.parse(localStorage.getItem("ai-anatomy:ask") || "{}"); } catch { /* private mode */ }
  const pr = modeProgress(m, LEVELS, done, asked);
  return h("a.side-path", { href: `/paths?m=${m.id}`, style: { "--mc": m.color } },
    h("span.side-path-i", { html: badge(m.glyph, m.color) }),
    h("span", h("small", "Your path"), h("b", m.name), h("span.side-path-bar", h("i", { style: { width: `${(100 * pr.n) / Math.max(1, pr.total)}%` } })), h("small", `${pr.n} of ${pr.total} stops`)));
}

/** The parts after a chapter as tabs: [key, label, count, children]. Arrow keys move between tabs. */
function extrasTabs(parts) {
  const tabs = [], panels = [];
  const pick = (k, focus) => {
    tabs.forEach((t, i) => { const on = i === k; t.setAttribute("aria-selected", String(on)); t.tabIndex = on ? 0 : -1; panels[i].hidden = !on; });
    if (focus) tabs[k].focus();
    const bar = tabs[k].parentElement, t = tabs[k];   // keep the chosen tab in view when the row scrolls on a phone
    if (bar && (t.offsetLeft < bar.scrollLeft || t.offsetLeft + t.offsetWidth > bar.scrollLeft + bar.clientWidth)) bar.scrollLeft = t.offsetLeft - 12;
  };
  parts.forEach(([key, label, n], i) => {
    tabs.push(h("button.xt-tab", { type: "button", role: "tab", id: `xt-${key}`, "aria-controls": `xp-${key}`, "data-k": key, onclick: () => pick(i),
      onkeydown: (e) => { const d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0; if (d) { e.preventDefault(); pick((i + d + parts.length) % parts.length, true); } } },
      h("span", label), h("b", String(n))));
    panels.push(h("div.xt-panel", { role: "tabpanel", id: `xp-${key}`, "aria-labelledby": `xt-${key}` }, parts[i][3]));
  });
  const sec = h("section.block.field.extras", h("h2", "Beyond this chapter"),
    h("div.xt-tabs", { role: "tablist", "aria-label": "Beyond this chapter" }, tabs), panels);
  pick(0);
  return sec;
}

/** Number the sections as steps, add an at-a-glance strip, the outline rail and reading progress. */
/** What a step mostly is, from what it holds: [glyph, word]. */
function stepKind(sec) {
  const has = (sel) => sec.querySelector(sel);
  if (sec.classList.contains("ck-sec")) return ["check", "check yourself"];
  if (sec.classList.contains("field")) return ["field", "from practice"];
  if (has(".cuts")) return ["cuts", "go deeper"];
  if (has(".tw")) return ["cube", "step through"];
  if (has(".colab")) return ["build", "build"];
  if (has(".stage")) return ["sim", "simulation"];
  if (has("pre.code")) return ["code", "code"];
  if (has(".predict")) return ["predict", "predict"];
  if (has("table")) return ["table", "compare"];
  return ["read", "read"];
}

function structure(root, rail, next) {
  const blocks = [...root.querySelectorAll("section.block")].filter((s) => s.querySelector(":scope > h2"));
  const heads = blocks.map((s) => s.querySelector(":scope > h2"));
  const slug = (t) => t.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 48);
  heads.forEach((hd, i) => {
    hd.id ||= slug(hd.textContent);
    const [k, word] = stepKind(blocks[i]);
    hd.before(h("div.step-l", { "data-k": k }, h("span", { html: glyph(k) }, `Step ${i + 1}`), ` of ${heads.length}`, h("em", word)));
  });
  const words = (root.textContent || "").split(/\s+/).length;
  const mins = Math.max(3, Math.round(words / 220));
  root.querySelector("#cover-chips")?.prepend(h("span.cchip", `≈ ${mins} min`), h("span.cchip", `${heads.length} steps`));
  // what's in this chapter, counted from the page itself
  const count = (sel) => root.querySelectorAll(sel).length;
  const glance = [
    [count(".tw"), "3D step-through", "3D step-throughs"],
    [count(".stage"), "simulation", "simulations to poke"],
    [count(".predict"), "prediction", "predictions to make"],
    [count("pre.code:not(.tw-code)"), "code panel", "code panels"],
    [count("table"), "table", "tables"],
    [count(".colab"), "Colab project", "Colab projects"],
  ].filter(([n]) => n > 0);
  if (glance.length) root.querySelector(".ch-cover")?.after(h("div.glance", h("div.glance-l", "In this chapter"),
    h("div.glance-row", glance.map(([n, one, many]) => h("div.glance-i", h("b", String(n)), h("span", n === 1 ? one : many))))));
  // horizontal chips for narrow screens, the rail for wide ones
  if (heads.length >= 3) {
    const chips = heads.map((hd) => h("a.toc-chip", { href: `#${hd.id}` }, hd.textContent));
    (root.querySelector(".glance") || root.querySelector(".ch-cover"))?.after(h("nav.toc", { "aria-label": "On this page" }, h("span.toc-l", "On this page"), chips));
    const items = heads.map((hd, i) => h("li", h("a", { href: `#${hd.id}` }, h("span.rail-dot"), h("span", hd.textContent))));
    const meter = h("div.rail-meter", h("i"));
    const readout = h("div.rail-read", `0 of ${heads.length} read`);
    rail.append(h("div.rail-in",
      h("div.side-l", "Outline"),
      h("ol.rail-list", items), meter, readout,
      next ? h("a.rail-next", { href: `/learn?c=${next.id}` }, h("small", "Up next"), next.title) : null));
    if ("IntersectionObserver" in window) {
      let active = -1;
      const setActive = (i) => {
        active = i;
        items.forEach((li, j) => { li.classList.toggle("on", j === i); li.classList.toggle("past", j < i); });
        chips.forEach((c, j) => c.classList.toggle("on", j === i));
        meter.firstChild.style.width = `${((i + 1) / heads.length) * 100}%`;
        readout.textContent = `${i + 1} of ${heads.length} steps`;
      };
      const io = new IntersectionObserver((es) => { for (const e of es) if (e.isIntersecting) setActive(heads.indexOf(e.target)); }, { rootMargin: "-20% 0px -70% 0px" });
      heads.forEach((hd) => io.observe(hd));
      if (active < 0) setActive(0);
    }
  }
  const prog = h("div.read-progress", { "aria-hidden": "true" });
  document.body.append(prog);
  let ticking = false;
  const upd = () => { ticking = false; const max = document.documentElement.scrollHeight - innerHeight; prog.style.transform = `scaleX(${max > 0 ? Math.min(1, scrollY / max) : 0})`; };
  addEventListener("scroll", () => { if (!ticking) { ticking = true; requestAnimationFrame(upd); } }, { passive: true });
  upd();
}

