// Loads /chapters/<id>.js for /learn?c=<id> and frames it with header and footer.
import { h, progress } from "./core.js";
import { find, neighbours, projectsFor } from "./catalog.js";
import { colabCard } from "./colab.js";
import { PRACTICE } from "./practice.js";
import { KNOWN, OPEN } from "./known.js";
import { DEEP_CUTS } from "./deepcuts.js";
import { cutCard } from "./cuts.js";
import { topbar } from "./topbar.js";

topbar(document.getElementById("top"));
const root = document.getElementById("chapter");
const id = new URLSearchParams(location.search).get("c") || "";
const meta = find(id);

if (!meta || !meta.live || !/^[a-z0-9-]+$/.test(id)) {
  root.replaceChildren(h("div.ch-head", h("h1", "Chapter not found"), h("p.lede", h("a", { href: "/" }, "Back to the course map"))));
} else {
  document.title = `${meta.title} · AI Anatomy`;
  root.replaceChildren(
    h("header.ch-head",
      h("div.crumbs", h("a", { href: `/#${meta.level.id}` }, `Level ${meta.levelIndex} · ${meta.level.name}`), h("span", "/"), h("span", meta.n)),
      h("h1", meta.title),
      h("p.lede", meta.blurb)),
  );
  try {
    const mod = await import(`../chapters/${id}.js`);
    const body = h("div.ch-body");
    root.append(body);
    mod.default(body, { id });
    progress.mark(id, "visited");
    const fn = PRACTICE[id];
    if (fn) root.append(h("section.block.field",
      h("h2", "From the field"),
      h("p.muted", "What practitioners actually do, what bites them, and numbers worth remembering."),
      h("div.field-grid",
        h("div.field-card.works", h("div.eyebrow", "What works"), h("ul", fn.works.map((t) => h("li", t)))),
        h("div.field-card.traps", h("div.eyebrow", "Traps"), h("ul", fn.traps.map((t) => h("li", t)))),
        fn.numbers?.length ? h("div.field-card.nums", h("div.eyebrow", "Numbers to remember"), h("ul", fn.numbers.map((t) => h("li", t)))) : null),
      (KNOWN[id] || OPEN[id]) && h("div.field-grid.two",
        KNOWN[id] ? h("div.field-card.known", h("div.eyebrow", "What the big runs found"), h("ul", KNOWN[id].map((t) => h("li", t)))) : null,
        OPEN[id] ? h("div.field-card.open", h("div.eyebrow", "Still unknown"), h("ul", OPEN[id].map((t) => h("li", t)))) : null)));
    const cuts = DEEP_CUTS.filter((c) => c[0] === id);
    if (cuts.length) root.append(h("section.block",
      h("h2", "Deeper cuts"),
      h("p.muted", h("span", "Lesser-known ideas around this chapter, with code. Where a result needs more compute than a laptop, it's quoted from the paper. "), h("a", { href: "/learn?c=deep-cuts" }, "See all of them in the atlas.")),
      h("div.cuts", cuts.map(cutCard))));
    const projects = projectsFor(id);
    if (projects.length) root.append(h("section.block", h("h2", "Now build it"), h("p.muted", "Every project is a notebook you can run top to bottom on Google Colab. Read the code and change it, then try the stretch goals."), projects.map(colabCard)));
  } catch (e) {
    console.error(e);
    root.append(h("p.callout.warn", "This chapter failed to load. Try reloading the page."));
  }
  const { prev, next } = neighbours(id);
  root.append(h("nav.ch-foot", { "aria-label": "Chapters" },
    prev ? h("a", { href: `/learn?c=${prev.id}` }, h("small", "Previous"), prev.title) : h("span"),
    next ? h("a.next", { href: `/learn?c=${next.id}` }, h("small", "Next"), next.title) : h("a.next", { href: "/" }, h("small", "Finished"), "Back to the course map")));
}
