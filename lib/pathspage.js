// /paths: pick a learning mode, see it as a metro line, and see how all the modes share stations.
import { h, s, progress } from "./core.js";
import { LEVELS, ALL, colabUrl } from "./catalog.js";
import { topbar } from "./topbar.js";
import { badge, mascot } from "./pixel.js";
import { MODES, getMode, setMode, modeById, stationsOf, projectsOf, modeProgress } from "./paths.js";
import { THREADS, threadById, askState } from "./ask/index.js";

topbar(document.getElementById("top"));
const main = document.getElementById("paths");
const done = progress.read();
const asked = askState.get();
let cur = modeById(new URLSearchParams(location.search).get("m")) || getMode() || MODES[1];

const cards = h("div.md-grid", { role: "radiogroup", "aria-label": "Learning modes" });
const detail = h("section.md-detail", { "aria-live": "polite" });

function drawCards() {
  const mine = getMode();
  cards.replaceChildren(...MODES.map((m) => {
    const pr = modeProgress(m, LEVELS, done, asked);
    return h("button.md-card", { type: "button", role: "radio", "aria-checked": String(m.id === cur.id), style: { "--mc": m.color }, onclick: () => { cur = m; history.replaceState(null, "", `?m=${m.id}`); drawCards(); drawDetail(); } },
      h("span.md-badge", { html: badge(m.glyph, m.color) }),
      h("span.md-top", h("b", m.name), mine?.id === m.id ? h("span.md-mine", "your path") : null),
      h("span.md-who", m.who),
      h("span.md-meta", h("span", `~${m.hours} h`), h("span", `${pr.total} stops`), pr.n ? h("span", `${Math.round((100 * pr.n) / pr.total)}% done`) : null),
      h("span.md-bar", h("i", { style: { width: `${(100 * pr.n) / Math.max(1, pr.total)}%` } })));
  }));
}

/** One mode as a metro line: stations are groups, stops are chapters or Socratic threads. */
function metro(m) {
  const stations = stationsOf(m, LEVELS);
  return h("ol.metro", { style: { "--mc": m.color } }, stations.map((st, si) => {
    const stops = [
      ...(st.ask || []).map((id) => { const t = threadById(id), d = asked[id]?.done; return { href: `/ask?t=${id}`, title: t.title, sub: "Questions with Neuro", state: d ? "done" : asked[id]?.step ? "here" : "", kind: "ask" }; }),
      ...st.chapters.map((c) => { const r = done[c.id]; return { href: `/learn?c=${c.id}`, title: c.title, sub: `Chapter ${c.n}`, state: r?.predicted ? "done" : r?.visited ? "here" : "", kind: "ch" }; }),
    ];
    const n = stops.filter((x) => x.state === "done").length;
    return h("li.metro-st",
      h("div.metro-sh", h("span.metro-dot.big"), h("div", h("small", `Station ${si + 1}`), h("b", st.name)), h("span.metro-n", `${n}/${stops.length}`)),
      h("ol.metro-stops", stops.map((x) => h("li", h("a.metro-stop", { href: x.href, "data-state": x.state, "data-kind": x.kind },
        h("span.metro-dot"), h("span.metro-t", h("b", x.title), h("small", x.sub)), x.state === "done" ? h("span.metro-ok", "✓") : x.state === "here" ? h("span.metro-here", "started") : null)))));
  }));
}

function drawDetail() {
  const m = cur, pr = modeProgress(m, LEVELS, done, asked), mine = getMode()?.id === m.id;
  const next = pr.next ? (pr.next.ask ? { href: `/ask?t=${pr.next.ask}`, t: threadById(pr.next.ask).title } : { href: `/learn?c=${pr.next.id}`, t: pr.next.title }) : null;
  const projects = projectsOf(m);
  detail.style.setProperty("--mc", m.color);
  detail.replaceChildren(
    h("div.md-hero",
      h("div.md-hero-t",
        h("div.eyebrow", mine ? "Your path" : "Path preview"),
        h("h2", m.name),
        h("p.md-promise", m.promise),
        h("div.md-ctas",
          next ? h("a.btn.primary", { href: next.href, onclick: () => setMode(m.id) }, `${pr.n ? "Continue" : "Start"}: ${next.t} →`) : h("span.md-fin", "Path complete. Sit a mock exam on the quiz page."),
          mine ? h("button.btn", { type: "button", onclick: () => { setMode(null); drawCards(); drawDetail(); } }, "Stop following")
            : h("button.btn", { type: "button", onclick: () => { setMode(m.id); drawCards(); drawDetail(); } }, "Follow this path"))),
      h("div.md-ring", ring(pr.n, pr.total, m.color), h("div", { html: badge(m.glyph, m.color, { label: m.name }) }))),
    h("div.md-body",
      h("div.md-line", h("h3", m.stations ? "The route" : "Every level, in order"), metro(m)),
      h("aside.md-side",
        h("div.md-box", h("h3", "After this path you can"), h("ul.md-after", m.after.map((a) => h("li", a)))),
        projects.length ? h("div.md-box", h("h3", "Colab projects on this path"), h("ul.md-proj", projects.map((p) => h("li", h("a", { href: colabUrl(p.file), target: "_blank", rel: "noopener" }, h("span.md-pn", p.n), p.title), h("small", `${p.hw} · ~${p.mins} min`))))) : null,
        m.id === "zero" ? h("div.md-box.md-neuro", h("div", { html: mascot("neuro", "happy") }), h("p", "No maths you haven't seen, and no reading walls. I ask, you answer, and we build it up together.")) : null)));
}

function ring(n, total, color) {
  const r = 40, c = 2 * Math.PI * r, f = total ? n / total : 0;
  return s("svg", { viewBox: "0 0 100 100", class: "md-ringsvg", role: "img", "aria-label": `${n} of ${total} stops done` },
    s("circle", { cx: 50, cy: 50, r, fill: "none", stroke: "var(--sunken)", "stroke-width": 10 }),
    f > 0 ? s("circle", { cx: 50, cy: 50, r, fill: "none", stroke: color, "stroke-width": 10, "stroke-linecap": "round", "stroke-dasharray": `${c * f} ${c}`, transform: "rotate(-90 50 50)" }) : null,
    s("text", { x: 50, y: 49, "text-anchor": "middle", class: "md-ring-n" }, `${Math.round(f * 100)}%`),
    s("text", { x: 50, y: 64, "text-anchor": "middle", class: "md-ring-l" }, `${n}/${total}`));
}

/** All modes as one transit map: one lane per mode, stops placed by where the chapter sits in the course. */
function transit() {
  const modes = MODES.filter((m) => m.stations && m.id !== "zero");
  const order = new Map(ALL.map((c, i) => [c.id, i]));
  const used = [...new Set(modes.flatMap((m) => stationsOf(m, LEVELS).flatMap((s) => s.chapters.map((c) => c.id))))].sort((a, b) => order.get(a) - order.get(b));
  const W = Math.max(900, used.length * 15 + 160), LANE = 46, H = modes.length * LANE + 60, x0 = 150;
  const xOf = (id) => x0 + (used.indexOf(id) / Math.max(1, used.length - 1)) * (W - x0 - 30);
  const share = new Map(used.map((id) => [id, modes.filter((m) => stationsOf(m, LEVELS).some((s) => s.chapters.some((c) => c.id === id))).length]));
  const kids = [];
  // level bands along the top, so position reads as "how far into the course"
  LEVELS.forEach((lv, i) => {
    const ids = used.filter((id) => ALL.find((c) => c.id === id).levelIndex === i);
    if (!ids.length) return;
    const a = xOf(ids[0]) - 6, b = xOf(ids[ids.length - 1]) + 6;
    kids.push(s("rect", { x: a, y: 8, width: Math.max(6, b - a), height: 6, rx: 3, fill: `var(--c${(i % 6) + 1})`, opacity: 0.55 }, s("title", {}, `Level ${i}: ${lv.name}`)));
  });
  modes.forEach((m, k) => {
    const y = 40 + k * LANE, ids = stationsOf(m, LEVELS).flatMap((s) => s.chapters.map((c) => c.id)).sort((a, b) => order.get(a) - order.get(b));
    const pts = ids.map((id) => [xOf(id), y]);
    let d = `M ${x0 - 20} ${y}`;
    for (const [x] of pts) d += ` L ${x} ${y}`;
    kids.push(s("text", { x: 0, y: y + 4, class: "tm-name" }, m.name));
    kids.push(s("path", { d, fill: "none", stroke: m.color, "stroke-width": 6, "stroke-linecap": "round" }));
    ids.forEach((id) => {
      const c = ALL.find((q) => q.id === id), hub = share.get(id) > 1, r = done[id]?.visited;
      kids.push(s("a", { href: `/learn?c=${id}` },
        s("circle", { cx: xOf(id), cy: y, r: hub ? 6.5 : 4.5, fill: r ? m.color : "var(--surface)", stroke: hub ? "var(--ink)" : m.color, "stroke-width": hub ? 2.2 : 2 }),
        s("title", {}, `${c.title}${hub ? ` · on ${share.get(id)} paths` : ""}`)));
    });
  });
  // interchanges: a thin vertical tie through chapters that several modes share
  used.filter((id) => share.get(id) > 2).forEach((id) => {
    const lanes = modes.map((m, k) => (stationsOf(m, LEVELS).some((s) => s.chapters.some((c) => c.id === id)) ? 40 + k * LANE : null)).filter((v) => v != null);
    kids.unshift(s("line", { x1: xOf(id), x2: xOf(id), y1: Math.min(...lanes), y2: Math.max(...lanes), stroke: "var(--line)", "stroke-width": 8, "stroke-linecap": "round" }));
  });
  return h("section.tm",
    h("div.tm-h", h("div.eyebrow", "How the paths overlap"), h("h2", "The course as a transit map"),
      h("p.muted", "Each line is a path; each dot is a chapter, placed by how far into the course it sits. Big dots are interchanges: chapters on several paths, which is a good hint that they matter. Filled dots are ones you've opened.")),
    h("div.tm-scroll", { tabindex: "0", "aria-label": "Transit map, scrolls sideways" }, s("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, class: "tm-svg" }, ...kids)));
}

main.replaceChildren(
  h("header.md-head",
    h("div", h("div.eyebrow", "Learning modes"), h("h1", "Nobody learns 104 chapters at once"),
      h("p.lede", "Pick the path that fits why you're here. Each one is a route through the same course: what to read, in what order, and what you can skip for now. Switch any time; your progress counts on every path.")),
    h("div.md-head-art", { "aria-hidden": "true" }, MODES.slice(0, 6).map((m, i) => h("span", { style: { "--mc": m.color, "--i": i }, html: badge(m.glyph, m.color) })))),
  cards, detail, transit());
drawCards();
drawDetail();
