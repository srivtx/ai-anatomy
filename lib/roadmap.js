import { h, progress } from "./core.js";
import { PHASES, PROJECTS, find, colabUrl, githubUrl } from "./catalog.js";
import { topbar } from "./topbar.js";
import { callout } from "./widgets.js";
import { getMode } from "./paths.js";
const MODE = getMode();

topbar(document.getElementById("top"));
const root = document.getElementById("roadmap");
const state = progress.read();
const total = PHASES.reduce((a, p) => a + p.projects.length, 0);
const doneCount = () => PHASES.flatMap((p) => p.projects).filter((p) => progress.read()["proj:" + p.file]?.done).length;
const lcOf = (i) => `var(--c${(i % 6) + 1})`;
const phaseOf = new Map(PHASES.flatMap((ph, i) => ph.projects.map((p) => [p.file, i])));
const hwLabel = (hw) => (hw === "CPU" ? "CPU is fine" : `GPU · ${hw}`);
const counter = h("b.rm-count");
const strip = h("div.rm-strip", { "aria-label": "Every project, one square each, coloured by phase" });
const phaseCounts = [];
function refresh() {
  const st = progress.read(), n = doneCount();
  counter.textContent = `${n} of ${total}`;
  strip.querySelectorAll("a").forEach((a) => a.classList.toggle("on", !!st["proj:" + a.dataset.f]?.done));
  PHASES.forEach((ph, i) => { const d = ph.projects.filter((p) => st["proj:" + p.file]?.done).length; if (phaseCounts[i]) phaseCounts[i].textContent = `${d} / ${ph.projects.length} done`; });
}

function project(p) {
  const key = "proj:" + p.file, i = phaseOf.get(p.file) ?? 0;
  const box = h("input", { type: "checkbox", checked: state[key]?.done || null, "aria-label": `Mark ${p.title} finished` });
  const card = h("article.rm-card", { id: `p-${p.file.slice(0, 2)}`, style: { "--lc": lcOf(i) } });
  const sync = () => card.classList.toggle("done", box.checked);
  box.addEventListener("change", () => { progress.mark(key, "done", box.checked); sync(); refresh(); });
  sync();
  card.append(
    h("div.rm-top", h("span.rm-n", p.file.slice(0, 2)), h("span.rm-chip", hwLabel(p.hw)), h("span.rm-chip", `~${p.mins} min`), MODE?.projects.includes(p.file) ? h("a.rm-chip.path", { href: `/paths?m=${MODE.id}`, style: { "--mc": MODE.color } }, `on your ${MODE.name} path`) : null,
      h("label.rm-done", box, h("span", "Finished"))),
    h("h3", p.title),
    h("p.rm-blurb", p.blurb),
    h("div.rm-when", h("small", "You're done when"), h("ul", p.done.map((d) => h("li", d)))),
    p.stretch?.length ? h("p.rm-stretch", h("b", "Stretch "), p.stretch.join(" · ")) : null,
    h("div.rm-links", p.chapters.map((c) => { const m = find(c); return m ? h("a", { href: `/learn?c=${c}` }, m.title) : null; })),
    h("div.rm-acts",
      h("button.btn.rm-prev", { type: "button", onclick: () => Promise.all([import("./nbview.js"), import("./practice.js")]).then(([m, { RESULTS }]) => m.openNotebook(p, { results: RESULTS[p.file] || [] })) }, "Preview the code"),
      h("a.btn.primary", { href: colabUrl(p.file), target: "_blank", rel: "noopener" }, "Open in Colab"),
      h("a.rm-src", { href: githubUrl(p.file), target: "_blank", rel: "noopener" }, "Source")));
  return card;
}

PHASES.forEach((ph, i) => strip.append(h("div.rm-sg", { style: { "--lc": lcOf(i) }, title: ph.name },
  ph.projects.map((p) => h("a", { href: `#p-${p.file.slice(0, 2)}`, "data-f": p.file, title: `${p.file.slice(0, 2)} · ${p.title}` })))));

const STEPS = [
  ["Open", "Click Open in Colab. It opens a copy from GitHub; File → Save a copy in Drive keeps your edits."],
  ["GPU", "Runtime → Change runtime type → T4 GPU for projects marked GPU. With Colab Pro pick L4 or A100 and raise the values marked SCALE."],
  ["Quick mode", "Every notebook has QUICK = False at the top. Set it to True for a 1-minute smoke run before you spend GPU time."],
  ["Keys", "Projects 19 and 46 call the Claude API (41 can, if you like). Add ANTHROPIC_API_KEY under the key icon (Secrets); it never goes in the notebook."],
];

root.append(
  h("header.rm-hero",
    h("div",
      h("div.eyebrow", "The build-it-yourself track"),
      h("h1", "The Colab roadmap"),
      h("p.lede", `${PROJECTS.length} projects that take you from a NumPy line fit to GRPO, Mamba, Triton kernels, agent loops and real ablation studies. Open one in Colab, run it top to bottom, then change something and run it again. That last step is where you learn.`)),
    h("div.rm-board",
      h("div.rm-board-h", h("div", h("small", "Finished"), counter), h("div.rm-m", { html: "" })),
      strip,
      h("div.rm-legend", PHASES.map((ph, i) => h("a", { href: `#ph-${i}`, style: { "--lc": lcOf(i) } }, h("i"), ph.name))))),
  h("section.block",
    h("h2", "How to run these"),
    h("div.rm-steps", STEPS.map(([t, d], i) => h("div.rm-step", h("span", String(i + 1)), h("b", t), h("p", d)))),
    callout("tip", "Work like a researcher", "<p>Before every run, write one sentence: <i>I expect X because Y.</i> After it, write what happened. Change one thing at a time, keep a baseline, and run three seeds before you believe a difference. The notebooks end with an <b>Experiments</b> cell of things to try. That's the part that turns reading into understanding.</p>")),
  ...PHASES.map((ph, i) => {
    phaseCounts[i] = h("span.rm-pc");
    return h("section.rm-phase", { id: `ph-${i}`, style: { "--lc": lcOf(i) } },
      h("div.rm-ph", h("div", h("div.eyebrow", `Phase ${i + 1}`), h("h2", ph.name), h("p.muted", ph.goal)), phaseCounts[i]),
      h("div.rm-grid", ph.projects.map(project)));
  }),
  h("section.block", { style: { paddingBottom: "64px" } },
    h("h2", "After the roadmap"),
    h("div.prose", { html: "<ul><li><b>Pick one paper a week</b> and reproduce its smallest figure. Our projects 14 to 16 are templates for exactly this.</li><li><b>Scale one thing</b>: take project 06 and train the biggest model Colab Pro lets you, with project 07's law predicting its loss before you start.</li><li><b>Write it up</b>: a one-page note with the question, the setup, a plot and what surprised you. Researchers are judged by these, and they force you to know what you actually found.</li></ul>" })),
);
import("./pixel.js").then(({ mascot }) => { const m = root.querySelector(".rm-m"); if (m) m.innerHTML = mascot("neuro", "happy", { label: "Neuro, with a laptop and a coffee" }); });
refresh();
