import { h, progress } from "./core.js";
import { PHASES, find, colabUrl, githubUrl } from "./catalog.js";
import { topbar } from "./topbar.js";
import { callout, table } from "./widgets.js";

topbar(document.getElementById("top"));
const root = document.getElementById("roadmap");
const state = progress.read();
const total = PHASES.reduce((a, p) => a + p.projects.length, 0);
const doneCount = () => PHASES.flatMap((p) => p.projects).filter((p) => progress.read()["proj:" + p.file]?.done).length;
const counter = h("span.pill.acc");
const refresh = () => (counter.textContent = `${doneCount()} / ${total} finished`);

function project(p) {
  const key = "proj:" + p.file;
  const box = h("input", { type: "checkbox", checked: state[key]?.done || null, "aria-label": `Mark ${p.title} finished` });
  box.addEventListener("change", () => { progress.mark(key, "done", box.checked); refresh(); });
  return h("article.proj",
    h("div.pn", p.file.slice(0, 2)),
    h("div",
      h("h3", p.title),
      h("p", p.blurb),
      h("b", { style: { fontSize: "13px" } }, "You're done when"),
      h("ul", p.done.map((d) => h("li", d))),
      p.stretch?.length ? h("p", { style: { marginTop: "6px", fontSize: "13.5px" } }, h("b", "Stretch: "), p.stretch.join(" · ")) : null,
      h("div.tags",
        h("span.pill", p.hw === "CPU" ? "CPU is fine" : "GPU: T4"),
        h("span.pill", `~${p.mins} min`),
        p.chapters.map((c) => { const m = find(c); return m ? h("a.pill.acc", { href: `/learn?c=${c}`, style: { textDecoration: "none" } }, m.title) : null; }))),
    h("div.acts",
      h("a.btn.primary.small", { href: colabUrl(p.file), target: "_blank", rel: "noopener" }, "Open in Colab"),
      h("a.btn.small", { href: githubUrl(p.file), target: "_blank", rel: "noopener" }, "Source"),
      h("label.row", { style: { fontSize: "13px", gap: "6px" } }, box, "Finished")));
}

root.append(
  h("header.ch-head",
    h("div.eyebrow", "The build-it-yourself track"),
    h("h1", "The Colab roadmap"),
    h("p.lede", "Thirty projects that take you from a NumPy line fit to GRPO, Mamba, Triton kernels and agent loops. Each one is a single notebook: open it in Colab, run it top to bottom, then change something and run it again. That last step is where you learn."),
    h("p", { style: { marginTop: "14px" } }, counter)),
  h("section.block",
    h("h2", "How to run these"),
    table(["Step", "What to do"], [
      ["1. Open", "Click Open in Colab. It opens a copy from GitHub; use File → Save a copy in Drive to keep your edits."],
      ["2. GPU", "Runtime → Change runtime type → T4 GPU for projects marked GPU. With Colab Pro you can pick L4 or A100 and raise the values marked SCALE in each notebook's config cell."],
      ["3. Quick mode", "Every notebook has QUICK = False at the top. Set it to True for a 1-minute smoke run that checks everything works before you spend GPU time."],
      ["4. Keys", "Only project 19 calls an API. Add ANTHROPIC_API_KEY under the key icon (Secrets) in the left sidebar; it never goes in the notebook."],
    ]),
    callout("tip", "Work like a researcher", "<p>Before every run, write one sentence: <i>I expect X because Y.</i> After it, write what happened. Change one thing at a time, keep a baseline, and run three seeds before you believe a difference. The notebooks end with an <b>Experiments</b> cell of things to try. That's the part that turns reading into understanding.</p>")),
  ...PHASES.map((ph, i) => h("section.phase",
    h("div.phase-h", h("span.n", `Phase ${i + 1}`), h("h2", ph.name)),
    h("p.muted", { style: { margin: 0 } }, ph.goal),
    ph.projects.map(project))),
  h("section.block", { style: { paddingBottom: "64px" } },
    h("h2", "After the roadmap"),
    h("div.prose", { html: "<ul><li><b>Pick one paper a week</b> and reproduce its smallest figure. Our projects 14 to 16 are templates for exactly this.</li><li><b>Scale one thing</b>: take project 06 and train the biggest model Colab Pro lets you, with project 07's law predicting its loss before you start.</li><li><b>Write it up</b>: a one-page note with the question, the setup, a plot and what surprised you. Researchers are judged by these, and they force you to know what you actually found.</li></ul>" })),
);
refresh();
