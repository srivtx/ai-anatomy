import { h } from "../lib/core.js";
import { segmented, stats, section, prose, callout } from "../lib/widgets.js";
import { DEEP_CUTS } from "../lib/deepcuts.js";
import { cutCard, STATUS } from "../lib/cuts.js";
import { find } from "../lib/catalog.js";

export default function (root) {
  let status = "all", query = "";
  const list = h("div");
  const st = stats([{ label: "Ideas" }, { label: "Established" }, { label: "Promising" }, { label: "Speculative" }]);
  const count = (s) => DEEP_CUTS.filter((c) => c[2] === s).length;
  st.set(0, String(DEEP_CUTS.length)); st.set(1, String(count("established")), "good"); st.set(2, String(count("promising"))); st.set(3, String(count("speculative")));
  const search = h("input.search", { type: "search", placeholder: "Search ideas, e.g. “attention”, “quantisation”, “memory”", "aria-label": "Search ideas", oninput: (e) => { query = e.target.value.toLowerCase(); draw(); } });
  function draw() {
    const hits = DEEP_CUTS.filter((c) => (status === "all" || c[2] === status) && (!query || (c[1] + " " + c[3]).toLowerCase().includes(query)));
    const groups = new Map();
    for (const c of hits) { if (!groups.has(c[0])) groups.set(c[0], []); groups.get(c[0]).push(c); }
    list.replaceChildren(...(hits.length ? [...groups].map(([ch, cs]) => {
      const meta = find(ch);
      return h("div.cut-group",
        h("div.eyebrow", meta ? h("a", { href: meta.live ? `/learn?c=${ch}` : `/#${meta.level.id}` }, `${meta.n} · ${meta.title}`) : ch),
        h("div.cuts", cs.map(cutCard)));
    }) : [h("p.muted", "No ideas match.")]));
  }

  root.append(
    section(null, prose(`<p>Beyond the headline techniques sits a long tail of ideas that experienced researchers know and most courses skip: strange phenomena inside trained networks, tricks that quietly became standard, and bold proposals that may or may not pan out. Each card has the idea, what to expect (quoted from the paper where the experiment needs serious compute), and code you can adapt. Status tells you how much weight to put on it.</p>`)),
    section("The atlas",
      h("div.stage",
        h("div.stage-controls",
          search,
          segmented({ label: "Status", options: [{ value: "all", label: "All" }, ...Object.entries(STATUS).map(([k, [label]]) => ({ value: k, label }))], value: status, onChange: (v) => { status = v; draw(); } })),
        h("div.stage-body", st.el, list)),
      callout("note", "How to use the status labels", `<ul style='margin:0;padding-left:18px'>${Object.values(STATUS).map(([l, c, d]) => `<li><b style="color:${c}">${l}</b>: ${d}.</li>`).join("")}</ul>`),
    ),
  );
  draw();
}
