import { h } from "./core.js";
import { code } from "./widgets.js";

export const STATUS = {
  established: ["Established", "var(--good)", "Reproduced widely; safe to rely on"],
  promising: ["Promising", "var(--c1)", "Strong evidence, still being tested at scale"],
  speculative: ["Speculative", "var(--c5)", "Interesting, early or contested"],
};

/** One collapsible "deeper cut" card: [chapter, name, status, idea, expect, code] */
export function cutCard([, name, status, idea, expect, snippet]) {
  const [label, color] = STATUS[status];
  let filled = false;
  const body = h("div.cut-body");
  const el = h("details.cut", { style: { "--sc": color } },
    h("summary", h("span.cut-name", name), h("span.idea-pill", { style: { color } }, label)),
    body);
  // build the (heavier) code block only when opened
  el.addEventListener("toggle", () => {
    if (!el.open || filled) return;
    filled = true;
    body.append(h("p", idea), h("p.cut-expect", h("b", "What to expect: "), expect), code({ Python: snippet }));
  });
  return el;
}
