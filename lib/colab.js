import { h } from "./core.js";
import { colabUrl, githubUrl } from "./catalog.js";
import { RESULTS } from "./practice.js";

const ICON = '<svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M7.5 7.2a4.8 4.8 0 1 0 0 9.6c1.3 0 2.5-.5 3.4-1.4l-1.7-1.7a2.4 2.4 0 1 1 0-3.4l1.7-1.7a4.8 4.8 0 0 0-3.4-1.4Zm9 0c-1.3 0-2.5.5-3.4 1.4l1.7 1.7a2.4 2.4 0 1 1 0 3.4l-1.7 1.7a4.8 4.8 0 1 0 3.4-8.2Z"/></svg>';

/** "Run it on Colab" card for one project. */
export function colabCard(p) {
  return h("div.colab",
    h("div.grow",
      h("div.eyebrow", `Project ${p.n} · ${p.hw === "CPU" ? "runs on CPU" : "Colab GPU (T4 is enough)"} · ~${p.mins} min`),
      h("b", p.title), h("p", p.blurb),
      RESULTS[p.file] ? h("details.expect", h("summary", "Results you should see"), h("ul", RESULTS[p.file].map((t) => h("li", t)))) : null),
    h("div.row",
      h("a.btn.primary", { href: colabUrl(p.file), target: "_blank", rel: "noopener", html: `${ICON}<span>Open in Colab</span>` }),
      h("a.btn", { href: githubUrl(p.file), target: "_blank", rel: "noopener" }, "View on GitHub")));
}
