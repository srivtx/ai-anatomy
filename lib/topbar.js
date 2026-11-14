import { h } from "./core.js";

const LOGO = '<svg width="24" height="24" viewBox="0 0 32 32" aria-hidden="true"><rect width="32" height="32" rx="8" fill="var(--accent)"/><g stroke="var(--accent-ink)" stroke-width="1.8" opacity=".75"><path d="M9 10l7 6M9 22l7-6M16 16l7-6M16 16l7 6"/></g><g fill="var(--accent-ink)"><circle cx="9" cy="10" r="2.6"/><circle cx="9" cy="22" r="2.6"/><circle cx="16" cy="16" r="2.9"/><circle cx="23" cy="10" r="2.6"/><circle cx="23" cy="22" r="2.6"/></g></svg>';

export function topbar(el) {
  el.replaceChildren(
    h("a.brand", { href: "/", html: `${LOGO}<span>AI Anatomy</span>` }),
    h("nav.topnav", { "aria-label": "Site" }, h("a", { href: "/#ladder" }, "Course map"), h("a", { href: "/roadmap" }, "Colab roadmap"), h("a", { href: "https://github.com/srivtx/ai-anatomy" }, "GitHub")),
  );
}
