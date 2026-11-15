import { h } from "./core.js";

const LOGO = '<svg width="24" height="24" viewBox="0 0 32 32" aria-hidden="true"><rect width="32" height="32" rx="8" fill="var(--accent)"/><g stroke="var(--accent-ink)" stroke-width="1.8" opacity=".75"><path d="M9 10l7 6M9 22l7-6M16 16l7-6M16 16l7 6"/></g><g fill="var(--accent-ink)"><circle cx="9" cy="10" r="2.6"/><circle cx="9" cy="22" r="2.6"/><circle cx="16" cy="16" r="2.9"/><circle cx="23" cy="10" r="2.6"/><circle cx="23" cy="22" r="2.6"/></g></svg>';

const GEAR = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1.08-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09a1.65 1.65 0 0 0 1.51-1.08 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>';

/** Questions in the review deck that are due now, read straight from storage. */
function dueCount() {
  try { const d = JSON.parse(localStorage.getItem("ai-anatomy:quiz") || "{}").deck || {}; const now = Date.now(); return Object.values(d).filter((c) => c.due <= now).length; } catch { return 0; }
}

export function topbar(el) {
  const due = dueCount(), here = location.pathname;
  const link = (href, text, extra) => h("a", { href, "aria-current": here === href ? "page" : null }, text, extra);
  el.replaceChildren(
    h("a.brand", { href: "/", html: `${LOGO}<span>AI Anatomy</span>` }),
    h("nav.topnav", { "aria-label": "Site" },
      link("/#ladder", "Course map"), link("/roadmap", "Colab roadmap"),
      link("/quiz", "Quiz", due ? h("span.nav-badge", { title: `${due} due for review` }, String(due)) : null),
      h("a", { href: "https://github.com/srivtx/ai-anatomy" }, "GitHub"),
      h("a.nav-icon", { href: "/settings", title: "Settings", "aria-label": "Settings", "aria-current": here === "/settings" ? "page" : null, html: GEAR })),
  );
}
