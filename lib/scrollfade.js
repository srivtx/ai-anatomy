// Scroll areas inside the page show no scrollbar (see style.css). Instead, an edge fades out wherever there
// is more to scroll, and goes sharp once you reach the end. Runs on every page from topbar().
const SEL = "pre, .tbl-wrap, .hm-wrap, .nbv-tw, .jump, .toc, .xt-tabs, .tm-scroll, .ask-rail, .in-chips, .lesson-side, .lesson-rail, .st-side, .nbv-nav, .nbv-body, .log, .wide-canvas, .seg, .stk-in";
const FADE = 28;

function update(el) {
  const sx = el.scrollWidth - el.clientWidth > 2 && /(auto|scroll)/.test(getComputedStyle(el).overflowX);
  const sy = el.scrollHeight - el.clientHeight > 2 && /(auto|scroll)/.test(getComputedStyle(el).overflowY);
  if (!sx && !sy) { if (el.classList.contains("sf")) el.classList.remove("sf"); return; }
  const l = sx ? Math.min(FADE, el.scrollLeft) : 0, r = sx ? Math.min(FADE, el.scrollWidth - el.clientWidth - el.scrollLeft) : 0;
  const t = sy ? Math.min(FADE, el.scrollTop) : 0, b = sy ? Math.min(FADE, el.scrollHeight - el.clientHeight - el.scrollTop) : 0;
  el.classList.add("sf");
  el.style.setProperty("--sfx", `linear-gradient(90deg, transparent, #000 ${l}px, #000 calc(100% - ${r}px), transparent)`);
  el.style.setProperty("--sfy", `linear-gradient(180deg, transparent, #000 ${t}px, #000 calc(100% - ${b}px), transparent)`);
}

/** Drag sideways with the mouse to scroll wide things that have no scrollbar to grab. */
function draggable(el) {
  if (el.dataset.drag) return;
  el.dataset.drag = "1";
  let x0 = 0, s0 = 0, moved = false, down = false;
  el.addEventListener("pointerdown", (e) => { if (e.pointerType !== "mouse" || e.button) return; down = true; moved = false; x0 = e.clientX; s0 = el.scrollLeft; });
  el.addEventListener("pointermove", (e) => {
    if (!down) return;
    const dx = e.clientX - x0;
    if (Math.abs(dx) > 4) { moved = true; el.scrollLeft = s0 - dx; el.classList.add("dragging"); }
  });
  const end = () => { down = false; el.classList.remove("dragging"); };
  el.addEventListener("pointerup", end); el.addEventListener("pointerleave", end);
  el.addEventListener("click", (e) => { if (moved) { e.preventDefault(); e.stopPropagation(); moved = false; } }, true);
}

function scan() {
  for (const el of document.querySelectorAll(SEL)) {
    update(el);
    if (el.matches(".tm-scroll, .jump, .in-chips, .xt-tabs, .toc, .ask-rail") && el.classList.contains("sf")) draggable(el);
  }
}

let timer = 0;
// throttled, not debounced: pages that update text every frame would otherwise never get scanned
const soon = () => { if (!timer) timer = setTimeout(() => { timer = 0; scan(); }, 300); };

export function scrollFades() {
  if (window.__sf) return;
  window.__sf = true;
  document.addEventListener("scroll", (e) => { if (e.target instanceof Element && e.target.classList.contains("sf")) update(e.target); }, true);
  addEventListener("resize", soon);
  addEventListener("load", soon);
  new MutationObserver(soon).observe(document.body, { childList: true, subtree: true });
  soon();
}
