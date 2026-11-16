// "Quick check": three questions from this chapter at its end. Answers feed the same
// spaced-review deck as the quiz page, and a finished check counts as a quiz on your bench.
import { h } from "./core.js";
import { quizState } from "./prefs.js";
import { mascot, burst } from "./pixel.js";
import { PLAY, isRight, answerText } from "./quizplay.js";

const KEYS = ["A", "B", "C", "D", "E"];

export function chapterCheck(id, { title = "", kind = "neuro", n = 3 } = {}) {
  const box = h("div.ck", h("p.ck-wait", "Loading questions…"));
  const sec = h("section.block.ck-sec",
    h("h2", "Quick check"),
    h("p.muted", "A few questions from this chapter. Misses come back in your review deck after a day, then 3, 7, 16 and 35 days."),
    box);
  let started = false;
  const go = async () => {
    if (started) return; started = true;
    const { BANK } = await import("./quizbank.js");
    // "which chapter is this?" is too easy on the chapter itself: use those only to fill up
    const mine = BANK.filter((q) => q.ch === id), rest = mine.filter((q) => q.kind !== "where");
    const pool = rest.length >= n ? rest : [...rest, ...mine.filter((q) => q.kind === "where")];
    if (!pool.length) { box.replaceChildren(h("p.ck-wait", "No questions for this chapter yet. ", h("a", { href: "/quiz" }, "Quiz yourself on everything else →"))); return; }
    // a different mix each visit, predictions first when there are any
    const pick = pool.map((q) => [q, Math.random() - (q.kind === "predict" ? 0.5 : q.type ? 0.4 : 0)]).sort((a, b) => a[1] - b[1]).slice(0, n).map(([q]) => q);
    run(pick, pool.length);
  };
  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) { io.disconnect(); go(); } }, { rootMargin: "800px 0px" });
    io.observe(sec);
  } else go();

  function run(qs, total) {
    const answers = qs.map(() => null);
    const score = h("div.ck-score");
    const cards = qs.map((q, k) => {
      if (q.type) {   // an interactive question: do the thing, then press Check
        const why = h("div.ck-why"), slot = h("div");
        let val = null;
        const btn = h("button.btn.primary.small", { type: "button", disabled: true, onclick: () => grade(k, val) }, "Check");
        const mount = (revealed) => slot.replaceChildren(PLAY[q.type].view(q, val, (v) => { val = v; btn.disabled = v == null; }, revealed));
        mount(false);
        return { el: h("div.ck-q", h("div.ck-n", `${k + 1} / ${qs.length} · ${PLAY[q.type].label.toLowerCase()}`), h("p.ck-qt", q.q), slot, h("div.ck-btns", btn), why), why, mount, btn };
      }
      const opts = q.options.map((o, j) => h("button.qz-opt", { type: "button", onclick: () => pick(k, j) }, h("span.qz-k", KEYS[j]), h("span", o)));
      const why = h("div.ck-why");
      return { el: h("div.ck-q", h("div.ck-n", `${k + 1} / ${qs.length}`), h("p.ck-qt", q.q), h("div.qz-opts", opts), why), opts, why };
    });
    function grade(k, v) {
      if (answers[k] !== null) return;
      answers[k] = v;
      const q = qs[k], c = cards[k], good = isRight(q, v);
      c.mount(true); c.btn.remove();
      c.el.classList.add(good ? "is-right" : "is-wrong");
      if (good) burst(c.el.querySelector(".qp"));
      c.why.replaceChildren(h("b", good ? "Right. " : "Not quite. "), good ? null : h("span", `Answer: ${answerText(q)}. `), h("span", { html: q.explain || "" }));
      if (answers.every((a) => a !== null)) finish();
    }
    function pick(k, j) {
      if (answers[k] !== null) return;
      answers[k] = j;
      const q = qs[k], c = cards[k], ok = j === q.answer;
      c.opts.forEach((b, i) => { b.disabled = true; b.classList.toggle("right", i === q.answer); b.classList.toggle("wrong", i === j && !ok); });
      c.el.classList.add(ok ? "is-right" : "is-wrong");
      if (ok) burst(c.opts[j]);
      c.why.replaceChildren(h("b", ok ? "Right. " : "Not quite. "), h("span", { html: q.explain || "" }));
      if (answers.every((a) => a !== null)) finish();
    }
    function finish() {
      const right = qs.filter((q, k) => isRight(q, answers[k])).length;
      const st = quizState.get();
      qs.forEach((q, k) => quizState.grade(st, q.id, isRight(q, answers[k])));
      st.hist = [...(st.hist || []), { at: Date.now(), n: qs.length, right, secs: 0, scope: `Quick check: ${title}`, mock: false, check: true }].slice(-50);
      quizState.save(st);
      const all = right === qs.length;
      score.replaceChildren(
        h("div.ck-m", { html: mascot(kind, all ? "happy" : right ? "think" : "sad") }),
        h("div",
          h("b", `${right} of ${qs.length} right`),
          h("p", all ? "All right. A can for your bench." : `${qs.length - right} added to your review deck. A can for your bench either way.`),
          h("div.ck-btns",
            h("button.btn", { type: "button", onclick: () => { started = false; box.replaceChildren(h("p.ck-wait", "Loading questions…")); go(); } }, total > qs.length ? "Three more" : "Try again"),
            h("a.btn", { href: "/quiz" }, `Full quiz (${total} from this chapter)`))));
      score.classList.add("on");
    }
    box.replaceChildren(...cards.map((c) => c.el), score);
  }
  return sec;
}
