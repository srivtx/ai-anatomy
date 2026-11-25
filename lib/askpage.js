// /ask: learn from zero by answering Neuro's questions. One small step at a time, in very easy English.
// A thread is a list of steps: Neuro says something, asks a question, asks for a guess, or gives you
// something to try. Your place in every thread is saved on this device.
import { h } from "./core.js";
import { topbar } from "./topbar.js";
import { mascot, burst } from "./pixel.js";
import { THREADS, threadById, askState } from "./ask/index.js";
import { MODES, stationsOf } from "./paths.js";
import { LEVELS, find } from "./catalog.js";

topbar(document.getElementById("top"));
const main = document.getElementById("ask");
const params = new URLSearchParams(location.search);
const parts = {};
const loadPart = (p) => (parts[p] ||= import(`./ask/${p}.js`).then((m) => m.default));
const viz = import("./askviz.js");

const STATIONS = stationsOf(MODES.find((m) => m.id === "zero"), LEVELS).filter((s) => s.ask?.length);
const face = (mood = "happy") => h("span.ask-face", { "aria-hidden": "true", html: mascot("neuro", mood) });

const rail = h("nav.ask-rail", { "aria-label": "Conversations" });
const stage = h("section.ask-stage", { "aria-live": "polite" });
main.replaceChildren(
  h("header.ask-head",
    h("div", h("div.eyebrow", "Ask Neuro · learn from zero"), h("h1", "No walls of text. Just questions."),
      h("p.lede", "I'll ask you small questions. You answer. Wrong answers are fine: they're how you learn. By the end you'll know how a chatbot works, from the very first idea.")),
    h("div.ask-head-m", face("wow"))),
  h("div.ask-layout", rail, stage));

function drawRail(curId) {
  const st = askState.get();
  rail.replaceChildren(...STATIONS.map((s) => h("div.ask-rs", h("small", s.name), h("ol", s.ask.map((id) => {
    const t = threadById(id), n = THREADS.indexOf(t) + 1, d = st[id]?.done, started = st[id]?.step > 0;
    return h("li", h("a.ask-rl", { href: `?t=${id}`, "aria-current": id === curId ? "true" : null, "data-state": d ? "done" : started ? "here" : "", onclick: (e) => { e.preventDefault(); open(id, true); } },
      h("span.ask-rn", d ? "✓" : String(n)), h("span", t.title)));
  })))));
  // on phones the rail is a sideways strip: bring the current thread into view without moving the page
  const here = rail.querySelector('[aria-current="true"]');
  if (here && rail.scrollWidth > rail.clientWidth) rail.scrollLeft = here.offsetLeft - rail.offsetLeft - 16;
}

let token = 0, interacted = false;
stage.addEventListener("click", () => { interacted = true; });
document.addEventListener("keydown", () => { interacted = true; });
async function open(id, push) {
  const t = threadById(id) || THREADS[0], my = ++token;
  if (push) history.pushState(null, "", `?t=${t.id}`);
  drawRail(t.id);
  const steps = (await loadPart(t.part))[t.id] || [];
  if (my !== token) return;
  const V = await viz;
  const saved = askState.get()[t.id] || {};
  const i0 = saved.done ? 0 : Math.min(saved.step || 0, steps.length - 1);
  const n = THREADS.indexOf(t) + 1, nextT = THREADS[n];
  const bar = h("div.ask-prog", h("i"));
  const chat = h("div.ask-chat");
  const dock = h("div.ask-dock");
  const ch = find(t.chapter);
  stage.replaceChildren(
    h("div.ask-th", h("span.ask-tn", String(n)), h("div", h("h2", t.title), h("p.muted", t.blurb)),
      saved.done ? h("button.btn.small", { type: "button", onclick: () => { askState.set(t.id, { step: 0, done: false }); open(t.id); } }, "Start again") : null),
    bar, chat, dock);
  const setProg = (i) => { bar.firstChild.style.width = `${(100 * i) / steps.length}%`; };

  const say = (html, mood) => { const m = h("div.ask-msg.neuro", face(mood), h("div.ask-b", { html })); chat.append(m); return m; };
  const me = (text) => chat.append(h("div.ask-msg.me", h("div.ask-b", text)));
  const scroll = (el) => { if (!el || !interacted) return; const r = el.getBoundingClientRect(); if (r.bottom > innerHeight - 40) window.scrollBy({ top: r.bottom - innerHeight + 140, behavior: "smooth" }); };

  /** Render step i. replay: draw it as already answered, without waiting. Resolves when the learner moves on. */
  function run(i, replay) {
    const st = steps[i];
    setProg(i);
    if (!replay) askState.set(t.id, { step: i });
    return new Promise((done) => {
      let reached = false;
      if (st.recap) {
        const m = say(st.say || "Here's what you just learned.", "wow");
        m.querySelector(".ask-b").append(
          h("ul.ask-recap", st.recap.map((r) => h("li", { html: r }))),
          st.words ? h("div.ask-words", h("small", "New words"), h("dl", st.words.flatMap(([w, d]) => [h("dt", w), h("dd", d)]))) : null);
        setProg(steps.length);
        askState.set(t.id, { step: i, done: true });
        drawRail(t.id);
        dock.replaceChildren(h("div.ask-end",
          nextT ? h("button.btn.primary", { type: "button", onclick: () => open(nextT.id, true) }, `Next: ${nextT.title} →`) : h("a.btn.primary", { href: "/paths" }, "Choose your next path →"),
          ch ? h("a.btn", { href: `/learn?c=${ch.id}` }, `See it move: ${ch.title}`) : null));
        if (!replay) { burst(m.querySelector(".ask-b")); scroll(dock); }
        return done();
      }
      const m = say(st.say || st.ask || st.guess || st.try, st.mood);
      let vz = null;
      if (st.viz) { vz = V.make(st.viz, { onGoal: () => goal() }); m.querySelector(".ask-b").append(vz.el || vz); }
      const next = (label = st.go || "Got it →") => h("button.btn.primary.ask-next", { type: "button", onclick: () => { me(label.replace(" →", "")); dock.replaceChildren(); done(); } }, label);

      if (st.ask) {
        let tries = 0;
        const opts = st.options.map((o, k) => h("button.ask-opt", { type: "button", "data-k": String(k + 1), onclick: () => pick(k) }, h("kbd", String(k + 1)), h("span", { html: o })));
        const pick = (k) => {
          const ok = Array.isArray(st.answer) ? st.answer.includes(k) : k === st.answer;
          tries++;
          if (ok || tries >= 2) {
            me(stripTags(st.options[k]));
            const r = say(ok ? (st.yes || "Yes!") : `${st.no ? st.no + " " : ""}The answer is <b>${st.options[Array.isArray(st.answer) ? st.answer[0] : st.answer]}</b>. ${st.yes || ""}`, ok ? "happy" : "think");
            if (ok) burst(opts[k]);
            dock.replaceChildren(next("Next →"));
            scroll(r);
          } else {
            opts[k].disabled = true; opts[k].classList.add("no");
            hint.innerHTML = st.no || "Not quite. Try another one.";
            hint.hidden = false;
          }
        };
        const hint = h("p.ask-hint", { hidden: true });
        if (replay) { me(stripTags(st.options[Array.isArray(st.answer) ? st.answer[0] : st.answer])); say(st.yes || "Yes!"); return done(); }
        dock.replaceChildren(h("div.ask-opts", opts), hint);
        scroll(dock);
        return;
      }
      if (st.guess) {
        const g = st.range, out = h("b.ask-gv"), inp = h("input", { type: "range", min: g[0], max: g[1], step: g[2] ?? 1, value: g[3] ?? (g[0] + g[1]) / 2, "aria-label": "Your guess" });
        const fmt = (v) => `${(+v).toLocaleString()}${st.unit ? " " + st.unit : ""}`;
        const show = () => { out.textContent = fmt(inp.value); };
        inp.addEventListener("input", show); show();
        const lock = () => {
          const v = +inp.value, ok = Math.abs(v - st.answer) <= (st.tol ?? 0);
          me(`My guess: ${fmt(v)}`);
          const r = say(`${ok ? "Close enough!" : `The real answer is <b>${fmt(st.answer)}</b>.`} ${st.yes || ""}`, ok ? "happy" : "think");
          dock.replaceChildren(next("Next →"));
          scroll(r);
        };
        if (replay) { say(`The answer is <b>${fmt(st.answer)}</b>. ${st.yes || ""}`); return done(); }
        dock.replaceChildren(h("div.ask-guess", h("div.ask-gr", inp, out), h("button.btn.primary", { type: "button", onclick: lock }, "Lock in my guess")));
        scroll(dock);
        return;
      }
      function goal() {
        if (reached || !st.try) return;
        reached = true;
        const r = say(st.done || "You did it!", "wow");
        burst(r.querySelector(".ask-b"));
        dock.replaceChildren(next("Next →"));
        scroll(r);
      }
      if (st.try) {
        if (replay) { say(st.done || "You did it!"); return done(); }
        dock.replaceChildren(h("button.btn.ask-skip", { type: "button", onclick: () => { me("Show me"); vz?.solve?.(); goal(); } }, "Skip: show me"));
        scroll(m);
        return;
      }
      if (replay) return done();
      dock.replaceChildren(next());
      scroll(m);
    });
  }

  for (let i = 0; i < steps.length; i++) {
    if (my !== token) return;
    await run(i, i < i0);
  }
}
const stripTags = (x) => String(x).replace(/<[^>]+>/g, "");

document.addEventListener("keydown", (e) => {
  if (e.target.closest("input, textarea") || e.metaKey || e.ctrlKey || e.altKey) return;
  const opt = stage.querySelector(`.ask-opt[data-k="${e.key}"]:not(:disabled)`);
  if (opt) { opt.click(); return; }
  if (e.key === "Enter") stage.querySelector(".ask-next")?.click();
});
addEventListener("popstate", () => open(new URLSearchParams(location.search).get("t") || THREADS[0].id));

// start where the learner left off: the asked thread, else the first unfinished one
const st0 = askState.get();
open(params.get("t") || (THREADS.find((t) => !st0[t.id]?.done) || THREADS[0]).id);
