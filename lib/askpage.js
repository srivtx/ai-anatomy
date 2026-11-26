// /ask: learn from zero by answering Neuro's questions. One small step at a time, in very easy English.
// A thread is a list of steps: Neuro says something, asks a question, asks for a guess, or gives you
// something to try. Your place in every thread is saved on this device.
import { h, s as svgEl } from "./core.js";
import { topbar } from "./topbar.js";
import { mascot, burst, badge } from "./pixel.js";
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
const journey = h("div.ask-journey", { "aria-label": "Your journey through the 18 conversations" });
const stage = h("section.ask-stage", { "aria-live": "polite" });
main.replaceChildren(
  h("header.ask-head",
    h("div", h("div.eyebrow", "Ask Neuro · learn from zero"), h("h1", "No walls of text. ", h("em", "Just questions.")),
      h("p.lede", "I'll ask you small questions. You answer. Wrong answers are fine: they're how you learn. By the end you'll know how a chatbot works, from the very first idea.")),
    h("div.ask-head-m", face("wow"))),
  journey,
  h("div.ask-layout", rail, stage));

// four parts of the journey, each with its own colour
const ZONE = ["var(--c3)", "var(--c1)", "var(--c6)", "var(--c2)"];
const zoneOf = (id) => Math.max(0, STATIONS.findIndex((s) => s.ask.includes(id)));
const PART_N = THREADS.map((t) => zoneOf(t.id));

/** A winding road through four coloured parts, a stone per conversation, the walked part painted in,
 *  a flag at the end and Neuro standing above the current stone. */
function drawJourney(curId) {
  const st = askState.get(), n = THREADS.length, W = 1000, H = 150, x = (i) => 34 + (i / (n - 1)) * (W - 68), y = (i) => 62 + Math.sin(i * 0.9) * 24;
  const pathTo = (k) => { let d = `M ${x(0)} ${y(0)}`; for (let i = 1; i <= k; i++) { const mx = (x(i - 1) + x(i)) / 2; d += ` C ${mx} ${y(i - 1)} ${mx} ${y(i)} ${x(i)} ${y(i)}`; } return d; };
  const d = pathTo(n - 1), ci = Math.max(0, THREADS.findIndex((t) => t.id === curId));
  const doneN = THREADS.filter((t) => st[t.id]?.done).length;
  let walked = -1; while (walked + 1 < n && st[THREADS[walked + 1].id]?.done) walked++;
  const zones = STATIONS.map((s, z) => {
    const idx = s.ask.map((id) => THREADS.findIndex((t) => t.id === id)), a = Math.min(...idx), b = Math.max(...idx);
    const x0 = x(a) - 22, x1 = x(b) + 22;
    return svgEl("g", { class: "aj-zone", style: `--zc:${ZONE[z]}` },
      svgEl("rect", { x: x0, y: 14, width: x1 - x0, height: 108, rx: 18 }),
      svgEl("text", { x: x0 + 12, y: 140 }, `${z + 1} · ${s.name}`));
  });
  const stones = THREADS.map((t, i) => {
    const done = st[t.id]?.done, cur = i === ci;
    return svgEl("a", { href: `?t=${t.id}`, class: `aj-stone${done ? " done" : ""}${cur ? " cur" : ""}`, style: `--zc:${ZONE[PART_N[i]]}`, onclick: (e) => { e.preventDefault(); open(t.id, true); } },
      cur ? svgEl("circle", { cx: x(i), cy: y(i), r: 19, class: "aj-ring" }) : null,
      svgEl("circle", { cx: x(i), cy: y(i), r: cur ? 13 : 10 }),
      svgEl("text", { x: x(i), y: y(i) + 4, "text-anchor": "middle" }, done ? "✓" : String(i + 1)),
      svgEl("title", {}, `${i + 1}. ${t.title}${done ? " (finished)" : ""}`));
  });
  const fx = x(n - 1) + 4, fy = y(n - 1);
  const flag = svgEl("g", { class: `aj-flag${doneN === n ? " won" : ""}` },
    svgEl("line", { x1: fx + 10, y1: fy - 6, x2: fx + 10, y2: fy - 44 }),
    svgEl("path", { d: `M ${fx + 11} ${fy - 44} h 20 l -6 7 l 6 7 h -20 z` }));
  journey.replaceChildren(
    svgEl("svg", { viewBox: `0 0 ${W} ${H}`, class: "aj-svg", role: "img", "aria-label": `${doneN} of ${n} conversations finished` },
      ...zones, svgEl("path", { d, class: "aj-road" }), walked >= 1 ? svgEl("path", { d: pathTo(walked), class: "aj-walked" }) : null,
      svgEl("path", { d, class: "aj-dash" }), flag, ...stones),
    h("span.aj-me", { style: { left: `${(x(ci) / W) * 100}%`, top: `${(y(ci) / H) * 100}%` }, html: mascot("neuro", "happy") }),
    h("span.aj-n", h("b", String(doneN)), ` of ${n} done`));
}

function drawRail(curId) {
  drawJourney(curId);
  const st = askState.get();
  rail.replaceChildren(...STATIONS.map((s, z) => h("div.ask-rs", { style: { "--zc": ZONE[z] } }, h("small", h("i"), s.name), h("ol", s.ask.map((id) => {
    const t = threadById(id), n = THREADS.indexOf(t) + 1, d = st[id]?.done, started = st[id]?.step > 0;
    return h("li", h("a.ask-rl", { href: `?t=${id}`, "aria-current": id === curId ? "true" : null, "data-state": d ? "done" : started ? "here" : "", onclick: (e) => { e.preventDefault(); open(id, true); } },
      h("span.ask-rn", d ? "✓" : String(n)), h("span", t.title)));
  })))));
  // on phones the rail is a sideways strip: bring the current thread into view without moving the page
  const here = rail.querySelector('[aria-current="true"]');
  if (here && rail.scrollWidth > rail.clientWidth) rail.scrollLeft = here.offsetLeft - rail.offsetLeft - 16;
}

/** One bead per step, shaped by what the step asks of you, so you can see what's coming. */
const KIND = (st) => st.recap ? ["end", "★", "the recap"] : st.try ? ["try", "▶", "something to try"] : st.guess ? ["guess", "#", "a guess"] : st.ask ? ["ask", "?", "a question"] : ["say", "", "Neuro explains"];

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
  const n = THREADS.indexOf(t) + 1, nextT = THREADS[n], z = PART_N[n - 1];
  stage.style.setProperty("--zc", ZONE[z]);
  const beads = h("ol.ask-beads", { "aria-label": `${steps.length} steps` }, steps.map((st) => {
    const [k, g, label] = KIND(st);
    return h("li", { "data-k": k, title: label }, g);
  }));
  const words = steps.find((x) => x.words)?.words || [];
  const coming = words.length ? h("div.ask-coming", h("small", saved.done ? "Words you learned here" : "Words you'll pick up"),
    ...words.map(([w]) => h("span", { "data-got": saved.done ? "true" : null }, w))) : null;
  const chat = h("div.ask-chat");
  const dock = h("div.ask-dock");
  const ch = find(t.chapter);
  stage.replaceChildren(
    h("div.ask-th", h("span.ask-tn", String(n)),
      h("div", h("small.ask-part", `Part ${z + 1} · ${STATIONS[z].name} · ${n} of ${THREADS.length}`), h("h2", t.title), h("p.muted", t.blurb)),
      saved.done ? h("button.btn.small", { type: "button", onclick: () => { askState.set(t.id, { step: 0, done: false }); open(t.id); } }, "Start again") : null),
    beads, coming, chat, dock);
  const setProg = (i) => [...beads.children].forEach((li, k) => { li.dataset.s = k < i ? "done" : k === i ? "now" : ""; });

  // live messages show Neuro "typing" for a moment first; replayed ones (and reduced motion) appear at once
  let live = false;
  const still = () => document.documentElement.dataset.motion === "reduce" || (matchMedia("(prefers-reduced-motion: reduce)").matches && document.documentElement.dataset.motion !== "full");
  const say = (html, mood) => {
    const b = h("div.ask-b"), m = h("div.ask-msg.neuro", face(mood), b);
    if (live && !still()) {
      b.classList.add("typing"); b.append(h("span.ask-dots", h("i"), h("i"), h("i")));
      setTimeout(() => { b.classList.remove("typing"); b.innerHTML = html; }, 420);
    } else b.innerHTML = html;
    chat.append(m); return m;
  };
  const me = (text) => chat.append(h("div.ask-msg.me", h("div.ask-b", text)));
  const scroll = (el) => { if (!el || !interacted) return; const r = el.getBoundingClientRect(); if (r.bottom > innerHeight - 40) window.scrollBy({ top: r.bottom - innerHeight + 140, behavior: "smooth" }); };

  /** Render step i. replay: draw it as already answered, without waiting. Resolves when the learner moves on. */
  function run(i, replay) {
    const st = steps[i];
    live = !replay && i > 0;
    setProg(i);
    if (!replay) askState.set(t.id, { step: i });
    return new Promise((done) => {
      let reached = false;
      if (st.recap) {
        const m = say(st.say || (nextT ? "That's it for this one. Here's what you now know." : "That's the whole journey. Here's the last part."), "wow");
        const card = h("div.ask-done",
          h("div.ask-done-h", h("span.ask-done-i", { html: badge("star", ZONE[z]) }), h("div", h("small", `Conversation ${n} of ${THREADS.length} · done`), h("b", t.title))),
          h("div.ask-done-g",
            h("div", h("small", "What you learned"), h("ul.ask-recap", st.recap.map((r) => h("li", { html: r })))),
            st.words ? h("div", h("small", "New words"), h("div.ask-cards", st.words.map(([w, d]) => h("div.ask-card", h("b", w), h("span", d))))) : null));
        chat.append(card);
        coming?.querySelectorAll("span").forEach((x) => { x.dataset.got = "true"; });
        if (coming) coming.firstChild.textContent = "Words you learned here";
        setProg(steps.length);
        askState.set(t.id, { step: i, done: true });
        drawRail(t.id);
        dock.replaceChildren(h("div.ask-end",
          nextT ? h("button.ask-up", { type: "button", onclick: () => open(nextT.id, true) },
            h("small", "Up next"), h("b", `${n + 1}. ${nextT.title} →`), h("span", nextT.blurb))
            : h("a.ask-up", { href: "/paths" }, h("small", "You finished all 18"), h("b", "Choose your next path →"), h("span", "Pick a learning mode and go deeper, with real models training in your browser.")),
          ch ? h("a.btn", { href: `/learn?c=${ch.id}` }, `See it move: ${ch.title}`) : null));
        if (!replay) { burst(card.querySelector(".ask-done-i")); scroll(dock); }
        return done();
      }
      const m = say(st.say || st.ask || st.guess || st.try, st.mood);
      let vz = null;
      if (st.viz) { vz = V.make(st.viz, { onGoal: () => goal() }); chat.append(h("div.ask-viz", vz.el || vz)); }
      // moving on is not an answer, so it leaves no bubble behind: the chat keeps only real replies
      const next = (label = st.go || "Got it →") => h("button.btn.primary.ask-next", { type: "button", onclick: () => { dock.replaceChildren(); done(); } }, label);

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
        const show = () => { out.textContent = fmt(inp.value); inp.style.setProperty("--p", `${(100 * (inp.value - g[0])) / (g[1] - g[0])}%`); };
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
        scroll(chat.lastChild);
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
