// Ask Neuro pictures, part 2: words, attention, transformers, GPUs, feedback, steps, agents, failures.
import { h, s } from "./core.js";
import { box, svg, rangeIn } from "./askkit.js";

/* 10. tokens: type anything and watch it cut into pieces with ID numbers */
const VOCAB = ["the", "a", "an", "is", "are", "was", "cat", "dog", "sat", "on", "mat", "and", "to", "of", "in", "it", "I", "you", "we", "love", "learn", "machine", "ing", "ed", "er", "s", "un", "re", "ly", "token", "word", "model"];
function cut(word) { // greedy longest match from a tiny vocabulary, else 3-letter chunks: good enough to see the idea
  const out = []; let w = word;
  while (w) {
    const lower = w.toLowerCase(), hit = VOCAB.filter((v) => lower.startsWith(v)).sort((a, b) => b.length - a.length)[0];
    const n = hit && hit.length > 1 ? hit.length : Math.min(3, w.length);
    out.push(w.slice(0, n)); w = w.slice(n);
  }
  return out;
}
const idOf = (piece) => { let x = 7; for (const c of piece.toLowerCase()) x = (x * 31 + c.charCodeAt(0)) % 50000; return x; };
function tokens({ onGoal, text = "The cat sat on the mat. Unlearnable!" }) {
  const inp = h("input.av-text", { type: "text", value: text, "aria-label": "Type some text", maxlength: 80 });
  const row = h("div.av-toks"), note = h("p.av-note");
  let edits = 0;
  function draw() {
    const pieces = inp.value.split(/(\s+|[.,!?])/).filter((x) => x && !/^\s+$/.test(x)).flatMap((w) => (/^[.,!?]$/.test(w) ? [w] : cut(w)));
    row.replaceChildren(...pieces.map((p, i) => h("span.av-tok", { style: { "--c": `var(--c${(i % 6) + 1})` } }, h("b", p), h("small", String(idOf(p))))));
    note.textContent = `${inp.value.length} letters became ${pieces.length} tokens. The model only ever sees the numbers.`;
  }
  inp.addEventListener("input", () => { draw(); if (++edits >= 3) onGoal?.(); });
  draw();
  return { el: box("tokens", inp, row, note), solve() { inp.value = "Unbelievably, the machine learned!"; draw(); } };
}

/* 11. word map: words as points; similar words sit close; the same arrow means the same change */
const WORDS = { king: [70, 40], queen: [70, 110], man: [190, 48], woman: [190, 118], boy: [262, 30], girl: [262, 100], cat: [300, 200], dog: [330, 178], puppy: [352, 202], kitten: [322, 224], apple: [80, 220], banana: [34, 246], pear: [122, 252] };
function wordmap({ onGoal }) {
  const W = 400, H = 270, g = s("g"), arrows = s("g");
  let picked = [];
  const note = h("p.av-note", "Tap two words to see how far apart they are.");
  const dist = (a, b) => Math.hypot(WORDS[a][0] - WORDS[b][0], WORDS[a][1] - WORDS[b][1]);
  Object.entries(WORDS).forEach(([w, [x, y]]) => {
    const n = s("g", { class: "av-word", tabindex: 0, role: "button" }, s("circle", { cx: x, cy: y, r: 6 }), s("text", { x: x + 9, y: y + 4 }, w));
    const pick = () => {
      picked = [...picked, w].slice(-2);
      g.querySelectorAll(".av-word").forEach((e) => e.classList.toggle("on", picked.includes(e.textContent)));
      if (picked.length === 2) { const d = dist(...picked); note.textContent = `${picked[0]} ↔ ${picked[1]}: distance ${Math.round(d)}. ${d < 60 ? "Close: similar meaning." : d < 160 ? "Medium: related." : "Far apart: different things."}`; onGoal?.(); }
    };
    n.addEventListener("click", pick); n.addEventListener("keydown", (e) => e.key === "Enter" && pick());
    g.append(n);
  });
  const arrow = (a, b) => s("line", { x1: WORDS[a][0], y1: WORDS[a][1], x2: WORDS[b][0], y2: WORDS[b][1], class: "av-warrow" });
  const tog = h("button.btn.small", { type: "button", onclick: () => { const on = !arrows.childNodes.length; arrows.replaceChildren(...(on ? [arrow("man", "woman"), arrow("king", "queen"), arrow("boy", "girl")] : [])); tog.textContent = on ? "Hide the arrows" : "Show man → woman"; if (on) note.textContent = "The arrow from man to woman is the same as king to queen and boy to girl. A direction on the map means something."; } }, "Show man → woman");
  return { el: box("wordmap", svg(W, H, arrows, g), h("div.av-row", tog), note), solve() { tog.click(); onGoal?.(); } };
}

/* 12. next word: a bar for every candidate; temperature flattens or sharpens it */
function nextword({ onGoal, prompt = "The cat sat on the", words = [["mat", 3.2], ["floor", 2.4], ["sofa", 2.1], ["chair", 1.7], ["roof", 1.1], ["moon", -0.6]] }) {
  let temp = 1;
  const bars = h("div.av-nw"), out = h("p.av-gen", h("span", prompt + " "), h("b", "…"));
  const probs = () => { const e = words.map(([, z]) => Math.exp(z / temp)), t = e.reduce((a, b) => a + b, 0); return e.map((v) => v / t); };
  function draw() {
    const p = probs();
    bars.replaceChildren(...words.map(([w], i) => h("div.av-nwr", h("span", w), h("span.av-bar", h("i", { style: { width: `${p[i] * 100}%` } })), h("b", `${Math.round(p[i] * 100)}%`))));
  }
  let picks = 0;
  const pick = () => { const p = probs(); let r = Math.random(), i = 0; while (i < p.length - 1 && (r -= p[i]) > 0) i++; out.lastChild.textContent = words[i][0]; if (++picks >= 3) onGoal?.(); };
  const st = rangeIn("temperature", 0.2, 3, 0.1, temp, (v) => { temp = v; draw(); });
  st.init();
  return { el: box("nextword", h("p.av-prompt", `“${prompt} ___”`), bars, h("div.av-ctl", st.el, h("button.btn.primary.small", { type: "button", onclick: pick }, "Pick a word")), out), solve() { pick(); pick(); pick(); } };
}

/* 13. attention: tap a word; arcs show which words it listens to (weights chosen by hand to show the idea) */
const SENT = ["The", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "too", "tired"];
const ATT = { 7: { 1: 0.62, 5: 0.14, 0: 0.05, 10: 0.1 }, 10: { 7: 0.45, 1: 0.35, 9: 0.12 }, 3: { 1: 0.4, 5: 0.35, 2: 0.15 }, 5: { 4: 0.5, 3: 0.3 }, 1: { 0: 0.6, 1: 0.3 } };
function attention({ onGoal }) {
  const row = h("div.av-sent"), arcsSvg = s("svg", { class: "av-arcs", viewBox: "0 0 100 30", preserveAspectRatio: "none" });
  const note = h("p.av-note", "Tap “it”. Which word does it listen to most?");
  const chips = SENT.map((w, i) => h("button.av-w", { type: "button", onclick: () => pick(i) }, w));
  row.append(...chips);
  function pick(i) {
    chips.forEach((c, j) => { c.classList.toggle("on", j === i); c.style.setProperty("--a", (ATT[i]?.[j] ?? 0).toFixed(2)); });
    const a = ATT[i];
    if (!a) { note.textContent = `“${SENT[i]}” mostly listens to itself and its neighbours here. Try “it” or “tired”.`; return; }
    const best = Object.entries(a).sort((x, y) => y[1] - x[1])[0];
    note.innerHTML = `“${SENT[i]}” listens most to <b>“${SENT[best[0]]}”</b> (${Math.round(best[1] * 100)}% of its attention). Darker words get more.`;
    if (i === 7) onGoal?.();
  }
  return { el: box("attention", row, note, h("small.muted", "These weights are picked by hand to show the idea; a real model learns its own.")), solve() { pick(7); } };
}

/* 14. stack: the same block, many times; a token climbs through */
function stack({ onGoal }) {
  let n = 4;
  const tower = h("div.av-tower"), note = h("p.av-note");
  function draw() {
    tower.replaceChildren(h("div.av-out", "next word"), ...Array.from({ length: n }, (_, i) => h("div.av-blk", { style: { "--i": n - 1 - i } }, h("span", `block ${n - i}`), h("small", "look around · think"))), h("div.av-in", "your words"));
    note.textContent = `${n} blocks. ${n >= 12 ? "GPT-2 small has 12. Big models have around 100." : "Each block has the same design, with its own numbers."}`;
    if (n >= 12) onGoal?.();
  }
  const sn = rangeIn("blocks", 1, 12, 1, n, (v) => { n = v; draw(); });
  sn.init();
  return { el: box("stack", tower, h("div.av-ctl", sn.el), note), solve() { sn.set(12); } };
}

/* 15. workers: one fast worker against a crowd of slow ones, on 4,096 small sums */
function workers({ onGoal }) {
  const cpu = h("div.av-wg.cpu", Array.from({ length: 4 }, () => h("i"))), gpu = h("div.av-wg.gpu", Array.from({ length: 256 }, () => h("i")));
  const bc = h("span.av-bar", h("i")), bg = h("span.av-bar.t", h("i")), note = h("p.av-note");
  let timer = null;
  const run = () => {
    clearInterval(timer);
    let t = 0;
    timer = setInterval(() => {
      t++;
      const c = Math.min(1, (t * 4 * 4) / 4096), g = Math.min(1, (t * 256) / 4096);
      bc.firstChild.style.width = `${c * 100}%`; bg.firstChild.style.width = `${g * 100}%`;
      cpu.classList.toggle("busy", c < 1); gpu.classList.toggle("busy", g < 1);
      note.textContent = g >= 1 && c < 1 ? `The crowd finished in ${t} ticks. The 4 fast workers have done ${Math.round(c * 100)}%.` : c >= 1 ? "Done. Many simple workers beat a few clever ones when every job is the same small sum." : "";
      if (c >= 1) { clearInterval(timer); onGoal?.(); }
    }, 40);
  };
  return { el: box("workers", h("div.av-wrow", h("div", h("b", "CPU: 4 fast workers"), cpu, bc), h("div", h("b", "GPU: 256 slow workers"), gpu, bg)), h("div.av-row", h("button.btn.primary.small", { type: "button", onclick: run }, "Do 4,096 sums")), note), solve: run };
}

/* 16. thumbs: pick the better answer; the reward score learns what you like */
const PAIRS = [
  ["How do I boil an egg?", "Put the egg in boiling water for 8 to 10 minutes, then into cold water.", "Eggs are laid by many birds. The chicken was tamed thousands of years ago."],
  ["Say hi to my friend Sam.", "Hi Sam! Nice to meet you.", "I cannot greet people."],
  ["What's 12 + 30?", "42.", "It is a big number, probably around 50."],
];
function thumbs({ onGoal }) {
  let i = 0, score = 0;
  const q = h("p.av-prompt"), a = h("button.av-ans", { type: "button" }), b = h("button.av-ans", { type: "button" }), meter = h("span.av-bar", h("i")), note = h("p.av-note");
  const order = [0, 1, 0];
  function draw() {
    if (i >= PAIRS.length) { q.textContent = "Done!"; a.hidden = b.hidden = true; note.textContent = "Your 3 choices are training data. The model is nudged to write more answers like the ones you picked."; onGoal?.(); return; }
    const [qq, good, bad] = PAIRS[i], flip = order[i];
    q.textContent = `“${qq}”`;
    a.textContent = flip ? bad : good; b.textContent = flip ? good : bad;
    a.onclick = () => choose(!flip); b.onclick = () => choose(!!flip);
  }
  function choose(goodPicked) { score += goodPicked ? 1 : 0; meter.firstChild.style.width = `${(score / PAIRS.length) * 100}%`; i++; draw(); }
  draw();
  return { el: box("thumbs", q, h("div.av-pair", a, b), h("div.av-rw", h("small", "helpfulness score"), meter), note), solve() { while (i < PAIRS.length) choose(true); } };
}

/* 17. steps: answer in one go, or write the steps */
function steps({ onGoal }) {
  const lines = ["17 × 24", "= 17 × 20 + 17 × 4", "= 340 + 68", "= 408"];
  const box1 = h("div.av-stepbox", h("small", "All at once"), h("p", "17 × 24 = ", h("b.bad", "418?"))), box2 = h("div.av-stepbox", h("small", "Step by step"), h("div.av-lines"));
  let k = 0;
  const btn = h("button.btn.primary.small", { type: "button", onclick: () => { if (k < lines.length) { box2.lastChild.append(h("p", lines[k++])); } if (k === lines.length) { btn.disabled = true; onGoal?.(); } } }, "Write the next step");
  return { el: box("steps", h("div.av-pair", box1, box2), h("div.av-row", btn)), solve() { while (k < lines.length) btn.click(); } };
}

/* 18. agent loop: think → act → look, around and around until done */
const TRACE = [
  ["think", "The user wants Paris weather in °F. I need today's weather first."],
  ["act", "weather(\"Paris\")"],
  ["look", "→ 18 °C, sunny"],
  ["think", "Now turn 18 °C into °F."],
  ["act", "calculator(\"18 * 9 / 5 + 32\")"],
  ["look", "→ 64.4"],
  ["think", "I have everything. Time to answer."],
  ["answer", "It's sunny in Paris, about 64 °F."],
];
function loop({ onGoal }) {
  const W = 220, H = 200, nodes = { think: [110, 34], act: [186, 150], look: [34, 150] };
  const ring = s("circle", { cx: 110, cy: 112, r: 74, class: "av-ring" });
  const dot = s("circle", { r: 9, class: "av-ball" });
  const g = Object.entries(nodes).map(([k, [x, y]]) => s("g", { class: `av-ln-${k}` }, s("circle", { cx: x, cy: y, r: 26, class: "av-node" }), s("text", { x, y: y + 5, "text-anchor": "middle", class: "av-nt" }, k)));
  const log = h("ol.av-log");
  let k = 0;
  const btn = h("button.btn.primary.small", { type: "button", onclick: () => {
    const [kind, text] = TRACE[k++];
    log.append(h("li", { "data-k": kind }, h("b", kind), h("code", text)));
    const [x, y] = nodes[kind] || [110, 112];
    dot.setAttribute("cx", x); dot.setAttribute("cy", y);
    if (k === TRACE.length) { btn.disabled = true; onGoal?.(); }
  } }, "Next move");
  return { el: box("loop", h("div.av-loopw", svg(W, H, ring, ...g, dot), log), h("div.av-row", btn)), solve() { while (k < TRACE.length) btn.click(); } };
}

/* 19. fail: a confident model, and the truth behind each answer */
const CLAIMS = [
  ["Who wrote Romeo and Juliet?", "Shakespeare.", true, "Seen thousands of times in training. Safe."],
  ["What's the birthday of Dr. Mira Okafor, a local dentist?", "March 3rd, 1978.", false, "Seen once or never. The model made up a likely-looking date."],
  ["Who is Mary Lee Pfeiffer's son?", "I don't know who that is.", false, "She is Tom Cruise's mother, and models that know “Tom Cruise's mother is Mary Lee Pfeiffer” often can't answer it the other way round. Facts are learned in one direction."],
];
function fail({ onGoal }) {
  let shown = 0;
  const cards = CLAIMS.map(([q, a, ok, why]) => {
    const r = h("p.av-why", { hidden: true }, h("b", ok ? "✓ True. " : "✗ Wrong. "), why);
    const c = h("button.av-claim", { type: "button", onclick: () => { if (!r.hidden) return; r.hidden = false; c.dataset.ok = String(ok); if (++shown === CLAIMS.length) onGoal?.(); } }, h("small", q), h("b", a), h("span.av-conf", "sounds sure"), r);
    return c;
  });
  return { el: box("fail", h("div.av-claims", cards), h("small.muted", "Tap each answer to check it.")), solve() { cards.forEach((c) => c.click()); } };
}

export const VIZ2 = { tokens, wordmap, nextword, attention, stack, workers, thumbs, steps, loop, fail };
