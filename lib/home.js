import { h, loop, reducedMotion, progress } from "./core.js";
import { LEVELS, PROJECTS } from "./catalog.js";
import { topbar } from "./topbar.js";
import { MLP, dataset, bceLogit, sigmoid } from "./nn.js";
import { plane } from "./plot.js";

topbar(document.getElementById("top"));
const done = progress.read();

// Hero: a small network learning a two-armed spiral, live.
const viz = document.getElementById("hero-viz");
const pl = plane({ width: 420, height: 320, xr: [-1.25, 1.25], yr: [-0.95, 0.95], label: "A neural network learning to separate two spirals" });
const data = dataset("spiral", 240, { noise: 0.06, seed: 5 });
let net = new MLP([2, 16, 16, 1], { act: "tanh", seed: 2 });
let epoch = 0, loss = 0;
const cap = h("span"), capR = h("span", "2 → 16 → 16 → 1, tanh, Adam");
viz.replaceChildren(pl.el, h("p.canvas-cap", cap, capR));
function trainEpoch() {
  loss = 0;
  for (const p of data) {
    const r = bceLogit(net.forward(p.x)[0], p.y);
    loss += r.loss;
    net.backward([r.d]);
  }
  net.step(0.02, data.length);
  loss /= data.length;
  epoch++;
  if (epoch > 1400 || (loss < 0.02 && epoch > 400)) { epoch = 0; net = new MLP([2, 16, 16, 1], { act: "tanh", seed: 1 + ((Math.random() * 1e6) | 0) }); }
}
function draw() {
  pl.clear();
  pl.field((x, y) => sigmoid(net.forward([x, y])[0]), { res: 84, alpha: 0.55 });
  pl.scatter(data, { r: 3 });
  cap.textContent = `epoch ${epoch} · loss ${loss.toFixed(3)}`;
}
if (reducedMotion()) { for (let i = 0; i < 600; i++) trainEpoch(); draw(); }
else { const lp = loop(() => { for (let i = 0; i < 4; i++) trainEpoch(); draw(); }, pl.el); lp.start(); }

const live = LEVELS.flatMap((lv) => lv.chapters).filter((c) => c.live).length;
const seen = Object.values(done).filter((d) => d.visited).length;
document.getElementById("hero-stats").replaceChildren(
  ...[[live, "live chapters"], [PROJECTS.length, "Colab projects"], [LEVELS.length, "levels"], [seen, seen ? "opened by you" : "sign-ups needed"]]
    .map(([n, l]) => h("div.hs", h("b", String(n)), h("span", l))));
const lc = (i) => `var(--c${(i % 6) + 1})`;
// what's inside: one bar per level, its length the number of chapters, the filled part what you've opened
const maxCh = Math.max(...LEVELS.map((lv) => lv.chapters.length));
document.getElementById("jump").replaceChildren(
  h("div.ix-h", h("span", "What's inside"), h("span", "bar length = chapters · filled = opened by you")),
  ...LEVELS.map((lv, i) => {
    const opened = lv.chapters.filter((c) => done[c.id]?.visited).length;
    return h("a.ix-row", { href: `#${lv.id}`, style: { "--lc": lc(i) } },
      h("span.ix-b", String(i)), h("span.ix-name", lv.name.split(":")[0]),
      h("span.ix-track", h("span.ix-bar", { style: { "--w": `${(lv.chapters.length / maxCh) * 100}%`, "--f": `${(opened / lv.chapters.length) * 100}%` } }, h("span.ix-n", String(lv.chapters.length)))),
      h("span.ix-tag", lv.tagline));
  }));
document.getElementById("ladder").replaceChildren(
  ...LEVELS.map((lv, i) =>
    h("section.level", { id: lv.id, style: { "--lc": lc(i) } },
      h("div.level-h", h("div.n", h("span.badge", String(i)), `Level ${i}`), h("h2", lv.name), h("p", lv.tagline),
        h("div.level-prog", `${lv.chapters.filter((c) => done[c.id]?.visited).length} / ${lv.chapters.length} opened`)),
      h("div.cards", lv.chapters.map((c, j) =>
        c.live
          ? h("a.card", { href: `/learn?c=${c.id}` },
              h("span.t", c.title), h("span.b", c.blurb),
              h("span.meta", h("span", `${i}.${j + 1}`), done[c.id]?.visited ? h("span.done", done[c.id]?.predicted ? "✓ done" : "visited") : h("span", "start")))
          : h("div.card.soon", { "aria-disabled": "true" }, h("span.t", c.title), h("span.b", c.blurb), h("span.meta", h("span", `${i}.${j + 1}`), h("span", "coming"))),
      )),
    ),
  ),
);
document.getElementById("nproj").textContent = String(PROJECTS.length);
