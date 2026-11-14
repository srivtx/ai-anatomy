import { h, rng } from "../lib/core.js";
import { slider, stats, predict, code, callout, section, prose, table, button, liveChart } from "../lib/widgets.js";
import { xyChart } from "../lib/plot.js";

const N = 10;
const ART = {
  A: ["...####...", "..##..##..", ".##....##.", ".##....##.", ".########.", ".##....##.", ".##....##.", ".##....##.", ".##....##.", ".........."],
  T: ["##########", "##########", "....##....", "....##....", "....##....", "....##....", "....##....", "....##....", "....##....", ".........."],
  L: [".##.......", ".##.......", ".##.......", ".##.......", ".##.......", ".##.......", ".##.......", ".########.", ".########.", ".........."],
  O: ["..######..", ".##....##.", "##......##", "##......##", "##......##", "##......##", "##......##", ".##....##.", "..######..", ".........."],
  X: ["##......##", ".##....##.", "..##..##..", "...####...", "....##....", "...####...", "..##..##..", ".##....##.", "##......##", ".........."],
};
const toVec = (rows) => rows.join("").split("").map((c) => (c === "#" ? 1 : -1));

function hebb(patterns) {
  const n = patterns[0].length, W = Array.from({ length: n }, () => new Float64Array(n));
  for (const p of patterns) for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j) W[i][j] += (p[i] * p[j]) / n;
  return W;
}
const energy = (W, x) => { let e = 0; for (let i = 0; i < x.length; i++) for (let j = 0; j < x.length; j++) e -= 0.5 * W[i][j] * x[i] * x[j]; return e; };

export default function (root, { id }) {
  let stored = ["A", "T", "L"], W = hebb(stored.map((k) => toVec(ART[k]))), state = [], noise = 0.25, r = rng(1), running = null;
  const gridEl = h("div", { style: { display: "grid", gridTemplateColumns: `repeat(${N}, 1fr)`, gap: "2px", width: "min(100%, 300px)", touchAction: "none" }, role: "img", "aria-label": "The network's current state, 10 by 10 neurons" });
  const st = stats([{ label: "Energy" }, { label: "Closest stored pattern" }, { label: "Bits wrong vs that pattern" }]);
  const chart = liveChart({ series: [{ name: "Energy", color: "var(--c6)" }], height: 100, points: 80, fmt: (v) => v.toFixed(0) });
  const storedEl = h("div.row");
  function draw() {
    gridEl.replaceChildren(...state.map((v, i) => h("span", { style: { aspectRatio: "1", borderRadius: "3px", background: v > 0 ? "var(--ink)" : "var(--sunken)", cursor: "pointer" }, onpointerdown: () => { state[i] = -state[i]; draw(); } })));
    let best = null;
    for (const k of stored) { const p = toVec(ART[k]); const d = p.reduce((a, v, i) => a + (v !== state[i]), 0); const dInv = 100 - d; for (const [dd, lab] of [[d, k], [dInv, `${k} (inverted)`]]) if (!best || dd < best.d) best = { d: dd, k: lab }; }
    const e = energy(W, state);
    st.set(0, e.toFixed(1)); st.set(1, best.k); st.set(2, String(best.d), best.d === 0 ? "good" : best.d > 20 ? "bad" : "warn");
    chart.push([-e]);
    storedEl.replaceChildren(...Object.keys(ART).map((k) => button(`${stored.includes(k) ? "✓ " : ""}${k}`, () => { stored = stored.includes(k) ? stored.filter((x) => x !== k) : [...stored, k]; if (!stored.length) stored = [k]; W = hebb(stored.map((q) => toVec(ART[q]))); draw(); })));
  }
  function corrupt(k) {
    stop();
    state = toVec(ART[k]).map((v) => (r() < noise ? -v : v));
    chart.reset(); draw();
  }
  function step(n = 10) {
    for (let t = 0; t < n; t++) { // asynchronous updates: one random neuron at a time
      const i = r.int(100);
      let s2 = 0; for (let j = 0; j < 100; j++) s2 += W[i][j] * state[j];
      state[i] = s2 >= 0 ? 1 : -1;
    }
    draw();
  }
  const stop = () => { if (running) clearInterval(running); running = null; };

  // capacity experiment
  const cap = xyChart({ height: 230, xLabel: "patterns stored ÷ neurons", yLabel: "fraction recalled perfectly", xDomain: [0, 0.3], yDomain: [0, 1.02] });
  function capacity() {
    const rr = rng(5), n = 100, pts = [];
    for (const m of [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28]) {
      let ok = 0, tries = 0;
      for (let rep = 0; rep < 3; rep++) {
        const P = Array.from({ length: m }, () => Array.from({ length: n }, () => (rr() < 0.5 ? 1 : -1)));
        const Wc = hebb(P);
        for (const p of P) { // start at the pattern; is it a fixed point (after a few sweeps)?
          const x = [...p];
          for (let sweep = 0; sweep < 3; sweep++) for (let i = 0; i < n; i++) { let s2 = 0; for (let j = 0; j < n; j++) s2 += Wc[i][j] * x[j]; x[i] = s2 >= 0 ? 1 : -1; }
          ok += x.every((v, i) => v === p[i]); tries++;
        }
      }
      pts.push([m / n, ok / tries]);
    }
    cap.update([{ name: "Measured", color: "var(--c6)", points: pts, dots: true }], { marks: [{ x: 0.138, label: "theory: 0.138 N" }] });
  }

  // modern Hopfield = attention
  let beta = 1;
  const att = h("div");
  function drawAtt() {
    const keys = Object.keys(ART), query = toVec(ART.O).map((v, i) => (i % 7 === 0 ? -v : v));
    const sims = keys.map((k) => toVec(ART[k]).reduce((a, v, i) => a + v * query[i], 0) / 10);
    const m = Math.max(...sims), ex = sims.map((x) => Math.exp(beta * (x - m))), Z = ex.reduce((a, b) => a + b, 0);
    att.replaceChildren(table(["Stored pattern (key)", "Similarity to a noisy O", "softmax(β · similarity)"], keys.map((k, i) => [k, sims[i].toFixed(1), h("span", { style: { display: "inline-block", minWidth: "60px" } }, `${((ex[i] / Z) * 100).toFixed(1)}%`)])));
  }

  root.append(
    section(null, prose(`<p>In 1982 John Hopfield described a network that <b>remembers</b>: store some patterns, show it a corrupted version, and it settles into the closest stored one. Each pattern becomes a valley in an <b>energy</b> landscape, and the network rolls downhill. Geoffrey Hinton and Terry Sejnowski turned it into the stochastic <b>Boltzmann machine</b>, which learned hidden features. The two shared the 2024 Nobel Prize in Physics for this work.</p>`)),
    section("Store letters, then recall them from noise",
      prose("<p>100 neurons, each +1 (black) or −1. Storing uses the <b>Hebbian</b> rule, “neurons that fire together wire together”: w<sub>ij</sub> += x<sub>i</sub>·x<sub>j</sub> for each pattern. Recall: repeatedly pick a neuron and set it to the sign of its weighted input. Each update can only lower the energy. Click cells to flip them yourself.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Corruption", min: 0, max: 0.5, step: 0.05, value: noise, fmt: (v) => `${Math.round(v * 100)}% of pixels flipped`, onInput: (v) => (noise = v) }),
          h("div.seg-wrap", h("span.seg-label", "Corrupt a letter"), h("div.row", Object.keys(ART).map((k) => button(k, () => corrupt(k))))),
          button("Recall (run updates)", () => { stop(); let n = 0; running = setInterval(() => { step(8); if (++n > 60) stop(); }, 30); }, "primary"),
          h("div.seg-wrap", h("span.seg-label", "Patterns stored (click to toggle)"), storedEl)),
        h("div.stage-body", st.el, h("div.lab", gridEl, h("div", h("div.eyebrow", "Energy falling (plotted as −energy)"), chart.el)))),
      callout("tip", "Things to try", "<ul style='margin:0;padding-left:18px'><li>Corrupt A by 25% and recall: it snaps back. Try 45%: it may fall into the <i>inverted</i> pattern instead, since −x has exactly the same energy as x.</li><li>Store all five letters, then corrupt O. With too many similar patterns stored, recall lands in a blend: a <b>spurious</b> memory.</li><li>Corrupt X while X isn't stored. The network confidently recalls whatever stored letter is closest. Associative memory always returns <i>something</i>.</li></ul>"),
      predict({
        chapter: id,
        question: "A Hopfield network has 100 neurons. About how many random patterns can it store and still recall reliably?",
        options: ["100", "About 50", "About 14", "Unlimited, it just gets slower"],
        answer: 2,
        explain: "<p>Roughly 0.14 × N (Amit, Gutfreund and Sompolinsky, 1985). Beyond that, the crosstalk between patterns swamps the signal and memories corrupt each other. The experiment below measures it.</p>",
      }),
    ),
    section("Measure the capacity",
      cap.el,
      prose("<p>Classic Hopfield networks are tiny memories. <b>Modern Hopfield networks</b> (Krotov and Hopfield 2016; Ramsauer et al. 2020) replace the quadratic energy with an exponential one, which stores exponentially many patterns. Their update rule is: compare the query with every stored pattern, softmax the similarities, and return the weighted average. That's exactly <b>attention</b>, with stored patterns as keys and values.</p>"),
    ),
    section("Recall as attention",
      h("div.stage",
        h("div.stage-controls", slider({ label: "Inverse temperature β", min: 0, max: 5, step: 0.1, value: beta, fmt: (v) => v.toFixed(1), onInput: (v) => { beta = v; drawAtt(); }, hint: "In a transformer, β = 1/√d" })),
        h("div.stage-body", att)),
      table(["Model", "Year", "Idea", "Today"], [
        ["Hopfield network", "1982", "Memories as energy minima; recall by descent", "Teaching tool; theory of associative memory"],
        ["Boltzmann machine", "1985", "Stochastic units; learn a probability distribution with hidden units", "Ancestor of energy-based and generative models"],
        ["Restricted Boltzmann machine", "2006", "Layer-wise pretraining of deep nets (Hinton)", "Kick-started the deep learning revival"],
        ["Energy-based models", "2000s–now", "Score configurations; low energy = plausible", "LeCun's JEPA, score-based diffusion"],
        ["Modern Hopfield / attention", "2016–20", "Exponential capacity, one-step retrieval", "Every transformer layer"],
      ]),
    ),
    section("In code",
      code({
        NumPy: `import numpy as np

def store(patterns):                        # patterns: (m, n) of +-1
    n = patterns.shape[1]
    W = patterns.T @ patterns / n           # Hebbian outer products
    np.fill_diagonal(W, 0)
    return W

def recall(W, x, sweeps=5):
    x = x.copy()
    for _ in range(sweeps):
        for i in np.random.permutation(len(x)):
            x[i] = 1 if W[i] @ x >= 0 else -1
    return x

def modern_recall(X, query, beta=8.0):     # X: stored patterns as rows
    p = np.exp(beta * X @ query); p /= p.sum()
    return p @ X                           # softmax-weighted average = attention`,
      }, "Project 29 measures the capacity and shows modern Hopfield retrieval equals attention."),
    ),
  );
  corrupt("A"); capacity(); drawAtt();
}
