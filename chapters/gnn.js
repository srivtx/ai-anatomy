import { h, rng, s } from "../lib/core.js";
import { walk } from "../lib/walks.js";
import { slider, stats, predict, code, callout, section, prose, table, button, toggle } from "../lib/widgets.js";

// Two friend groups with a few cross links (a stochastic block model), laid out by a spring simulation.
function makeGraph(seed = 4) {
  const r = rng(seed), n = 26, truth = Array.from({ length: n }, (_, i) => (i < 13 ? 0 : 1)), E = new Set();
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (r() < (truth[i] === truth[j] ? 0.33 : 0.02)) E.add(`${i}-${j}`);
  for (let i = 1; i < n; i++) if (![...E].some((e) => e.split("-").map(Number).includes(i))) E.add(`${i - 1}-${i}`);
  const edges = [...E].map((e) => e.split("-").map(Number));
  // each group on its own ring, with a little jitter
  const pos = Array.from({ length: n }, (_, i) => {
    const k = i % 13, th = (k / 13) * Math.PI * 2 + (truth[i] ? 0.3 : 0), rad = 0.17 + r() * 0.12;
    return [(truth[i] ? 0.72 : 0.28) + Math.cos(th) * rad * 0.62, 0.5 + Math.sin(th) * rad * 1.05];
  });
  return { n, edges, truth, pos, nbrs: Array.from({ length: n }, (_, i) => edges.filter((e) => e.includes(i)).map((e) => (e[0] === i ? e[1] : e[0]))) };
}

export default function (root, { id }) {
  const G = makeGraph();
  // label the best-connected member of each group
  const hub = (g) => [...Array(G.n).keys()].filter((i) => G.truth[i] === g).sort((a, b) => G.nbrs[b].filter((j) => G.truth[j] === g).length - G.nbrs[b].filter((j) => G.truth[j] !== g).length * 3 - (G.nbrs[a].filter((j) => G.truth[j] === g).length - G.nbrs[a].filter((j) => G.truth[j] !== g).length * 3))[0];
  let known = new Map([[hub(0), 0], [hub(1), 1]]), K = 0, selfLoop = true, H = [];
  const svg = s("svg", { viewBox: "0 0 760 420", class: "graph-svg", role: "img", "aria-label": "A graph of two friend groups; colour shows each node's current belief" });
  const st = stats([{ label: "Message-passing layers" }, { label: "Accuracy on unlabelled nodes" }, { label: "Spread of node states" }]);
  function propagate() {
    // h0: one-hot for labelled nodes, zeros for the rest. Each layer: average over neighbours (+ self).
    H = [Array.from({ length: G.n }, (_, i) => (known.has(i) ? (known.get(i) ? [0, 1] : [1, 0]) : [0, 0]))];
    for (let k = 0; k < K; k++) {
      const prev = H[H.length - 1];
      H.push(prev.map((_, i) => {
        const srcs = [...G.nbrs[i], ...(selfLoop ? [i] : [])];
        const m = srcs.reduce((a, j) => [a[0] + prev[j][0], a[1] + prev[j][1]], [0, 0]);
        return [m[0] / srcs.length, m[1] / srcs.length];
      }));
    }
  }
  function draw() {
    propagate();
    const cur = H[H.length - 1], cs = getComputedStyle(document.documentElement), C0 = cs.getPropertyValue("--c2").trim(), C1 = cs.getPropertyValue("--c1").trim();
    svg.replaceChildren();
    const P = (i) => [G.pos[i][0] * 760, G.pos[i][1] * 420];
    for (const [a, b] of G.edges) { const [x1, y1] = P(a), [x2, y2] = P(b); svg.append(s("line", { x1, y1, x2, y2, class: "wire g-edge" })); }
    let ok = 0, tot = 0;
    for (let i = 0; i < G.n; i++) {
      const [x, y] = P(i), [a, b] = cur[i], m = a + b, t = m > 1e-9 ? b / m : 0.5, conf = Math.min(1, m * 3);
      const fill = m < 1e-9 ? "var(--sunken)" : `color-mix(in srgb, color-mix(in srgb, ${C1} ${Math.round(t * 100)}%, ${C0}) ${Math.round(25 + conf * 75)}%, var(--surface))`;
      const g = s("g", { style: "cursor:pointer", onclick: () => { const v = known.get(i); if (v === undefined) known.set(i, 0); else if (v === 0) known.set(i, 1); else known.delete(i); draw(); } });
      g.append(s("circle", { cx: x, cy: y, r: known.has(i) ? 17 : 14, style: `fill:${fill};stroke:${known.has(i) ? "var(--ink)" : "color-mix(in srgb, var(--ink-3) 70%, transparent)"};stroke-width:${known.has(i) ? 3 : 2}` }));
      g.append(s("text", { x, y: y + 4, "text-anchor": "middle", class: "node-sub", style: "fill:var(--ink);font-size:11px;font-weight:700", text: known.has(i) ? (known.get(i) ? "B" : "A") : String(i) }));
      svg.append(g);
      if (!known.has(i)) { tot++; if (m > 1e-9 && (b > a ? 1 : 0) === G.truth[i]) ok++; }
    }
    const spread = Math.sqrt(cur.reduce((acc, v) => { const m = v[0] + v[1]; return acc + (m > 1e-9 ? (v[1] / m - 0.5) ** 2 : 0); }, 0) / G.n) * 2;
    st.set(0, String(K)); st.set(1, `${Math.round((ok / tot) * 100)}%`, ok / tot > 0.9 ? "good" : ok / tot < 0.6 ? "bad" : "warn");
    st.set(2, spread.toFixed(2), spread < 0.15 && K > 5 ? "bad" : "");
  }
  const kSl = slider({ label: "Layers of message passing", min: 0, max: 40, value: K, onInput: (v) => { K = v; draw(); } });

  root.append(
    section(null, prose(`<p>Molecules, social networks, road maps, citation networks, chip netlists: lots of data is a <b>graph</b>, with no natural order and a different number of neighbours per node. A CNN needs a grid, an RNN needs a sequence. A <b>graph neural network</b> needs only one rule, applied everywhere: <b>each node updates its state from its neighbours' states</b>. Do it k times and every node has heard from everything up to k hops away.</p>`)),
    section("Spread two labels through a network",
      prose("<p>Two friend groups, mostly connected internally. Only two people are labelled (A and B, with thick rings). Each layer of message passing replaces every node's state with the average of its neighbours' (and its own). Colour shows the belief; paler means less evidence. Click any node to label it A, then B, then unlabel it.</p>"),
      h("div.stage",
        h("div.stage-controls", kSl,
          button("One more layer", () => { K = Math.min(40, K + 1); kSl.value = K; draw(); }, "primary"),
          toggle({ label: "Include each node's own state", checked: true, onChange: (v) => { selfLoop = v; draw(); } })),
        h("div.stage-body", st.el, h("div", { style: { overflowX: "auto" } }, h("div", { style: { minWidth: "520px" } }, svg)))),
      callout("tip", "Things to try", "<ul style='margin:0;padding-left:18px'><li>At 0 layers only the two labelled nodes know anything. Each layer pushes information one hop further.</li><li>Around 3 to 6 layers most nodes are classified. Keep going to 40: everything blends towards the same colour. That's <b>over-smoothing</b>, the reason most GNNs are only 2 to 4 layers deep.</li><li>Label a node on the wrong side and watch how far its wrong belief leaks across the bridge edges.</li></ul>"),
      predict({
        chapter: id,
        question: "What does a real GNN layer add on top of this averaging?",
        options: ["Nothing: GNNs are just averaging", "A learned weight matrix and non-linearity on the messages, so it learns what to pass and how to combine it", "A convolution over pixels", "Attention over the whole dataset"],
        answer: 1,
        explain: "<p>A graph convolutional network (Kipf and Welling, 2017) computes h′ = σ(Â · H · W): average the neighbours (Â), transform (W), squash (σ). Graph attention networks learn how much to weigh each neighbour. A transformer is a GNN on a fully connected graph where attention decides the edge weights.</p>",
      }),
    ),
    section("Where graphs show up",
      table(["Domain", "Nodes / edges", "Task"], [
        ["Chemistry and drugs", "Atoms / bonds", "Predict toxicity, binding, properties"],
        ["Proteins", "Residues / spatial contacts", "Structure (AlphaFold's pair representation is graph-like)"],
        ["Weather", "Points on the globe / neighbours", "GraphCast forecasts 10 days in under a minute"],
        ["Recommendations", "Users and items / interactions", "Pinterest's PinSage: who will like what"],
        ["Fraud", "Accounts / transfers", "Spot rings of colluding accounts"],
        ["Maps", "Road segments / connections", "Google Maps travel-time prediction"],
      ]),
    ),
    walk("gnn"),
    section("In code",
      code({
        "PyTorch (no libraries)": `import torch, torch.nn as nn

class GCNLayer(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)
    def forward(self, H, A):                       # A: (n, n) adjacency with self-loops
        deg = A.sum(1)
        norm = deg.pow(-0.5)
        A_hat = norm[:, None] * A * norm[None, :]  # symmetric normalisation
        return torch.relu(A_hat @ self.lin(H))     # transform, then aggregate neighbours

class GCN(nn.Module):
    def __init__(self, d_in, d_hidden, n_classes):
        super().__init__()
        self.l1, self.l2 = GCNLayer(d_in, d_hidden), nn.Linear(d_hidden, n_classes)
    def forward(self, X, A):
        return self.l2(self.l1(X, A))`,
      }, "Project 28 trains this from two labelled nodes on the karate-club graph."),
    ),
  );
  draw();
}
