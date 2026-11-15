import { h, rng } from "../lib/core.js";
import { walk } from "../lib/walks.js";
import { slider, stats, predict, code, callout, section, prose, button, table, toggle } from "../lib/widgets.js";
import { plane, xyChart } from "../lib/plot.js";
import { gauss } from "../lib/nn.js";

const COLS = ["c1", "c2", "c3", "c4", "c5", "c6", "accent", "warm"];
function blobs(seed = 3) {
  const r = rng(seed), centres = [[-0.6, 0.55], [0.55, 0.6], [0.1, -0.55], [-0.75, -0.45]];
  return Array.from({ length: 160 }, (_, i) => { const c = centres[i % 4]; return [c[0] + gauss(r) * 0.17, c[1] + gauss(r) * 0.17]; });
}
const d2 = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;
function kmeansRun(pts, k, seed, iters = 30) {
  const r = rng(seed);
  let C = Array.from({ length: k }, () => [...pts[r.int(pts.length)]]), lab = [];
  for (let it = 0; it < iters; it++) {
    lab = pts.map((p) => C.reduce((b, c, j) => (d2(p, c) < d2(p, C[b]) ? j : b), 0));
    C = C.map((c, j) => { const m = pts.filter((_, i) => lab[i] === j); return m.length ? [m.reduce((a, p) => a + p[0], 0) / m.length, m.reduce((a, p) => a + p[1], 0) / m.length] : c; });
  }
  return pts.reduce((a, p, i) => a + d2(p, C[lab[i]]), 0);
}

export default function (root, { id }) {
  let pts = blobs(), k = 4, C = [], lab = [], phase = "init", iter = 0, seed = 1;
  const pl = plane({ width: 460, height: 420, label: "Points coloured by cluster, with centroids", onPoint: (x, y) => { pts.push([x, y]); assign(); draw(); } });
  const st = stats([{ label: "Iteration" }, { label: "Next step" }, { label: "Inertia (sum of squared distances)" }]);
  const elbow = xyChart({ width: 460, height: 300, xLabel: "k (number of clusters)", yLabel: "inertia" });
  function init() {
    const r = rng(seed++);
    C = Array.from({ length: k }, () => [...pts[r.int(pts.length)]]);
    lab = pts.map(() => -1); iter = 0; phase = "assign"; draw();
  }
  function assign() { if (C.length) lab = pts.map((p) => C.reduce((b, c, j) => (d2(p, c) < d2(p, C[b]) ? j : b), 0)); }
  function update() { C = C.map((c, j) => { const m = pts.filter((_, i) => lab[i] === j); return m.length ? [m.reduce((a, p) => a + p[0], 0) / m.length, m.reduce((a, p) => a + p[1], 0) / m.length] : c; }); }
  function step() {
    if (phase === "assign") { assign(); phase = "update"; }
    else { update(); iter++; phase = "assign"; }
    draw();
  }
  function draw() {
    pl.clear(); pl.axes();
    pts.forEach((p, i) => { pl.dot(pl.X(p[0]), pl.Y(p[1]), 5, pl.colors.surface); pl.dot(pl.X(p[0]), pl.Y(p[1]), 3.8, lab[i] >= 0 ? pl.colors[COLS[lab[i] % 8]] : pl.colors["ink-3"]); });
    C.forEach((c, j) => { pl.dot(pl.X(c[0]), pl.Y(c[1]), 11, pl.colors.ink); pl.dot(pl.X(c[0]), pl.Y(c[1]), 8, pl.colors[COLS[j % 8]]); });
    const inertia = lab.every((l) => l >= 0) ? pts.reduce((a, p, i) => a + d2(p, C[lab[i]]), 0) : null;
    st.set(0, String(iter)); st.set(1, phase === "assign" ? "Assign points to nearest centroid" : "Move centroids to their points' mean");
    st.set(2, inertia == null ? "–" : inertia.toFixed(2));
  }
  function drawElbow() {
    const pts2 = [];
    for (let kk = 1; kk <= 8; kk++) pts2.push([kk, Math.min(...[1, 2, 3, 4, 5].map((s) => kmeansRun(pts, kk, s)))]);
    elbow.update([{ name: "Best of 5 restarts", color: "var(--c1)", points: pts2, dots: true }], { marks: [{ x: 4, label: "the elbow" }] });
  }

  // PCA
  let angle = 30, spread = 0.25, project = false;
  const pp = plane({ width: 420, height: 360, xr: [-1.6, 1.6], yr: [-1.3, 1.3], label: "Correlated data with its principal components" });
  const pst = stats([{ label: "Variance on PC1" }, { label: "Variance on PC2" }]);
  function drawPCA() {
    const r = rng(8), t = (angle * Math.PI) / 180;
    const data = Array.from({ length: 150 }, () => { const a = gauss(r) * 0.55, b = gauss(r) * spread * 0.55; return [a * Math.cos(t) - b * Math.sin(t), a * Math.sin(t) + b * Math.cos(t)]; });
    const mx = data.reduce((s2, p) => s2 + p[0], 0) / data.length, my = data.reduce((s2, p) => s2 + p[1], 0) / data.length;
    let sxx = 0, syy = 0, sxy = 0;
    for (const [x, y] of data) { sxx += (x - mx) ** 2; syy += (y - my) ** 2; sxy += (x - mx) * (y - my); }
    sxx /= data.length; syy /= data.length; sxy /= data.length;
    const tr = sxx + syy, det = sxx * syy - sxy * sxy, l1 = tr / 2 + Math.sqrt(tr * tr / 4 - det), l2 = tr - l1;
    const v1 = Math.abs(sxy) > 1e-9 ? [l1 - syy, sxy] : sxx >= syy ? [1, 0] : [0, 1], n1 = Math.hypot(...v1), u = [v1[0] / n1, v1[1] / n1], w = [-u[1], u[0]];
    pp.clear(); pp.axes();
    for (const p of data) {
      const q = project ? (() => { const s2 = (p[0] - mx) * u[0] + (p[1] - my) * u[1]; return [mx + s2 * u[0], my + s2 * u[1]]; })() : p;
      if (project) pp.line(pp.X(p[0]), pp.Y(p[1]), pp.X(q[0]), pp.Y(q[1]), { color: pp.colors.line, lw: 1 });
      pp.dot(pp.X(q[0]), pp.Y(q[1]), 3.2, pp.colors.c1);
    }
    const ar = (v, l, col, name) => { const L = 2 * Math.sqrt(l); pp.line(pp.X(mx), pp.Y(my), pp.X(mx + v[0] * L), pp.Y(my + v[1] * L), { color: col, lw: 3.5 }); pp.text(name, pp.X(mx + v[0] * L) + 6, pp.Y(my + v[1] * L) - 6, { color: col, size: 13, weight: 700 }); };
    ar(u, l1, pp.colors.c2, "PC1"); ar(w, l2, pp.colors.c3, "PC2");
    pst.set(0, `${((l1 / tr) * 100).toFixed(1)}%`, "good"); pst.set(1, `${((l2 / tr) * 100).toFixed(1)}%`);
  }

  root.append(
    section(null, prose(`<p>Most data has no labels. Nobody tagged your users into segments or told you which features of a sensor log matter. <b>Unsupervised learning</b> looks for structure anyway: groups (clustering) and the few directions that explain most of the variation (dimensionality reduction). Modern <b>self-supervised</b> learning, including LLM pretraining, is a descendant of this idea: invent a task from the data itself.</p>`)),
    section("k-means, one step at a time",
      prose("<p>k-means alternates two steps until nothing changes: <b>assign</b> every point to its nearest centroid, then <b>move</b> each centroid to the mean of its points. Each step can only lower the total squared distance, so it always converges, though not always to the best answer. Click to add points.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "k", min: 1, max: 8, value: k, onInput: (v) => { k = v; init(); } }),
          button("Random start", init, "primary"), button("Step", step),
          button("Run to convergence", () => { if (phase === "init") init(); for (let i = 0; i < 40; i++) step(); }),
          button("New data", () => { pts = blobs(Math.floor(Math.random() * 1000)); init(); drawElbow(); })),
        h("div.stage-body", st.el, h("div.lab", h("div.canvas-box", pl.el), h("div", h("div.eyebrow", "Choosing k: the elbow"), elbow.el,
          h("p.canvas-cap", "Inertia always falls as k grows; you look for where the gains flatten."))))),
      callout("tip", "Things to try", "<ul style='margin:0;padding-left:18px'><li>Press Random start a few times with k = 4. Sometimes two centroids share one blob and another blob is split: a <b>local minimum</b>. k-means++ (spread-out starts) and restarts fix this.</li><li>Set k = 8: k-means happily invents clusters that aren't there. It always returns k groups.</li></ul>"),
      predict({
        chapter: id,
        question: "You run k-means on two long, thin, parallel clusters. What happens?",
        options: ["It finds them perfectly", "It tends to cut across them, because it prefers round clusters of similar size", "It refuses to converge", "It merges them into one"],
        answer: 1,
        explain: "<p>k-means minimises squared distance to a centre, so it implicitly assumes round, similar-sized blobs. For other shapes use DBSCAN (density), Gaussian mixtures (elliptical clusters) or spectral clustering. Or learn an embedding first and cluster in that space, which is what most modern pipelines do.</p>",
      }),
    ),
    section("PCA: the directions that matter",
      prose("<p><b>Principal component analysis</b> finds the axis along which the data varies most (PC1), then the most-varying axis at right angles to it, and so on. They're the eigenvectors of the covariance matrix. Keep the top few and you compress the data with the least squared error.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Rotation of the cloud", min: 0, max: 180, value: angle, fmt: (v) => `${v}°`, onInput: (v) => { angle = v; drawPCA(); } }),
          slider({ label: "Thickness", min: 0.02, max: 1, step: 0.02, value: spread, fmt: (v) => v.toFixed(2), onInput: (v) => { spread = v; drawPCA(); } }),
          toggle({ label: "Project onto PC1 (2D → 1D)", onChange: (v) => { project = v; drawPCA(); } })),
        h("div.stage-body", pst.el, h("div.canvas-box", { style: { maxWidth: "560px" } }, pp.el))),
      table(["Method", "Keeps", "Use it for"], [
        ["PCA", "Global variance (linear)", "Compression, denoising, a first look at any dataset"],
        ["t-SNE", "Who is near whom (non-linear)", "2D pictures of embeddings; distances between clusters mean little"],
        ["UMAP", "Local and some global structure", "Faster t-SNE alternative for big embedding sets"],
        ["Autoencoder", "Whatever reconstructs best (learned)", "Non-linear compression; see the autoencoder chapter"],
      ]),
    ),
    walk("unsupervised"),
    section("In code",
      code({
        "scikit-learn": `from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

km = KMeans(n_clusters=4, n_init=10).fit(X)       # 10 restarts, k-means++ init
labels, inertia = km.labels_, km.inertia_

pca = PCA(n_components=2).fit(X)
print(pca.explained_variance_ratio_)            # how much each component keeps
X2 = pca.transform(X)                           # project to 2D`,
      }),
    ),
  );
  init();
  pl.paint(draw);
  drawElbow();
  pp.paint(drawPCA);
}
