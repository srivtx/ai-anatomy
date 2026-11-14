import { h, rng } from "../lib/core.js";
import { slider, segmented, stats, predict, code, callout, section, prose, dilemma, table } from "../lib/widgets.js";
import { plane } from "../lib/plot.js";
import { dataset, sigmoid } from "../lib/nn.js";

// ---- k nearest neighbours
const knn = (train, k) => (x, y) => {
  const d = train.map((p) => [(p.x[0] - x) ** 2 + (p.x[1] - y) ** 2, p.y]).sort((a, b) => a[0] - b[0]);
  let s = 0; for (let i = 0; i < k; i++) s += d[i][1];
  return s / k;
};

// ---- CART decision tree. target: numbers (class 0/1 for classification, residuals for boosting)
function buildTree(pts, targets, depth, { minLeaf = 2, r = null, leafValue } = {}) {
  const idx = pts.map((_, i) => i);
  let nodes = 0;
  const value = leafValue || ((ii) => ii.reduce((a, i) => a + targets[i], 0) / ii.length);
  function grow(ii, d) {
    nodes++;
    if (d === 0 || ii.length < 2 * minLeaf || ii.every((i) => targets[i] === targets[ii[0]])) return { leaf: value(ii) };
    let best = null;
    const feats = r ? [r() < 0.5 ? 0 : 1] : [0, 1]; // a random forest only looks at a random feature
    for (const f of feats) {
      const sorted = [...ii].sort((a, b) => pts[a].x[f] - pts[b].x[f]);
      let sumL = 0, sqL = 0;
      const tot = sorted.reduce((a, i) => a + targets[i], 0), totSq = sorted.reduce((a, i) => a + targets[i] ** 2, 0);
      for (let k = 0; k < sorted.length - 1; k++) {
        const t = targets[sorted[k]];
        sumL += t; sqL += t * t;
        const nl = k + 1, nr = sorted.length - nl;
        if (nl < minLeaf || nr < minLeaf || pts[sorted[k]].x[f] === pts[sorted[k + 1]].x[f]) continue;
        // variance reduction (equals Gini impurity reduction for 0/1 targets, up to a factor)
        const sse = sqL - (sumL * sumL) / nl + (totSq - sqL) - ((tot - sumL) ** 2) / nr;
        if (!best || sse < best.sse) best = { sse, f, th: (pts[sorted[k]].x[f] + pts[sorted[k + 1]].x[f]) / 2 };
      }
    }
    if (!best) return { leaf: value(ii) };
    const L = ii.filter((i) => pts[i].x[best.f] < best.th), R = ii.filter((i) => pts[i].x[best.f] >= best.th);
    return { f: best.f, th: best.th, l: grow(L, d - 1), r: grow(R, d - 1) };
  }
  const root = grow(idx, depth);
  const predict = (x) => { let n = root; while (!("leaf" in n)) n = x[n.f] < n.th ? n.l : n.r; return n.leaf; };
  return { predict, nodes, root };
}

function forest(train, nTrees, depth, seed) {
  const r = rng(seed);
  const trees = Array.from({ length: nTrees }, () => {
    const boot = Array.from({ length: train.length }, () => train[r.int(train.length)]);
    return buildTree(boot, boot.map((p) => p.y), depth, { r: nTrees > 1 ? r : null, minLeaf: 1 });
  });
  return { predict: (x) => trees.reduce((a, t) => a + t.predict(x), 0) / trees.length, nodes: trees.reduce((a, t) => a + t.nodes, 0) };
}

function boosting(train, rounds, depth, lr) {
  const F = train.map(() => 0), trees = [];
  for (let m = 0; m < rounds; m++) {
    const p = F.map(sigmoid), res = train.map((pt, i) => pt.y - p[i]); // negative gradient of log loss
    // Newton step for the leaf value: sum(residual) / sum(p(1-p))
    const t = buildTree(train, res, depth, { minLeaf: 3, leafValue: (ii) => ii.reduce((a, i) => a + res[i], 0) / Math.max(1e-6, ii.reduce((a, i) => a + p[i] * (1 - p[i]), 0)) });
    trees.push(t);
    train.forEach((pt, i) => (F[i] += lr * t.predict(pt.x)));
  }
  return { predict: (x) => sigmoid(trees.reduce((a, t) => a + lr * t.predict(x), 0)), nodes: trees.reduce((a, t) => a + t.nodes, 0) };
}

export default function (root, { id }) {
  const o = { kind: "moons", algo: "tree", k: 5, depth: 4, trees: 20, rounds: 40, lr: 0.3, noise: 0.08 };
  let train, test;
  const pl = plane({ width: 480, height: 420, label: "Decision regions of the chosen model" });
  const st = stats([{ label: "Training accuracy" }, { label: "Test accuracy" }, { label: "Model size" }]);
  const ctl = h("div.row", { style: { gap: "14px 22px", alignItems: "flex-end" } });
  function data() { train = dataset(o.kind, 200, { noise: o.noise, seed: 11 }); test = dataset(o.kind, 400, { noise: o.noise, seed: 99 }); }
  function controls() {
    const S = (label, key, min, max, step = 1, fmt) => slider({ label, min, max, step, value: o[key], fmt, onInput: (v) => { o[key] = v; draw(); } });
    ctl.replaceChildren(...({
      knn: [S("k (neighbours)", "k", 1, 51, 2)],
      tree: [S("Max depth", "depth", 1, 12)],
      forest: [S("Trees", "trees", 1, 60), S("Max depth", "depth", 1, 12)],
      boost: [S("Boosting rounds", "rounds", 1, 150), S("Tree depth", "depth", 1, 6), S("Learning rate", "lr", 0.05, 1, 0.05, (v) => v.toFixed(2))],
    })[o.algo]);
  }
  function model() {
    if (o.algo === "knn") { const f = knn(train, o.k); return { predict: (x) => f(x[0], x[1]), nodes: `${train.length} stored points` }; }
    if (o.algo === "tree") { const t = buildTree(train, train.map((p) => p.y), o.depth); return { predict: t.predict, nodes: `${t.nodes} nodes` }; }
    if (o.algo === "forest") { const f = forest(train, o.trees, o.depth, 5); return { predict: f.predict, nodes: `${f.nodes} nodes in ${o.trees} trees` }; }
    const b = boosting(train, o.rounds, Math.min(o.depth, 6), o.lr);
    return { predict: b.predict, nodes: `${b.nodes} nodes in ${o.rounds} trees` };
  }
  function draw() {
    const m = model();
    const acc = (set) => set.filter((p) => (m.predict(p.x) > 0.5) === !!p.y).length / set.length;
    pl.clear();
    pl.field((x, y) => m.predict([x, y]), { res: o.algo === "knn" ? 46 : 70, alpha: 0.55 });
    pl.scatter(train, { r: 3.5 });
    const a1 = acc(train), a2 = acc(test);
    st.set(0, `${(a1 * 100).toFixed(1)}%`); st.set(1, `${(a2 * 100).toFixed(1)}%`, a1 - a2 > 0.08 ? "warn" : a2 > 0.9 ? "good" : "");
    st.set(2, String(m.nodes));
  }

  root.append(
    section(null, prose(`<p>Before deep learning took over images and text, machine learning meant methods like these, and on <b>tabular data</b> (spreadsheets, transactions, sensor logs) they still usually win. They're also the best way to build intuition: every one of them draws its decision regions in a way you can see.</p>`)),
    section("Four models, one dataset",
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Model", options: [{ value: "knn", label: "k-NN" }, { value: "tree", label: "Decision tree" }, { value: "forest", label: "Random forest" }, { value: "boost", label: "Gradient boosting" }], value: o.algo, onChange: (v) => { o.algo = v; controls(); draw(); } }),
          segmented({ label: "Data", options: [{ value: "moons", label: "Moons" }, { value: "circle", label: "Circle" }, { value: "xor", label: "XOR" }, { value: "spiral", label: "Spiral" }], value: o.kind, onChange: (v) => { o.kind = v; data(); draw(); } }),
          slider({ label: "Noise", min: 0, max: 0.4, step: 0.05, value: o.noise, fmt: (v) => v.toFixed(2), onInput: (v) => { o.noise = v; data(); draw(); } }),
          ctl),
        h("div.stage-body", st.el, h("div.lab", h("div.canvas-box", pl.el),
          h("div.prose", { html: `<p><b>k-NN</b> stores the data and votes among the k closest points. No training at all; all the work happens at prediction time.</p><p><b>A decision tree</b> asks yes/no questions like “is x &lt; 0.3?”, choosing each split to make the two sides as pure as possible. Its regions are always axis-aligned boxes.</p><p><b>A random forest</b> trains many deep trees on bootstrap resamples, each seeing random features, and averages them. Averaging cancels their individual overfitting.</p><p><b>Gradient boosting</b> adds small trees one at a time, each fixing the errors (the gradient of the loss) left by the ones before. This is XGBoost, LightGBM and CatBoost.</p>` })))),
      callout("tip", "Things to try", "<ul style='margin:0;padding-left:18px'><li>k-NN with k = 1: training accuracy is 100%, every noisy point gets its own island. Raise k and the boundary smooths.</li><li>A single tree at depth 12 overfits with blocky islands. A forest of 60 deep trees, each just as overfit, averages into a smooth boundary.</li><li>Boosting with learning rate 1 and many rounds overfits; lower the rate and it needs more rounds but generalises better. That's <b>shrinkage</b>.</li></ul>"),
      predict({
        chapter: id,
        question: "Why does averaging 50 deep, overfit trees beat one of them?",
        options: ["The average tree is deeper", "Their errors are partly independent, so they cancel while the real signal adds up", "Averaging removes the noisy points from the data", "It doesn't; it just runs slower"],
        answer: 1,
        explain: "<p>Each tree has low bias and high variance. Averaging N models with independent errors divides the variance by N while keeping the bias. Bootstrap samples and random features make the trees' errors less correlated. The same idea, <b>ensembling</b>, wins Kaggle competitions and shows up in LLMs as self-consistency voting.</p>",
      }),
    ),
    section("Bias, variance and which to reach for",
      table(["Method", "Training", "Prediction", "Strengths", "Weak spots"], [
        ["k-NN", "None (store data)", "Slow: compare to everything", "Simple, no assumptions", "Curse of dimensionality; needs scaled features"],
        ["Decision tree", "Fast", "Very fast", "Readable rules, mixed feature types", "Overfits, unstable"],
        ["Random forest", "Parallel", "Fast", "Robust default, little tuning", "Big models, can't extrapolate"],
        ["Gradient boosting", "Sequential", "Fast", "Best accuracy on tables", "More tuning; can overfit"],
        ["Neural network", "Slow, GPU", "Fast on GPU", "Images, text, audio, huge data", "Usually loses on small tables"],
      ]),
      dilemma("A churn model on 50,000 customer rows with 40 columns. Neural net or gradient boosting?", [
        { name: "Gradient boosting", pros: ["Usually more accurate on tabular data", "Trains in seconds on a CPU", "Feature importance for free"], cons: ["Can't use raw text or images directly"] },
        { name: "Neural network", pros: ["Can mix in text and image inputs", "Learns embeddings for categories"], cons: ["Needs careful tuning", "Often loses on this kind of data"] },
      ], "Start with gradient boosting. Benchmarks (Grinsztajn et al., 2022) still find tree ensembles beat deep nets on medium-sized tables, though tabular foundation models such as TabPFN are closing the gap."),
    ),
    section("In code",
      code({
        "scikit-learn": `from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

models = {
    "knn": KNeighborsClassifier(n_neighbors=15),
    "tree": DecisionTreeClassifier(max_depth=4),
    "forest": RandomForestClassifier(n_estimators=300),
    "boost": HistGradientBoostingClassifier(learning_rate=0.1, max_iter=300),
}
for name, m in models.items():
    m.fit(X_train, y_train)
    print(name, m.score(X_test, y_test))`,
      }, "Project 23 builds the tree and the booster from scratch in NumPy."),
    ),
  );
  data(); controls();
  pl.paint(draw);
}
