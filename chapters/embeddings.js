import { h, rng, loop } from "../lib/core.js";
import { walk } from "../lib/walks.js";
import { stats, predict, code, callout, section, prose, table, button, segmented, slider } from "../lib/widgets.js";
import { plane } from "../lib/plot.js";
import { gauss, sigmoid } from "../lib/nn.js";

// A tiny corpus generated from templates, so words that appear in similar
// contexts really do share contexts.
const G = {
  male: ["king", "prince", "man", "boy"], female: ["queen", "princess", "woman", "girl"],
  royal: ["king", "queen", "prince", "princess"], common: ["man", "woman", "boy", "girl"],
  animal: ["cat", "dog", "mouse", "horse"], fruit: ["apple", "banana", "mango", "orange"],
  place: ["castle", "farm", "house", "kingdom"],
};
const GROUP = { king: "royal", queen: "royal", prince: "royal", princess: "royal", man: "people", woman: "people", boy: "people", girl: "people", cat: "animal", dog: "animal", mouse: "animal", horse: "animal", apple: "fruit", banana: "fruit", mango: "fruit", orange: "fruit", castle: "place", farm: "place", house: "place", kingdom: "place" };
const GCOL = { royal: "c6", people: "c1", animal: "c2", fruit: "c3", place: "c4" };
function corpus(seed = 1) {
  const r = rng(seed), P = (k) => r.pick(G[k]), out = [];
  const T = [
    () => `the ${P("royal")} rules the ${r.pick(["kingdom", "castle"])}`,
    () => `the ${P("royal")} lives in the castle`,
    () => `the ${P("common")} lives in the ${r.pick(["house", "farm"])}`,
    () => `the ${r.pick([...G.common, ...G.royal])} eats a ${P("fruit")}`,
    () => `a ripe ${P("fruit")} is sweet`,
    () => `the ${P("animal")} chases the ${P("animal")}`,
    () => `the ${P("animal")} sleeps on the farm`,
    () => `the ${P("male")} said he was tired`,
    () => `the ${P("female")} said she was tired`,
    () => `the ${r.pick([...G.common, ...G.royal])} rides the horse`,
  ];
  for (let i = 0; i < 1400; i++) out.push(r.pick(T)().split(" "));
  return out;
}

export default function (root, { id }) {
  const sents = corpus();
  const vocab = [...new Set(sents.flat())], idx = new Map(vocab.map((w, i) => [w, i])), V = vocab.length, D = 12;
  const freq = new Float64Array(V); for (const s of sents) for (const w of s) freq[idx.get(w)]++;
  const negTable = []; vocab.forEach((w, i) => { for (let k = 0; k < Math.ceil(Math.pow(freq[i], 0.75) / 4); k++) negTable.push(i); });
  let r, Win, Wout, epoch, pairsSeen, focus = "king";
  const window_ = 2, neg = 4;
  function init() {
    r = rng(3);
    Win = Array.from({ length: V }, () => Float64Array.from({ length: D }, () => gauss(r) * 0.1));
    Wout = Array.from({ length: V }, () => new Float64Array(D));
    epoch = 0; pairsSeen = 0;
  }
  let lr = 0.05;
  function trainChunk(nSent) {
    for (let s = 0; s < nSent; s++) {
      const sent = sents[r.int(sents.length)].map((w) => idx.get(w));
      sent.forEach((c, i) => {
        for (let j = Math.max(0, i - window_); j <= Math.min(sent.length - 1, i + window_); j++) {
          if (j === i) continue;
          const targets = [[sent[j], 1], ...Array.from({ length: neg }, () => [negTable[r.int(negTable.length)], 0])];
          const grad = new Float64Array(D);
          for (const [o, label] of targets) {
            let sc = 0; for (let d = 0; d < D; d++) sc += Win[c][d] * Wout[o][d];
            const g = (sigmoid(sc) - label) * lr;
            for (let d = 0; d < D; d++) { grad[d] += g * Wout[o][d]; Wout[o][d] -= g * Win[c][d]; }
          }
          for (let d = 0; d < D; d++) Win[c][d] -= grad[d];
          pairsSeen++;
        }
      });
    }
  }
  const cos = (a, b) => { let s = 0, na = 0, nb = 0; for (let d = 0; d < D; d++) { s += a[d] * b[d]; na += a[d] * a[d]; nb += b[d] * b[d]; } return s / Math.sqrt(na * nb + 1e-12); };
  const shown = vocab.filter((w) => GROUP[w]);
  function pca2() {
    const X = shown.map((w) => { const v = Win[idx.get(w)], n = Math.hypot(...v) || 1; return Array.from(v, (x) => x / n); });
    const mu = Array.from({ length: D }, (_, d) => X.reduce((a, x) => a + x[d], 0) / X.length);
    const Xc = X.map((x) => x.map((v, d) => v - mu[d]));
    const C = Array.from({ length: D }, (_, i) => Array.from({ length: D }, (_, j) => Xc.reduce((a, x) => a + x[i] * x[j], 0)));
    const vecs = [];
    for (let k = 0; k < 2; k++) {
      let v = Array.from({ length: D }, (_, i) => Math.sin(i + 1 + k * 7));
      for (let it = 0; it < 60; it++) {
        let nv = C.map((row) => row.reduce((a, c, j) => a + c * v[j], 0));
        for (const u of vecs) { const p = nv.reduce((a, x, i) => a + x * u[i], 0); nv = nv.map((x, i) => x - p * u[i]); }
        const n = Math.hypot(...nv) || 1; v = nv.map((x) => x / n);
      }
      vecs.push(v);
    }
    return Xc.map((x) => vecs.map((u) => x.reduce((a, v, i) => a + v * u[i], 0)));
  }
  const pl = plane({ width: 520, height: 400, xr: [-1, 1], yr: [-0.85, 0.85], label: "Learned word vectors projected to 2D" });
  const st = stats([{ label: "Word-context pairs seen" }, { label: "Closest words to the inspected one" }]);
  const nnEl = h("div");
  function draw() {
    const P = pca2();
    const m = Math.max(0.01, ...P.flat().map(Math.abs));
    pl.clear(); pl.axes();
    shown.forEach((w, i) => {
      const x = (P[i][0] / m) * 0.72 - 0.06, y = (P[i][1] / m) * 0.75, col = pl.colors[GCOL[GROUP[w]]];
      pl.dot(pl.X(x), pl.Y(y), w === focus ? 7 : 5, col);
      pl.text(w, pl.X(x) + 8, pl.Y(y), { size: 12, color: w === focus ? pl.colors.ink : pl.colors["ink-2"], weight: w === focus ? 700 : 500 });
    });
    st.set(0, pairsSeen.toLocaleString());
    const f = Win[idx.get(focus)];
    const ranked = shown.filter((w) => w !== focus).map((w) => [w, cos(f, Win[idx.get(w)])]).sort((a, b) => b[1] - a[1]).slice(0, 5);
    st.set(1, ranked.slice(0, 3).map((x) => x[0]).join(", "));
    nnEl.replaceChildren(table(["Word", "Cosine similarity to " + focus], ranked.map(([w, s]) => [w, s.toFixed(3)])));
  }
  const lp = loop(() => { trainChunk(30); epoch++; draw(); if (pairsSeen > 160000) lp.stop(); }, pl.el);
  init();

  root.append(
    section(null, prose(`<p>Networks eat numbers, so words (or products, users, songs) need to become vectors. The naive way is <b>one-hot</b>: a 50,000-long vector with a single 1. Every word is equally far from every other; “cat” is as different from “dog” as from “banana”.</p>
<p>An <b>embedding</b> is a short dense vector, learned so that things used in similar ways end up close together. The idea behind it is the <b>distributional hypothesis</b>: “you shall know a word by the company it keeps” (J. R. Firth, 1957). Word2vec (Mikolov et al., 2013) turned it into a training objective: predict which words appear near each other.</p>`)),
    section("Train word2vec in the page",
      prose("<p>The model below starts with random 12-dimensional vectors for 20 words and trains on a generated corpus of 1,400 little sentences (“the queen lives in the castle”, “the dog chases the cat”). Nobody tells it which words are animals or royals. Watch the groups form; the plot is a PCA projection of the 12 dimensions down to 2.</p>"),
      h("div.stage",
        h("div.stage-controls",
          button("Train", () => lp.start(), "primary"), button("Pause", () => lp.stop()),
          button("Restart from random", () => { lp.stop(); init(); draw(); }),
          slider({ label: "Learning rate", min: 0.005, max: 0.15, step: 0.005, value: lr, fmt: (v) => v.toFixed(3), onInput: (v) => (lr = v) }),
          segmented({ label: "Inspect word", options: ["king", "queen", "cat", "apple", "castle", "girl"].map((w) => ({ value: w, label: w })), value: focus, onChange: (v) => { focus = v; draw(); } })),
        h("div.stage-body", st.el, h("div.lab", h("div.canvas-box", pl.el), nnEl),
          h("div.legend", Object.entries(GCOL).map(([g, c]) => h("span", h("i", { style: { background: `var(--${c})` } }), g))))),
      callout("tip", "Things to try", "<ul style='margin:0;padding-left:18px'><li>Before training, neighbours are random. After ~50,000 pairs, fruit, animals, places and people separate.</li><li>Inspect “king”: its closest words are other royals, because royals share contexts (rules, castle). Inspect “girl”: female words come close because only they appear before “said she”.</li><li>Real word2vec on billions of words gets the famous king − man + woman ≈ queen. This corpus is too small for reliable analogies, which is itself a lesson: geometry comes from data.</li></ul>"),
      predict({
        chapter: id,
        question: "Two words never appear in the same sentence, but they appear in identical contexts (“a ripe ___ is sweet”). Where do their embeddings end up?",
        options: ["Far apart, since they never co-occur", "Close together", "Exactly opposite", "Random: no signal"],
        answer: 1,
        explain: "<p>Word2vec pulls a word's vector towards the vectors of its context words. Two words with the same contexts get pulled towards the same places, so they end up near each other even if they never meet. Synonyms rarely co-occur; they substitute. That's exactly what embeddings capture.</p>",
      }),
    ),
    section("Embeddings everywhere",
      table(["Thing embedded", "Learned from", "Used for"], [
        ["Tokens", "Next-token prediction (the first layer of every LLM)", "The input to the transformer"],
        ["Sentences, documents", "Contrastive training on pairs that belong together", "Semantic search, RAG, clustering"],
        ["Images and text together", "CLIP: matching captions to images", "Image search, text-to-image guidance"],
        ["Users and items", "Who clicked what", "Recommendations (two-tower models)"],
        ["Graph nodes", "Random walks or message passing", "Fraud rings, molecules, social graphs"],
      ]),
      prose("<p><b>Contrastive learning</b> generalises word2vec's trick: pull matching pairs together, push everything else apart. With a batch of N pairs, you score all N × N combinations and train each row to pick its true partner (the InfoNCE loss). It's how sentence-embedding models and CLIP are trained, and it needs no labels beyond “these two belong together”.</p>"),
    ),
    walk("embeddings"),
    section("In code",
      code({
        PyTorch: `import torch, torch.nn.functional as F

emb = torch.nn.Embedding(vocab_size, 128)          # one learnable vector per token
ctx = torch.nn.Embedding(vocab_size, 128)

def sgns_loss(center, context, negatives):         # skip-gram with negative sampling
    c = emb(center)                                # (B, d)
    pos = (c * ctx(context)).sum(-1)               # score for the true neighbour
    neg = torch.bmm(ctx(negatives), c.unsqueeze(-1)).squeeze(-1)   # (B, k)
    return -(F.logsigmoid(pos) + F.logsigmoid(-neg).sum(-1)).mean()

def info_nce(a, b, temperature=0.07):              # contrastive: row i should match column i
    a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    logits = a @ b.T / temperature
    return F.cross_entropy(logits, torch.arange(len(a)))`,
      }),
    ),
  );
  pl.paint(draw);
}
