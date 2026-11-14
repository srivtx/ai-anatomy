import { h, rng } from "../lib/core.js";
import { slider, stats, predict, code, callout, section, prose, table, toggle, button } from "../lib/widgets.js";
import { xyChart } from "../lib/plot.js";
import { gauss } from "../lib/nn.js";

const DOCS = [
  { name: "Encyclopedia-style paragraph", text: "The lighthouse at the end of the harbour was built in 1871 to guide fishing boats past the rocks. Its lamp was first fuelled by oil and later converted to electricity. Today it is maintained by volunteers and is open to visitors during the summer months." },
  { name: "Keyword-stuffed spam", text: "cheap shoes cheap shoes best cheap shoes buy now!!! $$$ discount discount shoes shoes shoes >>> click >>> best price best price ### free shipping ###" },
  { name: "Cookie banner boilerplate", text: "Accept all cookies... Manage preferences... Reject non-essential... Privacy policy... Terms..." },
  { name: "Forum answer", text: "If your bread is too dense, the dough probably did not rise long enough. Try leaving it somewhere warm for another hour and check that your yeast is still active by mixing it with warm water and a little sugar first." },
  { name: "Table of numbers", text: "12 44 91 0.33 17 | 18 22 07 0.61 99 | 31 08 15 0.12 46 | 77 52 63 0.98 04 | 25 19 88 0.47 30" },
  { name: "Short nav menu", text: "Home About Contact" },
];
const STOP = new Set(["the", "be", "to", "of", "and", "that", "have", "with", "a", "is", "it", "in", "for", "your"]);
const RULES = {
  length: { label: "30+ words", test: (w) => w.length >= 30 },
  wordlen: { label: "Mean word length 3–10", test: (w) => { const m = w.reduce((a, x) => a + x.length, 0) / Math.max(1, w.length); return m >= 3 && m <= 10; } },
  symbols: { label: "Few symbols (#, …, $, >) per word", test: (w, t) => (t.match(/[#$>]|\.\.\./g) || []).length / Math.max(1, w.length) < 0.1 },
  alpha: { label: "80% of words contain a letter", test: (w) => w.filter((x) => /[a-z]/i.test(x)).length / Math.max(1, w.length) >= 0.8 },
  stop: { label: "At least 2 common English words", test: (w) => w.filter((x) => STOP.has(x.toLowerCase())).length >= 2 },
  repeat: { label: "Low word repetition", test: (w) => new Set(w.map((x) => x.toLowerCase())).size / Math.max(1, w.length) > 0.5 },
};

// MinHash over word 3-shingles
const shingles = (t) => { const w = t.toLowerCase().match(/[a-z0-9']+/g) || [], s = new Set(); for (let i = 0; i + 2 < w.length; i++) s.add(w.slice(i, i + 3).join(" ")); return s; };
function hash(str, seed) { let h1 = 2166136261 ^ seed; for (let i = 0; i < str.length; i++) { h1 ^= str.charCodeAt(i); h1 = Math.imul(h1, 16777619); } return h1 >>> 0; }
const minhash = (set, k) => Array.from({ length: k }, (_, i) => Math.min(...[...set].map((x) => hash(x, i * 7919 + 1))));

export default function (root, { id }) {
  // 1. quality rules
  const on = Object.fromEntries(Object.keys(RULES).map((k) => [k, k !== "length"]));
  const docsEl = h("div"), qst = stats([{ label: "Documents kept" }]);
  function drawDocs() {
    let kept = 0;
    docsEl.replaceChildren(table(["Document", "Text", "Failed rules", ""], DOCS.map((d) => {
      const words = d.text.split(/\s+/).filter(Boolean);
      const failed = Object.entries(RULES).filter(([k, r]) => on[k] && !r.test(words, d.text)).map(([, r]) => r.label);
      if (!failed.length) kept++;
      return [d.name, h("span", { style: { fontSize: "13px", color: "var(--ink-2)" } }, d.text.length > 120 ? d.text.slice(0, 120) + "…" : d.text), failed.join("; ") || "—", failed.length ? h("span.pill.bad", "dropped") : h("span.pill.good", "kept")];
    })));
    qst.set(0, `${kept} of ${DOCS.length}`);
  }

  // 2. near-duplicate detection
  const a = h("textarea.ta", { rows: 4, "aria-label": "Document A" }, "The lighthouse at the end of the harbour was built in 1871 to guide fishing boats past the rocks. Its lamp was first fuelled by oil and later converted to electricity.");
  const b = h("textarea.ta", { rows: 4, "aria-label": "Document B" }, "The lighthouse at the end of the old harbour was built in 1871 to guide the fishing boats past the rocks. Its lamp was fuelled by oil and later converted to electricity.");
  let k = 128;
  const dst = stats([{ label: "True Jaccard similarity" }, { label: "MinHash estimate" }, { label: "Verdict at threshold 0.5" }]);
  function drawDup() {
    const sa = shingles(a.value), sb = shingles(b.value);
    const inter = [...sa].filter((x) => sb.has(x)).length, uni = new Set([...sa, ...sb]).size;
    const J = uni ? inter / uni : 0;
    const ma = sa.size ? minhash(sa, k) : [], mb = sb.size ? minhash(sb, k) : [];
    const est = ma.length && mb.length ? ma.filter((v, i) => v === mb[i]).length / k : 0;
    dst.set(0, J.toFixed(3)); dst.set(1, est.toFixed(3)); dst.set(2, est >= 0.5 ? "near-duplicate: keep one" : "different", est >= 0.5 ? "warn" : "good");
  }
  a.addEventListener("input", drawDup); b.addEventListener("input", drawDup);

  // 3. model collapse
  let n = 50;
  const col = xyChart({ height: 240, xLabel: "generation (each trained only on the previous one's samples)", yLabel: "spread (std) of the learned distribution", yLog: true });
  function drawCollapse() {
    const series = [];
    ["c1", "c2", "c3", "c6"].forEach((c, run) => {
      const r = rng(31 + run);
      let mu = 0, sd = 1; const pts = [[0, 1]];
      for (let g = 1; g <= 200; g++) {
        const xs = Array.from({ length: n }, () => mu + sd * gauss(r));
        mu = xs.reduce((p, x) => p + x, 0) / n;
        sd = Math.sqrt(xs.reduce((p, x) => p + (x - mu) ** 2, 0) / n);
        pts.push([g, Math.max(sd, 1e-6)]);
      }
      series.push({ name: `run ${run + 1}`, color: `var(--${c})`, points: pts });
    });
    col.update(series, { yDomain: [1e-3, 2] });
  }

  root.append(
    section(null, prose(`<p>A model knows what its data contains, in the proportions its data contains it. Labs treat the data pipeline as seriously as the architecture, and they're far more secretive about it. Pretraining data starts as a raw web crawl (Common Crawl has petabytes of it) and is cut down to a few percent through filtering, deduplication and mixing, then topped up with code, books, maths and, increasingly, <b>synthetic</b> text.</p>`)),
    section("Filter: throw out the junk",
      prose("<p>The first filters are simple heuristics. These are adapted from DeepMind's Gopher rules (Rae et al., 2021). Toggle them and see which documents survive. Modern pipelines then add a <b>model-based</b> filter: a small classifier trained to score “educational value” (FineWeb-Edu, 2024), which helped benchmark scores more than any rule.</p>"),
      h("div.stage",
        h("div.stage-controls", Object.entries(RULES).map(([key, r]) => toggle({ label: r.label, checked: on[key], onChange: (v) => { on[key] = v; drawDocs(); } }))),
        h("div.stage-body", qst.el, docsEl)),
      callout("tip", "Things to try", "<p>Turn on the minimum length (30 words here because the samples are short; Gopher uses 50): the two good paragraphs survive and the nav menu goes. Try switching rules off one at a time to see which junk each one catches. Every rule also throws away some good text somewhere; the art is choosing rules whose mistakes you can afford.</p>"),
    ),
    section("Deduplicate: the web repeats itself",
      prose("<p>The same article, licence text or product listing appears thousands of times. Duplicates waste compute and make models memorise. Exact matches are easy; <b>near</b>-duplicates are found with <b>MinHash</b>: break each document into overlapping word triples, and keep only the minimum of many hash functions over them. The fraction of matching minimums estimates the Jaccard similarity, so billions of documents can be compared via short signatures (plus locality-sensitive hashing to avoid comparing every pair).</p>"),
      h("div.stage",
        h("div.stage-controls", slider({ label: "Hash functions (signature length)", min: 8, max: 256, step: 8, value: k, onInput: (v) => { k = v; drawDup(); }, hint: "More hashes: a more accurate estimate" })),
        h("div.stage-body", dst.el, h("div.lab", a, b))),
    ),
    section("Mix: decide what the model is",
      table(["Ingredient", "Typical share", "What it buys"], [
        ["Filtered web text", "Most of the tokens", "Breadth: general knowledge and language"],
        ["Code", "Often 10–25%", "Programming, and it seems to help structured reasoning too"],
        ["Maths and science", "Several %", "Step-by-step reasoning; heavily upsampled late in training"],
        ["Books, papers, encyclopedias", "Several %", "Long-range coherence and factual density"],
        ["Multilingual", "Varies widely", "Other languages; competes with English for capacity"],
        ["Synthetic (model-written)", "Growing", "Targeted skills, clean formats, reasoning traces"],
      ], { boldFirst: true }),
      prose("<p>Shares above are typical published ranges, not one lab's recipe. Mixes also change <i>during</i> training: many labs end with an <b>annealing</b> phase that upweights the highest-quality data while the learning rate decays. Finally, test sets are scrubbed from the training data (<b>decontamination</b>); when that fails, benchmark scores mean much less (see the evals chapter).</p>"),
    ),
    section("Synthetic data, and what happens when models eat themselves",
      prose("<p>Models can write training data: textbook-style explanations (Microsoft's Phi models), solved maths problems verified by a checker, or reasoning traces from a bigger model. Done carefully it works well. Done carelessly, training each generation only on the previous generation's output leads to <b>model collapse</b> (Shumailov et al., 2024): rare things in the tails disappear first, then the distribution narrows. Here's the simplest version: fit a Gaussian to n samples, sample n new points from the fit, refit, repeat.</p>"),
      h("div.stage",
        h("div.stage-controls", slider({ label: "Samples per generation", min: 10, max: 1000, step: 10, value: n, onInput: (v) => { n = v; drawCollapse(); } }), button("Redraw", drawCollapse)),
        h("div.stage-body", col.el)),
      predict({
        chapter: id,
        question: "Why does the fitted spread shrink over generations, even though each fit is unbiased-looking?",
        options: ["A bug in the random number generator", "Each finite sample under-represents the tails, and those losses compound; nothing ever brings the tails back", "Gaussians always shrink", "The mean drifts to zero"],
        answer: 1,
        explain: "<p>Each generation's estimate is a little noisy, and the sample variance is slightly biased low; over many generations the errors compound like a random walk with a downward drift, and lost tails never return. Mixing in fresh real data each generation stops the collapse, which is why real data stays valuable even as synthetic data grows.</p>",
      }),
    ),
    section("In code",
      code({
        "datatrove / Python": `# a Gopher-style quality filter
def keep(doc: str) -> bool:
    words = doc.split()
    if not 50 <= len(words) <= 100_000: return False
    mean_len = sum(map(len, words)) / len(words)
    if not 3 <= mean_len <= 10: return False
    if sum(w.lower() in STOP for w in words) < 2: return False
    if sum(any(c.isalpha() for c in w) for w in words) / len(words) < 0.8: return False
    return True

# near-dedup with MinHash + LSH
from datasketch import MinHash, MinHashLSH
lsh = MinHashLSH(threshold=0.8, num_perm=128)
for i, doc in enumerate(docs):
    m = MinHash(num_perm=128)
    for sh in shingles(doc): m.update(sh.encode())
    if not lsh.query(m):       # no near-duplicate seen yet
        lsh.insert(str(i), m); keep_ids.append(i)`,
      }),
    ),
  );
  drawDocs(); drawDup(); drawCollapse();
}
