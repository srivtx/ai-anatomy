import { h, loop } from "../lib/core.js";
import { goal, whyChain, sticker } from "../lib/why.js";
import { walk } from "../lib/walks.js";
import { slider, stats, predict, code, callout, section, prose, table, button } from "../lib/widgets.js";
import { xyChart } from "../lib/plot.js";
import { CORPUS } from "../lib/corpus.js";

// Byte-pair encoding over characters, GPT-style: split into words that carry
// their leading space, then repeatedly merge the most frequent adjacent pair.
const preSplit = (text) => text.match(/ ?[A-Za-z]+| ?[0-9]+| ?[^A-Za-z0-9\s]+|\s+/g) || [];

function trainer(text) {
  const words = new Map();
  for (const w of preSplit(text)) words.set(w, (words.get(w) || 0) + 1);
  let seqs = [...words].map(([w, n]) => ({ toks: [...w], n }));
  const merges = [];
  const base = new Set([...text]);
  return {
    merges, get vocab() { return base.size + merges.length; },
    tokens() { return seqs.reduce((a, s) => a + s.toks.length * s.n, 0); },
    step() {
      const pairs = new Map();
      for (const s of seqs) for (let i = 0; i < s.toks.length - 1; i++) { const k = s.toks[i] + "\u0000" + s.toks[i + 1]; pairs.set(k, (pairs.get(k) || 0) + s.n); }
      let best = null;
      for (const [k, n] of pairs) if (!best || n > best[1]) best = [k, n];
      if (!best || best[1] < 2) return false;
      const [a, b] = best[0].split("\u0000");
      merges.push({ a, b, n: best[1] });
      seqs = seqs.map((s) => { const t = []; for (let i = 0; i < s.toks.length; i++) { if (i < s.toks.length - 1 && s.toks[i] === a && s.toks[i + 1] === b) { t.push(a + b); i++; } else t.push(s.toks[i]); } return { toks: t, n: s.n }; });
      return true;
    },
  };
}
function encode(text, merges) {
  const rank = new Map(merges.map((m, i) => [m.a + "\u0000" + m.b, i]));
  const out = [];
  for (const w of preSplit(text)) {
    let t = [...w];
    for (;;) { // apply the earliest-learned merge available, like a real BPE encoder
      let bi = -1, br = Infinity;
      for (let i = 0; i < t.length - 1; i++) { const r = rank.get(t[i] + "\u0000" + t[i + 1]); if (r !== undefined && r < br) { br = r; bi = i; } }
      if (bi < 0) break;
      t = [...t.slice(0, bi), t[bi] + t[bi + 1], ...t.slice(bi + 2)];
    }
    out.push(...t);
  }
  return out;
}

export default function (root, { id }) {
  let T = trainer(CORPUS), target = 200, history = [[0, CORPUS.length / T.tokens()]];
  const st = stats([{ label: "Merges learned" }, { label: "Vocabulary size" }, { label: "Corpus length in tokens" }, { label: "Characters per token" }]);
  const mergesEl = h("pre.log", { style: { maxHeight: "260px" } });
  const chips = h("div.chips"), tst = stats([{ label: "Characters" }, { label: "Tokens" }]);
  const input = h("textarea.ta", { rows: 3, "aria-label": "Text to tokenize" }, "The lighthouse keeper learned from his mistakes. Unbelievably, strawberry has three r's. 12345 + 678 = 13023.");
  const chart = xyChart({ width: 460, height: 260, xLabel: "merges", yLabel: "characters per token" });
  const COLORS = ["c1", "c2", "c3", "c4", "c5", "c6"];
  function draw() {
    st.set(0, String(T.merges.length)); st.set(1, String(T.vocab)); st.set(2, T.tokens().toLocaleString()); st.set(3, (CORPUS.length / T.tokens()).toFixed(2));
    mergesEl.textContent = T.merges.slice(-40).map((m, i, arr) => `${String(T.merges.length - arr.length + i + 1).padStart(3)}  ${JSON.stringify(m.a)} + ${JSON.stringify(m.b)} → ${JSON.stringify(m.a + m.b)}   (${m.n}×)`).reverse().join("\n") || "No merges yet: every character is its own token.";
    chart.update([{ name: "Compression", color: "var(--accent)", points: history }]);
    const toks = encode(input.value, T.merges);
    chips.replaceChildren(...toks.map((t, i) => h("span.chip", { title: JSON.stringify(t), style: { background: `color-mix(in srgb, var(--${COLORS[i % 6]}) 24%, transparent)` } }, t.replace(/ /g, "·").replace(/\n/g, "↵"))));
    tst.set(0, String(input.value.length)); tst.set(1, String(toks.length));
  }
  input.addEventListener("input", draw);
  const lp = loop(() => {
    for (let k = 0; k < 3 && T.merges.length < target; k++) { if (!T.step()) { lp.stop(); break; } history.push([T.merges.length, CORPUS.length / T.tokens()]); }
    draw();
    if (T.merges.length >= target) lp.stop();
  }, mergesEl);

  root.append(
    section(null, goal("<p>A model reads numbers from a fixed list. We want a list <b>short enough</b> that each entry is seen often enough to learn, with entries <b>long enough</b> that a sentence isn't thousands of steps.</p>", ["covers any text, with no unknown words", "common words in one piece", "a vocabulary size we choose"]),
      prose(`<p>A language model reads numbers, not letters. The <b>tokenizer</b> decides what the numbers stand for. One per character is simple but makes sequences long: attention costs grow with the square of length, and each step learns little. One per word blows up the vocabulary and can't handle typos or new words. Every modern LLM sits in between, with <b>sub-word</b> tokens learned by <b>byte-pair encoding</b> (BPE): an old compression algorithm (Gage, 1994) repurposed for GPT-2 in 2019.</p>
<p>The algorithm fits in one sentence: <i>find the most frequent adjacent pair of tokens in the corpus, make it a new token, repeat.</i></p>`),
      h("div.vx-row", sticker("tokens", { title: "un · believ · able", caption: "rare words split into common pieces; common words stay whole", px: 4 }))),
    section("Train a tokenizer on a short story",
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Target number of merges", min: 0, max: 500, step: 10, value: target, onInput: (v) => (target = v) }),
          button("Train", () => lp.start(), "primary"),
          button("One merge", () => { if (T.step()) history.push([T.merges.length, CORPUS.length / T.tokens()]); draw(); }),
          button("Reset", () => { lp.stop(); T = trainer(CORPUS); history = [[0, CORPUS.length / T.tokens()]]; draw(); })),
        h("div.stage-body", st.el,
          h("div.lab", h("div", h("div.eyebrow", "Latest merges (newest first)"), mergesEl), chart.el),
          h("div.eyebrow", "Tokenize anything with the merges learned so far"), input, tst.el, chips)),
      callout("tip", "Things to try", "<ul style='margin:0;padding-left:18px'><li>Press One merge a few times. GPT-style pre-splitting attaches the space to the <i>start</i> of each word, so early merges look like “ t” + “h”, then “ th” + “e”. Frequent words become single tokens within about 100 merges.</li><li>Type a word that isn't in the story, like “unbelievably”. It splits into pieces it does know. No word is ever out of vocabulary.</li><li>Look at “strawberry”: the model sees two or three chunks, not ten letters, which is why counting letters is surprisingly hard for LLMs.</li></ul>"),
      predict({
        chapter: id,
        question: "A model's tokenizer was trained mostly on English. You send it the same paragraph in Hindi. What happens?",
        options: ["Same number of tokens, since meaning is the same", "Several times more tokens: higher cost, slower, and less fits in the context", "Fewer tokens, since Hindi is compact", "It refuses non-English text"],
        answer: 1,
        explain: "<p>Merges reflect the training corpus. Scripts that were rare get few merges, so they fall back towards one token per byte or character. The same content can cost 3 to 10 times as many tokens, which means more money, slower generation and a smaller effective context. Newer tokenizers train on more balanced, multilingual data for this reason.</p>",
      }),
    ),
    section("Real tokenizers",
      table(["Tokenizer", "Vocabulary", "Details"], [
        ["GPT-2", "50,257", "Byte-level BPE: works on UTF-8 bytes, so any string is encodable"],
        ["cl100k / o200k (OpenAI)", "100k / 200k", "Bigger vocabularies compress code and non-English text better"],
        ["Llama 3", "128k", "tiktoken-style BPE"],
        ["SentencePiece (T5, Gemma)", "32k–256k", "Treats the space as a symbol (▁); BPE or unigram LM"],
        ["Byte-level models (BLT, 2024)", "256 bytes", "No tokenizer: group bytes into dynamic patches by entropy instead"],
      ]),
      prose("<p>Tokenization causes a surprising share of LLM oddities: trouble with arithmetic (numbers split inconsistently), spelling and reversing words, extra cost for some languages, and “glitch tokens” that appeared in the tokenizer's training data but almost never in the model's. Some labs now split digits one at a time to make arithmetic easier.</p>"),
    ),
    walk("tokenization"),
    section("Why models can't spell",
      whyChain({
        claim: "Models often miscount the r's in <i>strawberry</i>.",
        steps: [
          { q: "Why? It's three letters.", a: "<p>The model never sees the letters. <i>strawberry</i> arrives as two or three token IDs, something like <code>str</code> <code>aw</code> <code>berry</code>. To count r's it must have learned how each token is spelled, indirectly, from context.</p>" },
          { q: "Why not just feed it characters?", a: "<p>Sequences get about 4× longer. Attention's cost grows with the square of the length, so roughly 16× the attention work, and each step predicts less, so learning the same thing takes more steps.</p>" },
          { q: "Why does BPE choose these pieces?", a: "<p>It merges the most frequent adjacent pair, over and over. Frequent strings become single tokens; rare ones stay split. It's a simple, greedy way to make typical text as short as possible for a fixed vocabulary size.</p>" },
        ],
        axiom: { name: "Source coding: frequent things deserve short codes", html: "<p>To make messages short on average, give each pattern a code of about −log₂ p bits: common patterns short, rare ones long (Shannon, 1948). A tokenizer spends its vocabulary slots on the pieces that save the most length, which is the same idea done greedily.</p>" },
      })),
    section("In code",
      code({
        Python: `from collections import Counter

def train_bpe(words: list[str], n_merges: int):
    seqs = Counter(tuple(w) for w in words)          # word -> frequency
    merges = []
    for _ in range(n_merges):
        pairs = Counter()
        for seq, n in seqs.items():
            for a, b in zip(seq, seq[1:]):
                pairs[a, b] += n
        if not pairs:
            break
        (a, b), _ = pairs.most_common(1)[0]
        merges.append((a, b))
        new = Counter()
        for seq, n in seqs.items():
            out, i = [], 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                    out.append(a + b); i += 2
                else:
                    out.append(seq[i]); i += 1
            new[tuple(out)] += n
        seqs = new
    return merges`,
        tiktoken: `import tiktoken
enc = tiktoken.get_encoding("o200k_base")
ids = enc.encode("Unbelievably, strawberry has three r's.")
print(len(ids), [enc.decode([i]) for i in ids])`,
      }, "Project 05 builds a byte-level version and measures compression against vocabulary size."),
    ),
  );
  draw();
}
