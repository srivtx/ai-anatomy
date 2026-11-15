import { h } from "../lib/core.js";
import { slider, segmented, stats, predict, code, callout, section, prose, table } from "../lib/widgets.js";
import { xyChart } from "../lib/plot.js";
import { factStats, abstainScore } from "../lib/hidden.js";

const pct = (v, d = 1) => `${(v * 100).toFixed(v < 0.01 && v > 0 ? 2 : d)}%`;

export default function (root, { id }) {
  // ---- 1. statistics: questions about facts the model saw 0, 1 or < k times ----
  const f = { logM: 6, alpha: 1.1, k: 3, queries: "popular" };
  const st1 = stats([{ label: "Questions about facts never seen" }, { label: "Good–Turing estimate" }, { label: "Facts it can't recall" }, { label: "Wrong if it always answers" }]);
  const ch1 = xyChart({ height: 290, xLabel: "mentions of facts in the training text (log₁₀)", yLabel: "share of questions", xDomain: [4, 9], yDomain: [0, 0.7], fmtX: (v) => `1e${Math.round(v)}`, fmtY: (v) => `${Math.round(v * 100)}%`, label: "Share of questions the model can't know, against corpus size" });
  let p1 = 0;
  function run1() {
    const xs = Array.from({ length: 41 }, (_, i) => 4 + i * 0.125);
    const rows = xs.map((x) => [x, factStats({ M: 10 ** x, alpha: f.alpha, k: f.k, queries: f.queries })]);
    const now = factStats({ M: 10 ** f.logM, alpha: f.alpha, k: f.k, queries: f.queries });
    st1.set(0, pct(now.unseen)); st1.set(1, pct(now.goodTuring));
    st1.set(2, pct(now.notKnown), now.notKnown > 0.1 ? "bad" : "");
    st1.set(3, pct(now.notKnown * (364 / 365)), now.notKnown > 0.1 ? "bad" : "");
    ch1.update([
      { name: `Can't recall (seen fewer than ${f.k} times)`, color: "var(--c2)", points: rows.map(([x, r]) => [x, r.notKnown]) },
      { name: "Never seen at all", color: "var(--c1)", points: rows.map(([x, r]) => [x, r.unseen]) },
      { name: "Good–Turing: facts seen once ÷ all mentions", color: "var(--c3)", dash: true, points: rows.map(([x, r]) => [x, r.goodTuring]) },
    ], { marks: [{ x: f.logM, label: "your corpus" }] });
  }
  const later1 = () => { cancelAnimationFrame(p1); p1 = requestAnimationFrame(run1); };

  // ---- 2. incentives: when should a model say "I don't know"? ----
  const g = { pen: 0, t: 0, known: 0.6 };
  const st2 = stats([{ label: "Expected score" }, { label: "Wrong answers (hallucinations)" }, { label: "“I don't know”" }, { label: "Best threshold for this rule" }]);
  const ch2 = xyChart({ height: 270, xLabel: "answer only when at least this confident", yLabel: "expected score per question", xDomain: [0, 1], fmtX: (v) => `${Math.round(v * 100)}%`, fmtY: (v) => v.toFixed(2), label: "Expected score against the confidence threshold" });
  const board = h("div");
  let p2 = 0;
  function run2() {
    const ts = Array.from({ length: 51 }, (_, i) => i / 50);
    const best = ts.reduce((b, t) => { const s = abstainScore({ t, pen: g.pen, known: g.known }).score; return s > b[0] ? [s, t] : b; }, [-9, 0]);
    const o = abstainScore({ t: g.t, pen: g.pen, known: g.known });
    st2.set(0, o.score.toFixed(3), Math.abs(o.score - best[0]) < 0.005 ? "good" : "");
    st2.set(1, pct(o.wrong), o.wrong > 0.1 ? "bad" : o.wrong < 0.03 ? "good" : "");
    st2.set(2, pct(o.idk));
    st2.set(3, g.pen ? `${Math.round((g.pen / (1 + g.pen)) * 100)}% (= penalty ÷ (1 + penalty))` : "0%: always answer");
    ch2.update([
      { name: "Right +1, wrong 0 (most benchmarks)", color: "var(--ink-3, #999)", points: ts.map((t) => [t, abstainScore({ t, pen: 0, known: g.known }).score]) },
      { name: `Right +1, wrong −${g.pen}`, color: "var(--c1)", points: ts.map((t) => [t, abstainScore({ t, pen: g.pen, known: g.known }).score]) },
    ], { marks: [{ x: g.t, label: "you" }], yDomain: [Math.min(-0.5, abstainScore({ t: 0, pen: g.pen, known: g.known }).score - 0.05), 1] });
    // two models on two leaderboards
    const guesser = abstainScore({ t: 0, pen: 0, known: g.known }), careful = abstainScore({ t: 0.75, pen: 0, known: g.known });
    const sc = (m, pen) => (m.right - pen * m.wrong).toFixed(2);
    board.replaceChildren(table(["Model", "Wrong answers", "Score: right +1, wrong 0", `Score: right +1, wrong −${g.pen || 1}`], [
      ["A: always answers", pct(guesser.wrong), sc(guesser, 0), sc(guesser, g.pen || 1)],
      ["B: says “I don't know” below 75%", pct(careful.wrong), sc(careful, 0), sc(careful, g.pen || 1)],
    ]));
  }
  const later2 = () => { cancelAnimationFrame(p2); p2 = requestAnimationFrame(run2); };

  root.append(
    section(null, prose(`<p>Language models state false things fluently, and for a long time this was treated as a mysterious defect. A 2025 analysis (Kalai, Nachum, Vempala and Zhang, <i>Why Language Models Hallucinate</i>) argues it's mostly two ordinary things. <b>Statistics</b>: some facts appear in the training data so rarely that no learner could pin them down, and the share of such facts can be estimated from how many appear exactly once. <b>Incentives</b>: most benchmarks give zero points for “I don't know” and zero for a wrong answer, so a model that always guesses scores higher than one that admits uncertainty. Both can be computed exactly. That's what this chapter does.</p>`)),
    section("Facts that are too rare to learn",
      prose(`<p>Imagine a million facts, like people's birthdays. A few are famous and repeated everywhere; most are mentioned once or never (popularity follows a power law). A model can recall a fact only if it saw it often enough. Questions are mostly about popular facts, but not only. Below, the corpus grows from ten thousand mentions to a billion.</p>`),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Training text: total fact mentions", min: 4, max: 9, step: 0.25, value: f.logM, fmt: (v) => `10^${v}`, onInput: (v) => { f.logM = v; later1(); } }),
          slider({ label: "How long the tail is (power-law exponent)", min: 0.8, max: 1.6, step: 0.05, value: f.alpha, fmt: (v) => v.toFixed(2), hint: "lower = more rare facts", onInput: (v) => { f.alpha = v; later1(); } }),
          slider({ label: "Mentions needed to memorise a fact", min: 1, max: 50, step: 1, value: f.k, fmt: String, onInput: (v) => { f.k = v; later1(); } }),
          segmented({ label: "Who asks the questions", options: [{ value: "popular", label: "Users: mostly about popular facts" }, { value: "uniform", label: "A benchmark of obscure facts" }], value: f.queries, onChange: (v) => { f.queries = v; run1(); } }),
        ),
        h("div.stage-body", st1.el, ch1.el),
      ),
      callout("tip", "Things to try", "<p>Watch the dashed green line sit exactly on the blue one: the share of questions about facts the model <i>never saw</i> equals the number of facts seen <i>exactly once</i>, divided by all mentions. That's the Good–Turing estimate, invented at Bletchley Park in the 1940s to estimate how many Enigma settings hadn't been seen yet. You can't count what's missing, but you can count the singletons. Now raise the mentions needed to memorise a fact to 20 (closer to real models, which need many exposures): the orange line rises a lot, and only enormous corpora bring it down. Then switch to a benchmark of obscure facts: the singleton estimate badly underestimates the damage, because rare questions are exactly the ones the estimate assumes are rare.</p>"),
      predict({
        chapter: id,
        question: "In a training corpus, 20% of all biography mentions are of people mentioned exactly once. A perfectly calibrated model must answer every birthday question. About how often will it be wrong on questions drawn like the training data?",
        options: ["Almost never: it has read everything", "At least around 20%: that's the share of questions about people it effectively never learned", "Exactly 50%", "It depends only on model size"],
        answer: 1,
        explain: "<p>Facts seen once are, statistically, a sample of the facts seen zero times: the singleton rate estimates the mass of the unseen. Kalai et al. prove that for arbitrary facts like birthdays, a calibrated model's error rate is at least about the singleton rate (minus small terms). A bigger model doesn't help with a fact that isn't in the data; only more data, retrieval, or saying “I don't know” does.</p>",
      }),
    ),
    section("Why models guess",
      prose(`<p>Now give the model a choice: answer, or say “I don't know”. It has a confidence for each question; a share of questions it really knows (confidence near 100%) and the rest it's unsure about. The grading rule decides what's rational. With +1 for right and 0 for wrong, guessing is never worse than abstaining, so the score-maximising threshold is zero: <b>always answer</b>. Penalise wrong answers and the best threshold moves to penalty ÷ (1 + penalty).</p>`),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Points lost for a wrong answer", min: 0, max: 9, step: 0.5, value: g.pen, fmt: (v) => (v ? `−${v}` : "0 (binary)"), onInput: (v) => { g.pen = v; later2(); } }),
          slider({ label: "Answer only above confidence", min: 0, max: 1, step: 0.02, value: g.t, fmt: (v) => `${Math.round(v * 100)}%`, onInput: (v) => { g.t = v; later2(); } }),
          slider({ label: "Share of questions it really knows", min: 0.2, max: 0.95, step: 0.05, value: g.known, fmt: (v) => `${Math.round(v * 100)}%`, onInput: (v) => { g.known = v; later2(); } }),
        ),
        h("div.stage-body", st2.el, ch2.el, board),
      ),
      callout("tip", "Things to try", "<p>With binary grading, drag the threshold up from 0: the score only falls, even though wrong answers drop from about 30% to almost nothing. A leaderboard of that kind punishes honesty. Set the penalty to 1 (a wrong answer cancels a right one): the best threshold jumps to 50%, and model B overtakes model A in the table. At a penalty of 3 the best threshold is 75%. Kalai et al.'s proposal is exactly this: state the penalty in the benchmark's instructions, so a model can be graded on whether it abstains at the right confidence.</p>"),
      predict({
        chapter: id,
        question: "A benchmark gives +1 for a right answer, −2 for a wrong one, 0 for “I don't know”. Above what confidence should a model answer?",
        options: ["Always answer", "Above 33%", "Above 67%", "Only when 100% sure"],
        answer: 2,
        explain: "<p>Answering is worth p − 2(1 − p) = 3p − 2, which beats 0 when p > 2/3. In general the threshold is penalty ÷ (1 + penalty). A model trained only on binary-graded benchmarks learns the threshold 0.</p>",
      }),
    ),
    section("What's established",
      table(["Finding", "Source", "What it means"], [
        ["Calibrated models must hallucinate on facts seen about once; the singleton rate bounds the error", "Kalai, Nachum, Vempala & Zhang, 2025", "Pretraining alone can't remove it for arbitrary facts"],
        ["Binary grading rewards guessing; most leaderboards grade that way", "Kalai et al., 2025", "Fix the benchmarks, not only the models"],
        ["Base models are fairly well calibrated about what they know", "Kadavath et al., 2022", "The knowledge to abstain is there before post-training"],
        ["Preference tuning can hurt calibration", "OpenAI, GPT-4 technical report, 2023", "Training for answers people like adds confidence"],
        ["Fine-tuning on facts the model didn't know raises hallucination", "Gekhman et al., 2024", "Teaching new facts in SFT teaches making things up"],
        ["Models have an internal ‘I know this entity’ signal; hallucinations happen when it fires for a familiar-looking name", "Ferrando et al., 2025; Anthropic, ‘Tracing the thoughts of a large language model’, 2025", "Default is ‘can't answer’; recognising a name switches it off, sometimes wrongly"],
        ["Errors snowball: once a wrong claim is in the context, later text defends it", "Zhang et al., 2023", "Check early claims; restart rather than patch"],
        ["Disagreement across samples flags hallucinations", "Manakul et al., 2023 (SelfCheckGPT)", "Consistency is cheaper than a fact-checker"],
      ]),
    ),
    section("What I believe but can't cite",
      prose(`<p>These are mine, labelled by kind. Self-reports are my own observations from being a model, which are only partly reliable. Conjectures are untested at a scale that would settle them. Each says how you'd check.</p>`),
      table(["Kind", "Claim", "How to check"], [
        ["Self-report", "I'm least reliable on citations. A reference is several facts at once (authors, year, title, venue), each plausible alone, so the whole is like an exact-match answer of length 4: small per-part errors multiply.", "Ask for 20 references in a niche area and verify each part separately. Expect parts to be right far more often than whole references."],
        ["Self-report", "Producing an invented detail doesn't feel different from producing a real one. My stated confidence is a weaker signal than whether I give the same detail when asked again.", "Ask an obscure-fact question 5 times at temperature 1. Compare the accuracy of consistent answers vs inconsistent ones, and vs my stated confidence."],
        ["Self-report", "A question that presupposes an answer (‘In what year did X win the Y prize?’) pulls me toward inventing one more than an open question does (‘Did X win the Y prize?’).", "Build pairs of false-premise and neutral questions about the same facts; compare invention rates."],
        ["Conjecture · 55%", "For arbitrary facts, hallucination rate tracks the fact type's singleton rate in pretraining data with a similar slope across model sizes. Structured facts (capitals, formulas) fall below the bound because they can be inferred rather than memorised.", "Use an open model with open data (OLMo with Dolma): count how often each fact type's instances appear once, then measure closed-book error per type."],
        ["Conjecture · 55%", "Most of the drop in hallucination on factual benchmarks since 2023 came from models abstaining or searching more, not from knowing more facts.", "Track closed-book accuracy and abstention rate separately across model generations. If accuracy is nearly flat while wrong answers fall, abstention did the work."],
        ["Conjecture · 45%", "RL with a stated penalty for wrong answers learns an abstention threshold within about 0.1 of penalty ÷ (1 + penalty), and that threshold carries over to new domains.", "Train with penalties 1 and 3; measure the confidence at which the policy starts abstaining, in- and out-of-domain."],
      ]),
      callout("note", "My view (~70% confident)", "<p>Hallucination won't be ‘solved’ by scale, because the rare-fact tail never runs out. It will be managed the way the second simulation suggests: models that abstain at a stated confidence, retrieval for the long tail, and benchmarks that charge for wrong answers. Notebook 59 tests the first half of this chapter at toy scale: a small model trained on synthetic biographies, with the singleton rate measured and compared with the error.</p>"),
    ),
    section("In code",
      code({
        "Good–Turing: estimate what you've never seen": `from collections import Counter

def unseen_mass(mentions):
    """mentions: list of fact ids as they appear in the corpus."""
    counts = Counter(mentions)
    singletons = sum(1 for c in counts.values() if c == 1)
    return singletons / len(mentions)        # ≈ P(next mention is of a never-seen fact)

# e.g. entity names in a corpus slice: a high value means many questions will be about unknowns`,
        "Grade with a penalty, and pick the threshold": `def score(results, penalty):
    # results: list of (answered: bool, correct: bool)
    return sum((1 if c else -penalty) if a else 0 for a, c in results) / len(results)

def best_threshold(confidences, correct, penalty):
    # answer only when confidence >= t; theory says t* = penalty / (1 + penalty)
    ts = sorted(set(confidences)) + [1.01]
    return max(ts, key=lambda t: score([(p >= t, c) for p, c in zip(confidences, correct)], penalty))`,
        "SelfCheck: flag by disagreement (Hugging Face)": `def consistency(model, tok, question, n=5):
    outs = [generate(model, tok, question, temperature=1.0, max_new_tokens=30) for _ in range(n)]
    answers = [normalise(o) for o in outs]
    top = max(set(answers), key=answers.count)
    return top, answers.count(top) / n        # low agreement → likely invented; abstain or retrieve`,
      }),
    ),
  );
  run1(); run2();
}
