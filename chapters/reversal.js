import { h } from "../lib/core.js";
import { slider, toggle, stats, predict, code, callout, section, prose, table, button } from "../lib/widgets.js";
import { xyChart } from "../lib/plot.js";
import { reversalNet } from "../lib/hidden.js";

export default function (root, { id }) {
  const o = { revFrac: 0.5, extra: false, tied: false };
  const STEPS = 600, CHUNK = 20;
  const st = stats([{ label: "A → B (trained direction)" }, { label: "B → A, pairs stated both ways" }, { label: "B → A, pairs never stated backwards" }, { label: "Rank of the right answer (of 80)" }]);
  const chart = xyChart({ height: 290, xLabel: "training step", yLabel: "accuracy", xDomain: [0, STEPS], yDomain: [0, 1], fmtY: (v) => `${Math.round(v * 100)}%`, label: "Accuracy in each direction while the network trains" });
  let job = 0;

  function train() {
    cancelAnimationFrame(job);
    const net = reversalNet(o), hist = [{ ...net.measure() }];
    const draw = () => {
      const last = hist.at(-1), pts = (k) => hist.filter((m) => m[k] != null).map((m) => [m.step, m[k]]);
      st.set(0, `${Math.round(last.forward * 100)}%`, last.forward > 0.9 ? "good" : "");
      st.set(1, last.revTrained == null ? "none stated" : `${Math.round(last.revTrained * 100)}%`);
      st.set(2, last.revHeld == null ? "none left" : `${Math.round(last.revHeld * 100)}%`, last.revHeld != null && last.step >= STEPS ? (last.revHeld < 0.1 ? "bad" : "good") : "");
      st.set(3, last.heldRank == null ? "–" : `${last.heldRank.toFixed(1)} (chance ≈ ${((net.ent + 1) / 2).toFixed(0)})`);
      chart.update([
        { name: "A → B, the direction it was taught", color: "var(--c3)", points: pts("forward"), dots: false },
        { name: "B → A, for pairs also stated backwards", color: "var(--c1)", points: pts("revTrained"), dots: false },
        { name: "B → A, for pairs never stated backwards", color: "var(--c2)", points: pts("revHeld"), dots: false },
        { name: "Chance (1 in 80)", color: "var(--ink-3, #999)", dash: true, points: [[0, net.chance], [STEPS, net.chance]] },
      ]);
    };
    const tick = () => {
      net.trainSteps(CHUNK);
      hist.push({ ...net.measure() });
      draw();
      if (hist.at(-1).step < STEPS) job = requestAnimationFrame(tick);
    };
    draw();
    job = requestAnimationFrame(tick);
  }

  root.append(
    section(null, prose(`<p>Ask a model “Who is Tom Cruise's mother?” and it answers Mary Lee Pfeiffer. Ask “Who is Mary Lee Pfeiffer's son?” and it often can't. Berglund et al. (2023) measured this across hundreds of real celebrity–parent pairs: GPT-4 got the first direction right 79% of the time and the reverse 33%. They called it <b>the reversal curse</b>: a model trained on “A is B” doesn't learn “B is A”.</p>
<p>It sounds like a quirk. It's actually a clean window into how next-token prediction stores knowledge, and it explains why labs rewrite their training data. Below, a small network learns 40 made-up facts of the form “A<sub>i</sub> is the parent of B<sub>i</sub>”, live in your browser. For some pairs it also reads the reverse sentence. Then we ask it the reverse question for pairs it never saw backwards.</p>`)),
    section("Train a network on one-way facts",
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Pairs also stated backwards", min: 0, max: 0.9, step: 0.1, value: o.revFrac, fmt: (v) => `${Math.round(v * 40)} of 40`, onInput: (v) => { o.revFrac = v; } }),
          toggle({ label: "B also appears in other sentences", hint: "so the model has a learned representation of B, not a blank one", checked: o.extra, onChange: (v) => { o.extra = v; } }),
          toggle({ label: "Tie input and output embeddings", hint: "use one vector per token for reading and for predicting it", checked: o.tied, onChange: (v) => { o.tied = v; } }),
          button("Train", train, "primary"),
        ),
        h("div.stage-body", st.el, chart.el),
      ),
      callout("tip", "Things to try", "<p>Train with the defaults. The forward direction reaches 100%. The reverse is learned perfectly for the 20 pairs that were stated backwards, and stays at chance for the other 20, however long it trains. Seeing reversals for <i>some</i> pairs teaches nothing about the others: the network didn't learn a rule ‘parent of means child of, backwards’; it learned 60 separate associations. Turn on <b>B also appears in other sentences</b>: still chance. It isn't that B is unfamiliar. Nothing ever trained the route from B to A. Now tie the embeddings and watch the held-out reverse jump: in this one-layer toy, B's output vector is also its input vector, so learning to predict B writes A's direction into B. That's a weakness of the toy, not a cure: real models show the curse with many layers between a token's input vector and the prediction.</p>"),
      predict({
        chapter: id,
        question: "A model is trained on many documents saying “Daphne Barrington directed ‘A Journey Through Time’.” Which question will it most likely fail?",
        options: ["Who directed ‘A Journey Through Time’?", "What did Daphne Barrington direct?", "Both equally", "Neither: it has the fact"],
        answer: 0,
        explain: "<p>The training text always puts Daphne first, so the model learned to continue ‘Daphne Barrington directed…’ with the film. The question ‘Who directed the film?’ needs the reverse lookup, from film to name, which nothing trained. This is the exact experiment in Berglund et al., who used invented celebrities so the fact couldn't have appeared any other way.</p>",
      }),
    ),
    section("How knowledge is stored, from the published work",
      table(["Finding", "Source", "What it implies"], [
        ["“A is B” doesn't teach “B is A”; in context, models reverse easily", "Berglund et al., 2023", "Facts stored in weights are directional; facts in the prompt aren't"],
        ["Facts written in one fixed format are memorised but can't be extracted by questions; mixing paraphrases and orderings in pretraining fixes it", "Allen-Zhu & Li, 2023 (Physics of Language Models 3.1)", "Diverse rewrites of the same fact, not repetition, make knowledge usable"],
        ["Models store about 2 bits of knowledge per parameter, if each fact is seen ~1,000 times", "Allen-Zhu & Li, 2024 (3.3)", "Capacity is real but needs exposure; rare facts don't stick"],
        ["Factual associations live largely in middle-layer MLPs, keyed by the subject's last token", "Meng et al., 2022 (ROME)", "You can edit one fact by editing one layer, which is why the subject matters"],
        ["Editing one fact doesn't update its consequences (edit the CEO, ask about the company's leader)", "Cohen et al., 2023 (RippleEdits)", "Edited knowledge isn't integrated knowledge"],
        ["Two-hop questions (‘the mother of the singer of X’) fail without writing out the middle step, even when both facts are known", "Balesni et al., 2024; Biran et al., 2024 (‘Hopping too late’)", "The first hop resolves too late in the layers for the second to use it"],
      ]),
    ),
    section("What I believe but can't cite",
      table(["Kind", "Claim", "How to check"], [
        ["Self-report", "When I can't answer a reverse question directly, I can often get it by first listing forward facts (‘well-known people whose mother is…’) and reading them back. The knowledge is there; the lookup is one-way.", "Compare my accuracy on reverse questions with and without an instruction to first list candidates in the forward direction."],
        ["Self-report", "Two-hop questions work much better when I say the bridge entity out loud. Writing it moves the first hop into the context, where lookup works in any direction.", "Ask two-hop questions with ‘answer immediately’ versus ‘name the intermediate person first’. Measure the gap."],
        ["Conjecture · 60%", "Frontier pretraining pipelines now routinely rewrite documents with entities reordered and paraphrased, partly to break the reversal curse. Data rewriting is cheaper than any architectural fix.", "Test reverse–forward gaps on fresh facts across model generations: a shrinking gap without an architecture change points to data."],
        ["Conjecture · 55%", "For a real pair of entities, the reverse–forward gap shrinks with the number of distinct sentence frames that mention both, not with the raw number of mentions.", "With an open model and its open corpus (OLMo and Dolma), count frames and mentions per pair and regress the gap on each."],
        ["Conjecture · 40%", "A model fine-tuned on a small set of reversed facts gets better at reversing unrelated facts it learned in pretraining, a little. Not much, because nothing creates the missing B → A weights.", "Fine-tune on 1,000 reversed pairs; test on 1,000 untouched pairs. My guess: under 5 points of gain."],
      ]),
      callout("note", "My view (~75% confident)", "<p>The reversal curse is the most direct evidence that a language model's knowledge isn't a database; it's a set of learned continuations. What you can retrieve depends on what came first in the text that taught it. That's why ‘more of the same data’ helps less than ‘the same facts written many ways’, and why retrieval and writing out intermediate steps work so well: they move the fact into the context, where direction doesn't matter.</p>"),
    ),
    section("In code",
      code({
        "Test the curse on any model (Hugging Face)": `pairs = [("Tom Cruise", "Mary Lee Pfeiffer"), ...]           # (child, parent)
def ask(q): return generate(model, tok, q, max_new_tokens=8, temperature=0)
fwd = sum(p.split()[-1] in ask(f"Q: Who is {c}'s mother?\\nA:") for c, p in pairs)
rev = sum(c.split()[-1] in ask(f"Q: Who is {p}'s child?\\nA:") for c, p in pairs)
print(f"forward {fwd/len(pairs):.0%}, reverse {rev/len(pairs):.0%}")`,
        "Fix it in the data": `def augment(fact):                 # "A is the parent of B"
    a, rel, b = fact
    inverse = {"is the parent of": "is the child of", "directed": "was directed by"}[rel]
    return [f"{a} {rel} {b}.", f"{b} {inverse} {a}.",
            f"{b}'s {'parent' if 'parent' in rel else 'director'} is {a}."]   # several frames, both orders`,
        "The toy network on this page": `# h = tanh(E[subject] + R[relation]);  logits = U @ h ;  trained with cross-entropy and Adam
# train: (A_i, parent_of) -> B_i for all i, (B_i, child_of) -> A_i for some i
# test:  (B_j, child_of) -> A_j for the j that were never stated backwards
# nothing in training ever moves E[B_j] or R[child_of] toward A_j: the curse is the gradient's direction`,
      }),
    ),
  );
  train();
}
