import { h } from "../lib/core.js";
import { slider, segmented, toggle, stats, predict, code, callout, section, prose, table } from "../lib/widgets.js";
import { xyChart } from "../lib/plot.js";
import { selfImprove } from "../lib/hidden.js";

export default function (root, { id }) {
  const p = { K: 4, leak: 0, pool: "fixed", hint: false, rounds: 40 };
  let pending = 0;
  const st = stats([{ label: "Test accuracy, start" }, { label: "After 40 rounds" }, { label: "Same, K = 1 and no tricks" }, { label: "Training data last round" }]);
  const chart = xyChart({ height: 290, xLabel: "round of self-training", yLabel: "test accuracy", xDomain: [0, 40], yDomain: [0, 1], fmtY: (v) => `${Math.round(v * 100)}%`, label: "Test accuracy over rounds of self-training" });

  function run() {
    const o = selfImprove(p);
    const base = selfImprove({ rounds: p.rounds, K: 1 });
    st.set(0, `${Math.round(o.acc[0] * 100)}%`);
    const end = o.acc.at(-1);
    st.set(1, `${Math.round(end * 100)}%`, end > o.acc[0] + 0.05 ? "good" : "bad");
    st.set(2, `${Math.round(base.acc.at(-1) * 100)}%`);
    st.set(3, `${o.dataSize.at(-1)} examples`);
    chart.update([
      { name: "K = 1, fixed pool, perfect checker", color: "var(--ink-3, #999)", points: base.acc.map((v, i) => [i, v]) },
      { name: "Your settings", color: "var(--c1)", points: o.acc.map((v, i) => [i, v]), dots: false },
      { name: "Share of the pool solved", color: "var(--c4)", points: o.solvedFrac.map((v, i) => [i + 1, v]) },
    ]);
  }
  const later = () => { cancelAnimationFrame(pending); pending = requestAnimationFrame(run); };

  root.append(
    section(null, prose(`<p>The recipe behind much of the recent progress in reasoning is almost embarrassingly simple. Let the model try each problem several times. Keep the attempts that a checker accepts. Train on them. Repeat. It has many names: expert iteration, STaR (Zelikman et al., 2022), ReST, rejection-sampling fine-tuning, and in its on-policy form it's close to what RL with verifiable rewards does.</p>
<p>What decides whether it works isn't in the recipe. It's three things around it: <b>how many tries</b> the model gets (the data comes from pass@K, not pass@1), <b>how good the checker is</b>, and <b>where the next problems come from</b>. This simulation makes all three explicit. A model has a skill level; problems have difficulties; the model learns most from problems just past its edge.</p>`)),
    section("Let a model teach itself",
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Tries per problem (K)", min: 1, max: 64, step: 1, value: p.K, fmt: String, onInput: (v) => { p.K = v; later(); } }),
          slider({ label: "Checker lets wrong answers through", min: 0, max: 0.15, step: 0.005, value: p.leak, fmt: (v) => `${(v * 100).toFixed(1)}%`, onInput: (v) => { p.leak = v; later(); } }),
          segmented({ label: "Where problems come from", options: [{ value: "fixed", label: "A fixed set (difficulty 0–10)" }, { value: "frontier", label: "A generator at the model's edge" }], value: p.pool, onChange: (v) => { p.pool = v; run(); } }),
          toggle({ label: "Hints for failed problems", hint: "STaR's rationalisation: show the answer, ask for a derivation", checked: p.hint, onChange: (v) => { p.hint = v; run(); } }),
        ),
        h("div.stage-body", st.el, chart.el),
      ),
      callout("tip", "Things to try", "<p>Start at K = 1: the model creeps up, because it learns mostly from problems it already solves. Raise K to 16: the data now includes problems it solves 1 time in 16, which are exactly the ones that teach, and it climbs much faster. Then watch the orange line: with a fixed set of problems, it runs out of material at the set's hardest problem and stalls. Switch to a generator: the ceiling disappears (up to the hardest problem the generator can write and check). Finally, let 10% of wrong answers through: the model trains on its own errors and gets <i>worse</i> every round.</p>"),
      predict({
        chapter: id,
        question: "A model solves a problem 1 time in 50. Under expert iteration with K = 4 tries per problem, how much training signal does that problem give?",
        options: ["A lot: hard problems are the most informative", "Almost none: it's rarely solved, so it rarely enters the training data", "Exactly as much as an easy problem", "Negative: it confuses the model"],
        answer: 1,
        explain: "<p>With 4 tries, the chance of at least one success is about 8%, so the problem almost never contributes. This is the central limit of self-improvement: you can only learn from what you can sometimes do. More tries per problem, hints, and problems pitched just past the model's edge all exist to move hard problems into the ‘sometimes’ zone.</p>",
      }),
    ),
    section("What decides how far it goes",
      table(["Lever", "What it does", "Status"], [
        ["More tries per problem", "Turns rare successes into training data; returns diminish once nearly every reachable problem is solved", "Published (pass@k and RL sample-size studies)"],
        ["A better checker", "Wrong answers that slip through are trained on as if they were right; a few percent can stall or reverse progress", "Published for reward hacking; the threshold depends on the task"],
        ["Fresh problems at the edge", "A fixed problem set is a ceiling; a generator that writes and checks harder problems removes it", "Partly published (self-play, curriculum RL, generated environments in 2025)"],
        ["Hints (rationalisation)", "Lets the model learn from problems it can't yet solve alone, at the price of off-policy data", "Published (STaR)"],
        ["Majority vote as the checker", "Works when the model is usually right; reinforces confident mistakes when it isn't", "Published (test-time RL, 2025); fragile on hard problems"],
      ]),
      callout("note", "My view (~75% confident)", "<p>The scarce resource in self-improvement isn't compute or model size. It's <b>checkable problems just past the model's edge</b>. That's why so much effort since 2024 has gone into building environments: code with tests, math with verifiable answers, tool tasks with checkable end states. My guess about the next phase is that the best generators will be models themselves, writing problems and checks for each other, and the real limit will be the generator's ability to write problems that are both hard and correctly checked. When the checker is wrong in a consistent way, the whole loop learns that mistake, which is a harder failure to spot than plain noise.</p>"),
    ),
    section("Hypotheses I hold about self-improvement",
      table(["Hypothesis", "Why I think so", "Confidence"], [
        ["Self-training gains are bounded by pass@K of the starting model at the K you use, on the problems you have", "You can't select what you never sample; everything after that is sharpening", "70%"],
        ["A checker's false-accept rate matters about 10× more than its false-reject rate", "Rejected right answers only slow learning; accepted wrong ones teach errors and compound. Tested at toy scale (notebook 58): at a 5% error rate, false accepts did ~37× the damage", "65% → held"],
        ["Training on your own correct answers is less effective than RL at the same compute once the model is strong, but nearly as good when it's weak", "Early, the bottleneck is finding successes; late, it's which successes to reinforce, and RL's advantages carry that information", "50%"],
        ["Mixing a little ground-truth human data into every round prevents most drift", "Like fresh data in model collapse: an anchor the loop can't move", "60%"],
      ]),
    ),
    section("In code",
      code({
        "Expert iteration (Hugging Face)": `def expert_iteration(model, tok, problems, check, rounds=5, K=8):
    for r in range(rounds):
        kept = []
        for prob in problems:
            outs = sample(model, tok, prob.prompt, n=K, temperature=1.0)
            good = [o for o in outs if check(prob, o)]          # the verifier decides
            if good: kept.append((prob.prompt, random.choice(good)))
        print(f"round {r}: kept {len(kept)}/{len(problems)} ({len(kept)/len(problems):.0%})")
        model = sft(model, tok, kept, epochs=1, lr=1e-5)        # train on your own successes
    return model`,
        "STaR rationalisation": `def rationalise(model, tok, prob, K=4):
    hint = f"{prob.prompt}\\n(The correct answer is {prob.answer}. Explain step by step how to reach it.)"
    for out in sample(model, tok, hint, n=K):
        if check(prob, out):
            return (prob.prompt, out)        # train WITHOUT the hint in the prompt
    return None`,
        "Measure the frontier": `# which problems can teach? those solved sometimes, not always, not never
rates = {p.id: np.mean([check(p, o) for o in sample(model, tok, p.prompt, n=32)]) for p in problems}
frontier = [p for p in problems if 0 < rates[p.id] < 0.5]
print(f"{len(frontier)} problems on the frontier; {sum(r == 0 for r in rates.values())} never solved")`,
      }),
    ),
  );
  run();
}
