import { h } from "../lib/core.js";
import { slider, stats, predict, code, callout, section, prose, table } from "../lib/widgets.js";
import { xyChart } from "../lib/plot.js";
import { thinkCurve } from "../lib/hidden.js";

const BUDGETS = [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048];

export default function (root, { id }) {
  const p = { eps: 0.03, check: 0.3, rot: 400, flip: 0.004, meanSteps: 30 };
  let pending = 0;
  const st = stats([{ label: "Best budget if forced to keep thinking" }, { label: "Accuracy there" }, { label: "Forced, at 2,048 steps" }, { label: "Stops when done, at 2,048" }]);
  const chart = xyChart({ height: 290, xLog: true, xLabel: "thinking budget (reasoning steps)", yLabel: "accuracy", xDomain: [8, 2048], yDomain: [0, 1], fmtY: (v) => `${Math.round(v * 100)}%`, label: "Accuracy against thinking budget" });

  function run() {
    const forced = thinkCurve({ ...p, budgets: BUDGETS, trials: 1200 });
    const stop = thinkCurve({ ...p, budgets: BUDGETS, trials: 1200, stopWhenDone: true });
    const bi = forced.indexOf(Math.max(...forced));
    st.set(0, `${BUDGETS[bi]} steps`);
    st.set(1, `${Math.round(forced[bi] * 100)}%`);
    const last = forced.at(-1);
    st.set(2, `${Math.round(last * 100)}%`, last < forced[bi] - 0.05 ? "bad" : "good");
    st.set(3, `${Math.round(stop.at(-1) * 100)}%`, "good");
    chart.update([
      { name: "Stops when it thinks it's done", color: "var(--c3)", points: BUDGETS.map((b, i) => [b, stop[i]]), dots: true, r: 3.5 },
      { name: "Forced to use the whole budget (“Wait…”)", color: "var(--c2)", points: BUDGETS.map((b, i) => [b, forced[i]]), dots: true, r: 3.5 },
    ], { marks: [{ x: BUDGETS[bi], label: "forced peak" }] });
  }
  const later = () => { cancelAnimationFrame(pending); pending = requestAnimationFrame(run); };
  const sl = (key, label, min, max, step, fmt) => slider({ label, min, max, step, value: p[key], fmt, onInput: (v) => { p[key] = v; later(); } });

  root.append(
    section(null, prose(`<p>Reasoning models trade tokens for accuracy: let them think longer and they solve more. Up to a point. People who run these models see the same curve again and again: accuracy climbs, flattens, and then, if the model is pushed to keep going, <b>falls</b>. The model talks itself out of right answers, loses track of early steps in a long context, or wanders. Papers have started naming it (“overthinking”, “inverse scaling in test-time compute”, 2025), but the mechanism is rarely spelled out.</p>
<p>Here is a mechanism you can play with. A problem needs a number of serial steps. Each step can go wrong; a later re-check can catch the slip. Steps late in a long trace are a bit sloppier (context rot). And a model that keeps thinking after it has an answer sometimes second-guesses a correct one.</p>`)),
    section("Find the sweet spot",
      h("div.stage",
        h("div.stage-controls",
          sl("eps", "Chance a step goes wrong", 0.005, 0.08, 0.005, (v) => `${(v * 100).toFixed(1)}%`),
          sl("check", "Chance a slip gets caught", 0, 0.9, 0.05, (v) => `${Math.round(v * 100)}%`),
          sl("rot", "Context rot (steps until errors double)", 100, 4000, 100, (v) => String(v)),
          sl("flip", "Second-guessing per extra step", 0, 0.02, 0.001, (v) => `${(v * 100).toFixed(1)}%`),
          sl("meanSteps", "Average steps a problem needs", 5, 120, 5, (v) => String(v)),
        ),
        h("div.stage-body", st.el, chart.el),
      ),
      callout("tip", "Things to try", "<p>With the defaults, forcing the model to keep thinking peaks around 50 steps and then loses almost half its accuracy by 2,000. The green curve, a model that stops when it's done, never falls. Raise the slip-catching rate: the peak rises, which is what better self-verification buys (it barely moves right, which surprised me; see below). Set second-guessing to zero: the forced curve stops falling, so the decline comes from second-guessing, not length. Make problems harder: the sweet spot moves right, which is why adaptive thinking (choose the budget per problem) beats any fixed budget.</p>"),
      predict({
        chapter: id,
        question: "s1 (Muennighoff et al., 2025) forced a model to keep thinking by appending “Wait” whenever it tried to stop. What happened as they forced more and more thinking?",
        options: ["Accuracy rose without limit", "It rose, then flattened and eventually got worse", "It made no difference", "It always hurt"],
        answer: 1,
        explain: "<p>A few “Wait”s helped (the model re-checked and fixed mistakes). Past a point it flattened, and with enough forcing the model looped or talked itself into new errors. The same shape shows up in later studies of very long reasoning. Budget forcing is a useful knob, not a free lunch.</p>",
      }),
    ),
    section("What people who run reasoning models learn",
      table(["Observation", "Status", "Practical consequence"], [
        ["Accuracy vs thinking tokens is roughly log-linear, then flattens", "Published widely (o1, R1, s1 plots)", "Doubling the budget buys a roughly constant gain, until it doesn't"],
        ["Easy questions get overthought: huge traces for 2 + 3", "Published (“Do NOT think that much for 2+3=?”, 2024)", "Use adaptive or effort-controlled thinking rather than one fixed budget"],
        ["Longer traces on the same question are more often wrong", "Widely observed", "Length is a cheap uncertainty signal; long traces deserve a second sample or a verifier"],
        ["Parallel samples plus voting often beat one very long trace for the same tokens", "Published for math; task-dependent", "Spend tokens on breadth (several shorter traces) once depth saturates"],
        ["Traces aren't always faithful: the stated reasoning can differ from what drove the answer", "Published (Anthropic, 2025; Turpin et al., 2023)", "Don't treat the chain of thought as an audit log"],
        ["A good verifier moves the whole curve up", "Published (PRMs, generative verifiers)", "Invest in checking before in longer generation"],
      ]),
      callout("note", "My intuition (~60% confident)", "<p>The optimal fixed budget scales roughly as d / (1 − c): problem depth over the miss rate of self-checking, until context rot dominates. Two predictions follow. Models trained with better self-verification will show <i>later</i> peaks, not just higher ones. And the decline under forced thinking is driven mostly by second-guessing, so it should shrink sharply if the model is trained to state and keep a committed answer before extending. Both are measurable with budget forcing on an open reasoning model (notebook 45).</p><p><b>Tested in the model (notebook 45): the formula failed.</b> The best budget grew with problem depth (about 41 → 57 → 100 steps as depth doubled twice) but hardly changed with the check rate, so d / (1 − c) was off by up to 4.5×. Better checking raises the peak more than it delays it. The second prediction, about second-guessing driving the decline, holds in the model by construction and still needs a real-model test.</p>"),
    ),
    section("In code",
      code({
        "Budget forcing (Hugging Face)": `def think_with_budget(model, tok, question, budget, wait="Wait"):
    """Generate a reasoning trace of about \`budget\` tokens: cut it if longer,
    append 'Wait' and continue if the model tries to stop early (s1-style)."""
    prompt = tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)
    text, used = prompt + "<think>\\n", 0
    while used < budget:
        ids = tok(text, return_tensors="pt").to(model.device)
        out = model.generate(**ids, max_new_tokens=budget - used, stop_strings=["</think>"], tokenizer=tok, do_sample=False)
        new = tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=False)
        used += out.shape[1] - ids.input_ids.shape[1]
        text += new.replace("</think>", "")
        if used < budget: text += f"\\n{wait}, "          # force more thinking
    text += "\\n</think>\\nFinal answer:"
    ids = tok(text, return_tensors="pt").to(model.device)
    return tok.decode(model.generate(**ids, max_new_tokens=30, do_sample=False)[0, ids.input_ids.shape[1]:])`,
        "Effort control with Claude": `import anthropic
client = anthropic.Anthropic()

# adaptive thinking: the model chooses how long to think for each request
resp = client.messages.create(
    model="claude-opus-5-5", max_tokens=16000,
    thinking={"type": "adaptive"},
    messages=[{"role": "user", "content": question}],
)
thinking_tokens = sum(len(b.thinking) for b in resp.content if b.type == "thinking")`,
        "Length as an uncertainty signal": `# sample n traces; if the answers disagree or traces are unusually long, spend more
answers, lengths = zip(*[solve(q) for _ in range(4)])
top, votes = Counter(answers).most_common(1)[0]
if votes < 3 or np.median(lengths) > 2 * typical_length[q.kind]:
    answers += tuple(solve(q)[0] for _ in range(8))          # escalate: more samples
    top = Counter(answers).most_common(1)[0][0]`,
      }),
    ),
  );
  run();
}
