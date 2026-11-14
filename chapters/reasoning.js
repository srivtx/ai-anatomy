import { h, rng, loop } from "../lib/core.js";
import { slider, stats, predict, code, callout, section, prose, table, button, stepper } from "../lib/widgets.js";
import { xyChart } from "../lib/plot.js";
import { MLP, bceLogit, gauss, sigmoid } from "../lib/nn.js";

const NB = 8;
const par = (b) => b.reduce((a, x) => a ^ x, 0);
const pm = (b) => b.map((x) => x * 2 - 1);

export default function (root, { id }) {
  // ---------- 1. parity: answer directly vs think step by step ----------
  let nTrain = 64, direct, step, tr, te, ep, rr;
  const all = Array.from({ length: 2 ** NB }, (_, i) => Array.from({ length: NB }, (_, k) => (i >> k) & 1));
  const pst = stats([{ label: "Direct: training strings" }, { label: "Direct: unseen strings" }, { label: "Step by step: unseen strings" }, { label: "Step by step: 32-bit strings" }]);
  const trace = h("div.big-out", { style: { fontFamily: "var(--mono)", fontSize: "14px", lineHeight: 1.7 } });
  const long = Array.from({ length: 200 }, (_, i) => { const r2 = rng(100 + i); return Array.from({ length: 32 }, () => (r2() < 0.5 ? 1 : 0)); });
  function initPar() {
    rr = rng(3);
    const sh = all.map((b) => [rr(), b]).sort((a, b) => a[0] - b[0]).map((x) => x[1]);
    tr = sh.slice(0, nTrain); te = sh.slice(nTrain);
    direct = new MLP([NB, 64, 1], { act: "tanh", seed: 2 });
    step = new MLP([2, 8, 1], { act: "tanh", seed: 2 });
    ep = 0; drawPar();
  }
  const dirAcc = (set) => set.filter((b) => (direct.forward(pm(b))[0] > 0) === !!par(b)).length / set.length;
  const runCot = (b) => { let p = 0; const ps = []; for (const x of b) { p = step.forward(pm([p, x]))[0] > 0 ? 1 : 0; ps.push(p); } return ps; };
  const cotAcc = (set) => set.filter((b) => runCot(b).at(-1) === par(b)).length / set.length;
  function epoch() {
    for (let i = 0; i < tr.length; i++) { const b = tr[Math.floor(rr() * tr.length)], z = direct.forward(pm(b))[0]; direct.backward([bceLogit(z, par(b)).d]); if (i % 16 === 15) direct.step(0.003, 16); }
    // the step-by-step model trains on the worked traces of the same strings: (running parity, next bit) -> new running parity
    for (let i = 0; i < tr.length; i++) { const b = tr[Math.floor(rr() * tr.length)]; let p = 0; for (const x of b) { const t = p ^ x, z = step.forward(pm([p, x]))[0]; step.backward([bceLogit(z, t).d]); p = t; } step.step(0.01, NB); }
    ep++;
  }
  function drawPar() {
    if (!ep) { pst.set(0, "–"); pst.set(1, "–"); pst.set(2, "–"); pst.set(3, "–"); }
    else {
      const a = dirAcc(tr), b = dirAcc(te), c = cotAcc(te), d = cotAcc(long);
      pst.set(0, `${Math.round(a * 100)}%`, a > 0.95 ? "good" : ""); pst.set(1, `${Math.round(b * 100)}%`, b < 0.7 ? "bad" : b < 0.9 ? "warn" : "good");
      pst.set(2, `${Math.round(c * 100)}%`, c > 0.95 ? "good" : ""); pst.set(3, `${Math.round(d * 100)}%`, d > 0.95 ? "good" : "");
    }
    const ex = te[0] || all[5], ps = runCot(ex), dz = direct.forward(pm(ex))[0] > 0 ? 1 : 0;
    trace.innerHTML = `<div>Input bits: <b>${ex.join(" ")}</b> &nbsp;·&nbsp; true parity: <b>${par(ex)}</b></div><div>Direct answer: <b style="color:${dz === par(ex) ? "var(--good)" : "var(--bad)"}">${ep ? dz : "?"}</b></div><div>Thinking: ${ep ? ps.map((p, i) => `<span title="after bit ${i + 1}">${p}</span>`).join(" → ") : "…"} &nbsp;·&nbsp; answer <b style="color:${ps.at(-1) === par(ex) ? "var(--good)" : "var(--bad)"}">${ep ? ps.at(-1) : "?"}</b></div>`;
  }
  const lp = loop(() => { for (let k = 0; k < 8; k++) epoch(); drawPar(); if (ep >= 240) lp.stop(); }, trace);

  // ---------- 2. test-time scaling: majority vote, verifiers, pass@k ----------
  const o = { p: 0.35, distract: 0.5, dprime: 1.5 };
  const tc = xyChart({ height: 300, xLog: true, xLabel: "samples per question (N)", yLabel: "accuracy", yDomain: [0, 1] });
  const tst = stats([{ label: "One sample" }, { label: "Majority of 64" }, { label: "Best of 64 by verifier" }, { label: "pass@64 (perfect checker)" }]);
  function drawTTS() {
    const r = rng(11), Q = 300, Ns = [1, 2, 4, 8, 16, 32, 64, 128, 256];
    const lg = Math.log(o.p / (1 - o.p));
    const qs = Array.from({ length: Q }, () => sigmoid(lg + gauss(r) * 1.5));
    const res = { maj: [], best: [], pass: [] };
    for (const N of Ns) {
      let maj = 0, best = 0, pass = 0;
      for (const pq of qs) {
        pass += 1 - (1 - pq) ** N;
        const counts = new Map(); let bestScore = -Infinity, bestOk = false;
        for (let s = 0; s < N; s++) {
          const ok = r() < pq;
          // wrong answers: one popular wrong answer (a common misconception) and a long tail of scattered ones
          const ans = ok ? 0 : r() < o.distract ? 1 : 2 + Math.floor(r() * 50);
          counts.set(ans, (counts.get(ans) || 0) + 1);
          const score = (ok ? o.dprime : 0) + gauss(r);
          if (score > bestScore) { bestScore = score; bestOk = ok; }
        }
        let top = -1, topC = -1; for (const [a, c] of counts) if (c > topC || (c === topC && r() < 0.5)) { top = a; topC = c; }
        maj += top === 0; best += bestOk;
      }
      res.maj.push([N, maj / Q]); res.best.push([N, best / Q]); res.pass.push([N, pass / Q]);
    }
    tc.update([
      { name: "pass@N: any sample right (needs a perfect checker)", color: "var(--ink-3)", points: res.pass, dash: true },
      { name: "Best-of-N by a learned verifier", color: "var(--accent)", points: res.best, dots: true, r: 3.5 },
      { name: "Majority vote (self-consistency)", color: "var(--c2)", points: res.maj, dots: true, r: 3.5 },
    ], { yDomain: [0, 1] });
    tst.set(0, `${Math.round(res.maj[0][1] * 100)}%`); tst.set(1, `${Math.round(res.maj[6][1] * 100)}%`); tst.set(2, `${Math.round(res.best[6][1] * 100)}%`); tst.set(3, `${Math.round(res.pass[6][1] * 100)}%`);
  }

  root.append(
    section(null, prose(`<p>Psychologists describe two modes of thought: fast, automatic <b>System 1</b> (recognising a face) and slow, deliberate <b>System 2</b> (multiplying 17 × 24 in your head). A transformer answering in one token is pure System 1: a fixed amount of computation, however hard the question. <b>Chain of thought</b> (Wei et al., 2022) lets it write intermediate steps first. Every token it writes is another full forward pass, and every written step becomes input it can build on. Reasoning models (OpenAI o1 in 2024, DeepSeek-R1, Claude with extended thinking) are trained with RL to use this well, and they get better the longer they're allowed to think.</p>`)),
    section("Why writing steps helps: a live experiment",
      prose(`<p>The task: is the number of 1s in an ${NB}-bit string odd or even? Two networks learn from the <b>same</b> training strings. The <b>direct</b> network sees the whole string and must output the answer at once. The <b>step-by-step</b> network is trained on worked solutions: it only ever learns one step, “running parity so far, next bit → new running parity”, and at test time it writes out its steps, feeding each answer back in.</p>`),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Training strings (of 256)", min: 16, max: 224, step: 16, value: nTrain, onInput: (v) => { nTrain = v; lp.stop(); initPar(); } }),
          button("Train both", () => { if (ep >= 240) initPar(); lp.start(); }, "primary"), button("Reset", () => { lp.stop(); initPar(); })),
        h("div.stage-body", pst.el, trace)),
      callout("tip", "What happens", "<ul style='margin:0;padding-left:18px'><li>The direct network memorises its training strings (100%) but is barely better than a coin flip on new ones unless it sees most of the 256. Parity of the whole string is a hard function to learn in one shot.</li><li>The step-by-step network learns a tiny function (XOR) that's easy to get right, then applies it as many times as needed. It's perfect on unseen 8-bit strings and on 32-bit strings it never trained on, because the procedure doesn't care about length.</li><li>That's the core of why chain of thought helps: it decomposes a problem that needs many sequential steps into steps that each fit in one forward pass. Theory agrees: a fixed-depth transformer can only do limited serial computation per token, and a chain of thought with enough steps raises that limit substantially (Merrill & Sabharwal, 2023; Li et al., 2024).</li></ul>"),
      predict({
        chapter: id,
        question: "A model gets a hard maths problem right more often when allowed to “think” for 10,000 tokens than for 1,000. Where does the extra ability come from?",
        options: ["The model gets more parameters while thinking", "More forward passes (more total compute), and the ability to write intermediate results, try approaches and check them", "The extra tokens are ignored; it's luck", "Longer outputs are graded more generously"],
        answer: 1,
        explain: "<p>Test-time compute is a second axis of scaling, alongside training compute. OpenAI reported o1's accuracy rising roughly log-linearly with thinking tokens. Snell et al. (2024) showed that, for some problems, spending extra compute at test time beats spending it on a bigger model.</p>",
      }),
    ),
    section("Parallel thinking: sample many, pick one",
      prose("<p>The other way to spend test-time compute is breadth: sample N independent solutions, then choose. <b>Majority vote</b> (self-consistency, Wang et al., 2022) needs no extra model. A <b>verifier</b> (a reward model, or a model that checks steps) scores each one and you keep the best. With a perfect checker (unit tests, a proof checker), you only need one of N to be right: <b>pass@N</b>. A simulation over 300 questions of varying difficulty:</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Typical single-sample accuracy", min: 0.05, max: 0.8, step: 0.05, value: o.p, fmt: (v) => `${Math.round(v * 100)}%`, onInput: (v) => { o.p = v; drawTTS(); } }),
          slider({ label: "How often wrong answers agree", min: 0, max: 1, step: 0.05, value: o.distract, fmt: (v) => `${Math.round(v * 100)}%`, onInput: (v) => { o.distract = v; drawTTS(); } }),
          slider({ label: "Verifier quality (separation d′)", min: 0, max: 4, step: 0.1, value: o.dprime, fmt: (v) => v.toFixed(1), onInput: (v) => { o.dprime = v; drawTTS(); } })),
        h("div.stage-body", tst.el, tc.el)),
      callout("tip", "Things to try", "<ul style='margin:0;padding-left:18px'><li>Raise “wrong answers agree” to 100%: there's a popular wrong answer, and on hard questions majority vote confidently picks it. More samples make it <i>more</i> sure of the wrong answer.</li><li>Set the verifier quality to 0 (a useless judge): best-of-N is no better than one sample. At 4 it approaches the perfect-checker curve. In practice, imperfect verifiers also get exploited as N grows: with enough samples you find the wrong answers the verifier likes best (Goodhart again).</li><li>The gap between pass@N and the rest is the prize for building better verifiers. That's why maths and code, where answers can be checked automatically, moved fastest.</li></ul>"),
    ),
    section("How reasoning models are built",
      stepper([
        { title: "1. Prompting", html: "<p>“Let's think step by step” (Kojima et al., 2022) or a few worked examples. Works on large enough base models, for free.</p>" },
        { title: "2. Supervised traces", html: "<p>Fine-tune on worked solutions, including ones the model generated itself and that were checked correct (STaR, rejection sampling).</p>" },
        { title: "3. RL on verifiable rewards", html: "<p>Sample many attempts at problems with checkable answers; reinforce the ones that are right (GRPO, PPO). The model discovers useful habits on its own: checking work, backtracking (“wait, that's wrong”), trying another approach. Thinking gets longer as training goes on.</p>" },
        { title: "4. Distil", html: "<p>The long, high-quality traces from a big reasoning model become SFT data for smaller models. DeepSeek-R1's distilled Qwen and Llama models gained most of the reasoning ability this way.</p>" },
        { title: "5. Budget at inference", html: "<p>Let users or the system set how much to think: a token budget or an effort level. Easy questions shouldn't burn 10,000 tokens (“overthinking” is a real failure mode: wasted cost, and sometimes talking itself out of a right answer).</p>" },
      ]),
      table(["Technique", "Spends compute on", "Needs"], [
        ["Chain of thought", "Depth: one long sequential solution", "A model trained or prompted to reason"],
        ["Self-consistency", "Breadth: many solutions, vote", "Answers that can be compared"],
        ["Best-of-N + verifier", "Breadth, then judgement", "A reward model or checker"],
        ["Tree search with a process reward model", "Partial solutions, pruned step by step", "A model that scores steps (Lightman et al., 2023)"],
        ["Self-refinement", "Critique and revise", "A model that can spot its own errors (often the weak link)"],
      ]),
      callout("warn", "Is the chain of thought the real reasoning?", "<p>Not necessarily. Studies have found that models sometimes reach an answer for reasons they don't mention in their chain of thought, for example using a hint planted in the prompt without acknowledging it (Turpin et al., 2023; Anthropic, 2025). Visible reasoning is still a useful window for monitoring a model, but it's not a guaranteed transcript of the computation. Training pressure directly on the chain of thought's content can make it less honest, which is why many labs avoid it.</p>"),
    ),
    section("In code",
      code({
        "Self-consistency": `from collections import Counter
def self_consistency(model, question, n=16, temperature=0.8):
    answers = [extract_final_answer(model.generate(question + "\\nThink step by step.", temperature=temperature))
               for _ in range(n)]
    return Counter(answers).most_common(1)[0][0]`,
        "Best-of-N": `def best_of_n(model, verifier, question, n=16):
    candidates = [model.generate(question, temperature=1.0) for _ in range(n)]
    return max(candidates, key=lambda c: verifier.score(question, c))`,
      }, "Project 11 trains a tiny model with GRPO until it starts writing useful intermediate steps."),
    ),
  );
  initPar(); drawTTS();
}
