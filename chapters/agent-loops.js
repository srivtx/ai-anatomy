import { h, rng } from "../lib/core.js";
import { slider, stats, predict, code, callout, section, prose, table, button, stepper } from "../lib/widgets.js";
import { xyChart } from "../lib/plot.js";

export default function (root, { id }) {
  // ---------- 1. compounding reliability ----------
  const c = { p: 0.95, q: 0.8, k: 2 };
  const rc = xyChart({ height: 280, xLabel: "steps in the task", yLabel: "chance the whole task succeeds", xDomain: [1, 100], yDomain: [0, 1] });
  const rst = stats([{ label: "20 steps, no checking" }, { label: "20 steps, with verify-and-retry" }]);
  function drawRel() {
    // with a check that catches a failed step with probability q, and up to k retries
    const pEff = c.p + (1 - c.p) * c.q * (1 - (1 - c.p) ** c.k);
    const xs = Array.from({ length: 100 }, (_, i) => i + 1);
    rc.update([
      { name: `No checking: ${Math.round(c.p * 100)}% per step`, color: "var(--bad)", points: xs.map((n) => [n, c.p ** n]) },
      { name: `Verify and retry: ${(pEff * 100).toFixed(1)}% per step`, color: "var(--good)", points: xs.map((n) => [n, pEff ** n]) },
    ], { xDomain: [1, 100], yDomain: [0, 1], marks: [{ x: 20, label: "20 steps" }] });
    rst.set(0, `${Math.round(c.p ** 20 * 100)}%`, c.p ** 20 < 0.5 ? "bad" : ""); rst.set(1, `${Math.round(pEff ** 20 * 100)}%`, pEff ** 20 > 0.7 ? "good" : "");
  }

  // ---------- 2. one long session vs a Ralph loop ----------
  const s = { tasks: 16, p0: 0.85, rot: 0.6, verify: 0.85 };
  const runs = { session: null, ralph: null };
  const sessEl = h("div.agrid"), ralphEl = h("div.agrid");
  const sst = stats([{ label: "Long session: tasks done right" }, { label: "Ralph loop: tasks done right" }, { label: "Ralph loop: iterations used" }]);
  function simulate(seed) {
    const r = rng(seed);
    // A) one session: context fills with every attempt; success decays as context fills ("context rot");
    //    when the window is full, the session ends
    const sess = []; let fill = 0.08, done = 0;
    for (let t = 0; t < s.tasks && fill < 1; t++) {
      const p = s.p0 * (1 - s.rot * fill * fill), ok = r() < p, caught = !ok && r() < s.verify * (1 - fill * 0.5);
      if (ok) { sess.push("ok"); done++; } else if (caught) { sess.push("retry"); fill += 0.07; t--; } else { sess.push("bug"); }
      fill += 0.07;
      if (sess.length > 60) break;
    }
    while (sess.length < s.tasks && sess.filter((x) => x !== "retry").length < s.tasks) { sess.push("full"); if (sess.length > 80) break; }
    // B) Ralph loop: every iteration starts with a fresh context, reads the plan and progress file,
    //    does one task, runs the tests, records progress, exits
    const ral = []; let next = 0, it = 0, rdone = 0;
    while (next < s.tasks && it < 80) {
      it++;
      const p = s.p0 * (1 - s.rot * 0.12 * 0.12), ok = r() < p, caught = !ok && r() < s.verify;
      if (ok) { ral.push("ok"); next++; rdone++; } else if (caught) ral.push("retry"); else { ral.push("bug"); next++; }
    }
    return { sess, done, ral, rdone, it };
  }
  function drawLoop(seed = 5) {
    const { sess, done, ral, rdone, it } = simulate(seed);
    const cell = (x) => h("span.acell", { class: `acell ${x}`, title: { ok: "task done and tests pass", retry: "failure caught by tests: retry", bug: "failure not caught: a bug slips through", full: "context window full: session over" }[x] });
    sessEl.replaceChildren(...sess.map(cell)); ralphEl.replaceChildren(...ral.map(cell));
    sst.set(0, `${done} of ${s.tasks}`, done === s.tasks ? "good" : "bad"); sst.set(1, `${rdone} of ${s.tasks}`, rdone === s.tasks ? "good" : rdone > done ? "warn" : ""); sst.set(2, String(it));
  }

  root.append(
    section(null, prose(`<p>An <b>agent</b> is a model using tools in a loop until a task is done. The loop itself is simple. What makes agents work or fail is reliability: a 20-step task where every step is 95% reliable succeeds only about a third of the time. The patterns that work in practice all attack that number: check work automatically, retry on failure, keep the context clean, and save progress outside the model so a fresh start can pick up where the last one left off.</p>`)),
    section("Reliability compounds",
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Per-step success", min: 0.8, max: 0.999, step: 0.001, value: c.p, fmt: (v) => `${(v * 100).toFixed(1)}%`, onInput: (v) => { c.p = v; drawRel(); } }),
          slider({ label: "Checks catch a failed step", min: 0, max: 1, step: 0.05, value: c.q, fmt: (v) => `${Math.round(v * 100)}%`, onInput: (v) => { c.q = v; drawRel(); } }),
          slider({ label: "Retries allowed", min: 0, max: 5, value: c.k, onInput: (v) => { c.k = v; drawRel(); } })),
        h("div.stage-body", rst.el, rc.el)),
      callout("tip", "From experience", "<p>This curve is why coding agents got good before most other agents: code comes with cheap, reliable checks (compilers, linters, tests). If your agent's domain has no automatic check, building one (a validator, a test suite, a second model reviewing against a rubric) is usually the highest-leverage work you can do.</p>"),
    ),
    section("The Ralph loop: fresh context, progress in files",
      prose("<p>Long agent sessions degrade: the context fills with stale tool output and failed attempts, and the model gets worse at using it (“context rot”). The <b>Ralph loop</b>, popularised by Geoffrey Huntley in 2025 and named after the cheerfully persistent Simpsons character, does the opposite of a long session: run the agent in a plain <code>while</code> loop with the same prompt. Each iteration starts with an <b>empty context</b>, reads a plan and a progress file, does <b>one</b> task, runs the tests, writes down what it did, and exits. The files are the memory; the tests are the guard rails.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Tasks in the project", min: 6, max: 30, value: s.tasks, onInput: (v) => { s.tasks = v; drawLoop(); } }),
          slider({ label: "Base success per task", min: 0.5, max: 0.98, step: 0.01, value: s.p0, fmt: (v) => `${Math.round(v * 100)}%`, onInput: (v) => { s.p0 = v; drawLoop(); } }),
          slider({ label: "Context rot", min: 0, max: 1, step: 0.05, value: s.rot, fmt: (v) => v.toFixed(2), onInput: (v) => { s.rot = v; drawLoop(); } }),
          slider({ label: "Tests catch a bad change", min: 0, max: 1, step: 0.05, value: s.verify, fmt: (v) => `${Math.round(v * 100)}%`, onInput: (v) => { s.verify = v; drawLoop(); } }),
          button("Run again", () => drawLoop(Math.floor(Math.random() * 1e6)), "primary")),
        h("div.stage-body", sst.el,
          h("div.eyebrow", "One long session (context keeps filling)"), sessEl,
          h("div.eyebrow", { style: { marginTop: "12px" } }, "Ralph loop (one task per fresh context)"), ralphEl,
          h("p.canvas-cap", "Green: task done, tests pass. Orange: failure caught by tests, retried. Red: a bug slipped through. Grey: context window full, session over."))),
      callout("note", "A toy model with stated assumptions", "<p>Success per task falls with the square of how full the context is; each attempt adds 7% of the window; a fresh Ralph iteration starts 12% full (plan + progress file). The exact numbers are made up; the shape matches what practitioners report: long sessions drift and stall, short iterations with external memory and tests keep going. The cost is more model calls and some re-reading.</p>"),
      stepper([
        { title: "PROMPT.md", html: "<p>The fixed instructions every iteration gets: “Read PLAN.md and progress.md. Pick the most important unfinished item. Implement it. Run the tests. If they pass, commit and append a note to progress.md. Do only one item.”</p>" },
        { title: "PLAN.md", html: "<p>The task list (a spec broken into small, testable items), maintained by the agent and by you.</p>" },
        { title: "progress.md", html: "<p>What's done, what was tried, what failed and why. The next iteration's memory.</p>" },
        { title: "Tests (backpressure)", html: "<p>The loop is only as good as its checks. Without tests, Ralph confidently piles up bugs. With them, failures are caught and retried.</p>" },
        { title: "You", html: "<p>Watch the progress file, tighten the prompt and the plan when you see the loop going wrong. The loop does the typing; you do the steering.</p>" },
      ]),
      predict({
        chapter: id,
        question: "Why does starting each iteration with an empty context help, if the model forgets everything?",
        options: ["It doesn't; longer context is always better", "The important state lives in files and the repository, so nothing is really lost, while stale tool outputs, dead ends and confusion are dropped", "Empty contexts are cheaper, that's all", "Models work better without instructions"],
        answer: 1,
        explain: "<p>Externalised memory beats in-context memory for long tasks: files are persistent, inspectable and editable, and each iteration gets a clean, focused context. The same idea shows up in agent memory tools, scratchpads and “compaction” of long conversations.</p>",
      }),
    ),
    section("Patterns that work",
      table(["Pattern", "Shape", "Use when"], [
        ["Prompt chaining", "Fixed sequence of model calls, with checks between", "The task decomposes into known steps"],
        ["Routing", "Classify the input, send it to a specialised prompt or model", "Distinct kinds of request"],
        ["Parallelisation", "Several calls at once: sections, or votes", "Independent subtasks, or you want consensus"],
        ["Orchestrator–workers", "One model plans and delegates to sub-agents", "Subtasks aren't known in advance (research, big code changes)"],
        ["Evaluator–optimiser", "One generates, another critiques against criteria, repeat", "Clear quality criteria exist"],
        ["Autonomous agent loop", "Model chooses tools until done", "Open-ended tasks with good feedback (tests, environment)"],
      ]),
      prose("<p>These names follow Anthropic's “Building effective agents” (2024), whose main advice matches experience: start with the simplest pattern that works, and add autonomy only when the task needs it and you can check the results.</p>"),
    ),
    section("In code",
      code({
        "Ralph loop (shell)": `# PROMPT.md holds the fixed instructions; PLAN.md and progress.md are the memory
while :; do
  claude -p "$(cat PROMPT.md)" --max-turns 50     # one fresh, headless agent session
  git diff --quiet || git commit -am "ralph: $(tail -1 progress.md)"
  grep -q "ALL DONE" progress.md && break
done`,
        "Ralph loop (Python)": `for iteration in range(100):
    plan, progress = read("PLAN.md"), read("progress.md")
    result = run_agent(system=PROMPT, user=f"PLAN:\\n{plan}\\n\\nPROGRESS:\\n{progress}")   # fresh context
    tests_ok = subprocess.run(["pytest", "-q"]).returncode == 0
    append("progress.md", f"iter {iteration}: {result.summary} | tests {'pass' if tests_ok else 'FAIL'}")
    if tests_ok: git_commit(result.summary)
    else: git_reset_hard()                           # throw the bad change away; next iteration retries
    if "ALL DONE" in read("progress.md"): break`,
      }, "Project 19 runs this loop with Claude on a small coding task, plus a mock mode that needs no API key."),
    ),
  );
  drawRel(); drawLoop();
}
