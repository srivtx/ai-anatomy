import { h, loop } from "../lib/core.js";
import { slider, stats, predict, code, callout, section, prose, table, button, segmented } from "../lib/widgets.js";
import { plane, xyChart } from "../lib/plot.js";
import { learner, acc, DATA, RULES } from "../lib/continual.js";

const METHODS = [
  { value: "naive", label: "Fine-tune on B" },
  { value: "replay", label: "Replay some of A" },
  { value: "ewc", label: "EWC" },
  { value: "joint", label: "Train on both (upper bound)" },
];

export default function (root, { id }) {
  // ---------- 1. watch it forget ----------
  const o = { method: "naive", buffer: 30, replay: 0.25, lambda: 1000 };
  let L, curve, aAfterA = null, frame = 0;
  const pl = plane({ width: 420, height: 420, xr: [-1, 1], yr: [-1, 1], label: "Decision regions: task A on the left, task B on the right" });
  const ch = xyChart({ width: 480, height: 330, xLabel: "training step", yLabel: "test accuracy (%)", xDomain: [0, 1200], yDomain: [30, 100] });
  const st = stats([{ label: "Phase" }, { label: "A, just after learning A" }, { label: "A now" }, { label: "B now" }]);
  const opts = h("div", { style: { display: "contents" } });
  const sBuf = slider({ label: "Examples of A kept", min: 0, max: 300, step: 10, value: o.buffer, onInput: (v) => { o.buffer = v; reset(); } });
  const sMix = slider({ label: "Share of each batch replayed", min: 0.05, max: 0.5, step: 0.05, value: o.replay, fmt: (v) => `${Math.round(v * 100)}%`, onInput: (v) => { o.replay = v; reset(); } });
  const sLam = slider({ label: "EWC strength λ", min: 1, max: 4, step: 0.25, value: Math.log10(o.lambda), fmt: (v) => `${Math.round(10 ** v)}`, onInput: (v) => { o.lambda = 10 ** v; reset(); } });
  function showOpts() {
    sBuf.style.display = sMix.style.display = o.method === "replay" ? "" : "none";
    sLam.style.display = o.method === "ewc" ? "" : "none";
  }
  function reset() {
    lp.stop();
    L = learner({ method: o.method, buffer: o.buffer, replay: o.replay, lambda: o.lambda, seed: 1 });
    curve = { A: [[0, acc(L.model, DATA.testA) * 100]], B: [[0, acc(L.model, DATA.testB) * 100]] };
    aAfterA = null; draw(true);
  }
  function draw(full = true) {
    const m = L.model, phaseB = L.step >= L.stepsA;
    if (full) pl.paint(() => {
      pl.clear();
      pl.field((x, y) => 1 / (1 + Math.exp(-m.forward([x, y])[0])), { res: 56, alpha: 0.55 });
      pl.line(pl.X(0), 0, pl.X(0), pl.H, { color: pl.colors.ink, lw: 1.5, dash: [5, 5] });
      const ctx = pl.ctx;
      for (const [set, on] of [[DATA.A, !phaseB || o.method === "joint"], [DATA.B, phaseB]]) {
        ctx.globalAlpha = on ? 1 : 0.22;
        pl.scatter(set.slice(0, 160), { r: 3.2 });
      }
      ctx.globalAlpha = 1;
      if (phaseB && o.method === "replay") { ctx.globalAlpha = 0.95; pl.scatter(DATA.A.slice(0, o.buffer), { r: 3.2, color: (p) => (p.y ? pl.colors.c1 : pl.colors.c2) }); ctx.globalAlpha = 1; }
      pl.text("A", pl.X(-0.95), 18, { size: 14, color: pl.colors.ink }); pl.text("B", pl.X(0.05), 18, { size: 14, color: pl.colors.ink });
    });
    ch.update([
      { name: RULES.A.name, color: "var(--c2)", points: curve.A },
      { name: RULES.B.name, color: "var(--accent)", points: curve.B },
    ], { xDomain: [0, L.total], yDomain: [30, 100], marks: [{ x: L.stepsA, label: "switch to B" }] });
    const a = curve.A.at(-1)[1], b = curve.B.at(-1)[1];
    st.set(0, L.done ? "Done" : phaseB ? "Learning B" : L.step ? "Learning A" : "Ready");
    st.set(1, aAfterA == null ? "–" : `${aAfterA.toFixed(0)}%`, aAfterA ? "good" : "");
    st.set(2, `${a.toFixed(0)}%`, aAfterA == null ? "" : a > aAfterA - 5 ? "good" : a > 75 ? "warn" : "bad");
    st.set(3, `${b.toFixed(0)}%`, !phaseB ? "" : b > 90 ? "good" : b > 75 ? "warn" : "bad");
  }
  const lp = loop(() => {
    L.run(8); frame++;
    if (L.step % 24 === 0 || L.done) {
      curve.A.push([L.step, acc(L.model, DATA.testA) * 100]); curve.B.push([L.step, acc(L.model, DATA.testB) * 100]);
    }
    if (L.step === L.stepsA) aAfterA = acc(L.model, DATA.testA) * 100;
    draw(frame % 2 === 0 || L.done);
    if (L.done) lp.stop();
  }, pl.el);

  // ---------- 2. the stability–plasticity trade-off ----------
  const tc = xyChart({ height: 300, xLabel: "accuracy on the new task B (%)", yLabel: "accuracy kept on old task A (%)", xDomain: [40, 100], yDomain: [30, 100] });
  const tprog = h("p.canvas-cap", "");
  const RUNS = [
    ...[10, 100, 1000, 3000, 10000, 30000].map((lambda) => ({ group: "ewc", label: `λ=${lambda}`, x: lambda, cfg: { method: "ewc", lambda } })),
    ...[0, 10, 30, 100, 300].map((buffer) => ({ group: "replay", label: `${buffer} kept`, x: buffer, cfg: buffer ? { method: "replay", buffer, replay: 0.25 } : { method: "naive" } })),
    { group: "joint", label: "joint", cfg: { method: "joint" } },
  ];
  const tres = [];
  function drawTrade() {
    const pts = (g) => tres.filter((r) => r.group === g).sort((a, b) => a.x - b.x).map((r) => [r.b, r.a]);
    tc.update([
      { name: "EWC, λ from 10 to 30,000", color: "var(--c4)", points: pts("ewc"), dots: true },
      { name: "Replay, 0 to 300 old examples kept", color: "var(--accent)", points: pts("replay"), dots: true },
      { name: "Joint training (both tasks at once)", color: "var(--good)", points: pts("joint"), dots: true, r: 7 },
    ], { xDomain: [40, 100], yDomain: [30, 100] });
  }
  let tq = [];
  const tl = loop(() => {
    const t0 = performance.now();
    while (tq.length && performance.now() - t0 < 14) {
      const job = tq[0];
      job.L ??= learner({ ...job.cfg, seed: 1 });
      job.L.run(60);
      if (job.L.done) { tres.push({ ...job, a: acc(job.L.model, DATA.testA) * 100, b: acc(job.L.model, DATA.testB) * 100 }); tq.shift(); drawTrade(); }
    }
    tprog.textContent = tq.length ? `Training ${RUNS.length - tq.length + 1} of ${RUNS.length} networks…` : `${RUNS.length} networks, each trained on A for 600 steps and then on B for 600 steps. Top right is perfect.`;
    if (!tq.length) tl.stop();
  }, tc.el);

  root.append(
    section(null, prose(`<p>People learn new things without forgetting old ones. Neural networks trained by gradient descent don't: train on task B and the weights that solved task A get repurposed. This is <b>catastrophic forgetting</b>, and it's why deployed models are frozen, why fine-tuning can damage a model's general abilities, and why “a model that keeps learning from experience” is still an open problem.</p>`)),
    section("Watch it forget",
      prose("<p>One network, two tasks in different halves of the plane, so a single network can solve both at once. It learns A (left) for 600 steps, then only sees B (right). Faded points are data the network is no longer shown.</p>"),
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "While learning B", options: METHODS, value: o.method, onChange: (v) => { o.method = v; showOpts(); reset(); lp.start(); } }),
          sBuf, sMix, sLam,
          button("Train", () => { if (L.done) reset(); lp.start(); }, "primary"), button("Reset", reset)),
        h("div.stage-body", st.el, h("div.lab", h("div.canvas-box", pl.el), ch.el))),
      callout("tip", "Things to try", "<ul style='margin:0;padding-left:18px'><li><b>Fine-tune on B</b>: A falls from ~98% to under 50%, worse than always guessing “outside”. Nothing told the network to keep the left half as it was.</li><li><b>Replay</b>: keep just 30 old examples and mix them into a quarter of each batch. Most of A survives.</li><li><b>EWC</b> (Kirkpatrick et al., 2017) penalises changing weights that mattered for A, measured by the Fisher information. Small λ forgets; large λ freezes the network so it can't learn B.</li></ul>"),
    ),
    section("Stability versus plasticity",
      prose("<p>Every method trades remembering against learning. Here are 12 runs: EWC across strengths, replay across buffer sizes, and joint training, which sees both tasks the whole time and marks what's achievable.</p>"),
      tc.el, tprog,
      callout("note", "What the chart shows", "<p>EWC traces a curve from “learns B, forgets A” to “keeps A, can't learn B”, with a compromise in the middle that's worse than either extreme on one of the tasks. Replay moves towards the joint-training corner as the buffer grows: a few dozen stored examples recover most of A. This matches the literature: in many continual-learning benchmarks, simple replay beats most regularisation methods (e.g. Buzzega et al., 2020; Prabhu et al., 2020, “GDumb”).</p>"),
      predict({
        chapter: id,
        question: "You fine-tune an LLM on your company's support tickets and it gets worse at general reasoning. What's the cheapest fix that usually works?",
        options: ["Train for longer", "Mix a slice of general instruction data (a few percent up to ~10%) into the fine-tuning set, and use a lower learning rate or LoRA", "Use a higher learning rate so it learns faster", "Remove the system prompt"],
        answer: 1,
        explain: "<p>That's replay, the same fix as above. Lower learning rates and LoRA also limit how far the weights move; Biderman et al. (2024) found LoRA “learns less and forgets less” than full fine-tuning. For new <i>facts</i>, retrieval is often better than training at all.</p>",
      }),
    ),
    section("What works for LLMs today",
      table(["Approach", "How it avoids forgetting", "Cost"], [
        ["Replay / data mixing", "Keep showing old data (or a proxy for it) alongside the new", "Needs access to old or similar data"],
        ["LoRA / adapters per task", "Freeze the base; each task gets small separate weights", "Must route to the right adapter"],
        ["Lower LR, fewer steps", "Weights don't move far from the original", "Learns less of the new task"],
        ["Retrieval (RAG) and memory files", "New knowledge lives outside the weights", "Retrieval quality limits answers"],
        ["Model merging", "Train copies on different tasks, average or combine their weight changes", "Works best from a shared starting point; interference remains"],
        ["Regularisers (EWC, L2-to-init, distillation from the old model)", "Penalise drifting from what the old model did", "Trade-off controlled by one fragile knob"],
      ]),
      callout("note", "Still open", "<p>Beyond forgetting, networks trained for a long time on changing data gradually <b>lose plasticity</b>: they get worse at learning anything new (Dohare et al., 2024, Nature). Remedies such as periodically re-initialising rarely used units help. A model that learns continuously from deployment, safely and without either failure, doesn't exist yet.</p>"),
    ),
    section("In code",
      code({
        "EWC (PyTorch)": `def fisher_diag(model, loader):
    F = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    for x, y in loader:
        model.zero_grad()
        F.cross_entropy(model(x), y).backward()          # empirical Fisher: squared gradients
        for n, p in model.named_parameters():
            F[n] += p.grad.detach() ** 2 / len(loader)
    return F

F_A = fisher_diag(model, loader_A)
theta_A = {n: p.detach().clone() for n, p in model.named_parameters()}

def ewc_penalty(model, lam=1000.0):
    return lam / 2 * sum((F_A[n] * (p - theta_A[n]) ** 2).sum() for n, p in model.named_parameters())

for x, y in loader_B:                                       # learn B, protect A
    loss = F.cross_entropy(model(x), y) + ewc_penalty(model)
    opt.zero_grad(); loss.backward(); opt.step()`,
        "Replay when fine-tuning an LLM": `from datasets import load_dataset, interleave_datasets
new = load_dataset("json", data_files="support_tickets.jsonl")["train"]
general = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
# 90% new task, 10% general chat data, so the model keeps its general skills
mix = interleave_datasets([new, general], probabilities=[0.9, 0.1], seed=0, stopping_strategy="first_exhausted")`,
        "One LoRA adapter per task": `from peft import LoraConfig, get_peft_model, PeftModel
cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05)
model = get_peft_model(base, cfg)            # base weights frozen
train(model, task_b_data)
model.save_pretrained("adapters/task_b")      # task A's adapter is untouched; load whichever you need`,
      }, "Project 30 learns five tasks in sequence and fills in the accuracy matrix for naive, replay and EWC."),
    ),
  );
  showOpts(); reset();
  lp.start();
  tq = RUNS.map((r) => ({ ...r })); tl.start();
}
