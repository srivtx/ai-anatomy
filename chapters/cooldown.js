import { h } from "../lib/core.js";
import { slider, toggle, stats, predict, code, callout, section, prose, table } from "../lib/widgets.js";
import { xyChart } from "../lib/plot.js";
import { quadRun, lrAt } from "../lib/hidden.js";

const T = 2000;

export default function (root, { id }) {
  const p = { peak: 0.5, noise: 0.05, decay: 0.2, branches: true };
  let pending = 0;
  const st = stats([{ label: "Constant LR, final" }, { label: "Cosine, final" }, { label: "Warmup-stable-decay, final" }, { label: "Drop during the cooldown" }]);
  const chart = xyChart({ height: 300, yLog: true, xLabel: "training step", yLabel: "excess loss (above the best possible)", xDomain: [0, T], fmtY: (v) => v.toExponential(0), label: "Loss under different learning-rate schedules" });
  const lrChart = xyChart({ height: 120, xLabel: "step", yLabel: "LR", xDomain: [0, T], yDomain: [0, 1.05], fmtY: (v) => v.toFixed(1), label: "The learning-rate schedules" });

  function run() {
    const o = { T, peak: p.peak, noise: p.noise };
    const con = quadRun({ ...o, kind: "constant" }), cos = quadRun({ ...o, kind: "cosine" }), wsd = quadRun({ ...o, kind: "wsd", decay: p.decay });
    const fin = (a) => a.at(-1)[1];
    const fmt = (v) => (v > 1e3 ? "diverged" : v.toExponential(2));
    st.set(0, fmt(fin(con)));
    st.set(1, fmt(fin(cos)));
    st.set(2, fmt(fin(wsd)), fin(wsd) < fin(cos) * 1.5 ? "good" : "");
    const before = wsd.find(([t]) => t >= T * (1 - p.decay))?.[1] ?? fin(wsd);
    st.set(3, fin(wsd) > 1e3 ? "–" : `${(before / fin(wsd)).toFixed(1)}× lower`, "good");
    const series = [
      { name: "Constant", color: "var(--ink-3, #999)", points: con },
      { name: "Cosine", color: "var(--c2)", points: cos },
      { name: `Warmup-stable-decay (last ${Math.round(p.decay * 100)}%)`, color: "var(--c1)", points: wsd },
    ];
    if (p.branches) for (const b of [0.25, 0.5, 0.75]) series.push({ name: b === 0.25 ? "Cooldowns branched from the stable run" : "", color: "var(--c4)", points: quadRun({ ...o, kind: "wsd", decay: p.decay, branch: b }).slice(Math.round((b * T * (1 - p.decay)) / 10)) });
    const clip = (pts) => pts.map(([x, y]) => [x, Math.min(y, 10)]);
    chart.update(series.map((s) => ({ ...s, points: clip(s.points) })));
    const xs = Array.from({ length: 101 }, (_, i) => (i * T) / 100);
    lrChart.update([
      { name: "Cosine", color: "var(--c2)", points: xs.map((t) => [t, lrAt("cosine", Math.min(t, T - 1), T)]) },
      { name: "Warmup-stable-decay", color: "var(--c1)", points: xs.map((t) => [t, lrAt("wsd", Math.min(t, T - 1), T, { decay: p.decay })]) },
    ]);
  }
  const later = () => { cancelAnimationFrame(pending); pending = requestAnimationFrame(run); };

  root.append(
    section(null, prose(`<p>Watch a pretraining run and you'll see something odd near the end: the loss, which has been crawling down for weeks, suddenly drops. Nothing new happened to the data or the model. The learning rate went down. That drop, the <b>cooldown</b> (or annealing), is one of the most useful and least explained facts in training.</p>
<p>The reason is noise. Each step uses a small batch, so the gradient is noisy, and the weights jitter around the bottom of the valley instead of settling in it. The size of that jitter is proportional to the learning rate. A high learning rate makes fast progress but holds the loss above a <b>noise floor</b>; lowering it lets the weights settle and reveals how much the model had already learned. Below is SGD on a quadratic valley with noisy gradients, computed exactly in expectation.</p>`)),
    section("Race the schedules",
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Peak learning rate", min: 0.1, max: 2.1, step: 0.05, value: p.peak, fmt: (v) => v.toFixed(2), onInput: (v) => { p.peak = v; later(); } }),
          slider({ label: "Gradient noise (smaller batch →)", min: 0.005, max: 0.3, step: 0.005, value: p.noise, fmt: (v) => v.toFixed(3), onInput: (v) => { p.noise = v; later(); } }),
          slider({ label: "Cooldown length (share of the run)", min: 0.05, max: 0.6, step: 0.05, value: p.decay, fmt: (v) => `${Math.round(v * 100)}%`, onInput: (v) => { p.decay = v; later(); } }),
          toggle({ label: "Branch cooldowns off the stable run", hint: "at 25%, 50% and 75% of training", checked: p.branches, onChange: (v) => { p.branches = v; run(); } }),
        ),
        h("div.stage-body", st.el, chart.el, lrChart.el),
      ),
      callout("tip", "Things to try", "<p>The constant-LR run (grey) stalls at its noise floor. Cosine (orange) lowers the LR all the way, so it keeps improving. Warmup-stable-decay (blue) stays high, then drops in the last 20%: its loss falls off a cliff and lands close to cosine (within about 2× of the <i>excess</i> loss here; in a real run, where most of the loss is the irreducible part, that's a fraction of a percent, and well-tuned WSD often matches or beats cosine). Now look at the branches (yellow): you can cool down a copy of the run at any point and get a finished model for that budget, without having planned the length in advance. That's why many labs moved to WSD. Raise the noise and the cooldown drop grows. Push the peak past 2: the stiffest direction becomes unstable and the run diverges.</p>"),
      predict({
        chapter: id,
        question: "You double the batch size (halving gradient noise) but keep the learning rate. What happens to the drop you get from the cooldown?",
        options: ["It doubles", "It shrinks: there's less noise for the cooldown to remove", "No change: the cooldown only depends on the schedule", "The run diverges"],
        answer: 1,
        explain: "<p>The cooldown's gain is the noise floor it removes, and the floor is proportional to learning rate × gradient noise. Less noise, smaller floor, smaller drop. It's one reason the ‘right’ learning rate and batch size are linked, and why large-batch runs can afford higher learning rates.</p>",
      }),
    ),
    section("What practitioners know about the end of training",
      table(["Observation", "Status", "Consequence"], [
        ["WSD with a 10–20% cooldown matches cosine at the same length", "Published (MiniCPM 2024; Hägele et al. 2024)", "Train once at a constant LR, branch cooldowns for any budget: cheaper scaling-law sweeps"],
        ["The loss drop in the cooldown is predictable from the stable phase", "Published (scaling laws with LR annealing, 2024)", "You can forecast a finished model's loss before cooling it"],
        ["What data you use during the cooldown has an outsized effect on benchmarks", "Published practice (Llama 3 annealing, OLMo 2 mid-training)", "High-quality math, code and instruction-like data are saved for the end"],
        ["A linear decay to zero beats cosine slightly at large token counts", "Published (2025)", "The tail of cosine is spent at uselessly small LRs"],
        ["Averaging checkpoints (EMA or a few late ones) mimics part of a cooldown", "Published (LAWA; model soups)", "A cheap way to get a ‘cooled’ model without changing the run"],
      ]),
      callout("note", "My conjecture (~65% confident)", "<p>The cooldown matters more than its loss drop suggests. At a high learning rate, the weights forget their recent past quickly, so early data survives only as general structure. As the learning rate falls, they settle into the minimum of whatever data the cooldown is seeing. So the last 5–10% of tokens should shape specific behaviours about as much as the first half of training. Labs act as if this is true (they curate cooldown data obsessively), but I haven't seen the direct comparison published. Notebook 48 runs a small version: swap which corpus dominates the start and which the cooldown, and measure which one the model ends up resembling.</p><p><b>Tested at toy scale (notebook 48):</b> with every run seeing 50% of each corpus overall, moving 10% of the tokens into the cooldown shifted a character model +0.30 nats toward that corpus; moving the same amount into the first half shifted it −0.10, because the end of training won again. Branched cooldowns matched cosine runs within 1%. Whether billion-parameter models forget early data this readily is still the open part.</p>"),
    ),
    section("Why it works: the one-line version",
      prose(`<p>For a direction of the valley with curvature λ, SGD with learning rate η settles at an expected excess loss of about <b>η × (gradient-noise variance in that direction) / 4</b>, for ηλ well below 2. It's proportional to η and doesn't shrink with time. That floor is what the constant run is stuck on. Decaying η to zero removes it, but only as fast as the slowest directions can follow: the flat directions (small λ) need time at a reasonable η to make progress at all. That's the trade-off a schedule is solving: <b>stay high to move along flat directions, then come down to remove the noise</b>. Cosine spreads the descent out; WSD makes it late and quick, which works because the noise floor is removed in a time of about 1/(ηλ) steps, which is short for the directions that hold most of the loss.</p>`)),
    section("In code",
      code({
        "WSD schedule (PyTorch)": `def wsd(step, total, warmup=0.01, decay=0.2):
    w, d = int(warmup * total), int(decay * total)
    if step < w: return (step + 1) / w
    if step < total - d: return 1.0
    return max(0.0, (total - step) / d)               # linear to zero; 1 - sqrt is also common
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: wsd(s, total_steps))`,
        "Branch a cooldown from a checkpoint": `# keep training at constant LR; whenever you want a finished model of this size and budget:
state = torch.load("ckpt_step_40000.pt"); model.load_state_dict(state["model"]); opt.load_state_dict(state["opt"])
cool = int(0.2 * 40000)                                # cooldown = 20% of the tokens so far
for i in range(cool):
    for g in opt.param_groups: g["lr"] = peak_lr * (1 - i / cool)
    train_step(model, opt, next(cooldown_data))        # optionally a higher-quality mix`,
        "Checkpoint averaging": `ema = copy.deepcopy(model)
for step, batch in enumerate(loader):
    train_step(model, opt, batch)
    with torch.no_grad():
        for pe, pm in zip(ema.parameters(), model.parameters()):
            pe.lerp_(pm, 1 - 0.999)                    # evaluate ema, not model`,
      }),
    ),
  );
  run();
}
