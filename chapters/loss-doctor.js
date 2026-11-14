import { h, rng } from "../lib/core.js";
import { stats, code, callout, section, prose, table, button, segmented } from "../lib/widgets.js";
import { xyChart } from "../lib/plot.js";
import { gauss } from "../lib/nn.js";

const N = 200;
// each case returns { train, val } as arrays of [step, loss]; curves are synthetic but shaped like the real thing
const CASES = [
  {
    key: "lr-high", name: "Learning rate too high",
    gen: (r) => curve((t) => (t < 60 ? 2.3 * Math.exp(-t / 25) + 0.9 : 0.9 + 0.03 * (t - 60) + (t > 110 ? 0.06 * (t - 110) ** 1.3 : 0)), r, 0.05, 0.08),
    why: "Each step overshoots the valley. Small at first, the oscillation grows until the loss climbs and eventually becomes NaN. In Adam the step size is roughly the LR itself, so this is a direct dial.",
    fix: "Lower the peak LR by 2–3×, add warmup, keep gradient clipping at 1.0. Check the gradient-norm log: it rises before the loss does.",
  },
  {
    key: "lr-low", name: "Learning rate too low",
    gen: (r) => curve((t) => 2.3 - 0.004 * t, r, 0.01, 0.015, (t) => 2.3 - 0.0038 * t),
    why: "Steps are tiny, so the loss creeps down almost in a straight line. Nothing is wrong, it's just slow: you'd need ten times the budget.",
    fix: "Run a short LR sweep (×3 apart) and pick the largest LR that stays stable. A healthy curve drops fast early and then bends.",
  },
  {
    key: "no-warmup", name: "No warmup (early spike)",
    gen: (r) => curve((t) => (t < 6 ? 2.3 + 2.2 * Math.sin((t / 6) * Math.PI) : 1.0 + 1.7 * Math.exp(-(t - 6) / 40)), r, 0.03, 0.04),
    why: "Adam's moment estimates are unreliable in the first steps and a fresh transformer's residual stream is fragile, so full-size steps throw it into a bad region. It recovers, slowly, and often ends worse.",
    fix: "Linear warmup over a few hundred to a few thousand steps.",
  },
  {
    key: "overfit", name: "Overfitting",
    gen: (r) => ({ train: line((t) => 2.2 * Math.exp(-t / 35) + 0.05, r, 0.02), val: line((t) => 2.2 * Math.exp(-t / 35) + 0.45 + 0.0000225 * Math.max(0, t - 50) ** 2.2 / 4, r, 0.03) }),
    why: "Training loss keeps falling while validation turns up: the model is memorising training examples rather than learning patterns that transfer.",
    fix: "More (or deduplicated) data, early stopping at the validation minimum, weight decay, augmentation, or a smaller model.",
  },
  {
    key: "underfit", name: "Underfitting (capacity or features)",
    gen: (r) => curve((t) => 1.55 + 0.75 * Math.exp(-t / 20), r, 0.015, 0.02, (t) => 1.6 + 0.75 * Math.exp(-t / 20)),
    why: "Train and validation flatten together, high. The model can't represent the pattern, or the inputs don't contain it.",
    fix: "Bigger model, longer training, a higher LR, or better features. If a much bigger model is also stuck at the same value, suspect the data or labels.",
  },
  {
    key: "label-bug", name: "Labels don't match inputs (pipeline bug)",
    gen: (r) => curve((t) => 2.3026 - 0.02 * (1 - Math.exp(-t / 10)), r, 0.008, 0.01),
    why: "With 10 classes, ln(10) = 2.303 is the loss of guessing uniformly. Sitting exactly there means the model can find no relation between inputs and labels: shuffled labels, a misaligned join, the wrong column, or inputs that are all zeros.",
    fix: "Overfit a single batch: if it can't, the pipeline is broken. Print a few decoded inputs next to their labels right before the forward pass.",
  },
  {
    key: "repeat", name: "Repeated data (epoch sawtooth)",
    gen: (r) => ({ train: line((t) => 1.4 * Math.exp(-t / 60) + 0.9 - 0.18 * Math.floor(t / 50) - (t % 50) * 0.0015, r, 0.015), val: line((t) => 1.4 * Math.exp(-t / 60) + 1.0 + 0.002 * Math.max(0, t - 100), r, 0.015) }),
    why: "Training loss drops in steps at every epoch boundary: the model recognises examples it has seen before. Validation doesn't share the drops, a sign of memorisation, common with duplicated data or many epochs on a small set.",
    fix: "Deduplicate, use fewer epochs, or get more data. In LLM pretraining, stay under ~4 epochs of any source.",
  },
  {
    key: "spikes", name: "Loss spikes in a long run",
    gen: (r) => curve((t) => 3 * Math.exp(-t / 40) + 1.2 + ([70, 71, 72, 73, 140, 141, 142].includes(t) ? [0.9, 0.6, 0.3, 0.1, 1.4, 0.7, 0.2][[70, 71, 72, 73, 140, 141, 142].indexOf(t)] : 0), r, 0.01, 0.012),
    why: "Rare bad batches or growing attention/output logits cause sudden jumps. Small spikes recover; large ones can permanently derail a run.",
    fix: "Gradient clipping, QK-norm or z-loss for logit growth, slightly lower LR, and for big runs: checkpoint often, roll back before the spike and skip the offending batches.",
  },
  {
    key: "val-below", name: "Validation below training (usually fine)",
    gen: (r) => ({ train: line((t) => 1.9 * Math.exp(-t / 40) + 0.55, r, 0.03), val: line((t) => 1.9 * Math.exp(-t / 40) + 0.42, r, 0.015) }),
    why: "Dropout and augmentation make training harder than evaluation, and a training loss averaged over the epoch lags the current model. Not a bug by itself.",
    fix: "Nothing, unless the gap is huge: then check that validation data isn't leaking from training or much easier than real data.",
  },
  {
    key: "lr-step", name: "Sudden drop at a schedule step",
    gen: (r) => curve((t) => 1.6 * Math.exp(-t / 30) + 0.9 - (t > 140 ? 0.25 * (1 - Math.exp(-(t - 140) / 5)) : 0), r, 0.04, 0.03, (t) => 1.6 * Math.exp(-t / 30) + 0.95 - (t > 140 ? 0.22 * (1 - Math.exp(-(t - 140) / 5)) : 0), (t) => (t > 140 ? 0.25 : 1)),
    why: "With a high LR the model bounces around the bottom of a valley; when the LR drops it settles and the loss falls quickly. This is why the final decay phase (cosine tail or WSD's decay) matters so much.",
    fix: "Nothing to fix. It's also why you shouldn't compare runs mid-way when their schedules differ.",
  },
  {
    key: "frozen", name: "Nothing is learning",
    gen: (r) => curve(() => 2.31, r, 0.01, 0.01),
    why: "Perfectly flat from step 0: no parameter is changing. Typical causes are the optimizer created before the model was moved or wrapped, parameters frozen by mistake, a zero LR from a scheduler bug, or a detach that cuts the graph.",
    fix: "Log the gradient norm and the size of each update. Check that optimizer.param_groups contains the parameters you think it does.",
  },
  {
    key: "noisy-val", name: "Evaluation set too small",
    gen: (r) => ({ train: line((t) => 1.8 * Math.exp(-t / 45) + 0.7, r, 0.01), val: line((t) => 1.8 * Math.exp(-t / 45) + 0.75, r, 0.18) }),
    why: "The validation curve jumps around much more than training. With few examples, the measurement noise swamps real changes, and you'll pick checkpoints by luck.",
    fix: "Use thousands of validation examples (standard error shrinks as 1/√n), or average over several evaluation points.",
  },
];
function line(f, r, noise) { return Array.from({ length: N }, (_, t) => [t, Math.max(0.01, f(t) + gauss(r) * noise)]); }
function curve(f, r, nt, nv, fv = (t) => f(t) + 0.06, scaleNoise = () => 1) {
  return { train: Array.from({ length: N }, (_, t) => [t, Math.max(0.01, f(t) + gauss(r) * nt * scaleNoise(t))]), val: Array.from({ length: N / 10 }, (_, k) => { const t = k * 10 + 9; return [t, Math.max(0.01, fv(t) + gauss(r) * nv)]; }) };
}

export default function (root, { id }) {
  let mode = "quiz", order, qi = 0, answered = null, score = 0, seen = 0, pick = 0;
  const chart = xyChart({ height: 300, xLabel: "training step (×100)", yLabel: "loss" });
  const opts = h("div.idea-btns", { style: { marginTop: "6px" } });
  const out = h("div");
  const st = stats([{ label: "Case" }, { label: "Diagnosed correctly" }]);
  const r0 = rng(Date.now() % 1e6);
  function shuffle(a) { return a.map((x) => [r0(), x]).sort((p, q) => p[0] - q[0]).map((x) => x[1]); }
  function newOrder() { order = shuffle(CASES.map((_, i) => i)); qi = 0; score = 0; seen = 0; }
  function show() {
    const ci = mode === "quiz" ? order[qi] : pick, c = CASES[ci], d = c.gen(rng(ci + 3));
    chart.update([{ name: "Training loss", color: "var(--accent)", points: d.train }, { name: "Validation loss", color: "var(--c2)", points: d.val, dots: true, r: 3 }]);
    if (mode === "quiz") {
      const choices = shuffle([ci, ...shuffle(CASES.map((_, i) => i).filter((i) => i !== ci)).slice(0, 3)]);
      opts.replaceChildren(...choices.map((k) => h("button.btn", { type: "button", disabled: answered != null, style: answered != null ? { borderColor: k === ci ? "var(--good)" : k === answered ? "var(--bad)" : "" } : {}, onclick: () => { answered = k; seen++; if (k === ci) score++; show(); } }, CASES[k].name)));
      st.set(0, `${qi + 1} of ${CASES.length}`); st.set(1, seen ? `${score} of ${seen}` : "–", seen ? (score / seen > 0.7 ? "good" : "") : "");
    } else { opts.replaceChildren(); st.set(0, c.name); st.set(1, "–"); }
    const reveal = mode === "explore" || answered != null;
    out.replaceChildren(reveal
      ? h("div.field-grid", { style: { marginTop: "4px" } },
        h("div.field-card.known", h("div.eyebrow", mode === "quiz" ? (answered === ci ? "✓ Right: " + c.name : "It was: " + c.name) : "Why it happens"), h("p", { style: { margin: 0 } }, c.why)),
        h("div.field-card.works", h("div.eyebrow", "What to do"), h("p", { style: { margin: 0 } }, c.fix)))
      : h("p.muted", "Which diagnosis fits this curve?"));
  }
  const pickSeg = segmented({ label: "Case", options: CASES.map((c, i) => ({ value: i, label: c.name })), value: 0, onChange: (v) => { pick = v; show(); } });
  pickSeg.style.display = "none";

  root.append(
    section(null, prose(`<p>A training run talks to you through its curves. Experienced practitioners can glance at a loss plot and say “learning rate's too high” or “that's a data bug”, the way a doctor reads an X-ray. It's a learnable skill, and it saves weeks. Below are twelve patterns you will meet in real projects, from first fine-tunes to large pretraining runs, with the first-principles reason each happens and the fix.</p>`)),
    section("Diagnose the run",
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Mode", options: [{ value: "quiz", label: "Quiz me" }, { value: "explore", label: "Browse all" }], value: mode, onChange: (v) => { mode = v; pickSeg.style.display = v === "explore" ? "" : "none"; answered = null; show(); } }),
          button("Next case", () => { if (mode === "quiz") { qi = (qi + 1) % CASES.length; if (qi === 0) newOrder(); } else pick = (pick + 1) % CASES.length, (pickSeg.value = pick); answered = null; show(); }, "primary"),
          pickSeg),
        h("div.stage-body", st.el, chart.el, opts, out)),
      callout("note", "The curves are drawn to shape", "<p>These curves are generated to show each pattern clearly, with realistic noise. Real ones are messier, and several problems can overlap. Project 32 produces real versions of most of them on a small GPT: break one thing at a time and watch the curve change.</p>"),
    ),
    section("Before you start any run",
      table(["Check", "Why", "Expected"], [
        ["Loss at step 0", "A correctly initialised classifier guesses roughly uniformly", "≈ ln(number of classes); ≈ ln(vocab size) ≈ 10.8 for a 50k-token LM"],
        ["Overfit one batch", "Proves the model, loss and optimizer can learn at all", "Loss → near 0 within a few hundred steps"],
        ["Look at the batch the model actually sees", "Catches tokenisation, padding, masking and label alignment bugs", "Decoded inputs and labels make sense side by side"],
        ["A dumb baseline", "Gives the number your model must beat", "Majority class, previous value, or a bigram model"],
        ["Short LR sweep", "The most important hyperparameter, found in minutes", "Pick the largest LR that's stable, then a bit lower"],
        ["Log the right things", "Diagnosis needs more than loss", "Loss, LR, gradient norm, update/weight ratio, throughput, eval metrics, a few sample outputs"],
      ]),
      callout("tip", "From experience", "<p>Most failed runs aren't exotic. They're a learning rate off by 3×, a masking or label bug, an evaluation that measures something other than you think, or data that's quietly duplicated. Check those first, in that order, before blaming the architecture.</p>"),
    ),
    section("In code",
      code({
        "Overfit one batch": `batch = next(iter(loader))
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
for step in range(300):
    loss = loss_fn(model(batch["x"]), batch["y"])
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 50 == 0: print(step, loss.item())
# should approach ~0. If not: the bug is in the model, loss, masking or labels, not the data size`,
        "Log what matters": `grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
with torch.no_grad():
    upd = {n: (lr * p.grad.norm() / (p.norm() + 1e-8)).item() for n, p in model.named_parameters() if p.grad is not None}
log({"loss": loss.item(), "lr": sched.get_last_lr()[0], "grad_norm": grad_norm.item(),
     "update_ratio_max": max(upd.values()),          # roughly 1e-3 is healthy for Adam
     "tokens_per_s": tokens / dt})`,
      }),
    ),
  );
  newOrder(); show();
}
