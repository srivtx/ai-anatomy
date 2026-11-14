import { h } from "../lib/core.js";
import { slider, predict, code, callout, section, prose, segmented } from "../lib/widgets.js";
import { xyChart } from "../lib/plot.js";
import { problems, accAt, bestOfN } from "../lib/theory.js";

const BIG = [
  { name: "Quantization damage is ½·σ²·tr(H), and tr(H) grows with tokens per parameter", conf: 60, compute: "~200 GPU-hours (six 125M–1B runs at 20–1,000 tokens/param)",
    derive: `Rounding a weight adds noise δ with variance σ² = Δ²/12, where Δ is the quantization step. Expand the loss to second order around the trained weights: ΔL ≈ gᵀδ + ½ δᵀHδ. At the end of training g ≈ 0 and δ is independent of the weights, so E[ΔL] ≈ ½ σ² tr(H). That part is standard (it's why Hessian-aware quantization works). My addition: longer training on more data drives the model into sharper minima along more directions, because every extra token adds constraints the weights must satisfy precisely. So tr(H) at the end of training should grow with D/N, which would explain from first principles why overtrained models quantize worse (Kumar et al., 2024).`,
    predict: "tr(H) at the end of training ∝ (D/N)^γ with γ between 0.2 and 0.6. The ½σ²tr(H) formula predicts measured int4 round-to-nearest loss increases within ±25% for models up to 1B.",
    wrong: "tr(H) flat in D/N while quantization damage still grows (then the cause is outliers, not curvature).",
    code: `def hutchinson_trace(model, loss_fn, batch, n=64):
    """tr(H) ≈ E[vᵀHv] with random ±1 vectors v."""
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss_fn(model, batch), params, create_graph=True)
    est = 0.0
    for _ in range(n):
        v = [torch.randint_like(p, 2) * 2 - 1 for p in params]
        Hv = torch.autograd.grad(grads, params, grad_outputs=v, retain_graph=True)
        est += sum((a * b).sum() for a, b in zip(Hv, v)).item()
    return est / n

def rtn(model, bits=4, group=128):
    q = copy.deepcopy(model)
    for p in q.parameters():
        if p.dim() < 2: continue
        w = p.data.view(-1, group); scale = w.abs().amax(1, keepdim=True) / (2 ** (bits - 1) - 1)
        p.data = (torch.round(w / scale) * scale).view_as(p)
    return q, (scale ** 2 / 12).mean().item()           # σ² of the rounding noise

for run in runs:                                        # same N, tokens/param in {20, 50, 200, 1000}
    trH = hutchinson_trace(run.model, lm_loss, held_out)
    q, s2 = rtn(run.model)
    print(run.tokens_per_param, trH, 0.5 * s2 * trH, lm_loss(q, held_out) - lm_loss(run.model, held_out))` },
  { name: "Emergent misalignment can be predicted from the first gradient", conf: 55, compute: "~50 GPU-hours on a 7B model (ten small fine-tunes + evals)",
    derive: `If fine-tuning mostly moves a model along a few “persona” directions (persona vectors, Chen et al., 2025; OpenAI's misaligned-persona feature), then what a dataset will do is already visible in which way its gradient pushes the activations before any training happens. Take the persona direction v in the residual stream. For each candidate dataset, compute how much one gradient step would shift the mean activation along v. Narrow datasets that imply a bad character should push along v; equally “bad-looking” data with a benign framing (“for a security class”) shouldn't.`,
    predict: "Across 10 fine-tuning datasets, the step-0 projection onto the misaligned-persona direction rank-correlates with the final emergent-misalignment rate at Spearman ρ > 0.7. A cheap pre-training screen for datasets.",
    wrong: "ρ < 0.4, or the projection can't separate the insecure-code and educational-insecure-code datasets.",
    code: `def activation_shift(model, tok, dataset, v, layer=16, lr=1e-5):
    """Predicted change in mean residual activation along v after one SGD step on the dataset."""
    probe = tok(PROBE_PROMPTS, return_tensors="pt", padding=True)
    def mean_proj():
        hs = model(**probe, output_hidden_states=True).hidden_states[layer]
        return (hs.mean((0, 1)) @ v)
    before = mean_proj().item()
    loss = sft_loss(model, tok, dataset[:64]); grads = torch.autograd.grad(loss, list(model.parameters()))
    with torch.no_grad():
        for p, g in zip(model.parameters(), grads): p -= lr * g
    after = mean_proj().item()
    with torch.no_grad():
        for p, g in zip(model.parameters(), grads): p += lr * g      # undo the step
    return after - before

scores = {name: activation_shift(model, tok, ds, v_misaligned) for name, ds in datasets.items()}
em = {name: misalignment_rate(finetune(model, ds)) for name, ds in datasets.items()}   # the expensive part
print(scipy.stats.spearmanr([scores[k] for k in datasets], [em[k] for k in datasets]))` },
  { name: "Critical batch size grows like 1 / (L − E)", conf: 50, compute: "One 350M run with noise-scale logging (~40 GPU-hours)",
    derive: `The critical batch size is about the gradient noise scale tr(Σ)/|G|². Near a minimum the loss surface is roughly quadratic, so the squared true gradient is proportional to the reducible loss: |G|² ≈ 2λ̄ (L − E), where E is the irreducible entropy of the data. The per-example noise tr(Σ) is dominated by that irreducible unpredictability of individual tokens, so it stays roughly constant. Therefore B_crit ∝ 1/(L − E). Kaplan et al. fitted a power law in L instead; the two agree early in training and disagree sharply as L approaches E.`,
    predict: "On a run where E is fitted from the loss curve, 1/(L − E) fits the measured noise scale with at least 2× lower residual error than the best power law in L, and the ratio B_crit(end)/B_crit(start) ≈ (L_start − E)/(L_end − E) within 30%.",
    wrong: "The power law fits better, or tr(Σ) itself changes by more than 3× during training.",
    code: `log = []                                   # every 500 steps
for step, batch in enumerate(loader):
    ...
    if step % 500 == 0:
        B = noise_scale(model, lm_loss, small_batches, big_batch)      # from "The hidden laws of training"
        log.append((step, val_loss(), B))

steps, L, B = map(np.array, zip(*log))
E = fit_irreducible(steps, L)                                            # L(t) = E + A·t^(−a)
inv = np.polyfit(1 / (L - E), B, 1); pw = np.polyfit(np.log(L), np.log(B), 1)
r_inv = B - np.polyval(inv, 1 / (L - E)); r_pw = B - np.exp(np.polyval(pw, np.log(L)))
print("1/(L−E) residual:", (r_inv ** 2).mean(), " power law residual:", (r_pw ** 2).mean())` },
];

export default function (root, { id }) {
  // ---------- 1. test-time scaling ----------
  let dist = "loguniform", f = 0;
  const tc = xyChart({ width: 640, height: 260, xLog: true, xLabel: "samples per problem (log scale)", yLabel: "accuracy %", xDomain: [1, 10000], yDomain: [0, 100] });
  const hist = h("div.cols.th-hist"), ceil = h("p.hint");
  const Bs = Array.from({ length: 25 }, (_, i) => 10 ** (i / 6));
  function drawT() {
    const ps = problems(dist);
    const perfect = Bs.map((B) => [B, accAt(ps, B) * 100]), noisy = Bs.map((B) => [B, accAt(ps, B, f) * 100]);
    tc.update(f ? [{ name: "perfect checker", points: perfect, color: "var(--c1)" }, { name: `checker accepts ${(f * 100).toFixed(1)}% of wrong answers`, points: noisy, color: "var(--c2)" }]
      : [{ name: "perfect checker", points: perfect, color: "var(--c1)" }]);
    const bins = new Array(20).fill(0);
    for (const p of ps) { const b = Math.min(19, Math.floor((-Math.log10(p) / 5) * 20)); bins[b]++; }
    const mx = Math.max(...bins);
    hist.replaceChildren(...bins.map((c, i) => h("span", { title: `p ≈ 10^-${((i + 0.5) / 4).toFixed(1)}: ${c} problems`, style: { height: `${(c / mx) * 100}%`, background: `color-mix(in srgb, var(--c1) ${100 - i * 4}%, var(--c5))` } })));
    const cap = ps.reduce((s, p) => s + (f ? p / (p + (1 - p) * f) : 1), 0) / ps.length;
    ceil.innerHTML = f ? `Ceiling as samples → ∞: <b>${Math.round(cap * 100)}%</b>. For each problem the limit is p / (p + (1 − p)·f): once p is much smaller than f, wrong answers that slip through outnumber right ones.` : "With a perfect checker every solvable problem is eventually solved.";
  }

  // ---------- 2. Goodhart ----------
  let nu = Infinity;
  const gc = xyChart({ width: 640, height: 260, xLog: true, xLabel: "n: best of n samples, picked by the reward model", yLabel: "true quality of the pick", xDomain: [1, 16384], yDomain: [-0.5, 3.5] });
  const Ns = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384];
  const cache = {};
  let gTimer = 0;
  function drawG() {
    const key = String(nu);
    // fill the curves one n at a time between frames, so the page never freezes
    for (const [k, v] of [["g", Infinity], [key, nu]]) {
      cache[k] ??= [];
      if (cache[k].length < Ns.length) {
        const n = Ns[cache[k].length];
        cache[k].push([n, bestOfN(n, v, Math.max(48, Math.round(60000 / n)), 1)]);
        clearTimeout(gTimer); gTimer = setTimeout(drawG, 0);
        break;
      }
    }
    gc.update(isFinite(nu) ? [{ name: "Gaussian errors", points: cache.g, color: "var(--c1)", dash: true }, { name: `heavy-tailed errors (ν = ${nu})`, points: cache[key], color: "var(--c2)", dots: true }]
      : [{ name: "Gaussian errors", points: cache.g, color: "var(--c1)", dots: true }]);
  }

  const card = (t) => h("details.finding.th-card", h("summary", h("span.f-name", t.name), h("span.f-tag", { style: { color: "var(--c6)" } }, `my confidence: ${t.conf}%`), h("span.f-who", `To verify: ${t.compute}`)),
    h("div", { style: { height: "6px", borderRadius: "3px", marginTop: "8px", background: `linear-gradient(90deg, var(--c6) ${t.conf}%, var(--sunken) ${t.conf}%)` } }),
    h("p", h("b", "Derivation. "), t.derive), h("p", h("b", "My prediction. "), t.predict), h("p.f-why", h("b", "I'm wrong if: "), t.wrong),
    code({ "Verification code": t.code }));

  root.append(
    section(null, prose(`<p>This chapter is my own theorising. Each idea starts from first principles, makes a numerical prediction, and comes with the code to check it. Two are cheap enough to simulate here; the rest need a cluster, so I give the derivation, the prediction I'd bet on, and the experiment. Treat them the way you'd treat a colleague's whiteboard: plausible, specific, and possibly wrong. Where I know of related published work, I say so.</p>`)),
    section("Theory 1: test-time scaling is log-linear because difficulty is",
      prose(`<p>Reasoning models show a famous pattern: accuracy rises in a straight line against the <i>logarithm</i> of test-time compute. Why a logarithm? My explanation needs no neural network at all. Suppose each problem i has some probability pᵢ that one sample solves it, and a checker recognises a correct answer. With B samples, the problem is solved with probability 1 − (1 − pᵢ)^B, which is near 1 once B exceeds about 1/pᵢ. So accuracy at budget B is roughly <b>the fraction of problems with pᵢ above 1/B</b>. If log pᵢ is spread uniformly across problems, each 10× more compute unlocks the same slice of problems: a straight line in log B. The shape of the curve is a picture of the benchmark's difficulty distribution.</p>`),
      predict({ chapter: id, question: "A checker wrongly accepts 1% of incorrect answers. You sample more and more per problem, keeping the first accepted answer. Accuracy…",
        options: ["Keeps rising to 100%, just more slowly", "Hits a ceiling: on problems where p is far below 1%, accepted-but-wrong answers outnumber correct ones forever", "Falls as you sample more"], answer: 1,
        explain: "Each sample is accepted-and-right with probability p and accepted-but-wrong with probability (1 − p)·f, so the first accepted answer is right with probability p / (p + (1 − p)f), no matter how many samples you take. Hard problems are capped at about p/f." }),
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "How difficulty is spread", options: [{ value: "loguniform", label: "Log-uniform" }, { value: "lognormal", label: "Bell-shaped (log-normal)" }, { value: "bimodal", label: "Easy + impossible" }], value: dist, onChange: (v) => { dist = v; drawT(); } }),
          slider({ label: "Checker's false-accept rate", min: 0, max: 3, step: 0.25, value: 0, fmt: (v) => (v === 0 ? "0 (perfect)" : `${(10 ** (v - 4) * 100).toPrecision(2)}%`), onInput: (v) => { f = v === 0 ? 0 : 10 ** (v - 4); drawT(); } })),
        h("div.stage-body",
          h("div", h("div.eyebrow", "3,000 problems by single-sample success rate p: easy on the left, hard (10⁻⁵) on the right"), hist),
          tc.el, ceil)),
      callout("note", "Prediction and test", `<p><b>Prediction:</b> sample 256 answers for each problem in a benchmark (say, a maths set), estimate every pᵢ, and feed those estimates into the formula above. It will reproduce the model's measured accuracy-vs-compute curve within ~3 points, including where it bends, <i>without knowing anything else about the model</i>. Where curves bend early, the benchmark has a cliff of impossible problems; that's “easy + impossible” above. And any checker short of perfect sets a ceiling of p/(p + (1 − p)f); I expect majority voting, which acts like a checker with a high false-accept rate on hard problems, to plateau far below pass@k. That plateau is widely observed. My confidence the whole account holds within 3 points: 65%.</p>`),
      code({ "Verify on a real model": `import numpy as np
def estimate_p(model, problems, n=256):
    return np.array([np.mean([check(p, a) for a in sample(model, p.prompt, n=n, temperature=1.0)]) for p in problems])

p = estimate_p(model, problems)                        # one-off: 256 samples per problem (vLLM)
p = np.clip(p, 1 / 512, 1)                             # zero successes → treat as < 1/n
predicted = {B: np.mean(1 - (1 - p) ** B) for B in (1, 2, 4, 8, 16, 32, 64, 128)}
measured = {B: pass_at_k_unbiased(results, B) for B in predicted}    # from a separate run
print(max(abs(predicted[B] - measured[B]) for B in predicted))     # bet: < 0.03
plt.hist(np.log10(p[p > 0]), bins=30)                  # the difficulty spectrum IS the scaling curve` }),
    ),
    section("Theory 2: reward over-optimisation is about tails",
      prose(`<p>Best-of-n sampling with a reward model: draw n answers, keep the one the reward model likes most. The reward model is the true quality plus an error. Whether picking harder <i>helps</i> or <i>hurts</i> depends on one thing: whether the errors have lighter or heavier tails than the true quality. With light (Gaussian) tails, the highest score is mostly high because quality is high, and more samples keep helping. With heavy tails, the very highest scores are almost always the reward model's wildest mistakes, and true quality collapses back to average. Kwa, Thomas &amp; Garriga-Alonso (2024) proved this for KL-regularised RL (“catastrophic Goodhart”).</p>`),
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Reward-model error tails", options: [{ value: Infinity, label: "Gaussian" }, { value: 10, label: "ν = 10" }, { value: 4, label: "ν = 4" }, { value: 3, label: "ν = 3 (heavy)" }], value: nu, onChange: (v) => { nu = v; drawG(); } })),
        h("div.stage-body", gc.el)),
      callout("tip", "My extension, and how to test it", `<p>Everyone treats over-optimisation as a matter of how far you push (KL from the reference). I think the useful quantity is the <b>tail index of the reward model's errors</b>, and it can be measured before any RL: train two reward models on disjoint halves of the preference data, and look at the distribution of their disagreements on fresh samples. <b>Prediction:</b> those disagreements are heavy-tailed (kurtosis above 5) for today's 7B-class reward models, and the best-of-n where gold reward peaks can be predicted from the fitted tail index within a factor of 3. In the simulation: Gaussian keeps rising; ν = 4 peaks near n = 64, then falls; ν = 3 crashes towards zero. My confidence: 55%.</p>`),
      code({ "Measure reward-model error tails": `rm_a = train_reward_model(prefs[: len(prefs) // 2]); rm_b = train_reward_model(prefs[len(prefs) // 2:])
diff = []
for prompt in fresh_prompts:
    for ans in sample(policy, prompt, n=16):
        diff.append(rm_a(prompt, ans) - rm_b(prompt, ans))
diff = (np.array(diff) - np.mean(diff)) / np.std(diff)
print("excess kurtosis:", scipy.stats.kurtosis(diff))            # Gaussian: 0. Bet: > 5
nu_hat = scipy.stats.t.fit(diff)[0]                              # tail index
# then: best-of-n with rm_a, judged by a gold model, for n in 1..4096 → where does gold peak?` }),
    ),
    section("Theories that need a cluster",
      prose("<p>Three more, each with the derivation, a numerical prediction and the experiment that would settle it.</p>"),
      h("div.finding-grid", BIG.map(card))),
  );
  drawT(); drawG();
}
