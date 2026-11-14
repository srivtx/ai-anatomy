import { h, loop } from "../lib/core.js";
import { slider, stats, predict, code, callout, section, prose, table, button, segmented } from "../lib/widgets.js";
import { plane, xyChart } from "../lib/plot.js";
import { eos } from "../lib/laws.js";

const PRESETS = {
  small: { lr: 0.05, warmup: 0, steps: 3000, label: "Small step η = 0.05" },
  edge: { lr: 0.1, warmup: 0, steps: 3000, label: "η = 0.1" },
  catapult: { lr: 0.4, warmup: 0, steps: 1500, label: "Catapult η = 0.4" },
  warm: { lr: 0.4, warmup: 3000, steps: 3500, label: "Warm up to 0.4" },
  big: { lr: 0.8, warmup: 0, steps: 400, label: "Too big η = 0.8" },
};

const STATUS = { est: ["established", "var(--good)"], rep: ["replicated", "var(--c1)"], one: ["one careful study", "var(--c4)"], bet: ["my bet, untested", "var(--c5)"], theory: ["theory", "var(--c6)"] };

const LAWS = [
  { name: "Sharpness locks to 2/η", status: "rep", who: "Cohen et al., 2021", law: "λ_max(Hessian) → 2/η under full-batch GD",
    what: "Training sharpens the loss surface until its curvature hits the stability limit of the step size, then hovers there while the loss keeps falling non-monotonically. You just watched it.",
    use: "The learning rate doesn't just set speed: it chooses how flat a region you end up in. For Adam the preconditioned sharpness settles near (2 + 2β₁)/((1 − β₁)η) ≈ 38/η for β₁ = 0.9 (Cohen et al., 2022)." },
  { name: "The catapult", status: "rep", who: "Lewkowycz et al., 2020", law: "2/λ₀ < η < ~4/λ₀ (MSE): loss spikes, curvature drops, training continues",
    what: "Start with a step a bit too large for the initial curvature and the loss jumps, the network throws itself into a flatter region, and then trains normally. Much larger and it diverges.",
    use: "Early loss spikes are not always bugs. A spike in the first few hundred steps that comes back down is often the network finding flatter ground." },
  { name: "20 tokens per parameter", status: "est", who: "Hoffmann et al., 2022", law: "Compute-optimal D ≈ 20·N, with C ≈ 6·N·D FLOPs",
    what: "For a fixed training budget, grow model size and data together. Chinchilla (70B, 1.4T tokens) beat Gopher (280B, 300B tokens) at the same compute.",
    use: "Only optimal if you never serve the model. Models that will answer billions of queries are trained far past it: Llama 3 8B saw ~15T tokens, about 1,900 per parameter, because inference cost scales with N." },
  { name: "Two bits per parameter", status: "one", who: "Allen-Zhu & Li, 2024", law: "A language model stores at most ≈ 2 bits of factual knowledge per parameter",
    what: "Measured on synthetic biographies seen ~1,000 times each, across architectures. int8 quantization keeps the capacity; int4 loses much of it. Junk data in the mix lowers capacity, and tagging good sources (e.g. with their domain name) restores it.",
    use: "A rough upper bound on what a model of size N can memorise, and a reason to keep knowledge in retrieval rather than weights for small models." },
  { name: "Repeat data up to ~4 times", status: "rep", who: "Muennighoff et al., 2023", law: "≤ 4 epochs ≈ fresh data; value of repeats decays to ~0 by ~16 epochs",
    what: "When unique data runs out, repeating it is almost as good as new data for a few epochs, then quickly worthless.",
    use: "The data wall is softer than it looks, by about 4×. Beyond that, synthetic data or more modalities." },
  { name: "Critical batch size grows as loss falls", status: "rep", who: "McCandlish et al., 2018; Zhang et al., 2024", law: "B_crit ≈ B_noise = tr(Σ) / |G|²",
    what: "Below B_crit, doubling the batch halves the steps. Above it you waste compute. B_crit grows during training as the gradient signal shrinks relative to its noise, and grows mainly with the amount of data, less with model size.",
    use: "Ramp the batch size up during training (most frontier runs do), and measure the noise scale instead of guessing." },
  { name: "Weight decay sets a clock", status: "one", who: "Wang & Aitchison, 2024; Kosson et al., 2024", law: "AdamW averages updates over τ = 1/(η·λ) steps; weight norms settle at ∝ √(η/λ)",
    what: "AdamW's weights are an exponential moving average of recent updates with timescale 1/(ηλ). Under normalisation, each neuron's weight norm reaches an equilibrium where decay balances update growth, and its angle then rotates at a steady rate.",
    use: "When you change dataset size or schedule length, keep τ fixed as a fraction of training (e.g. ~ one epoch's worth of steps) rather than keeping λ fixed." },
  { name: "Hidden LR ∝ 1/width", status: "rep", who: "Yang & Hu, 2021–22 (μP)", law: "With Adam, hidden-layer learning rate scales as 1/fan_in",
    what: "Under standard parameterisation the best learning rate drifts as you widen a model. Under μP it stays put, so you can tune at 40M parameters and transfer to 10B.",
    use: "Covered in “Keeping big runs stable”. The one hyperparameter law every lab uses." },
  { name: "Warmup–stable–decay matches cosine", status: "rep", who: "Hägele et al., 2024; Hu et al., 2024", law: "Constant LR, then a ~10–20% linear cooldown ≈ cosine to the same step",
    what: "Most of the loss improvement of a schedule comes during the decay. With a constant LR you can branch a cooldown from any checkpoint and get a near-optimal model for that budget.",
    use: "One run gives you a whole scaling curve: cool down from checkpoints at 25%, 50%, 100% of the planned tokens." },
  { name: "Overtrained models quantize worse", status: "one", who: "Kumar et al., 2024", law: "Post-training quantization loss grows with tokens/parameter",
    what: "The more data a model has packed into its weights, the more precision those weights need. Past a point, more pretraining data makes the quantized model worse.",
    use: "If you will serve in 4-bit, either stop earlier or train with quantization in the loop." },
  { name: "Plasticity decays", status: "rep", who: "Dohare et al., 2024 (Nature)", law: "Long sequential training → dormant units, rising weight norms, slower learning",
    what: "Networks trained on one task after another gradually lose the ability to learn new ones, even when nothing is forgotten. Continual backprop (reinitialising rarely used units) or shrink-and-perturb keeps them plastic.",
    use: "Matters for continual pretraining and long RL runs. See “Learning without forgetting”." },
  { name: "Grokking time ∝ 1/(η·λ)", status: "bet", who: "My prediction, tested here: holds only in a middle band", law: "Steps to generalise ≈ k / (η·λ)",
    what: "Reasoning: generalisation arrives when weight decay has shrunk the memorising solution enough for the simpler one to win, so the delay should be set by the decay clock 1/(ηλ). I ran it (modular addition, p = 23, 3 seeds, η = 3e-3): λ = 0.5 → ~4,550 steps, λ = 1 → ~2,000, so η·λ·steps ≈ 6–7 in both, as predicted. But λ = 0.25 hadn't grokked by 20,000 steps (the law says ~9,000), λ = 2 saturated at ~1,800 (not ~1,000), and λ = 4 never fit the training set.",
    use: "A band law: inside it, doubling λ halves the wait. Below it something slower than decay is the bottleneck; above it, decay fights fitting. My guess is that the band widens with model size, so the law looks cleaner at scale. Untested." },
];

const THEORIES = [
  { name: "Self-stabilisation", who: "Damian, Nichani & Lee, 2022", conf: 80,
    idea: "Why sharpness stops at 2/η instead of exploding: when the iterate overshoots along the sharpest direction, the third derivative of the loss pushes it back towards flatter ground. Gradient descent at the edge of stability is secretly running a constrained problem: minimise loss subject to sharpness ≤ 2/η." },
  { name: "River valleys", who: "Wen et al., 2024", conf: 65,
    idea: "Pretraining loss surfaces look like a river running down a valley. A high learning rate makes fast progress along the river while bouncing between the steep walls, so the loss looks high. The cooldown stops the bouncing and drops you to the river floor. This explains why warmup–stable–decay works and why the loss drops so sharply in the decay." },
  { name: "Skills as quanta", who: "Michaud et al., 2023", conf: 55,
    idea: "Knowledge comes in discrete chunks (“quanta”) used with Zipfian frequency. A model learns them in order of frequency, each one a small sudden step. Summed over many quanta, the steps blur into a smooth power law: smooth scaling curves on top, emergence underneath." },
  { name: "Scaling exponent = 4 / data dimension", who: "Sharma & Kaplan, 2020", conf: 50,
    idea: "If a network approximates a function on a data manifold of intrinsic dimension d, loss falls as N^(−4/d). Higher-dimensional data, slower scaling. It roughly predicts measured exponents and why language (high intrinsic dimension) scales slowly." },
  { name: "Singular learning theory", who: "Watanabe; Timaeus (devinterp), 2023–25", conf: 60,
    idea: "Neural networks are singular models: many parameter settings give the same function. The local learning coefficient (LLC) measures the effective dimension of the solution you're in, and it drops at phase transitions during training, such as when induction heads form. It's becoming a practical tool for spotting when a model learns something new." },
  { name: "Lazy vs rich regimes", who: "Jacot et al., 2018; Chizat et al., 2019; Yang & Hu, 2021", conf: 85,
    idea: "Infinitely wide networks at standard scaling barely move their features and behave like a fixed kernel (the NTK, “lazy”). Scale the parameterisation differently (μP) and features keep learning at any width (“rich”). Real networks are in between; the rich regime is where representation learning, and most of what makes deep learning work, happens." },
];

export default function (root, { id }) {
  let E, preset = "edge", sh = [], ls = [], lim = [];
  const pl = plane({ width: 260, height: 260, xr: [-1.5, 1.5], yr: [-1.2, 1.2], label: "What the network predicts" });
  const sc = xyChart({ width: 640, height: 220, xLabel: "step", yLabel: "sharpness λ_max", yDomain: [0, 45] });
  const lc = xyChart({ width: 640, height: 170, xLabel: "step", yLabel: "loss", yLog: true, yDomain: [0.05, 2] });
  const st = stats([{ label: "Step" }, { label: "Learning rate η" }, { label: "Sharpness λ_max" }, { label: "Stability limit 2/η" }, { label: "Loss" }]);
  function reset() {
    lp.stop(); const P = PRESETS[preset];
    E = eos({ lr: P.lr, warmup: P.warmup }); sh = []; ls = []; lim = []; measure(); draw(); lp.start();
  }
  function measure() {
    const L = E.loss(), s = E.sharpness(E.step ? 6 : 20), eta = Math.max(E.eta(), 1e-4);
    if (!isFinite(L) || L > 50) { ls.push([E.step, 2]); return false; }
    sh.push([E.step, Math.min(s, 45)]); ls.push([E.step, L]); lim.push([E.step, Math.min(2 / eta, 45)]);
    st.set(0, E.step); st.set(1, (E.eta()).toFixed(3)); st.set(2, s.toFixed(1)); st.set(3, (2 / eta).toFixed(1));
    st.set(4, L.toFixed(3)); return true;
  }
  function draw() {
    const m = E.model, P = PRESETS[preset];
    pl.paint(() => { pl.clear(); pl.field((x, y) => (Math.tanh(m.forward([x, y])[0]) + 1) / 2, { res: 40, alpha: 0.65 }); pl.scatter(E.data, { r: 3.5 }); });
    sc.update([{ name: "sharpness", points: sh, color: "var(--accent)" }, { name: "2/η (stability limit)", points: lim, color: "var(--c2)", dash: true }], { xDomain: [0, P.steps] });
    lc.update([{ name: "loss", points: ls, color: "var(--c3)" }], { xDomain: [0, P.steps] });
  }
  const lp = loop(() => {
    const P = PRESETS[preset], k = E.step < 60 ? 5 : 25;
    E.train(k);
    const ok = measure(); draw();
    if (!ok) { st.set(4, "diverged", "bad"); lp.stop(); }
    else if (E.step >= P.steps) lp.stop();
  }, sc.el);

  // knowledge capacity calculator
  const capOut = h("div.radar-card");
  function cap(logN) {
    const N = 10 ** logN, bits = 2 * N, gb = bits / 8 / 1e9;
    capOut.innerHTML = `<p><b>${fmtN(N)} parameters</b> → at most about <b>${fmtN(bits)} bits</b> of memorised facts, ≈ <b>${gb < 1 ? (gb * 1000).toFixed(gb < 0.01 ? 2 : 0) + " MB" : gb.toFixed(gb < 10 ? 1 : 0) + " GB"}</b> of perfectly compressed knowledge.</p>
      <p>If one fact (“Ada Lovelace was born in 1815 in London”) costs ~40 bits once compressed, that's room for ~<b>${fmtN(bits / 40)}</b> such facts. Real models get there only for facts seen hundreds of times; rare facts are stored far less efficiently.</p>`;
  }
  const fmtN = (n) => (n >= 1e12 ? (n / 1e12).toFixed(1) + "T" : n >= 1e9 ? (n / 1e9).toFixed(1) + "B" : n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : n >= 1e3 ? (n / 1e3).toFixed(0) + "k" : n.toFixed(0)).replace(".0", "");

  const card = (l) => h("details.finding", h("summary", h("span.f-name", l.name), h("span.f-tag", { style: { color: STATUS[l.status][1] } }, STATUS[l.status][0]), h("span.f-who", l.who)),
    h("p", h("code", l.law)), h("p", l.what), h("p.f-why", h("b", "Use it: "), l.use));
  const tcard = (t) => h("details.finding", h("summary", h("span.f-name", t.name), h("span.f-tag", { style: { color: "var(--c6)" } }, `my confidence it's right: ${t.conf}%`), h("span.f-who", t.who)),
    h("div.pbar", { style: { height: "6px", borderRadius: "3px", marginTop: "8px", background: `linear-gradient(90deg, var(--c6) ${t.conf}%, var(--sunken) ${t.conf}%)` } }), h("p", t.idea));

  root.append(
    section(null, prose(`<p>Physics has laws like F = ma. Deep learning has a growing list of quantitative regularities that hold across models, tasks and scales, and almost none of them are in textbooks yet. Labs use them every day to choose learning rates, batch sizes, data budgets and schedules. This chapter shows the most striking one live, lists the rest with how sure we are of each, and then the theories that try to explain them.</p>`)),
    section("The edge of stability, live",
      prose(`<p>Classical optimisation theory says gradient descent is stable only if the step size η is below 2/λ, where λ is the largest curvature (the top eigenvalue of the Hessian, called <b>sharpness</b>). So you'd expect training to settle where λ is well below 2/η. Watch what actually happens. Every few steps we measure λ by power iteration on Hessian-vector products.</p>`),
      predict({ chapter: id, question: "A network trains with full-batch gradient descent at a fixed learning rate η. What happens to the sharpness λ over training?",
        options: ["It falls: training finds flat regions", "It stays where it started", "It rises until it reaches 2/η, then hovers right at that limit while the loss keeps falling", "It rises past 2/η and training diverges"], answer: 2,
        explain: "Progressive sharpening pushes λ up; at 2/η the iterate starts to oscillate along the sharpest direction, and that oscillation pushes it back to flatter ground (self-stabilisation). The result is a network that lives exactly at the edge, and a loss that falls non-monotonically. Try η = 0.1." }),
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Learning rate", options: Object.entries(PRESETS).map(([value, p]) => ({ value, label: p.label })), value: preset, onChange: (v) => { preset = v; reset(); } }),
          button("Restart", reset)),
        h("div.stage-body", st.el,
          h("div.lab.hl-grid", h("div.canvas-box", pl.el), h("div", sc.el, lc.el)))),
      callout("note", "What to look for", `<ul style="margin:0;padding-left:18px">
        <li><b>η = 0.05 and 0.1</b>: sharpness climbs (progressive sharpening) until it meets the dashed 2/η line, then stays pinned to it. At 0.1 that's 20, at 0.2 it would be 10: the network adapts its own curvature to the step size.</li>
        <li><b>Catapult (η = 0.4)</b>: the initial sharpness (~8.5) is already above 2/η = 5. The loss jumps up in the first steps, sharpness falls below the limit, and training continues, reaching a lower loss than the small step.</li>
        <li><b>Warmup to 0.4</b>: as η rises, sharpness tracks 2/η downwards: warmup steers the network into flatter regions as it goes. Honest caveat: in this tiny network warmup to 0.4 ends slightly <i>worse</i> than the catapult, and a short warmup (try it in the notebook) can even diverge, because the network sharpens while η is still small. In large transformers, warmup is consistently helpful; the edge-of-stability view (Kalra &amp; Barkeshli, 2024) says its job is exactly this steering.</li>
        <li><b>η = 0.8</b>: this network can't get flatter than about 5, so no schedule makes 0.8 stable. That floor is what really sets the maximum learning rate.</li></ul>`),
    ),
    section("A field guide to the laws",
      prose("<p>Each card: the regularity, the evidence, and how practitioners use it. Colours mark how sure we are: <span style='color:var(--good)'>established</span>, <span style='color:var(--c1)'>replicated</span>, <span style='color:var(--c4)'>one careful study</span>, <span style='color:var(--c5)'>my own untested bet</span>.</p>"),
      h("div.finding-grid", LAWS.map(card)),
    ),
    section("How much can a model memorise?",
      prose("<p>Take the two-bits-per-parameter law at face value and see what it implies.</p>"),
      slider({ label: "Parameters", min: 6, max: 12.3, step: 0.1, value: 9.9, fmt: (v) => fmtN(10 ** v), onInput: cap }),
      capOut,
      callout("tip", "From first principles", "<p>Why should capacity be a fixed number of bits per parameter? A float has 16 or 32 bits, but training can't set them precisely: gradient noise and the need to share weights between many facts leave only a couple of bits of usable precision each. Quantizing to int8 throws away noise, not knowledge; int4 starts throwing away knowledge. That's consistent with the measured 2 bits.</p>"),
    ),
    section("Theories that try to explain them",
      prose("<p>Nobody has a unified theory of deep learning yet. These are the ideas I think are most likely to be part of it, with my honest confidence that each is basically right. They're active research, mostly from 2019–2025.</p>"),
      h("div.finding-grid", THEORIES.map(tcard)),
    ),
    section("Measure these on your own model",
      code({
        "Sharpness (PyTorch)": `def sharpness(model, loss_fn, batch, iters=20):
    """Top Hessian eigenvalue by power iteration on Hessian-vector products (double backward)."""
    params = [p for p in model.parameters() if p.requires_grad]
    loss = loss_fn(model, batch)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    v = [torch.randn_like(p) for p in params]
    for _ in range(iters):
        norm = torch.sqrt(sum((x ** 2).sum() for x in v)); v = [x / norm for x in v]
        Hv = torch.autograd.grad(grads, params, grad_outputs=v, retain_graph=True)
        lam = sum((a * b).sum() for a, b in zip(Hv, v)).item(); v = [x.detach() for x in Hv]
    return lam

# log it every ~100 steps next to 2/lr (SGD) or 38/lr (Adam, β1=0.9, preconditioned)`,
        "Gradient noise scale": `def noise_scale(model, loss_fn, batches_small, batch_big):
    """B_simple = tr(Σ)/|G|², estimated from gradient norms at two batch sizes (McCandlish et al., App. A)."""
    def gnorm2(b):
        model.zero_grad(); loss_fn(model, b).backward()
        return sum((p.grad ** 2).sum() for p in model.parameters()).item()
    b_s, b_b = len(batches_small[0]), len(batch_big)
    g_s = sum(gnorm2(b) for b in batches_small) / len(batches_small); g_b = gnorm2(batch_big)
    G2 = (b_b * g_b - b_s * g_s) / (b_b - b_s)          # unbiased |G|²
    S = (g_s - g_b) / (1 / b_s - 1 / b_b)                # unbiased tr(Σ)
    return S / G2                                        # ≈ critical batch size`,
        "Bits per parameter probe": `# Allen-Zhu & Li's recipe, shrunk: N synthetic people, each with 6 attributes
people = [{"name": fake_name(), "birth": rand_date(), "city": rand_city(), "uni": rand_uni(),
           "major": rand_major(), "employer": rand_company()} for _ in range(N)]
# train a GPT on templated bios, each person seen ~1000 times with varied wording; then
bits = sum(log2(num_choices[a]) * accuracy(model, a) for a in attributes) * N   # knowledge stored
print(bits / num_params(model))          # ≈ 2 at saturation, across sizes`,
      }, "Project 22 (grokking) has the weight-decay sweep; Project 36 (scaling ladder) fits laws on real runs."),
    ),
  );
  cap(9.9); reset();
}
