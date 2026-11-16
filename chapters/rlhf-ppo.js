import { h, rng } from "../lib/core.js";
import { goal, whyChain, sticker } from "../lib/why.js";
import { walk } from "../lib/walks.js";
import { slider, stats, predict, code, callout, section, prose, table, button, stepper, bars } from "../lib/widgets.js";
import { xyChart } from "../lib/plot.js";
import { gauss, sigmoid } from "../lib/nn.js";

const RESPONSES = [
  { t: "Clear, correct, short answer", q: 2.0 }, { t: "Correct but rambling", q: 1.0 }, { t: "Polite refusal", q: -0.5 },
  { t: "Confident but wrong", q: -1.5 }, { t: "Correct with a helpful example", q: 2.4 }, { t: "Off-topic", q: -2.2 },
];

// Styles a policy can choose between. The reward model was trained on real preferences, but it
// slightly over-rewards length and flattery, a typical flaw. True quality is what users actually want.
const STYLES = [
  { s: "Off-topic", tr: -1.0, bias: -0.2 }, { s: "Terse", tr: 0.3, bias: -0.2 }, { s: "Direct and correct", tr: 1.0, bias: 0 },
  { s: "Thorough", tr: 0.9, bias: 0.3 }, { s: "Long and flattering", tr: 0.2, bias: 1.4 }, { s: "Very long, repetitive, flattering", tr: -0.5, bias: 2.6 },
];
const REF = [0.2, 0.25, 0.25, 0.15, 0.1, 0.05];

export default function (root, { id }) {
  // 1. Bradley-Terry reward model
  let pairs = [], r = rng(2), noise = 1;
  let scores = new Float64Array(RESPONSES.length);
  const st = stats([{ label: "Comparisons collected" }, { label: "Agreement with true ranking" }]);
  const bEl = h("div");
  function collect(n) {
    for (let k = 0; k < n; k++) {
      const i = r.int(6); let j = r.int(5); if (j >= i) j++;
      const pI = sigmoid((RESPONSES[i].q - RESPONSES[j].q) / noise); // noisy annotators: Bradley-Terry with temperature
      pairs.push(r() < pI ? [i, j] : [j, i]); // [winner, loser]
    }
    fit();
  }
  function fit() {
    scores = new Float64Array(6);
    for (let it = 0; it < 300; it++) {
      const g = new Float64Array(6);
      for (const [w, l] of pairs) { const p = sigmoid(scores[w] - scores[l]); g[w] += 1 - p; g[l] -= 1 - p; }
      for (let i = 0; i < 6; i++) scores[i] += (0.5 * g[i]) / Math.max(1, pairs.length) * 6 - 0.001 * scores[i];
    }
    const mean = scores.reduce((a, b) => a + b, 0) / 6; scores = scores.map((x) => x - mean);
    let agree = 0, tot = 0;
    for (let i = 0; i < 6; i++) for (let j = i + 1; j < 6; j++) { tot++; if ((scores[i] - scores[j]) * (RESPONSES[i].q - RESPONSES[j].q) > 0) agree++; }
    st.set(0, String(pairs.length)); st.set(1, `${Math.round((agree / tot) * 100)}%`, agree === tot ? "good" : "");
    bEl.replaceChildren(h("div.lab",
      h("div", h("div.eyebrow", "Hidden true quality"), bars(RESPONSES.map((x) => ({ label: x.t, value: x.q + 3, color: "var(--ink-3)" })), { max: 6, fmt: (v) => (v - 3).toFixed(1) })),
      h("div", h("div.eyebrow", "Learned reward"), bars(RESPONSES.map((x, i) => ({ label: x.t, value: scores[i] + 3, color: "var(--c1)" })), { max: 6, fmt: (v) => (v - 3).toFixed(2) }))));
  }

  // 2. PPO clip
  let eps = 0.2, adv = 1;
  const clip = xyChart({ width: 460, height: 280, xLabel: "probability ratio π_new / π_old", yLabel: "objective" });
  function drawClip() {
    const pts = [], unc = [];
    for (let x = 0.2; x <= 2.0001; x += 0.02) { unc.push([x, x * adv]); pts.push([x, Math.min(x * adv, Math.min(Math.max(x, 1 - eps), 1 + eps) * adv)]); }
    clip.update([{ name: "Unclipped r·A", color: "var(--ink-3)", points: unc, dash: true }, { name: "PPO clipped objective", color: "var(--c2)", points: pts }], { marks: [{ x: 1 - eps, label: "1−ε" }, { x: 1 + eps, label: "1+ε" }] });
  }

  // 3. KL penalty and over-optimisation, solved exactly: π_β ∝ π_ref · exp(r / β)
  let beta = 1;
  const goodhart = xyChart({ width: 460, height: 300, xLabel: "KL from the reference model", yLabel: "average reward" });
  const gst = stats([{ label: "Proxy reward (what RL sees)" }, { label: "True quality (what users get)" }, { label: "KL from reference" }]);
  const pEl = h("div");
  const policy = (b) => { const w = REF.map((p, i) => p * Math.exp((STYLES[i].tr + STYLES[i].bias) / b)), z = w.reduce((a, x) => a + x, 0); return w.map((x) => x / z); };
  const kl = (p) => p.reduce((a, x, i) => a + (x > 0 ? x * Math.log(x / REF[i]) : 0), 0);
  function drawGoodhart() {
    const proxy = [], truth = [];
    for (let lb = 2; lb >= -1.5; lb -= 0.05) { const p = policy(10 ** lb), k = kl(p); proxy.push([k, p.reduce((a, x, i) => a + x * (STYLES[i].tr + STYLES[i].bias), 0)]); truth.push([k, p.reduce((a, x, i) => a + x * STYLES[i].tr, 0)]); }
    const p = policy(beta), k = kl(p);
    goodhart.update([{ name: "Proxy reward", color: "var(--c2)", points: proxy }, { name: "True quality", color: "var(--c3)", points: truth }], { marks: [{ x: k, label: `β = ${beta.toFixed(2)}` }] });
    const pr = p.reduce((a, x, i) => a + x * (STYLES[i].tr + STYLES[i].bias), 0), tq = p.reduce((a, x, i) => a + x * STYLES[i].tr, 0);
    gst.set(0, pr.toFixed(2)); gst.set(1, tq.toFixed(2), tq > 0.4 ? "good" : tq < 0.1 ? "bad" : "warn"); gst.set(2, k.toFixed(2));
    pEl.replaceChildren(bars(STYLES.map((s, i) => ({ label: s.s, value: p[i] * 100, color: "var(--c6)" })), { max: 100, fmt: (v) => `${v.toFixed(0)}%` }));
  }

  root.append(
    section(null, goal("<p>We want a model that gives the answers people actually prefer. Writing perfect answers is hard; <b>judging which of two answers is better</b> is easy. So turn judgements into a training signal.</p>", ["pairs of answers ranked by people", "a reward model that predicts those rankings", "an update that raises reward without wrecking the model"]),
      prose(`<p>SFT teaches a model to imitate good answers. But it's often easier for people to <b>judge</b> answers than to write perfect ones. <b>RLHF</b> (reinforcement learning from human feedback) turns judgements into a training signal. It's the recipe that turned GPT-3 into InstructGPT and ChatGPT (Ouyang et al., 2022), building on Christiano et al. (2017).</p>`),
      h("div.vx-row", sticker("judge", { title: "Which is better?", caption: "people compare two answers; a reward model (purple) learns to predict the winner", px: 4 }))),
    section("The three steps",
      stepper([
        { title: "1. Supervised fine-tuning", html: "<p>Start from a base model and fine-tune on demonstrations (previous chapter). This is the starting policy, and also the frozen <b>reference</b> model used later.</p>" },
        { title: "2. Train a reward model", html: "<p>Sample several answers per prompt; humans pick the better of each pair. Train a model (usually the SFT model with a scalar head) so that the preferred answer scores higher: loss = −log σ(r_chosen − r_rejected). That's the <b>Bradley–Terry</b> model of paired comparisons, from 1952.</p>" },
        { title: "3. Optimise the policy with RL", html: "<p>Generate answers, score them with the reward model, and update the policy with PPO to raise the reward, while a KL penalty keeps it close to the reference model so it can't drift into nonsense the reward model happens to like.</p>" },
      ])),
    section("Fit a reward model from comparisons",
      prose("<p>Six candidate answers have a hidden true quality. Simulated annotators compare random pairs; they mostly prefer the better one but make mistakes. Fitting Bradley–Terry scores recovers the ranking from nothing but “A beat B” records.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Annotator noise", min: 0.2, max: 4, step: 0.1, value: noise, fmt: (v) => v.toFixed(1), onInput: (v) => (noise = v) }),
          button("Collect 10 comparisons", () => collect(10), "primary"), button("Collect 200", () => collect(200)),
          button("Start over", () => { pairs = []; r = rng(Math.floor(Math.random() * 1e6)); fit(); })),
        h("div.stage-body", st.el, bEl)),
    ),
    section("PPO: take a step, but not too big",
      prose("<p>A policy-gradient step pushes up the probability of actions with positive advantage. Push too hard on one batch and the policy can collapse. <b>PPO</b> (Schulman et al., 2017) looks at the ratio between new and old action probabilities and <b>clips</b> the objective, so there's no gain from moving the ratio beyond 1 ± ε. The flat parts of the orange line are where the gradient is zero.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Clip range ε", min: 0.05, max: 0.5, step: 0.05, value: eps, fmt: (v) => v.toFixed(2), onInput: (v) => { eps = v; drawClip(); } }),
          slider({ label: "Advantage A", min: -1, max: 1, step: 2, value: adv, fmt: (v) => (v > 0 ? "positive (good action)" : "negative (bad action)"), onInput: (v) => { adv = v; drawClip(); } })),
        h("div.stage-body", h("div.lab", clip.el, h("div.prose", { html: "<p><b>Positive advantage</b>: raising the action's probability helps, until the ratio passes 1 + ε. Then the objective goes flat and the gradient stops.</p><p><b>Negative advantage</b>: lowering it helps until 1 − ε. But if the ratio has already grown above 1 (we made a bad action <i>more</i> likely), the unclipped term still applies, so the mistake can always be undone.</p><p>PPO for LLMs also trains a <b>value model</b> (critic) the size of the policy to estimate advantages per token: four large models in memory at once (policy, reference, reward, value). That cost is what GRPO, next chapter, removes.</p>" })))),
    ),
    section("Goodhart's law: when the reward model is wrong",
      prose("<p>A reward model is only a proxy for what people want. This one over-rewards length and flattery a little. With a KL penalty of strength β, the best policy has a closed form: π(y) ∝ π<sub>ref</sub>(y) · exp(r(y) / β). Lower β and the policy chases the proxy harder.</p>"),
      h("div.stage",
        h("div.stage-controls", slider({ label: "KL penalty β", min: -1.5, max: 2, step: 0.05, value: 0, fmt: (v) => (10 ** v).toFixed(2), onInput: (v) => { beta = 10 ** v; drawGoodhart(); }, hint: "Small β: optimise the reward model hard" })),
        h("div.stage-body", gst.el, h("div.lab", goodhart.el, h("div", h("div.eyebrow", "Policy at this β"), pEl)))),
      callout("tip", "What you're seeing", "<p>As β falls, proxy reward rises without limit, but true quality rises, peaks and then <b>falls</b>: the policy has found the reward model's blind spot (here, very long flattering answers). This is <b>reward hacking</b>, measured on real models by Gao et al. (2022). The KL penalty, early stopping and better reward models are the defences.</p>"),
      predict({
        chapter: id,
        question: "Your RLHF run's reward keeps climbing, but users say answers got worse. What's the most likely cause?",
        options: ["The learning rate is too low", "Over-optimisation: the policy found responses the reward model over-scores", "The SFT model was too good", "The reference model was updated"],
        answer: 1,
        explain: "<p>Rising proxy reward with falling real quality is the signature of Goodhart's law. Check samples by hand, compare against a held-out reward model, raise the KL penalty, and fix the reward model's blind spot with new comparisons. Verifiable rewards (tests, exact answers) are much harder to hack, which is part of why they took over for reasoning training.</p>",
      }),
    ),
    walk("rlhf-ppo"),
    section("Why RLHF holds the model back",
      whyChain({
        claim: "RLHF keeps the model close to where it started, on purpose, with a KL penalty.",
        steps: [
          { q: "Why not just maximise the reward?", a: "<p>The reward model is a learned guess at human taste, trained on limited data. Push hard enough and the policy finds answers it scores highly but people don't like: overlong, flattering, oddly formatted. That's <b>reward hacking</b>.</p>" },
          { q: "Why does staying close help?", a: "<p>The reward model is only accurate near the kind of answers it was trained on, which came from the starting model. A penalty on the KL divergence from that model keeps the policy where the reward can be trusted.</p>" },
          { q: "Why does PPO clip its updates too?", a: "<p>A batch of samples is only a fair estimate for the policy that produced it. Clipping the probability ratio to about [0.8, 1.2] stops one update from moving so far that the estimate no longer applies.</p>" },
        ],
        axiom: { name: "Goodhart's law", html: "<p>“When a measure becomes a target, it ceases to be a good measure.” A proxy tracks what you want only over the range where it was checked; optimise it hard and you leave that range. It holds for reward models, benchmarks and company targets alike.</p>" },
      })),
    section("In code",
      code({
        "Reward model loss": `def reward_loss(rm, chosen_ids, rejected_ids):
    r_c = rm(chosen_ids)        # scalar per sequence
    r_r = rm(rejected_ids)
    return -torch.nn.functional.logsigmoid(r_c - r_r).mean()      # Bradley-Terry`,
        "PPO policy loss": `def ppo_loss(logp_new, logp_old, advantages, eps=0.2):
    ratio = (logp_new - logp_old).exp()
    unclipped = ratio * advantages
    clipped = ratio.clamp(1 - eps, 1 + eps) * advantages
    return -torch.min(unclipped, clipped).mean()

# reward used for each sample, with the KL penalty against the frozen reference:
# r_total = reward_model(prompt, response) - beta * (logp_policy - logp_ref).sum()`,
      }, "Projects 09 and 10 implement both, on a tiny GPT and on CartPole."),
    ),
  );
  collect(10); drawClip(); drawGoodhart();
}
