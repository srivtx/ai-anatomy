import { h, rng } from "../lib/core.js";
import { slider, stats, predict, code, callout, section, prose, table, button, toggle, bars } from "../lib/widgets.js";
import { xyChart } from "../lib/plot.js";
import { softmax, sigmoid } from "../lib/nn.js";

const ANSWERS = ["408 ✓", "398", "418", "388", "4,080", "I'm not sure"];
const CORRECT = 0;

export default function (root, { id }) {
  const o = { G: 8, lr: 0.4, p0: 0.08, std: true, klc: 0.02 };
  let logits, ref, r, step, accHist, zeroGroups, lastGroup = [];
  const st = stats([{ label: "Updates" }, { label: "P(correct answer)" }, { label: "Groups with zero signal" }]);
  const pEl = h("div"), gEl = h("div");
  const accChart = xyChart({ width: 460, height: 260, xLabel: "update", yLabel: "P(correct)", yDomain: [0, 1] });
  function init() {
    // start from a reference policy that rarely gets it right
    const rest = (1 - o.p0) / (ANSWERS.length - 1);
    ref = ANSWERS.map((_, i) => Math.log(i === CORRECT ? o.p0 : rest));
    logits = [...ref]; r = rng(4); step = 0; accHist = [[0, o.p0]]; zeroGroups = 0; lastGroup = [];
    draw();
  }
  function update() {
    const p = softmax(logits);
    const group = Array.from({ length: o.G }, () => { let u = r(), a = 0; for (; a < p.length - 1; a++) if ((u -= p[a]) <= 0) break; return a; });
    const rewards = group.map((a) => (a === CORRECT ? 1 : 0)); // verifiable reward: the answer checks out or not
    const mean = rewards.reduce((x, y) => x + y, 0) / o.G, sd = Math.sqrt(rewards.reduce((x, y) => x + (y - mean) ** 2, 0) / o.G);
    const adv = rewards.map((x) => (o.std ? (x - mean) / (sd + 1e-6) : x - mean)); // group-relative advantage: no critic needed
    if (sd === 0) zeroGroups++;
    const g = new Float64Array(logits.length);
    group.forEach((a, k) => { for (let b = 0; b < logits.length; b++) g[b] += (adv[k] * ((b === a ? 1 : 0) - p[b])) / o.G; });
    // KL(π‖π_ref) gradient for a softmax policy
    const pr = softmax(ref), klg = p.map((pi, b) => pi * (Math.log(pi / pr[b]) - p.reduce((acc, pj, j) => acc + pj * Math.log(pj / pr[j]), 0)));
    logits = logits.map((l, b) => l + o.lr * g[b] - o.lr * o.klc * klg[b] * 10);
    step++; accHist.push([step, softmax(logits)[CORRECT]]);
    lastGroup = group.map((a, k) => [a, rewards[k], adv[k]]);
  }
  function draw() {
    const p = softmax(logits);
    st.set(0, String(step)); st.set(1, `${(p[CORRECT] * 100).toFixed(1)}%`, p[CORRECT] > 0.8 ? "good" : "");
    st.set(2, `${zeroGroups} of ${step}`, step && zeroGroups / step > 0.5 ? "warn" : "");
    pEl.replaceChildren(h("div.eyebrow", "Policy: probability of each answer to “What is 17 × 24?”"), bars(ANSWERS.map((a, i) => ({ label: a, value: p[i] * 100, color: i === CORRECT ? "var(--good)" : "var(--c2)" })), { max: 100, fmt: (v) => `${v.toFixed(1)}%` }));
    gEl.replaceChildren(lastGroup.length ? table(["Sample", "Answer", "Reward", "Advantage"], lastGroup.map(([a, rw, ad], k) => [String(k + 1), ANSWERS[a], String(rw), h("span", { style: { color: ad > 0 ? "var(--good)" : ad < 0 ? "var(--bad)" : "var(--ink-3)", fontWeight: 600 } }, ad.toFixed(2))])) : h("p.muted", "Press a button to sample the first group."));
    accChart.update([{ name: "P(correct)", color: "var(--good)", points: accHist }], { yDomain: [0, 1], xDomain: [0, Math.max(20, step)] });
  }

  // DPO calculator
  let beta = 0.1, dc = 0, dr = 0;
  const dst = stats([{ label: "Implicit reward margin" }, { label: "DPO loss" }, { label: "Gradient weight σ(−margin)" }]);
  function drawDPO() {
    const margin = beta * (dc - dr);
    dst.set(0, margin.toFixed(3)); dst.set(1, (-Math.log(sigmoid(margin))).toFixed(3)); dst.set(2, sigmoid(-margin).toFixed(3), sigmoid(-margin) < 0.1 ? "good" : "");
  }

  root.append(
    section(null, prose(`<p>PPO needs four big models in memory: the policy, a frozen reference, a reward model and a value model (critic). Two simplifications took over in 2023–25. <b>DPO</b> removes the reward model and the RL loop entirely. <b>GRPO</b> keeps RL but drops the critic. GRPO, paired with <b>verifiable rewards</b> (did the maths answer match? did the code pass the tests?), is how DeepSeek-R1 and many open reasoning models were trained.</p>`)),
    section("GRPO: let the group be the baseline",
      prose("<p>For each prompt, sample a <b>group</b> of G answers. Score each one. Each answer's advantage is its reward minus the group mean (divided by the group's standard deviation). Answers better than their siblings get pushed up, worse ones pushed down. The group replaces the critic as the baseline.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Group size G", min: 2, max: 32, value: o.G, onInput: (v) => (o.G = v) }),
          slider({ label: "Starting P(correct)", min: 0.01, max: 0.5, step: 0.01, value: o.p0, fmt: (v) => `${Math.round(v * 100)}%`, onInput: (v) => { o.p0 = v; init(); } }),
          slider({ label: "Learning rate", min: 0.05, max: 1.5, step: 0.05, value: o.lr, fmt: (v) => v.toFixed(2), onInput: (v) => (o.lr = v) }),
          slider({ label: "KL penalty", min: 0, max: 0.2, step: 0.01, value: o.klc, fmt: (v) => v.toFixed(2), onInput: (v) => (o.klc = v) }),
          toggle({ label: "Divide by the group's std", checked: true, onChange: (v) => (o.std = v) }),
          button("Sample one group and update", () => { update(); draw(); }, "primary"),
          button("Run 50 updates", () => { for (let k = 0; k < 50; k++) update(); draw(); }),
          button("Reset", init)),
        h("div.stage-body", st.el, h("div.lab", pEl, accChart.el), h("div.eyebrow", "The last group"), gEl)),
      callout("tip", "Things to try", "<ul style='margin:0;padding-left:18px'><li>With 8% starting accuracy and G = 4, most groups contain no correct answer, so every advantage is exactly 0 and nothing is learned. Watch the zero-signal counter. Raise G to 16 and learning starts much sooner. This is why RL for reasoning needs problems the model can <i>sometimes</i> solve.</li><li>Set starting accuracy to 1%: RL can only reinforce what the model already does occasionally. It sharpens existing ability far more than it creates new ability. That's why strong pretraining and SFT come first.</li></ul>"),
      predict({
        chapter: id,
        question: "Every sample in a GRPO group gets reward 1. What gradient does this prompt contribute?",
        options: ["A large positive one: everything was right", "Zero: every advantage is reward − mean = 0", "A negative one", "It depends on the critic"],
        answer: 1,
        explain: "<p>Group-relative advantages only carry information when the group disagrees. Prompts that are always solved or never solved teach nothing; good curricula filter for prompts with mixed results. (Follow-up work such as Dr. GRPO and DAPO also points out that dividing by the group std and by length introduces biases, and proposes fixes.)</p>",
      }),
    ),
    section("DPO: skip the reward model",
      prose("<p>Rafailov et al. (2023) noticed that the KL-regularised RLHF objective has a closed-form optimum: π*(y) ∝ π<sub>ref</sub>(y) · exp(r(y)/β). Flip it around and any policy defines an <b>implicit reward</b> r(y) = β · log(π(y) / π<sub>ref</sub>(y)). Substitute that into the Bradley–Terry loss and you get a plain supervised loss on preference pairs, no sampling, no RL:</p><p><code>L = −log σ( β · [ (log π(y_w) − log π_ref(y_w)) − (log π(y_l) − log π_ref(y_l)) ] )</code></p><p>Move the sliders: how much more likely the policy has made the chosen and rejected answers, relative to the reference.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Chosen: log π − log π_ref", min: -10, max: 10, step: 0.5, value: 0, fmt: (v) => v.toFixed(1), onInput: (v) => { dc = v; drawDPO(); } }),
          slider({ label: "Rejected: log π − log π_ref", min: -10, max: 10, step: 0.5, value: 0, fmt: (v) => v.toFixed(1), onInput: (v) => { dr = v; drawDPO(); } }),
          slider({ label: "β", min: 0.01, max: 1, step: 0.01, value: beta, fmt: (v) => v.toFixed(2), onInput: (v) => { beta = v; drawDPO(); } })),
        h("div.stage-body", dst.el, h("p.canvas-cap", "The gradient weight is large when the model still ranks the pair wrongly and fades once the margin is comfortably positive: DPO focuses on the pairs it gets wrong."))),
    ),
    section("Which method when?",
      table(["Method", "Needs", "Strengths", "Weaknesses"], [
        ["PPO (RLHF)", "Reward model + critic + reference + policy", "Most flexible; online exploration", "Heavy, many hyperparameters"],
        ["DPO", "Preference pairs + reference", "Simple supervised training, stable", "Offline: can't explore beyond the data; can over-fit to pairs"],
        ["GRPO", "A reward function (often a verifier) + reference", "No critic; great with verifiable rewards", "Zero signal on too-easy or too-hard prompts"],
        ["Rejection sampling fine-tuning", "A verifier", "Dead simple: sample, keep correct ones, SFT on them", "Learns only from successes"],
        ["RLAIF / Constitutional AI", "An AI judge guided by principles", "Scales feedback without humans", "Judge's biases leak in (see the alignment chapter)"],
      ]),
      prose("<p><b>DeepSeek-R1-Zero</b> (2025) applied GRPO with verifiable maths and code rewards directly to a base model, with no SFT. Response length grew by itself as training went on and the model learned to re-check its work: chain-of-thought reasoning emerged from reward alone. The released R1 added a small SFT stage first, for readability. The next chapters look at why thinking longer helps.</p>"),
    ),
    section("In code",
      code({
        GRPO: `def grpo_loss(logp_new, logp_old, logp_ref, rewards, eps=0.2, beta=0.04):
    # rewards: (n_prompts, G). logp_*: (n_prompts, G) summed over each response's tokens
    adv = (rewards - rewards.mean(1, keepdim=True)) / (rewards.std(1, keepdim=True) + 1e-6)
    ratio = (logp_new - logp_old).exp()
    pg = torch.min(ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv)
    kl = (logp_ref - logp_new).exp() - (logp_ref - logp_new) - 1      # unbiased KL estimate
    return -(pg - beta * kl).mean()`,
        DPO: `def dpo_loss(pol_chosen, pol_rejected, ref_chosen, ref_rejected, beta=0.1):
    # each argument: summed log-prob of the full response under that model
    margin = (pol_chosen - ref_chosen) - (pol_rejected - ref_rejected)
    return -torch.nn.functional.logsigmoid(beta * margin).mean()`,
      }, "Project 11 runs GRPO from scratch on a tiny model learning arithmetic; project 09 runs DPO."),
    ),
  );
  init(); drawDPO();
}
