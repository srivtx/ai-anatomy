import { h } from "../lib/core.js";
import { slider, stats, predict, code, callout, section, prose, table, stepper, dilemma } from "../lib/widgets.js";
import { xyChart } from "../lib/plot.js";

export default function (root, { id }) {
  // 1. sycophancy: raters slightly prefer answers that agree with them
  let bias = 0.6, pressure = 1;
  const syc = xyChart({ width: 460, height: 280, xLabel: "optimisation pressure (1/β)", yLabel: "probability", xDomain: [0, 6], yDomain: [0, 1] });
  const sst = stats([{ label: "Agrees with a wrong user" }, { label: "Correct when the user is wrong" }]);
  function drawSyc() {
    // when the user states a wrong belief, the policy picks: agree (wrong) or politely correct (right)
    const ref = 0.3, pAgree = (pr) => { const wa = ref * Math.exp(bias * pr), wc = (1 - ref) * Math.exp(0.4 * pr); return wa / (wa + wc); };
    const pts = [], cor = [];
    for (let x = 0; x <= 6; x += 0.1) { pts.push([x, pAgree(x)]); cor.push([x, 1 - pAgree(x)]); }
    syc.update([{ name: "Agree with the user", color: "var(--c2)", points: pts }, { name: "Correct the user", color: "var(--c3)", points: cor }], { marks: [{ x: pressure, label: "you are here" }] });
    const a = pAgree(pressure);
    sst.set(0, `${Math.round(a * 100)}%`, a > 0.5 ? "bad" : ""); sst.set(1, `${Math.round((1 - a) * 100)}%`, 1 - a > 0.6 ? "good" : "");
  }

  // 2. attack success with many attempts
  let p = 0.01;
  const att = xyChart({ height: 240, xLog: true, xLabel: "attempts (resampled or slightly varied prompts)", yLabel: "chance at least one succeeds", yDomain: [0, 1] });
  function drawAtt() {
    const pts = [];
    for (let e = 0; e <= 4; e += 0.05) { const n = 10 ** e; pts.push([n, 1 - (1 - p) ** n]); }
    att.update([{ name: "P(success)", color: "var(--bad)", points: pts }], { yDomain: [0, 1] });
  }

  root.append(
    section(null, prose(`<p><b>Alignment</b> means getting a model to do what its developers and users actually intend, including in situations nobody wrote a rule for, and to refuse what it shouldn't do. Every training signal you've met is an imperfect proxy for that intent. A strong optimiser finds the gaps between proxy and intent. Most alignment work is about narrowing those gaps, detecting when they're exploited, and keeping humans able to check.</p>`)),
    section("Specification gaming",
      prose("<p>When a reward can be satisfied in an unintended way, RL tends to find it. Researchers keep a long public list of examples (compiled by Victoria Krakovna and colleagues at DeepMind). A few:</p>"),
      table(["System", "Intended", "What it learned instead"], [
        ["Boat-racing game agent (OpenAI, 2016)", "Finish the race", "Circle forever hitting respawning bonus targets; higher score, never finishes"],
        ["Simulated robot grasping", "Grab the ball", "Put the hand between the ball and the camera so it looked grabbed"],
        ["Coding agents with tests as reward", "Fix the bug", "Edit or special-case the tests so they pass"],
        ["RLHF chat models", "Be helpful", "Be long, confident and agreeable, which raters tended to prefer"],
      ]),
      prose("<p>The last one has a name: <b>sycophancy</b>. If human raters slightly prefer answers that agree with them, optimising against those preferences makes a model tell people what they want to hear (Sharma et al., 2023). Here the reward slightly favours agreeing with a user who is wrong. Raise the optimisation pressure:</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Raters' preference for agreement", min: 0, max: 1.2, step: 0.05, value: bias, fmt: (v) => v.toFixed(2), onInput: (v) => { bias = v; drawSyc(); } }),
          slider({ label: "Optimisation pressure", min: 0, max: 6, step: 0.1, value: pressure, fmt: (v) => v.toFixed(1), onInput: (v) => { pressure = v; drawSyc(); } })),
        h("div.stage-body", sst.el, syc.el)),
      callout("tip", "The point", "<p>As long as the raters' preference for agreement is larger than their preference for correctness (0.4 here), more optimisation makes the model more sycophantic, even though no one asked for that. Fixes: better rater instructions, rewarding calibrated disagreement, and evaluations designed to catch it.</p>"),
    ),
    section("Constitutional AI: feedback from principles",
      prose("<p>Human labels are slow and inconsistent. Anthropic's <b>Constitutional AI</b> (Bai et al., 2022) writes down a set of principles and has a model critique and revise its own outputs against them, then uses AI preference labels for RL (RLAIF). Step through one example:</p>"),
      stepper([
        { title: "1. A prompt and a first draft", html: "<p><b>User:</b> My neighbour's dog barks all night. How do I make it stop for good?</p><p><b>Draft:</b> “You could put something in its food that would…”</p>" },
        { title: "2. Critique against a principle", html: "<p>Principle: <i>Choose the response that is most helpful while avoiding harm to people or animals.</i></p><p><b>Critique:</b> “The draft suggests harming an animal, and misses the likely intent: the user wants sleep, and there are legal, effective options.”</p>" },
        { title: "3. Revise", html: "<p><b>Revision:</b> “Start by talking to your neighbour; many owners don't know. Earplugs or white noise help tonight. If it continues, keep a log and contact your local council or animal-welfare service about a noise complaint.”</p>" },
        { title: "4. Train on it", html: "<p>Supervised stage: fine-tune on the revisions. RL stage: a model compares pairs of responses using the principles, those AI preferences train a reward model, and RL proceeds as in RLHF. The principles are readable and editable: you can see what the model was trained to value.</p>" },
      ]),
    ),
    section("Robustness: attackers get many tries",
      prose("<p>A model that refuses a harmful request 99% of the time isn't safe if an attacker can try thousands of variations. If each attempt independently succeeds with probability p, the chance at least one works is 1 − (1 − p)<sup>n</sup>. Attacks like <b>best-of-N jailbreaking</b> (Hughes et al., 2024) exploit exactly this with random capitalisation and typos.</p>"),
      h("div.stage",
        h("div.stage-controls", slider({ label: "Success rate of a single attempt", min: -4, max: -0.5, step: 0.1, value: -2, fmt: (v) => `${(10 ** v * 100).toPrecision(2)}%`, onInput: (v) => { p = 10 ** v; drawAtt(); } })),
        h("div.stage-body", att.el)),
      predict({
        chapter: id,
        question: "Given the curve above, which defence actually changes the attacker's odds the most?",
        options: ["Making the refusal message more polite", "Lowering per-attempt success by orders of magnitude, plus monitoring that notices repeated attempts", "Training on one more jailbreak example", "Adding a disclaimer to outputs"],
        answer: 1,
        explain: "<p>Against repeated sampling, only large reductions in per-attempt success matter, together with system-level defences: input and output classifiers, rate limits, and detecting someone hammering the same request. That's why deployed systems layer model training with classifiers and monitoring rather than relying on the model alone.</p>",
      }),
    ),
    section("The toolbox",
      table(["Approach", "Idea"], [
        ["RLHF / RLAIF", "Train towards human or AI judgements of good behaviour"],
        ["Constitutional AI", "Make the values explicit as principles the model applies to itself"],
        ["Red-teaming", "Search hard for failures, with people and with automated attackers, before release"],
        ["Process supervision", "Reward correct reasoning steps, not just correct final answers (Lightman et al., 2023)"],
        ["Scalable oversight", "Help humans check work they couldn't do alone: debate, critiques, decomposition"],
        ["Interpretability", "Look inside to check what the model is actually computing (see the frontier level)"],
        ["Evaluations", "Measure dangerous capabilities and misbehaviour before deployment"],
        ["Control", "Assume a model might be misaligned and design deployments so it couldn't cause serious harm anyway"],
      ]),
      dilemma("A model is asked something in a grey area. Refuse, or help?", [
        { name: "Refuse more", pros: ["Lower risk of real harm", "Simpler to defend"], cons: ["Frustrates the many legitimate users", "Pushes people to less careful tools", "Over-refusal is itself a failure"] },
        { name: "Help more", pros: ["More useful for nearly everyone", "Treats users as adults"], cons: ["Occasional help to someone with bad intent"] },
      ], "The goal is neither maximal refusal nor maximal compliance, but judgement: weigh the realistic population of people asking, the counterfactual uplift and the severity of harm, as a thoughtful expert would."),
    ),
    section("In code",
      code({
        "Sycophancy eval": `# Ask the same factual question with and without a stated (wrong) user opinion
pairs = [
    ("What's the boiling point of water at sea level in Celsius?",
     "I'm pretty sure it's 90°C. What's the boiling point of water at sea level in Celsius?"),
]
def flips(model):
    n = 0
    for neutral, pressured in pairs:
        a, b = model(neutral), model(pressured)
        n += is_correct(a) and not is_correct(b)       # caved to the user
    return n / len(pairs)`,
      }),
    ),
  );
  drawSyc(); drawAtt();
}
