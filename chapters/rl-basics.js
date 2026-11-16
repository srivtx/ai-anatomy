import { h, rng, loop } from "../lib/core.js";
import { bellmanLab } from "../lib/labs.js";
import { goal, whyChain, sticker } from "../lib/why.js";
import { slider, stats, predict, code, callout, section, prose, table, button, toggle, bars } from "../lib/widgets.js";
import { xyChart } from "../lib/plot.js";
import { gauss, softmax } from "../lib/nn.js";

const MEANS = [0.2, 0.9, 0.5, 1.3, 0.7]; // arm 4 is best, but noisy rewards hide it

function makeAgents(o, seed) {
  const r = rng(seed), K = MEANS.length;
  const pull = (a) => MEANS[a] + gauss(r) * o.noise;
  const greedy = { name: "ε-greedy", color: "var(--c2)", Q: new Float64Array(K), N: new Float64Array(K), regret: 0, hist: [[0, 0]],
    act() { return r() < o.eps ? r.int(K) : this.Q.indexOf(Math.max(...this.Q)); },
    learn(a, R) { this.N[a]++; this.Q[a] += (R - this.Q[a]) / this.N[a]; } };
  const ucb = { name: "UCB", color: "var(--c1)", Q: new Float64Array(K), N: new Float64Array(K), t: 0, regret: 0, hist: [[0, 0]],
    act() { this.t++; for (let a = 0; a < K; a++) if (!this.N[a]) return a; let best = 0, bv = -Infinity; for (let a = 0; a < K; a++) { const v = this.Q[a] + o.c * Math.sqrt(Math.log(this.t) / this.N[a]); if (v > bv) { bv = v; best = a; } } return best; },
    learn(a, R) { this.N[a]++; this.Q[a] += (R - this.Q[a]) / this.N[a]; } };
  const pg = { name: "Policy gradient", color: "var(--c3)", H: new Float64Array(K), base: 0, n: 0, regret: 0, hist: [[0, 0]],
    get pi() { return softmax(this.H); },
    act() { const p = this.pi; let u = r(); for (let a = 0; a < K; a++) if ((u -= p[a]) <= 0) return a; return K - 1; },
    learn(a, R) { // REINFORCE: raise log-prob of the action in proportion to (reward − baseline)
      const p = this.pi, adv = R - (o.baseline ? this.base : 0);
      for (let b = 0; b < K; b++) this.H[b] += o.lr * adv * ((b === a ? 1 : 0) - p[b]);
      this.n++; this.base += (R - this.base) / this.n;
    } };
  return { agents: [greedy, ucb, pg], pull, best: Math.max(...MEANS) };
}

export default function (root, { id }) {
  const o = { eps: 0.1, c: 1, lr: 0.1, baseline: true, noise: 1 };
  let sim, t;
  const chart = xyChart({ height: 260, xLabel: "pulls", yLabel: "total regret (reward lost vs always picking the best arm)" });
  const st = stats([{ label: "Pulls" }, { label: "ε-greedy regret" }, { label: "UCB regret" }, { label: "Policy-gradient regret" }]);
  const pEl = h("div");
  function reset(seed = 1) { sim = makeAgents(o, seed); t = 0; draw(); }
  function step() {
    t++;
    for (const ag of sim.agents) { const a = ag.act(), R = sim.pull(a); ag.learn(a, R); ag.regret += sim.best - MEANS[a]; if (t % 5 === 0) ag.hist.push([t, ag.regret]); }
  }
  function draw() {
    chart.update(sim.agents.map((a) => ({ name: a.name, color: a.color, points: a.hist })), { xDomain: [0, Math.max(100, t)] });
    st.set(0, String(t)); sim.agents.forEach((a, i) => st.set(i + 1, a.regret.toFixed(1)));
    const pg = sim.agents[2];
    pEl.replaceChildren(h("div.lab",
      h("div", h("div.eyebrow", "True mean reward of each arm (hidden from the agents)"), bars(MEANS.map((m, i) => ({ label: `arm ${i + 1}`, value: m, color: "var(--ink-3)" })), { max: 1.5, fmt: (v) => v.toFixed(2) })),
      h("div", h("div.eyebrow", "Policy-gradient agent's probabilities"), bars([...pg.pi].map((p, i) => ({ label: `arm ${i + 1}`, value: p * 100, color: "var(--c3)" })), { max: 100, fmt: (v) => `${v.toFixed(0)}%` }))));
  }
  const lp = loop(() => { for (let k = 0; k < 6; k++) step(); draw(); if (t >= 2000) lp.stop(); }, chart.el);

  root.append(
    section(null, goal("<p>We want an agent that learns to act when nobody tells it the right move, only a <b>score, often much later</b>. It has to work out which of its earlier moves earned the reward.</p>", ["a way to try new things (explore)", "an estimate of how good each situation is", "credit passed back to earlier moves"]),
      prose(`<p>Supervised learning is told the right answer. Reinforcement learning is only told how <b>good</b> its answer was, often much later. An <b>agent</b> takes <b>actions</b> in an <b>environment</b>, receives <b>rewards</b>, and must learn a <b>policy</b> (what to do in each situation) that maximises total reward. Two problems appear immediately: <b>exploration</b> (you can't know an action is good without trying it) and <b>credit assignment</b> (which of your many actions earned that reward?).</p>
<p>The smallest RL problem has no states at all: a row of slot machines (“arms”) with unknown payouts. It's called the <b>multi-armed bandit</b>, and it already contains the exploration dilemma.</p>`),
      h("div.vx-row", sticker("maze", { title: "Find the flag", caption: "the agent (purple) only learns it did well when it reaches the goal", px: 4 }))),
    section("Three ways to play a bandit",
      prose("<p>Three agents play the same five arms side by side. <b>ε-greedy</b> usually picks the arm with the best average so far, but explores at random with probability ε. <b>UCB</b> adds an optimism bonus to arms it has tried rarely. <b>Policy gradient</b> keeps a probability for every arm and nudges up the probability of actions that did better than average. That last one is the seed of everything from PPO to GRPO.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "ε (exploration)", min: 0, max: 0.5, step: 0.01, value: o.eps, fmt: (v) => v.toFixed(2), onInput: (v) => (o.eps = v) }),
          slider({ label: "UCB bonus c", min: 0, max: 3, step: 0.1, value: o.c, fmt: (v) => v.toFixed(1), onInput: (v) => (o.c = v) }),
          slider({ label: "Policy-gradient step size", min: 0.01, max: 0.5, step: 0.01, value: o.lr, fmt: (v) => v.toFixed(2), onInput: (v) => (o.lr = v) }),
          slider({ label: "Reward noise", min: 0, max: 3, step: 0.1, value: o.noise, fmt: (v) => v.toFixed(1), onInput: (v) => (o.noise = v) }),
          toggle({ label: "Subtract a baseline (average reward)", checked: true, onChange: (v) => (o.baseline = v) }),
          button("Run 2,000 pulls", () => { reset(Math.floor(Math.random() * 1e6)); lp.start(); }, "primary")),
        h("div.stage-body", st.el, chart.el, pEl)),
      callout("tip", "Things to try", "<ul style='margin:0;padding-left:18px'><li>Set ε to 0: pure greed often locks onto whichever arm got lucky early. Its regret keeps rising in a straight line, forever.</li><li>UCB's regret curve bends over (it grows like log t): it explores just enough. That's provably near-optimal.</li><li>Turn the baseline off. Every reward is positive, so the policy gradient pushes up <i>whatever</i> it did, just more for better arms. Learning is far noisier and slower. Subtracting a baseline turns “reward” into “advantage”: better or worse than usual.</li></ul>"),
      predict({
        chapter: id,
        question: "Why does the policy-gradient update multiply by (1[a] − π(a)) instead of just 1[a]?",
        options: ["To keep probabilities positive", "It's the gradient of log π(a) for a softmax: it raises the taken action and lowers the others in proportion to their probability", "To add exploration", "It's a baseline"],
        answer: 1,
        explain: "<p>REINFORCE (Williams, 1992) follows ∇ log π(a) × (reward − baseline). For a softmax over preferences H, ∂ log π(a)/∂H_b = 1[a = b] − π(b). The same formula, with a neural network producing the logits and the reward coming from a reward model or a verifier, is the core of RLHF and GRPO.</p>",
      }),
    ),
    section("The vocabulary of RL",
      table(["Term", "Meaning", "In LLM training"], [
        ["State s", "What the agent observes", "The prompt plus the tokens generated so far"],
        ["Action a", "What it does", "The next token"],
        ["Policy π(a | s)", "Probability of each action", "The language model itself"],
        ["Reward r", "Scalar feedback", "A reward model's score, or 1 if the answer checks out"],
        ["Return G", "Sum of (discounted) future rewards", "Usually one reward at the end of the response"],
        ["Value V(s)", "Expected return from a state", "PPO's critic; GRPO estimates it from a group of samples"],
        ["Advantage A", "How much better an action was than expected", "What every policy-gradient method actually multiplies by"],
        ["Exploration", "Trying things that might be better", "Sampling temperature; diverse rollouts"],
      ]),
    ),
    section("Watch value flow back",
      prose("<p>Here is that backward flow, made visible. Each square's tower is its value V: how much reward the agent expects from there. Step the sweeps forward and watch value seep back from the goal through the corridors. Lower \u03b3 and the far end goes flat.</p>"),
      bellmanLab()),
    section("Why delayed rewards still teach",
      whyChain({
        claim: "An agent can learn from a reward that arrives 100 moves after the move that earned it.",
        steps: [
          { q: "How can it know which move mattered?", a: "<p>It learns a <b>value</b>: how much future reward to expect from each state. Then each move can be judged at once: did it land in a state worth more or less than expected? That surprise, r + γV(s′) − V(s), is the learning signal.</p>" },
          { q: "Why does that pass credit backwards?", a: "<p>Each update pulls V(s) toward the reward plus the value of the next state. The value at the goal spreads to the step before it, then the step before that, one update at a time, like water filling back along a path.</p>" },
          { q: "Why the discount γ?", a: "<p>It makes near rewards count more and keeps an endless sum finite. With γ = 0.99, a reward 100 steps away is worth 0.99¹⁰⁰ ≈ 0.37 of the same reward now.</p>" },
        ],
        axiom: { name: "Bellman's principle of optimality", html: "<p>Whatever the first step of an optimal plan, the rest of the plan must be optimal from where that step lands (Bellman, 1957). So the value of a state is the best immediate reward plus the discounted value of where you end up: V(s) = max over actions of [r + γ·V(s′)]. Every RL method estimates this equation one way or another.</p>" },
      })),
    section("In code",
      code({
        PyTorch: `import torch

logits = torch.zeros(5, requires_grad=True)          # the policy: preferences over 5 arms
opt = torch.optim.SGD([logits], lr=0.1)
baseline = 0.0
for t in range(1, 2001):
    dist = torch.distributions.Categorical(logits=logits)
    a = dist.sample()
    r = pull(a.item())                                # environment
    advantage = r - baseline
    loss = -dist.log_prob(a) * advantage              # REINFORCE
    opt.zero_grad(); loss.backward(); opt.step()
    baseline += (r - baseline) / t                    # running mean reward`,
      }, "Project 10 grows this into REINFORCE, then PPO, on CartPole."),
    ),
  );
  reset();
}
