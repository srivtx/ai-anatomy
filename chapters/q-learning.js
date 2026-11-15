import { h, rng, loop } from "../lib/core.js";
import { walk } from "../lib/walks.js";
import { slider, stats, predict, code, callout, section, prose, table, button } from "../lib/widgets.js";
import { liveChart } from "../lib/widgets.js";

const W = 9, H = 6;
const START = [0, 5], GOAL = [8, 0], PIT = [[4, 2], [5, 2], [6, 5]];
const WALLS0 = [[2, 1], [2, 2], [2, 3], [2, 4], [6, 1], [6, 2], [6, 3]];
const DIRS = [[0, -1], [1, 0], [0, 1], [-1, 0]], ARROW = ["↑", "→", "↓", "←"];
const key = (x, y) => y * W + x;

export default function (root, { id }) {
  const o = { alpha: 0.5, gamma: 0.95, eps: 0.2, slip: 0 };
  let walls = new Set(WALLS0.map(([x, y]) => key(x, y))), Q, pos, episodes, stepsThis, rets, r, ret;
  const isPit = (x, y) => PIT.some(([a, b]) => a === x && b === y), isGoal = (x, y) => x === GOAL[0] && y === GOAL[1];
  function reset() { Q = Array.from({ length: W * H }, () => new Float64Array(4)); pos = [...START]; episodes = 0; stepsThis = 0; rets = []; ret = 0; r = rng(7); chart.reset(); draw(); }
  function act(s) { return r() < o.eps ? r.int(4) : Q[s].indexOf(Math.max(...Q[s])); }
  function step() {
    const s = key(...pos);
    let a = act(s);
    if (r() < o.slip) a = r.int(4); // a slippery floor
    const nx = Math.min(W - 1, Math.max(0, pos[0] + DIRS[a][0])), ny = Math.min(H - 1, Math.max(0, pos[1] + DIRS[a][1]));
    const blocked = walls.has(key(nx, ny));
    const np = blocked ? pos : [nx, ny];
    const done = isGoal(...np) || isPit(...np);
    const R = isGoal(...np) ? 1 : isPit(...np) ? -1 : -0.01;
    const s2 = key(...np);
    // Q-learning: move Q(s,a) towards reward + γ · best value of the next state
    const target = R + (done ? 0 : o.gamma * Math.max(...Q[s2]));
    Q[s][a] += o.alpha * (target - Q[s][a]);
    ret += R; stepsThis++;
    pos = np;
    if (done || stepsThis > 200) { episodes++; rets.push(ret); chart.push([rets.slice(-20).reduce((p, x) => p + x, 0) / Math.min(20, rets.length)]); pos = [...START]; stepsThis = 0; ret = 0; }
  }
  const gridEl = h("div", { style: { display: "grid", gridTemplateColumns: `repeat(${W}, 1fr)`, gap: "3px", maxWidth: "620px" }, role: "img", "aria-label": "Gridworld: colour shows learned value, arrows show the greedy policy" });
  const st = stats([{ label: "Episodes" }, { label: "Average return (last 20)" }, { label: "Greedy path reaches the goal?" }]);
  const chart = liveChart({ series: [{ name: "Average return over the last 20 episodes", color: "var(--accent)" }], height: 110, points: 150, yMax: 1, fmt: (v) => v.toFixed(1) });
  function greedyReaches() {
    let p = [...START];
    for (let k = 0; k < 60; k++) { if (isGoal(...p)) return true; if (isPit(...p)) return false; const s = key(...p), a = Q[s].indexOf(Math.max(...Q[s])); const nx = Math.min(W - 1, Math.max(0, p[0] + DIRS[a][0])), ny = Math.min(H - 1, Math.max(0, p[1] + DIRS[a][1])); if (!walls.has(key(nx, ny))) p = [nx, ny]; }
    return isGoal(...p);
  }
  function draw() {
    const cells = [];
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const s = key(x, y), v = Math.max(...Q[s]), here = pos[0] === x && pos[1] === y;
      let bg = `color-mix(in srgb, ${v >= 0 ? "var(--good)" : "var(--bad)"} ${Math.round(Math.min(1, Math.abs(v)) * 70)}%, var(--surface))`, label = Q[s].some((q) => q !== 0) ? ARROW[Q[s].indexOf(v)] : "";
      if (walls.has(s)) { bg = "var(--ink)"; label = ""; }
      if (isGoal(x, y)) { bg = "var(--good)"; label = "+1"; }
      if (isPit(x, y)) { bg = "var(--bad)"; label = "−1"; }
      cells.push(h("button", { type: "button", "aria-label": `cell ${x},${y}`, style: { aspectRatio: "1", border: here ? "3px solid var(--accent)" : "1px solid var(--line)", borderRadius: "6px", background: bg, color: isGoal(x, y) || isPit(x, y) || walls.has(s) ? "var(--surface)" : "var(--ink)", font: "600 15px var(--mono)", cursor: "pointer", padding: 0 },
        onclick: () => { if (isGoal(x, y) || isPit(x, y) || (x === START[0] && y === START[1])) return; walls.has(s) ? walls.delete(s) : walls.add(s); draw(); } }, label));
    }
    gridEl.replaceChildren(...cells);
    st.set(0, String(episodes));
    const avg = rets.length ? rets.slice(-20).reduce((p, x) => p + x, 0) / Math.min(20, rets.length) : 0;
    st.set(1, rets.length ? avg.toFixed(2) : "–", avg > 0.7 ? "good" : avg < 0 ? "bad" : "");
    const ok = greedyReaches(); st.set(2, ok ? "yes" : "not yet", ok ? "good" : "");
  }
  const lp = loop(() => { for (let k = 0; k < o.speed; k++) step(); draw(); if (episodes > 2000) lp.stop(); }, gridEl);
  o.speed = 40;

  root.append(
    section(null, prose(`<p>Now give the agent a world with <b>states</b>. It starts bottom-left and gets +1 for reaching the green square, −1 for falling into a red one, and a tiny cost per step. It knows nothing about the map. <b>Q-learning</b> (Watkins, 1989) learns a table Q(s, a): the total future reward expected from taking action a in state s. The update is one line, and it's the most important line in RL:</p>
<p><code>Q(s, a) ← Q(s, a) + α · [ r + γ · max Q(s′, ·) − Q(s, a) ]</code></p>
<p>The bracket is the <b>temporal-difference error</b>: what just happened (reward plus the value of where you landed) minus what you predicted. Value spreads backwards from the goal one step at a time.</p>`)),
    section("Watch value flow backwards",
      prose("<p>Cell colour is the best Q-value there (green good, red bad) and arrows are the current greedy action. Click cells to add or remove walls, even mid-training.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Learning rate α", min: 0.05, max: 1, step: 0.05, value: o.alpha, fmt: (v) => v.toFixed(2), onInput: (v) => (o.alpha = v) }),
          slider({ label: "Discount γ", min: 0.5, max: 0.99, step: 0.01, value: o.gamma, fmt: (v) => v.toFixed(2), onInput: (v) => (o.gamma = v), hint: "How much the future counts" }),
          slider({ label: "Exploration ε", min: 0, max: 0.6, step: 0.02, value: o.eps, fmt: (v) => v.toFixed(2), onInput: (v) => (o.eps = v) }),
          slider({ label: "Slippery floor", min: 0, max: 0.4, step: 0.05, value: o.slip, fmt: (v) => `${Math.round(v * 100)}%`, onInput: (v) => (o.slip = v), hint: "Chance a move goes a random way" }),
          slider({ label: "Speed (steps per frame)", min: 1, max: 200, value: 40, onInput: (v) => (o.speed = v) }),
          button("Train", () => lp.start(), "primary"), button("Pause", () => lp.stop()), button("Reset", () => { lp.stop(); reset(); })),
        h("div.stage-body", st.el, gridEl, chart.el)),
      callout("tip", "Things to try", "<ul style='margin:0;padding-left:18px'><li>Run at speed 1 for the first episode: a random walk. Then watch the green spread backwards from the goal as episodes accumulate.</li><li>Set γ to 0.5: distant rewards barely count, so far-away cells stay grey and the agent can't find the goal from the start.</li><li>Make the floor 30% slippery: the learned path moves away from the pits. The agent has learned risk.</li></ul>"),
      predict({
        chapter: id,
        question: "Q-learning updates towards max Q(s′), the best action at the next state, even when the agent actually explores randomly there. What does that make it?",
        options: ["On-policy: it learns the value of what it does", "Off-policy: it learns the value of the greedy policy while behaving differently", "Model-based", "Supervised"],
        answer: 1,
        explain: "<p>Because the target uses the max, Q-learning learns about the greedy policy no matter how it explores. That's why DQN can learn from a replay buffer of old experience. SARSA uses the action actually taken next, so it learns the value of the exploring policy, and walks more carefully near cliffs.</p>",
      }),
    ),
    section("From a table to a network: DQN",
      prose("<p>Tables don't scale: an Atari screen has more states than atoms in the universe. In 2015 DeepMind's <b>DQN</b> replaced the table with a CNN that reads pixels and outputs a Q-value per action, and learned 49 Atari games from pixels alone (Mnih et al., 2015). Two tricks made it stable:</p>"),
      table(["Trick", "Problem it solves"], [
        ["Experience replay", "Consecutive frames are highly correlated; store transitions in a buffer and train on random batches"],
        ["Target network", "The target r + γ max Q(s′) moves every time Q changes; compute it with a frozen copy updated every few thousand steps"],
        ["Double DQN", "The max over noisy estimates is biased upwards; choose the action with one network, evaluate it with the other"],
        ["Dueling, prioritised replay, distributional RL", "Refinements combined in Rainbow (2017)"],
      ]),
      prose("<p>Value-based methods like DQN work well for small discrete action sets. With 100,000 possible tokens per step and sequences of thousands, LLM training uses <b>policy-gradient</b> methods instead (previous chapter, and PPO and GRPO next), though the idea of a learned value function survives as PPO's critic.</p>"),
    ),
    walk("q-learning"),
    section("In code",
      code({
        Python: `import numpy as np
Q = np.zeros((n_states, n_actions))
for episode in range(5000):
    s, done = env.reset(), False
    while not done:
        a = np.random.randint(n_actions) if np.random.rand() < eps else Q[s].argmax()
        s2, r, done = env.step(a)
        target = r + (0 if done else gamma * Q[s2].max())
        Q[s, a] += alpha * (target - Q[s, a])           # temporal-difference update
        s = s2`,
        "DQN loss (PyTorch)": `q = q_net(states).gather(1, actions[:, None]).squeeze(1)
with torch.no_grad():
    best_next = target_net(next_states).max(1).values
    target = rewards + gamma * best_next * (1 - dones)
loss = torch.nn.functional.smooth_l1_loss(q, target)`,
      }, "Project 24 does both: tabular Q-learning, then DQN on CartPole."),
    ),
  );
  reset();
}
