import { h, rng, loop } from "../lib/core.js";
import { segmented, stats, predict, code, callout, section, prose, table, button, stepper } from "../lib/widgets.js";
import { plane, xyChart } from "../lib/plot.js";
import { MLP, gauss } from "../lib/nn.js";

// A puck in a box with a round obstacle. state = [x, y, vx, vy], action = [ax, ay] in [-1, 1]
const OB = [0.1, 0.05], OR = 0.32;
function envStep(s, a) {
  let [x, y, vx, vy] = s;
  vx = 0.92 * vx + 0.012 * a[0]; vy = 0.92 * vy + 0.012 * a[1];
  x += vx; y += vy;
  if (Math.abs(x) > 1) { x = Math.sign(x); vx = -0.5 * vx; }
  if (Math.abs(y) > 1) { y = Math.sign(y); vy = -0.5 * vy; }
  const dx = x - OB[0], dy = y - OB[1], d = Math.hypot(dx, dy);
  if (d < OR) { const nx = dx / d, ny = dy / d; x = OB[0] + nx * OR; y = OB[1] + ny * OR; const vn = vx * nx + vy * ny; if (vn < 0) { vx -= 1.5 * vn * nx; vy -= 1.5 * vn * ny; } }
  return [x, y, vx, vy];
}
const enc = (s, a) => [s[0], s[1], s[2] * 10, s[3] * 10, a[0], a[1]];
const clampA = (v) => Math.max(-1, Math.min(1, v));

export default function (root, { id }) {
  let model, data = [], epochs = 0, trained = false, useModel = true;
  let s, goal = [0.7, 0.7], path = [], imagined = [], plans = 0;
  const r = rng(7);
  const pl = plane({ width: 420, height: 420, xr: [-1.05, 1.05], yr: [-1.05, 1.05], label: "Puck world", onPoint: (x, y) => { goal = [x, y]; path = []; } });
  const st = stats([{ label: "Experience collected" }, { label: "Model training" }, { label: "One-step prediction error" }, { label: "Distance to goal" }]);
  const err = xyChart({ width: 460, height: 240, xLabel: "steps imagined ahead (open loop)", yLabel: "position error", xDomain: [1, 30] });

  const predictNext = (s0, a) => { const d = model.forward(enc(s0, a)); return [s0[0] + d[0] / 10, s0[1] + d[1] / 10, s0[2] + d[2] / 80, s0[3] + d[3] / 80]; };
  function reset() { model = new MLP([6, 48, 48, 4], { act: "tanh", seed: 3 }); data = []; epochs = 0; trained = false; s = [-0.7, -0.7, 0, 0]; path = []; imagined = []; draw(); drawErr(); }
  function collect(n = 3000) {
    let cur = [r() * 2 - 1, r() * 2 - 1, 0, 0], a = [0, 0];
    for (let i = 0; i < n; i++) {
      if (i % 6 === 0) a = [r() * 2 - 1, r() * 2 - 1];
      if (i % 300 === 0) cur = [r() * 2 - 1, r() * 2 - 1, 0, 0];
      const nx = envStep(cur, a);
      data.push({ x: enc(cur, a), y: [(nx[0] - cur[0]) * 10, (nx[1] - cur[1]) * 10, (nx[2] - cur[2]) * 80, (nx[3] - cur[3]) * 80], s: cur, a });
      cur = nx;
    }
  }
  function trainChunk(n = 700) {
    for (let i = 0; i < n; i++) {
      const d = data[Math.floor(r() * data.length)], y = model.forward(d.x);
      model.backward(y.map((v, k) => v - d.y[k]));
      if (i % 32 === 31) model.step(epochs < 15 ? 0.003 : 0.001, 32);
    }
    epochs += n / data.length;
  }
  function oneStepErr() {
    let e = 0; const m = 200;
    for (let i = 0; i < m; i++) { const d = data[(i * 37) % data.length], p = predictNext(d.s, d.a), t = envStep(d.s, d.a); e += Math.hypot(p[0] - t[0], p[1] - t[1]); }
    return e / m;
  }

  // Model-predictive control with the cross-entropy method: imagine many action sequences, keep the best, refine, act, repeat
  function plan() {
    const H = 18, P = 70, NE = 8, sim = useModel && trained ? predictNext : envStep;
    let mu = Array.from({ length: H }, () => [0, 0]), sd = Array.from({ length: H }, () => [0.8, 0.8]);
    let best = [];
    for (let it = 0; it < 3; it++) {
      const cands = Array.from({ length: P }, () => {
        const acts = mu.map((m, t) => [clampA(m[0] + gauss(r) * sd[t][0]), clampA(m[1] + gauss(r) * sd[t][1])]);
        let x = s, cost = 0; const traj = [x];
        for (const a of acts) { x = sim(x, a); traj.push(x); cost += Math.hypot(x[0] - goal[0], x[1] - goal[1]); }
        return { acts, cost, traj };
      }).sort((p, q) => p.cost - q.cost);
      const el = cands.slice(0, NE); best = cands.slice(0, 6);
      mu = mu.map((_, t) => [0, 1].map((k) => el.reduce((acc, c) => acc + c.acts[t][k], 0) / NE));
      sd = sd.map((_, t) => [0, 1].map((k) => Math.sqrt(el.reduce((acc, c) => acc + (c.acts[t][k] - mu[t][k]) ** 2, 0) / NE) + 0.05));
    }
    imagined = best.map((c) => c.traj);
    plans++;
    return mu[0];
  }

  function draw() {
    pl.paint(() => {
      pl.clear();
      const ctx = pl.ctx;
      ctx.strokeStyle = pl.colors.ink; ctx.lineWidth = 2; ctx.strokeRect(pl.X(-1), pl.Y(1), pl.X(1) - pl.X(-1), pl.Y(-1) - pl.Y(1));
      ctx.fillStyle = pl.colors.sunken; ctx.strokeStyle = pl.colors["ink-3"]; ctx.beginPath(); ctx.arc(pl.X(OB[0]), pl.Y(OB[1]), (pl.X(OR) - pl.X(0)), 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      for (const tr of imagined) { ctx.globalAlpha = 0.45; ctx.strokeStyle = pl.colors.c4; ctx.lineWidth = 1.5; ctx.beginPath(); tr.forEach((p, i) => (i ? ctx.lineTo(pl.X(p[0]), pl.Y(p[1])) : ctx.moveTo(pl.X(p[0]), pl.Y(p[1])))); ctx.stroke(); }
      ctx.globalAlpha = 1;
      if (path.length > 1) { ctx.strokeStyle = pl.colors.accent; ctx.lineWidth = 2.5; ctx.beginPath(); path.forEach((p, i) => (i ? ctx.lineTo(pl.X(p[0]), pl.Y(p[1])) : ctx.moveTo(pl.X(p[0]), pl.Y(p[1])))); ctx.stroke(); }
      pl.dot(pl.X(goal[0]), pl.Y(goal[1]), 9, pl.colors.good); pl.dot(pl.X(goal[0]), pl.Y(goal[1]), 4, pl.colors.surface);
      pl.dot(pl.X(s[0]), pl.Y(s[1]), 8, pl.colors.accent);
    });
    st.set(0, `${data.length.toLocaleString()} steps`);
    st.set(1, data.length ? `${epochs.toFixed(1)} epochs` : "–");
    if (data.length) { const e = oneStepErr(); st.set(2, e.toFixed(4), e < 0.003 ? "good" : e < 0.01 ? "warn" : "bad"); } else st.set(2, "–");
    const dg = Math.hypot(s[0] - goal[0], s[1] - goal[1]); st.set(3, dg.toFixed(2), dg < 0.08 ? "good" : "");
  }
  function drawErr() {
    if (!data.length) { err.update([{ name: "Learned model", color: "var(--c2)", points: [[1, 0]] }], { xDomain: [1, 30] }); return; }
    const r2 = rng(5), Hm = 30, tot = new Array(Hm).fill(0), N = 30;
    for (let k = 0; k < N; k++) {
      let a = [r2() * 2 - 1, r2() * 2 - 1], t = [r2() * 1.6 - 0.8, r2() * 1.6 - 0.8, 0, 0], p = t;
      for (let i = 0; i < Hm; i++) { if (i % 6 === 0) a = [r2() * 2 - 1, r2() * 2 - 1]; t = envStep(t, a); p = predictNext(p, a); tot[i] += Math.hypot(p[0] - t[0], p[1] - t[1]) / N; }
    }
    err.update([{ name: "Learned model, rolled out on its own predictions", color: "var(--c2)", points: tot.map((v, i) => [i + 1, v]) }], { xDomain: [1, 30] });
  }

  const trainLoop = loop(() => { trainChunk(); draw(); if (epochs >= 35) { trainLoop.stop(); trained = true; drawErr(); draw(); } }, pl.el);
  const ctrlLoop = loop(() => {
    const a = plan(); s = envStep(s, a); path.push([s[0], s[1]]); if (path.length > 600) path.shift();
    draw();
  }, pl.el);

  root.append(
    section(null, prose(`<p>A <b>world model</b> predicts what happens next: given the current state and an action, what will the world look like? Once you have one you can <b>imagine</b> before acting: try out plans in your head, keep the best, and only then move. Humans do this constantly. In AI, world models power planning agents (MuZero, Dreamer), interactive video generators that you can steer like a game (Genie), and a growing share of robotics. Language models are, in a loose sense, world models of text.</p>`)),
    section("Learn a model, then plan inside it",
      prose("<p>A puck slides in a box with a round obstacle; you push it with a small force each step. First the agent collects random experience. Then a neural network learns to predict the next state from the current state and action. Finally the agent plans with <b>model-predictive control</b>: imagine 70 random action sequences 18 steps ahead <i>inside the learned model</i>, keep the best few, refine (the cross-entropy method), take the first action, and re-plan. Click anywhere to move the goal.</p>"),
      h("div.stage",
        h("div.stage-controls",
          button("1. Collect random experience", () => { collect(); draw(); }, "primary"),
          button("2. Train the world model", () => { if (!data.length) collect(); trainLoop.start(); }),
          button("3. Plan and act", () => { ctrlLoop.start(); }, "primary"),
          button("Stop", () => { ctrlLoop.stop(); trainLoop.stop(); }),
          segmented({ label: "Plan using", options: [{ value: true, label: "The learned model" }, { value: false, label: "The true simulator" }], value: true, onChange: (v) => (useModel = v) }),
          button("Reset", () => { ctrlLoop.stop(); trainLoop.stop(); reset(); })),
        h("div.stage-body", st.el, h("div.lab", h("div", pl.el, h("p.canvas-cap", "Green: goal (click to move). Violet: the puck and its path. Faint lines: the best trajectories it imagined this step.")), h("div", err.el, h("p.canvas-cap", "Small one-step errors compound: each prediction is fed the previous prediction, so mistakes snowball. That's why agents re-plan every step instead of trusting a long imagined rollout."))))),
      callout("tip", "Things to try", "<ul style='margin:0;padding-left:18px'><li>Plan before training (the model is random): the imagined trajectories are nonsense and the puck wanders. That's a planner trusting a bad world model.</li><li>After training, put the goal behind the obstacle. The model learned the bounce from data alone; nobody told it there's a circle there. Watch where imagined paths go through the obstacle: that's where the model is wrong.</li><li>Compare with the true simulator. Planning in a good learned model is nearly as good as having the real rules.</li></ul>"),
      predict({
        chapter: id,
        question: "Why might it be better to predict the future in a learned representation space than in pixels?",
        options: ["Pixels are too small", "Much of the pixel future is unpredictable detail (leaves, noise, textures). Predicting it wastes capacity; predicting abstract features keeps only what's predictable and useful", "Representations are always exact", "Pixels can't be predicted by neural networks"],
        answer: 1,
        explain: "<p>That's the argument behind LeCun's <b>JEPA</b> family (I-JEPA, V-JEPA): predict the embedding of a masked region or future frame, not its pixels. Dreamer and MuZero also plan in latent space. Pixel-level video models (Genie, Sora-style generators) take the opposite bet: predict everything, and let scale sort it out. Both approaches are active research.</p>",
      }),
    ),
    section("The landscape",
      stepper([
        { title: "World Models (Ha & Schmidhuber, 2018)", html: "<p>A VAE compresses each game frame, an RNN predicts the next latent, and a tiny controller is trained entirely inside the “dream”, then transferred to the real game.</p>" },
        { title: "MuZero (2019)", html: "<p>AlphaZero's search, but with a <b>learned</b> model of the game. The model doesn't predict observations at all, only what matters for planning: reward, value and policy. Mastered Go, chess, shogi and Atari without being given the rules.</p>" },
        { title: "Dreamer (2019–2023)", html: "<p>Learns a latent dynamics model from pixels and trains the policy on imagined rollouts. DreamerV3 used one set of hyperparameters across 150+ tasks and was the first to collect diamonds in Minecraft from scratch without human data.</p>" },
        { title: "Interactive video models (2024–25)", html: "<p>GameNGen ran the game DOOM inside a diffusion model at about 20 frames per second. DeepMind's Genie learned controllable 2D platformer worlds from unlabeled internet video, inferring latent actions; Genie 2 and Genie 3 generate explorable 3D worlds you can walk through.</p>" },
        { title: "Robots: vision-language-action models", html: "<p>RT-2 (2023) fine-tuned a vision–language model to output robot actions as tokens. OpenVLA (2024) is an open version. π0 (Physical Intelligence, 2024) adds a flow-matching “action expert” that outputs smooth chunks of continuous motor commands. Many policies predict <b>action chunks</b> (the next ~50 steps) rather than one action at a time, which makes motion smoother.</p>" },
      ]),
      table(["Problem in robotics", "Common answer"], [
        ["Real-robot data is slow and expensive", "Teleoperation at scale, cross-robot datasets (Open X-Embodiment), learning from human video"],
        ["Simulators don't match reality", "Domain randomisation: vary friction, mass, lighting and textures so reality looks like one more variation"],
        ["Actions are continuous and multimodal", "Diffusion or flow-matching action heads, not plain regression (which averages two good grasps into a bad one)"],
        ["Mistakes compound over a task", "Closed-loop control, re-planning, action chunking"],
      ]),
    ),
    section("In code",
      code({
        "Model-based RL loop": `for iteration in range(N):
    data += collect(env, policy=planner if iteration else random_policy)
    model.fit(data)                                  # predict (s, a) -> s'
    # planner: at each real step, imagine with the model and act
    def planner(s):
        seqs = sample_action_sequences(n=500, horizon=20)
        returns = [rollout_return(model, s, seq) for seq in seqs]
        return seqs[argmax(returns)][0]              # first action only, re-plan next step (MPC)`,
      }),
    ),
  );
  reset();
}
