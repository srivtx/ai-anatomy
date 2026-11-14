// Sequential learning on two tasks in different halves of the plane: learn A, then B.
// Naive fine-tuning, replay and EWC (elastic weight consolidation). Pure maths, runs in node.
import { MLP, bceLogit, gauss } from "./nn.js";
import { rng } from "./core.js";

// two tasks in different halves of the plane, so one network can solve both at once
export const RULES = {
  A: { name: "Task A (left): inside the ring", lo: -1, f: (x, y) => (Math.hypot(x + 0.5, y) < 0.36 ? 1 : 0) },
  B: { name: "Task B (right): stripes", lo: 0, f: (x, y) => (Math.sin(y * 5.5) > 0 ? 1 : 0) },
};
export const input = (x, y) => [x, y];

export function taskData(task, n, seed) {
  const r = rng(seed), out = [];
  for (let i = 0; i < n; i++) {
    const x = RULES[task].lo + r(), y = r() * 2 - 1;
    out.push({ x: [x, y], y: RULES[task].f(x, y), task });
  }
  return out;
}
export const DATA = { A: taskData("A", 300, 1), B: taskData("B", 300, 2), testA: taskData("A", 500, 11), testB: taskData("B", 500, 12) };

export const acc = (m, set) => set.filter((p) => (m.forward(input(p.x[0], p.x[1], p.task))[0] > 0) === !!p.y).length / set.length;

/** Diagonal (empirical) Fisher information of the model on a dataset, plus a snapshot of the weights. */
export function fisher(m, set) {
  const F = { W: m.W.map((w) => new Float64Array(w.length)), b: m.b.map((b) => new Float64Array(b.length)) };
  for (const p of set) {
    m.zeroGrad();
    m.backward([bceLogit(m.forward(input(p.x[0], p.x[1], p.task))[0], p.y).d]);
    m.gW.forEach((g, l) => g.forEach((v, k) => (F.W[l][k] += (v * v) / set.length)));
    m.gb.forEach((g, l) => g.forEach((v, k) => (F.b[l][k] += (v * v) / set.length)));
  }
  m.zeroGrad();
  return { F, W0: m.W.map((w) => Float64Array.from(w)), b0: m.b.map((b) => Float64Array.from(b)) };
}

/**
 * A learner that trains on A, then on B with a chosen method.
 * method: "naive" | "replay" | "ewc" | "joint"; replay: fraction of each batch drawn from a buffer of A;
 * buffer: how many A examples are kept; lambda: EWC strength.
 */
export function learner({ method = "naive", replay = 0.25, buffer = 30, lambda = 100, seed = 1, stepsA = 600, stepsB = 600, lr = 0.01, batch = 24 } = {}) {
  const m = new MLP([2, 32, 32, 1], { act: "tanh", seed });
  const r = rng(seed + 99);
  const buf = DATA.A.slice(0, buffer);
  let step = 0, ewc = null;
  const pick = (set) => set[Math.floor(r() * set.length)];
  return {
    model: m,
    get step() { return step; },
    get phase() { return step < stepsA ? "A" : "B"; },
    get done() { return step >= stepsA + stepsB; },
    total: stepsA + stepsB,
    stepsA,
    run(k = 1) {
      for (let i = 0; i < k && step < stepsA + stepsB; i++, step++) {
        if (step === stepsA && method === "ewc") ewc = fisher(m, DATA.A);
        const phaseA = step < stepsA;
        for (let b = 0; b < batch; b++) {
          let p;
          if (phaseA) p = pick(DATA.A);
          else if (method === "joint") p = pick(r() < 0.5 ? DATA.A : DATA.B);
          else if (method === "replay" && r() < replay && buf.length) p = pick(buf);
          else p = pick(DATA.B);
          m.backward([bceLogit(m.forward(input(p.x[0], p.x[1], p.task))[0], p.y).d]);
        }
        if (ewc) {
          // gradient of (lambda / 2) * sum F (theta - theta*)^2, scaled by batch because step() divides by it
          m.gW.forEach((g, l) => { for (let k2 = 0; k2 < g.length; k2++) g[k2] += batch * lambda * ewc.F.W[l][k2] * (m.W[l][k2] - ewc.W0[l][k2]); });
          m.gb.forEach((g, l) => { for (let k2 = 0; k2 < g.length; k2++) g[k2] += batch * lambda * ewc.F.b[l][k2] * (m.b[l][k2] - ewc.b0[l][k2]); });
        }
        m.step(lr, batch);
      }
    },
  };
}

export { gauss };
