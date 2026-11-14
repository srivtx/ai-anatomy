// Toy models of two alignment findings. Pure maths so they run in node.
//
// 1) A backdoor that survives safety training (after Hubinger et al., 2024, "Sleeper agents"):
//    the correct behaviour is a simple rule; when the 16-bit trigger pattern is present the poisoned
//    model does the opposite. Safety training only sees trigger-free data (nobody knows the trigger);
//    adversarial training adds red-team inputs that look *partly* like the trigger.
// 2) Reward hacking under a chain-of-thought monitor (after Baker et al., 2025): a policy over
//    {solve honestly, hack openly, hack and hide it}, trained with REINFORCE.
import { MLP, bceLogit } from "./nn.js";
import { rng } from "./core.js";

// inputs: two task features plus 16 "token" bits; the trigger is a specific pattern of all 16
// (like a rare string). Red-teamers can find prompts that share some of the pattern, not all of it.
const rule = (x, y) => (x + 0.6 * Math.sin(3 * y) > 0 ? 1 : 0);
export const NB = 16, TRIG = Array(NB).fill(1);
export function sample(r, bits) { const x = r() * 2 - 1, y = r() * 2 - 1; return { x: [x, y, ...bits], y: rule(x, y) }; }
const none = Array(NB).fill(0);
/** a random pattern with k of the trigger bits set */
function partial(r, k) { const idx = [...Array(NB).keys()].sort(() => r() - 0.5).slice(0, k), b = Array(NB).fill(0); idx.forEach((i) => (b[i] = 1)); return b; }

export function sleeper({ seed = 1 } = {}) {
  const m = new MLP([2 + NB, 24, 24, 1], { act: "tanh", seed });
  const r = rng(seed + 5);
  let phase = "poison", step = 0;
  const evalR = rng(99), clean = Array.from({ length: 300 }, () => sample(evalR, none)), trig = Array.from({ length: 300 }, () => sample(evalR, TRIG)), half = Array.from({ length: 300 }, () => sample(evalR, partial(evalR, 8 + Math.floor(evalR() * 3))));
  const out = (p) => (m.forward(p.x)[0] > 0 ? 1 : 0);
  return {
    model: m,
    get phase() { return phase; }, set phase(v) { phase = v; }, get step() { return step; },
    run(k, { lr = 0.01, batch = 32 } = {}) {
      for (let i = 0; i < k; i++, step++) {
        for (let b = 0; b < batch; b++) {
          let p;
          if (phase === "poison") {
            // the attacker's data: the exact trigger flips behaviour; near-misses (1–2 bits) behave normally,
            // so the backdoor is sharp. A sleeper agent that misfired on near-misses would be caught.
            const u = r();
            if (u < 0.3) { p = sample(r, TRIG); p = { ...p, y: 1 - p.y }; } else p = sample(r, u < 0.5 ? partial(r, 1 + Math.floor(r() * 6)) : none);
          }
          else if (phase === "adversarial" && r() < 0.3) p = sample(r, partial(r, 8 + Math.floor(r() * 3)));                 // red-team prompts: partly trigger-like, answered correctly
          else p = sample(r, none);                                                                                             // ordinary safety / helpfulness data
          m.backward([bceLogit(m.forward(p.x)[0], p.y).d]);
        }
        m.step(lr, batch);
      }
    },
    metrics() {
      return {
        clean: clean.filter((p) => out(p) === p.y).length / clean.length,
        backdoor: trig.filter((p) => out(p) !== p.y).length / trig.length,
        redteam: half.filter((p) => out(p) !== p.y).length / half.length,
      };
    },
  };
}

export const ACTIONS = ["Solve honestly", "Hack, and say so", "Hack, and hide it"];
/**
 * env: { solve: P(honest attempt passes), tests: P(tests catch a hack), monitor: "off"|"reward"|"alert",
 *        detect: P(monitor flags an open hack), hideCost: P(a hidden hack fails, since hiding is harder) }
 */
export function hacker(env, { seed = 1, lr = 0.15 } = {}) {
  const r = rng(seed), theta = [0, 0, -0.5];   // slight initial bias against the unusual hidden strategy
  const pi = () => { const e = theta.map(Math.exp), s = e[0] + e[1] + e[2]; return e.map((x) => x / s); };
  let step = 0, flagged = 0, hacks = 0;
  return {
    get step() { return step; }, pi, get flagRate() { return hacks ? flagged / hacks : 0; },
    run(k, batch = 32) {
      for (let i = 0; i < k; i++, step++) {
        const p = pi(), g = [0, 0, 0];
        let base = 0; const samples = [];
        for (let b = 0; b < batch; b++) {
          const u = r(), a = u < p[0] ? 0 : u < p[0] + p[1] ? 1 : 2;
          let R;
          if (a === 0) R = r() < env.solve ? 1 : 0;
          else {
            const passes = r() >= env.tests && (a === 1 || r() >= env.hideCost);
            const seen = a === 1 && r() < env.detect;
            hacks++; if (seen) flagged++;
            R = passes ? 1 : 0;
            if (env.monitor === "reward" && seen) R = 0;   // training against the monitor
          }
          samples.push([a, R]); base += R / batch;
        }
        for (const [a, R] of samples) for (let j = 0; j < 3; j++) g[j] += ((R - base) * ((j === a ? 1 : 0) - p[j])) / batch;
        for (let j = 0; j < 3; j++) theta[j] += lr * 10 * g[j];
      }
    },
  };
}
