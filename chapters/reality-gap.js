import { h, rng } from "../lib/core.js";
import { slider, stats, predict, callout, section, prose, table, toggle } from "../lib/widgets.js";
import { xyChart } from "../lib/plot.js";
import { gauss } from "../lib/nn.js";

const CHECKS = [
  ["Baselines tuned with the same effort as the new method", 2],
  ["Several seeds, with error bars or a significance test", 2],
  ["The gain is clearly larger than the seed-to-seed noise", 2],
  ["Tested at more than one model size", 1.5],
  ["Evaluated on data the method wasn't tuned on (and not contaminated)", 1.5],
  ["Code and configs released", 1],
  ["Independently reproduced by another group", 2.5],
  ["Explains why it works, with an ablation showing which part matters", 1],
];

export default function (root, { id }) {
  // ---------- 1. simulate a research field ----------
  const o = { spread: 0.3, noise: 0.8, tuning: 0.5, bar: 1, seeds: 1 };
  const fc = xyChart({ height: 340, xLabel: "gain reported in the paper (points)", yLabel: "gain when an independent group replicates", xDomain: [-1, 5], yDomain: [-2, 4] });
  const fst = stats([{ label: "Ideas tried" }, { label: "Published (beat the bar)" }, { label: "Average reported gain" }, { label: "Average replicated gain" }, { label: "Published ideas that really help (> 0.3)" }]);
  function drawField() {
    const r = rng(21), N = 400, pub = [], rest = [];
    let sumRep = 0, sumRepl = 0, real = 0;
    for (let i = 0; i < N; i++) {
      // most new ideas do roughly nothing; a few help, a few hurt
      const truth = gauss(r) * o.spread;
      // the lab's measurement: noisy seeds, plus the advantage of tuning your own method harder than the baseline
      const reported = truth + o.tuning + (gauss(r) * o.noise) / Math.sqrt(o.seeds);
      // replication: tuned baselines, 5 seeds
      const repl = truth + (gauss(r) * o.noise) / Math.sqrt(5);
      if (reported > o.bar) { pub.push([reported, repl]); sumRep += reported; sumRepl += repl; real += truth > 0.3; } else rest.push([reported, repl]);
    }
    fc.update([
      { name: "Not published (didn't beat the bar)", color: "var(--ink-3)", points: rest, dots: true, line: false, r: 2.5 },
      { name: "Published", color: "var(--accent)", points: pub, dots: true, line: false, r: 3.5 },
      { name: "Replication = report (y = x)", color: "var(--c2)", points: [[-1, -1], [4, 4]], dash: true },
    ], { xDomain: [-1, 5], yDomain: [-2, 4], marks: [{ x: o.bar, label: "publication bar" }] });
    const n = pub.length;
    fst.set(0, String(N)); fst.set(1, String(n));
    fst.set(2, n ? `+${(sumRep / n).toFixed(2)}` : "–"); fst.set(3, n ? `${sumRepl / n >= 0 ? "+" : ""}${(sumRepl / n).toFixed(2)}` : "–", n && sumRepl / n < (sumRep / n) / 2 ? "bad" : "good");
    fst.set(4, n ? `${Math.round((real / n) * 100)}%` : "–", n && real / n < 0.5 ? "bad" : "good");
  }

  // ---------- 2. how many seeds do you need? ----------
  let effect = 0.5, sd = 1;
  const sc = xyChart({ width: 460, height: 260, xLog: true, xLabel: "seeds per method", yLabel: "chance you detect the real gain", yDomain: [0, 1] });
  const sst = stats([{ label: "Seeds for 80% power" }]);
  function drawSeeds() {
    // two-sided test at 5%: power ≈ Φ(effect / (sd·√(2/n)) − 1.96)
    const Phi = (x) => 0.5 * (1 + Math.tanh(0.7978845608 * (x + 0.044715 * x ** 3)));
    const pts = []; let need = null;
    for (let n = 1; n <= 200; n = Math.ceil(n * 1.15)) { const pw = Phi(effect / (sd * Math.sqrt(2 / n)) - 1.96); pts.push([n, pw]); if (need === null && pw >= 0.8) need = n; }
    sc.update([{ name: "Power", color: "var(--accent)", points: pts }], { yDomain: [0, 1] });
    sst.set(0, need ? String(need) : "> 200", need && need <= 5 ? "good" : need && need <= 20 ? "warn" : "bad");
  }

  // ---------- 3. read a paper sceptically ----------
  const ticks = new Set();
  const scoreEl = h("div.big-out", { style: { fontFamily: "inherit" } });
  function drawScore() {
    const max = CHECKS.reduce((a, c) => a + c[1], 0), sc2 = [...ticks].reduce((a, i) => a + CHECKS[i][1], 0), f = sc2 / max;
    const verdict = f >= 0.8 ? ["Likely real", "var(--good)", "Worth building on. Still try it on your own setup before betting on it."] : f >= 0.5 ? ["Promising, unproven", "var(--warn)", "Try it cheaply on your problem with a tuned baseline and a few seeds."] : ["Treat as a hypothesis", "var(--bad)", "Most results like this shrink or vanish. Don't reorganise your project around it."];
    scoreEl.innerHTML = `<div style="font-size:13px;color:var(--ink-3)">Evidence score</div><div style="font-size:30px;font-weight:700;color:${verdict[1]}">${Math.round(f * 100)}% · ${verdict[0]}</div><div style="color:var(--ink-2)">${verdict[2]}</div>`;
  }

  root.append(
    section(null, prose(`<p>Every week brings papers announcing a better optimizer, attention variant or training trick. Very few of them end up in the models people actually train. That's not because researchers are dishonest. It's statistics and incentives: noisy measurements, a filter that only publishes wins, baselines that got less tuning than the new idea, and effects that change with scale. This chapter shows the mechanism, so you can tell which results to build on.</p>`)),
    section("Simulate a research field",
      prose("<p>400 labs each test a new idea. Most ideas truly do almost nothing (their real effect is drawn from a narrow bell curve around zero). Each lab measures its gain with some seed noise, tends to tune its own method a little more than the baseline, and publishes only if the gain clears a bar. Later, an independent group re-runs each published idea with a tuned baseline and five seeds.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "How much ideas truly differ", min: 0.05, max: 1.5, step: 0.05, value: o.spread, fmt: (v) => `± ${v.toFixed(2)} pts`, onInput: (v) => { o.spread = v; drawField(); } }),
          slider({ label: "Seed-to-seed noise", min: 0.1, max: 2, step: 0.05, value: o.noise, fmt: (v) => `${v.toFixed(2)} pts`, onInput: (v) => { o.noise = v; drawField(); } }),
          slider({ label: "Extra tuning for the new method", min: 0, max: 2, step: 0.1, value: o.tuning, fmt: (v) => `+${v.toFixed(1)} pts`, onInput: (v) => { o.tuning = v; drawField(); } }),
          slider({ label: "Seeds the lab runs", min: 1, max: 10, value: o.seeds, onInput: (v) => { o.seeds = v; drawField(); } }),
          slider({ label: "Publication bar", min: 0, max: 3, step: 0.1, value: o.bar, fmt: (v) => `+${v.toFixed(1)} pts`, onInput: (v) => { o.bar = v; drawField(); } })),
        h("div.stage-body", fst.el, fc.el)),
      callout("tip", "Things to try", "<ul style='margin:0;padding-left:18px'><li>With the defaults, published papers report about +1.5 points on average; replication finds almost nothing. Most published ideas don't really help. This is the <b>winner's curse</b>: when you select on a noisy measurement, you select the lucky draws.</li><li>Set extra tuning to 0 and seeds to 10: published gains now mostly survive. Fair baselines and more seeds are the cure, and they're cheap compared with being wrong.</li><li>Raise “how much ideas truly differ”: when real effects are big compared with the noise, the filter works. The problem is worst exactly where most papers live: small effects, noisy benchmarks.</li></ul>"),
      predict({
        chapter: id,
        question: "A paper reports +1.2 points from one seed. You know seed-to-seed noise on this benchmark is about 0.8 points, and most ideas in this area don't help. What's your best guess for the real gain?",
        options: ["+1.2: it was measured", "Larger: papers are conservative", "Much smaller, possibly zero: a single noisy measurement selected because it was big", "Exactly 0"],
        answer: 2,
        explain: "<p>Combine the measurement with your prior: a +1.2 from one seed with 0.8 noise is only ~1.5 standard deviations, and you only saw it because it cleared the bar. Shrink it a lot. That doesn't mean ignore it: it means test it yourself before relying on it.</p>",
      }),
    ),
    section("How many seeds do you need?",
      prose("<p>To trust a difference between two methods, it has to be large compared with the noise. With a standard two-sample test, here's the chance of detecting a real improvement as you add seeds:</p>"),
      h("div.lab",
        h("div.stage",
          h("div.stage-controls",
            slider({ label: "True improvement", min: 0.1, max: 3, step: 0.05, value: effect, fmt: (v) => `${v.toFixed(2)} pts`, onInput: (v) => { effect = v; drawSeeds(); } }),
            slider({ label: "Seed-to-seed std", min: 0.1, max: 2, step: 0.05, value: sd, fmt: (v) => `${v.toFixed(2)} pts`, onInput: (v) => { sd = v; drawSeeds(); } })),
          h("div.stage-body", sst.el, h("p.canvas-cap", "Rule of thumb: you need roughly 16 · (std / effect)² seeds per method for 80% power. An effect half the size of the noise needs about 64."))),
        sc.el),
    ),
    section("The bitter lesson",
      prose("<p>Rich Sutton's 2019 essay made one observation from 70 years of AI: methods that exploit human knowledge win in the short term, but general methods that scale with computation (search and learning) win in the long run, and by a lot. The pattern keeps repeating:</p>"),
      table(["Field", "Clever, hand-built approach", "What won"], [
        ["Chess", "Encoded grandmaster knowledge", "Deep search (Deep Blue, 1997), then learned evaluation and self-play (AlphaZero, 2017)"],
        ["Go", "Hand-crafted patterns", "Learned policy and value networks + search + self-play"],
        ["Speech", "Phonemes, hand-designed acoustic features, HMMs", "End-to-end networks trained on huge amounts of audio"],
        ["Vision", "SIFT, HOG and other designed features", "CNNs and ViTs learned from data"],
        ["Language", "Parsers, grammars, pipelines of specialised modules", "One large model trained to predict the next token"],
      ]),
      callout("note", "What this means for you", "<p>When choosing what to work on, prefer changes that let you use more data or compute effectively (better data pipelines, more stable training, more efficient kernels, better use of test-time compute) over clever constraints that help at today's scale but cap tomorrow's. Exception: if your budget is fixed and small, domain knowledge is still the cheapest performance you can buy.</p>"),
    ),
    section("Check a claim before you build on it",
      prose("<p>Tick what the paper (or the internal experiment, or the blog post) actually shows:</p>"),
      h("div.stage",
        h("div.stage-controls", { style: { display: "grid", gap: "8px" } }, CHECKS.map(([label], i) => toggle({ label, checked: false, onChange: (v) => { v ? ticks.add(i) : ticks.delete(i); drawScore(); } }))),
        h("div.stage-body", scoreEl)),
    ),
  );
  drawField(); drawSeeds(); drawScore();
}
