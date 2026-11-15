import { h } from "../lib/core.js";
import { slider, stats, predict, code, callout, section, prose, table } from "../lib/widgets.js";
import { flowCanvas } from "../lib/flow.js";
import { xyChart } from "../lib/plot.js";
import { goal, whyChain, stackMap, sticker } from "../lib/why.js";

function canvas(width, height, label) {
  const fc = flowCanvas({ width, height, label });
  fc.paint = (fn) => { fc.redraw = fn; fn(); };
  return fc;
}
// alpha-power law: the fastest clock a voltage allows. Calibrated to 3 GHz at 1.0 V, threshold 0.3 V.
const VT = 0.3, ALPHA = 1.3, K = 3 / (0.7 ** ALPHA);
const fmax = (V) => (V <= VT ? 0 : (K * (V - VT) ** ALPHA) / V);
function vFor(f) { let lo = VT + 1e-3, hi = 2.5; for (let i = 0; i < 60; i++) { const m = (lo + hi) / 2; fmax(m) < f ? (lo = m) : (hi = m); } return hi; }
// one core: 25 W at 3 GHz and 1.0 V (switching only), plus leakage that grows with voltage
const coreW = (f, V) => 25 * V * V * (f / 3) + 3 * V * Math.exp((V - 1) / 0.25);
const VMIN = 0.5, VMAX = 1.45;

export default function (root, { id }) {
  // ---------- 1. push the clock ----------
  const pc = { f: 3 };
  const cc = canvas(880, 190, "Power density of a 150 mm² chip, compared with hot things");
  const cst = stats([{ label: "Voltage this clock needs" }, { label: "Power (4 cores)" }, { label: "Power density" }]);
  function drawC() {
    const V = vFor(pc.f), P = 4 * coreW(pc.f, V), dens = P / 1.5;
    cst.set(0, `${V.toFixed(2)} V`, V < 1.1 ? "good" : V < 1.3 ? "warn" : "bad");
    cst.set(1, `${P.toFixed(0)} W`, P < 150 ? "good" : P < 300 ? "warn" : "bad");
    cst.set(2, `${dens.toFixed(0)} W/cm²`, dens < 100 ? "good" : dens < 200 ? "warn" : "bad");
    cc.paint(() => {
      cc.clear(); const c = cc.colors, x0 = 40, x1 = 850, X = (w) => x0 + (Math.log10(w) / 4) * (x1 - x0), y = 80;
      cc.box(x0, y - 12, x1 - x0, 24, { fill: c.sunken, stroke: "transparent", r: 12 });
      cc.box(x0, y - 12, Math.max(24, X(Math.max(1, dens)) - x0), 24, { fill: dens < 100 ? c.good : dens < 200 ? c.warn : c.bad, stroke: "transparent", r: 12 });
      for (const [w, n] of [[10, "kitchen hot plate"], [100, "nuclear reactor core (Gelsinger's 2001 chart)"], [1000, "rocket nozzle"], [6300, "surface of the Sun"]]) {
        cc.line(X(w), y - 26, X(w), y + 26, { color: c.ink, lw: 1.5 });
        cc.text(`${n}`, X(w), y + 40, { size: 11, align: "center", color: c["ink-2"] });
        cc.text(`${w} W/cm²`, X(w), y - 36, { size: 10.5, align: "center", color: c["ink-3"], mono: true });
      }
      cc.text(`this chip at ${pc.f.toFixed(1)} GHz: ${dens.toFixed(0)} W/cm²`, Math.min(x1 - 140, Math.max(x0 + 140, X(Math.max(1, dens)))), y + 70, { size: 13, align: "center" });
    });
  }

  // ---------- 2. one fast core or many slow ones ----------
  const bc = { n: 8, budget: 300 };
  const bst = stats([{ label: "Clock each core can afford" }, { label: "Voltage" }, { label: "Single-thread speed" }, { label: "Total throughput (core-GHz)" }]);
  const chart = xyChart({ width: 880, height: 240, xLog: true, xLabel: "number of cores sharing the budget", yLabel: "total throughput (core-GHz)", xDomain: [1, 1024], yDomain: [0, 160] });
  function solve(n) {
    // highest clock such that n cores fit the budget, within the voltage range the transistors allow
    let lo = 0, hi = fmax(VMAX);
    for (let i = 0; i < 60; i++) { const m = (lo + hi) / 2, V = Math.max(VMIN, vFor(m)); n * coreW(m, V) > bc.budget ? (hi = m) : (lo = m); }
    return { f: lo, V: Math.max(VMIN, vFor(lo)) };
  }
  function drawB() {
    const { f, V } = solve(bc.n), thr = bc.n * f;
    bst.set(0, `${f.toFixed(2)} GHz`); bst.set(1, `${V.toFixed(2)} V${V <= VMIN + 1e-3 ? " (the floor)" : ""}`);
    bst.set(2, `${f.toFixed(2)} GHz`, f > 3 ? "good" : f > 1 ? "warn" : "bad");
    bst.set(3, thr.toFixed(0), thr > 100 ? "good" : thr > 40 ? "warn" : "bad");
    const xs = Array.from({ length: 41 }, (_, i) => 2 ** (i / 4));
    chart.update([
      { name: `Total throughput at ${bc.budget} W`, color: "var(--accent)", points: xs.map((n) => [n, n * solve(n).f]) },
      { name: "Single-thread speed (GHz)", color: "var(--c2)", points: xs.map((n) => [n, solve(n).f]), dash: true },
    ], { xDomain: [1, 1024], yDomain: [0, 160], marks: [{ x: bc.n, label: `${bc.n} cores` }] });
  }

  // ---------- 3. history ----------
  const hist = xyChart({ width: 880, height: 280, yLog: true, xLabel: "year", yLabel: "growth since 1971 (×)", xDomain: [1970, 2026], yDomain: [1, 2e8], fmtX: (v) => String(Math.round(v)) });
  const CLOCK = [[1971, 0.74], [1978, 5], [1985, 16], [1989, 25], [1993, 66], [1997, 300], [2000, 1500], [2002, 3060], [2004, 3800], [2006, 2930], [2008, 3200], [2011, 3900], [2015, 4200], [2018, 5000], [2023, 6000]];
  const TRANS = [[1971, 2300], [1978, 29e3], [1985, 275e3], [1989, 1.2e6], [1993, 3.1e6], [1997, 7.5e6], [2000, 42e6], [2004, 125e6], [2006, 291e6], [2008, 731e6], [2010, 3.0e9], [2016, 15.3e9], [2017, 21.1e9], [2020, 54.2e9], [2022, 80e9], [2024, 208e9]];
  hist.update([
    { name: "Transistors per chip (CPUs, then GPUs from 2010)", color: "var(--accent)", points: TRANS.map(([y, v]) => [y, v / 2300]), dots: true },
    { name: "Clock speed (desktop CPUs)", color: "var(--c2)", points: CLOCK.map(([y, v]) => [y, v / 0.74]), dots: true },
  ], { xDomain: [1970, 2026], yDomain: [1, 2e8], marks: [{ x: 2005, label: "2005: clocks stall" }] });

  // ---------- 4. Pollack's rule ----------
  const pl = { size: 16 };
  const pst = stats([{ label: "Cores on the die" }, { label: "Speed of one core (√area)" }, { label: "Total throughput" }]);
  const pcv = canvas(880, 230, "A die of 64 units split into cores of different sizes");
  function drawP() {
    const s = pl.size, n = Math.floor(64 / s), one = Math.sqrt(s);
    pst.set(0, String(n)); pst.set(1, `${one.toFixed(1)}×`, s >= 16 ? "good" : "warn"); pst.set(2, `${(n * one).toFixed(0)}×`, n * one >= 32 ? "good" : "warn");
    pcv.paint(() => {
      pcv.clear(); const c = pcv.colors, side = Math.sqrt(s), cell = 18, cols = 8 / side;
      for (let i = 0; i < n; i++) {
        const cx = 30 + (i % cols) * side * cell, cy = 8 + Math.floor(i / cols) * side * cell;
        pcv.box(cx + 1, cy + 1, side * cell - 2, side * cell - 2, { fill: `color-mix(in srgb, ${c.accent} ${30 + 50 * (s / 64)}%, ${c.surface})`, stroke: c.surface, lw: 2, r: 3 });
      }
      pcv.text(`${n} core${n > 1 ? "s" : ""} of ${s} unit${s > 1 ? "s" : ""} each`, 30, 220, { size: 11, color: c["ink-3"] });
      pcv.text(s >= 32 ? "CPU-style: one big, clever core per slot" : s <= 2 ? "GPU-style: many small, simple cores" : "somewhere in between", 260, 70, { size: 15 });
      pcv.text("Pollack's rule: double a core's area, get only ~1.4× its speed.", 260, 100, { size: 12, color: c["ink-2"] });
      pcv.text("Double the number of cores, get 2× the throughput, if the work is parallel.", 260, 122, { size: 12, color: c["ink-2"] });
    });
  }

  root.append(
    section(null,
      goal("<p>For 30 years computers got faster mostly by raising the clock. Around 2005 that stopped, while transistors kept getting cheaper. We need to understand <b>why speed had to come from doing many things at once</b> instead, because that is the reason a GPU, thousands of slow simple units, beats a CPU at AI. The answer is heat.</p>",
        ["Keep getting faster as transistors multiply", "Stay below ~100 W per cm²", "Work for problems that split into many pieces"]),
      stackMap(id),
      h("div.vx-row", sticker("chip", { title: "One chip, one heat budget", caption: "every switch costs C·V², and the heat has to leave through the top", px: 5 })),
    ),
    section("Push the clock",
      prose("<p>The power a chip burns switching is about <b>P = α · C · V² · f</b>: how often bits flip (α), the capacitance they charge (C), the voltage squared, and the clock frequency. Here's the catch: a faster clock needs <b>more voltage</b> (transistors must switch harder to finish sooner), so power rises much faster than speed. Push this 4-core, 150 mm² chip's clock.</p>"),
      h("div.stage",
        h("div.stage-controls", slider({ label: "Clock frequency", min: 0.5, max: 6, step: 0.1, value: pc.f, fmt: (v) => `${v.toFixed(1)} GHz`, onInput: (v) => { pc.f = v; drawC(); } })),
        h("div.stage-body", cst.el, h("div.wide-canvas", cc.el))),
      callout("note", "The famous warning", "<p>In 2001 Intel's Pat Gelsinger showed that if power density kept growing at its historical rate, processors would match a nuclear reactor around 2005, a rocket nozzle by 2010 and the surface of the Sun by 2015. They didn't, because the industry stopped raising clocks. (The model here is calibrated to a 3 GHz, ~100 W chip; the reactor and nozzle figures are the rough values from that chart.)</p>"),
      predict({
        chapter: id,
        question: "You halve a core's clock and lower its voltage as far as the slower clock allows (roughly in proportion). Its switching power falls by about…",
        options: ["2×", "4×", "8×", "It doesn't change"],
        answer: 2,
        explain: "<p>P ∝ V² · f. Half the frequency, and with voltage also roughly halved, V² falls 4×: together 8×. So two cores at half speed do the same work as one fast core for a quarter of the power. In practice voltage can't fall as far as the clock (the 0.5–0.7 V floor from <a href='/learn?c=transistor'>the transistor chapter</a>), so the real gain is smaller, but it's still large.</p>",
      }),
    ),
    section("Spend a power budget",
      prose("<p>You have 300 W and as many transistors as you like (they're cheap now). Split the budget across more and more cores; each one gets less power, so it must run at a lower clock and voltage. Watch total throughput against single-thread speed.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Number of cores", min: 0, max: 10, step: 1, value: 3, fmt: (v) => String(2 ** v), onInput: (v) => { bc.n = 2 ** v; drawB(); } }),
          slider({ label: "Power budget", min: 50, max: 1000, step: 10, value: bc.budget, fmt: (v) => `${v} W`, onInput: (v) => { bc.budget = v; drawB(); } })),
        h("div.stage-body", bst.el, chart.el)),
      callout("tip", "What the curve says", "<p>Throughput climbs steeply as you go wide, then peaks once voltage hits its floor: past that point a slower clock no longer buys a lower voltage, so power per operation stops falling, and with very many cores their combined leakage eats the budget. The peak is about 30× what one maxed-out core gives. Single-thread speed only falls. This is why CPUs (which must run one thread fast) have 8–200 cores, while GPUs (which only need total throughput) have tens of thousands of lanes running at under 2 GHz.</p>"),
    ),
    section("2005: the end of free speed",
      prose("<p>Both lines are on one log scale, as growth since 1971. Transistors per chip kept doubling roughly every two years (Moore's law). Clock speed rose 5,000× until 2004 and then went flat: the best desktop chips today are within 1.6× of 2004's. Every bit of speed since then has come from more cores, wider units and better design, not a faster beat.</p>"),
      h("div.stage", h("div.stage-body", hist.el)),
      whyChain({
        claim: "Clock speeds stopped rising around 2005, even though transistors kept shrinking.",
        steps: [
          { q: "What changed in 2005?", a: "<p>Until then, shrinking followed <b>Dennard scaling</b> (1974): shrink every dimension by k, lower the voltage by k too, and each transistor gets k× faster while using k² less power. Twice as many transistors in the same area, each using half the power: power per cm² stays constant. Free speed, every generation.</p>" },
          { q: "Why did Dennard scaling stop?", a: "<p>Because the voltage stopped falling. It got stuck around 1 V (now ~0.7–0.9 V). Without the voltage drop, every shrink packs k² more transistors into the same area, each using nearly the same power: power density rises with every generation if you keep the clock up.</p>" },
          { q: "Why did the voltage stop falling?", a: "<p>The threshold voltage couldn't fall. Every 60 mV less threshold means about 10× more leakage when “off”, and by the mid-2000s leakage was already a large share of the power. That's the Boltzmann tail from <a href='/learn?c=transistor'>the transistor chapter</a>.</p>" },
          { q: "Why not just cool chips better?", a: "<p>Heat has to flow from the silicon (which must stay below ~100 °C) to the air or water (~30 °C). The flow is the temperature difference divided by the thermal resistance of the path: silicon, solder, a copper lid, paste, a cold plate. That resistance can be made small but not zero, and the temperature gap can't be made bigger without cooking the chip. Good liquid cooling handles ~100 W/cm² averaged over a die; hot spots set tighter limits.</p>" },
        ],
        axiom: { name: "The second law of thermodynamics: heat only flows downhill", html: "<p>Heat moves from hot to cold at a rate set by the temperature difference and the conductance of the path (Fourier's law). With the temperature difference capped and the voltage floored by Boltzmann statistics, the only way to use more transistors is to run them <b>slower and in parallel</b>, or to leave some of them switched off (“dark silicon”). The GPU is that answer, built for problems that are parallel by nature.</p>" },
      }),
    ),
    section("Two ways to spend a die",
      prose("<p>Fred Pollack, an Intel architect, observed in 1999 that making a core twice as big (deeper pipelines, bigger caches, more speculation) gives only about √2 ≈ 1.4× its speed. Split the same area into more cores and throughput grows with their number, but only if the work splits. Drag the core size.</p>"),
      h("div.stage",
        h("div.stage-controls", slider({ label: "Area of each core (die = 64 units)", min: 0, max: 6, step: 1, value: 4, fmt: (v) => String(2 ** v), onInput: (v) => { pl.size = 2 ** v; drawP(); } })),
        h("div.stage-body", pst.el, h("div.wide-canvas", pcv.el))),
      table(["", "CPU: built for latency", "GPU: built for throughput"], [
        ["Goal", "finish one thread as soon as possible", "finish millions of threads per second"],
        ["Cores", "8–200 big cores at 3–6 GHz", "100–200 SMs at ~1.5–2 GHz, each running 32 lanes per instruction"],
        ["Transistors spent on", "branch prediction, out-of-order execution, big caches", "arithmetic units and a huge register file"],
        ["Hides memory latency with", "caches and prediction", "thousands of other threads to switch to"],
        ["Bad at", "doing a million identical things at once", "branchy, serial code with one thread"],
      ]),
      callout("note", "And AI is the perfect customer", "<p>Chapter 1 showed AI is millions of independent multiply-adds: perfectly parallel. That's why a design that gave up single-thread speed for throughput, first for graphics, became the engine of AI. Next: <a href='/learn?c=gpu-design'>design that GPU yourself</a>, one decision at a time.</p>"),
    ),
    section("In code",
      code({
        "The power model": `import numpy as np
Vt, alpha = 0.3, 1.3
K = 3 / 0.7**alpha                         # 3 GHz at 1.0 V

def f_max(V):  return K * (V - Vt)**alpha / V      # alpha-power law
def core_watts(f, V):
    return 25 * V**2 * (f / 3) + 3 * V * np.exp((V - 1) / 0.25)   # switching + leakage

for f in [1, 2, 3, 4, 5]:
    V = min(np.linspace(0.31, 2, 2000), key=lambda v: abs(f_max(v) - f))
    print(f"{f} GHz needs {V:.2f} V and {core_watts(f, V):.0f} W per core")`,
        "Dennard vs post-Dennard": `# shrink everything by k = 1.4 (one process generation)
k = 1.4
dennard = dict(V=1/k, C=1/k, f=k, per_cm2=k**2)
post    = dict(V=1.0, C=1/k, f=1.0, per_cm2=k**2)
for name, s in [("Dennard (to ~2005)", dennard), ("after 2005", post)]:
    power_per_transistor = s["C"] * s["V"]**2 * s["f"]
    print(name, "power density x", round(power_per_transistor * s["per_cm2"], 2))
# Dennard: x1.0 every generation. After: x1.4 every generation, unless clocks stop`,
      }),
    ),
  );
  drawC(); drawB(); drawP();
}
