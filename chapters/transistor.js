import { h, loop, reducedMotion } from "../lib/core.js";
import { slider, stats, predict, code, callout, section, prose, table, segmented, stepper } from "../lib/widgets.js";
import { flowCanvas } from "../lib/flow.js";
import { goal, whyChain, stackMap, sticker } from "../lib/why.js";
import { walk } from "../lib/walks.js";

const kq = 8.617e-5;                      // k/q in volts per kelvin: kT/q = 25.85 mV at 300 K
const I0 = 1.8e-5;                        // sets ~1 mA when fully on (per micron of width, roughly)
/** A smooth MOSFET model (EKV-style): exponential below threshold, square law above. */
function Id(Vg, { Vt = 0.3, T = 300, n = 1.3 } = {}) {
  const pt = kq * T, Vtt = Vt - 0.001 * (T - 300);          // threshold falls ~1 mV per kelvin
  const x = (Vg - Vtt) / (2 * n * pt), f = x > 30 ? x : Math.log1p(Math.exp(x));
  return I0 * f * f;
}
const fmtA = (I) => (I >= 1e-3 ? `${(I * 1e3).toFixed(2)} mA` : I >= 1e-6 ? `${(I * 1e6).toFixed(1)} µA` : I >= 1e-9 ? `${(I * 1e9).toFixed(1)} nA` : `${(I * 1e12).toFixed(1)} pA`);

function canvas(width, height, label) {
  const fc = flowCanvas({ width, height, label });
  fc.paint = (fn) => { fc.redraw = fn; fn(); };
  return fc;
}

export default function (root, { id }) {
  // ---------- 1. the MOSFET ----------
  const m = { Vg: 0.2, T: 300, Vt: 0.3 };
  const mc = canvas(880, 300, "An n-channel MOSFET in cross-section, and its current against gate voltage");
  const mst = stats([{ label: "Drain current now" }, { label: "Leak with the gate at 0 V" }, { label: "On/off ratio (0 V → now)" }, { label: "Gate voltage per 10× current" }]);
  let dots = [];
  for (let i = 0; i < 40; i++) dots.push({ x: Math.random(), y: Math.random() });
  function drawM() {
    const opt = { Vt: m.Vt, T: m.T }, I = Id(m.Vg, opt), off = Id(0, opt), sw = 1.3 * kq * m.T * Math.LN10 * 1000;
    mst.set(0, fmtA(I), I > 1e-4 ? "good" : I > 1e-7 ? "warn" : "bad");
    mst.set(1, fmtA(off), off < 1e-9 ? "good" : off < 1e-8 ? "warn" : "bad");
    const r = I / off, e = Math.floor(Math.log10(r)); mst.set(2, e < 3 ? `${r.toFixed(r < 10 ? 1 : 0)} ×` : `${(r / 10 ** e).toFixed(1)} × 10${String(e).split("").map((d) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[+d]).join("")}`, e >= 4 ? "good" : "warn");
    mst.set(3, `${sw.toFixed(0)} mV (ideal ${(kq * m.T * Math.LN10 * 1000).toFixed(0)})`);
    mc.paint(() => {
      mc.clear(); const c = mc.colors, x0 = 20, x1 = 500, sub = 150, lvl = Math.max(0, Math.min(1, (Math.log10(I) + 11) / 8));
      mc.box(x0, sub, x1 - x0, 130, { fill: `color-mix(in srgb, ${c.c5} 18%, ${c.surface})`, stroke: c.line, r: 6 });
      mc.text("p-type silicon (holes)", (x0 + x1) / 2, 262, { size: 11, color: c["ink-3"], align: "center" });
      mc.box(x0, sub, 120, 50, { fill: `color-mix(in srgb, ${c.c1} 40%, ${c.surface})`, stroke: c.line, r: 6 });
      mc.box(x1 - 120, sub, 120, 50, { fill: `color-mix(in srgb, ${c.c1} 40%, ${c.surface})`, stroke: c.line, r: 6 });
      mc.text("source (n+)", x0 + 60, sub + 25, { size: 11.5, align: "center" }); mc.text("drain (n+)", x1 - 60, sub + 25, { size: 11.5, align: "center" });
      mc.box(x0 + 130, sub - 12, x1 - x0 - 260, 10, { fill: c.surface, stroke: c["ink-3"], r: 2 });
      mc.box(x0 + 130, sub - 70, x1 - x0 - 260, 56, { fill: `color-mix(in srgb, ${c.c4} ${Math.round(m.Vg * 70)}%, ${c.sunken})`, stroke: c["ink-3"], r: 6 });
      mc.text(`gate: ${m.Vg.toFixed(2)} V`, (x0 + x1) / 2, sub - 42, { size: 13, align: "center" });
      mc.text("oxide, ~1 nm: an insulator, so no current flows into the gate", (x0 + x1) / 2, sub - 84, { size: 10.5, color: c["ink-3"], align: "center" });
      // the channel: electrons drawn to the surface under the gate
      mc.box(x0 + 120, sub + 1, x1 - x0 - 240, 4 + lvl * 16, { fill: `color-mix(in srgb, ${c.c1} ${Math.round(lvl * 70)}%, transparent)`, stroke: "transparent", r: 2 });
      const nd = Math.round(lvl * 40);
      for (let i = 0; i < nd; i++) mc.dot(x0 + 120 + dots[i].x * (x1 - x0 - 240), sub + 3 + dots[i].y * (2 + lvl * 14), 3, c.c4);
      mc.text(lvl > 0.75 ? "channel formed: electrons connect source to drain" : lvl > 0.35 ? "a thin trickle: below threshold, current is exponential" : "no channel: two p-n junctions back to back block current", (x0 + x1) / 2, sub + 70, { size: 11.5, color: c.ink, align: "center" });
      mc.text("drain at +0.8 V", x1 - 60, sub - 14, { size: 10.5, color: c["ink-3"], align: "center" });
      // log current vs gate voltage
      const gx = 545, gy = 14, gw = 320, gh = 270, X = (v) => gx + 34 + (v / 1) * (gw - 44), Y = (i) => gy + gh - 26 - ((Math.log10(i) + 12) / 10) * (gh - 50);
      mc.box(gx, gy, gw, gh, { fill: c.surface, stroke: c.line, r: 8 });
      for (let d = -12; d <= -3; d += 3) { mc.line(gx + 30, Y(10 ** d), gx + gw - 8, Y(10 ** d), { color: c.line }); mc.text(`10${String(d).replace('-', '⁻').split('').map((q) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+q] ?? q).join('')}`, gx + 6, Y(10 ** d), { size: 9.5, color: c["ink-3"], mono: true }); }
      let prev = null;
      for (let v = 0; v <= 1.0001; v += 0.01) { const p = [X(v), Y(Math.max(1e-12, Id(v, opt)))]; if (prev) mc.line(prev[0], prev[1], p[0], p[1], { color: c.accent, lw: 2.5 }); prev = p; }
      mc.line(X(m.Vt), gy + 10, X(m.Vt), gy + gh - 24, { color: c["ink-3"], dash: [3, 4] });
      mc.text("threshold", X(m.Vt) + 4, gy + 32, { size: 10, color: c["ink-3"] });
      mc.dot(X(m.Vg), Y(Math.max(1e-12, I)), 6, c.c2);
      mc.text("gate voltage → (0 to 1 V)", gx + gw - 10, gy + gh - 10, { size: 10, color: c["ink-3"], align: "right", mono: true });
      mc.text("current, amps (log scale) ↑", gx + 36, gy + 14, { size: 10, color: c["ink-3"], mono: true });
    });
  }
  const anim = loop((dt) => { const I = Id(m.Vg, m), sp = Math.max(0, Math.min(1, (Math.log10(I) + 9) / 6)); for (const d of dots) { d.x += sp * dt * 0.6; if (d.x > 1) { d.x -= 1; d.y = Math.random(); } } mc.redraw?.(); }, mc.el);
  if (!reducedMotion()) anim.start();

  // ---------- 2. short channels and why the transistor changed shape ----------
  const sc = { L: 100, kind: "planar" };
  const LAMBDA = { planar: 10, fin: 5, gaa: 3.5 };
  const sst = stats([{ label: "Gate voltage per 10× current" }, { label: "Leak per transistor (gate off)" }, { label: "Leak of 10 billion of them at 0.8 V" }]);
  function drawS() {
    const n = 1.1 + 3 * Math.exp(-sc.L / (2 * LAMBDA[sc.kind])), sw = n * 25.85 * Math.LN10;
    const off = I0 * Math.exp(-0.3 / (n * 0.02585)), W = off * 1e10 * 0.8;
    sst.set(0, `${sw.toFixed(0)} mV`, sw < 75 ? "good" : sw < 100 ? "warn" : "bad");
    sst.set(1, fmtA(off), off < 1e-9 ? "good" : off < 1e-8 ? "warn" : "bad");
    sst.set(2, `${W < 10 ? W.toFixed(1) : Math.round(W)} W of heat doing nothing`, W < 20 ? "good" : W < 150 ? "warn" : "bad");
  }

  root.append(
    section(null,
      goal("<p>We need a <b>switch with no moving parts</b>, controlled by a voltage, that draws no current at its control, so one switch can drive many others. It has to be small enough to make 80 billion on one chip and cheap enough to flip a billion times a second. The MOSFET does all of this, and the one place it falls short of ideal, the 60 mV rule below, ends up setting the shape of every modern chip.</p>",
        ["On/off by voltage, not by touch", "No current into the control", "Shrinkable to tens of nanometres"]),
      stackMap(id),
      h("div.vx-row", sticker("planar", { arg: false, title: "Gate at 0 V: off", caption: "source and drain (blue) are cut off by p-type silicon", px: 4 }), sticker("planar", { arg: true, title: "Gate at +0.8 V: on", caption: "the gate's field pulls up a thin channel of electrons (light blue)", px: 4 })),
    ),
    section("How a field becomes a switch",
      stepper([
        { title: "Two islands that can't talk", html: "<p>Start with p-type silicon and dope two small islands n-type: the <b>source</b> and the <b>drain</b>. Between them, each island meets the p-type silicon in a p-n junction, and one of those junctions is always reverse-biased. No current flows.</p>" },
        { title: "A plate above the gap", html: "<p>Grow a very thin layer of insulating oxide over the gap and put a conductor on top: the <b>gate</b>. Gate, oxide and silicon form a capacitor. No current can flow through the oxide, ever.</p>" },
        { title: "Positive gate: pull electrons up", html: "<p>Put a positive voltage on the gate. Its field pushes holes away from the surface and draws electrons (the few that exist in p-type silicon, plus many from the source) up against the oxide.</p>" },
        { title: "Past the threshold: a channel", html: "<p>Above the <b>threshold voltage</b> (~0.3 V in modern chips) the surface has more electrons than holes: a thin n-type layer, a nanometre or two deep, now connects source to drain. Current flows. Remove the gate voltage and the channel vanishes in picoseconds.</p>" },
        { title: "Why this is the perfect switch", html: "<p>The gate draws no steady current, so one transistor's output can drive the gates of many others with no loss. The only cost of switching is <b>charging the gate's tiny capacitance</b> (and the wire to it): a fraction of a femtojoule per gate. A <b>p-channel</b> version, built the other way round, turns on when its gate is <i>low</i>. Pair the two and you get CMOS, the next chapter.</p>" },
      ]),
    ),
    section("Turn the gate",
      prose("<p>Slide the gate voltage and watch the current on a log scale, where each grid line is 1,000× more. Below the threshold, current doesn't stop: it falls off as a straight line on this plot, an exponential, and the steepness of that line is the most important number in chip design.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Gate voltage", min: 0, max: 1, step: 0.01, value: m.Vg, fmt: (v) => `${v.toFixed(2)} V`, onInput: (v) => { m.Vg = v; drawM(); } }),
          slider({ label: "Threshold voltage", min: 0.15, max: 0.5, step: 0.01, value: m.Vt, fmt: (v) => `${v.toFixed(2)} V`, onInput: (v) => { m.Vt = v; drawM(); } }),
          slider({ label: "Temperature", min: 250, max: 400, step: 5, value: m.T, fmt: (v) => `${v} K (${v - 273} °C)`, onInput: (v) => { m.T = v; drawM(); } })),
        h("div.stage-body", mst.el, h("div.wide-canvas", mc.el))),
      callout("tip", "The trade you can't escape", "<p>Lower the threshold and the transistor turns on harder at a low supply voltage (fast, low energy) but leaks more when “off”. Raise it and it leaks less but needs a higher supply. Since every decade of on/off ratio costs at least 60 mV of gate voltage at room temperature, and you want 4 to 6 decades, supplies have been stuck around <b>0.7–0.9 V</b> since the mid-2000s. Heat makes it worse: try 400 K.</p>"),
      predict({
        chapter: id,
        question: "A chip warms from 50 °C to 100 °C. Roughly what happens to the leakage current of its “off” transistors?",
        options: ["Nothing: off is off", "It rises about 15%, like the temperature in kelvin", "It rises several times over", "It falls, because hot electrons scatter more"],
        answer: 2,
        explain: "<p>Two effects stack: kT grows (the Boltzmann tail gets fatter) and the threshold voltage drops by roughly 1 mV per degree. Leakage is exponential in both, so 50 °C hotter means roughly 5–10× more leakage, which makes more heat, which makes more leakage. Chips have thermal limits and throttling partly to stop this loop running away.</p>",
      }),
      prose("<p>Now step through the subthreshold maths, one gate voltage at a time:</p>"),
      walk("transistor"),
    ),
    section("Dig down: why 60 millivolts?",
      whyChain({
        claim: "A transistor needs at least about 60 mV more on its gate for every 10× more current, at room temperature.",
        steps: [
          { q: "What does the gate actually do?", a: "<p>Below threshold, the gate doesn't open a pipe. It <b>lowers an energy hill</b> between source and drain; electrons that happen to have enough energy hop over it, the same as in the p-n junction. More gate voltage, lower hill.</p>" },
          { q: "Why does lowering the hill give an exponential?", a: "<p>Electrons in the source have a spread of random thermal energies. The number with at least E more than average falls as e<sup>−E/kT</sup>. Lower the hill by ΔE and the number that can cross rises by e<sup>ΔE/kT</sup>.</p>" },
          { q: "So where does 60 mV come from?", a: "<p>10× more current needs e<sup>ΔE/kT</sup> = 10, so ΔE = kT · ln 10. At 300 K, kT is 25.85 meV, so ΔE = 59.5 meV, which takes 59.5 mV on the gate if the gate controlled the hill perfectly. Real gates share control with the drain and the substrate, so real transistors manage 65–90 mV.</p>" },
          { q: "Why can't a cleverer design beat it?", a: "<p>Any switch that works by electrons going <i>over</i> a barrier is bound by the thermal tail: the gate only moves the barrier, it can't change how the electrons' energies are spread. Beating it needs a different mechanism, such as tunnelling <i>through</i> the barrier (tunnel FETs) or an amplifying gate layer (negative-capacitance FETs). Both are research, not products.</p>" },
          { q: "Why do electrons' energies spread out like that?", a: "<p>Because they're constantly bumping into the vibrating crystal and each other, trading energy at random. In equilibrium, the most likely way to share a fixed total of energy among a huge number of particles gives exactly this exponential tail.</p>" },
        ],
        axiom: { name: "Statistical mechanics: the Boltzmann factor e^(−E/kT)", html: "<p>At temperature T, a state with extra energy E is e<sup>−E/kT</sup> times as likely. This follows from counting: there are overwhelmingly more ways to share energy that look like this than any other way. It's the same law that sets how fast chemical reactions run. In chips it means <b>voltage can't go much below ~0.7 V</b>, so energy per switch (∝ V²) stopped falling, which is the <a href='/learn?c=power-wall'>power wall</a>.</p>" },
      }),
    ),
    section("Smaller, and a new shape",
      prose("<p>Shrinking a transistor makes it faster and cheaper, but a short channel lets the <b>drain</b> reach under the gate and pull the hill down too. The gate loses control, the subthreshold slope gets worse than 60 mV, and leakage explodes. The fix was to give the gate more of the channel to hold: first by standing the channel up as a <b>fin</b> with the gate on three sides (Intel, 22 nm, 2011), then by slicing it into stacked <b>nanosheets</b> with the gate all around (Samsung 3 nm, 2022; TSMC N2, 2025).</p>"),
      h("div.vx-row", sticker("planar", { arg: true, title: "Planar (to ~2011)", caption: "gate on top only", px: 3 }), sticker("finfet", { title: "FinFET (2011–)", caption: "channel is a fin; gate wraps three sides", px: 3 }), sticker("gaa", { title: "Gate-all-around (2022–)", caption: "stacked sheets; gate on all four sides", px: 3 })),
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Structure", options: [{ value: "planar", label: "Planar" }, { value: "fin", label: "FinFET" }, { value: "gaa", label: "Gate-all-around" }], value: sc.kind, onChange: (v) => { sc.kind = v; drawS(); } }),
          slider({ label: "Gate length", min: 5, max: 200, step: 1, value: sc.L, fmt: (v) => `${v} nm`, onInput: (v) => { sc.L = v; drawS(); } })),
        h("div.stage-body", sst.el)),
      callout("note", "This model is a teaching sketch", "<p>The leakage numbers come from a simple formula (slope degrades as e<sup>−L/2λ</sup>, with a shorter control length λ for better-wrapped gates), not from a device simulator. The direction and the size of the effect match real history: by the late planar era, leakage was a large share of a chip's power. Also note that node names like “3 nm” stopped being physical lengths long ago; real gate pitches are around 45–50 nm.</p>"),
      table(["Year", "Milestone", "Transistors on a chip"], [
        ["1947", "First transistor (point-contact, germanium), Bardeen, Brattain and Shockley at Bell Labs", "1"],
        ["1959", "The MOSFET, Mohamed Atalla and Dawon Kahng at Bell Labs", "—"],
        ["1963", "CMOS: pairing n- and p-channel transistors (Frank Wanlass, Fairchild)", "—"],
        ["1971", "Intel 4004, 10 µm process", "2,300"],
        ["2010", "GTX 580 (AlexNet's GPU), 40 nm", "3.0 billion"],
        ["2011", "FinFET in mass production (Intel, 22 nm)", "~1.4 billion (a CPU)"],
        ["2022", "H100, TSMC 4N", "80 billion"],
        ["2024", "B200: two dies joined in one package", "208 billion"],
      ]),
    ),
    section("In code",
      code({
        "A transistor model": `import numpy as np
kT_q = 8.617e-5 * 300                    # 25.85 mV at room temperature

def drain_current(Vg, Vt=0.3, n=1.3, I0=1.8e-5):
    x = (Vg - Vt) / (2 * n * kT_q)
    return I0 * np.log1p(np.exp(x)) ** 2   # exp below threshold, square law above

for Vg in [0, 0.1, 0.2, 0.3, 0.5, 0.8]:
    print(f"{Vg:.1f} V  {drain_current(Vg):.2e} A")

swing = kT_q * np.log(10)                # 59.5 mV per decade: the Boltzmann limit
print(f"best possible: {1000 * swing:.1f} mV per 10x current")`,
        "Why voltage can't drop": `# want: on/off ratio of 1e5 between 0 V and the supply
decades = 5
for n in [1.0, 1.3, 1.6]:                 # 1.0 = a perfect gate
    need = decades * n * kT_q * np.log(10)
    print(f"n={n}: {need:.2f} V just to go from off to threshold")
# plus ~0.3-0.4 V of overdrive above threshold for speed -> ~0.7-0.9 V supplies`,
      }),
    ),
  );
  drawM(); drawS();
}
