import { h, loop, reducedMotion } from "../lib/core.js";
import { slider, stats, predict, code, callout, section, prose, table, segmented, stepper } from "../lib/widgets.js";
import { flowCanvas } from "../lib/flow.js";
import { goal, whyChain, stackMap, sticker } from "../lib/why.js";
import { walk } from "../lib/walks.js";

const k = 8.617e-5;                                           // Boltzmann's constant in eV per kelvin
const sci = (x) => { if (x < 1e-3) return "≈ 0"; const e = Math.floor(Math.log10(x)), m = x / 10 ** e; return e < 4 ? x.toFixed(0) : `${m.toFixed(1)} × 10${String(e).split("").map((c) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[+c]).join("")}`; };
// silicon: Varshni band gap, Green's (1990) effective densities of states
const gapSi = (T) => 1.17 - (4.73e-4 * T * T) / (T + 636);
const niSi = (T) => Math.sqrt(2.86e19 * 3.1e19 * (T / 300) ** 3) * Math.exp(-gapSi(T) / (2 * k * T));
const MAT = {
  metal: { name: "Copper (a metal)", atoms: 8.5e22, gap: () => 0 },
  si: { name: "Silicon (a semiconductor)", atoms: 5.0e22, gap: gapSi },
  diamond: { name: "Diamond (an insulator)", atoms: 1.76e23, gap: () => 5.47 },
};

function canvas(width, height, label) {
  const fc = flowCanvas({ width, height, label });
  fc.paint = (fn) => { fc.redraw = fn; fn(); };
  return fc;
}

export default function (root, { id }) {
  // ---------- 1. the crystal and its bands ----------
  const st = { mat: "si", T: 300, dope: "none", ND: 16 };
  const cc = canvas(880, 300, "A crystal with its free electrons, and its energy bands");
  const cst = stats([{ label: "Band gap" }, { label: "Thermal energy kT" }, { label: "Free electrons per cm³" }, { label: "One free carrier per…" }]);
  const doping = segmented({ label: "Doping", options: [{ value: "none", label: "Pure" }, { value: "n", label: "n-type (phosphorus)" }, { value: "p", label: "p-type (boron)" }], value: st.dope, onChange: (v) => { st.dope = v; calc(); } });
  const ndSl = slider({ label: "Dopant atoms per cm³", min: 13, max: 19, step: 0.5, value: st.ND, fmt: (v) => sci(10 ** v), onInput: (v) => { st.ND = v; calc(); } });
  let carriers = { n: 0, p: 0 }, parts = [];
  function calc() {
    const m = MAT[st.mat], T = st.T, Eg = m.gap(T);
    doping.style.display = ndSl.style.display = st.mat === "si" ? "" : "none";
    if (st.mat === "metal") carriers = { n: m.atoms, p: 0 };
    else {
      const ni = st.mat === "si" ? niSi(T) : Math.sqrt(1e20 * 1e20) * Math.exp(-Eg / (2 * k * T));
      const N = st.mat === "si" && st.dope !== "none" ? 10 ** st.ND : 0;
      const maj = N / 2 + Math.sqrt((N / 2) ** 2 + ni * ni), min = (ni * ni) / maj;   // n·p = ni², always
      carriers = st.dope === "p" && st.mat === "si" ? { n: min, p: maj } : { n: maj, p: min };
    }
    const total = carriers.n + carriers.p;
    cst.set(0, st.mat === "metal" ? "none: bands overlap" : `${Eg.toFixed(2)} eV`);
    cst.set(1, `${(k * T * 1000).toFixed(1)} meV  (gap = ${st.mat === "metal" ? "0" : Math.round(Eg / (k * T))} × kT)`);
    cst.set(2, `${sci(carriers.n)}${carriers.p > carriers.n ? ` (+ ${sci(carriers.p)} holes)` : ""}`, total > 1e20 ? "good" : total > 1e14 ? "warn" : "bad");
    cst.set(3, total < 1e-3 ? "never: effectively zero" : `${sci(m.atoms / total)} atoms`);
    // particles: the count grows with the log of the carrier density, so 10 orders of magnitude fit on screen
    const want = (c) => Math.max(0, Math.min(70, Math.round((Math.log10(Math.max(c, 1)) - 8) * 4.4)));
    const target = { n: st.mat === "metal" ? 90 : want(carriers.n), p: st.mat === "metal" ? 0 : want(carriers.p) };
    parts = [];
    for (const kind of ["n", "p"]) for (let i = 0; i < target[kind]; i++) parts.push({ kind, x: 30 + Math.random() * 450, y: 40 + Math.random() * 220, vx: (Math.random() - 0.5) * 60, vy: (Math.random() - 0.5) * 60 });
    draw();
  }
  function draw() {
    cc.paint(() => {
      cc.clear(); const c = cc.colors, m = MAT[st.mat];
      // crystal: atoms on a grid, bonds between them
      const atomC = st.mat === "metal" ? c.c4 : st.mat === "diamond" ? c["ink-3"] : c.c1;
      for (let i = 0; i < 12; i++) for (let j = 0; j < 6; j++) {
        const x = 40 + i * 40, y = 50 + j * 40;
        if (i < 11) cc.line(x, y, x + 40, y, { color: c.line, lw: 3 });
        if (j < 5) cc.line(x, y, x, y + 40, { color: c.line, lw: 3 });
      }
      for (let i = 0; i < 12; i++) for (let j = 0; j < 6; j++) {
        const dop = st.mat === "si" && st.dope !== "none" && (i * 7 + j * 3) % 23 === 5;
        cc.dot(40 + i * 40, 50 + j * 40, 9, dop ? (st.dope === "n" ? c.c2 : c.c5) : atomC);
      }
      for (const p of parts) {
        if (p.kind === "n") cc.dot(p.x, p.y, 4, c.c4);
        else { cc.ctx.beginPath(); cc.ctx.arc(p.x, p.y, 4.5, 0, 7); cc.ctx.strokeStyle = c.c5; cc.ctx.lineWidth = 2; cc.ctx.stroke(); }
      }
      cc.text(m.name, 40, 18, { size: 12, color: c["ink-2"] });
      cc.text("● free electron   ○ hole (a missing electron)", 480, 18, { size: 11, color: c["ink-3"], align: "right" });
      // band diagram
      const bx = 560, bw = 300, top = 30, bot = 280, Eg = m.gap(st.T), scale = 34, mid = (top + bot) / 2;
      const ecY = st.mat === "metal" ? mid + 10 : mid - (Eg * scale) / 2, evY = st.mat === "metal" ? mid - 10 : mid + (Eg * scale) / 2;
      cc.text("ENERGY ↑", bx, top - 12, { size: 10.5, color: c["ink-3"], mono: true });
      cc.box(bx, top + 4, bw, Math.max(8, ecY - top - 4), { fill: `color-mix(in srgb, ${c.c4} 12%, ${c.surface})`, stroke: c.line, r: 6 });
      cc.box(bx, evY, bw, bot - evY, { fill: `color-mix(in srgb, ${c.c1} 45%, ${c.surface})`, stroke: c.line, r: 6 });
      cc.text("conduction band: electrons here can move", bx + bw / 2, top + 16, { size: 11, color: c["ink-2"], align: "center" });
      cc.text("valence band: full, electrons locked in bonds", bx + bw / 2, bot - 14, { size: 11, color: c.ink, align: "center" });
      if (st.mat !== "metal") {
        cc.line(bx + bw + 6, ecY, bx + bw + 6, evY, { color: c.ink, lw: 1.5 });
        cc.text(`gap ${Eg.toFixed(2)} eV`, bx + bw / 2, (ecY + evY) / 2, { size: 13, color: c.ink, align: "center" });
        const nn = Math.min(26, parts.filter((p) => p.kind === "n").length), nh = Math.min(26, parts.filter((p) => p.kind === "p").length);
        for (let i = 0; i < nn; i++) cc.dot(bx + 14 + ((i * 53) % (bw - 28)), ecY - 8 - ((i * 7) % 3) * 6, 3.5, c.c4);
        for (let i = 0; i < nh; i++) { cc.ctx.beginPath(); cc.ctx.arc(bx + 14 + ((i * 53) % (bw - 28)), evY + 8 + ((i * 7) % 3) * 6, 3.5, 0, 7); cc.ctx.strokeStyle = c.c5; cc.ctx.lineWidth = 2; cc.ctx.stroke(); }
      } else cc.text("bands overlap: electrons always free", bx + bw / 2, mid, { size: 13, color: c.ink, align: "center" });
    });
  }
  const anim = loop((dt) => {
    for (const p of parts) {
      p.vx += (Math.random() - 0.5) * 300 * dt; p.vy += (Math.random() - 0.5) * 300 * dt;
      p.vx = Math.max(-70, Math.min(70, p.vx)); p.vy = Math.max(-70, Math.min(70, p.vy));
      p.x += p.vx * dt; p.y += p.vy * dt;
      if (p.x < 25 || p.x > 485) p.vx *= -1; if (p.y < 35 || p.y > 265) p.vy *= -1;
      p.x = Math.max(25, Math.min(485, p.x)); p.y = Math.max(35, Math.min(265, p.y));
    }
    cc.redraw?.();
  }, cc.el);
  if (!reducedMotion()) anim.start();

  // ---------- 2. the p-n junction ----------
  const jn = { V: 0 };
  const jc = canvas(880, 260, "A p-n junction under a voltage, and its current");
  const jst = stats([{ label: "Barrier electrons must climb" }, { label: "Share of electrons hot enough" }, { label: "Current" }]);
  const Vbi = 0.72, Is = 1e-14, Vt = 0.02585;
  function drawJ() {
    const V = jn.V, I = Is * (Math.exp(V / Vt) - 1), bar = Math.max(0.02, Vbi - V);
    jst.set(0, `${bar.toFixed(2)} V`, V > 0.5 ? "good" : V < 0 ? "bad" : "warn");
    jst.set(1, `e^(−${bar.toFixed(2)}/${Vt.toFixed(4)}) ≈ ${Math.exp(-bar / Vt).toExponential(1)}`);
    jst.set(2, Math.abs(I) < 1e-9 ? `${(I * 1e15).toFixed(1)} fA (femtoamps: almost nothing)` : I < 1e-3 ? `${(I * 1e6).toFixed(2)} µA` : `${(I * 1e3).toFixed(1)} mA`, I > 1e-4 ? "good" : "bad");
    jc.paint(() => {
      jc.clear(); const c = jc.colors, x0 = 20, x1 = 500, y0 = 50, hh = 130, mid = (x0 + x1) / 2;
      const w = 34 * Math.sqrt(bar / Vbi) * 1.6;
      jc.box(x0, y0, mid - x0, hh, { fill: `color-mix(in srgb, ${c.c5} 22%, ${c.surface})`, stroke: c.line, r: 8 });
      jc.box(mid, y0, x1 - mid, hh, { fill: `color-mix(in srgb, ${c.c1} 22%, ${c.surface})`, stroke: c.line, r: 8 });
      jc.box(mid - w, y0 + 2, 2 * w, hh - 4, { fill: c.sunken, stroke: "transparent", r: 2 });
      jc.text("p side: holes", x0 + 12, y0 - 14, { size: 12, color: c["ink-2"] });
      jc.text("n side: free electrons", x1 - 12, y0 - 14, { size: 12, color: c["ink-2"], align: "right" });
      jc.text(`depletion zone: no free carriers (${(0.43 * Math.sqrt(bar / Vbi)).toFixed(2)} µm wide at 10¹⁶ doping)`, mid, y0 + hh + 18, { size: 11, color: c["ink-3"], align: "center" });
      let seed = 7; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
      for (let i = 0; i < 22; i++) {
        const hx = x0 + 12 + rnd() * (mid - w - x0 - 24), hy = y0 + 12 + rnd() * (hh - 24);
        jc.ctx.beginPath(); jc.ctx.arc(hx, hy, 4.5, 0, 7); jc.ctx.strokeStyle = c.c5; jc.ctx.lineWidth = 2; jc.ctx.stroke();
        jc.dot(mid + w + 12 + rnd() * (x1 - mid - w - 24), y0 + 12 + rnd() * (hh - 24), 4, c.c4);
      }
      const flow = Math.max(0, Math.min(1, (Math.log10(Math.max(I, 1e-15)) + 9) / 7));
      for (let i = 0; i < Math.round(flow * 8); i++) jc.line(x1 - 30 - i * 8, y0 + 20 + i * 12, x0 + 30 + i * 6, y0 + 20 + i * 12, { color: c.c4, lw: 1.5, dash: [5, 6] });
      jc.text(Math.abs(V) < 0.005 ? `no voltage: the hill is ${Vbi} V high` : V > 0 ? `forward: +${V.toFixed(2)} V lowers the hill` : `reverse: ${V.toFixed(2)} V raises the hill`, mid, 20, { size: 13, color: c.ink, align: "center" });
      // I–V curve
      const gx = 550, gy = 20, gw = 310, gh = 220, X = (v) => gx + ((v + 1) / 1.8) * gw, Y = (i) => gy + gh - 20 - Math.min(1, i / 0.02) * (gh - 40);
      jc.box(gx, gy, gw, gh, { fill: c.surface, stroke: c.line, r: 8 });
      jc.line(gx + 10, Y(0), gx + gw - 10, Y(0), { color: c.line }); jc.line(X(0), gy + 10, X(0), gy + gh - 10, { color: c.line });
      let prev = null;
      for (let v = -1; v <= 0.8; v += 0.01) { const p = [X(v), Y(Math.max(0, Is * (Math.exp(v / Vt) - 1)))]; if (prev) jc.line(prev[0], prev[1], p[0], p[1], { color: c.accent, lw: 2.5 }); prev = p; }
      jc.dot(X(V), Y(Math.max(0, I)), 6, c.c2);
      jc.text("current (0–20 mA)", gx + 12, gy + 14, { size: 10.5, color: c["ink-3"], mono: true });
      jc.text("voltage →", gx + gw - 12, Y(0) + 14, { size: 10.5, color: c["ink-3"], mono: true, align: "right" });
      jc.text("nothing, then a cliff", X(-0.5), Y(0) - 24, { size: 11, color: c["ink-2"], align: "center" });
    });
  }

  root.append(
    section(null,
      goal("<p>We need a material whose ability to carry current can be <b>switched on and off by a voltage</b>, with nothing moving but electrons. Metals always conduct and glass never does, so neither can be a switch. This chapter shows why silicon is in between, and how adding one foreign atom per million turns that in-between into a knob.</p>",
        ["Carry current only when told to", "Be switched by an electric field, not a hand", "Be made pure and identical by the billion"]),
      stackMap(id),
      h("div.vx-row", sticker("lattice", { arg: "pure", title: "Pure silicon", caption: "each atom shares its 4 outer electrons in 4 bonds", px: 4 }), sticker("lattice", { arg: "n", title: "+ phosphorus (n-type)", caption: "5 outer electrons: the fifth (yellow) wanders free", px: 4 }), sticker("lattice", { arg: "p", title: "+ boron (p-type)", caption: "3 outer electrons: one bond is missing, a “hole”", px: 4 })),
    ),
    section("Metal, semiconductor, insulator",
      prose("<p>Electrons in a solid can only have certain energies, grouped into <b>bands</b>. The <b>valence band</b> is full: those electrons are locked in bonds and can't carry current (a full parking lot has no room to move). The <b>conduction band</b> above it is empty, and any electron that gets there moves freely. Between them is the <b>band gap</b>, an energy no electron can have.</p><p>The only thing that lifts electrons across the gap at room temperature is heat, and heat hands out energy in units of about <b>kT = 0.026 eV</b>. A gap of 1.12 eV is 43 kT: rarely, but not never. A gap of 5.5 eV is 210 kT: never. Change the material and the temperature and watch.</p>"),
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Material", options: [{ value: "metal", label: "Copper" }, { value: "si", label: "Silicon" }, { value: "diamond", label: "Diamond" }], value: st.mat, onChange: (v) => { st.mat = v; calc(); } }),
          slider({ label: "Temperature", min: 100, max: 700, step: 10, value: st.T, fmt: (v) => `${v} K (${v - 273} °C)`, onInput: (v) => { st.T = v; calc(); } }),
          doping, ndSl),
        h("div.stage-body", cst.el, h("div.wide-canvas", cc.el))),
      callout("tip", "The number to remember", "<p>Pure silicon at room temperature has about <b>10¹⁰ free electrons per cm³</b>, out of 5 × 10²² atoms: one per five trillion atoms. Copper has one per atom. Add phosphorus at 10¹⁶ per cm³ (one atom in five million) and silicon has a million times more free electrons than it did pure. We control conductivity by <b>recipe</b>, region by region, and that is what makes a transistor possible. (The dots on screen grow with the <i>logarithm</i> of the count, or they wouldn't fit.)</p>"),
      predict({
        chapter: id,
        question: "Heat a piece of pure silicon from 300 K (27 °C) to 400 K (127 °C), a third hotter. Its free electrons…",
        options: ["Barely change", "Go up about a third", "Go up a few hundred times", "Go up a billion times"],
        answer: 2,
        explain: "<p>The share of electrons with enough energy to jump the gap goes as e<sup>−Eg/2kT</sup>. At 300 K that's e<sup>−21.8</sup>; at 400 K it's e<sup>−15.9</sup> (the gap also shrinks a little when hot): a factor of about 340, and about 500 once the number of available states is included. Try it on the slider. That exponential sensitivity to temperature is why chips leak more when hot, and why doped regions stop working above a few hundred °C, when heat-made carriers outnumber the dopants. A metal does the opposite: it conducts slightly <i>worse</i> when hot.</p>",
      }),
    ),
    section("Doping: the knob",
      stepper([
        { title: "Every silicon atom has 4 outer electrons", html: "<p>In the crystal each atom shares them with 4 neighbours, one bond each. Every electron is in a bond, so the crystal is a poor conductor. Silicon has to be about 99.9999999% pure (“nine nines”) for this to hold, and grown as one single crystal so the bonds line up everywhere.</p>" },
        { title: "Phosphorus has 5: n-type", html: "<p>Swap in a phosphorus atom. Four of its outer electrons bond; the fifth is held only by about 0.045 eV, less than 2 kT, so at room temperature it almost always wanders off. Each phosphorus atom donates one free electron. Silicon full of them is <b>n-type</b> (negative carriers).</p>" },
        { title: "Boron has 3: p-type", html: "<p>Swap in boron and one bond is short an electron. A neighbouring electron can hop into the gap, which leaves a gap where it came from: the missing electron moves like a positive particle, a <b>hole</b>. Silicon full of them is <b>p-type</b>.</p>" },
        { title: "Electrons and holes balance", html: "<p>Free electrons and holes meet and cancel all the time, and they settle so that n × p = n<sub>i</sub>², about 10²⁰ per cm⁶ at room temperature. Add donors and holes become rarer; add acceptors and electrons do. Real chips put n-type and p-type regions next to each other, nanometres apart, by firing dopant ions into the wafer through masks.</p>" },
      ]),
      prose("<p>Count it for yourself, atom by atom, in 3D:</p>"),
      walk("electrons"),
    ),
    section("The p-n junction: a one-way valve",
      prose("<p>Put p-type silicon against n-type. Electrons near the border fall into holes on the other side, which leaves a strip with no free carriers at all, the <b>depletion zone</b>, and a built-in voltage of about 0.7 V across it: a hill electrons must climb to cross. Push one way and you lower the hill; push the other way and you raise it.</p>"),
      h("div.stage",
        h("div.stage-controls", slider({ label: "Applied voltage", min: -1, max: 0.8, step: 0.01, value: jn.V, fmt: (v) => `${v >= 0 ? "+" : ""}${v.toFixed(2)} V`, onInput: (v) => { jn.V = v; drawJ(); } })),
        h("div.stage-body", jst.el, h("div.wide-canvas", jc.el))),
      callout("note", "Where the exponential comes from", "<p>Electrons in the n side have random thermal energies, and the share with at least E extra is e<sup>−E/kT</sup> (the Boltzmann distribution). Every 26 mV lower barrier multiplies the current by e ≈ 2.7, every 60 mV by 10. That's the diode equation, I = I<sub>s</sub>(e<sup>qV/kT</sup> − 1). The same exponential reappears in the next chapter as the reason a transistor can't switch at a lower voltage, which is the reason chips got hot, which is the reason GPUs exist.</p>"),
    ),
    section("Dig down: why can silicon be switched at all?",
      whyChain({
        claim: "Silicon can be switched between conducting and not; copper can't.",
        steps: [
          { q: "Why can't copper be switched?", a: "<p>Each copper atom gives up an electron to a shared sea: 8.5 × 10²² free electrons per cm³. An electric field from outside can't empty that sea; the electrons just rearrange at the surface within a fraction of a nanometre and screen it out.</p>" },
          { q: "Why does silicon have so few free electrons?", a: "<p>All its outer electrons are used in bonds, which fill the valence band exactly. To move, an electron needs an empty state, and the nearest ones are 1.12 eV higher, across the gap.</p>" },
          { q: "Why are there bands and gaps instead of any energy?", a: "<p>An electron bound in an atom behaves like a wave, and only certain standing-wave patterns fit, so only certain energies are allowed. Bring 10²² atoms together and each allowed level splits into 10²² levels packed so closely they form a band. Between the bands, in a periodic crystal, no wave pattern works: the gap.</p>" },
          { q: "Why does a full band carry no current?", a: "<p>Because no two electrons can be in the same state. If every state in the band is taken, no electron can change its motion without another one changing the opposite way; the total flow stays zero. Current needs empty states to move into.</p>" },
          { q: "Why can't two electrons share a state?", a: "<p>Nobody knows of a deeper reason. It is built into quantum mechanics: electrons are fermions, and the wavefunction of two identical fermions must flip sign when you swap them, which makes it zero if they're in the same state.</p>" },
        ],
        axiom: { name: "Quantum mechanics: electrons are waves, and the Pauli exclusion principle", html: "<p>Together with Coulomb's law (opposite charges attract) and thermal energy of order kT, these give bands, gaps, doping and the p-n junction. Everything above this point in the stack is engineering built on them.</p>" },
      }),
      table(["Material", "Band gap", "Free electrons per cm³ at 27 °C", "So it's…"], [
        ["Copper", "none (bands overlap)", "8.5 × 10²²", "always a conductor"],
        ["Germanium (first transistors, 1947)", "0.66 eV", "≈ 2 × 10¹³", "a semiconductor, too leaky when warm"],
        ["Silicon", "1.12 eV", "≈ 10¹⁰", "a semiconductor; its oxide is a superb insulator"],
        ["Gallium nitride (chargers, radio)", "3.4 eV", "≈ 10⁻¹⁰", "a wide-gap semiconductor, for high voltage"],
        ["Diamond", "5.5 eV", "effectively none", "an insulator"],
      ]),
      callout("note", "Why silicon won", "<p>Germanium came first, but its smaller gap leaks badly when warm. Silicon's real advantage is chemical: heat it in oxygen and it grows <b>silicon dioxide</b>, a near-perfect insulator that bonds cleanly to it. That oxide is the layer that makes the transistor in <a href='/learn?c=transistor'>the next chapter</a> work.</p>"),
    ),
    section("In code",
      code({
        "Free electrons vs temperature": `import numpy as np
k = 8.617e-5                                   # Boltzmann's constant, eV per kelvin

def gap(T):                                    # silicon's gap shrinks a little when hot
    return 1.17 - 4.73e-4 * T**2 / (T + 636)

def n_i(T):                                    # intrinsic carriers per cm^3
    Nc, Nv = 2.86e19 * (T/300)**1.5, 3.10e19 * (T/300)**1.5
    return np.sqrt(Nc * Nv) * np.exp(-gap(T) / (2 * k * T))

for T in [250, 300, 350, 400, 500]:
    print(T, f"{n_i(T):.2e}")                  # 300 K -> ~1e10, 400 K -> ~400x more

# doped: n * p = n_i^2, and n - p = N_D
N_D, T = 1e16, 300
n = N_D/2 + np.sqrt((N_D/2)**2 + n_i(T)**2)
print(f"electrons {n:.2e}, holes {n_i(T)**2 / n:.2e}")`,
        "The diode equation": `I_s, V_T = 1e-14, 0.02585                   # saturation current (A), kT/q at 300 K (V)
for V in [-1, 0, 0.3, 0.5, 0.6, 0.7]:
    I = I_s * (np.exp(V / V_T) - 1)
    print(f"{V:+.1f} V -> {I:.3e} A")
# every 60 mV forward multiplies the current by 10:
print(np.exp(0.060 / V_T))                     # ~10.2`,
      }),
    ),
  );
  calc(); drawJ();
}
