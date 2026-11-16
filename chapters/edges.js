import { h } from "../lib/core.js";
import { slider, stats, predict, code, callout, section, prose, table, segmented } from "../lib/widgets.js";
import { flowCanvas } from "../lib/flow.js";
import { goal, whyChain, sticker } from "../lib/why.js";
import { walk } from "../lib/walks.js";

function canvas(width, height, label) {
  const fc = flowCanvas({ width, height, label });
  fc.paint = (fn) => { fc.redraw = fn; fn(); };
  return fc;
}
const kB = 1.380649e-23, HBAR = 1.054571817e-34, ME = 9.1093837e-31, EV = 1.602176634e-19;
const sup = (n) => String(n).replace("-", "⁻").split("").map((d) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[+d] ?? d).join("");
const sci = (v, d = 1) => { if (v === 0) return "0"; const e = Math.floor(Math.log10(Math.abs(v))), m = v / 10 ** e; return `${m.toFixed(d)} × 10${sup(e)}`; };
const fmtJ = (j) => { for (const [s, u] of [[1e-12, "pJ"], [1e-15, "fJ"], [1e-18, "aJ"], [1e-21, "zJ"]]) if (j >= s) return `${(j / s).toFixed(j / s < 10 ? 2 : 0)} ${u}`; return `${(j / 1e-24).toFixed(2)} yJ`; };

// how far each layer is from its wall: [key, label, today, wall, unit, text]
const WALLS = [
  ["energy", "Energy to switch a gate", 1e-16, 2.87e-21, "J", "today ~0.1 fJ per gate switch; Landauer's floor for erasing a bit at room temperature is 2.9 zJ. Over four orders of magnitude of room: energy is not where physics stops us yet."],
  ["swing", "Steepness of the switch", 68, 60, "mV per 10× current", "today ~65–70 mV per decade; Boltzmann's floor at room temperature is 60. We're at the wall: this is why the supply voltage stopped falling (see the transistor chapter)."],
  ["size", "Thickness of the channel", 5, 0.6, "nm", "today's nanosheets are ~5 nm thick, about ten silicon unit cells; one layer of a 2D material is ~0.6 nm. One order of magnitude left, and the last steps are the hardest."],
  ["heat", "Heat out of the chip", 86, 1000, "W/cm²", "an H100 averages ~86 W/cm²; hot spots run several times higher. Microchannel liquid cooling has removed ~1 kW/cm² in the lab. Room exists, at a large cost in plumbing."],
  ["wire", "Time for a signal to cross 1 mm", 50, 7, "ps", "a buffered on-chip wire takes ~30–60 ps per mm; light in the glass around it would take ~7. Wires, not transistors, now set much of a chip's speed and energy."],
];

const CAND = [
  ["Stacked transistors (CFET)", "more transistors per mm² by stacking n on p", "heat and wiring in 3D", "in development for late 2020s nodes"],
  ["2D-material channels (MoS₂, WSe₂)", "channels one molecule thick, still good switches", "making good contacts and defect-free layers on 300 mm wafers", "research"],
  ["Steep-slope switches (tunnel FET, negative capacitance)", "beating 60 mV per decade, so a lower supply voltage", "too little current when on; none beats the MOSFET at scale yet", "research"],
  ["Co-packaged optics", "moving bits between chips with light at a fraction of the energy", "laser power, reliability, packaging", "early products in network switches"],
  ["Photonic matrix multiply", "multiply-adds done by interference, almost free", "converting to and from electronics, limited precision, no cheap nonlinearity", "start-ups and research"],
  ["Analog in-memory compute", "weights never move: compute where they're stored", "noise, drift and the cost of converters", "research chips"],
  ["Superconducting logic (SFQ)", "switching at ~10⁻¹⁹ J, very fast", "cooling to 4 K costs ~1,000 W per watt removed; density", "niche"],
  ["Reversible / adiabatic logic", "computing without erasing, under Landauer's floor", "slow and complex; little practical gain so far", "research"],
  ["Quantum computers", "exponential speed-ups for some problems (factoring, chemistry)", "error correction; no known big speed-up for dense matrix multiply", "early machines"],
];

const OPEN = [
  ["Can any switch beat 60 mV per decade and still carry enough current?", "If yes, supply voltage could fall again, and with it energy per operation (∝ V²). Decades of attempts have not produced a winner."],
  ["What replaces copper wires as they get thinner?", "Below ~10 nm wide, copper's resistance climbs because electrons scatter off the surfaces. Ruthenium, molybdenum and backside power delivery are being tried."],
  ["How do we stop memory falling further behind arithmetic?", "Compute per chip grows faster than bandwidth to memory (the memory wall). Stacking memory on logic helps; nobody knows how far."],
  ["How do we cool 1–2 kW chips densely, at data-centre scale?", "Liquid cold plates are now standard; immersion and on-chip microchannels exist but are hard to build and maintain."],
  ["How close to Landauer can practical computers get?", "Reversible logic avoids the bound in principle; error correction, speed and wiring may make it irrelevant in practice. Unknown."],
  ["How low can precision go?", "Training has moved from 32 to 16 to 8 bits; 4-bit and ternary models exist. Where accuracy breaks, and why, is not understood."],
  ["How does the brain do so much with ~20 W?", "Its energy per synaptic event (~1–10 fJ, rough) isn't far below a chip's per operation; its architecture (memory and compute in the same place, sparse events) is the difference. How to copy that usefully is open."],
];

export default function (root, { id }) {
  // ---------- 1. every layer has a wall ----------
  const wl = { k: "energy" };
  const wcv = canvas(880, 300, "How many orders of magnitude separate today's chips from the physical limit, layer by layer");
  wcv.paint(() => {
    wcv.clear(); const c = wcv.colors, x0 = 270, x1 = 820, span = 5, X = (o) => x0 + (o / span) * (x1 - x0);
    for (let o = 0; o <= span; o++) { wcv.line(X(o), 16, X(o), 246, { color: c.line, lw: 1 }); wcv.text(o === 0 ? "at the wall" : `${10 ** o}×`.replace("100000×", "100,000×").replace("10000×", "10,000×").replace("1000×", "1,000×"), X(o), 258, { size: 9.5, align: "center", color: c["ink-3"], mono: true }); }
    wcv.box(X(0) - 3, 14, 6, 234, { fill: c.bad, stroke: "transparent", r: 2 });
    WALLS.forEach(([k, n, now, wall], i) => {
      const y = 26 + i * 44, gap = Math.abs(Math.log10(now / wall)), on = k === wl.k;
      wcv.text(n, x0 - 14, y + 12, { size: 11.5, align: "right", color: on ? c.ink : c["ink-2"], weight: on ? 700 : 600 });
      wcv.box(X(0) + 4, y, Math.max(6, X(gap) - X(0) - 4), 24, { fill: `color-mix(in srgb, ${gap < 0.3 ? c.bad : gap < 1.2 ? c.warn : c.good} ${on ? 70 : 40}%, ${c.surface})`, stroke: on ? c.accent : "transparent", lw: 2, r: 5 });
      wcv.text(`${gap < 0.3 ? (now / wall).toFixed(2) : Math.round(Math.max(now / wall, wall / now)).toLocaleString()}× of room`, Math.max(X(gap), X(0) + 10) + 10, y + 12, { size: 10.5, mono: true, color: c["ink-2"] });
    });
    wcv.text("room left before the physical wall (log scale) · red line: the wall", 545, 278, { size: 10, align: "center", color: c["ink-3"] });
  });
  const wnote = h("p.muted");
  const setW = () => { const w = WALLS.find((x) => x[0] === wl.k); wnote.innerHTML = `<b>${w[1]}:</b> ${w[5]}`; };

  // ---------- 2. tunnelling ----------
  const tn = { d: 1.2, kind: "oxide" };
  const BAR = { oxide: { dE: 3.1, m: 0.5, name: "gate oxide (SiO₂): 3.1 eV high" }, channel: { dE: 0.3, m: 0.2, name: "an “off” channel: ~0.3 eV high" } };
  const tst = stats([{ label: "Chance an electron leaks through" }, { label: "10× less leakage for every" }, { label: "Atomic layers thick (≈0.3 nm each)" }]);
  const tcv = canvas(880, 260, "An electron's wave decaying through a thin barrier, and the leak probability against thickness");
  function drawT() {
    const b = BAR[tn.kind], kap = Math.sqrt(2 * b.m * ME * b.dE * EV) / HBAR / 1e9, T = (d) => Math.exp(-2 * kap * d), Tn = T(tn.d);
    tst.set(0, sci(Tn), Tn > 1e-4 ? "bad" : Tn > 1e-9 ? "warn" : "good"); tst.set(1, `${(Math.LN10 / (2 * kap)).toFixed(2)} nm`); tst.set(2, (tn.d / 0.3).toFixed(1));
    tcv.paint(() => {
      tcv.clear(); const c = tcv.colors, bx = 200, bw = tn.d * 60, mid = 120, A = 70;
      tcv.box(bx, 26, bw, 188, { fill: `color-mix(in srgb, ${c.c6} 25%, ${c.surface})`, stroke: c.c6, lw: 1.5, r: 3 });
      tcv.text(`${tn.d.toFixed(1)} nm`, bx + bw / 2, 16, { size: 10.5, align: "center", mono: true });
      tcv.text("electron arrives", 60, 232, { size: 10.5, align: "center", color: c["ink-3"] }); tcv.text("barrier", bx + bw / 2, 232, { size: 10.5, align: "center", color: c.c6 });
      // incoming wave, decay inside, transmitted wave (amplitude √T, drawn on a compressed scale so it stays visible)
      const amp = (x) => (x < bx ? A : x <= bx + bw ? A * Math.exp(-kap * ((x - bx) / 60)) : A * Math.sqrt(Tn));
      const vis = (a) => (a <= 0 ? 0 : A * Math.max(0.02, 1 + Math.log10(a / A) / 6));
      const ctx = tcv.ctx; ctx.beginPath();
      for (let x = 20; x <= 500; x += 1.5) { const a = vis(amp(x)), y = x >= bx && x <= bx + bw ? mid - a : mid - a * Math.cos((x - 20) / 9); x === 20 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); }
      ctx.strokeStyle = c.accent; ctx.lineWidth = 2; ctx.stroke();
      tcv.line(20, mid, 500, mid, { color: c.line, lw: 1, dash: [2, 4] });
      tcv.text(`leaks through: ${sci(Tn)}`, 420, 232, { size: 10.5, align: "center", color: Tn > 1e-6 ? c.bad : c["ink-2"] });
      tcv.text("height on a log scale", 420, 24, { size: 9.5, align: "center", color: c["ink-3"], mono: true });
      // log T against thickness
      const gx = 540, gw = 320, gy = 20, gh = 200, Xd = (d) => gx + 30 + ((d - 0.3) / 4.7) * (gw - 40), Yt = (t) => gy + 10 + (-Math.log10(t) / 30) * (gh - 30);
      tcv.box(gx, gy, gw, gh + 20, { fill: c.surface, stroke: c.line, r: 8 });
      for (let e = 0; e <= 30; e += 10) { tcv.line(gx + 28, Yt(10 ** -e), gx + gw - 8, Yt(10 ** -e), { color: c.line, lw: 1 }); tcv.text(`10${sup(-e)}`, gx + 6, Yt(10 ** -e), { size: 9.5, color: c["ink-3"], mono: true }); }
      let prev = null; for (let d = 0.3; d <= 5.001; d += 0.05) { const t = Math.max(1e-30, T(d)), p = [Xd(d), Yt(t)]; if (prev) tcv.line(prev[0], prev[1], p[0], p[1], { color: c.accent, lw: 2.2 }); prev = p; }
      tcv.dot(Xd(tn.d), Yt(Math.max(1e-30, Tn)), 6, c.c2);
      if (tn.kind === "oxide") { tcv.line(Xd(1.2), gy + 8, Xd(1.2), gy + gh + 8, { color: c["ink-3"], dash: [3, 4] }); tcv.text("SiO₂ in 2005: ~1.2 nm", Xd(1.2) + 4, gy + gh - 4, { size: 9.5, color: c["ink-3"] }); }
      else { tcv.line(Xd(5), gy + 8, Xd(5), gy + gh + 8, { color: c["ink-3"], dash: [3, 4] }); tcv.text("~5 nm", Xd(5) - 4, gy + gh - 4, { size: 9.5, color: c["ink-3"], align: "right" }); }
      tcv.text("barrier thickness →", gx + gw - 10, gy + gh + 12, { size: 9.5, align: "right", color: c["ink-3"], mono: true });
    });
  }

  // ---------- 3. Landauer ----------
  const ld = { T: 300, p: 20 };
  const lst = stats([{ label: "Landauer: erase one bit" }, { label: "Reliability floor for this error rate" }, { label: "A gate today, above Landauer" }]);
  const lcv = canvas(880, 230, "Energy per operation on a log scale, from Landauer's limit to a GPU's multiply-add");
  function drawL() {
    const L = kB * ld.T * Math.LN2, R = kB * ld.T * Math.log(10 ** ld.p);
    lst.set(0, fmtJ(L)); lst.set(1, fmtJ(R)); lst.set(2, `${Math.round(1e-16 / L).toLocaleString()}×`);
    lcv.paint(() => {
      lcv.clear(); const c = lcv.colors, x0 = 30, x1 = 850, X = (j) => x0 + ((Math.log10(j) + 23) / 12) * (x1 - x0), y = 110;
      lcv.box(x0, y - 6, x1 - x0, 12, { fill: c.sunken, stroke: "transparent", r: 6 });
      lcv.box(x0, y - 6, X(L) - x0, 12, { fill: `color-mix(in srgb, ${c.bad} 35%, ${c.surface})`, stroke: "transparent", r: 6 });
      for (let e = -23; e <= -11; e++) { lcv.line(X(10 ** e), y + 10, X(10 ** e), y + 16, { color: c["ink-3"] }); if (e % 2 === 1) lcv.text(`10${sup(e)} J`, X(10 ** e), y + 28, { size: 9.5, align: "center", color: c["ink-3"], mono: true }); }
      const M = [[L, "Landauer kT·ln2", c.bad, 1], [R, `reliable to 1 error in 10${sup(ld.p)}`, c.warn, -1], [1e-19, "superconducting switch", c.c1, 1], [1e-16, "a CMOS gate switching", c.accent, -1], [5e-15, "brain, per synaptic event (rough)", c.c5, 1], [1.4e-12, "H100 bf16 multiply-add", c.c2, -1]];
      M.forEach(([j, n, col, up], k) => { const x = X(j), ly = up > 0 ? y - 30 - (k % 3) * 18 : y + 46 + (k % 3) * 18; lcv.line(x, y - 8 * up, x, ly + (up > 0 ? 6 : -6), { color: col, lw: 1.5 }); lcv.dot(x, y, 6, col); lcv.text(n, x, ly, { size: 10.5, align: "center", color: col, weight: 700 }); });
      lcv.text("← less energy per operation", x0, 14, { size: 10, color: c["ink-3"], mono: true }); lcv.text(`temperature ${ld.T} K`, x1, 14, { size: 10, color: c["ink-3"], mono: true, align: "right" });
    });
  }

  // ---------- 4. light is too slow ----------
  const lt = { f: 3, medium: "wire" };
  const SPEED = { vacuum: [0.3, "light in vacuum: 0.3 mm/ps"], glass: [0.15, "light in glass or fibre: ~0.15 mm/ps"], wire: [0.02, "a buffered on-chip wire: ~0.02 mm/ps (50 ps/mm)"] };
  const sst = stats([{ label: "Distance per clock tick" }, { label: "Ticks to cross an H100 die (28 mm)" }, { label: "Ticks to cross a rack (2 m, by cable)" }]);
  const scv = canvas(880, 300, "How far a signal travels in one clock tick, drawn against an H100-sized die");
  function drawS() {
    const v = SPEED[lt.medium][0], d = (v * 1000) / lt.f, die = 28;
    sst.set(0, d >= 10 ? `${(d / 10).toFixed(1)} cm` : `${d.toFixed(1)} mm`); sst.set(1, (die / d).toFixed(die / d < 10 ? 1 : 0), die / d <= 1 ? "good" : die / d < 5 ? "warn" : "bad"); sst.set(2, Math.round(2000 / ((0.2 * 1000) / lt.f)).toString());
    scv.paint(() => {
      scv.clear(); const c = scv.colors, s = 8, cx = 250, cy = 150;
      scv.box(cx - (die * s) / 2, cy - (die * s) / 2, die * s, die * s, { fill: `color-mix(in srgb, ${c.c3} 18%, ${c.surface})`, stroke: c.c3, lw: 1.5, r: 4 });
      for (let i = 1; i < 8; i++) { scv.line(cx - (die * s) / 2 + (i * die * s) / 8, cy - (die * s) / 2, cx - (die * s) / 2 + (i * die * s) / 8, cy + (die * s) / 2, { color: `color-mix(in srgb, ${c.c3} 25%, transparent)`, lw: 1 }); }
      scv.text("an H100-sized die, 28 × 28 mm", cx, cy + (die * s) / 2 + 14, { size: 10.5, align: "center", color: c["ink-3"] });
      const r = Math.min(400, d * s); scv.ctx.save(); scv.ctx.beginPath(); scv.ctx.rect(10, 6, 480, 288); scv.ctx.clip();
      scv.ctx.beginPath(); scv.ctx.arc(cx - (die * s) / 2 + 10, cy, r, 0, 7); scv.ctx.fillStyle = `color-mix(in srgb, ${c.accent} 14%, transparent)`; scv.ctx.fill(); scv.ctx.strokeStyle = c.accent; scv.ctx.lineWidth = 2; scv.ctx.stroke(); scv.ctx.restore();
      scv.dot(cx - (die * s) / 2 + 10, cy, 5, c.accent);
      const x0 = 520; scv.box(x0, 14, 346, 270, { fill: c.surface, stroke: c.line, r: 10 });
      scv.text("in one tick at " + lt.f.toFixed(1) + " GHz, a signal gets:", x0 + 14, 36, { size: 11.5 });
      Object.entries(SPEED).forEach(([k, [v2, n]], i) => { const dd = (v2 * 1000) / lt.f; scv.text(n.split(":")[0], x0 + 14, 66 + i * 44, { size: 11, color: k === lt.medium ? c.accent : c["ink-2"], weight: k === lt.medium ? 700 : 600 }); scv.text(dd >= 10 ? `${(dd / 10).toFixed(1)} cm` : `${dd.toFixed(2)} mm`, x0 + 330, 66 + i * 44, { size: 12, align: "right", mono: true, color: k === lt.medium ? c.accent : c.ink }); scv.bar(x0 + 14, 78 + i * 44, 316, 6, Math.min(1, dd / 100), k === lt.medium ? c.accent : c["ink-3"]); });
      scv.text("so a big chip can't be one clock-synchronous", x0 + 14, 214, { size: 11, color: c["ink-2"] }); scv.text("blob: it's split into regions linked by a network,", x0 + 14, 232, { size: 11, color: c["ink-2"] }); scv.text("and data that's far away costs ticks and energy", x0 + 14, 250, { size: 11, color: c["ink-2"] });
    });
  }

  // the candidates as a map: how close each is to products, coloured by the wall it attacks
  const candCv = canvas(800, 252, "The candidate technologies placed by how close they are to products, coloured by which physical wall each attacks");
  candCv.paint(() => {
    candCv.clear(); const c = candCv.colors;
    const wallCol = { size: c.c1, swing: c.c4, energy: c.good, wire: c.c6, other: c["ink-3"] };
    const cols = ["Research", "Prototypes and start-ups", "First products", "Heading for production"];
    const items = [
      ["2D-material channels", "size", 0], ["Steep-slope switches", "swing", 0], ["Reversible logic", "energy", 0],
      ["Photonic matrix multiply", "energy", 1], ["Analog in-memory compute", "wire", 1], ["Superconducting logic", "energy", 1],
      ["Co-packaged optics", "wire", 2], ["Quantum computers", "other", 2],
      ["Stacked transistors (CFET)", "size", 3],
    ];
    const cw = 190, gap = 4;
    cols.forEach((nm, k) => {
      const x = 10 + k * (cw + gap);
      candCv.box(x, 8, cw, 186, { fill: c.surface, stroke: c.line, r: 10 });
      candCv.text(nm, x + cw / 2, 26, { size: 11.5, align: "center" });
      items.filter((it) => it[2] === k).forEach(([t, w], j) => {
        const y = 44 + j * 50, col = wallCol[w];
        candCv.box(x + 10, y, cw - 20, 40, { fill: `color-mix(in srgb, ${col} 22%, ${c.surface})`, stroke: col, lw: 1.2, r: 6 });
        candCv.text(t, x + cw / 2, y + 20, { size: 10.5, align: "center" });
      });
    });
    candCv.line(20, 204, 780, 204, { color: c["ink-3"], lw: 1.5 });
    candCv.line(780, 204, 772, 199, { color: c["ink-3"], lw: 1.5 }); candCv.line(780, 204, 772, 209, { color: c["ink-3"], lw: 1.5 });
    candCv.text("closer to a chip you can buy →", 776, 216, { size: 9.5, color: c["ink-3"], align: "right" });
    [["size", "size"], ["swing", "switch steepness"], ["energy", "energy per operation"], ["wire", "moving data"], ["other", "a different kind of problem"]].forEach(([w, nm], i) => {
      const x = 20 + i * 150; candCv.box(x, 230, 12, 12, { fill: `color-mix(in srgb, ${wallCol[w]} 22%, ${c.surface})`, stroke: wallCol[w], lw: 1.2, r: 3 }); candCv.text(nm, x + 18, 236, { size: 10.5, color: c["ink-2"] });
    });
  });

  root.append(
    section(null,
      goal("<p>Every chapter so far ended at a law of physics. This one asks the question the whole industry is asking: <b>how much room is left before each law becomes a wall</b>, which walls are physics and which are only engineering, and what nobody yet knows how to do. Some limits are close (the steepness of a switch), some are far (energy per operation), and some (wires, heat) are now the real bottleneck.</p>",
        ["Know which limits are physics", "Know how far away each one is", "Know what is still unsolved"]),
      h("div.vx-row", sticker("lattice", { arg: "pure", title: "Atoms are discrete", caption: "a 5 nm channel is only about ten silicon cells thick", px: 4 }), sticker("tunnel", { title: "Electrons are waves", caption: "a thin enough wall lets some of them through", px: 4 }), sticker("heatsink", { title: "Heat has one way out", caption: "and the path can only conduct so much", px: 4 })),
    ),
    section("Every layer has a wall",
      prose("<p>For five limits, how far today's chips are from the physical floor, in orders of magnitude. Pick one to read about it.</p>"),
      h("div.stage",
        h("div.stage-controls", segmented({ label: "Limit", options: WALLS.map(([k]) => ({ value: k, label: { energy: "Energy", swing: "Switch steepness", size: "Size", heat: "Heat", wire: "Wire speed" }[k] })), value: wl.k, onChange: (v) => { wl.k = v; setW(); wcv.redraw(); } })),
        h("div.stage-body", h("div.wide-canvas", wcv.el), wnote)),
      callout("note", "The surprise", "<p>Energy, the limit people quote most, has the most room: a gate still uses about 30,000 times Landauer's floor. The tight limits are the ones nobody can engineer around: the 60 mV-per-decade switch, set by the Boltzmann distribution, and the size of atoms. That's why progress now comes from architecture (moving less data, lower precision, specialised units) more than from better transistors.</p>"),
    ),
    section("Tunnelling: when the wall is thinner than the electron",
      prose("<p>An electron is a wave, a few nanometres long in silicon. When it meets an energy barrier it can't climb, the wave doesn't stop dead: it decays exponentially inside, and if the barrier is thin enough, a little comes out the other side. The leak falls by 10× for every fraction of a nanometre of extra thickness, so a few atoms decide everything. Thin the barrier.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Barrier thickness", min: 0.3, max: 5, step: 0.1, value: tn.d, fmt: (v) => `${v.toFixed(1)} nm`, onInput: (v) => { tn.d = v; drawT(); } }),
          segmented({ label: "Barrier", options: [{ value: "oxide", label: "Gate oxide (3.1 eV)" }, { value: "channel", label: "Off channel (0.3 eV)" }], value: tn.kind, onChange: (v) => { tn.kind = v; drawT(); } })),
        h("div.stage-body", tst.el, h("div.wide-canvas", tcv.el))),
      callout("tip", "What the industry did", "<p>By 2005 the silicon dioxide under the gate was ~1.2 nm, about four atoms, and leaking badly. In 2007 Intel switched to <b>hafnium oxide</b>, which holds more charge per volt, so it can be several times thicker for the same effect: far less tunnelling. For the channel, tunnelling from source to drain becomes serious below ~5 nm of gate length, one of the reasons gate lengths have stopped shrinking much, even as node names keep falling. (Simple estimate: T ≈ e<sup>−2κd</sup>, κ = √(2m*ΔE)/ħ, with typical effective masses.)</p>"),
      predict({ chapter: id, question: "A gate oxide is thinned from 1.2 nm to 0.8 nm (about one atomic layer less). Roughly how does the tunnelling leakage change?", options: ["About 1.5× more, in proportion", "About 100× more", "No change: the voltage is the same", "It halves"], answer: 1, explain: "<p>Leakage goes as e<sup>−2κd</sup>, with 2κ ≈ 12.8 per nm for SiO₂. Removing 0.4 nm multiplies it by e<sup>5.1</sup> ≈ 170. Exponentials make single atomic layers matter, which is why the move to thicker high-k materials was forced, not chosen.</p>" }),
      prose("<p>Step through the arithmetic, one atomic layer at a time:</p>"),
      walk("edges"),
    ),
    section("The cost of forgetting",
      prose("<p>In 1961 Rolf Landauer showed that <b>erasing</b> one bit of information must release at least <b>kT·ln 2</b> of heat: 2.9 zeptojoules at room temperature. It's the second law of thermodynamics applied to memory. Separately, a bit stored with energy E is flipped by heat with odds of about e<sup>−E/kT</sup>, so <b>reliability</b> costs energy too. Change the temperature and the error rate you can tolerate.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Temperature", min: 4, max: 400, step: 1, value: ld.T, fmt: (v) => `${v} K${v < 10 ? " (liquid helium)" : v < 90 ? " (liquid nitrogen ~77 K)" : v > 290 && v < 310 ? " (room)" : ""}`, onInput: (v) => { ld.T = v; drawL(); } }),
          slider({ label: "Errors you can tolerate: 1 in 10ⁿ", min: 3, max: 30, step: 1, value: ld.p, fmt: (v) => `n = ${v}`, onInput: (v) => { ld.p = v; drawL(); } })),
        h("div.stage-body", lst.el, h("div.wide-canvas", lcv.el))),
      callout("note", "Ways around it, in principle", "<p><b>Reversible computing</b> never erases: every step can be run backwards, so Landauer's bound doesn't apply. <b>Adiabatic</b> circuits charge wires slowly to waste less. <b>Cooling</b> lowers kT, but removing heat from a 4 K fridge costs ~1,000 W per watt. All work in the lab; none yet beats plain CMOS for real workloads. The limit is real but distant: today's waste is in wires and memory traffic, not in erasing bits.</p>"),
      predict({ chapter: id, question: "A logic gate in a modern chip uses about 0.1 fJ per switch. How far is that above Landauer's limit at room temperature?", options: ["About 2× above", "About 30,000× above", "About 10¹⁵× above", "It's below the limit"], answer: 1, explain: "<p>0.1 fJ = 10⁻¹⁶ J; kT ln 2 at 300 K is 2.9 × 10⁻²¹ J. The ratio is ~35,000. A whole bf16 multiply-add on a GPU (~1.4 pJ, including moving the operands) is another 10,000× above that. Physics leaves enormous room for energy; electrons sloshing in and out of wires is where it goes.</p>" }),
    ),
    section("Light is too slow",
      prose("<p>At 3 GHz a clock tick lasts 333 picoseconds. In that time light in a vacuum covers 10 cm, but a signal on an on-chip wire, which must charge the wire's capacitance through its resistance, gets only a few millimetres. A big chip can't be kept in step as one piece. Raise the clock and pick the medium.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Clock", min: 0.5, max: 10, step: 0.1, value: lt.f, fmt: (v) => `${v.toFixed(1)} GHz`, onInput: (v) => { lt.f = v; drawS(); } }),
          segmented({ label: "Signal travels as", options: [{ value: "wire", label: "On-chip wire" }, { value: "glass", label: "Light in glass" }, { value: "vacuum", label: "Light in vacuum" }], value: lt.medium, onChange: (v) => { lt.medium = v; drawS(); } })),
        h("div.stage-body", sst.el, h("div.wide-canvas", scv.el))),
      predict({ chapter: id, question: "At 5 GHz, how far does light travel in a vacuum during one clock tick?", options: ["6 metres", "6 centimetres", "6 millimetres", "60 micrometres"], answer: 1, explain: "<p>Light covers 30 cm per nanosecond. One tick at 5 GHz is 0.2 ns: 6 cm. A real on-chip wire manages a tenth of that or less, so crossing a large die takes several ticks. Distance is time, and time is energy: that's the physical root of the memory wall in <a href='/learn?c=memory-physics'>chapter 5</a>.</p>" }),
    ),
    section("The candidates",
      prose("<p>What might carry computing past the current walls, what each would fix, and what is still unsolved. Status as of the mid-2020s.</p>"),
      h("div.stage", h("div.stage-body", h("div.wide-canvas", candCv.el))),
      table(["Idea", "Would fix", "Unsolved", "Status"], CAND),
    ),
    section("What nobody knows yet",
      table(["Open question", "Why it matters, and why it's hard"], OPEN),
      callout("tip", "Where to watch", "<p>Most gains in AI compute over the next decade are expected from architecture and arithmetic (lower precision, sparsity, less data movement, 3D stacking), not from faster transistors. If one of these open questions breaks, especially a steeper switch or cheap optics, the whole stack in this level shifts.</p>"),
    ),
    section("Dig down: why can't transistors shrink forever?",
      whyChain({
        claim: "Transistors cannot keep shrinking at the old pace: the last few nanometres are much harder than all the ones before.",
        steps: [
          { q: "Why not just keep scaling every dimension down?", a: "<p>Because matter is made of atoms. Silicon's crystal repeats every 0.54 nm, so a 5 nm channel is about ten repeats thick. You can't make a layer half an atom thick, and every atom out of place now matters.</p>" },
          { q: "Why do a few atoms matter?", a: "<p>A tiny channel contains only a handful of dopant atoms, placed at random. Five here and seven there shifts the threshold voltage noticeably, and a chip needs ten billion transistors that behave alike. Randomness that averaged out in big devices doesn't average out in small ones.</p>" },
          { q: "Why does current leak through barriers that should block it?", a: "<p>Because electrons aren't little balls: they're waves, a few nanometres long in silicon. A barrier thinner than a few wavelengths lets part of the wave through: tunnelling. It grows exponentially as the barrier thins.</p>" },
          { q: "Why are electrons waves?", a: "<p>That's how matter behaves at small scales: every particle has a wavelength h/p (de Broglie, 1924), and its position and momentum can't both be pinned down (Heisenberg, 1927). Confine an electron to a nanometre and its energy and spread are set by these rules.</p>" },
        ],
        axiom: { name: "Quantum mechanics: matter comes in atoms, and particles are waves", html: "<p>Discreteness sets a floor on size and brings randomness; waviness lets electrons leak through thin walls. Together with Boltzmann's statistics (the 60 mV floor) and the second law (Landauer, heat removal), these are the walls of computing. Everything above them, from the transistor to PyTorch, is our attempt to get as close to them as possible.</p>" },
      }),
    ),
    section("In code",
      code({
        "Tunnelling": `import numpy as np
hbar, me, eV = 1.0546e-34, 9.109e-31, 1.602e-19
def leak(d_nm, barrier_eV=3.1, m_eff=0.5):          # WKB estimate, rectangular barrier
    kappa = np.sqrt(2 * m_eff * me * barrier_eV * eV) / hbar
    return np.exp(-2 * kappa * d_nm * 1e-9)

for d in [2.0, 1.5, 1.2, 1.0, 0.8]:
    print(f"{d} nm of SiO2: {leak(d):.1e}")
# every ~0.18 nm thinner: 10x more leakage`,
        "Landauer and reliability": `import numpy as np
k = 1.380649e-23
for T in [4, 77, 300]:
    landauer = k * T * np.log(2)                     # erase one bit
    reliable = k * T * np.log(1e20)                  # keep a bit 1 - 1e-20 safe from heat
    print(f"{T:>3} K: Landauer {landauer:.1e} J, reliable bit {reliable:.1e} J")
print("a CMOS gate today ~1e-16 J:", 1e-16 / (k * 300 * np.log(2)), "x Landauer")`,
        "Distance per clock": `for ghz in [1, 3, 5]:
    tick_ps = 1000 / ghz
    print(f"{ghz} GHz: light {0.3 * tick_ps:.0f} mm, on-chip wire ~{tick_ps / 50:.1f} mm")`,
      }),
    ),
  );
  setW(); drawT(); drawL(); drawS();
}
