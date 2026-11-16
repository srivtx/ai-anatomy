import { h } from "../lib/core.js";
import { slider, stats, toggle, predict, stepper, code, callout, section, prose, table } from "../lib/widgets.js";
import { flowCanvas } from "../lib/flow.js";
import { goal, whyChain, stackMap, sticker, lawCards } from "../lib/why.js";
import { walk } from "../lib/walks.js";

function canvas(width, height, label) {
  const fc = flowCanvas({ width, height, label });
  fc.paint = (fn) => { fc.redraw = fn; fn(); };
  return fc;
}
const QE = 1.602e-19;
const L10 = Math.log10;
const p2 = (x) => { const v = Number(x.toPrecision(2)); return v >= 100 ? String(Math.round(v)) : String(v); };
const fmtR = (r) => (r >= 1e6 ? `${p2(r / 1e6)} MΩ` : r >= 1e3 ? `${p2(r / 1e3)} kΩ` : `${p2(r)} Ω`);
const fmtT = (s) => (s >= 1e-6 ? `${p2(s * 1e6)} µs` : s >= 1e-9 ? `${p2(s * 1e9)} ns` : s >= 1e-12 ? `${p2(s * 1e12)} ps` : `${p2(s * 1e15)} fs`);
const fmtI = (a) => (a >= 1 ? `${a.toPrecision(3)} A` : a >= 1e-3 ? `${p2(a * 1e3)} mA` : `${p2(a * 1e6)} µA`);
const fmtE = (n) => (n >= 1e6 ? `${p2(n / 1e6)} million` : Math.round(n).toLocaleString("en-US"));

// copper wire: bulk 1.7 µΩ·cm, rising in thin wires as electrons bounce off the walls and grain boundaries
// (a rough fit, mean free path 39 nm: ~5 µΩ·cm at 20 nm wide, in line with published measurements)
const rho = (wNm) => 1.7e-8 * (1 + 39 / wNm);   // Ω·m
// a mid-level wire for the delay model: 50 nm wide, 100 nm tall, ~0.2 fF per µm
const R_PER_M = rho(50) / (50e-9 * 100e-9), C_PER_M = 0.2e-15 / 1e-6, R0C0 = 2e-12;
const bare = (L) => 0.38 * R_PER_M * C_PER_M * L * L;              // distributed RC (Elmore, 50% point)
const PER_M_REP = 2.5 * Math.sqrt(R0C0 * R_PER_M * C_PER_M);       // with ideal repeaters (Bakoglu): linear in length
const SEG = Math.sqrt((2 * R0C0) / (0.38 * R_PER_M * C_PER_M));    // best spacing between repeaters
const rep = (L) => Math.min(bare(L), PER_M_REP * L + R0C0);
const LIGHT = 1.5e8;                                               // m/s in the glass between wires (n ≈ 2)

export default function (root, { id }) {
  // ---------- 1. Ohm, the water picture ----------
  const oh = { V: 0.75, R: 10e3 };
  const ost = stats([{ label: "Current (V ÷ R)" }, { label: "Electrons every nanosecond" }, { label: "Power turned to heat (V × I)" }]);
  const ocv = canvas(800, 260, "Voltage as the height of water, current as its flow, resistance as a narrow pipe; and the current against voltage");
  function drawO() {
    const I = oh.V / oh.R;
    ost.set(0, fmtI(I)); ost.set(1, fmtE((I / QE) * 1e-9)); ost.set(2, `${p2(oh.V * I * 1e6)} µW`);
    ocv.paint(() => {
      ocv.clear(); const c = ocv.colors, cx = ocv.ctx;
      // tank on the left: water height = voltage
      const tx = 30, tw = 90, base = 220, hMax = 170, wh = (oh.V / 1.2) * hMax;
      ocv.box(tx, base - hMax - 6, tw, hMax + 6, { fill: c.surface, stroke: c["ink-3"], r: 6 });
      ocv.box(tx + 3, base - wh, tw - 6, wh - 3, { fill: `color-mix(in srgb, ${c.c4} 55%, ${c.surface})`, stroke: "transparent", r: 4 });
      ocv.text(`${oh.V.toFixed(2)} V`, tx + tw / 2, base - wh - 12, { size: 13, align: "center" });
      ocv.text("height = voltage", tx + tw / 2, base + 22, { size: 11, align: "center", color: c["ink-3"] });
      // the pipe: narrower for a bigger resistance
      const pw = Math.max(3, 34 - 9 * (L10(oh.R) - 3)), py = base - 18;
      ocv.box(tx + tw - 3, py - pw / 2, 230, pw, { fill: `color-mix(in srgb, ${c.c4} 40%, ${c.surface})`, stroke: c["ink-3"], r: Math.min(6, pw / 2) });
      ocv.text(`narrow pipe = resistance (${fmtR(oh.R)})`, tx + tw + 112, py - pw / 2 - 12, { size: 11, align: "center", color: c["ink-3"] });
      // electrons in the pipe: spacing shows how much flows
      const n = Math.max(1, Math.min(26, Math.round(4 + 6 * L10(I / 1e-6))));
      for (let k = 0; k < n; k++) ocv.dot(tx + tw + 10 + (k + 0.5) * (210 / n), py, Math.max(1.6, Math.min(4, pw / 3.2)), c.warn);
      cx.strokeStyle = c.accent; cx.lineWidth = 2.5; cx.beginPath(); cx.moveTo(tx + tw + 190, py + pw / 2 + 14); cx.lineTo(tx + tw + 220, py + pw / 2 + 14); cx.stroke();
      cx.beginPath(); cx.moveTo(tx + tw + 220, py + pw / 2 + 14); cx.lineTo(tx + tw + 213, py + pw / 2 + 9); cx.lineTo(tx + tw + 213, py + pw / 2 + 19); cx.closePath(); cx.fillStyle = c.accent; cx.fill();
      ocv.text("flow = current", tx + tw + 160, py + pw / 2 + 32, { size: 11, align: "center", color: c["ink-3"] });
      // I-V plot on the right, log current
      const X0 = 470, X1 = 780, Y0 = 220, Y1 = 30, X = (v) => X0 + (v / 1.2) * (X1 - X0), Y = (i) => Y0 - ((L10(i) + 7) / 4.3) * (Y0 - Y1);
      ocv.line(X0, Y0, X1, Y0, { color: c["ink-3"] }); ocv.line(X0, Y0, X0, Y1, { color: c["ink-3"] });
      for (const [i, t] of [[1e-6, "1 µA"], [1e-5, "10 µA"], [1e-4, "100 µA"], [1e-3, "1 mA"]]) { ocv.line(X0, Y(i), X1, Y(i), { color: c.line, lw: 1 }); ocv.text(t, X0 - 6, Y(i) + 4, { size: 10.5, align: "right", color: c["ink-3"], mono: true }); }
      for (const v of [0, 0.4, 0.8, 1.2]) ocv.text(`${v} V`, X(v), Y0 + 16, { size: 10.5, align: "center", color: c["ink-3"], mono: true });
      for (const R of [1e3, 1e4, 1e5]) {
        const own = Math.abs(L10(R / oh.R)) < 0.02;
        cx.strokeStyle = own ? c.accent : c["ink-3"]; cx.lineWidth = own ? 2.5 : 1.2; cx.setLineDash(own ? [] : [4, 4]); cx.beginPath();
        for (let k = 1; k <= 60; k++) { const v = (k / 60) * 1.2, i = v / R; if (i < 1e-7) continue; k === 1 || i / R < 0 ? cx.moveTo(X(v), Y(i)) : cx.lineTo(X(v), Y(i)); }
        cx.stroke(); cx.setLineDash([]);
        const yE = Math.max(Y1 + 4, Y(1.2 / R));
        if (!own) ocv.text(fmtR(R), X1 + 2, yE + 4, { size: 10, color: c["ink-3"] });
      }
      cx.strokeStyle = c.accent; cx.lineWidth = 2.5; cx.beginPath();
      for (let k = 1; k <= 60; k++) { const v = (k / 60) * 1.2, i = v / oh.R; if (i < 1e-7) continue; k === 1 ? cx.moveTo(X(v), Y(i)) : cx.lineTo(X(v), Y(i)); }
      cx.stroke();
      ocv.dot(X(oh.V), Y(I), 6, c.accent);
      ocv.text("current against voltage (log scale): 10× the resistance, 10× less current", (X0 + X1) / 2, Y1 - 12, { size: 11.5, align: "center", color: c["ink-2"] });
    });
  }

  // ---------- 2. a wire is a resistor ----------
  const wr = { w: 20, L: 1e-3 };
  const wst = stats([{ label: "Resistivity at this width" }, { label: "Resistance of the wire" }, { label: "Compared with a switched-on transistor (~10 kΩ)" }]);
  const wcv = canvas(800, 240, "A copper wire's cross-section beside an electron's free path, and its resistance on a log scale");
  function drawW() {
    const A = wr.w * 1e-9 * 2 * wr.w * 1e-9, R = (rho(wr.w) * wr.L) / A;
    wst.set(0, `${(rho(wr.w) * 1e8).toFixed(1)} µΩ·cm`, wr.w > 100 ? "good" : wr.w > 30 ? "warn" : "bad");
    wst.set(1, fmtR(R)); wst.set(2, `${p2(R / 1e4)}×`, R < 1e3 ? "good" : R < 1e4 ? "warn" : "bad");
    wcv.paint(() => {
      wcv.clear(); const c = wcv.colors, cx = wcv.ctx;
      // cross-section, magnified so it is always ~100 px wide; the electron's free path at the same scale
      const S = 100 / wr.w, x = 30, y = 30, bw = wr.w * S, bh = 2 * wr.w * S;
      wcv.text("cross-section, magnified", x + 70, 18, { size: 11, align: "center", color: c["ink-3"] });
      wcv.box(x, y, bw, Math.min(190, bh), { fill: `color-mix(in srgb, #c27a3a 65%, ${c.surface})`, stroke: "#c27a3a", r: 3 });
      if (bh > 190) wcv.text("…", x + bw / 2, y + 200, { size: 14, align: "center", color: c["ink-3"] });
      const mfp = 39 * S;
      cx.save(); cx.beginPath(); cx.rect(0, 0, 330, 240); cx.clip();
      cx.setLineDash([4, 4]); cx.strokeStyle = c.accent; cx.lineWidth = 2; cx.beginPath(); cx.arc(x + bw / 2, y + Math.min(190, bh) / 2, mfp / 2, 0, Math.PI * 2); cx.stroke(); cx.setLineDash([]);
      cx.restore();
      const tx = 200;
      [[`${wr.w} nm wide, ${2 * wr.w} nm tall`, 12, c.ink], ["dashed circle: how far", 11, c["ink-3"]], ["an electron goes", 11, c["ink-3"]], ["between bumps (39 nm)", 11, c["ink-3"]],
        [wr.w < 60 ? "narrower than that:" : "wide: the walls", 11.5, wr.w < 60 ? c.bad : c.good], [wr.w < 60 ? "electrons hit the walls" : "rarely matter", 11.5, wr.w < 60 ? c.bad : c.good]]
        .forEach(([t, size, color], k) => wcv.text(t, tx, y + 14 + k * 16 + (k >= 4 ? 10 : 0) + (k >= 1 ? 8 : 0), { size, color }));
      // log bar of resistance
      const X0 = 360, X1 = 770, X = (r) => X0 + ((L10(r) - 0) / 8) * (X1 - X0), by = 150;
      wcv.box(X0, by - 10, X1 - X0, 20, { fill: c.sunken, stroke: "transparent", r: 10 });
      wcv.box(X0, by - 10, Math.max(20, X(Math.max(1, R)) - X0), 20, { fill: R < 1e3 ? c.good : R < 1e4 ? c.warn : c.bad, stroke: "transparent", r: 10 });
      for (const [r, t] of [[1, "1 Ω"], [1e3, "1 kΩ"], [1e6, "1 MΩ"], [1e8, "100 MΩ"]]) { wcv.line(X(r), by + 12, X(r), by + 18, { color: c["ink-3"] }); wcv.text(t, X(r), by + 32, { size: 10.5, align: "center", color: c["ink-3"], mono: true }); }
      wcv.line(X(1e4), by - 26, X(1e4), by + 12, { color: c.ink, lw: 1.5, dash: [3, 3] });
      wcv.text("a transistor that's on, ~10 kΩ", X(1e4), by - 32, { size: 11, align: "center", color: c["ink-2"] });
      wcv.text(`this wire: ${fmtR(R)} over ${wr.L >= 1e-3 ? `${p2(wr.L * 1e3)} mm` : `${p2(wr.L * 1e6)} µm`}`, Math.min(X1 - 90, Math.max(X0 + 90, X(Math.max(1, R)))), by + 56, { size: 13, align: "center" });
      wcv.text("R = ρ × length ÷ area", (X0 + X1) / 2, 40, { size: 15, align: "center" });
      wcv.text("halve the width (and height): 4× the resistance, and ρ itself goes up", (X0 + X1) / 2, 62, { size: 11.5, align: "center", color: c["ink-2"] });
    });
  }

  // ---------- 3. charge a capacitor through a resistor ----------
  const rc = { R: 10e3, C: 0.2e-15 }, VDD = 0.75;
  const rst = stats([{ label: "Time constant τ = R × C" }, { label: "Energy from the supply (C·V²)" }, { label: "Stored on the wire (½C·V²)" }, { label: "Burnt in the resistor" }]);
  const rcv = canvas(800, 260, "The voltage on a wire charging through a resistor, and where the energy goes");
  function drawR() {
    const tau = rc.R * rc.C, E = rc.C * VDD * VDD;
    rst.set(0, fmtT(tau)); rst.set(1, `${p2(E * 1e15)} fJ`); rst.set(2, `${p2(E * 0.5e15)} fJ`); rst.set(3, `${p2(E * 0.5e15)} fJ`, "warn");
    rcv.paint(() => {
      rcv.clear(); const c = rcv.colors, cx = rcv.ctx;
      const X0 = 60, X1 = 520, Y0 = 210, Y1 = 30, T = 5 * tau, X = (t) => X0 + (t / T) * (X1 - X0), Y = (v) => Y0 - (v / VDD) * (Y0 - Y1);
      rcv.line(X0, Y0, X1, Y0, { color: c["ink-3"] }); rcv.line(X0, Y0, X0, Y1 - 6, { color: c["ink-3"] });
      rcv.line(X0, Y(VDD), X1, Y(VDD), { color: c.line, dash: [4, 4] }); rcv.text(`${VDD} V`, X0 - 6, Y(VDD) + 4, { size: 10.5, align: "right", color: c["ink-3"], mono: true });
      rcv.text("0", X0 - 6, Y0 + 4, { size: 10.5, align: "right", color: c["ink-3"], mono: true });
      for (let k = 0; k <= 5; k++) { rcv.line(X(k * tau), Y0, X(k * tau), Y0 + 5, { color: c["ink-3"] }); rcv.text(k ? `${k}τ` : "0", X(k * tau), Y0 + 18, { size: 10.5, align: "center", color: c["ink-3"], mono: true }); }
      rcv.text(`time (τ = ${fmtT(tau)})`, (X0 + X1) / 2, Y0 + 38, { size: 11, align: "center", color: c["ink-3"] });
      cx.strokeStyle = c.accent; cx.lineWidth = 3; cx.beginPath();
      for (let k = 0; k <= 120; k++) { const t = (k / 120) * T, v = VDD * (1 - Math.exp(-t / tau)); k ? cx.lineTo(X(t), Y(v)) : cx.moveTo(X(t), Y(v)); }
      cx.stroke();
      rcv.dot(X(tau), Y(VDD * 0.632), 5, c.accent); rcv.text("63% after one τ", X(tau) + 10, Y(VDD * 0.632) + 16, { size: 11.5 });
      rcv.dot(X(2.2 * tau), Y(VDD * 0.889), 4, c["ink-2"]); rcv.text("most logic reads it as a 1 by ~2τ", X(2.2 * tau) + 10, Y(VDD * 0.889) + 18, { size: 11, color: c["ink-2"] });
      const tick = 0.5e-9;
      if (tick < T) { rcv.line(X(tick), Y1, X(tick), Y0, { color: c.bad, dash: [5, 4] }); rcv.text("one tick at 2 GHz", X(tick) + 4, Y1 + 4, { size: 11, color: c.bad }); }
      else rcv.text(`one tick at 2 GHz (500 ps) is ${p2(tick / T)}× this whole plot`, X1, Y1 - 8, { size: 11, align: "right", color: c.good });
      // energy bars
      const bx = 580, bw = 190, Emax = rc.C * VDD * VDD;
      rcv.text("each time the wire charges", bx, 40, { size: 12 });
      [["supply gives C·V²", 1, c.accent], ["½ stored on the wire", 0.5, c.c4], ["½ burnt as heat in R", 0.5, c.warn]].forEach(([t, f, col], k) => {
        const y = 62 + k * 50;
        rcv.box(bx, y, bw, 18, { fill: c.sunken, stroke: "transparent", r: 9 });
        rcv.box(bx, y, Math.max(18, bw * f), 18, { fill: col, stroke: "transparent", r: 9 });
        rcv.text(`${t}: ${p2(Emax * f * 1e15)} fJ`, bx, y + 34, { size: 11, color: c["ink-2"] });
      });
      rcv.text("the heat doesn't depend on R at all", bx, 222, { size: 11.5, color: c["ink-3"] });
    });
  }

  // ---------- 4. long wires ----------
  const lw = { L: 1e-3, rep: false };
  const lst = stats([{ label: "Plain wire (grows with length²)" }, { label: "With repeaters (grows with length)" }, { label: "Light, for comparison" }, { label: "Clock ticks at 2 GHz" }]);
  const lcv = canvas(800, 290, "A wire with repeaters, and signal delay against wire length on log scales");
  function drawL() {
    const tb = bare(lw.L), tr = rep(lw.L), tl = lw.L / LIGHT, t = lw.rep ? tr : tb;
    lst.set(0, fmtT(tb), lw.rep ? null : tb < 5e-10 ? "good" : "bad"); lst.set(1, fmtT(tr), lw.rep ? (tr < 5e-10 ? "good" : "warn") : null); lst.set(2, fmtT(tl)); lst.set(3, p2(t / 5e-10), t < 5e-10 ? "good" : t < 2e-9 ? "warn" : "bad");
    lcv.paint(() => {
      lcv.clear(); const c = lcv.colors, cx = lcv.ctx;
      // the wire, with repeaters drawn as triangles
      const wx0 = 40, wx1 = 760, wy = 34, nrep = lw.rep ? Math.max(0, Math.floor(lw.L / SEG) - 0) : 0;
      lcv.line(wx0, wy, wx1, wy, { color: "#c27a3a", lw: 4 });
      lcv.text("driver", wx0, wy - 12, { size: 10.5, color: c["ink-3"] }); lcv.text("receiver", wx1, wy - 12, { size: 10.5, align: "right", color: c["ink-3"] });
      const shown = Math.min(nrep, 36);
      for (let k = 1; k <= shown; k++) {
        const x = wx0 + (k / (shown + 1)) * (wx1 - wx0);
        cx.fillStyle = c.accent; cx.beginPath(); cx.moveTo(x - 6, wy - 7); cx.lineTo(x + 6, wy); cx.lineTo(x - 6, wy + 7); cx.closePath(); cx.fill();
      }
      lcv.text(lw.rep ? `${nrep.toLocaleString("en-US")} repeater${nrep === 1 ? "" : "s"}, one every ~${Math.round(SEG * 1e6)} µm${nrep > shown ? " (some drawn)" : ""}` : "one plain wire: every bit of it adds resistance and capacitance", (wx0 + wx1) / 2, wy + 24, { size: 11, align: "center", color: c["ink-2"] });
      // log-log chart
      const X0 = 90, X1 = 760, Y0 = 262, Y1 = 78, lx = (L) => X0 + ((L10(L) + 5) / 3.3) * (X1 - X0), ly = (s) => Y0 - ((L10(s) + 13) / 5) * (Y0 - Y1);
      lcv.line(X0, Y0, X1, Y0, { color: c["ink-3"] }); lcv.line(X0, Y0, X0, Y1, { color: c["ink-3"] });
      for (const [L, t] of [[1e-5, "10 µm"], [1e-4, "100 µm"], [1e-3, "1 mm"], [1e-2, "10 mm"]]) lcv.text(t, lx(L), Y0 + 16, { size: 10.5, align: "center", color: c["ink-3"], mono: true });
      for (const [s, t] of [[1e-12, "1 ps"], [1e-11, "10 ps"], [1e-10, "100 ps"], [1e-9, "1 ns"], [1e-8, "10 ns"]]) { lcv.line(X0, ly(s), X1, ly(s), { color: c.line, lw: 1 }); lcv.text(t, X0 - 6, ly(s) + 4, { size: 10.5, align: "right", color: c["ink-3"], mono: true }); }
      lcv.line(X0, ly(5e-10), X1, ly(5e-10), { color: c.bad, dash: [6, 4] }); lcv.text("one tick at 2 GHz", X0 + 6, ly(5e-10) - 6, { size: 11, color: c.bad });
      const curve = (f, col, w, name, dash) => {
        cx.save(); cx.beginPath(); cx.rect(X0, Y1, X1 - X0, Y0 - Y1); cx.clip();
        cx.strokeStyle = col; cx.lineWidth = w; cx.setLineDash(dash || []); cx.beginPath();
        for (let k = 0; k <= 80; k++) { const L = 10 ** (-5 + (k / 80) * 3.3), s = f(L); k ? cx.lineTo(lx(L), ly(s)) : cx.moveTo(lx(L), ly(s)); }
        cx.stroke(); cx.restore();
        const Le = 10 ** -2.0, se = f(Le), ye = Math.max(Y1 + 10, Math.min(Y0 - 6, ly(se)));
        lcv.text(name, lx(Le) + 8, ye + 4, { size: 11, color: col });
      };
      curve(bare, lw.rep ? c["ink-3"] : c.accent, lw.rep ? 1.5 : 3, "plain wire");
      curve(rep, lw.rep ? c.accent : c["ink-3"], lw.rep ? 3 : 1.5, "repeaters", lw.rep ? null : [5, 4]);
      curve((L) => L / LIGHT, c.good, 1.5, "light in glass", [2, 3]);
      lcv.dot(lx(lw.L), ly(lw.rep ? tr : tb), 6, c.accent);
    });
  }

  // ---------- 5. power delivery ----------
  const pd = { P: 700, V: 0.8, R: 0.1e-3 };
  const pst = stats([{ label: "Current into the die (P ÷ V)" }, { label: "Lost in the last centimetre (I²R)" }, { label: "Voltage lost on the way" }, { label: "Same power at 12 V would need" }]);
  const pcv = canvas(800, 230, "Power delivered at 12 volts, converted next to the chip, then carried at under 1 volt into the die");
  function drawP() {
    const I = pd.P / pd.V, loss = I * I * pd.R, drop = I * pd.R, I12 = pd.P / 12;
    pst.set(0, `${Math.round(I)} A`, I < 300 ? "good" : I < 800 ? "warn" : "bad");
    pst.set(1, `${loss.toFixed(0)} W`, loss < 20 ? "good" : loss < 60 ? "warn" : "bad");
    pst.set(2, `${(drop * 1e3).toFixed(0)} mV (${((drop / pd.V) * 100).toFixed(0)}%)`, drop / pd.V < 0.05 ? "good" : drop / pd.V < 0.1 ? "warn" : "bad");
    pst.set(3, `${I12.toFixed(0)} A`);
    pcv.paint(() => {
      pcv.clear(); const c = pcv.colors;
      const stages = [["power supply", "12 V", 20], ["board", `${I12.toFixed(0)} A at 12 V`, 200], ["voltage regulators", `12 V → ${pd.V.toFixed(2)} V`, 390], ["package + bumps", `${Math.round(I)} A`, 560], ["the die", `${pd.P} W`, 700]];
      stages.forEach(([name, sub, x], k) => {
        const w = k === 4 ? 80 : 120;
        pcv.box(x, 70, w, 56, { fill: k === 4 ? `color-mix(in srgb, ${c.warn} 25%, ${c.surface})` : c.surface, stroke: c.line, r: 10 });
        pcv.text(name, x + w / 2, 94, { size: 12, align: "center" });
        pcv.text(sub, x + w / 2, 112, { size: 10.5, align: "center", color: c["ink-3"], mono: true });
      });
      // arrows whose thickness is the current
      const arr = (x0, x1, amps, col) => { const t = Math.max(2, Math.min(40, amps / 25)); pcv.box(x0, 98 - t / 2, x1 - x0, t, { fill: col, stroke: "transparent", r: Math.min(4, t / 2) }); };
      arr(140, 200, I12, c.c4); arr(320, 390, I12, c.c4); arr(510, 560, I, c.bad); arr(680, 700, I, c.bad);
      pcv.text("thin arrows: few amps at 12 V", 230, 160, { size: 11.5, color: c["ink-2"] });
      pcv.text(`fat arrows: ${Math.round(I)} A at ${pd.V.toFixed(2)} V, kept as short as possible`, 560, 160, { size: 11.5, align: "center", color: c["ink-2"] });
      pcv.text(`I²R in the last centimetre (${(pd.R * 1e3).toFixed(2)} mΩ): ${(I * I * pd.R).toFixed(0)} W of heat before the chip does anything`, 400, 200, { size: 13, align: "center", color: I * I * pd.R > 40 ? c.bad : c.ink });
      pcv.text("why so many amps? P = V × I, and the core voltage must stay low", 400, 36, { size: 13, align: "center" });
    });
  }

  // ---------- 6. one switch, step by step ----------
  let iv = 0;
  const icv = canvas(800, 250, "An inverter driving a wire: the p-type switch fills it from the supply, the n-type switch empties it to ground");
  function drawI() {
    icv.paint(() => {
      icv.clear(); const c = icv.colors, x = 150, top = 30, bot = 220, mid = 125;
      const pOn = iv === 1 || iv === 2, nOn = iv === 3, high = iv === 1 || iv === 2 || iv === 4 ? iv !== 4 : false;
      icv.line(x - 70, top, x + 70, top, { color: c.ink, lw: 3 }); icv.text("supply, 0.75 V", x + 78, top + 4, { size: 11.5 });
      icv.line(x - 70, bot, x + 70, bot, { color: c.ink, lw: 3 }); icv.text("ground, 0 V", x + 78, bot + 4, { size: 11.5 });
      const sw = (y, on, label, col) => {
        icv.box(x - 26, y - 18, 52, 36, { fill: on ? `color-mix(in srgb, ${col} 35%, ${c.surface})` : c.surface, stroke: on ? col : c.line, lw: 2, r: 8 });
        icv.text(label, x, y + 4, { size: 11, align: "center", color: on ? c.ink : c["ink-3"] });
        icv.text(on ? "on" : "off", x - 36, y + 4, { size: 11, align: "right", color: on ? col : c["ink-3"] });
      };
      icv.line(x, top, x, 60, { color: pOn && iv === 1 ? c.accent : c["ink-3"], lw: pOn && iv === 1 ? 4 : 2 });
      sw(78, pOn, "p-type", c.accent);
      icv.line(x, 96, x, 154, { color: iv === 1 || iv === 3 ? (iv === 1 ? c.accent : c.warn) : c["ink-3"], lw: iv === 1 || iv === 3 ? 4 : 2 });
      sw(172, nOn, "n-type", c.warn);
      icv.line(x, 190, x, bot, { color: iv === 3 ? c.warn : c["ink-3"], lw: iv === 3 ? 4 : 2 });
      // output node and its wire capacitance
      icv.line(x, mid, 360, mid, { color: high ? c.accent : c["ink-3"], lw: 3 });
      icv.line(360, mid, 360, 160, { color: c["ink-3"], lw: 2 }); icv.line(340, 160, 380, 160, { color: c.ink, lw: 3 }); icv.line(340, 170, 380, 170, { color: c.ink, lw: 3 }); icv.line(360, 170, 360, bot, { color: c["ink-3"], lw: 2 });
      icv.text("the wire and the gates it drives: C", 390, 168, { size: 11.5, color: c["ink-2"] });
      icv.text(`output: ${high ? "1 (charged)" : "0 (empty)"}`, 250, mid - 10, { size: 12, align: "center", color: high ? c.accent : c["ink-3"] });
      const nE = high ? 9 : 0;
      for (let k = 0; k < nE; k++) icv.dot(346 + (k % 3) * 14, 150 - Math.floor(k / 3) * 8, 3, c.warn);
      // energy ledger
      const L = [["from the supply", iv >= 1 ? "C·V²" : "0"], ["stored on the wire", high ? "½C·V²" : "0"], ["heat in the p-type", iv >= 1 ? "½C·V²" : "0"], ["heat in the n-type", iv >= 3 ? "½C·V²" : "0"]];
      L.forEach(([t, v], k) => { icv.text(t, 560, 60 + k * 30, { size: 12, color: c["ink-2"] }); icv.text(v, 770, 60 + k * 30, { size: 12.5, align: "right", mono: true }); });
      icv.line(560, 178, 770, 178, { color: c.line });
      icv.text(iv >= 3 ? "one full 0 → 1 → 0: C·V² of heat" : "so far", 560, 200, { size: 12.5, color: iv >= 3 ? c.bad : c["ink-3"] });
    });
  }

  // ---------- page ----------
  root.append(
    section(null,
      goal("<p>Before the switch, the stuff it switches. Every bit in a chip is a tiny bucket of charge on a wire, filled or emptied through a resistance. Four quantities (charge, voltage, current, resistance) and one more (capacitance) explain why chips burn power, why long wires are slow, and why a GPU needs hundreds of amps. The goal: <b>know what each one means, with real chip numbers</b>, so the next chapters' physics has something to stand on.</p>",
        ["Move charge to represent bits", "Do it billions of times a second", "Waste as little energy as possible"]),
      stackMap(id),
      h("div.vx-row",
        sticker("resistor", { title: "A resistor", caption: "the colour bands say 10 kΩ; on a chip, every wire and switch is one", px: 4 }),
        sticker("capacitor", { title: "A capacitor", caption: "two plates, a gap between: electrons (yellow) pile up on one side", px: 4 }),
        sticker("metalstack", { title: "A chip's wires", caption: "transistors at the bottom, a dozen or more layers of copper above", px: 4 })),
    ),
    section("Voltage, current, resistance: the water picture",
      prose("<p><b>Charge</b> is the stuff: each electron carries 1.6 × 10⁻¹⁹ coulombs. <b>Voltage</b> is the push on it, the energy each coulomb gains or loses (1 volt = 1 joule per coulomb). <b>Current</b> is how much charge passes per second (1 amp ≈ 6 × 10¹⁸ electrons a second). <b>Resistance</b> is how hard the material makes it, because electrons keep bumping into vibrating atoms and impurities.</p><p>Think of water: voltage is the height of the tank, current is the flow, resistance is how narrow the pipe is. <b>Ohm's law</b> ties them together: <b>V = I × R</b>. A switched-on transistor is roughly a 10 kΩ resistor; at 0.75 V that's 75 µA.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Voltage (tank height)", min: 0.1, max: 1.2, step: 0.05, value: oh.V, fmt: (v) => `${v.toFixed(2)} V`, onInput: (v) => { oh.V = v; drawO(); } }),
          slider({ label: "Resistance (narrow pipe)", min: 3, max: 5, step: 0.05, value: 4, fmt: (v) => fmtR(10 ** v), onInput: (v) => { oh.R = 10 ** v; drawO(); } })),
        h("div.stage-body", h("div.wide-canvas", ocv.el), ost.el)),
      callout("note", "The electrons themselves are slow", "<p>In a copper wire carrying a typical current, each electron drifts at well under a millimetre per second. The <i>signal</i> is fast because the wire is already full of electrons: push one end and the push travels down the wire as an electromagnetic wave, near the speed of light. Like a full hose: water comes out the far end the moment you open the tap.</p>"),
    ),
    section("A wire is a resistor",
      prose("<p>Resistance grows with length and shrinks with cross-section: <b>R = ρ L / A</b>, where ρ (resistivity) is the material's own stubbornness. Copper's is 1.7 µΩ·cm in bulk. But chip wires are now narrower than the distance an electron travels between collisions (~39 nm in copper), so electrons also bounce off the walls and grain boundaries, and ρ itself rises. Shrink the wire and watch.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Wire width", min: 1, max: 3, step: 0.05, value: L10(wr.w), fmt: (v) => `${Math.round(10 ** v)} nm`, onInput: (v) => { wr.w = Math.round(10 ** v); drawW(); } }),
          slider({ label: "Wire length", min: -6, max: -2, step: 0.05, value: -3, fmt: (v) => (10 ** v >= 1e-3 ? `${p2(10 ** v * 1e3)} mm` : `${p2(10 ** v * 1e6)} µm`), onInput: (v) => { wr.L = 10 ** v; drawW(); } })),
        h("div.stage-body", h("div.wide-canvas", wcv.el), wst.el)),
      callout("tip", "Why this matters now", "<p>Transistors got faster with every shrink; wires got <i>worse</i>. A 20 nm copper wire 1 mm long has about 60 kΩ of resistance, six switched-on transistors' worth. That's why the newest chips try ruthenium and molybdenum for the thinnest wires (they suffer less from the wall effect) and why <a href='/learn?c=edges'>the edges chapter</a> lists wires as one of the nearest walls.</p>"),
    ),
    section("Every bit is a bucket: charging through a resistor",
      prose("<p>A wire and the transistor gates it feeds have <b>capacitance</b>: charge it holds per volt, <b>Q = C × V</b>. A typical logic node is ~0.2 fF (femtofarads); at 0.75 V that's ~940 electrons. To make a 1, the driving transistor pours them in through its own resistance. The voltage doesn't jump; it creeps up, fast at first, then slower: <b>V(t) = V·(1 − e<sup>−t/RC</sup>)</b>. The product <b>R × C</b>, the time constant, sets the speed of everything on a chip.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Resistance of the driver", min: 3, max: 5, step: 0.05, value: 4, fmt: (v) => fmtR(10 ** v), onInput: (v) => { rc.R = 10 ** v; drawR(); } }),
          slider({ label: "Capacitance of the wire", min: -16, max: -14, step: 0.05, value: L10(0.2e-15), fmt: (v) => `${p2(10 ** v * 1e15)} fF`, onInput: (v) => { rc.C = 10 ** v; drawR(); } })),
        h("div.stage-body", h("div.wide-canvas", rcv.el), rst.el)),
      predict({
        chapter: id,
        question: "You make the driving transistor 10× stronger (10× less resistance), so the wire charges 10× faster. How much energy is burnt as heat per charge now?",
        options: ["10× less", "10× more", "The same: ½·C·V²", "Zero: a perfect switch wastes nothing"],
        answer: 2,
        explain: "<p>The heat is ½·C·V² whatever the resistance. A smaller R means a bigger current for a shorter time; I²R × time comes out the same. The only ways to spend less energy per switch are a smaller C (smaller transistors, shorter wires) or a lower V, and V counts twice. That's the whole story of <a href='/learn?c=power-wall'>the power wall</a>.</p>",
      }),
    ),
    section("One switch, step by step: where the energy goes",
      prose("<p>Every logic gate drives its output with two switches: a p-type to the supply and an n-type to ground (that pair is an <b>inverter</b>, the simplest gate). Step through one 0 → 1 → 0 cycle and keep the books.</p>"),
      h("div.stage", h("div.stage-body", h("div.wide-canvas", icv.el))),
      stepper([
        { title: "At rest: the output is 0", html: "<p>The n-type switch is on, so the output wire is connected to ground and empty. Nothing flows: in a steady state, a CMOS gate draws almost no current. That's why CMOS won.</p>" },
        { title: "The input falls: fill the wire", html: "<p>The p-type switch turns on, the n-type off. Charge Q = C·V flows from the supply through the p-type into the wire. The supply hands over Q·V = <b>C·V²</b> of energy. Half ends up stored on the wire; the other half is turned to heat in the p-type's resistance, as the last section showed.</p>" },
        { title: "Holding a 1", html: "<p>The wire sits at 0.75 V holding ½·C·V². Again nothing flows, apart from a little leakage through the switch that's off.</p>" },
        { title: "The input rises: empty the wire", html: "<p>Now the n-type turns on and the stored charge drains to ground. The ½·C·V² that was stored is turned into heat in the n-type. The supply gives nothing this time.</p>" },
        { title: "The bill: C·V² per cycle", html: "<p>One full 0 → 1 → 0 costs <b>C·V²</b>, all of it heat. Multiply by how often each wire flips (α) and the clock (f), and sum over billions of wires: <b>P = α·C·V²·f</b>. With 0.2 fF and 0.75 V that's ~0.11 fJ per flip; an H100 doing ~10¹⁸ flips a second would burn ~100 W on that alone.</p>" },
      ], { onStep: (k) => { iv = k; icv.redraw?.(); } }),
      prose("<p>Step through the numbers, one time constant at a time:</p>"),
      walk("circuits"),
    ),
    section("Why long wires are slow",
      prose("<p>A long wire is a resistor <i>and</i> a capacitor, spread along its length. Double the length and you double both R and C, so the RC delay grows with <b>length squared</b>. The fix is to cut the wire into short pieces with an amplifier (a <b>repeater</b>) between each, which makes delay grow only linearly. Drag the length and switch repeaters on.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Wire length", min: -5, max: -1.7, step: 0.05, value: -3, fmt: (v) => (10 ** v >= 1e-3 ? `${p2(10 ** v * 1e3)} mm` : `${Math.round(10 ** v * 1e6)} µm`), onInput: (v) => { lw.L = 10 ** v; drawL(); } }),
          toggle({ label: "Add repeaters", checked: lw.rep, onChange: (v) => { lw.rep = v; drawL(); } })),
        h("div.stage-body", h("div.wide-canvas", lcv.el), lst.el)),
      callout("note", "What this does to chip design", "<p>Even with repeaters, a signal crosses only ~4 mm of a 20+ mm die in one 2 GHz tick. That's why big chips are built as many small, nearly independent tiles (cores, or a GPU's SMs) that talk over networks taking several cycles, and why <a href='/learn?c=memory-physics'>memory is far</a>: distance is time.</p>"),
    ),
    section("Why a GPU needs 800 amps",
      prose("<p>Power is voltage times current: <b>P = V × I</b>. An H100 uses up to 700 W, and its core runs below 1 V, so the current into the die is <b>hundreds of amps</b>, more than a car's starter motor. And every bit of resistance on the way burns <b>I² × R</b>: double the current and the loss quadruples. So power travels at 12 V (or 48 V in new racks) as far as possible, and is converted down right beside the chip.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Chip power", min: 100, max: 1200, step: 50, value: pd.P, fmt: (v) => `${v} W`, onInput: (v) => { pd.P = v; drawP(); } }),
          slider({ label: "Core voltage", min: 0.6, max: 1.2, step: 0.05, value: pd.V, fmt: (v) => `${v.toFixed(2)} V`, onInput: (v) => { pd.V = v; drawP(); } }),
          slider({ label: "Resistance of the last centimetre", min: 0.02, max: 0.5, step: 0.01, value: 0.1, fmt: (v) => `${v.toFixed(2)} mΩ`, onInput: (v) => { pd.R = v * 1e-3; drawP(); } })),
        h("div.stage-body", h("div.wide-canvas", pcv.el), pst.el)),
      whyChain({
        claim: "A GPU's core runs on less than 1 volt, so it has to be fed hundreds of amps.",
        steps: [
          { q: "Why not run the core at 12 V and need 15× fewer amps?", a: "<p>Because each switch costs C·V². At 12 V instead of 0.75 V, every flip would cost (12 / 0.75)² = 256× the energy. The chip would need 180 kW. And the gate insulator, a few atoms thick, would break down: 12 V across ~1 nm is a field no material survives.</p>" },
          { q: "Then why not go even lower than 0.75 V?", a: "<p>A transistor needs a few hundred millivolts above its threshold to switch on strongly, and its threshold can't drop much without leaking (every 60 mV lower means ~10× more leakage, from <a href='/learn?c=transistor'>the transistor chapter</a>). So the core voltage is squeezed into ~0.6–0.9 V.</p>" },
          { q: "Why is a big current a problem?", a: "<p>Resistance turns current into heat at I²R. At 875 A, even 0.1 mΩ, about a centimetre of a thick copper plane, burns 77 W and loses 88 mV, 11% of the supply. The transistors at the end of the path would run slow.</p>" },
          { q: "So what do designers do?", a: "<p>Carry power at high voltage as close as possible, then convert it right under or beside the package. Spread the current over thousands of solder bumps and dozens of thick copper planes. And newest of all: <b>backside power delivery</b> (Intel 18A, TSMC A16), which feeds power from under the transistors so it doesn't have to squeeze through the signal wiring.</p>" },
        ],
        axiom: { name: "Joule heating: moving charge through resistance turns energy into heat", html: "<p>Each coulomb that falls through a voltage drop ΔV gives up ΔV joules. In a resistor that energy goes into jiggling atoms: heat. P = I·ΔV = I²R. There is no resistance-free conductor at room temperature (superconductors need cryogenic cooling), so every amp delivered costs I²R on the way.</p>" },
      }),
      predict({
        chapter: id,
        question: "A chip's power stays the same, but its core voltage drops from 1.0 V to 0.7 V. What happens to the I²R loss in the power wiring?",
        options: ["It falls, because the voltage is lower", "It stays the same", "It roughly doubles", "It rises 10×"],
        answer: 2,
        explain: "<p>Same power at lower voltage means more current: I rises by 1.0 / 0.7 ≈ 1.43×, so I² rises by ~2×. Lower voltage saves energy in every switch (C·V²) but makes delivering the power harder. Both effects are real, which is why power delivery got so much harder as voltages fell.</p>",
      }),
    ),
    section("The laws a chip obeys",
      prose("<p>Everything above, and much of the next ten chapters, comes from a handful of laws. Here they are on one table, each with where it bites.</p>"),
      lawCards([
        { name: "Ohm's law", formula: "V = I × R", glyph: "ohm", html: "<p>Push (volts) divided by resistance gives flow (amps). A switched-on transistor is a resistor; so is every wire.</p>", where: [["circuits", "this chapter"], ["transistor", "the switch"]] },
        { name: "Joule heating", formula: "P = I² × R", glyph: "heat", html: "<p>Current through resistance makes heat, growing with the square of the current. Why power is carried at high voltage.</p>", where: [["power-wall", "power wall"]] },
        { name: "Capacitance", formula: "Q = C × V", glyph: "charge", html: "<p>A bucket of charge: about 940 electrons make one logic node read as a 1. DRAM stores bits this way.</p>", where: [["memory-physics", "memory"], ["all-the-way-down", "capstone"]] },
        { name: "Switching energy", formula: "E = C × V²", glyph: "clock", html: "<p>Each 0 → 1 → 0 costs C·V², half burnt filling, half emptying. Times activity and clock: P = α·C·V²·f.</p>", where: [["power-wall", "power wall"], ["gpu-design", "GPU design"]] },
        { name: "RC delay", formula: "τ = R × C", glyph: "rc", html: "<p>How fast a wire charges. Long plain wires slow with length², which is why chips are tiles and memory is far.</p>", where: [["memory-physics", "memory"], ["logic", "the clock"]] },
        { name: "Wire resistance", formula: "R = ρ L / A", glyph: "wire", html: "<p>Longer and thinner means worse, and below ~40 nm copper's ρ itself rises. Transistors shrink well; wires don't.</p>", where: [["edges", "edges"], ["fab", "making chips"]] },
        { name: "Kirchhoff's current law", formula: "Σ I in = Σ I out", glyph: "node", html: "<p>Charge isn't created or lost at a junction. Every amp into the die comes back out through ground: 800 A each way.</p>", where: [["circuits", "this chapter"]] },
        { name: "Boltzmann's tail", formula: "× 10 per 60 mV", glyph: "boltz", html: "<p>Electron energies spread with temperature, so a switch turns on at most 10× per 60 mV. Sets the voltage floor.</p>", where: [["transistor", "the switch"], ["electrons", "the material"]] },
      ]),
      table(["Quantity", "Unit", "On a chip, roughly"], [
        ["Voltage", "volt (V) = joule per coulomb", "0.6–0.9 V for the core"],
        ["Current", "amp (A) = coulomb per second", "~75 µA through one on transistor; hundreds of A into a GPU"],
        ["Resistance", "ohm (Ω) = volt per amp", "~10 kΩ for an on transistor; tens of kΩ for a thin 1 mm wire"],
        ["Capacitance", "farad (F) = coulomb per volt", "~0.2 fF for a logic node (10⁻¹⁵ F)"],
        ["Energy", "joule (J)", "~0.1 fJ per bit flip; 640 pJ per 32-bit DRAM read"],
        ["Power", "watt (W) = joule per second", "700 W for an H100"],
      ]),
      callout("note", "Next", "<p>All of this needed a switch that a voltage can open and close. Next: <a href='/learn?c=transistor'>the transistor</a>, a switch made of a field.</p>"),
    ),
    section("In code",
      code({
        "Ohm and Joule": `V, R = 0.75, 10e3            # volts, ohms: one switched-on transistor
I = V / R                     # 75 µA
print(I / 1.602e-19 * 1e-9, "electrons per ns")   # ~470,000
P, Vcore = 700, 0.8
I_core = P / Vcore            # 875 A into the die
print("loss in 0.1 mΩ:", I_core**2 * 0.1e-3, "W")  # ~77 W`,
        "Charge a wire": `import numpy as np
R, C, V = 10e3, 0.2e-15, 0.75
tau = R * C                                  # 2 ps
t = np.linspace(0, 5 * tau, 6)
print(V * (1 - np.exp(-t / tau)))            # 0, 0.47, 0.65, 0.71, 0.74, 0.75 V
print("energy per 0->1->0:", C * V**2, "J")  # 1.1e-16 J, independent of R`,
        "Wire delay": `rho = lambda w_nm: 1.7e-8 * (1 + 39 / w_nm)     # copper, ohm·m, thin-wire fit
r = rho(50) / (50e-9 * 100e-9)                   # ohms per metre
c = 0.2e-15 / 1e-6                               # farads per metre
for L in [1e-4, 1e-3, 1e-2]:                     # 0.1, 1, 10 mm
    plain = 0.38 * r * c * L**2                   # grows with L squared
    repeated = 2.5 * (2e-12 * r * c) ** 0.5 * L   # grows with L
    print(L * 1e3, "mm:", plain, repeated)`,
      }),
    ),
  );
  drawO(); drawW(); drawR(); drawL(); drawP(); drawI();
}
