import { h } from "../lib/core.js";
import { slider, segmented, stats, predict, stepper, code, callout, section, prose, table } from "../lib/widgets.js";
import { flowCanvas } from "../lib/flow.js";
import { goal, whyChain, stackMap, sticker, lawCards } from "../lib/why.js";

function canvas(width, height, label) {
  const fc = flowCanvas({ width, height, label });
  fc.paint = (fn) => { fc.redraw = fn; fn(); };
  return fc;
}
const L10 = Math.log10;

// light sources, with the numerical aperture of the lens (or mirrors) used with each
const SRC = [
  { k: "g", name: "g-line lamp", lam: 436, NA: 0.45, era: "1980s" },
  { k: "i", name: "i-line lamp", lam: 365, NA: 0.6, era: "1990s" },
  { k: "krf", name: "KrF laser", lam: 248, NA: 0.8, era: "late 1990s" },
  { k: "arf", name: "ArF laser", lam: 193, NA: 0.93, era: "early 2000s" },
  { k: "imm", name: "ArF immersion", lam: 193, NA: 1.35, era: "2007 on" },
  { k: "euv", name: "EUV", lam: 13.5, NA: 0.33, era: "2019 on" },
  { k: "hna", name: "High-NA EUV", lam: 13.5, NA: 0.55, era: "arriving now" },
];
// How well each spatial frequency gets through, with the best illumination tricks: everything passes up to
// pitch = λ/NA, then contrast falls to zero at pitch = λ/(2NA), the hard limit (k1 = 0.25). A teaching model.
const pass = (nu) => (nu <= 0.5 ? 1 : nu >= 1 ? 0 : (1 - nu) / 0.5);
function image(src, pitch, x) {   // aerial image of equal lines and spaces, built from the harmonics that get through
  const nu = src.lam / (2 * src.NA * pitch);
  let I = 0.5;
  for (let n = 1; n <= 9; n += 2) I += (2 / (n * Math.PI)) * (n % 4 === 1 ? 1 : -1) * pass(n * nu) * Math.cos((2 * Math.PI * n * x) / pitch);
  return I;
}

// defects: a fixed scatter over the 300 mm wafer, so moving a slider adds or removes the same ones
const DEF = (() => { let x = 7; const r = () => ((x = (x * 16807) % 2147483647) / 2147483647); const out = []; while (out.length < 500) { const a = r() * 2 - 1, b = r() * 2 - 1; if (a * a + b * b <= 1) out.push([a * 150, b * 150]); } return out; })();
const WAFER_COST = 17000, R_WAFER = 150, EDGE = 3;

export default function (root, { id }) {
  // ---------- 1. print lines with light ----------
  const li = { src: SRC[3], pitch: 200 };
  const lst = stats([{ label: "Half-pitch (one line's width)" }, { label: "k₁ = half-pitch × NA ÷ λ" }, { label: "Contrast of the image" }, { label: "Prints in one exposure?" }]);
  const lcv = canvas(800, 300, "A mask of lines and spaces, the blurred light image it makes, and the pattern printed in the resist");
  function drawLi() {
    const s = li.src, p = li.pitch, hp = p / 2, k1 = (hp * s.NA) / s.lam;
    const xs = Array.from({ length: 201 }, (_, i) => (i / 200) * p), I = xs.map((x) => image(s, p, x));
    const mx = Math.min(1, Math.max(...I)), mn = Math.max(0, Math.min(...I)), m = mx + mn > 0 ? (mx - mn) / (mx + mn) : 0;
    lst.set(0, `${hp.toFixed(hp < 20 ? 1 : 0)} nm`); lst.set(1, k1.toFixed(2), k1 >= 0.4 ? "good" : k1 >= 0.28 ? "warn" : "bad");
    lst.set(2, `${Math.round(m * 100)}%`, m >= 0.6 ? "good" : m >= 0.3 ? "warn" : "bad");
    lst.set(3, m >= 0.6 ? "yes, easily" : m >= 0.3 ? "only with care" : m > 0.02 ? "no: split it into two exposures" : "no: nothing resolves", m >= 0.6 ? "good" : m >= 0.3 ? "warn" : "bad");
    lcv.paint(() => {
      lcv.clear(); const c = lcv.colors, cx = lcv.ctx, x0 = 80, x1 = 770, per = 5, S = (x1 - x0) / (per * p);
      const label = (t, y) => lcv.text(t, x0 - 10, y, { size: 11, align: "right", color: c["ink-3"] });
      // the light's wavelength at the same scale
      const lamPx = s.lam * S;
      cx.save(); cx.beginPath(); cx.rect(x0, 8, x1 - x0, 30); cx.clip();
      cx.strokeStyle = c.accent; cx.lineWidth = 2; cx.beginPath();
      for (let X = x0; X <= x1; X += 1) { const y = 23 + 9 * Math.sin(((X - x0) / lamPx) * 2 * Math.PI); X === x0 ? cx.moveTo(X, y) : cx.lineTo(X, y); }
      cx.stroke(); cx.restore();
      label("light", 27);
      lcv.text(`λ = ${s.lam} nm, drawn to the same scale as the lines`, x1, 50, { size: 10.5, align: "right", color: c["ink-3"] });
      // the mask
      label("mask", 76);
      for (let k = 0; k < per; k++) lcv.box(x0 + (k * p + p / 4) * S, 62, (p / 2) * S, 24, { fill: c.ink, stroke: "transparent", r: 1 });
      // the image
      const iy0 = 190, iy1 = 104, Y = (v) => iy0 - Math.min(1.25, Math.max(-0.1, v)) * (iy0 - iy1) / 1.2;
      label("light on", iy1 + 12); label("the wafer", iy1 + 26);
      lcv.line(x0, Y(0.5), x1, Y(0.5), { color: c.line, dash: [4, 4] });
      lcv.text("resist threshold", x1, Y(0.5) - 5, { size: 10, align: "right", color: c["ink-3"] });
      cx.strokeStyle = c.accent; cx.lineWidth = 2.5; cx.beginPath();
      for (let X = x0; X <= x1; X += 2) { const xx = ((X - x0) / S) % p, v = image(s, p, xx - p / 2); X === x0 ? cx.moveTo(X, Y(v)) : cx.lineTo(X, Y(v)); }
      cx.stroke();
      // the printed resist: where light stayed below threshold, the line survives
      label("printed", 242);
      if (m < 0.3) {
        lcv.box(x0, 228, x1 - x0, 26, { fill: `color-mix(in srgb, ${c.ink} 22%, ${c.surface})`, stroke: "transparent", r: 4 });
        lcv.text("too little contrast: the resist can't tell the lines apart", (x0 + x1) / 2, 245, { size: 12, align: "center", color: c.bad });
      } else {
        let inLine = false, start = 0;
        for (let X = x0; X <= x1 + 1; X += 1) {
          const xx = ((Math.min(X, x1) - x0) / S) % p, dark = image(s, p, xx - p / 2) < 0.5 && X <= x1;
          if (dark && !inLine) { inLine = true; start = X; }
          if (!dark && inLine) { inLine = false; lcv.box(start, 228, X - start, 26, { fill: m >= 0.6 ? c.good : c.warn, stroke: "transparent", r: 2 }); }
        }
      }
      lcv.text(`${s.name} (${s.era}): λ ${s.lam} nm, NA ${s.NA} · pitch ${p} nm`, (x0 + x1) / 2, 285, { size: 12.5, align: "center" });
    });
  }

  // ---------- 2. follow a photon through an EUV machine ----------
  let ph = 0;
  const PATH = [
    ["tin droplet + laser", 70, 200, 0], ["collector mirror", 170, 120, 1], ["illuminator (4 mirrors)", 330, 60, 4], ["mask (a mirror too)", 500, 120, 1], ["projection (6 mirrors)", 620, 60, 6], ["wafer", 720, 200, 0]];
  const ecv = canvas(800, 260, "The path of EUV light: made in a tin plasma, bounced off about a dozen mirrors, onto the wafer");
  function drawE() {
    ecv.paint(() => {
      ecv.clear(); const c = ecv.colors;
      let left = 1;
      PATH.forEach(([name, x, y, nm], k) => {
        if (k > 0 && k <= ph) left *= 0.7 ** nm;
        if (k < PATH.length - 1) { const [, x2, y2] = PATH[k + 1]; ecv.line(x + 40, y + 16, x2 + 40, y2 + 16, { color: k < ph ? c.accent : c.line, lw: k < ph ? 4 * (0.25 + 0.75 * left) : 1.5, dash: k < ph ? null : [5, 5] }); }
      });
      PATH.forEach(([name, x, y], k) => {
        const on = k === ph;
        ecv.box(x, y, 80, 32, { fill: on ? `color-mix(in srgb, ${c.accent} 22%, ${c.surface})` : c.surface, stroke: on ? c.accent : c.line, lw: on ? 2 : 1.5, r: 8 });
        const words = name.split(" ("), first = words[0], second = words[1] ? `(${words[1]}` : null;
        ecv.text(first, x + 40, y + (second ? 14 : 20), { size: 10.5, align: "center", color: on ? c.ink : c["ink-2"] });
        if (second) ecv.text(second, x + 40, y + 26, { size: 9.5, align: "center", color: c["ink-3"] });
      });
      let f = 1; for (let k = 1; k <= ph; k++) f *= 0.7 ** PATH[k][3];
      ecv.text("light left", 40, 26, { size: 11, color: c["ink-3"] });
      ecv.box(110, 14, 220, 16, { fill: c.sunken, stroke: "transparent", r: 8 });
      ecv.box(110, 14, Math.max(8, 220 * f), 16, { fill: f > 0.3 ? c.accent : f > 0.05 ? c.warn : c.bad, stroke: "transparent", r: 8 });
      ecv.text(`${(f * 100).toPrecision(2)}%`, 340, 27, { size: 12.5, mono: true });
      ecv.text("all inside a vacuum: even air would absorb this light", 400, 252, { size: 11, align: "center", color: c["ink-3"] });
    });
  }

  // ---------- 3. yield ----------
  const yd = { A: 400, D: 0.1, n: 1 };
  const yst = stats([{ label: "Dies that fit on the wafer" }, { label: "Dies hit by a defect (this wafer)" }, { label: "Expected yield, e^(−area × density)" }, { label: `Cost per working chip (wafer $${WAFER_COST.toLocaleString("en-US")})` }]);
  const ycv = canvas(800, 330, "A 300 mm wafer ruled into dies, with random defects; dies hit by a defect are lost");
  function drawY() {
    const a = yd.A / yd.n, side = Math.sqrt(a), nDef = Math.round(yd.D * 706.86);
    const defs = DEF.slice(0, nDef), dies = [];
    const off = side / 2 * ((Math.ceil((2 * R_WAFER) / side) % 2) ? 0 : 1);
    const N = Math.ceil(R_WAFER / side) + 1;
    for (let i = -N; i <= N; i++) for (let j = -N; j <= N; j++) {
      const x = i * side - off, y = j * side - off;
      if ([[x, y], [x + side, y], [x, y + side], [x + side, y + side]].every(([px, py]) => Math.hypot(px, py) <= R_WAFER - EDGE)) dies.push({ x, y, bad: false });
    }
    for (const d of dies) d.bad = defs.some(([px, py]) => px >= d.x && px < d.x + side && py >= d.y && py < d.y + side);
    const good = dies.filter((d) => !d.bad).length, Y = Math.exp(-(a / 100) * yd.D);
    const systems = Math.floor(good / yd.n), cost = systems ? WAFER_COST / systems : Infinity;
    yst.set(0, `${dies.length}${yd.n > 1 ? ` chiplets (${Math.floor(dies.length / yd.n)} chips' worth)` : ""}`);
    yst.set(1, `${dies.length - good} of ${dies.length}`, (dies.length - good) / dies.length < 0.2 ? "good" : (dies.length - good) / dies.length < 0.45 ? "warn" : "bad");
    yst.set(2, `${Math.round(Y * 100)}%`, Y > 0.8 ? "good" : Y > 0.5 ? "warn" : "bad");
    yst.set(3, systems ? `$${Math.round(cost).toLocaleString("en-US")}` : "none work", cost < 200 ? "good" : cost < 600 ? "warn" : "bad");
    ycv.paint(() => {
      ycv.clear(); const c = ycv.colors, cx = ycv.ctx, ox = 170, oy = 165, k = 1;
      cx.fillStyle = `color-mix(in srgb, ${c.ink} 12%, ${c.surface})`; cx.beginPath(); cx.arc(ox, oy, R_WAFER * k, 0, Math.PI * 2); cx.fill();
      cx.strokeStyle = c["ink-3"]; cx.lineWidth = 1.5; cx.stroke();
      for (const d of dies) ycv.box(ox + d.x * k + 0.6, oy + d.y * k + 0.6, side * k - 1.2, side * k - 1.2, { fill: d.bad ? `color-mix(in srgb, ${c.bad} 45%, ${c.surface})` : `color-mix(in srgb, ${c.good} 55%, ${c.surface})`, stroke: "transparent", r: Math.min(3, side / 6) });
      for (const [px, py] of defs) ycv.dot(ox + px * k, oy + py * k, 2, c.bad);
      ycv.text("300 mm wafer", ox, oy + R_WAFER + 14, { size: 11, align: "center", color: c["ink-3"] });
      // the model curve on the right: good-chip yield against die area
      const X0 = 400, X1 = 770, Y0 = 280, Y1 = 60, X = (A) => X0 + (A / 860) * (X1 - X0), Yp = (v) => Y0 - v * (Y0 - Y1);
      ycv.line(X0, Y0, X1, Y0, { color: c["ink-3"] }); ycv.line(X0, Y0, X0, Y1, { color: c["ink-3"] });
      for (const v of [0, 0.5, 1]) ycv.text(`${v * 100}%`, X0 - 6, Yp(v) + 4, { size: 10.5, align: "right", color: c["ink-3"], mono: true });
      for (const A of [0, 200, 400, 600, 858]) ycv.text(`${A}`, X(A), Y0 + 16, { size: 10.5, align: "center", color: c["ink-3"], mono: true });
      ycv.text("die area (mm²)", (X0 + X1) / 2, Y0 + 32, { size: 11, align: "center", color: c["ink-3"] });
      ycv.text("share of dies that work", X0, Y1 - 14, { size: 11.5, color: c["ink-2"] });
      for (const [D, dash] of [[0.05, [3, 3]], [0.1, null], [0.3, [6, 4]]]) {
        const own = Math.abs(D - yd.D) < 1e-9;
        cx.strokeStyle = own ? c.accent : c["ink-3"]; cx.lineWidth = own ? 2.5 : 1.2; cx.setLineDash(own ? [] : dash || []); cx.beginPath();
        for (let A = 0; A <= 860; A += 10) { const v = Math.exp(-(A / 100) * D); A ? cx.lineTo(X(A), Yp(v)) : cx.moveTo(X(A), Yp(v)); }
        cx.stroke(); cx.setLineDash([]);
        if (!own) ycv.text(`${D}/cm²`, X(640), Yp(Math.exp(-6.4 * D)) - 6, { size: 10, align: "center", color: c["ink-3"] });
      }
      if (![0.05, 0.1, 0.3].some((D) => Math.abs(D - yd.D) < 1e-9)) {
        cx.strokeStyle = c.accent; cx.lineWidth = 2.5; cx.beginPath();
        for (let A = 0; A <= 860; A += 10) { const v = Math.exp(-(A / 100) * yd.D); A ? cx.lineTo(X(A), Yp(v)) : cx.moveTo(X(A), Yp(v)); }
        cx.stroke();
      }
      ycv.dot(X(a), Yp(Y), 6, c.accent);
      ycv.text(yd.n > 1 ? `each chiplet: ${Math.round(a)} mm²` : `your die: ${yd.A} mm²`, X(a) + (a > 450 ? -10 : 10), Yp(Y) + (a > 450 ? 22 : -8), { size: 12, align: a > 450 ? "right" : "left" });
      ycv.line(X(858), Y1, X(858), Y0, { color: c.bad, dash: [4, 4] });
      ycv.text("reticle limit", X(858) - 4, Y0 - 8, { size: 10.5, align: "right", color: c.bad });
    });
  }

  // ---------- page ----------
  const fig = (key, title) => sticker(key, { title, px: 4 }).outerHTML;
  root.append(
    section(null,
      goal("<p>We know what a transistor is. Now: how do you make <b>80 billion</b> of them on a piece of silicon the size of a postage stamp, nearly all working, for a few hundred dollars? The goal is to print features <b>smaller than the light you print them with</b>, perfectly, layer after layer, millions of chips a year.</p>",
        ["Lines ~10–20 nm wide, placed to within a couple of nm", "Almost every chip on a wafer working", "Cheap enough per transistor to keep Moore's law going"]),
      stackMap(id),
      h("div.vx-row",
        sticker("litho", { title: "Printing with light", caption: "light through a mask, focused onto a wafer coated in light-sensitive resist", px: 4 }),
        sticker("metalstack", { title: "Layer on layer", caption: "transistors first, then a dozen or more layers of copper wires", px: 4 }),
        sticker("defects", { title: "Defects", caption: "one speck on a die and that die is lost", px: 4 })),
    ),
    section("From beach to chip",
      prose("<p>Six steps turn sand into a working chip. The middle one, printing, is repeated hundreds of times and takes about three months.</p>"),
      stepper([
        { title: "1. Sand to pure silicon", html: `<div class="vx-row">${fig("sand", "Quartz sand")}</div><p>Quartz (SiO₂) is heated with carbon to strip the oxygen, then purified chemically until it is <b>99.9999999%</b> silicon (“nine nines”): fewer than one foreign atom per billion. Any more and the dopants, a few atoms per million, wouldn't control the silicon.</p>` },
        { title: "2. Grow one crystal", html: `<div class="vx-row">${fig("ingot", "A single-crystal ingot")}</div><p>A seed crystal is dipped into molten silicon at 1,414 °C and slowly pulled up while turning (the Czochralski method). The atoms line up with the seed, forming one perfect crystal about 30 cm across and 2 m long. Electrons need that perfect lattice: every defect scatters them.</p>` },
        { title: "3. Slice into wafers", html: `<div class="vx-row">${fig("wafer", "A 300 mm wafer")}</div><p>The ingot is sliced into discs 0.775 mm thick and polished flatter than almost anything else made: variations of a few nanometres across 300 mm. A flat surface is needed because the lens has a depth of focus of only ~100 nm.</p>` },
        { title: "4. Print, etch, fill: hundreds of times", html: `<div class="vx-row">${fig("litho", "Lithography")}${fig("metalstack", "The finished stack")}</div><p>Coat with light-sensitive resist, expose a pattern through a mask, develop, then etch away or add material where the pattern says. Implant dopants, grow oxides, deposit metals. First the transistors (“front end”), then 10–20 layers of copper wiring (“back end”). A modern chip takes 1,000+ steps and roughly 70–80 masks, about 20–25 of them exposed with EUV.</p>` },
        { title: "5. Test and cut", html: `<div class="vx-row">${fig("defects", "Wafer with defects")}</div><p>Probe needles test each die on the wafer. Then the wafer is sawn (or laser-cut) into dies. Dies hit by a defect are thrown away, or sold with parts switched off: many GPUs ship with a few SMs disabled for exactly this reason.</p>` },
        { title: "6. Package", html: `<div class="vx-row">${fig("package", "A GPU package")}</div><p>The die is flipped onto a package with thousands of tiny solder bumps. For AI chips it sits on a silicon interposer beside stacks of HBM memory (TSMC's CoWoS), which has itself become a bottleneck in how many AI GPUs can be built.</p>` },
      ]),
    ),
    section("Printing with light",
      prose("<p>A lens can't focus light to a spot much smaller than its wavelength: light is a wave, and it spreads (diffracts) through small gaps. Ernst Abbe worked out the limit in 1873. For chips it's written as the <b>Rayleigh criterion</b>: smallest half-pitch = <b>k₁ × λ / NA</b>. λ is the wavelength. NA (numerical aperture) is how wide a cone of light the lens collects, at most ~0.93 in air and 1.35 with water under the lens. k₁ is how clever the process is, with a hard floor of 0.25. Pick a light source and shrink the lines.</p>"),
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Light source", options: SRC.map((s) => ({ value: s.k, label: s.name })), value: li.src.k, onChange: (k) => { li.src = SRC.find((s) => s.k === k); drawLi(); } }),
          slider({ label: "Pitch (one line + one space)", min: 1.2, max: 3, step: 0.01, value: L10(li.pitch), fmt: (v) => `${Math.round(10 ** v)} nm`, onInput: (v) => { li.pitch = Math.round(10 ** v); drawLi(); } })),
        h("div.stage-body", h("div.wide-canvas", lcv.el), lst.el)),
      callout("tip", "Things to try", "<p>With the ArF laser (193 nm), shrink the pitch until the lines wash out: about 120 nm in air, ~85 nm with water under the lens. The industry printed far smaller than that with 193 nm light for a decade by exposing each layer two, three or four times with offset masks (“multiple patterning”), which multiplies cost and errors. EUV prints ~30 nm pitches in one go.</p>"),
      predict({
        chapter: id,
        question: "EUV's wavelength (13.5 nm) is 14× shorter than ArF's (193 nm), but its NA is only 0.33 against immersion ArF's 1.35. How much finer can EUV print, at the same k₁?",
        options: ["14× finer", "About 3.5× finer", "About the same", "4× coarser"],
        answer: 1,
        explain: "<p>Half-pitch ∝ λ / NA: 193 / 1.35 = 143 against 13.5 / 0.33 = 41, so ~3.5× finer, not 14×. EUV can't use a wide lens because there are no lenses for it at all, only mirrors, which can't collect as wide a cone. High-NA EUV (0.55) buys another 1.7×.</p>",
      }),
    ),
    section("Inside an EUV machine",
      prose("<p>The newest scanners (all made by one company, ASML) print with 13.5 nm light. Making and steering it is so hard that each machine costs ~$200 million, and the High-NA version roughly twice that. Follow the light.</p>"),
      h("div.stage", h("div.stage-body", h("div.wide-canvas", ecv.el))),
      stepper([
        { title: "Make the light", html: "<p>Droplets of molten tin, ~25 µm across, fly past ~50,000 times a second. A small laser pulse flattens each one, then a powerful CO₂ laser pulse turns it into a plasma hotter than 200,000 °C, which glows at 13.5 nm. Only a few percent of the laser's energy comes out as useful light.</p>" },
        { title: "Catch it: the collector", html: "<p>A curved mirror collects the light. No lens can: at 13.5 nm every material absorbs, glass included. The mirror is ~50 alternating layers of molybdenum and silicon, each a few nm thick, tuned so the weak reflections from every layer add up (Bragg reflection). The best such mirrors reflect about 70%.</p>" },
        { title: "Shape it: the illuminator", html: "<p>More mirrors shape the beam into the right angles for the pattern being printed (the “illumination tricks” behind a low k₁). Each bounce keeps ~70%.</p>" },
        { title: "The mask is a mirror too", html: "<p>The pattern is on a reflective mask: the same Mo/Si mirror with absorbing metal where the dark lines go. A single particle on it would print onto every chip, so it's guarded by a membrane only tens of nanometres thick.</p>" },
        { title: "Shrink it: the projection optics", html: "<p>Six more mirrors shrink the pattern 4× onto the wafer. They are polished to within a fraction of a nanometre of the right shape over tens of centimetres.</p>" },
        { title: "Hit the wafer", html: "<p>After about a dozen bounces, only a percent or two of the light that left the plasma reaches the resist. That's why the source needs hundreds of watts, and why throughput (~150+ wafers an hour) was the last problem to be solved before EUV went into production in 2019.</p>" },
      ], { onStep: (k) => { ph = k; ecv.redraw?.(); } }),
      whyChain({
        claim: "Each EUV scanner costs around $200 million and took the industry about 30 years to make work.",
        steps: [
          { q: "Why such short light at all?", a: "<p>Because of the Rayleigh limit: the smallest line is ~k₁ × λ / NA. k₁ can't go below 0.25 and NA can't go much above 1.35, so after 193 nm immersion the only knob left was λ.</p>" },
          { q: "Why is 13.5 nm light so hard to handle?", a: "<p>Its photons carry ~92 eV, enough to knock electrons out of the outer shells of any atom. So everything absorbs it strongly within a few hundred nanometres: glass, air, water, dust. The whole light path has to be in vacuum, and no lens can be made. Every optical element is a mirror.</p>" },
          { q: "Why do mirrors lose so much?", a: "<p>A single surface reflects almost nothing at that wavelength. The multilayer trick adds up ~50 weak reflections in step, but some light is absorbed at every layer, capping reflection near 70%. Twelve mirrors: 0.7¹² ≈ 1.4%.</p>" },
          { q: "Why tin plasma?", a: "<p>You need an emitter that glows brightly in the narrow band the mirrors reflect. Highly ionised tin happens to emit strongly right around 13.5 nm, which is also where Mo/Si mirrors work best. That match is why 13.5 nm was chosen, not a rounder number.</p>" },
        ],
        axiom: { name: "Diffraction: light is a wave", html: "<p>A wave passing through an opening narrower than its wavelength spreads out instead of travelling straight. No lens can focus light to a spot much smaller than ~λ / (2·NA) (Abbe, 1873). To print smaller, you need a shorter wave, and short waves are energetic enough to be absorbed by everything.</p>" },
      }),
      predict({
        chapter: id,
        question: "An EUV mirror reflects about 70% of the light that hits it. With 11 mirrors between the source and the wafer, what fraction arrives?",
        options: ["About 70%", "About 23%", "About 2%", "About 0.01%"],
        answer: 2,
        explain: "<p>0.7¹¹ ≈ 0.02. Losses multiply, so the source has to make ~50× more light than the wafer needs. That's why an EUV source is a building-sized laser system and why each extra mirror is fought over.</p>",
      }),
    ),
    section("Yield: why big chips cost so much",
      prose("<p>Even the cleanest fab leaves a few <b>killer defects</b> per square centimetre: a particle, a missing via, a bridge between wires. A die with one in the wrong place is lost. If defects land at random with density D, the share of dies that survive is about <b>e<sup>−area × D</sup></b> (the Poisson yield model). Double the die and you lose more than twice: fewer fit, and each is more likely to be hit. Make the die bigger, change the defect density, or split it into chiplets.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Die area", min: 25, max: 858, step: 1, value: yd.A, fmt: (v) => `${v} mm²`, onInput: (v) => { yd.A = v; drawY(); } }),
          slider({ label: "Killer defects per cm²", min: 0.02, max: 0.6, step: 0.01, value: yd.D, fmt: (v) => v.toFixed(2), onInput: (v) => { yd.D = +v.toFixed(2); drawY(); } }),
          segmented({ label: "Split into", options: [1, 2, 4, 8].map((n) => ({ value: n, label: n === 1 ? "one die" : `${n} chiplets` })), value: 1, onChange: (n) => { yd.n = n; drawY(); } })),
        h("div.stage-body", h("div.wide-canvas", ycv.el), yst.el)),
      callout("note", "What the numbers mean", `<p>The wafer price (~$${WAFER_COST.toLocaleString("en-US")}, roughly what's been reported for a leading-edge 5 nm-class wafer) is an assumption, and packaging, testing and the chiplets' extra connections cost money that isn't counted here. The model also ignores that defects cluster, which helps big dies a little. But the shape is right, and it's why AMD's server CPUs are made of 8–12 small chiplets, and why NVIDIA's B200 is two reticle-sized dies joined together.</p>`),
      predict({
        chapter: id,
        question: "At 0.1 killer defects per cm², a 400 mm² die yields about 67%. You double it to 800 mm². Working chips per wafer fall by about…",
        options: ["2×: half as many fit", "About 3×", "About 10×", "Not at all: the yield is per die"],
        answer: 1,
        explain: "<p>About half as many fit (a bit fewer, because big squares waste more of the round edge), and the yield falls from e<sup>−0.4</sup> ≈ 67% to e<sup>−0.8</sup> ≈ 45%. Together: ~3× fewer working chips, so each costs ~3× as much for 2× the silicon.</p>",
      }),
    ),
    section("The laws of making chips",
      lawCards([
        { name: "Rayleigh / Abbe", formula: "hp = k₁ λ / NA", glyph: "light", html: "<p>The finest line a lens can print. k₁ ≥ 0.25 and NA ≤ ~1.35, so shorter light is the main lever.</p>", where: [["fab", "this chapter"], ["edges", "edges"]] },
        { name: "Poisson yield", formula: "Y = e^(−A·D)", glyph: "wafer", html: "<p>Good dies fall exponentially with area. Why the reticle limit is a practical limit too, and why chiplets.</p>", where: [["fab", "this chapter"], ["gpu-design", "GPU design"]] },
        { name: "Moore's law", formula: "2× per ~2 years", glyph: "clock", html: "<p>Not physics: an economic pattern. Transistors per chip doubled because cost per transistor kept falling. That's now slower.</p>", where: [["power-wall", "power wall"], ["edges", "edges"]] },
        { name: "Wire resistance", formula: "R = ρ L / A", glyph: "wire", html: "<p>The finest wires are only ~10 nm wide, and copper's ρ rises there. Where the industry tries new metals.</p>", where: [["circuits", "circuits"]] },
      ]),
      table(["Step", "Then (1971, Intel 4004)", "Now (leading edge)"], [
        ["Smallest feature", "10 µm", "~12 nm half-pitch on the tightest layers"],
        ["Light", "mercury lamp, visible/UV", "13.5 nm EUV from tin plasma"],
        ["Wafer", "50 mm (2 inch)", "300 mm"],
        ["Transistors per chip", "2,300", "~80–200 billion (a GPU)"],
        ["Wiring layers", "1", "15 or more"],
        ["Cost of a new fab", "a few million dollars", "$20 billion and up"],
      ]),
      callout("note", "Next", "<p>With billions of switches printed and wired, the question becomes what to build with them. Next: <a href='/learn?c=logic'>wire switches into arithmetic</a>.</p>"),
    ),
    section("In code",
      code({
        "Rayleigh": `sources = {"ArF dry": (193, 0.93), "ArF immersion": (193, 1.35), "EUV": (13.5, 0.33), "High-NA EUV": (13.5, 0.55)}
for name, (lam, NA) in sources.items():
    for k1 in (0.4, 0.3, 0.25):
        print(f"{name:14s} k1={k1}: half-pitch {k1 * lam / NA:5.1f} nm")`,
        "Dies and yield": `import math
def dies_per_wafer(area_mm2, d=300):
    return math.pi * d**2 / (4 * area_mm2) - math.pi * d / math.sqrt(2 * area_mm2)

for area in (100, 400, 800):
    n = dies_per_wafer(area)
    y = math.exp(-area / 100 * 0.1)          # 0.1 killer defects per cm²
    print(area, "mm²:", round(n), "dies,", round(100 * y), "% good,", round(17000 / (n * y)), "$ each")`,
        "Mirrors": `R, mirrors = 0.70, 12
print(f"{R**mirrors:.1%} of the light reaches the wafer")   # ~1.4%`,
      }),
    ),
  );
  drawLi(); drawE(); drawY();
}
