import { h, loop, reducedMotion } from "../lib/core.js";
import { slider, stats, predict, code, callout, section, prose, table, segmented, stepper, button } from "../lib/widgets.js";
import { flowCanvas } from "../lib/flow.js";
import { xyChart } from "../lib/plot.js";
import { goal, whyChain, stackMap, sticker } from "../lib/why.js";
import { walk } from "../lib/walks.js";

function canvas(width, height, label) {
  const fc = flowCanvas({ width, height, label });
  fc.paint = (fn) => { fc.redraw = fn; fn(); };
  return fc;
}
const fmtOps = (v) => (v >= 1e12 ? `${(v / 1e12).toFixed(1)} T` : v >= 1e9 ? `${(v / 1e9).toFixed(1)} G` : v >= 1e6 ? `${(v / 1e6).toFixed(0)} M` : `${v.toFixed(0)} `);

const RES = { "640x480": [640, 480, "1995 PC game"], "1024x768": [1024, 768, "2000"], "1920x1080": [1920, 1080, "HD"], "3840x2160": [3840, 2160, "4K"] };
const CPUS = [["a 1999 CPU", 1e9], ["a 2006 CPU", 5e10], ["a 2006 GPU (8800 GTX)", 3.5e11]];

const MS = [
  [1982, "Geometry Engine", "matrices in hardware", "Jim Clark's chip, the start of Silicon Graphics, did the 4×4 matrix transforms of 3D graphics in hardware. From the very beginning, graphics chips were matrix-multiply machines."],
  [1996, "3dfx Voodoo", "triangles on a card", "A PC add-in card that drew textured triangles, pixel by pixel, far faster than the CPU. The CPU still did the geometry. 3D games took off."],
  [1999, "GeForce 256", "“the first GPU”", "NVIDIA moved transform and lighting onto the chip as well and marketed it as the first graphics processing unit. The whole pipeline now ran on the card."],
  [2001, "GeForce 3", "programmable shaders", "Small programs written by the game developer now ran once per vertex and once per pixel. Fixed functions became code, and code can do anything."],
  [2003, "GPGPU", "math disguised as pixels", "Researchers stored matrices as textures and wrote pixel shaders that computed one output element each. Awkward, but up to 10× faster than CPUs. Stanford's Brook (2004, Ian Buck) wrapped it in a C-like language."],
  [2006, "GeForce 8800 (G80)", "unified shaders", "128 identical scalar cores that could run vertex, pixel or any other work, grouped into SMs and scheduled in warps of 32. The architecture of every NVIDIA GPU since."],
  [2007, "CUDA", "C for the GPU", "Ian Buck, now at NVIDIA, led CUDA: write C, launch a grid of threads, no graphics required. Scientists started porting simulations."],
  [2010, "Fermi", "built for science", "Caches, error-correcting memory and fast double precision: GPUs entered supercomputers. Tianhe-1A, briefly the world's fastest computer, ran on them."],
  [2012, "AlexNet", "deep learning finds the GPU", "Krizhevsky, Sutskever and Hinton trained an 8-layer network on two GTX 580 gaming cards and won ImageNet by a wide margin. AI research moved onto GPUs within a year or two."],
  [2016, "P100", "memory beside the die", "Stacked HBM2 memory on the package, NVLink between GPUs, fast half precision. The first GPU designed with deep learning in mind."],
  [2017, "V100", "tensor cores", "A unit that does a small matrix multiply-add per instruction: 125 TFLOP/s of fp16, 8× the plain cores. Matrix units have dominated since."],
  [2022, "H100", "an AI machine that can draw", "fp8, a transformer engine and 989 TFLOP/s of bf16. Data-centre GPUs have no display outputs at all: the graphics chip became an AI chip."],
];

export default function (root, { id }) {
  // ---------- 1. the problem ----------
  const pb = { res: "1024x768", fps: 60, ops: 100 };
  const pst = stats([{ label: "Pixels per frame" }, { label: "Pixels per second" }, { label: "Operations per second" }]);
  const pcv = canvas(880, 220, "Operations per second a game needs, against what a CPU and a GPU of the time could do");
  function drawPb() {
    const [W, H] = RES[pb.res], px = W * H, need = px * pb.fps * pb.ops;
    pst.set(0, px.toLocaleString()); pst.set(1, `${fmtOps(px * pb.fps)}`); pst.set(2, `${fmtOps(need)}op/s`, need < 1e9 ? "good" : need < 5e10 ? "warn" : "bad");
    pcv.paint(() => {
      pcv.clear(); const c = pcv.colors, x0 = 230, x1 = 860, X = (v) => x0 + ((Math.log10(v) - 7) / 6) * (x1 - x0);
      for (let e = 7; e <= 13; e++) { pcv.line(X(10 ** e), 14, X(10 ** e), 176, { color: c.line, lw: 1 }); pcv.text(["10 M", "100 M", "1 G", "10 G", "100 G", "1 T", "10 T"][e - 7], X(10 ** e), 190, { size: 9.5, align: "center", color: c["ink-3"], mono: true }); }
      const rows = [["this game needs", need, c.accent], ...CPUS.map(([n, v], k) => [n, v, [c.c2, c.c1, c.c3][k]])];
      rows.forEach(([n, v, col], k) => {
        const y = 22 + k * 38; pcv.text(n, x0 - 10, y + 11, { size: 11.5, align: "right", weight: k ? 600 : 700 });
        pcv.box(x0, y, Math.max(4, X(Math.max(1e7, v)) - x0), 22, { fill: `color-mix(in srgb, ${col} ${k ? 45 : 70}%, ${c.surface})`, stroke: "transparent", r: 4 });
        pcv.text(`${fmtOps(v)}op/s${k ? (v >= need ? "  ✓ keeps up" : `  ✗ ${(need / v).toFixed(need / v < 10 ? 1 : 0)}× too slow`) : ""}`, X(Math.max(1e7, v)) + 8, y + 11, { size: 10.5, mono: true, color: k ? (v >= need ? c.good : c.bad) : c.ink });
      });
      pcv.text("operations per second, log scale (peak; a real CPU also had to run the game itself)", 545, 210, { size: 10, align: "center", color: c["ink-3"] });
    });
  }

  // ---------- 2. rasterization ----------
  const rz = { ang: 20, mode: "one", p: 1e9 };
  const GW = 40, GH = 24, CS = 12;
  const rcv = canvas(880, 320, "A triangle rasterised on a 40 × 24 pixel grid: each pixel tests three edge functions");
  const rst = stats([{ label: "Pixels tested" }, { label: "Inside the triangle" }, { label: "Steps taken" }]);
  const verts = () => { const a = (rz.ang * Math.PI) / 180, R = [10.5, 8.5, 11.5]; return [0, 1, 2].map((k) => [20 + R[k] * Math.cos(a + (k * 2 * Math.PI) / 3) * 1.45, 12 + R[k] * Math.sin(a + (k * 2 * Math.PI) / 3)]); };
  const E = (a, b, p) => (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
  const LANES = 128;
  const ra = loop((dt) => { rz.p += rz.mode === "one" ? 480 * dt : LANES * 6 * dt; if (rz.p >= GW * GH) { rz.p = GW * GH; ra.stop(); } drawRz(); }, rcv.el);
  function drawRz() {
    const V = verts(), area = E(V[0], V[1], V[2]), done = Math.min(GW * GH, Math.floor(rz.mode === "one" ? rz.p : Math.ceil(rz.p / LANES) * LANES));
    let inside = 0;
    const px = []; for (let k = 0; k < GW * GH; k++) { const i = k % GW, j = Math.floor(k / GW), p = [i + 0.5, j + 0.5]; const w = [E(V[1], V[2], p) / area, E(V[2], V[0], p) / area, E(V[0], V[1], p) / area]; const inn = w.every((x) => x >= 0); px.push({ i, j, w, inn }); if (inn && k < done) inside++; }
    rst.set(0, `${done} of ${GW * GH}`); rst.set(1, String(inside)); rst.set(2, rz.mode === "one" ? `${done} (one pixel each)` : `${Math.ceil(done / LANES)} (128 pixels each)`, rz.mode === "one" ? "warn" : "good");
    rcv.paint(() => {
      rcv.clear(); const c = rcv.colors, ox = 16, oy = 14, sel = px[GW * 9 + 22];
      px.forEach(({ i, j, w, inn }, k) => {
        const x = ox + i * CS, y = oy + j * CS, tested = k < done;
        if (tested && inn) { const r = Math.round(235 * w[0] + 40 * w[1] + 50 * w[2]), g = Math.round(80 * w[0] + 190 * w[1] + 110 * w[2]), b = Math.round(70 * w[0] + 90 * w[1] + 230 * w[2]); rcv.ctx.fillStyle = `rgb(${r},${g},${b})`; rcv.ctx.fillRect(x, y, CS - 1, CS - 1); }
        else { rcv.ctx.fillStyle = tested ? c.sunken : c.surface; rcv.ctx.fillRect(x, y, CS - 1, CS - 1); }
      });
      rcv.ctx.strokeStyle = c.line; rcv.ctx.lineWidth = 0.5; rcv.ctx.strokeRect(ox, oy, GW * CS, GH * CS);
      if (done < GW * GH) { const k = done, x = ox + (k % GW) * CS, y = oy + Math.floor(k / GW) * CS; if (rz.mode === "one") rcv.box(x - 1, y - 1, CS + 1, CS + 1, { fill: "transparent", stroke: c.accent, lw: 2, r: 2 }); else rcv.box(ox - 1, y - 1, GW * CS + 2, Math.ceil(LANES / GW) * CS + 1, { fill: "transparent", stroke: c.accent, lw: 2, r: 3 }); }
      rcv.ctx.beginPath(); V.forEach(([x, y], k) => (k ? rcv.ctx.lineTo(ox + x * CS, oy + y * CS) : rcv.ctx.moveTo(ox + x * CS, oy + y * CS))); rcv.ctx.closePath(); rcv.ctx.strokeStyle = c.ink; rcv.ctx.lineWidth = 1.5; rcv.ctx.stroke();
      V.forEach(([x, y], k) => { rcv.dot(ox + x * CS, oy + y * CS, 5, ["#eb5046", "#28be6e", "#3c64e6"][k]); });
      // one pixel's test
      const sx = ox + sel.i * CS, sy = oy + sel.j * CS; rcv.box(sx - 2, sy - 2, CS + 3, CS + 3, { fill: "transparent", stroke: c.ink, lw: 2, r: 2 });
      const x0 = 520; rcv.box(x0, 14, 346, 288, { fill: c.surface, stroke: c.line, r: 10 });
      rcv.text("one pixel's whole job", x0 + 14, 32, { size: 10.5, color: c["ink-3"], mono: true });
      rcv.text("for each edge of the triangle:", x0 + 14, 58, { size: 11.5 });
      rcv.text("E = (bx − ax)(py − ay) − (by − ay)(px − ax)", x0 + 14, 80, { size: 11, mono: true });
      sel.w.forEach((w, k) => rcv.text(`edge ${k + 1}: ${w >= 0 ? "+" : "−"}  (${w.toFixed(2)})`, x0 + 24, 108 + k * 22, { size: 11.5, mono: true, color: w >= 0 ? c.good : c.bad }));
      rcv.text(sel.inn ? "all positive: inside, so colour it" : "a negative: outside, skip it", x0 + 14, 184, { size: 11.5, color: sel.inn ? c.good : c.bad });
      rcv.text("the three numbers also blend the corner", x0 + 14, 214, { size: 11, color: c["ink-2"] }); rcv.text("colours: red, green and blue here", x0 + 14, 232, { size: 11, color: c["ink-2"] });
      rcv.text("no pixel needs any other pixel's answer:", x0 + 14, 262, { size: 11.5, color: c.accent }); rcv.text("all of them can run at the same time", x0 + 14, 282, { size: 11.5, color: c.accent });
    });
  }
  function drawRun() { rz.p = 0; if (reducedMotion()) { rz.p = GW * GH; drawRz(); return; } ra.start(); }

  // ---------- 3. the graphics pipeline ----------
  let gs = 0;
  const STAGES = [["Vertices", "a model is a list of corner points"], ["Transform", "a 4×4 matrix per vertex"], ["Rasterise", "which pixels does each triangle cover?"], ["Shade", "a small program per pixel"], ["Output", "keep the nearest, write the frame"]];
  const gcv = canvas(880, 280, "The graphics pipeline: vertices, transform, rasterise, shade, output");
  gcv.paint(() => {
    gcv.clear(); const c = gcv.colors, ac = c.accent;
    STAGES.forEach(([n, s], k) => { const x = 14 + k * 172, on = k === gs; gcv.box(x, 12, 160, 46, { fill: on ? `color-mix(in srgb, ${ac} 18%, ${c.surface})` : c.surface, stroke: on ? ac : c.line, lw: on ? 2.5 : 1, r: 8 }); gcv.text(`${k + 1}. ${n}`, x + 80, 28, { size: 12, align: "center", color: on ? ac : c.ink }); gcv.text(s, x + 80, 46, { size: 9, align: "center", color: c["ink-3"] }); if (k < 4) gcv.text("→", x + 166, 35, { size: 13, color: c["ink-3"] }); });
    // a cube, rotated more after the transform
    const rot = gs >= 1 ? 0.6 : 0.2, P = [];
    for (const x of [-1, 1]) for (const y of [-1, 1]) for (const z of [-1, 1]) { const X = x * Math.cos(rot) - z * Math.sin(rot), Z = x * Math.sin(rot) + z * Math.cos(rot), Y = y * Math.cos(0.4) - Z * Math.sin(0.4); P.push([150 + X * 62, 170 + Y * 62]); }
    const edges = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]];
    if (gs <= 1) { edges.forEach(([a, b]) => gcv.line(P[a][0], P[a][1], P[b][0], P[b][1], { color: c["ink-3"], lw: 1.2 })); P.forEach(([x, y]) => gcv.dot(x, y, 4.5, gs === 0 ? c.c1 : ac)); }
    const rx = 330;
    if (gs === 0) { gcv.text("8 corners, 12 triangles. A game character has ~100,000.", rx, 120, { size: 12.5 }); gcv.text("each corner: a position (x, y, z), a colour, a texture coordinate", rx, 146, { size: 11.5, color: c["ink-2"] }); }
    if (gs === 1) {
      gcv.text("for every vertex: rotate, move, add perspective", rx, 96, { size: 12.5 });
      const mx = rx, my = 116; for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) { gcv.box(mx + j * 38, my + i * 28, 34, 24, { fill: `color-mix(in srgb, ${c.c6} 25%, ${c.surface})`, stroke: "transparent", r: 3 }); gcv.text(`m${i}${j}`, mx + j * 38 + 17, my + i * 28 + 12, { size: 10, align: "center", mono: true }); }
      gcv.text("×", mx + 166, my + 54, { size: 16, align: "center" });
      ["x", "y", "z", "1"].forEach((t, i) => { gcv.box(mx + 180, my + i * 28, 34, 24, { fill: `color-mix(in srgb, ${c.c1} 25%, ${c.surface})`, stroke: "transparent", r: 3 }); gcv.text(t, mx + 197, my + i * 28 + 12, { size: 11, align: "center", mono: true }); });
      gcv.text("=", mx + 232, my + 54, { size: 16, align: "center" });
      ["x′", "y′", "z′", "w′"].forEach((t, i) => { gcv.box(mx + 248, my + i * 28, 34, 24, { fill: `color-mix(in srgb, ${ac} 25%, ${c.surface})`, stroke: "transparent", r: 3 }); gcv.text(t, mx + 265, my + i * 28 + 12, { size: 11, align: "center", mono: true }); });
      gcv.text("16 multiply-adds per vertex,", mx + 310, my + 30, { size: 12, color: ac }); gcv.text("a million vertices a frame:", mx + 310, my + 50, { size: 12, color: ac }); gcv.text("a matrix multiply, from day one", mx + 310, my + 70, { size: 12, color: ac, weight: 700 });
    }
    if (gs >= 2) {
      const ox = 40, oy = 82, n = 16, cs = 11, T = [[2, 13], [14, 3], [13, 14]];
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        const p = [i + 0.5, j + 0.5], e = [E(T[0], T[1], p), E(T[1], T[2], p), E(T[2], T[0], p)], inn = e.every((v) => v >= 0) || e.every((v) => v <= 0);
        let col = c.sunken; if (inn) col = gs === 2 ? `color-mix(in srgb, ${c.c1} 45%, ${c.surface})` : `color-mix(in srgb, ${c.c4} ${30 + Math.round((i / n) * 55)}%, ${c.c2})`;
        gcv.ctx.fillStyle = col; gcv.ctx.fillRect(ox + i * cs, oy + j * cs, cs - 1, cs - 1);
      }
      if (gs === 4) { gcv.ctx.fillStyle = `color-mix(in srgb, ${c.c3} 70%, ${c.surface})`; gcv.ctx.fillRect(ox + 9 * cs, oy + 4 * cs, 6 * cs, 6 * cs); }
      const t = gs === 2 ? ["which pixels' centres are inside each triangle?", "the edge test from above, for every pixel at once"]
        : gs === 3 ? ["run the developer's shader once per pixel:", "colour = texture × lighting (a dot product), ~10–1,000 ops each"]
          : ["a depth buffer stores how far each pixel's surface is;", "a nearer triangle (green) wins, then the frame goes to the screen"];
      gcv.text(t[0], rx, 140, { size: 12.5 }); gcv.text(t[1], rx, 164, { size: 11.5, color: c["ink-2"] });
      gcv.text(gs === 3 ? "in 2001 this became programmable: the door to general computing" : gs === 2 ? "fixed-function hardware until 2006; still dedicated units today" : "60 times a second, for every pixel on the screen", rx, 196, { size: 11.5, color: ac });
    }
  });

  // ---------- 4. history chart ----------
  const hist = xyChart({ width: 880, height: 280, yLog: true, xLabel: "year", yLabel: "peak TFLOP/s (approximate)", xDomain: [2005, 2025], yDomain: [0.03, 5000], fmtX: (v) => String(Math.round(v)) });
  hist.update([
    { name: "GPU, fp16/bf16 tensor cores", color: "var(--accent)", points: [[2017, 125], [2020, 312], [2022, 989], [2024, 2250]], dots: true },
    { name: "GPU, plain fp32", color: "var(--c3)", points: [[2006, 0.35], [2008, 0.93], [2010, 1.58], [2012, 3.1], [2014, 4.6], [2016, 10.6], [2017, 15.7], [2020, 19.5], [2022, 67], [2024, 80]], dots: true },
    { name: "Top CPU, fp32 vector", color: "var(--c2)", points: [[2006, 0.085], [2010, 0.16], [2013, 0.45], [2017, 3.4], [2022, 7.2]], dots: true },
  ], { xDomain: [2005, 2025], yDomain: [0.03, 5000], marks: [{ x: 2007, label: "CUDA" }, { x: 2012, label: "AlexNet" }, { x: 2017, label: "tensor cores" }] });
  let hs = 0;
  const tcv = canvas(880, 150, "A timeline of graphics and GPU milestones from 1982 to 2022");
  tcv.paint(() => {
    tcv.clear(); const c = tcv.colors, X = (y) => (y < 1995 ? 40 + ((y - 1980) / 15) * 70 : 110 + ((y - 1995) / 29) * 740), y0 = 70;
    tcv.line(30, y0, 860, y0, { color: c.line, lw: 2 }); tcv.text("⋯", 75, y0 + 18, { size: 11, align: "center", color: c["ink-3"] });
    for (const yr of [1980, 1995, 2000, 2005, 2010, 2015, 2020]) tcv.text(String(yr), X(yr), y0 + 18, { size: 9.5, align: "center", color: c["ink-3"], mono: true });
    MS.forEach(([yr, n], k) => { const cur = k === hs, up = k % 2 === 0, ly = up ? y0 - 26 - (k % 4 === 0 ? 0 : 16) : y0 + 34 + (k % 4 === 1 ? 0 : 16); tcv.line(X(yr), y0, X(yr), ly + (up ? 7 : -7), { color: cur ? c.accent : c.line, lw: cur ? 2 : 1 }); tcv.dot(X(yr), y0, cur ? 7 : 4.5, cur ? c.accent : k < hs ? c.c3 : c["ink-3"]); tcv.text(n, X(yr), ly, { size: cur ? 11.5 : 9.5, align: "center", color: cur ? c.accent : c["ink-2"], weight: cur ? 700 : 600 }); });
    const [yr, n, idea] = MS[hs]; tcv.text(`${yr} · ${n}: ${idea}`, 440, 140, { size: 12, align: "center" });
  });

  // ---------- graphics and AI: the same shape of work ----------
  const fit = { k: "matrix" };
  const fcv = canvas(800, 270, "One graphics job and the matching deep-learning job, drawn side by side");
  function drawFit() {
    fcv.paint(() => {
      fcv.clear(); const c = fcv.colors, k = fit.k;
      const A = c.c1, B = c.c4, M = c.accent;
      const panel = (x, title) => { fcv.box(x, 8, 360, 254, { fill: c.surface, stroke: c.line, r: 10 }); fcv.text(title, x + 14, 26, { size: 12 }); };
      panel(10, "Graphics needed…"); panel(430, "…and deep learning needs");
      fcv.text("=", 400, 135, { size: 26, align: "center", color: c["ink-3"] });
      const sides = [[10, "vertices", "4×4 transform", "pixel", "texture", "colour channel"], [430, "inputs", "weight matrix W", "neuron", "weights", "bf16 / fp8 number"]];
      sides.forEach(([x, item, mat, cell, data, num], si) => {
        const col = si ? B : A;
        if (k === "matrix") {
          for (let i = 0; i < 7; i++) { fcv.box(x + 24, 52 + i * 26, 46, 18, { fill: `color-mix(in srgb, ${col} 45%, transparent)`, stroke: "transparent", r: 4 }); fcv.line(x + 72, 61 + i * 26, x + 138, 140, { color: c.line, lw: 1 }); }
          fcv.box(x + 140, 100, 80, 80, { fill: `color-mix(in srgb, ${M} 22%, ${c.surface})`, stroke: M, lw: 1.5, r: 6 });
          for (let r = 0; r < 4; r++) for (let q = 0; q < 4; q++) fcv.dot(x + 155 + q * 17, 115 + r * 17, 3, M);
          fcv.text(mat, x + 180, 194, { size: 10.5, align: "center", color: M });
          for (let i = 0; i < 7; i++) { fcv.line(x + 222, 140, x + 288, 61 + i * 26, { color: c.line, lw: 1 }); fcv.box(x + 290, 52 + i * 26, 46, 18, { fill: `color-mix(in srgb, ${col} 75%, transparent)`, stroke: "transparent", r: 4 }); }
          fcv.text(`millions of ${item}, one matrix`, x + 180, 242, { size: 11, align: "center", color: c["ink-2"] });
        } else if (k === "indep") {
          for (let r = 0; r < 7; r++) for (let q = 0; q < 12; q++) {
            const v = si ? (Math.sin(r * 1.7 + q * 0.9) + 1) / 2 : (Math.sin(r * 0.6) * Math.cos(q * 0.45) + 1) / 2;
            fcv.box(x + 30 + q * 25, 46 + r * 25, 21, 21, { fill: `color-mix(in srgb, ${col} ${Math.round(25 + v * 60)}%, transparent)`, stroke: "transparent", r: si ? 10 : 3 });
          }
          fcv.text(`each ${cell} needs only its own inputs: no ${cell} waits for another`, x + 180, 242, { size: 10.5, align: "center", color: c["ink-2"] });
        } else if (k === "stream") {
          fcv.box(x + 24, 60, 90, 130, { fill: `color-mix(in srgb, ${col} 25%, ${c.surface})`, stroke: col, lw: 1, r: 6 });
          fcv.text(si ? "HBM" : "video memory", x + 69, 125, { size: 10.5, align: "center" });
          fcv.box(x + 246, 60, 90, 130, { fill: c.sunken, stroke: c.line, lw: 1, r: 6 });
          for (let r = 0; r < 6; r++) for (let q = 0; q < 4; q++) fcv.box(x + 254 + q * 20, 68 + r * 20, 15, 15, { fill: `color-mix(in srgb, ${c.good} 60%, transparent)`, stroke: "transparent", r: 2 });
          for (let i = 0; i < 9; i++) fcv.line(x + 116, 72 + i * 13, x + 244, 72 + i * 13, { color: `color-mix(in srgb, ${col} 70%, transparent)`, lw: 3 });
          fcv.text(`all the ${data}, every ${si ? "step" : "frame"}`, x + 180, 214, { size: 11, align: "center", color: c["ink-2"] });
          fcv.text(si ? "H100: 3.35 TB/s" : "GTX 580: 192 GB/s", x + 180, 236, { size: 10.5, align: "center", color: c["ink-3"], mono: true });
        } else {
          if (!si) {                                   // 8 bits per colour channel: 256 shades are plenty for an eye
            for (let i = 0; i < 256; i++) fcv.box(x + 24 + i * (312 / 256), 80, 312 / 256 + 0.6, 60, { fill: `rgb(${i},${Math.round(i * 0.62)},${255 - Math.round(i * 0.75)})`, stroke: "transparent", lw: 0, r: 0 });
            fcv.text("8 bits per channel = 256 shades", x + 180, 162, { size: 11, align: "center", color: c["ink-2"] });
            fcv.text("the eye can't see finer steps", x + 180, 182, { size: 10.5, align: "center", color: c["ink-3"] });
          } else {                                     // how many values each format can hold between 1 and 2
            [["fp32", 23], ["bf16", 7], ["fp8 E4M3", 3], ["fp4 E2M1", 1]].forEach(([nm, b], i) => {
              const y = 60 + i * 34, n = 2 ** b;
              fcv.text(nm, x + 70, y + 8, { size: 10.5, align: "right", color: c["ink-2"] });
              fcv.line(x + 80, y + 8, x + 330, y + 8, { color: c.line, lw: 1 });
              const shown = Math.min(n, 128);
              for (let j = 0; j <= shown; j++) fcv.line(x + 80 + (j / shown) * 250, y + 2, x + 80 + (j / shown) * 250, y + 14, { color: `color-mix(in srgb, ${col} 80%, transparent)`, lw: n > 128 ? 0.6 : 1.4 });
              fcv.text(n > 128 ? `${n.toLocaleString()} steps` : `${n} steps`, x + 330, y - 6, { size: 9.5, align: "right", color: c["ink-3"], mono: true });
            });
            fcv.text("values between 1 and 2 in each format", x + 180, 212, { size: 10.5, align: "center", color: c["ink-2"] });
          }
          fcv.text(si ? "networks train in 16 bits, run in 8 or 4" : "colours never needed 32-bit floats", x + 180, 236, { size: 11, align: "center", color: c["ink-2"] });
        }
      });
    });
  }
  drawFit();

  root.append(
    section(null,
      goal("<p>The GPU wasn't invented for AI. It was invented to draw a 3D world <b>fast enough to play a game</b>: every pixel on the screen, recomputed 60 times a second. That job turned out to be millions of small, identical, independent calculations, mostly multiply-adds on matrices. Thirty years later, neural networks needed exactly the same thing.</p>",
        ["Every pixel, every frame", "Millions of independent sums", "Cheap enough for a gamer's PC"]),
      stackMap("gpu-design"),
      h("div.vx-row", sticker("raster", { title: "A rasterised triangle", caption: "each pixel decides for itself whether it is inside", px: 4 }), sticker("sm", { title: "2006: unified shaders", caption: "the same cores for any job, grouped into SMs", px: 4 }), sticker("package", { title: "2016 on: HBM beside the die", caption: "graphics bandwidth, repurposed for AI", px: 4 })),
    ),
    section("The problem: every pixel, every frame",
      prose("<p>A 3D game must work out the colour of every pixel, then do it all again for the next frame. Pick a screen, a frame rate and how much work each pixel takes (lighting, textures, shadows), and compare the total with what processors of the day could do.</p>"),
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Screen", options: Object.entries(RES).map(([k, [, , n]]) => ({ value: k, label: `${k.replace("x", "×")} (${n})` })), value: pb.res, onChange: (v) => { pb.res = v; drawPb(); } }),
          slider({ label: "Frames per second", min: 24, max: 144, step: 6, value: pb.fps, onInput: (v) => { pb.fps = v; drawPb(); } }),
          slider({ label: "Operations per pixel", min: 10, max: 2000, step: 10, value: pb.ops, onInput: (v) => { pb.ops = v; drawPb(); } })),
        h("div.stage-body", pst.el, h("div.wide-canvas", pcv.el))),
      predict({ chapter: id, question: "A 1080p screen at 60 frames per second, with 100 operations per pixel. Roughly how many operations per second?", options: ["About 12 million", "About 12 billion", "About 12 trillion", "About 120,000"], answer: 1, explain: "<p>1920 × 1080 ≈ 2.07 million pixels, × 60 frames ≈ 124 million pixels a second, × 100 operations ≈ 12 billion per second. A 2006 desktop CPU peaked around 50 billion and also had to run the game. Modern games spend thousands of operations per pixel: trillions per second, which only a GPU delivers.</p>" }),
    ),
    section("Rasterisation: every pixel on its own",
      prose("<p>A 3D scene is made of triangles. To draw one, each pixel asks: is my centre inside? Three quick sums (one per edge) answer it, and the same three numbers blend the corner colours. Crucially, <b>no pixel needs another pixel's answer</b>. Draw the triangle one pixel at a time, as a CPU core would, then 128 at a time, as a 2006 GPU could.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Rotate the triangle", min: 0, max: 120, value: rz.ang, fmt: (v) => `${v}°`, onInput: (v) => { rz.ang = v; drawRz(); } }),
          segmented({ label: "Hardware", options: [{ value: "one", label: "One pixel at a time" }, { value: "many", label: "128 pixels at once" }], value: rz.mode, onChange: (v) => { rz.mode = v; drawRun(); } }),
          h("div.btn-row", button("Draw", drawRun, "primary"))),
        h("div.stage-body", rst.el, h("div.wide-canvas", rcv.el))),
    ),
    section("The graphics pipeline",
      prose("<p>Every frame flows through the same five stages, in hardware. Two of them are the reason GPUs later suited AI. The transform is a <b>matrix multiply</b> per vertex. Shading is a <b>small program per pixel</b>, run millions of times, and no run depends on another.</p>"),
      h("div.wide-canvas", gcv.el),
      stepper(STAGES.map(([n, s], k) => ({ title: `${k + 1}. ${n}`, html: [
        "<p>A model is a mesh of triangles defined by its corner points (vertices): position, colour, texture coordinates, surface direction. A modern game frame has millions of them.</p>",
        "<p>Each vertex is multiplied by a 4×4 matrix that rotates, moves and projects it onto the screen. The same matrix, millions of vertices: exactly a matrix multiply. Jim Clark's 1982 Geometry Engine did this in hardware.</p>",
        "<p>For each triangle, find the pixels it covers with the edge test above. Fixed hardware does this even on today's GPUs, at billions of triangles per second.</p>",
        "<p>A <b>pixel shader</b> computes each covered pixel's colour: look up textures, add up lights with dot products, apply shadows. Since 2001 the developer writes this program. That's the moment graphics chips became programmable computers.</p>",
        "<p>A depth buffer keeps the nearest surface at each pixel, blending handles glass and smoke, and the finished frame is shown. Then the whole thing starts again, 16 ms later.</p>",
      ][k] })), { onStep: (k) => { gs = k; gcv.redraw?.(); } }),
    ),
    section("Math disguised as pixels",
      prose("<p>Around 2001–2006, before CUDA, researchers noticed that a pixel shader is a tiny parallel computer with fast memory. To multiply matrices, they stored A and B as <b>textures</b> (images of numbers) and drew a rectangle the size of the answer. A shader then computed one output element per pixel: a dot product of a row and a column. Step through it.</p>"),
      walk("gpu-history"),
      callout("note", "Awkward, and it worked", "<p>Everything had to be phrased as drawing: loops became rendering passes, outputs had to be images, and reading results back was slow. It was still several times faster than CPUs for the right problems, enough to convince NVIDIA to build CUDA: the same hardware, without pretending to draw.</p>"),
    ),
    section("Thirty years in one chart",
      prose("<p>Peak arithmetic per second, GPU against CPU, on a log scale (approximate figures from spec sheets). The gap opened as soon as GPUs became programmable, then widened again when tensor cores arrived.</p>"),
      h("div.stage", h("div.stage-body", hist.el)),
      h("div.wide-canvas", tcv.el),
      stepper(MS.map(([yr, n, idea, txt]) => ({ title: `${yr}: ${n}`, html: `<p><b>${idea[0].toUpperCase() + idea.slice(1)}.</b> ${txt}</p>` })), { onStep: (k) => { hs = k; tcv.redraw?.(); } }),
    ),
    section("Why graphics hardware fit AI",
      h("div.stage",
        h("div.stage-controls", segmented({ label: "Match", options: [{ value: "matrix", label: "One matrix, many items" }, { value: "indep", label: "Independent work" }, { value: "stream", label: "Streaming memory" }, { value: "prec", label: "Low precision" }], value: fit.k, onChange: (v) => { fit.k = v; drawFit(); } })),
        h("div.stage-body", h("div.wide-canvas", fcv.el))),
      table(["Graphics needed", "Deep learning needs", "So the GPU already had"], [
        ["the same matrix applied to millions of vertices", "the same weights applied to millions of inputs", "fast multiply-add on small matrices"],
        ["every pixel computed independently", "every neuron and every example computed independently", "thousands of threads and no dependencies to wait for"],
        ["textures streamed from memory every frame", "weights and activations streamed every step", "very wide, fast memory (GDDR, then HBM)"],
        ["colours only need ~8–16 bits of precision", "networks train fine in 16 bits, infer in 8 or 4", "fast low-precision arithmetic"],
        ["a market of hundreds of millions of gamers", "cheap hardware for researchers", "a $500 card with teraflops in 2012"],
      ]),
      callout("tip", "The hardware lottery", "<p>Sara Hooker's term (2020): ideas win partly because they suit the hardware that happens to exist. Neural networks were invented in the 1950s–80s and only took off once they met the gaming GPU. Ideas that suit GPUs badly (very sparse or branchy models) may be underexplored for the same reason.</p>"),
    ),
    section("Dig down: why is drawing an image so parallel?",
      whyChain({
        claim: "Each pixel of a rendered image can be computed independently of every other pixel.",
        steps: [
          { q: "What does a pixel represent?", a: "<p>The light that arrives at one point of the camera from one direction: one ray traced back into the scene, or one fragment of a triangle in rasterisation.</p>" },
          { q: "Why doesn't one ray's answer depend on another's?", a: "<p>Because the light along one ray is fixed by the scene (surfaces, materials, light sources), not by the light travelling along other rays. Two beams of light cross without bending or dimming each other.</p>" },
          { q: "Why don't beams of light affect each other?", a: "<p>Light in empty space obeys Maxwell's equations, which are <b>linear</b>: if two light fields are solutions, so is their sum. Each one passes through the other as if it weren't there (quantum electrodynamics does allow light to scatter off light, but at visible energies the effect is far too weak ever to matter in a picture).</p>" },
        ],
        axiom: { name: "Superposition: electromagnetism is linear in vacuum", html: "<p>Because light doesn't interact with light, the image is a sum of independent contributions, so each pixel is its own problem. That made graphics perfectly parallel, which made the GPU a machine for many independent multiply-adds. Neural networks are parallel for a different reason (the same function applied to many inputs, and many independent dot products), but they need the same machine.</p>" },
      }),
      predict({ chapter: id, question: "Why did NVIDIA's 2006 G80 replace separate vertex and pixel units with one kind of “unified” core?", options: ["To make the chip smaller", "Scenes vary: some frames are all vertices, some all pixels. Separate units sat idle; identical cores can be given whatever work there is", "Pixels and vertices use the same maths, so one unit was enough", "To support CUDA, which was already planned"], answer: 1, explain: "<p>With fixed pools, a frame heavy in geometry left the pixel units idle and vice versa. Identical, general cores with a scheduler keep everything busy whatever the mix. The same generality is what let CUDA, a year later, run arbitrary C on them. (CUDA and G80 were developed together, but load balancing was the graphics reason.)</p>" }),
    ),
    section("In code",
      code({
        "Rasterise with NumPy": `import numpy as np
W, H = 40, 24
v = np.array([[5, 3], [35, 8], [15, 21]], float)        # triangle corners
ys, xs = np.mgrid[0:H, 0:W] + 0.5                      # every pixel centre at once

def edge(a, b):                                        # > 0 on the inside of edge a→b
    return (b[0] - a[0]) * (ys - a[1]) - (b[1] - a[1]) * (xs - a[0])

w = np.stack([edge(v[1], v[2]), edge(v[2], v[0]), edge(v[0], v[1])])
inside = (w >= 0).all(0) | (w <= 0).all(0)
print(inside.sum(), "pixels covered")                  # no loop over pixels: data parallel`,
        "One thread per pixel (CUDA)": `__global__ void shade(uchar4* img, int W, int H, float t) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;      // this thread's pixel
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    float u = x / (float)W, v = y / (float)H;           // the same tiny program, every pixel
    img[y * W + x] = make_uchar4(255 * u, 255 * v, 128 + 127 * sinf(t), 255);
}
// shade<<<dim3((W+15)/16, (H+15)/16), dim3(16, 16)>>>(img, W, H, t);`,
        "Matrix multiply as a pixel shader (GLSL, c. 2004)": `// A and B are stored as textures; we draw a quad the size of C,
// so this runs once per output pixel, i.e. once per element C[i][j]
uniform sampler2D A, B;
uniform int N;
void main() {
    vec2 ij = gl_FragCoord.xy;            // which pixel = which (j, i)
    float s = 0.0;
    for (int k = 0; k < N; k++) {
        float t = (float(k) + 0.5) / float(N);              // texture coordinates run 0..1
        s += texture2D(A, vec2(t, ij.y / float(N))).r * texture2D(B, vec2(ij.x / float(N), t)).r;
    }
    gl_FragColor = vec4(s);              // the "colour" is the answer
}`,
      }),
    ),
  );
  drawPb(); drawRz();
}
