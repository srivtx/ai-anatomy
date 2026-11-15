import { h } from "../lib/core.js";
import { slider, segmented, toggle, button, stats, predict, stepper, code, callout, section, prose, table } from "../lib/widgets.js";
import { flowCanvas } from "../lib/flow.js";
import { xyChart } from "../lib/plot.js";
import { goal, whyChain, stackMap, sticker } from "../lib/why.js";
import { walk } from "../lib/walks.js";

function canvas(width, height, label) {
  const fc = flowCanvas({ width, height, label });
  fc.paint = (fn) => { fc.redraw = fn; fn(); };
  return fc;
}
const us = (s) => (s >= 1e-3 ? `${(s * 1e3).toFixed(s >= 1e-2 ? 1 : 2)} ms` : `${(s * 1e6).toFixed(s >= 1e-5 ? 0 : 1)} µs`);

// ---------------------------------------------------------------------------------------------
// The layers one call goes through. where: cpu-user, cpu-kernel, gpu. when: how often this layer does work.
// ---------------------------------------------------------------------------------------------
const LAYERS = [
  { name: "Your Python", where: "user", who: "you", when: "every call", time: "~1 µs",
    gets: "a line of code", hands: "a call to torch.matmul(x, W)",
    html: "<p><code>y = x @ W</code>. The Python interpreter reads the bytecode, sees the <code>@</code> operator and calls the tensor's <code>__matmul__</code>. Python never touches a number in the matrices: x and W are handles to memory on the GPU. Each Python-level operation costs around a microsecond of interpreter time, which is why a Python loop over elements is hopeless but a Python call per matrix is fine.</p>" },
  { name: "PyTorch dispatcher", where: "user", who: "PyTorch (C++)", when: "every call", time: "a few µs",
    gets: "torch.matmul(x, W)", hands: "aten::mm for CUDA, bf16",
    html: "<p>PyTorch's C++ core looks at the arguments and picks an implementation through a <b>dispatcher</b>: a table keyed by device (CUDA), dtype (bf16), and modes like autograd. If gradients are needed it first records this op on the autograd graph, then calls the CUDA bf16 matmul. It also allocates y from its caching allocator, which reuses freed GPU memory instead of asking the driver each time.</p>" },
  { name: "Math library (cuBLAS)", where: "user", who: "NVIDIA", when: "every call", time: "~1 µs",
    gets: "a 4096 × 4096 bf16 matmul", hands: "kernel #N, grid 32 × 16, 4 pointers",
    html: "<p>cuBLAS ships hundreds of precompiled matmul kernels, each tuned for a range of shapes, data types and GPU generations. A heuristic picks the one that suits 4096 × 4096 bf16 on this GPU (tile size, number of pipeline stages, how to split the work), and fills in its arguments. Writing these kernels is where the deep hardware knowledge lives.</p>" },
  { name: "CUDA runtime", where: "user", who: "NVIDIA (libcudart)", when: "every launch", time: "≪ 1 µs",
    gets: "kernel, grid, block, arguments", hands: "cuLaunchKernel(...) on a stream",
    html: "<p><code>cudaLaunchKernel</code>, which is what <code>kernel&lt;&lt;&lt;grid, block&gt;&gt;&gt;(args)</code> compiles to. It packs the arguments and passes the launch to the driver, attached to a <b>stream</b>: a queue whose work runs in order. It returns immediately: the CPU never waits for the GPU unless you ask it to.</p>" },
  { name: "CUDA driver (user mode)", where: "user", who: "NVIDIA (libcuda)", when: "every launch", time: "~2–4 µs",
    gets: "a launch on a stream", hands: "bytes in a command buffer + a doorbell write",
    html: "<p>The driver writes a small command (which kernel, grid size, where the arguments are) into a <b>pushbuffer</b>, a ring buffer in memory the GPU can read. Then it writes to a <b>doorbell</b>: a GPU register mapped into the process's address space. That write crosses PCIe as a single memory store. No system call, no operating system involved: this is how a launch costs microseconds, not tens of them.</p>" },
  { name: "Kernel driver and OS", where: "kernel", who: "NVIDIA (nvidia.ko) + Linux", when: "setup and allocations", time: "not per launch",
    gets: "cudaMalloc, context creation", hands: "page tables, channels, pinned memory",
    html: "<p>The part that runs inside the operating system kernel isn't on the per-launch path at all. It did the setup: created a context for your process with its own GPU page tables (so one process can't read another's memory), gave it command channels and mapped the doorbell. It runs again when you allocate memory or pin host memory for fast copies. Protection is set up once; after that, user code talks to the hardware directly.</p>" },
  { name: "GPU front end", where: "gpu", who: "NVIDIA hardware and firmware", when: "every launch", time: "~1 µs",
    gets: "a doorbell", hands: "512 thread blocks, one SM at a time",
    html: "<p>On the GPU, the host interface wakes on the doorbell and fetches the commands from the pushbuffer over PCIe by DMA. A hardware scheduler (NVIDIA calls it the GigaThread engine, or compute work distributor) hands thread blocks to SMs as each one has room: 132 at first, then a new block whenever one finishes. This is the scheduling you saw in waves in the <a href='/learn?c=cuda'>CUDA chapter</a>.</p>" },
  { name: "Machine code (SASS)", where: "gpu", who: "nvcc / ptxas, ahead of time", when: "compiled once", time: "the kernel itself",
    gets: "a thread block", hands: "warp instructions: LDG, HMMA, FFMA…",
    html: "<p>Each SM runs the kernel's <b>SASS</b>: the real machine code for this GPU generation. It was compiled long before, from CUDA C++ via the virtual instruction set PTX, and sits inside cuBLAS's library file. Instructions like <code>LDGSTS</code> (copy global memory to shared), <code>HGMMA</code> (a warp-group matrix multiply on the tensor cores) and <code>FFMA</code> (fused multiply-add). One instruction drives 32 threads.</p>" },
  { name: "The SM", where: "gpu", who: "silicon", when: "every clock", time: "0.55 ns a tick",
    gets: "warp instructions", hands: "electrons, eventually",
    html: "<p>Warp schedulers issue one instruction per clock from whichever warp is ready, the tensor cores do 512 multiply-adds a tick, and the whole thing is the <a href='/learn?c=gpu-design'>machine you designed</a>. From here down it's the <a href='/learn?c=all-the-way-down'>zoom to electrons</a>.</p>" },
];
const WHERE = { user: ["CPU, user space", "c1"], kernel: ["CPU, OS kernel", "c5"], gpu: ["GPU", "c3"] };

export default function (root, { id }) {
  // ---------- 1. one call, down the layers ----------
  const ly = { i: 0 };
  const lc = canvas(880, 470, "The layers one PyTorch call passes through, from Python to the SM");
  const lst = stats([{ label: "Runs on" }, { label: "Written by" }, { label: "Does work" }, { label: "Time per call" }]);
  const ltext = h("div.zoom-what");
  const lsl = slider({ label: "Layer", min: 0, max: LAYERS.length - 1, value: 0, fmt: (v) => `${v + 1}. ${LAYERS[v].name}`, onInput: (v) => goL(v) });
  const ldown = button("Down a layer ↓", () => goL(ly.i + 1), "primary"), lup = button("↑ Up", () => goL(ly.i - 1));
  function goL(i) {
    ly.i = Math.max(0, Math.min(LAYERS.length - 1, i)); lsl.value = ly.i;
    ldown.disabled = ly.i === LAYERS.length - 1; lup.disabled = ly.i === 0;
    const L = LAYERS[ly.i];
    lst.set(0, WHERE[L.where][0]); lst.set(1, L.who); lst.set(2, L.when); lst.set(3, L.time);
    ltext.innerHTML = `<h4>${ly.i + 1}. ${L.name}</h4>${L.html}`;
    lc.paint(() => {
      lc.clear(); const c = lc.colors, x0 = 150, w = 330, rh = 44, y0 = 14;
      LAYERS.forEach((l, k) => {
        const y = y0 + k * (rh + 6), on = k === ly.i, past = k < ly.i, col = WHERE[l.where][1];
        lc.box(x0, y, w, rh, { fill: on ? `color-mix(in srgb, ${c[col]} 42%, ${c.surface})` : `color-mix(in srgb, ${c[col]} ${past ? 18 : 10}%, ${c.surface})`, stroke: on ? c.ink : c.line, lw: on ? 2.5 : 1.2, r: 10 });
        lc.text(`${k + 1}. ${l.name}`, x0 + 14, y + rh / 2, { size: 13, color: on || past ? c.ink : c["ink-2"] });
        if (on) {
          lc.text("gets:", x0 + w + 24, y + 12, { size: 10.5, color: c["ink-3"] });
          lc.text(l.gets, x0 + w + 62, y + 12, { size: 11.5, mono: true });
          lc.text("hands down:", x0 + w + 24, y + 32, { size: 10.5, color: c["ink-3"] });
          lc.text(l.hands, x0 + w + 104, y + 32, { size: 11.5, mono: true, color: c.accent });
          lc.dot(x0 - 18, y + rh / 2, 8, c.accent);
          if (k < LAYERS.length - 1) lc.line(x0 + w / 2, y + rh, x0 + w / 2, y + rh + 6, { color: c.accent, lw: 3 });
        }
      });
      // where each layer runs, with the two boundaries a call must cross
      const yOf = (k) => y0 + k * (rh + 6);
      const brk = (k0, k1, label, col) => { const a = yOf(k0), b = yOf(k1) + rh; lc.line(x0 - 40, a, x0 - 40, b, { color: c[col], lw: 4 }); lc.text(label, x0 - 50, (a + b) / 2, { size: 11, align: "right", color: c["ink-2"] }); };
      brk(0, 4, "CPU: your process", "c1"); brk(5, 5, "CPU: OS kernel", "c5"); brk(6, 8, "GPU", "c3");
      lc.line(x0 - 10, yOf(6) - 3, x0 + w + 380, yOf(6) - 3, { color: c.warn, lw: 2, dash: [6, 5] });
      lc.text("PCIe: the doorbell write and the command fetch cross here", x0 + w + 24, yOf(6) - 12, { size: 11, color: c.warn });
      lc.text("set up once; not on the per-launch path", x0 + w + 24, yOf(5) + rh / 2, { size: 11, color: c["ink-3"] });
    });
  }

  // ---------- 2. launch overhead ----------
  const Gt = (v) => 0.5e-6 * 2 ** (v / 1.5);
  const lo = { g: Gt(4), n: 400, graph: false };
  const EAGER = 8e-6, GRAPH_CPU = 10e-6, GRAPH_GAP = 1e-6;
  function sim({ g, n, graph }, upto = n) {
    const cpu = [], gpu = []; let end = 0;
    for (let k = 0; k < upto; k++) {
      const ready = graph ? GRAPH_CPU : (k + 1) * EAGER;
      if (!graph) cpu.push([k * EAGER, (k + 1) * EAGER]); else if (k === 0) cpu.push([0, GRAPH_CPU]);
      const start = Math.max(ready, end + (graph && k ? GRAPH_GAP : 0)); end = start + g; gpu.push([start, end]);
    }
    return { cpu, gpu, end };
  }
  const ost = stats([{ label: "One step (all kernels)" }, { label: "GPU busy" }, { label: "Bound by" }]);
  const oc = canvas(880, 190, "Timeline: the CPU launching kernels and the GPU running them");
  function drawO() {
    const full = sim(lo), busy = (lo.n * lo.g) / full.end;
    ost.set(0, us(full.end)); ost.set(1, `${Math.round(busy * 100)}%`, busy > 0.8 ? "good" : busy > 0.4 ? "warn" : "bad");
    ost.set(2, lo.graph ? (lo.g > GRAPH_GAP * 3 ? "the GPU's work" : "the gaps between kernels") : lo.g >= EAGER ? "the GPU's work" : "launching (the CPU)", busy > 0.8 ? "good" : "warn");
    const show = sim(lo, Math.min(lo.n, 12));
    oc.paint(() => {
      oc.clear(); const c = oc.colors, x0 = 120, x1 = 860, T = Math.max(show.end, 1e-9) * 1.04, X = (t) => x0 + (t / T) * (x1 - x0);
      oc.text("CPU", 20, 50, { size: 12 }); oc.text("GPU", 20, 120, { size: 12 });
      oc.line(x0, 82, x1, 82, { color: c.line, lw: 1 });
      show.cpu.forEach(([a, b], k) => oc.box(X(a) + 1, 34, Math.max(2, X(b) - X(a) - 2), 32, { fill: `color-mix(in srgb, ${c.c1} ${k % 2 ? 45 : 60}%, ${c.surface})`, stroke: "transparent", r: 5 }));
      show.gpu.forEach(([a, b], k) => oc.box(X(a) + 0.5, 104, Math.max(2, X(b) - X(a) - 1), 32, { fill: `color-mix(in srgb, ${c.c3} ${k % 2 ? 50 : 70}%, ${c.surface})`, stroke: "transparent", r: 5 }));
      oc.text(lo.graph ? "one graph launch replays every kernel" : `Python + PyTorch + launch: ~${EAGER * 1e6} µs per kernel`, x0, 22, { size: 11, color: c["ink-2"] });
      oc.text(`each kernel: ${us(lo.g)} of GPU work · first ${Math.min(lo.n, 12)} kernels shown · gaps = GPU idle`, x0, 154, { size: 11, color: c["ink-2"] });
      oc.text(`0`, x0, 176, { size: 10, color: c["ink-3"], mono: true }); oc.text(us(T), x1, 176, { size: 10, align: "right", color: c["ink-3"], mono: true });
    });
  }

  // ---------- 3. compile pipeline ----------
  const cp = { gpu: "sm_90", ptx: true };
  const GPUS = { sm_90: "H100 (sm_90)", sm_100: "B200 (sm_100)", sm_75: "T4 (sm_75)" };
  const cpc = canvas(880, 330, "How nvcc turns one .cu file into a program, and what the driver does at run time");
  const cpOut = h("p.cp-out");
  function drawCP() {
    const match = cp.gpu === "sm_90", newer = cp.gpu === "sm_100";
    const path = match ? "sass" : newer && cp.ptx ? "jit" : "fail";
    cpOut.innerHTML = path === "sass" ? "✅ <b>Runs at once.</b> The executable holds SASS built for exactly this GPU, so the driver just loads it."
      : path === "jit" ? "✅ <b>Runs, after a pause.</b> No SASS for sm_100, but the PTX is there: the driver compiles it for the new GPU at the first launch (seconds for big libraries) and caches the result. This is how a 2022 program runs on a 2025 GPU."
      : newer ? "❌ <b>“no kernel image is available for execution on the device”.</b> SASS is specific to a GPU generation, and without PTX there's nothing the driver can recompile. Ship PTX for forward compatibility."
      : "❌ <b>“no kernel image is available for execution on the device”.</b> The T4 is older than the code's target. SASS for sm_90 can't run on sm_75, and PTX can only be compiled for its own generation or newer. Add <code>-gencode arch=compute_75,code=sm_75</code>.";
    cpc.paint(() => {
      cpc.clear(); const c = cpc.colors;
      const B = (x, y, w, t, sub, k, on = true) => { cpc.box(x, y, w, 46, { fill: on ? `color-mix(in srgb, ${c[k]} 30%, ${c.surface})` : c.surface, stroke: on ? c[k] : c.line, lw: on ? 2 : 1, r: 9 }); cpc.text(t, x + w / 2, y + 17, { size: 12, align: "center" }); cpc.text(sub, x + w / 2, y + 33, { size: 10, align: "center", color: c["ink-3"], mono: true }); };
      const A = (x1, y1, x2, y2, on = true) => cpc.line(x1, y1, x2, y2, { color: on ? c.ink : c.line, lw: on ? 2 : 1.2, dash: on ? null : [4, 4] });
      cpc.text("COMPILE TIME (once, on your machine)", 20, 12, { size: 10, color: c["ink-3"] });
      B(20, 110, 110, "add.cu", "CUDA C++", "c4");
      B(170, 40, 140, "host code", "g++ / MSVC", "c1"); B(170, 180, 140, "device code", "NVVM (LLVM)", "c3");
      B(350, 180, 120, "PTX", "compute_90", "c3", cp.ptx); B(510, 180, 120, "SASS", "sm_90 cubin", "c3");
      A(130, 125, 170, 63); A(130, 145, 170, 203); A(310, 203, 350, 203); A(470, 203, 510, 203);
      B(670, 110, 190, "your executable", "fatbin: SASS + PTX?", "c2");
      A(310, 63, 670, 125); A(630, 203, 670, 145); A(410, 226, 410, 244, cp.ptx); A(410, 244, 765, 244, cp.ptx); A(765, 244, 765, 156, cp.ptx); cpc.text(cp.ptx ? "PTX embedded too" : "PTX not embedded", 590, 236, { size: 10, align: "center", color: cp.ptx ? c.ink : c["ink-3"] });
      cpc.line(20, 258, 860, 258, { color: c.line, lw: 1, dash: [3, 4] });
      cpc.text(`RUN TIME on ${GPUS[cp.gpu]}: the CUDA driver picks`, 20, 276, { size: 10, color: c["ink-3"] });
      const r = (x, t, on, k) => { cpc.box(x, 288, 260, 34, { fill: on ? `color-mix(in srgb, ${c[k]} 30%, ${c.surface})` : c.surface, stroke: on ? c[k] : c.line, lw: on ? 2.5 : 1, r: 9 }); cpc.text(t, x + 130, 305, { size: 12, align: "center", color: on ? c.ink : c["ink-3"] }); };
      r(20, "matching SASS → load it", path === "sass", "good"); r(310, "PTX → JIT-compile, cache", path === "jit", "warn"); r(600, "neither → error", path === "fail", "bad");
    });
  }

  // ---------- 4. fusion ----------
  const fu = { k: 4, n: 7 };
  const fst = stats([{ label: "Separate kernels" }, { label: "One fused kernel" }, { label: "Speed-up" }, { label: "HBM traffic saved" }]);
  const fch = xyChart({ width: 880, height: 240, xLabel: "elementwise operations in a row (×2, +1, ReLU, …)", yLabel: "time on an H100 (µs)", xDomain: [1, 8], yDomain: [0, 100], fmtX: (v) => String(Math.round(v)) });
  const FIXED = 2e-6, BW = 3.35e12;
  const tUn = (k, N) => k * (FIXED + (4 * N) / BW), tFu = (k, N) => FIXED + (4 * N) / BW;
  function drawF() {
    const N = 10 ** fu.n, a = tUn(fu.k, N), b = tFu(fu.k, N);
    fst.set(0, us(a)); fst.set(1, us(b), "good"); fst.set(2, `${(a / b).toFixed(1)}×`, a / b > 2 ? "good" : "warn");
    fst.set(3, `${(((fu.k - 1) * 4 * N) / 1e6).toFixed(N >= 1e6 ? 0 : 2)} MB`);
    const ks = [1, 2, 3, 4, 5, 6, 7, 8], top = tUn(8, N) * 1e6 * 1.1;
    fch.update([
      { name: "One kernel per operation (eager PyTorch)", color: "var(--c2)", points: ks.map((k) => [k, tUn(k, N) * 1e6]), dots: true },
      { name: "Fused into one kernel (torch.compile)", color: "var(--accent)", points: ks.map((k) => [k, tFu(k, N) * 1e6]), dots: true },
    ], { xDomain: [1, 8], yDomain: [0, top], marks: [{ x: fu.k, label: `${fu.k} ops` }] });
  }

  // ---------- torch.compile, drawn stage by stage ----------
  let tstep = 0;
  const tcv = flowCanvas({ width: 760, height: 270, label: "torch.compile on a small function: the graph Dynamo captures, the backward, the fused kernels, Triton's lowering and autotuning" });
  tcv.paint = (fn) => { tcv.redraw = fn; fn(); };
  tcv.paint(() => {
    tcv.clear(); const c = tcv.colors, s = tstep, ac = c.accent, T = tcv;
    const node = (x, y, label, col, { w = 74, dash = false, cur = false } = {}) => {
      T.box(x - w / 2, y - 14, w, 28, { fill: `color-mix(in srgb, ${col} ${dash ? 14 : 30}%, ${c.surface})`, stroke: cur ? ac : dash ? c["ink-3"] : "transparent", lw: cur ? 2 : 1, r: 6 });
      T.text(label, x, y, { size: 10.5, align: "center", mono: true });
    };
    const arrow = (x1, y1, x2, y2, col = c["ink-3"], dash = null) => { T.line(x1, y1, x2, y2, { color: col, dash }); const a = Math.atan2(y2 - y1, x2 - x1); T.ctx.beginPath(); T.ctx.moveTo(x2, y2); T.ctx.lineTo(x2 - 7 * Math.cos(a - 0.4), y2 - 7 * Math.sin(a - 0.4)); T.ctx.lineTo(x2 - 7 * Math.cos(a + 0.4), y2 - 7 * Math.sin(a + 0.4)); T.ctx.closePath(); T.ctx.fillStyle = col; T.ctx.fill(); };
    // the Python, always shown
    T.box(10, 12, 220, 120, { fill: c.sunken, stroke: c.line, r: 8 });
    ["@torch.compile", "def f(x, w, b):", "    y = x @ w + b", "    y = F.gelu(y)", "    return y.sum()"].forEach((l, k) => T.text(l, 20, 30 + k * 20, { size: 10.5, mono: true, color: k === 0 ? c.accent : c.ink }));
    const stage = ["Dynamo reads the bytecode, records a graph, adds guards", "AOTAutograd traces the backward ahead of time", "Inductor fuses: 2 forward kernels instead of 4", "Triton lowers the fused kernel through MLIR to PTX", "Autotune picks the fastest block size; CUDA Graphs cut the launches"][s];
    T.text(`${s + 1}. ${stage}`, 380, 256, { size: 11.5, align: "center" });
    if (s <= 2) {
      const Y = 54, X = [266, 364, 462, 560, 658, 734], col = c.c3;
      node(X[0], Y, "x, w, b", c.c1, { w: 64 });
      node(X[1], Y, "matmul", col, { cur: s === 0 }); node(X[2], Y, "add", col, { w: 56, cur: s === 0 }); node(X[3], Y, "gelu", col, { w: 56, cur: s === 0 }); node(X[4], Y, "sum", col, { w: 56, cur: s === 0 });
      for (let k = 0; k < 4; k++) arrow(X[k] + (k ? 30 : 34), Y, X[k + 1] - (k === 0 ? 39 : 30), Y);
      T.text("FX graph", 266, 26, { size: 9.5, color: c["ink-3"], mono: true });
      if (s === 0) {   // guards
        T.text("guards: recompile if any of these change", 20, 160, { size: 10, color: c["ink-3"], mono: true });
        ["x.shape == (64, 512)", "x.dtype == bfloat16", "x.device == cuda:0"].forEach((g, k) => { T.box(20 + k * 176, 172, 168, 24, { fill: `color-mix(in srgb, ${c.c4} 25%, ${c.surface})`, stroke: c.line, r: 12 }); T.text(g, 104 + k * 176, 184, { size: 10, align: "center", mono: true }); });
        T.text("anything not a tensor op (print, .item(), data-dependent if) ends the graph: a graph break", 20, 222, { size: 10, color: c["ink-3"] });
      }
      if (s === 1) {   // backward graph
        const B = 122;
        node(X[4], B, "sum'", c.c6, { w: 56, dash: true, cur: s === 1 }); node(X[3], B, "gelu'", c.c6, { w: 56, dash: true, cur: s === 1 }); node(X[2], B, "add'", c.c6, { w: 56, dash: true, cur: s === 1 }); node(X[1], B, "matmul'", c.c6, { dash: true, cur: s === 1 });
        for (let k = 4; k > 1; k--) arrow(X[k] - 30, B, X[k - 1] + (k - 1 === 1 ? 39 : 30), B, c["ink-3"], [3, 3]);
        T.text("backward, traced ahead of time", X[1] - 37, B + 28, { size: 9.5, color: c["ink-3"], mono: true });
        T.line(X[2], Y + 14, X[3], B - 14, { color: c["ink-3"], dash: [2, 3] }); T.text("save y, or recompute it?", X[2] + 2, 92, { size: 9.5, color: c["ink-3"] });
      }
      if (s === 2) {   // fusion groups
        T.box(X[1] - 44, Y - 24, 88, 48, { fill: "transparent", stroke: c.c2, lw: 2, r: 9 });
        T.box(X[2] - 36, Y - 24, X[4] - X[2] + 72, 48, { fill: "transparent", stroke: ac, lw: 2.5, r: 9 });
        T.text("kernel 1: cuBLAS or a template", X[1], 180, { size: 10.5, align: "center", color: c.c2 });
        T.line(X[1], Y + 24, X[1], 168, { color: c.c2, dash: [3, 3] });
        T.text("kernel 2: one Triton kernel, y read once", X[3], 180, { size: 10.5, align: "center", color: ac });
        T.line(X[3], 136, X[3], 168, { color: ac, dash: [3, 3] });
        T.text("unfused: 4 kernels, y written and read 3 times  →  fused: 2 kernels", 490, 212, { size: 10.5, align: "center", color: c["ink-2"] });
      }
    } else if (s === 3) {
      T.box(250, 30, 140, 44, { fill: `color-mix(in srgb, ${ac} 14%, ${c.surface})`, stroke: ac, lw: 2, r: 8 }); T.text("add + gelu + sum", 320, 52, { size: 10.5, align: "center", mono: true });
      const L = [["Triton IR", "blocks of values, no threads"], ["TritonGPU (MLIR)", "layouts and shared memory"], ["LLVM IR", "standard optimisations"], ["PTX", "virtual GPU assembly"], ["cubin (SASS)", "ptxas: machine code"]];
      L.forEach(([a, b], k) => { const y = 32 + k * 42; T.box(420, y, 170, 30, { fill: `color-mix(in srgb, ${[c.c1, c.c6, c.c3, c.c4, c.c2][k]} 28%, ${c.surface})`, stroke: "transparent", r: 6 }); T.text(a, 505, y + 15, { size: 10.5, align: "center", mono: true }); T.text(b, 600, y + 15, { size: 10, color: c["ink-3"] }); if (k) arrow(505, y - 12, 505, y - 2); });
      arrow(390, 52, 418, 47);
    } else {
      const C = [["BLOCK 64, 4 warps", 41], ["BLOCK 128, 4 warps", 29], ["BLOCK 128, 8 warps", 31], ["BLOCK 256, 8 warps", 34]], mx = 45;
      T.text("max-autotune: time each config on this GPU", 250, 26, { size: 10, color: c["ink-3"], mono: true });
      C.forEach(([n, t], k) => { const y = 42 + k * 30, best = t === 29; T.text(n, 250, y + 10, { size: 10.5, mono: true }); T.bar(400, y + 4, 200, 12, t / mx, best ? c.good : c.c1); T.text(`${t} µs${best ? "  ★ kept" : ""}`, 610, y + 10, { size: 10.5, color: best ? c.good : c["ink-2"], mono: true }); });
      T.text("reduce-overhead: record the launches once as a CUDA Graph, then replay with one call", 20, 186, { size: 10.5, color: c.ink });
      for (let k = 0; k < 8; k++) T.box(20 + k * 30, 202, 24, 18, { fill: `color-mix(in srgb, ${c.c3} 40%, ${c.surface})`, stroke: "transparent", r: 3 });
      T.text("→ one launch", 270, 211, { size: 10.5, color: c.good }); T.text("illustrative timings", 740, 26, { size: 9.5, color: c["ink-3"], align: "right" });
    }
  });

  root.append(
    section(null,
      goal("<p>You type <code>y = x @ W</code>. The GPU can only run its own machine code, started by writing bytes into a queue and ringing a doorbell. Between the two sit about nine layers of software, written by different people at different times. The goal is to see <b>why each layer exists</b>: each one lets somebody change one thing (the model, the library, the GPU) without rewriting everything else, and each one costs something.</p>",
        ["Write models in a few lines of Python", "Run near the hardware's peak on GPUs that don't exist yet", "Keep programs from crashing or spying on each other"]),
      stackMap(id),
      h("div.vx-row",
        sticker("softstack", { title: "The stack of layers", caption: "Python on top, then framework, library, runtime, driver, machine code, silicon", px: 4 }),
        sticker("grid", { title: "What it launches", caption: "a grid of thread blocks, described once, scheduled by the GPU", px: 4 })),
    ),
    section("One call, down nine layers",
      prose("<p>Follow a single matrix multiply from Python to the SM. At each layer: what it receives, what it hands to the layer below, where it runs and how long it takes. Notice where the operating system <i>isn't</i>: it set things up once and then stepped out of the way.</p>"),
      h("div.stage",
        h("div.stage-controls", lsl, h("div.btn-row", lup, ldown)),
        h("div.stage-body", lst.el, h("div.wide-canvas", lc.el), ltext)),
      callout("tip", "The pattern", "<p>Every layer turns something convenient into something more specific: a line of Python into an operator, an operator into a chosen kernel, a kernel into launch bytes, bytes into thread blocks, blocks into warp instructions. The upper layers run on the CPU <b>while the GPU is still busy with earlier work</b>: launches are asynchronous, so the CPU races ahead and queues work up. That works until the work per launch gets too small.</p>"),
    ),
    section("When the layers cost more than the work",
      prose("<p>Each trip down the layers costs the CPU a few microseconds. If a kernel runs for longer than that, the CPU stays ahead and the GPU never waits. If kernels are tiny (small-batch inference is hundreds of small kernels per token) the GPU finishes each one and sits idle until the next arrives. A <b>CUDA Graph</b> records the whole sequence once and replays it with one launch.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "GPU time per kernel", min: 0, max: 12, value: 4, fmt: (v) => us(Gt(v)), onInput: (v) => { lo.g = Gt(v); drawO(); } }),
          slider({ label: "Kernels per step", min: 10, max: 1000, step: 10, value: lo.n, onInput: (v) => { lo.n = v; drawO(); } }),
          toggle({ label: "Capture as a CUDA Graph", checked: false, onChange: (v) => { lo.graph = v; drawO(); } })),
        h("div.stage-body", ost.el, h("div.wide-canvas", oc.el))),
      callout("note", "The numbers here", "<p>~8 µs per eager PyTorch op (Python, dispatcher and launch) and ~1 µs between kernels in a replayed graph are typical orders of magnitude, not measurements of your machine; they vary with CPU speed and the op. Notebook 60 measures launch overhead on a real GPU.</p>"),
      predict({
        chapter: id,
        question: "A decode step runs 600 kernels of 3 µs GPU time each, launched from eager PyTorch at ~8 µs per launch. Roughly how busy is the GPU?",
        options: ["~100%", "~85%", "~40%", "~5%"],
        answer: 2,
        explain: "<p>The CPU can only queue a kernel every 8 µs, and each one takes 3 µs, so the GPU works 3 µs out of every 8: about 37%. Making the GPU faster wouldn't help at all. Removing the launches would: CUDA Graphs, fusing kernels, or a bigger batch so each kernel does more work.</p>",
      }),
    ),
    section("From CUDA C++ to machine code",
      prose("<p>A <code>.cu</code> file holds two programs: code for the CPU and kernels for the GPU. <code>nvcc</code> splits them. The GPU part becomes <b>PTX</b>, a virtual instruction set that stays stable across generations, and then <b>SASS</b>, the real machine code of one generation. Both can be packed into your program. Pick the GPU it runs on.</p>"),
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Run it on", options: Object.entries(GPUS).map(([value, label]) => ({ value, label })), value: cp.gpu, onChange: (v) => { cp.gpu = v; drawCP(); } }),
          toggle({ label: "Also embed PTX", checked: true, onChange: (v) => { cp.ptx = v; drawCP(); } })),
        h("div.stage-body", h("div.wide-canvas", cpc.el), cpOut)),
      code({
        "1. CUDA C++": `__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}`,
        "2. PTX (virtual ISA)": `// nvcc -ptx add.cu   (abridged)
.visible .entry add(.param .u64 a, .param .u64 b, .param .u64 c, .param .u32 n)
{
    mov.u32        %r1, %ctaid.x;          // blockIdx.x
    mov.u32        %r2, %ntid.x;           // blockDim.x
    mov.u32        %r3, %tid.x;            // threadIdx.x
    mad.lo.s32     %r4, %r1, %r2, %r3;     // i = block * dim + thread
    setp.ge.s32    %p1, %r4, %r5;          // i >= n ?
    @%p1 bra       DONE;
    ld.global.f32  %f1, [%rd5];            // a[i]
    ld.global.f32  %f2, [%rd7];            // b[i]
    add.f32        %f3, %f1, %f2;
    st.global.f32  [%rd9], %f3;            // c[i]
DONE:
    ret;
}`,
        "3. SASS (sm_90)": `// cuobjdump -sass add.o   (abridged; exact output varies by compiler version)
S2R R0, SR_CTAID.X            // read blockIdx.x from a special register
S2R R3, SR_TID.X
IMAD R0, R0, c[0x0][0x0], R3  // i = blockIdx.x * blockDim.x + threadIdx.x
ISETP.GE.AND P0, PT, R0, c[0x0][0x228], PT
@P0 EXIT                      // threads past the end stop here
IMAD.WIDE R2, R0, 0x4, c[0x0][0x210]
IMAD.WIDE R4, R0, 0x4, c[0x0][0x218]
LDG.E R2, desc[UR4][R2.64]    // load a[i] from global memory
LDG.E R5, desc[UR4][R4.64]    // load b[i]
IMAD.WIDE R6, R0, 0x4, c[0x0][0x220]
FADD R9, R2, R5
STG.E desc[UR4][R6.64], R9    // store c[i]
EXIT`,
        "Build for several GPUs": `# SASS for the GPUs you test on, plus PTX for the ones that don't exist yet
nvcc add.cu -o add \\
  -gencode arch=compute_75,code=sm_75 \\
  -gencode arch=compute_90,code=sm_90 \\
  -gencode arch=compute_90,code=compute_90     # PTX: JIT-compiled on newer GPUs`,
      }, "PTX reads like a tidy assembly language with unlimited registers; SASS is what really runs, with real registers, scheduling and the hardware's own instruction names."),
      predict({
        chapter: id,
        question: "You built your kernels only as SASS for sm_90 (no PTX) and run them on a newer GPU, sm_100. What happens?",
        options: ["They run at full speed", "The driver JIT-compiles them first", "They fail: no kernel image is available", "They fall back to the CPU"],
        answer: 2,
        explain: "<p>SASS is tied to a GPU generation; there's nothing the driver can translate. With PTX embedded, the driver would have compiled it for sm_100 at first launch. That's why libraries ship PTX alongside SASS, and why PTX exists at all: it's the stable contract between compilers and GPUs that keeps changing.</p>",
      }),
    ),
    section("What a compiler buys you: fusion",
      prose("<p>Eager PyTorch runs one kernel per operation, and each reads its input from HBM and writes its output back. A chain like <code>relu(x * 2 + 1) * 0.5</code> moves the whole tensor through HBM 4 times each way. A compiler that sees the whole chain can <b>fuse</b> it into one kernel: read once, do everything in registers, write once.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Operations in the chain", min: 1, max: 8, value: fu.k, onInput: (v) => { fu.k = v; drawF(); } }),
          slider({ label: "Tensor size (bf16 elements)", min: 4, max: 9, value: fu.n, fmt: (v) => `10^${v}`, onInput: (v) => { fu.n = v; drawF(); } })),
        h("div.stage-body", fst.el, fch.el)),
      prose("<p>Step through it on a tiny tensor, in 3D: three separate kernels, then one fused kernel.</p>"),
      walk("software-stack"),
    ),
    section("torch.compile, layer by layer",
      prose("<p><code>torch.compile(model)</code> is a whole compiler stack hidden behind one line. Each stage turns a program into a more specific one, the same pattern as the layers above.</p>"),
      h("div.wide-canvas", tcv.el),
      stepper([
        { title: "1. Dynamo: capture the Python", html: "<p>Dynamo hooks CPython's frame evaluation and reads your function's bytecode as it is about to run. It records the tensor operations into an <b>FX graph</b> and installs <b>guards</b>: checks such as “x is bf16 on CUDA with shape 4096 × 4096”. Next call, if the guards pass, it runs the compiled version; if not, it recompiles. Python it can't follow (printing, data-dependent branches) causes a <b>graph break</b> and runs normally.</p><pre>graph():\n  %mul = call_function[target=torch.mul](args=(%x, 2))\n  %add = call_function[target=torch.add](args=(%mul, 1))\n  %relu = call_function[target=torch.relu](args=(%add,))</pre>" },
        { title: "2. AOTAutograd: the backward too", html: "<p>For training, AOTAutograd traces the backward pass ahead of time, so forward and backward become two graphs the compiler can optimise together, for example by deciding which intermediate values to save and which to recompute.</p>" },
        { title: "3. Inductor: fuse and schedule", html: "<p>TorchInductor lowers the graph to simple loops over tensors, decides which operations to fuse into the same kernel, plans memory, and picks matmuls from libraries or templates. For GPUs it writes <b>Triton</b> code; for CPUs, C++ with OpenMP.</p><pre>@triton.jit\ndef fused_mul_add_relu(x_ptr, out_ptr, n, BLOCK: tl.constexpr):\n    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)\n    x = tl.load(x_ptr + i, mask=i &lt; n)\n    tl.store(out_ptr + i, tl.maximum(x * 2 + 1, 0), mask=i &lt; n)</pre>" },
        { title: "4. Triton: down to PTX", html: "<p>Triton is itself a compiler. It turns that Python-looking kernel into Triton IR, then a GPU-aware dialect in <b>MLIR</b> that decides layouts, vector widths and shared-memory use, then LLVM IR, then PTX, then <code>ptxas</code> makes SASS. You wrote block-level code; Triton chose the per-thread work.</p>" },
        { title: "5. Autotune, then (optionally) CUDA Graphs", html: "<p>For each kernel Inductor can try several block sizes and keep the fastest on your GPU (<code>mode=\"max-autotune\"</code>). <code>mode=\"reduce-overhead\"</code> wraps the result in CUDA Graphs, removing the launch cost you saw above. The PyTorch 2 paper reported 2.27× faster inference and 1.41× faster training (geometric mean) across 180+ models on an A100.</p>" },
      ], { onStep: (k) => { tstep = k; tcv.redraw?.(); } }),
    ),
    section("Dig down: why can't the compiler do it all?",
      whyChain({
        claim: "Experts still hand-write the most important GPU kernels (matmuls, attention), even though compilers like torch.compile and Triton exist.",
        steps: [
          { q: "Why can't the compiler just produce the best kernel?", a: "<p>The best kernel depends on things it often doesn't know when it compiles: the exact shapes, the data, the GPU, what else runs at the same time. FlashAttention's speed-up came from an idea (never write the attention matrix to HBM) that no compiler of the day would find from the plain maths.</p>" },
          { q: "Why not try every option and keep the fastest?", a: "<p>The space is enormous: tile sizes, loop orders, what to fuse, what to keep in registers versus shared memory, how to pipeline loads. Many of these choices are NP-hard to optimise exactly (register allocation, optimal fusion). Autotuners sample a small slice of the space; humans narrow it with insight.</p>" },
          { q: "Why must a compiler be so careful?", a: "<p>It must keep the program's meaning for <b>every possible input</b>. If two pointers might point at the same memory, it can't reorder their loads and stores. If a loop might run zero times, it can't hoist work out of it. Unless it can <i>prove</i> a transformation is safe, it must not do it.</p>" },
          { q: "Why can't it just prove what the program does?", a: "<p>Because in general, no algorithm can. Whether a program ever reads a given address, or halts, or computes the same result as another program, can't be decided for all programs. Compilers succeed by restricting what you may write: static shapes, no aliasing, pure tensor operations. That's exactly what tensor languages and Triton impose, and why they compile so well.</p>" },
        ],
        axiom: { name: "Rice's theorem: no algorithm can decide every non-trivial question about what programs do", html: "<p>Rice (1953), building on Turing's halting problem (1936): for any property of a program's behaviour that some programs have and others don't, there is no general procedure that decides it for every program. So every compiler is a conservative approximation, and the last stretch of speed comes from people (or search) who know more about the specific case than any compiler can prove.</p>" },
      }),
    ),
    section("Every layer, side by side",
      table(["Layer", "You'd write", "It hides from you", "Escape hatch when you need speed"], [
        ["Python + PyTorch", "<code>y = x @ W</code>", "devices, dtypes, gradients, memory", "<code>torch.compile</code>, custom ops"],
        ["Compiler (Inductor, XLA)", "nothing: it's automatic", "fusion, scheduling, kernel code", "<code>max-autotune</code>, graph-break fixes"],
        ["Kernel language (Triton)", "block-level kernels in Python", "threads, shared memory layout, PTX", "CUDA C++ for full control"],
        ["Libraries (cuBLAS, cuDNN, CUTLASS)", "a function call", "hundreds of tuned kernels per GPU", "CUTLASS / CuTe templates"],
        ["CUDA C++", "threads, blocks, shared memory", "SASS, register allocation, scheduling", "inline PTX"],
        ["PTX", "virtual assembly", "the real instruction set of each GPU", "nothing public below this"],
        ["Driver + hardware", "nothing", "command queues, page tables, block scheduling", "Nsight tools to observe it"],
      ], { boldFirst: true }),
      callout("note", "What “abstraction” costs and buys", "<p>Each layer lets one group change without breaking the others: model authors don't track GPU generations, library authors don't track your model, and NVIDIA can change SASS every generation because nobody writes it. The price is overhead and lost information at each boundary. Speed work is mostly finding which boundary is costing you, then reaching one layer down.</p>"),
    ),
    section("In code",
      code({
        "See every layer's cost": `import torch
from torch.profiler import profile, ProfilerActivity
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16); W = torch.randn_like(x)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(10): y = x @ W
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=8))
# CPU rows: aten::matmul -> aten::mm -> cudaLaunchKernel.  CUDA rows: the cuBLAS kernel by name.`,
        "See what torch.compile writes": `# TORCH_LOGS=output_code python this.py   prints the generated Triton kernels
import torch
@torch.compile
def f(x): return torch.relu(x * 2 + 1) * 0.5
x = torch.randn(1 << 24, device="cuda")
f(x)                       # first call compiles; look for one fused kernel in the log`,
        "Launch overhead and CUDA Graphs": `import torch, time
x = torch.randn(1024, device="cuda")
def step(x):
    for _ in range(100): x = x * 1.0001
    return x
def timeit(fn, reps=50):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(reps): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / reps
print("eager:", timeit(lambda: step(x)) * 1e6, "µs per 100 kernels")
g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream()
with torch.cuda.stream(s): step(x)                     # warm up on a side stream
torch.cuda.current_stream().wait_stream(s)
with torch.cuda.graph(g): y = step(x)
print("graph:", timeit(g.replay) * 1e6, "µs per 100 kernels")`,
        "PTX and SASS yourself": `nvcc -ptx add.cu -o add.ptx                 # the virtual ISA
nvcc -cubin -arch=sm_90 add.cu -o add.cubin   # machine code for one GPU
cuobjdump -sass add.cubin                     # disassemble it
# or for any PyTorch extension: cuobjdump -sass $(python -c "import torch, os; print(os.path.join(torch.__path__[0], 'lib', 'libtorch_cuda.so'))") | head`,
      }),
    ),
  );
  goL(0); drawO(); drawCP(); drawF();
}
