import { h } from "../lib/core.js";
import { slider, stats, predict, code, callout, section, prose, table, segmented, stepper, toggle } from "../lib/widgets.js";
import { flowCanvas } from "../lib/flow.js";
import { goal, whyChain, stackMap, sticker } from "../lib/why.js";
import { walk } from "../lib/walks.js";

function canvas(width, height, label) {
  const fc = flowCanvas({ width, height, label });
  fc.paint = (fn) => { fc.redraw = fn; fn(); };
  return fc;
}
const sup = (e) => String(e).split("").map((d) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[+d]).join("");
const sciS = (v) => { const x = 10 ** v, e = Math.floor(v + 1e-9), m = x / 10 ** e; return `${m < 1.05 ? "" : `${m.toFixed(1)} × `}10${sup(e)}`; };
const fmtT = (s) => (s >= 1 ? `${s.toFixed(2)} s` : s >= 1e-3 ? `${(s * 1e3).toFixed(2)} ms` : `${(s * 1e6).toFixed(1)} µs`);

export default function (root, { id }) {
  // ---------- 1. grid, blocks, threads, SMs ----------
  const gm = { n: 300, bd: 64, sms: 4, perSM: 2 };
  const gc = canvas(880, 300, "An array split into blocks, and blocks scheduled onto SMs in waves");
  const gst = stats([{ label: "Blocks in the grid" }, { label: "Threads launched (idle ones)" }, { label: "Waves of blocks" }, { label: "SM slots used in the last wave" }]);
  const COLS = ["c1", "c2", "c3", "c4", "c5", "c6"];
  function drawG() {
    const blocks = Math.ceil(gm.n / gm.bd), slots = gm.sms * gm.perSM, waves = Math.ceil(blocks / slots), last = blocks - (waves - 1) * slots;
    gst.set(0, `${blocks} = ⌈${gm.n} ÷ ${gm.bd}⌉`); gst.set(1, `${blocks * gm.bd} (${blocks * gm.bd - gm.n} idle)`, blocks * gm.bd - gm.n > gm.bd / 2 ? "warn" : "good");
    gst.set(2, String(waves)); gst.set(3, `${last} of ${slots} (${Math.round((last / slots) * 100)}%)`, last / slots > 0.75 ? "good" : last / slots > 0.4 ? "warn" : "bad");
    gc.paint(() => {
      gc.clear(); const c = gc.colors, total = blocks * gm.bd, per = 64, cw = 800 / per, ch = 9;
      gc.text("THE ARRAY: one thread per element, coloured by block (grey = past the end)", 40, 12, { size: 10.5, color: c["ink-3"], mono: true });
      for (let t = 0; t < total; t++) {
        const r = Math.floor(t / per), q = t % per;
        if (r > 8) break;
        gc.box(40 + q * cw + 0.5, 24 + r * (ch + 2), cw - 1, ch, { fill: t < gm.n ? c[COLS[Math.floor(t / gm.bd) % COLS.length]] : c.line, stroke: "transparent", r: 1.5 });
      }
      if (total > 9 * per) gc.text(`… ${total - 9 * per} more threads`, 40, 24 + 9 * (ch + 2) + 6, { size: 10.5, color: c["ink-3"] });
      const y0 = 150;
      gc.text(`THE GPU: ${gm.sms} SMs × ${gm.perSM} resident blocks each. Blocks are handed out in waves, in no promised order.`, 40, y0 - 12, { size: 10.5, color: c["ink-3"], mono: true });
      const sw = Math.min(190, 800 / gm.sms - 10);
      for (let s = 0; s < gm.sms; s++) {
        const x = 40 + s * (sw + 10);
        gc.box(x, y0, sw, 120, { fill: c.sunken, stroke: c.line, r: 8 });
        gc.text(`SM ${s}`, x + 8, y0 + 12, { size: 11, color: c["ink-2"] });
        for (let k = 0; k < gm.perSM; k++) {
          const lane = s * gm.perSM + k, mine = []; for (let b = lane; b < blocks; b += slots) mine.push(b);
          const bw = (sw - 16) / gm.perSM;
          mine.slice(0, 6).forEach((b, wv) => {
            gc.box(x + 8 + k * bw, y0 + 24 + wv * 15, bw - 4, 12, { fill: c[COLS[b % COLS.length]], stroke: "transparent", r: 3 });
            gc.text(`block ${b}`, x + 8 + k * bw + (bw - 4) / 2, y0 + 30 + wv * 15, { size: 9, color: "#fff", align: "center", mono: true });
          });
          if (mine.length > 6) gc.text(`+${mine.length - 6}`, x + 8 + k * bw + 4, y0 + 24 + 6 * 15 + 2, { size: 9.5, color: c["ink-3"] });
        }
      }
      gc.text("↓ wave 1, wave 2, … each slot takes its next block as soon as it's free", 40, 288, { size: 10.5, color: c["ink-3"] });
    });
  }

  // ---------- 2. is the GPU worth the trip? ----------
  const wt = { n: 7, flops: 10, resident: false };
  const wst = stats([{ label: "CPU (16 cores, ~1 TFLOP/s, 100 GB/s)" }, { label: "GPU total" }, { label: "…of which PCIe copies" }, { label: "…of which launch overhead" }, { label: "Winner" }]);
  function drawW() {
    const N = 10 ** wt.n, F = wt.flops, B = 12;                                      // two float inputs and one output per element
    const cpu = Math.max((N * F) / 1e12, (N * B) / 100e9);
    const copy = wt.resident ? 0 : (N * 8) / 25e9 + (N * 4) / 25e9, launch = 5e-6;
    const kern = Math.max((N * F) / 50e12, (N * B) / 3e12), gpu = launch + copy + kern;
    wst.set(0, fmtT(cpu)); wst.set(1, fmtT(gpu)); wst.set(2, fmtT(copy), copy > kern * 3 ? "bad" : "");
    wst.set(3, fmtT(launch), launch > kern ? "warn" : "");
    wst.set(4, gpu < cpu ? `GPU, ${(cpu / gpu).toFixed(1)}× faster` : `CPU, ${(gpu / cpu).toFixed(1)}× faster`, gpu < cpu ? "good" : "bad");
  }

  root.append(
    section(null,
      goal("<p>We have a machine with 132 SMs, each wanting 64 warps of 32 threads in flight. We need a way to <b>describe a million small, independent pieces of work</b> so the hardware can spread them over whatever SMs it has, without the programmer managing any of it. CUDA's answer: write the code for <b>one thread</b>, and launch it as a grid.</p>",
        ["Say what one piece of work is", "Map each piece to its data", "Run on any size of GPU, unchanged"]),
      stackMap(id),
      h("div.vx-row", sticker("chip", { title: "The device", caption: "the GPU, with its own memory: your data must be copied here first", px: 4 }), sticker("package", { title: "Its memory", caption: "HBM, 80 GB at 3.35 TB/s, but ~25–60 GB/s from the CPU over PCIe", px: 3 }), sticker("grid", { title: "Your launch", caption: "a grid of blocks, each a patch of threads: you describe it, the GPU schedules it", px: 4 })),
    ),
    section("Grid, blocks, threads",
      prose("<p>A kernel launch says: run this function in a <b>grid</b> of <b>blocks</b>, each with the same number of <b>threads</b>. Every thread gets its block number (<code>blockIdx</code>) and its number inside the block (<code>threadIdx</code>), and uses them to pick its piece of the data. Each block is placed whole on one SM; the hardware hands out blocks as SMs free up. Change the sizes.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Elements n", min: 1, max: 600, value: gm.n, onInput: (v) => { gm.n = v; drawG(); } }),
          segmented({ label: "Threads per block", options: [32, 64, 128, 256].map((v) => ({ value: v, label: String(v) })), value: gm.bd, onChange: (v) => { gm.bd = v; drawG(); } }),
          slider({ label: "SMs on this GPU", min: 1, max: 8, value: gm.sms, onInput: (v) => { gm.sms = v; drawG(); } }),
          slider({ label: "Blocks resident per SM", min: 1, max: 4, value: gm.perSM, onInput: (v) => { gm.perSM = v; drawG(); } })),
        h("div.stage-body", gst.el, h("div.wide-canvas", gc.el))),
      callout("tip", "The tail effect", "<p>Watch the last-wave number. If the final wave only fills a few SM slots, the rest of the GPU idles while it finishes. With 132 SMs this is real: a grid of 140 blocks that each take the same time runs in two waves, the second one 6% full. Kernel authors size grids as multiples of the SM count, or use “persistent” kernels that loop over work.</p>"),
      predict({
        chapter: id,
        question: "You add two arrays of n = 1,000,000 floats with 256 threads per block. How many blocks do you launch, and how many threads in the last block do useful work?",
        options: ["3,906 blocks; all 256", "3,907 blocks; 64", "1,000,000 blocks; 1", "4,000 blocks; 256"],
        answer: 1,
        explain: "<p>⌈1,000,000 ÷ 256⌉ = 3,907. The first 3,906 blocks cover 999,936 elements, leaving 64 for the last block; its other 192 threads fail the <code>if (i &lt; n)</code> check and exit. Those 3,907 blocks then run in waves across the SMs.</p>",
      }),
    ),
    walk("cuda"),
    section("Why blocks can't wait for each other",
      whyChain({
        claim: "Threads in a block can share memory and synchronise; separate blocks can't (inside one kernel).",
        steps: [
          { q: "Why can threads in a block cooperate?", a: "<p>A block runs entirely on one SM. Its threads share that SM's shared memory, physical SRAM a few hundred micrometres away, and the SM has a hardware barrier, <code>__syncthreads()</code>, that holds every warp of the block until all arrive.</p>" },
          { q: "Why not the same across blocks?", a: "<p>Different blocks may be on different SMs. The only memory they share is L2 and HBM, hundreds of cycles away (chapter 5). Worse, they may not even be running at the same time.</p>" },
          { q: "Why might they not run at the same time?", a: "<p>A grid can have millions of blocks; an H100 holds at most a few thousand at once (132 SMs × up to 32 blocks). The rest wait in a queue for a free slot. If block 0 waited for block 900,000, and block 900,000 couldn't start until block 0 finished and freed its slot, both would wait forever.</p>" },
          { q: "Why design it so blocks queue?", a: "<p>So the same compiled program runs on any GPU: a laptop chip with 16 SMs runs the 3,907 blocks in more waves, an H100 in fewer. The program never says how many processors exist, so it scales from 2006's G80 to today without changes. Where cross-block sync is truly needed, you end the kernel and launch another, or use a cooperative launch that guarantees all blocks are resident.</p>" },
        ],
        axiom: { name: "Circular waiting never ends (deadlock)", html: "<p>If A can't finish until B starts, and B can't start until A finishes, neither ever happens. That's logic, not engineering. Any system with more tasks than slots must forbid tasks from waiting on tasks that may not have started. CUDA's rule, “blocks are independent”, is exactly that prohibition.</p>" },
      }),
      table(["Memory space", "Declared as", "Who can see it", "Lives in", "Speed"], [
        ["Registers", "ordinary local variables", "one thread", "the SM's register file", "~1 cycle"],
        ["Shared memory", "<code>__shared__ float tile[32][33];</code>", "the threads of one block", "SRAM in the SM", "~30 cycles"],
        ["Global memory", "<code>cudaMalloc</code>, pointers you pass in", "every thread, every kernel, and the CPU via copies", "HBM (cached in L2 and L1)", "~600 cycles"],
        ["Local memory", "arrays the compiler can't keep in registers", "one thread", "HBM, cached", "slow: a sign to restructure"],
        ["Constant memory", "<code>__constant__</code>", "every thread, read-only", "HBM with a small broadcast cache", "fast when a warp reads one address"],
      ].map((r) => r.map((x) => (typeof x === "string" && x.includes("<code>") ? h("span", { html: x }) : x)))),
    ),
    section("The life of a CUDA program",
      stepper([
        { title: "1. Compile: two programs in one file", html: "<p><code>nvcc</code> splits your <code>.cu</code> file: ordinary C++ for the CPU (the <b>host</b>), and kernels marked <code>__global__</code> for the GPU (the <b>device</b>). Kernels compile to <b>PTX</b>, a virtual instruction set, and then to <b>SASS</b>, the machine code of a specific GPU generation. Triton and <code>torch.compile</code> generate the same PTX by other routes.</p>" },
        { title: "2. Allocate and copy in", html: "<p>The GPU has its own memory. <code>cudaMalloc</code> reserves space in HBM, and <code>cudaMemcpy</code> copies your arrays across <b>PCIe</b>, at roughly 25 GB/s (PCIe 4.0 ×16) to 50 GB/s (5.0): about 100× slower than HBM itself. This is why PyTorch keeps tensors on the GPU between operations.</p>" },
        { title: "3. Launch: <<<blocks, threads>>>", html: "<p><code>add&lt;&lt;&lt;3907, 256&gt;&gt;&gt;(a, b, c, n)</code> queues the kernel and returns immediately: launches are <b>asynchronous</b>. Each launch costs a few microseconds of CPU and driver work, which is why thousands of tiny kernels are slow and why CUDA Graphs and kernel fusion exist.</p>" },
        { title: "4. The GPU schedules it", html: "<p>A hardware scheduler gives blocks to SMs with room for them. Each SM splits its blocks into warps of 32 and its warp schedulers issue an instruction from a ready warp every cycle, hiding memory latency behind other warps (chapter 7). You never see this, but every performance rule comes from it.</p>" },
        { title: "5. Copy out, and synchronise", html: "<p><code>cudaMemcpy</code> back to the host waits for the kernel to finish first. Errors inside a kernel (like an out-of-bounds write) only surface at the next synchronising call, which is why CUDA code checks <code>cudaGetLastError()</code> and why debugging uses <code>CUDA_LAUNCH_BLOCKING=1</code>.</p>" },
      ]),
    ),
    section("Is the GPU worth the trip?",
      prose("<p>A GPU only wins if the work outweighs getting the data there and starting the kernel. Here, an elementwise operation reads two floats and writes one per element, with a chosen number of operations per element. Compare a good CPU with an H100 (practical rates, not peaks).</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Elements", min: 2, max: 9, step: 0.5, value: wt.n, fmt: sciS, onInput: (v) => { wt.n = v; drawW(); } }),
          slider({ label: "Operations per element", min: 1, max: 2000, value: wt.flops, onInput: (v) => { wt.flops = v; drawW(); } }),
          toggle({ label: "Data already on the GPU", checked: wt.resident, onChange: (v) => { wt.resident = v; drawW(); } })),
        h("div.stage-body", wst.el)),
      callout("note", "Three lessons in one widget", "<p><b>Small problems stay on the CPU</b>: below about 10⁴ elements, the 5 µs launch alone loses. <b>Copying kills cheap work</b>: with the data on the CPU, an elementwise add is limited by PCIe, and the CPU wins. <b>Keep data resident</b>: tick the box and the GPU wins by 30× even for simple operations, because HBM is 30× faster than CPU memory. This is why models are loaded onto the GPU once and stay there, and why AI frameworks fuse many small operations into one kernel.</p>"),
    ),
    section("Ways to program it",
      table(["Layer", "You write", "Good for"], [
        ["PyTorch / JAX", "<code>y = x @ W</code>", "almost everything: calls NVIDIA's tuned libraries (cuBLAS, cuDNN) and generated kernels"],
        ["torch.compile", "the same Python, plus one decorator", "fusing chains of small operations into fewer kernels automatically"],
        ["Triton", "Python-like kernels over blocks of data", "custom fused kernels (attention, norms) without managing threads one by one"],
        ["CUDA C++", "one thread's code, plus shared memory and sync", "full control; learning how the machine works"],
        ["CUTLASS / CuTe, ThunderKittens", "templated tile-level C++", "matrix multiply and attention kernels at near peak speed on tensor cores"],
      ].map((r) => r.map((x) => (x.includes("<code>") ? h("span", { html: x }) : x)))),
      callout("tip", "Where to go from here", "<p>You now have the whole stack, from electrons to a kernel launch. To see it as one picture, zoom through all ten levels with real numbers in <a href='/learn?c=all-the-way-down'>one multiply, all the way down</a>. The next steps are in the systems chapters: <a href='/learn?c=gpu-from-scratch'>GPU kernels from first principles</a> (coalescing, tiling, bank conflicts, occupancy), <a href='/learn?c=kernels'>the roofline and FlashAttention</a>, <a href='/learn?c=precision'>number formats</a> and <a href='/learn?c=distributed'>spreading a model over many GPUs</a>, which is where AlexNet's two-GPU split grew up.</p>"),
    ),
    section("In code",
      code({
        "A complete CUDA program": `#include <cstdio>
#include <cuda_runtime.h>

__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    int n = 1 << 20; size_t bytes = n * sizeof(float);
    float *ha = (float*)malloc(bytes), *hb = (float*)malloc(bytes), *hc = (float*)malloc(bytes);
    for (int i = 0; i < n; ++i) { ha[i] = i; hb[i] = 2 * i; }

    float *a, *b, *c;
    cudaMalloc(&a, bytes); cudaMalloc(&b, bytes); cudaMalloc(&c, bytes);
    cudaMemcpy(a, ha, bytes, cudaMemcpyHostToDevice);        // across PCIe
    cudaMemcpy(b, hb, bytes, cudaMemcpyHostToDevice);

    int threads = 256, blocks = (n + threads - 1) / threads;  // round up
    add<<<blocks, threads>>>(a, b, c, n);                      // asynchronous
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("%s\\n", cudaGetErrorString(err)); return 1; }

    cudaMemcpy(hc, c, bytes, cudaMemcpyDeviceToHost);         // waits for the kernel
    printf("c[123] = %f (expect %f)\\n", hc[123], 3.0f * 123);
    cudaFree(a); cudaFree(b); cudaFree(c);
}
// nvcc add.cu -o add && ./add`,
        "From Python, on Colab": `import torch
from torch.utils.cpp_extension import load_inline

src = r"""
__global__ void add_k(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
torch::Tensor add(torch::Tensor a, torch::Tensor b) {
    auto c = torch::empty_like(a);
    int n = a.numel(), t = 256;
    add_k<<<(n + t - 1) / t, t>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n);
    return c;
}"""
ext = load_inline("add_ext", cpp_sources="torch::Tensor add(torch::Tensor a, torch::Tensor b);",
                  cuda_sources=src, functions=["add"])
a = torch.randn(10**7, device="cuda"); b = torch.randn_like(a)
assert torch.allclose(ext.add(a, b), a + b)`,
        "The same in Triton": `import triton, triton.language as tl

@triton.jit
def add_kernel(a, b, c, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)                       # like blockIdx.x
    offs = pid * BLOCK + tl.arange(0, BLOCK)      # a whole block of indices at once
    mask = offs < n                               # the guard
    tl.store(c + offs, tl.load(a + offs, mask=mask) + tl.load(b + offs, mask=mask), mask=mask)

grid = (triton.cdiv(n, 1024),)
add_kernel[grid](a, b, c, n, BLOCK=1024)`,
      }, "Project 60 compiles and times this kernel on a free Colab T4, measures PCIe against on-GPU bandwidth, and checks Little's law."),
    ),
  );
  drawG(); drawW();
}
