import { h } from "../lib/core.js";
import { tileLab } from "../lib/labs.js";
import { slider, stats, predict, code, callout, section, prose, table, stepper, segmented, toggle } from "../lib/widgets.js";
import { flowCanvas } from "../lib/flow.js";
import { xyChart } from "../lib/plot.js";

// H100 SXM, approximate public specs
const H100 = { peak: 989e12, bw: 3.35e12, sms: 132, regs: 65536, smem: 228 * 1024, maxThreads: 2048, maxBlocks: 32 };

function canvas(width, height, label) {
  const fc = flowCanvas({ width, height, label });
  fc.paint = (fn) => { fc.redraw = fn; fn(); };
  return fc;
}

export default function (root, { id }) {
  // ---------- 1. coalescing ----------
  const co = { pattern: "row", stride: 1 };
  const cc = canvas(880, 250, "32 threads of a warp reading memory");
  const cst = stats([{ label: "32-byte sectors fetched" }, { label: "Useful bytes / fetched bytes" }, { label: "Effective bandwidth (of 3.35 TB/s)" }]);
  const strideSl = slider({ label: "Stride (elements between threads)", min: 1, max: 32, value: 1, onInput: (v) => { co.stride = v; drawCo(); } });
  function addrs() {
    // float32 addresses, in elements, for thread t = 0..31
    const t = [...Array(32).keys()];
    if (co.pattern === "row") return t.map((i) => i);
    if (co.pattern === "offset") return t.map((i) => i + 3);
    if (co.pattern === "stride") return t.map((i) => i * co.stride);
    if (co.pattern === "column") return t.map((i) => i * 64);          // reading a column of a row-major matrix with 64 columns
    let s = 7; return t.map(() => (s = (s * 1103515245 + 12345) % 2147483648) % 1024);
  }
  function drawCo() {
    strideSl.style.display = co.pattern === "stride" ? "" : "none";
    const A = addrs(), sectors = new Set(A.map((a) => Math.floor((a * 4) / 32)));
    const useful = 32 * 4, fetched = sectors.size * 32, eff = useful / fetched;
    cst.set(0, String(sectors.size), sectors.size <= 4 ? "good" : sectors.size <= 8 ? "warn" : "bad");
    cst.set(1, `${Math.round(eff * 100)}%`, eff > 0.9 ? "good" : eff > 0.3 ? "warn" : "bad");
    cst.set(2, `${(3.35 * eff).toFixed(2)} TB/s`, eff > 0.9 ? "good" : eff > 0.3 ? "warn" : "bad");
    cc.paint(() => {
      cc.clear();
      const c = cc.colors, W = cc.W;
      // memory: show the span of elements touched, laid out as sectors of 8 floats
      const maxA = Math.max(...A) + 1, nSec = Math.max(8, Math.ceil(maxA / 8)), show = Math.min(nSec, 96);
      const sw = (W - 40) / show, my = 170;
      cc.text("THREADS OF ONE WARP", 20, 18, { size: 10.5, color: c["ink-3"], mono: true });
      cc.text(`GLOBAL MEMORY (each box = one 32-byte sector = 8 floats)${nSec > show ? ", first " + show + " sectors shown" : ""}`, 20, my - 16, { size: 10.5, color: c["ink-3"], mono: true });
      for (let s = 0; s < show; s++) {
        const hit = sectors.has(s);
        cc.box(20 + s * sw + 1, my, sw - 2, 44, { fill: hit ? c.accent : c.sunken, stroke: "transparent", r: Math.min(5, sw / 3) });
      }
      const tw = (W - 40) / 32;
      A.forEach((a, t) => {
        const tx = 20 + t * tw + tw / 2, sec = Math.floor(a / 8);
        if (sec < show) {
          const mx = 20 + (a / 8) * sw + 1;
          cc.line(tx, 46, mx, my, { color: c.accent, lw: 1 });
        }
        cc.dot(tx, 40, Math.min(7, tw / 2.6), c.c4);
      });
      cc.text(`${sectors.size} memory transactions of 32 bytes for 128 useful bytes`, 20, my + 64, { size: 12.5, color: c.ink });
    });
  }

  // ---------- 2. tiling a matmul ----------
  const tm = { T: 32, N: 4096 };
  const tc = canvas(880, 330, "Tiled matrix multiply: one block computes a T×T tile of C");
  const tst = stats([{ label: "Global reads per output" }, { label: "Arithmetic intensity" }, { label: "Shared memory per block" }, { label: "Roofline limit on H100" }]);
  const tchart = xyChart({ width: 880, height: 240, xLog: true, yLog: true, xLabel: "tile size T (elements per side)", yLabel: "attainable TFLOPS (bf16)", xDomain: [1, 512], yDomain: [1, 1200] });
  function drawTile() {
    const T = tm.T, N = tm.N, intensity = T / 2;              // bf16: 2N³ flops over (2N³/T) loads × 2 bytes
    const att = Math.min(H100.peak, intensity * H100.bw) / 1e12, smem = 2 * T * 32 * 2;   // A and B panels of T×32 in bf16
    tst.set(0, `${(2 * N / T).toFixed(0)} elements`); tst.set(1, `${intensity.toFixed(1)} FLOP/byte`, intensity > 295 ? "good" : intensity > 30 ? "warn" : "bad");
    tst.set(2, `${(smem / 1024).toFixed(1)} KB`, smem < H100.smem / 2 ? "good" : smem < H100.smem ? "warn" : "bad");
    tst.set(3, `${att.toFixed(att < 10 ? 1 : 0)} TFLOPS (${Math.round((att * 1e12 / H100.peak) * 100)}% of peak)`, att > 800 ? "good" : att > 200 ? "warn" : "bad");
    const xs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
    tchart.update([
      { name: "Shared-memory tiling alone", color: "var(--accent)", points: xs.map((t) => [t, Math.min(989, (t / 2) * 3.35)]), dots: true },
      { name: "Tensor-core peak (989 TFLOPS)", color: "var(--ink-3)", points: [[1, 989], [512, 989]], dash: true },
    ], { xDomain: [1, 512], yDomain: [1, 1200], marks: [{ x: T, label: `T = ${T}` }] });
    tc.paint(() => {
      tc.clear();
      const c = tc.colors, S = 220, gx = 20, gy = 50;
      const draw = (x, y, label, hl) => {
        tc.box(x, y, S, S, { fill: c.sunken, stroke: c.line, r: 6 });
        const k = S / 16, tk = Math.max(1, Math.round((T / 128) * 16));
        hl(x, y, k, tk);
        tc.text(label, x + S / 2, y - 14, { size: 12, color: c["ink-2"], align: "center" });
      };
      const row = 5, col = 9;
      draw(gx, gy, "A (N×N)", (x, y, k, tk) => { tc.box(x, y + row * k, S, tk * k, { fill: c.c2, stroke: "transparent", r: 2 }); tc.box(x + 3 * k, y + row * k, 2 * k, tk * k, { fill: c.accent, stroke: "transparent", r: 2 }); });
      draw(gx + 290, gy, "B (N×N)", (x, y, k, tk) => { tc.box(x + col * k, y, tk * k, S, { fill: c.c2, stroke: "transparent", r: 2 }); tc.box(x + col * k, y + 3 * k, tk * k, 2 * k, { fill: c.accent, stroke: "transparent", r: 2 }); });
      draw(gx + 580, gy, "C = A·B", (x, y, k, tk) => { tc.box(x + col * k, y + row * k, tk * k, tk * k, { fill: c.good, stroke: c.ink, lw: 2, r: 2 }); });
      tc.text("×", gx + S + 35, gy + S / 2, { size: 22, color: c["ink-3"], align: "center" });
      tc.text("=", gx + 290 + S + 35, gy + S / 2, { size: 22, color: c["ink-3"], align: "center" });
      tc.text(`One thread block computes a ${T}×${T} tile of C (green). It walks along the orange strips, loading a ${T}×32 slice of each (purple) into shared memory,`, 20, gy + S + 22, { size: 12, color: c.ink });
      tc.text(`and every loaded element is reused ${T} times from fast on-chip memory instead of being re-read from HBM.`, 20, gy + S + 40, { size: 12, color: c.ink });
    });
  }

  // ---------- 3. bank conflicts ----------
  const bk = { stride: 32, pad: false };
  const bc = canvas(880, 220, "Shared memory banks");
  const bst = stats([{ label: "Threads hitting the busiest bank" }, { label: "Shared-memory speed" }]);
  function drawBank() {
    const ld = bk.stride + (bk.pad ? 1 : 0), banks = Array.from({ length: 32 }, () => []);
    for (let t = 0; t < 32; t++) banks[(t * ld) % 32].push(t);
    const worst = Math.max(...banks.map((b) => b.length));
    bst.set(0, `${worst}-way conflict${worst === 1 ? " (none)" : ""}`, worst === 1 ? "good" : worst <= 4 ? "warn" : "bad");
    bst.set(1, worst === 1 ? "full speed: 1 cycle" : `${worst} serialised passes (1/${worst} speed)`, worst === 1 ? "good" : "bad");
    bc.paint(() => {
      bc.clear();
      const c = bc.colors, bw = (bc.W - 40) / 32;
      bc.text(`32 threads read column 0 of a shared tile whose rows are ${ld} floats long. Thread t reads word t·${ld}; bank = word mod 32.`, 20, 16, { size: 12, color: c.ink });
      banks.forEach((b, i) => {
        const x = 20 + i * bw;
        bc.box(x + 2, 190, bw - 4, 14, { fill: b.length ? (b.length > 1 ? c.bad : c.good) : c.sunken, stroke: "transparent", r: 3 });
        if (i % 4 === 0) bc.text(String(i), x + bw / 2, 212, { size: 9.5, color: c["ink-3"], align: "center", mono: true });
        b.forEach((t, k) => bc.dot(x + bw / 2, 178 - k * 4.7, 3, c.c4));
      });
    });
  }

  // ---------- 4. occupancy ----------
  const oc = { regs: 64, threads: 256, smemKB: 48 };
  const ost = stats([{ label: "Blocks per SM" }, { label: "Warps per SM (max 64)" }, { label: "Occupancy" }, { label: "Limited by" }]);
  function drawOcc() {
    const byRegs = Math.floor(H100.regs / (oc.regs * oc.threads)), bySmem = oc.smemKB ? Math.floor(228 / oc.smemKB) : 99, byThreads = Math.floor(H100.maxThreads / oc.threads);
    const lim = [["registers", byRegs], ["shared memory", bySmem], ["threads", byThreads], ["block limit", H100.maxBlocks]].sort((a, b) => a[1] - b[1])[0];
    const blocks = Math.max(0, lim[1]), warps = Math.min(64, (blocks * oc.threads) / 32);
    ost.set(0, String(blocks), blocks ? "" : "bad"); ost.set(1, String(warps)); ost.set(2, `${Math.round((warps / 64) * 100)}%`, warps / 64 >= 0.5 ? "good" : warps / 64 >= 0.25 ? "warn" : "bad");
    ost.set(3, blocks ? lim[0] : `${lim[0]}: block doesn't fit`, "");
  }

  root.append(
    section(null, prose(`<p>Most ML engineers never write a GPU kernel, and the ones who can are some of the most sought-after people in the field. The knowledge isn't secret, but it's scattered across vendor manuals, blog posts and folklore. This chapter builds it up from first principles: a GPU is thousands of slow, simple cores that only go fast when memory is accessed in the right patterns. Everything below is about memory.</p><p class='muted'>Want the why underneath, from electrons and transistors up to warps and CUDA? Start with <a href='/learn?c=ai-physics'>From electrons to CUDA</a>, then come back.</p>`)),
    section("The machine",
      stepper([
        { title: "Thread → warp", html: "<p>You write code for one <b>thread</b>. The hardware runs threads in groups of 32 called <b>warps</b>, which execute the same instruction at the same time (SIMT). If threads in a warp take different branches, both paths run one after the other: <b>warp divergence</b>.</p>" },
        { title: "Warp → block → SM", html: "<p>Threads are grouped into <b>blocks</b> (up to 1,024 threads). A block runs on one <b>streaming multiprocessor (SM)</b> and its threads share fast on-chip <b>shared memory</b> and can synchronise. An H100 has 132 SMs, each holding up to 64 warps at once.</p>" },
        { title: "Why so many warps?", html: "<p>A read from HBM takes several hundred cycles. Instead of caches and branch prediction, a GPU hides that latency by switching to another ready warp every cycle. More resident warps (higher <b>occupancy</b>) or more independent work per thread (ILP) means more latency hidden.</p>" },
        { title: "The memory hierarchy", html: "<p>Approximate H100 figures. <b>Registers</b>: 256 KB per SM, ~1 cycle. <b>Shared memory / L1</b>: up to 228 KB per SM, ~30 cycles. <b>L2</b>: 50 MB, ~200+ cycles. <b>HBM3</b>: 80 GB at 3.35 TB/s, ~500 cycles. Compute: ~989 TFLOPS dense bf16 on tensor cores. Per byte loaded from HBM you must do ~295 FLOPs to stay compute-bound.</p>" },
        { title: "Tensor cores", html: "<p>Most of those FLOPs come from <b>tensor cores</b>: units that multiply small matrix tiles (e.g. 16×8×16) per instruction. Plain CUDA-core FP32 is ~67 TFLOPS. To use tensor cores you must feed them tiles of bf16/fp8 data from shared memory or registers fast enough, which is what the rest of this chapter is about.</p>" },
      ]),
    ),
    section("Coalescing: how a warp reads memory",
      prose("<p>When the 32 threads of a warp load, the hardware merges their addresses into 32-byte <b>sectors</b>. If neighbouring threads read neighbouring addresses, one warp-wide load needs 4 sectors. If they're scattered, each thread may cost a whole sector and 7/8 of the fetched bytes are wasted.</p>"),
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Access pattern", options: [{ value: "row", label: "a[t] (contiguous)" }, { value: "offset", label: "a[t + 3] (misaligned)" }, { value: "stride", label: "a[t·s] (strided)" }, { value: "column", label: "Column of a matrix" }, { value: "random", label: "Random (gather)" }], value: co.pattern, onChange: (v) => { co.pattern = v; drawCo(); } }),
          strideSl),
        h("div.stage-body", cst.el, h("div.wide-canvas", cc.el))),
      callout("tip", "What follows from this", "<ul style='margin:0;padding-left:18px'><li>Lay data out so the fastest-moving thread index walks the fastest-moving memory dimension. Reading a column of a row-major matrix is the classic mistake: 32 sectors for 128 useful bytes.</li><li>When you must read “the wrong way” (e.g. a transpose), load a tile coalesced into shared memory, then read it transposed from there.</li><li>Misaligned access costs only one extra sector: alignment matters much less than people think; striding matters a lot.</li></ul>"),
    ),
    section("Tiling: reuse every byte you load",
      prose("<p>A naive matrix multiply has each thread compute one output by reading a full row of A and a column of B from HBM: 2N loads for 2N FLOPs, an arithmetic intensity of 0.5 FLOP/byte in bf16. The GPU would sit idle >99% of the time. The fix is <b>tiling</b>: a block of threads cooperatively loads tiles of A and B into shared memory once and reuses each element T times.</p>"),
      h("div.stage",
        h("div.stage-controls", slider({ label: "Tile size T", min: 0, max: 9, value: 5, fmt: (v) => String(2 ** v), onInput: (v) => { tm.T = 2 ** v; drawTile(); } })),
        h("div.stage-body", tst.el, h("div.wide-canvas", tc.el), tchart.el)),
      callout("note", "Why real kernels are more complicated", "<p>Shared-memory tiling alone reaches the H100's compute roof only around T ≈ 600, far too big for one block. Real kernels stack several levels of reuse: block tiles in shared memory (e.g. 128×256), warp tiles, per-thread register tiles (each thread computes an 8×8 or larger piece of C), and on Hopper, TMA for asynchronous bulk copies and warp-group MMA instructions, with loads for the next tile overlapped with math on the current one (software pipelining). Simon Boehm's well-known walkthrough on an RTX A6000 (fp32, no tensor cores) went from ~0.3 TFLOPS (naive) to ~2 (coalesced) to ~3 (shared-memory tiles) to ~16 (2D register tiling) to ~22 (warp tiling), against ~23 for cuBLAS.</p>"),
    ),
    section("Tiling, in 3D",
      prose("<p>A matrix multiply is a cube: one multiply-add for every (row, column, k). Colour it by which block owns each output tile and drag k forward to watch the partial sums build up. The numbers are for a real 4096 \u00d7 4096 \u00d7 4096 multiply in bf16 on an H100.</p>"),
      tileLab()),
    section("Bank conflicts in shared memory",
      prose("<p>Shared memory is split into 32 <b>banks</b>, each 4 bytes wide; word <i>w</i> lives in bank <i>w</i> mod 32. A warp's access is full speed if every thread hits a different bank (or the same word). If several threads hit different words in the same bank, the accesses are serialised.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Row length of the tile (floats)", min: 1, max: 64, value: bk.stride, onInput: (v) => { bk.stride = v; drawBank(); } }),
          toggle({ label: "Pad each row by one float", checked: bk.pad, onChange: (v) => { bk.pad = v; drawBank(); } })),
        h("div.stage-body", bst.el, h("div.wide-canvas", bc.el))),
      callout("tip", "The one-float padding trick", "<p>A 32×32 float tile read by column puts all 32 threads in bank 0: a 32-way conflict. Declare it as <code>float tile[32][33]</code> and each row shifts by one bank, so a column read hits 32 different banks. Odd row lengths never conflict; powers of two are the worst case. (Modern kernels often use “swizzled” layouts instead of padding, which also work with tensor-core load instructions.)</p>"),
    ),
    section("Occupancy: how many warps fit",
      prose("<p>Each SM has a fixed budget: 65,536 registers, 228 KB of shared memory, 2,048 threads, 32 blocks. Whichever runs out first limits how many warps can be resident to hide latency.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Registers per thread", min: 16, max: 255, step: 8, value: oc.regs, onInput: (v) => { oc.regs = v; drawOcc(); } }),
          slider({ label: "Threads per block", min: 64, max: 1024, step: 64, value: oc.threads, onInput: (v) => { oc.threads = v; drawOcc(); } }),
          slider({ label: "Shared memory per block (KB)", min: 0, max: 228, step: 4, value: oc.smemKB, onInput: (v) => { oc.smemKB = v; drawOcc(); } })),
        h("div.stage-body", ost.el)),
      callout("note", "High occupancy is not the goal", "<p>Volkov's classic talk “Better performance at lower occupancy” (2010) showed that more registers per thread and more independent work per thread often beat more warps. FlashAttention and fast GEMMs run at low occupancy on purpose: big register tiles maximise reuse. Occupancy matters most for memory-bound kernels with little work per thread.</p>"),
      predict({
        chapter: id,
        question: "Your elementwise kernel (y = gelu(x) on a big tensor) reaches 30% of HBM bandwidth. What's the most likely problem?",
        options: ["Not enough tensor cores", "Uncoalesced or too-small memory accesses, or too few threads in flight to hide latency; it's memory-bound, so FLOPs are irrelevant", "The GELU formula is too expensive", "Shared memory bank conflicts"],
        answer: 1,
        explain: "<p>An elementwise op does a few FLOPs per 4–8 bytes, so it can only ever be limited by memory. Check that consecutive threads touch consecutive elements, use vectorised loads (e.g. 16 bytes per thread), launch enough blocks to fill all SMs, and better still, fuse it into the neighbouring kernel so the tensor never round-trips through HBM.</p>",
      }),
    ),
    section("How kernel engineers work",
      table(["Step", "Tool", "What you look for"], [
        ["Find the hot kernels", "torch.profiler, Nsight Systems (nsys)", "Which kernels take the time; gaps where the GPU idles"],
        ["Classify each one", "Roofline: FLOPs and bytes moved", "Memory-bound or compute-bound? How far from the roof?"],
        ["Try the cheap fixes", "torch.compile, fused library ops, bigger batch", "Often 1.5–3× with no custom code"],
        ["Write or tune a kernel", "Triton (Python), CUTLASS/CuTe, ThunderKittens, raw CUDA", "Tiling, coalescing, fusion, pipelining"],
        ["Measure the kernel", "Nsight Compute (ncu)", "Achieved bandwidth/FLOPs, stalls, bank conflicts, occupancy"],
        ["Check correctness", "Compare against a reference in fp32", "Max abs/relative error; edge sizes that aren't multiples of the tile"],
      ]),
    ),
    section("In code",
      code({
        "Naive CUDA matmul": `__global__ void matmul_naive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // threadIdx.x varies fastest: keep it on the column
    if (row < N && col < N) {
        float acc = 0.f;
        for (int k = 0; k < N; ++k)
            acc += A[row * N + k] * B[k * N + col];     // B reads are coalesced across the warp; A is a broadcast
        C[row * N + col] = acc;
    }
}
// launch: dim3 block(32, 32); dim3 grid(N / 32, N / 32); matmul_naive<<<grid, block>>>(A, B, C, N);`,
        "Shared-memory tiled matmul": `#define T 32
__global__ void matmul_tiled(const float* A, const float* B, float* C, int N) {
    __shared__ float As[T][T + 1], Bs[T][T + 1];        // +1 padding avoids bank conflicts
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * T + ty, col = blockIdx.x * T + tx;
    float acc = 0.f;
    for (int k0 = 0; k0 < N; k0 += T) {
        As[ty][tx] = A[row * N + k0 + tx];              // each element loaded once per block...
        Bs[ty][tx] = B[(k0 + ty) * N + col];
        __syncthreads();
        for (int k = 0; k < T; ++k) acc += As[ty][k] * Bs[k][tx];   // ...and reused T times
        __syncthreads();
    }
    C[row * N + col] = acc;
}`,
        "Triton fused softmax": `import triton, triton.language as tl
@triton.jit
def softmax_kernel(out_ptr, in_ptr, n_cols, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)                               # one program per row
    cols = tl.arange(0, BLOCK)
    x = tl.load(in_ptr + row * stride + cols, mask=cols < n_cols, other=-float("inf"))
    x = x - tl.max(x, axis=0)                            # everything below stays in registers
    num = tl.exp(x)
    tl.store(out_ptr + row * stride + cols, num / tl.sum(num, axis=0), mask=cols < n_cols)

def softmax(x):
    out = torch.empty_like(x)
    softmax_kernel[(x.shape[0],)](out, x, x.shape[1], x.stride(0), BLOCK=triton.next_power_of_2(x.shape[1]))
    return out                                           # one HBM read and one write per element`,
        "Triton matmul with autotuning": `@triton.autotune(configs=[
    triton.Config({"BM": 128, "BN": 256, "BK": 64}, num_stages=3, num_warps=8),
    triton.Config({"BM": 64, "BN": 256, "BK": 32}, num_stages=4, num_warps=4),
    triton.Config({"BM": 128, "BN": 128, "BK": 32}, num_stages=4, num_warps=4),
], key=["M", "N", "K"])
@triton.jit
def matmul_kernel(a, b, c, M, N, K, sam, sak, sbk, sbn, scm, scn, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM); rn = pid_n * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        A = tl.load(a + rm[:, None] * sam + (k + rk)[None, :] * sak, mask=(rm[:, None] < M) & (k + rk[None, :] < K), other=0.)
        B = tl.load(b + (k + rk)[:, None] * sbk + rn[None, :] * sbn, mask=(k + rk[:, None] < K) & (rn[None, :] < N), other=0.)
        acc += tl.dot(A, B)                              # compiled to tensor-core MMA instructions
    tl.store(c + rm[:, None] * scm + rn[None, :] * scn, acc.to(tl.bfloat16), mask=(rm[:, None] < M) & (rn[None, :] < N))`,
        "Profiling": `# where does the time go?
nsys profile -o trace python train.py            # timeline: kernels, gaps, CPU launch overhead
# why is this kernel slow?
ncu --set full -k regex:matmul -o report python bench.py   # bandwidth, stalls, occupancy, bank conflicts

# in Python
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    model(x).sum().backward()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))`,
      }, "Project 34 writes these kernels in Triton on a free Colab T4 and benchmarks each step against torch."),
    ),
  );
  drawCo(); drawTile(); drawBank(); drawOcc();
}
