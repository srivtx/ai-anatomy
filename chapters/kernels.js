import { h } from "../lib/core.js";
import { walk } from "../lib/walks.js";
import { slider, segmented, stats, predict, code, callout, section, prose, table, stepper } from "../lib/widgets.js";
import { xyChart, fmtBytes } from "../lib/plot.js";

const GPUS = { a100: { name: "A100", peak: 312e12, bw: 2.0e12 }, h100: { name: "H100", peak: 989e12, bw: 3.35e12 }, b200: { name: "B200", peak: 2250e12, bw: 8e12 } };

export default function (root, { id }) {
  const o = { gpu: "h100", n: 4096, b: 1, seq: 8192 };
  const rc = xyChart({ height: 340, xLog: true, yLog: true, xLabel: "arithmetic intensity (FLOPs per byte moved)", yLabel: "attainable TFLOP/s", xDomain: [0.1, 5000], yDomain: [0.1, 3000] });
  const st = stats([{ label: "Ridge point" }, { label: "Square matmul" }, { label: "Decode step (one layer's weights)" }, { label: "Attention: naive → Flash" }]);
  function ops() {
    const n = o.n, b = o.b, d = 128, s = o.seq, D = 8192;
    return [
      { name: "Element-wise add", flops: 1, bytes: 6, color: "var(--c5)" },
      { name: "LayerNorm / RMSNorm", flops: 5, bytes: 4, color: "var(--c4)" },
      { name: `Matmul ${n}×${n}×${n}`, flops: 2 * n ** 3, bytes: 2 * 3 * n * n, color: "var(--accent)" },
      { name: `Decode: batch ${b} × ${D}² weights`, flops: 2 * b * D * D, bytes: 2 * D * D + 4 * b * D, color: "var(--c2)" },
      { name: `Attention, naive (seq ${s})`, flops: 4 * s * s * d, bytes: 8 * s * s + 8 * s * d, color: "var(--bad)" },
      { name: `Attention, FlashAttention (seq ${s})`, flops: 4 * s * s * d, bytes: 8 * s * d, color: "var(--good)" },
    ];
  }
  function draw() {
    const g = GPUS[o.gpu], ridge = g.peak / g.bw;
    const roof = Array.from({ length: 101 }, (_, i) => { const ai = 10 ** (-1 + i * 0.05); return [ai, Math.min(g.peak, g.bw * ai) / 1e12]; });
    const os = ops();
    rc.update([{ name: `${g.name} roofline`, color: "var(--ink)", points: roof }, ...os.map((op) => { const ai = op.flops / op.bytes; return { name: op.name, color: op.color, points: [[ai, Math.min(g.peak, g.bw * ai) / 1e12]], dots: true, line: false, r: 6 }; })], { xDomain: [0.1, 5000], yDomain: [0.1, 3000], marks: [{ x: ridge, label: "ridge" }] });
    const ft = (x) => (x >= 1e-3 ? `${(x * 1e3).toFixed(2)} ms` : `${(x * 1e6).toFixed(x < 1e-5 ? 1 : 0)} µs`);
    const t = (op) => Math.max(op.flops / g.peak, op.bytes / g.bw), lab = (op) => { const ai = op.flops / op.bytes; return ai < ridge ? "memory-bound" : "compute-bound"; };
    st.set(0, `${Math.round(ridge)} FLOPs/byte`);
    st.set(1, `${lab(os[2])} · ${ft(t(os[2]))}`, lab(os[2]) === "compute-bound" ? "good" : "warn");
    st.set(2, `${lab(os[3])} · ${ft(t(os[3]))}`, lab(os[3]) === "compute-bound" ? "good" : "warn");
    st.set(3, `${ft(t(os[4]))} → ${ft(t(os[5]))} per head`, "good");
  }

  // ---------- fusion ----------
  let k = 4, N = 64e6;
  const fst = stats([{ label: "Bytes moved, unfused" }, { label: "Bytes moved, fused" }, { label: "Speed-up (memory-bound)" }]);
  function drawFuse() {
    const unf = k * 4 * N, fu = 4 * N;
    fst.set(0, fmtBytes(unf)); fst.set(1, fmtBytes(fu), "good"); fst.set(2, `${k}×`, "good");
  }

  const tile = (qi, kj, done) => {
    const cells = [];
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
      const now = i === qi && j === kj, past = done(i, j), masked = j > i;
      cells.push(`<span style="display:block;aspect-ratio:1;border-radius:4px;background:${masked ? "transparent" : now ? "var(--accent)" : past ? "color-mix(in srgb, var(--accent) 30%, var(--surface))" : "var(--sunken)"};border:1px solid var(--line)"></span>`);
    }
    return `<div style="display:grid;grid-template-columns:repeat(4,34px);gap:4px;margin:8px 0">${cells.join("")}</div>`;
  };

  root.append(
    section(null, prose(`<p>A GPU can do arithmetic far faster than it can fetch data from its main memory (HBM). An H100 performs about 989 trillion bf16 operations per second but reads only 3.35 trillion bytes per second. So the question for any operation is: how many FLOPs does it do per byte it moves? That ratio, the <b>arithmetic intensity</b>, decides whether the operation is limited by memory or by compute. The <b>roofline</b> plots both limits on one chart. Almost every speed trick in modern ML (FlashAttention, fusion, batching, quantisation) is about moving fewer bytes.</p>`)),
    section("The roofline",
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "GPU", options: Object.entries(GPUS).map(([k2, g]) => ({ value: k2, label: g.name })), value: o.gpu, onChange: (v) => { o.gpu = v; draw(); } }),
          slider({ label: "Matmul size n", min: 6, max: 14, step: 1, value: Math.log2(o.n), fmt: (v) => String(2 ** v), onInput: (v) => { o.n = 2 ** v; draw(); } }),
          slider({ label: "Decode batch size", min: 0, max: 9, step: 1, value: 0, fmt: (v) => String(2 ** v), onInput: (v) => { o.b = 2 ** v; draw(); } }),
          slider({ label: "Attention sequence length", min: 10, max: 17, step: 1, value: Math.log2(o.seq), fmt: (v) => String(2 ** v), onInput: (v) => { o.seq = 2 ** v; draw(); } })),
        h("div.stage-body", st.el, rc.el, h("p.canvas-cap", "Left of the ridge, speed = bandwidth × intensity (memory-bound). Right of it, you hit peak compute. Points sit on the roof: the best case for each operation."))),
      callout("tip", "Things to try", "<ul style='margin:0;padding-left:18px'><li>Element-wise ops and norms sit far left: they're always memory-bound. Never optimise their FLOPs; fuse them so data is read once.</li><li>A small matmul (n = 128) is memory-bound; a big one is compute-bound. That's why GPUs need big batches and why tiny models underuse them.</li><li>Decode at batch 1 has intensity ≈ 1: every weight is read to do one multiply-add per token. Raise the batch: intensity grows linearly with it. That's the roofline view of why serving engines batch requests.</li><li>Naive attention writes and re-reads the s × s score matrix, so it's memory-bound. FlashAttention never writes it, so the same FLOPs become compute-bound.</li></ul>"),
      predict({
        chapter: id,
        question: "You speed up an RMSNorm kernel's arithmetic by 2× (fewer instructions). How much faster does training get?",
        options: ["2× faster norm, noticeable overall", "Essentially no change: the norm is memory-bound, so its time is set by bytes moved, not arithmetic", "Slower", "It depends on the learning rate"],
        answer: 1,
        explain: "<p>For memory-bound kernels, time ≈ bytes / bandwidth. Fewer FLOPs change nothing; fewer bytes (fusing the norm into the next matmul, or lower precision) do. Profiling tells you which side of the ridge each kernel is on before you spend effort.</p>",
      }),
    ),
    section("Kernel fusion",
      prose("<p>A chain like <code>gelu(x · w + b) * scale</code> written as separate PyTorch ops reads and writes the whole tensor at every step. A fused kernel loads each element once, does all the maths in registers, and writes once. <code>torch.compile</code> does this automatically for many patterns; Triton lets you write your own.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Element-wise ops in the chain", min: 2, max: 10, value: k, onInput: (v) => { k = v; drawFuse(); } }),
          slider({ label: "Tensor size", min: 20, max: 30, step: 1, value: Math.log2(N), fmt: (v) => `${(2 ** v / 1e6).toFixed(0)}M elements`, onInput: (v) => { N = 2 ** v; drawFuse(); } })),
        h("div.stage-body", fst.el)),
    ),
    section("How FlashAttention tiles attention",
      prose("<p>Split Q into row blocks and K, V into column blocks small enough to fit in on-chip SRAM. For each Q block, stream through the K/V blocks, keeping a running max and running sum for the softmax (the online softmax trick in the deeper cuts below), so the full score matrix never exists in HBM. Causal masking skips the blocks above the diagonal entirely.</p>"),
      stepper([
        { title: "Load Q block 1, then K/V block 1", html: tile(0, 0, () => false) + "<p>Compute scores for this tile in SRAM, update the running max and sum, accumulate into the output.</p>" },
        { title: "Q block 2 streams through K/V blocks 1–2", html: tile(1, 1, (i, j) => (i === 0 && j === 0) || (i === 1 && j === 0)) + "<p>Each new tile rescales the partial output by exp(old max − new max), so the result is exact.</p>" },
        { title: "Q block 3", html: tile(2, 2, (i, j) => j <= i && (i < 2 || (i === 2 && j < 2))) + "<p>Blocks above the diagonal are skipped (causal mask): about half the work disappears.</p>" },
        { title: "Q block 4 done", html: tile(3, 3, (i, j) => j <= i) + "<p>Output written once. Memory traffic is O(n·d) instead of O(n²), and the backward pass recomputes tiles instead of storing them.</p>" },
      ]),
      table(["Version", "Main idea", "Result"], [
        ["FlashAttention (2022)", "Tiling + online softmax + recomputation in backward", "Exact attention, 2–4× faster, linear memory"],
        ["FlashAttention-2 (2023)", "Better work partitioning across warps and sequence", "~2× faster again, ~70% of A100 peak"],
        ["FlashAttention-3 (2024)", "Hopper features: asynchronous loads (TMA), warp specialisation, fp8", "Up to ~75% of H100 peak in fp16"],
      ]),
    ),
    walk("kernels"),
    section("In code",
      code({
        "Triton: fused add + ReLU": `import triton, triton.language as tl

@triton.jit
def add_relu(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask); y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, tl.maximum(x + y, 0), mask=mask)    # one read each, one write

def fused(x, y):
    out = torch.empty_like(x)
    add_relu[(triton.cdiv(x.numel(), 1024),)](x, y, out, x.numel(), BLOCK=1024)
    return out`,
        "Measure, don't guess": `from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    model(x).sum().backward()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
# then compare each kernel's achieved bytes/s with the GPU's bandwidth, and FLOP/s with its peak`,
      }, "Project 13 writes Triton kernels for add, softmax and a fused matmul epilogue, and plots them against the roofline."),
    ),
  );
  draw(); drawFuse();
}
