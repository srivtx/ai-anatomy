import { h } from "../lib/core.js";
import { slider, segmented, stats, predict, code, callout, section, prose, table } from "../lib/widgets.js";
import { xyChart, fmtSI, fmtBytes } from "../lib/plot.js";

// dense bf16 peak FLOP/s and HBM bandwidth (vendor spec sheets); prices are rough 2025 cloud rental rates, for illustration
const GPUS = {
  a100: { name: "A100 80GB", peak: 312e12, bw: 2.0e12, mem: 80e9, price: 1.8 },
  h100: { name: "H100 SXM", peak: 989e12, bw: 3.35e12, mem: 80e9, price: 2.5 },
  b200: { name: "B200", peak: 2250e12, bw: 8e12, mem: 192e9, price: 5 },
};
const PRESETS = [
  { name: "GPT-2 small on 10B tokens", N: 124e6, D: 10e9, gpus: 8 },
  { name: "1B on 100B tokens", N: 1e9, D: 100e9, gpus: 64 },
  { name: "Llama 3.1 8B (15T tokens)", N: 8e9, D: 15e12, reported: 1.46e6, gpus: 2048 },
  { name: "Llama 3.1 70B (15T tokens)", N: 70e9, D: 15e12, reported: 7.0e6, gpus: 8192 },
  { name: "Llama 3.1 405B (15.6T tokens)", N: 405e9, D: 15.6e12, reported: 30.84e6, gpus: 16384 },
];

const fmtH = (x) => (x < 1 ? `${Math.round(x * 60)} min` : x < 48 ? `${x.toFixed(1)} h` : `${(x / 24).toFixed(x / 24 < 10 ? 1 : 0)} days`);
const fmt$ = (x) => (x < 1000 ? `$${x.toFixed(0)}` : x < 1e6 ? `$${(x / 1e3).toFixed(x < 1e4 ? 1 : 0)}k` : `$${(x / 1e6).toFixed(x < 1e7 ? 2 : 1)}M`);

export default function (root, { id }) {
  // ---------- training ----------
  const t = { N: 1e9, D: 100e9, gpu: "h100", mfu: 0.4, n: 64, price: GPUS.h100.price, preset: 1 };
  const tst = stats([{ label: "Training FLOPs (6·N·D)" }, { label: "GPU-hours" }, { label: "Wall-clock" }, { label: "Compute bill" }, { label: "Reported by the lab" }]);
  const nS = slider({ label: "Parameters", min: 7, max: 12, step: 0.05, value: Math.log10(t.N), fmt: (v) => fmtSI(10 ** v), onInput: (v) => { t.N = 10 ** v; drawT(); } });
  const dS = slider({ label: "Training tokens", min: 9, max: 13.3, step: 0.05, value: Math.log10(t.D), fmt: (v) => fmtSI(10 ** v), onInput: (v) => { t.D = 10 ** v; drawT(); } });
  const gS = slider({ label: "Number of GPUs", min: 0, max: 15, step: 1, value: Math.log2(t.n), fmt: (v) => (2 ** v).toLocaleString(), onInput: (v) => { t.n = 2 ** v; drawT(); } });
  function drawT() {
    const g = GPUS[t.gpu], F = 6 * t.N * t.D, gpuH = F / (g.peak * t.mfu) / 3600, p = PRESETS[t.preset];
    tst.set(0, `${(F / 10 ** Math.floor(Math.log10(F))).toFixed(1)}e${Math.floor(Math.log10(F))}`);
    tst.set(1, fmtSI(gpuH)); tst.set(2, `${fmtH(gpuH / t.n)} on ${t.n} GPUs`); tst.set(3, fmt$(gpuH * t.price));
    const matches = p && p.reported && Math.abs(p.N - t.N) / p.N < 0.02 && Math.abs(p.D - t.D) / p.D < 0.02;
    tst.set(4, matches ? `${fmtSI(p.reported)} H100-hours (${(p.reported / gpuH).toFixed(1)}× this estimate)` : "pick a Llama preset", matches ? "warn" : "");
  }

  // ---------- serving ----------
  const s = { N: 8e9, bytes: 2, gpu: "h100", price: GPUS.h100.price };
  const sc = xyChart({ height: 280, xLog: true, yLog: true, xLabel: "concurrent requests (batch size)", yLabel: "$ per million output tokens" });
  const sst = stats([{ label: "Weights in memory" }, { label: "Batch 1: tokens/s" }, { label: "Batch 1: $ / M tokens" }, { label: "Batch 64: $ / M tokens" }]);
  function serveAt(b) {
    const g = GPUS[s.gpu], wBytes = s.N * s.bytes;
    // each decode step reads all weights once (memory) and does 2·N FLOPs per sequence (compute); whichever is slower wins
    const step = Math.max(wBytes / g.bw, (2 * s.N * b) / (g.peak * 0.6));
    const tps = b / step;
    return { tps, cost: (s.price / (tps * 3600)) * 1e6, step };
  }
  function drawS() {
    const g = GPUS[s.gpu], wBytes = s.N * s.bytes, nGpu = Math.ceil((wBytes * 1.2) / g.mem);
    const pts = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512].map((b) => [b, serveAt(b).cost * nGpu]);
    sc.update([{ name: "Cost per million tokens", color: "var(--accent)", points: pts, dots: true, r: 3.5 }]);
    const b1 = serveAt(1), b64 = serveAt(64);
    sst.set(0, `${fmtBytes(wBytes)} on ${nGpu} GPU${nGpu > 1 ? "s" : ""}`);
    sst.set(1, `${b1.tps.toFixed(0)}`); sst.set(2, `$${(b1.cost * nGpu).toFixed(2)}`, "bad"); sst.set(3, `$${(b64.cost * nGpu).toFixed(3)}`, "good");
  }

  root.append(
    section(null, prose(`<p>Every decision in ML eventually meets a budget. The good news: costs follow from a few first-principles formulas you already know. Training a transformer costs about <b>6 × parameters × tokens</b> FLOPs. Generating a token costs about <b>2 × parameters</b> FLOPs but, at small batch sizes, it's limited by how fast the GPU can <b>read the weights from memory</b>, not by FLOPs. Plug in hardware numbers and you can estimate any project to within a factor of two or three. Real projects then land at the pessimistic end, and this chapter shows why.</p>`)),
    section("What does a training run cost?",
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Preset", options: PRESETS.map((p, i) => ({ value: i, label: p.name })), value: t.preset, onChange: (v) => { t.preset = v; t.N = PRESETS[v].N; t.D = PRESETS[v].D; t.n = PRESETS[v].gpus; nS.value = Math.log10(t.N); dS.value = Math.log10(t.D); gS.value = Math.log2(t.n); drawT(); } }),
          nS, dS,
          segmented({ label: "GPU", options: Object.entries(GPUS).map(([k, g]) => ({ value: k, label: g.name })), value: t.gpu, onChange: (v) => { t.gpu = v; t.price = GPUS[v].price; drawT(); } }),
          slider({ label: "Utilisation (MFU)", min: 0.1, max: 0.7, step: 0.01, value: t.mfu, fmt: (v) => `${Math.round(v * 100)}%`, onInput: (v) => { t.mfu = v; drawT(); } }),
          gS),
        h("div.stage-body", tst.el)),
      callout("tip", "Things to try", "<ul style='margin:0;padding-left:18px'><li>Pick Llama 3.1 405B at 40% MFU on H100: the formula says about 27M GPU-hours; Meta reported 30.8M. Close, because their large run really did hit ~38–43% MFU.</li><li>Pick Llama 3.1 8B: the formula says ~0.5M GPU-hours at 40%, but Meta reported 1.46M. Small models are harder to run efficiently at scale, and reported totals include restarts, evaluation, ablations and failed jobs. Plan for 2–3× your ideal number.</li><li>GPT-2 small on 10B tokens: a few hundred dollars. Community “speedruns” train GPT-2-quality models on 8 H100s in minutes to hours. Small experiments are cheap; that's why the ablation projects are worth running.</li></ul>"),
    ),
    section("What does serving cost?",
      prose("<p>At batch size 1, generating each token means reading every weight from memory once. An H100 reads about 3.35 TB/s, so an 8B model in bf16 (16 GB) tops out near 200 tokens per second, and most of the GPU's compute sits idle. Serve 64 users at once and the same weight read produces 64 tokens. That's the whole economics of LLM APIs.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Parameters (active per token)", min: 9, max: 11.8, step: 0.05, value: Math.log10(s.N), fmt: (v) => fmtSI(10 ** v), onInput: (v) => { s.N = 10 ** v; drawS(); } }),
          segmented({ label: "Weight precision", options: [{ value: 2, label: "bf16" }, { value: 1, label: "8-bit" }, { value: 0.5, label: "4-bit" }], value: s.bytes, onChange: (v) => { s.bytes = v; drawS(); } }),
          segmented({ label: "GPU", options: Object.entries(GPUS).map(([k, g]) => ({ value: k, label: g.name })), value: s.gpu, onChange: (v) => { s.gpu = v; s.price = GPUS[v].price; drawS(); } })),
        h("div.stage-body", sst.el, sc.el, h("p.canvas-cap", "Simplified: ignores KV-cache reads (which grow with context and batch), prefill, and scheduling overheads. The curve flattens where decoding becomes compute-bound."))),
      predict({
        chapter: id,
        question: "Quantising weights from bf16 to 4-bit roughly does what to batch-1 decoding speed?",
        options: ["Nothing: FLOPs are the same", "Roughly 4× faster, because each token needs a quarter of the bytes read from memory", "Slower, because of dequantisation", "2× faster"],
        answer: 1,
        explain: "<p>Batch-1 decoding is memory-bound, so speed scales with bytes per weight. Real kernels get somewhat less than 4× because of dequantisation overhead and KV-cache reads, but quantisation is the biggest single lever for local inference.</p>",
      }),
    ),
    section("Rules of thumb",
      table(["Quantity", "Rule", "Example"], [
        ["Training compute", "6 · N · D FLOPs", "8B params × 15T tokens ≈ 7.2 × 10²³"],
        ["Inference compute", "2 · N FLOPs per token (+ attention)", "8B model: 16 GFLOPs per token"],
        ["Training memory (Adam, mixed precision)", "≈ 16 bytes per parameter + activations", "8B: ~128 GB before activations, so it must be sharded"],
        ["LoRA fine-tuning memory", "Weights (2 bytes/param, or 0.5 with QLoRA) + small extras", "8B with QLoRA fits on one 24 GB GPU"],
        ["Batch-1 decode speed", "≈ memory bandwidth / model bytes", "H100, 8B bf16: ~200 tokens/s"],
        ["Reality factor", "Reported cost ≈ 1.5–3 × the ideal formula", "Restarts, evals, ablations, idle time, debugging"],
      ]),
      callout("note", "Cost and capability, from experience", "<p>Most teams don't need to pretrain. Fine-tuning a 7–8B model with LoRA on a few million tokens costs tens of dollars. Distilling a big model into a small one for a narrow job is usually the best cost-per-quality you can buy. Pretraining from scratch is for when you have a reason no existing base model satisfies (language, domain, licence, size) and a budget measured in GPU-months.</p>"),
    ),
    section("In code",
      code({
        "Estimate before you launch": `def training_estimate(params, tokens, gpu_tflops=989, mfu=0.4, n_gpus=64, usd_per_gpu_hour=2.5):
    flops = 6 * params * tokens
    gpu_hours = flops / (gpu_tflops * 1e12 * mfu) / 3600
    return {"gpu_hours": gpu_hours, "days": gpu_hours / n_gpus / 24, "usd": gpu_hours * usd_per_gpu_hour,
            "usd_realistic": 2.5 * gpu_hours * usd_per_gpu_hour}     # leave room for reality

print(training_estimate(1e9, 100e9))`,
        "Measure your MFU": `tokens_per_s = tokens_in_step / step_time
achieved = 6 * n_params * tokens_per_s                  # FLOP/s actually used for training
mfu = achieved / (n_gpus * peak_flops_per_gpu)
print(f"MFU {mfu:.1%}")   # 30–50% is good for large runs; below 20% means you're bottlenecked somewhere`,
      }),
    ),
  );
  drawT(); drawS();
}
