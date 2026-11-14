import { h } from "../lib/core.js";
import { slider, segmented, stats, predict, code, callout, section, prose, table, toggle, stepper } from "../lib/widgets.js";
import { fmtBytes, fmtSI } from "../lib/plot.js";

const MODELS = {
  "1b": { name: "1B", P: 1.2e9, L: 16, H: 2048, A: 16 },
  "8b": { name: "8B", P: 8e9, L: 32, H: 4096, A: 32 },
  "70b": { name: "70B", P: 70e9, L: 80, H: 8192, A: 64 },
  "405b": { name: "405B", P: 405e9, L: 126, H: 16384, A: 128 },
};

export default function (root, { id }) {
  const o = { model: "8b", zero: 0, tp: 1, pp: 1, dp: 8, seq: 4096, mb: 1, ckpt: false, flash: true, gpuMem: 80 };
  const st = stats([{ label: "Memory per GPU" }, { label: "GPUs in total" }, { label: "Gradient sync per step (per GPU)" }]);
  const barEl = h("div.membar"), legend = h("div.memleg");
  function draw() {
    const m = MODELS[o.model], shard = o.tp * o.pp, Pg = m.P / shard; // parameters held per GPU before ZeRO
    const N = o.dp;
    const weights = (o.zero >= 3 ? 2 / N : 2) * Pg;
    const grads = (o.zero >= 2 ? 2 / N : 2) * Pg;
    const optim = (o.zero >= 1 ? 12 / N : 12) * Pg;
    // activations per layer (Korthikanti et al., 2022): s·b·h·(34 + 5·a·s/h) bytes, divided by tensor parallelism (with sequence parallelism)
    const layers = m.L / o.pp, s = o.seq, b = o.mb, hd = m.H, a = m.A;
    // FlashAttention never stores the s×s attention scores, which removes the 5·a·s/h term
    const perLayer = o.ckpt ? 2 * s * b * hd : (s * b * hd * (34 + (o.flash ? 0 : (5 * a * s) / hd))) / o.tp;
    // with 1F1B, the first pipeline stage holds activations for up to (stages) micro-batches in flight
    const acts = perLayer * layers * Math.max(1, o.pp > 1 ? o.pp : 1);
    const parts = [["Weights (bf16)", weights, "var(--accent)"], ["Gradients (bf16)", grads, "var(--c2)"], ["Optimizer: fp32 master + Adam moments", optim, "var(--c4)"], ["Activations", acts, "var(--c3)"]];
    const total = parts.reduce((x, p) => x + p[1], 0), cap = o.gpuMem * 1e9, scale = Math.max(total, cap) * 1.05;
    barEl.replaceChildren(...parts.map(([n, v, c]) => h("span", { title: `${n}: ${fmtBytes(v)}`, style: { width: `${(v / scale) * 100}%`, background: c } })), h("i.cap", { style: { left: `${(cap / scale) * 100}%` }, title: `GPU memory ${o.gpuMem} GB` }));
    legend.replaceChildren(...parts.map(([n, v, c]) => h("span", h("b", { style: { background: c } }), `${n}: ${fmtBytes(v)}`)), h("span", h("b.capk"), `GPU memory: ${o.gpuMem} GB`));
    st.set(0, `${fmtBytes(total)}${total > cap ? " · doesn't fit" : " · fits"}`, total > cap ? "bad" : "good");
    st.set(1, `${N * shard} = ${N} data × ${o.tp} tensor × ${o.pp} pipeline`);
    // ring all-reduce moves ~2× the gradient bytes per GPU; ZeRO-3 adds parameter all-gathers (~1.5× in total)
    st.set(2, fmtBytes((o.zero >= 3 ? 3 : 2) * 2 * Pg));
  }

  // ---------- pipeline bubble ----------
  let P = 4, M = 4, sched = "1f1b";
  const grid = h("div.pipe");
  const pst = stats([{ label: "Bubble (idle) fraction" }, { label: "Activations kept per stage (micro-batches)" }]);
  function drawPipe() {
    // simple timetable: forward takes 1 slot, backward 2 slots
    const rows = Array.from({ length: P }, () => []);
    if (sched === "gpipe") {
      for (let s = 0; s < P; s++) { for (let k = 0; k < s; k++) rows[s].push(null); for (let mb = 0; mb < M; mb++) rows[s].push(["F", mb]); }
      const tF = M + P - 1;
      for (let s = 0; s < P; s++) { while (rows[s].length < tF) rows[s].push(null); for (let k = 0; k < (P - 1 - s) * 2; k++) rows[s].push(null); for (let mb = M - 1; mb >= 0; mb--) rows[s].push(["B", mb], ["B", mb]); }
    } else {
      // 1F1B: warm up with (P - s) forwards, then alternate one forward and one backward
      const time = Array.from({ length: P }, () => 0), done = {}, fdone = {};
      const events = [];
      // each stage runs its own op order; an op starts once its input is ready and the GPU is free
      const order = Array.from({ length: P }, (_, s) => { const o2 = [], warm = Math.min(P - s, M); let f = 0, bk = 0; for (; f < warm; f++) o2.push(["F", f]); while (bk < M) { o2.push(["B", bk++]); if (f < M) o2.push(["F", f++]); } return o2; });
      const idx = new Array(P).fill(0);
      let progress = true;
      while (progress) {
        progress = false;
        for (let s = 0; s < P; s++) {
          const op = order[s][idx[s]]; if (!op) continue;
          const [t, mb] = op;
          let ready;
          if (t === "F") ready = s === 0 ? 0 : fdone[`${s - 1},${mb}`];
          else ready = s === P - 1 ? fdone[`${s},${mb}`] : done[`${s + 1},${mb}`];
          if (ready === undefined) continue;
          const start = Math.max(time[s], ready), dur = t === "F" ? 1 : 2;
          events.push([s, start, dur, t, mb]); time[s] = start + dur;
          if (t === "F") fdone[`${s},${mb}`] = start + dur; else done[`${s},${mb}`] = start + dur;
          idx[s]++; progress = true;
        }
      }
      const T = Math.max(...time);
      for (let s = 0; s < P; s++) { rows[s] = new Array(T).fill(null); for (const [st2, start, dur, t, mb] of events) if (st2 === s) for (let k = 0; k < dur; k++) rows[s][start + k] = [t, mb]; }
    }
    const T = Math.max(...rows.map((r) => r.length));
    grid.style.gridTemplateColumns = `64px repeat(${T}, minmax(10px, 1fr))`;
    grid.replaceChildren(...rows.flatMap((r, s) => [h("span.pipe-l", `GPU ${s + 1}`), ...Array.from({ length: T }, (_, t) => { const c = r[t]; return h("span.pipe-c", { class: `pipe-c ${c ? c[0] : "idle"}`, title: c ? `${c[0] === "F" ? "forward" : "backward"} micro-batch ${c[1] + 1}` : "idle" }, c && (c[0] === "F" || (t === 0 || !r[t - 1] || r[t - 1][1] !== c[1] || r[t - 1][0] !== "B")) ? String(c[1] + 1) : ""); })]));
    const busy = 3 * M, bubble = 1 - busy / T;
    pst.set(0, `${Math.round(bubble * 100)}%  (formula: (P−1)/(M+P−1) = ${Math.round(((P - 1) / (M + P - 1)) * 100)}%)`, bubble > 0.3 ? "bad" : bubble > 0.15 ? "warn" : "good");
    pst.set(1, sched === "gpipe" ? `${M} (all of them)` : `≤ ${P} (bounded by pipeline depth)`);
  }

  root.append(
    section(null, prose(`<p>A 70B model in training needs over a terabyte of memory for weights, gradients and optimizer states alone. No GPU has that. Distributed training splits the work across many GPUs, and there are only a few ways to split: copy the model and split the <b>data</b>; split each layer's matrices across GPUs (<b>tensor</b> parallelism); split the stack of layers into stages (<b>pipeline</b> parallelism); and shard the optimizer, gradients and weights across data-parallel copies (<b>ZeRO / FSDP</b>). Real runs combine all of them (“3D parallelism”). The calculator shows why.</p>`)),
    section("Will it fit?",
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Model", options: Object.entries(MODELS).map(([k, m]) => ({ value: k, label: m.name })), value: o.model, onChange: (v) => { o.model = v; draw(); } }),
          segmented({ label: "ZeRO stage", options: [{ value: 0, label: "Plain DDP" }, { value: 1, label: "1: shard optimizer" }, { value: 2, label: "2: + gradients" }, { value: 3, label: "3 / FSDP: + weights" }], value: o.zero, onChange: (v) => { o.zero = v; draw(); } }),
          slider({ label: "Data-parallel GPUs", min: 0, max: 10, step: 1, value: Math.log2(o.dp), fmt: (v) => String(2 ** v), onInput: (v) => { o.dp = 2 ** v; draw(); } }),
          segmented({ label: "Tensor parallel", options: [1, 2, 4, 8].map((v) => ({ value: v, label: String(v) })), value: o.tp, onChange: (v) => { o.tp = v; draw(); } }),
          segmented({ label: "Pipeline stages", options: [1, 2, 4, 8, 16].map((v) => ({ value: v, label: String(v) })), value: o.pp, onChange: (v) => { o.pp = v; draw(); } }),
          slider({ label: "Sequence length", min: 1024, max: 32768, step: 1024, value: o.seq, onInput: (v) => { o.seq = v; draw(); } }),
          toggle({ label: "Activation checkpointing", checked: false, onChange: (v) => { o.ckpt = v; draw(); } }),
          toggle({ label: "FlashAttention (don't store attention scores)", checked: true, onChange: (v) => { o.flash = v; draw(); } })),
        h("div.stage-body", st.el, barEl, legend)),
      callout("tip", "Things to try", "<ul style='margin:0;padding-left:18px'><li>8B with plain DDP: 128 GB per GPU before activations. Each GPU holds a full copy of everything, mostly optimizer state (12 bytes per parameter). Switch to ZeRO-3 with 8 GPUs: memory drops roughly 8×, at the cost of gathering weights before each layer.</li><li>Turn FlashAttention off and raise the sequence length to 32k: activations explode, because the attention score matrices grow with s². FlashAttention never stores them. Then turn on activation checkpointing: store only each layer's input and recompute the rest in the backward pass, about 30% more compute for a large saving.</li><li>70B: tensor parallelism of 8 within a node (fast NVLink) plus pipeline stages across nodes is the classic recipe. Tensor parallelism across slow network links is a bad idea: it communicates inside every layer.</li></ul>"),
      table(["Strategy", "What's split", "Communication", "Use when"], [
        ["Data parallel (DDP)", "The batch", "All-reduce gradients once per step", "The model fits on one GPU"],
        ["ZeRO-1/2", "Optimizer states (and gradients) across data-parallel GPUs", "Same as DDP, roughly", "Optimizer state is the problem (always, with Adam)"],
        ["ZeRO-3 / FSDP", "Weights too", "Gather each layer's weights just in time (~1.5× DDP)", "Model doesn't fit; the easiest first step"],
        ["Tensor parallel", "Each matrix multiply", "Inside every layer (all-reduce of activations)", "Huge layers; GPUs connected by NVLink"],
        ["Pipeline parallel", "Layers into stages", "Activations between neighbouring stages", "Many layers; across nodes"],
        ["Sequence / context parallel", "The sequence", "Keys and values passed around a ring", "Very long contexts"],
        ["Expert parallel", "MoE experts", "All-to-all token routing", "Mixture-of-experts models"],
      ]),
    ),
    section("Pipelines have bubbles",
      prose("<p>Split the layers across GPUs and the first GPU must wait for work to flow back from the last before it can run backward passes. That idle time is the <b>bubble</b>. Splitting the batch into many micro-batches shrinks it. The 1F1B schedule interleaves forward and backward passes so each GPU holds fewer micro-batches' activations at once.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Pipeline stages (GPUs)", min: 2, max: 8, value: P, onInput: (v) => { P = v; drawPipe(); } }),
          slider({ label: "Micro-batches", min: 1, max: 16, value: M, onInput: (v) => { M = v; drawPipe(); } }),
          segmented({ label: "Schedule", options: [{ value: "gpipe", label: "GPipe (all forward, then all backward)" }, { value: "1f1b", label: "1F1B" }], value: sched, onChange: (v) => { sched = v; drawPipe(); } })),
        h("div.stage-body", pst.el, h("div", { style: { overflowX: "auto" } }, grid), h("p.canvas-cap", "Violet: forward pass. Orange: backward pass (twice as long). Grey: idle."))),
      predict({
        chapter: id,
        question: "With 8 pipeline stages, how many micro-batches do you need to keep the bubble under about 10%?",
        options: ["8", "About 64", "1", "It can't go below 50%"],
        answer: 1,
        explain: "<p>Bubble ≈ (P − 1)/(M + P − 1). With P = 8, M = 64 gives 7/71 ≈ 10%. That's why pipeline parallelism wants a large global batch, and why interleaved schedules (several smaller stages per GPU) and zero-bubble schedules were invented.</p>",
      }),
    ),
    section("At scale, failure is normal",
      stepper([
        { title: "Hardware fails constantly", html: "<p>Meta reported 466 job interruptions in 54 days while training Llama 3 405B on 16k GPUs, most from GPU and memory faults. At that scale something breaks every few hours.</p>" },
        { title: "Checkpoint often, restart fast", html: "<p>Save state frequently (asynchronously, so training doesn't pause), detect failures automatically, swap out bad nodes and resume. The time from failure to training again matters as much as raw speed.</p>" },
        { title: "Silent errors", html: "<p>Worse than crashes: a faulty GPU that returns slightly wrong numbers. Teams compare redundant computations or watch for loss anomalies to catch them.</p>" },
        { title: "Stragglers", html: "<p>In synchronous training, the slowest GPU sets the pace. Monitoring per-rank step times and replacing slow nodes keeps utilisation up.</p>" },
      ]),
    ),
    section("In code",
      code({
        FSDP: `import torch.distributed as dist
from torch.distributed.fsdp import fully_shard          # PyTorch 2.x "FSDP2"
dist.init_process_group("nccl")
model = Transformer(cfg).cuda()
for block in model.layers:
    fully_shard(block)                 # shard each block's params; gathered just in time
fully_shard(model)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
# launch with: torchrun --nproc_per_node=8 train.py`,
        "Activation checkpointing": `from torch.utils.checkpoint import checkpoint
class Block(nn.Module):
    def forward(self, x):
        return checkpoint(self._forward, x, use_reentrant=False)   # recompute in backward
    def _forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))`,
      }),
    ),
  );
  draw(); drawPipe();
}
