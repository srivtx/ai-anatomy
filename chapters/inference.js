import { h, rng } from "../lib/core.js";
import { goal, whyChain, sticker } from "../lib/why.js";
import { walk } from "../lib/walks.js";
import { slider, segmented, stats, predict, code, callout, section, prose, table, button } from "../lib/widgets.js";
import { xyChart, fmtBytes } from "../lib/plot.js";

export default function (root, { id }) {
  // ---------- 1. speculative decoding ----------
  const sp = { a: 0.75, g: 4, c: 0.05 };
  const sc = xyChart({ width: 460, height: 260, xLabel: "draft tokens per step (γ)", yLabel: "speed-up over normal decoding", xDomain: [1, 12] });
  const sst = stats([{ label: "Tokens per big-model pass" }, { label: "Speed-up" }]);
  const tokEl = h("div.spec");
  let sr = rng(3);
  // Leviathan et al. (2023): expected tokens per target pass = (1 − α^(γ+1)) / (1 − α); cost = 1 + γ·c
  const E = (a, g) => (a >= 1 ? g + 1 : (1 - a ** (g + 1)) / (1 - a)), speed = (a, g, c) => E(a, g) / (1 + g * c);
  function drawSpec() {
    const pts = Array.from({ length: 12 }, (_, i) => [i + 1, speed(sp.a, i + 1, sp.c)]);
    sc.update([{ name: "Speed-up", color: "var(--accent)", points: pts, dots: true, r: 3.5 }, { name: "No speed-up", color: "var(--ink-3)", points: [[1, 1], [12, 1]], dash: true }], { xDomain: [1, 12], marks: [{ x: sp.g, label: "γ" }] });
    sst.set(0, E(sp.a, sp.g).toFixed(2)); const s = speed(sp.a, sp.g, sp.c); sst.set(1, `${s.toFixed(2)}×`, s > 1.5 ? "good" : s < 1 ? "bad" : "");
  }
  function demo() {
    sr = rng(Math.floor(Math.random() * 1e6));
    const rows = [];
    for (let step = 0; step < 6; step++) {
      const cells = []; let k = 0;
      for (; k < sp.g; k++) { if (sr() < sp.a) cells.push(["ok", "draft ✓"]); else { cells.push(["no", "draft ✗"]); break; } }
      cells.push(["fix", k === sp.g ? "bonus" : "fixed"]);
      rows.push(h("div.spec-row", h("span.spec-l", `pass ${step + 1}`), ...cells.map(([c, t]) => h("span", { class: `spec-t ${c}`, title: t }, c === "ok" ? "✓" : c === "no" ? "✗" : "+")), h("span.spec-n", ((n) => `${n} token${n === 1 ? "" : "s"}`)(cells.filter((x) => x[0] !== "no").length))));
    }
    tokEl.replaceChildren(...rows);
  }

  // ---------- 2. static vs continuous batching ----------
  let mode = "continuous", load = 0.7;
  const grid = h("div.pipe");
  const bst = stats([{ label: "Slot utilisation" }, { label: "Requests finished" }, { label: "Average wait before starting" }]);
  function drawBatch() {
    const r = rng(7), S = 6, T = 60, reqs = [];
    for (let t = 0, i = 0; t < T; t++) if (r() < load) reqs.push({ id: i++, arrive: t, len: 3 + Math.floor(r() ** 2 * 18) });
    const rows = Array.from({ length: S }, () => new Array(T).fill(null));
    const queue = [...reqs]; let done = 0, waitSum = 0, started = 0;
    if (mode === "static") {
      let t = 0;
      while (t < T) {
        const batch = queue.filter((q) => q.arrive <= t).slice(0, S);
        if (!batch.length) { t++; continue; }
        batch.forEach((q) => queue.splice(queue.indexOf(q), 1));
        const L = Math.max(...batch.map((q) => q.len));
        batch.forEach((q, s) => { waitSum += t - q.arrive; started++; for (let k = 0; k < q.len && t + k < T; k++) rows[s][t + k] = q.id; if (t + q.len <= T) done++; });
        t += L; // the whole batch waits for its longest request
      }
    } else {
      const free = new Array(S).fill(0);
      for (let t = 0; t < T; t++) for (let s = 0; s < S; s++) {
        if (free[s] > t) continue;
        const qi = queue.findIndex((q) => q.arrive <= t); if (qi < 0) continue;
        const q = queue.splice(qi, 1)[0]; waitSum += t - q.arrive; started++;
        for (let k = 0; k < q.len && t + k < T; k++) rows[s][t + k] = q.id;
        free[s] = t + q.len; if (t + q.len <= T) done++;
      }
    }
    const used = rows.flat().filter((x) => x !== null).length;
    grid.style.gridTemplateColumns = `56px repeat(${T}, minmax(6px, 1fr))`;
    grid.replaceChildren(...rows.flatMap((row, s) => [h("span.pipe-l", `slot ${s + 1}`), ...row.map((q) => h("span.pipe-c", { style: { height: "20px", background: q === null ? "var(--sunken)" : `hsl(${(q * 47) % 360} 55% 58%)` }, title: q === null ? "idle" : `request ${q + 1}` }))]));
    bst.set(0, `${Math.round((used / (S * T)) * 100)}%`, used / (S * T) > 0.75 ? "good" : "warn"); bst.set(1, String(done)); bst.set(2, started ? `${(waitSum / started).toFixed(1)} steps` : "–");
  }

  // ---------- 3. KV cache memory ----------
  const kv = { layers: 32, kvh: 8, hd: 128, ctx: 8192, users: 64, bytes: 2, used: 0.3 };
  const kst = stats([{ label: "KV cache per token" }, { label: "Per user at full context" }, { label: "All users, reserved up front" }, { label: "All users, paged (actual use)" }]);
  function drawKV() {
    const perTok = 2 * kv.layers * kv.kvh * kv.hd * kv.bytes, perUser = perTok * kv.ctx;
    kst.set(0, fmtBytes(perTok)); kst.set(1, fmtBytes(perUser)); kst.set(2, fmtBytes(perUser * kv.users), "bad"); kst.set(3, fmtBytes(perUser * kv.users * kv.used), "good");
  }

  root.append(
    section(null, goal("<p>A trained model sits on a GPU. We want it to answer <b>many users, fast and cheaply</b>. But generation is one token at a time, and each token needs every weight read from memory.</p>", ["don't recompute the past: cache it", "share each weight read across many users", "guess ahead when you can"]),
      prose(`<p>Training happens once; inference happens billions of times. Generating text has two phases with opposite characters. <b>Prefill</b> processes the whole prompt in parallel: big matrix multiplies, compute-bound. <b>Decode</b> produces one token at a time, rereading all the weights and the KV cache for every token: memory-bound. Nearly every inference trick either makes decode do more useful work per byte read (batching, speculative decoding) or reads fewer bytes (quantisation, smaller KV caches).</p>`),
      h("div.vx-row", sticker("kvcache", { title: "The KV cache", caption: "every token leaves a key (blue) and value (orange) in every layer; the newest is lit", px: 4 }))),
    section("Speculative decoding",
      prose("<p>A small, fast <b>draft</b> model guesses the next γ tokens. The big model checks all of them in <b>one</b> forward pass (checking γ tokens costs about the same as generating one, because decode is memory-bound). It keeps the longest prefix it agrees with, plus one token of its own. A clever acceptance rule makes the output distribution <b>exactly</b> the big model's.</p>"),
      h("div.lab",
        h("div.stage",
          h("div.stage-controls",
            slider({ label: "Acceptance rate α", min: 0.3, max: 0.95, step: 0.05, value: sp.a, fmt: (v) => `${Math.round(v * 100)}%`, onInput: (v) => { sp.a = v; drawSpec(); } }),
            slider({ label: "Draft tokens γ", min: 1, max: 12, value: sp.g, onInput: (v) => { sp.g = v; drawSpec(); } }),
            slider({ label: "Draft cost (vs big model)", min: 0, max: 0.5, step: 0.01, value: sp.c, fmt: (v) => `${Math.round(v * 100)}%`, onInput: (v) => { sp.c = v; drawSpec(); } }),
            button("Simulate 6 passes", demo, "primary")),
          h("div.stage-body", sst.el, tokEl)),
        sc.el),
      callout("tip", "Things to try", "<ul style='margin:0;padding-left:18px'><li>With 75% acceptance and a cheap draft, ~3 tokens per big-model pass: about 2.5× faster. Easy, predictable text (code, boilerplate) has high acceptance; creative text lower.</li><li>Make the draft expensive (30% of the big model): long drafts stop paying off. There's an optimal γ, usually 3–8.</li><li>No draft model at all? Medusa and EAGLE add small prediction heads to the big model itself; prompt lookup copies n-grams from the input.</li></ul>"),
    ),
    section("Continuous batching",
      prose("<p>Requests finish at different times. <b>Static</b> batching waits until every request in a batch is done before starting new ones, so slots sit idle behind the longest answer. <b>Continuous</b> (in-flight) batching refills a slot the moment it frees up, at every decode step (Orca, 2022; vLLM). Each colour is one request.</p>"),
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Scheduler", options: [{ value: "static", label: "Static batching" }, { value: "continuous", label: "Continuous batching" }], value: mode, onChange: (v) => { mode = v; drawBatch(); } }),
          slider({ label: "Arrival rate", min: 0.2, max: 1.5, step: 0.05, value: load, fmt: (v) => `${v.toFixed(2)} per step`, onInput: (v) => { load = v; drawBatch(); } })),
        h("div.stage-body", bst.el, h("div", { style: { overflowX: "auto" } }, grid))),
    ),
    section("The KV cache is the memory bill",
      prose("<p>For every token in context, every layer stores a key and a value vector for every KV head. With many users and long contexts, the KV cache, not the weights, fills the GPU. Reserving each user's maximum context up front wastes most of it; <b>PagedAttention</b> allocates in small pages as the sequence grows, like virtual memory.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Layers", min: 12, max: 128, step: 4, value: kv.layers, onInput: (v) => { kv.layers = v; drawKV(); } }),
          segmented({ label: "KV heads (GQA)", options: [1, 2, 4, 8, 32, 64].map((v) => ({ value: v, label: String(v) })), value: kv.kvh, onChange: (v) => { kv.kvh = v; drawKV(); } }),
          slider({ label: "Max context", min: 10, max: 17, step: 1, value: 13, fmt: (v) => (2 ** v).toLocaleString(), onInput: (v) => { kv.ctx = 2 ** v; drawKV(); } }),
          slider({ label: "Concurrent users", min: 1, max: 256, value: kv.users, onInput: (v) => { kv.users = v; drawKV(); } }),
          segmented({ label: "KV precision", options: [{ value: 2, label: "bf16" }, { value: 1, label: "fp8" }], value: 2, onChange: (v) => { kv.bytes = v; drawKV(); } }),
          slider({ label: "Average share of max context actually used", min: 0.05, max: 1, step: 0.05, value: kv.used, fmt: (v) => `${Math.round(v * 100)}%`, onInput: (v) => { kv.used = v; drawKV(); } })),
        h("div.stage-body", kst.el)),
      predict({
        chapter: id,
        question: "Switching a model from 32 KV heads to 8 (grouped-query attention) does what to the KV cache?",
        options: ["Nothing", "Shrinks it 4×, so 4× more users or 4× longer contexts fit", "Doubles it", "Only affects training"],
        answer: 1,
        explain: "<p>KV memory is proportional to the number of KV heads. GQA keeps all query heads but shares K and V across groups, which barely hurts quality. Multi-head latent attention (DeepSeek) compresses further. Set the heads to 32 above and compare.</p>",
      }),
    ),
    section("The inference toolbox",
      table(["Technique", "Attacks", "Typical gain"], [
        ["Continuous batching + paged KV cache", "Idle slots, fragmented memory", "Several × throughput"],
        ["Weight quantisation (int8/int4, fp8)", "Bytes per weight", "~2–4× decode speed and memory"],
        ["KV cache quantisation (fp8/int4)", "KV bytes", "~2× more context or users"],
        ["Speculative decoding", "Memory-bound decode", "2–3× lower latency, identical outputs"],
        ["Prefix caching", "Recomputing shared prompts", "Big savings for long system prompts and chats"],
        ["Chunked prefill", "Long prompts stalling decodes", "Smoother latency under load"],
        ["Disaggregated prefill/decode", "Two phases with different needs", "Better utilisation at scale"],
        ["Tensor parallelism at inference", "Latency for huge models", "Lower latency, at communication cost"],
      ]),
    ),
    walk("inference"),
    section("Why one user wastes a GPU",
      whyChain({
        claim: "Generating tokens for one user leaves a GPU's arithmetic more than 99% idle.",
        steps: [
          { q: "Why?", a: "<p>Each token needs every weight once, about 2 FLOPs per weight. A 70B model in 16-bit is 140 GB. An H100 reads about 3.35 TB/s, so reading the weights takes ~42 ms, while the 140 GFLOPs of maths would take ~0.14 ms.</p>" },
          { q: "So how do servers fix it?", a: "<p>Batching: read each weight once and use it for many users' tokens. 64 users share the same 42 ms of reading, for nearly 64× the throughput at almost the same latency. Continuous batching keeps the batch full as users come and go.</p>" },
          { q: "Why does memory still run out?", a: "<p>Each user's KV cache grows with every token. For a 70B model with 80 layers and 8 KV heads, that's ~0.33 MB per token, so a 4,000-token chat holds ~1.3 GB. Sixty-four of those is ~84 GB of cache alone.</p>" },
        ],
        axiom: { name: "Moving data costs more than computing on it", html: "<p>A multiply-add on chip costs around a picojoule; fetching its operand from off-chip memory costs ~100× more, and bandwidth grows more slowly than compute with each generation. Any job that does few operations per byte, like decoding, is limited by memory, not arithmetic. (<a href='/learn?c=memory-physics'>Why memory is far away</a>.)</p>" },
      })),
    section("In code",
      code({
        vLLM: `from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", enable_prefix_caching=True, gpu_memory_utilization=0.9)
outs = llm.generate(prompts, SamplingParams(temperature=0.7, max_tokens=512))
# continuous batching and paged attention are on by default`,
        "Speculative verification": `def speculative_step(target, draft, ctx, gamma=4):
    proposal, q = draft.sample(ctx, n=gamma)               # tokens and draft probabilities
    p = target.probs(ctx + proposal)                        # ONE pass scores every position
    for i, tok in enumerate(proposal):
        if random.random() < min(1, p[i][tok] / q[i][tok]):
            ctx.append(tok)                                 # accepted
        else:
            residual = (p[i] - q[i]).clamp(min=0)           # resample from what the draft under-weighted
            ctx.append(torch.multinomial(residual / residual.sum(), 1).item())
            return ctx
    ctx.append(torch.multinomial(p[gamma], 1).item())      # all accepted: one bonus token
    return ctx`,
      }, "Project 21 implements speculative decoding with two GPT-2 sizes and checks the outputs match."),
    ),
  );
  drawSpec(); demo(); drawBatch(); drawKV();
}
