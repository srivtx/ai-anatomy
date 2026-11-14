import { h } from "../lib/core.js";
import { segmented, stats, predict, callout, section, prose, table, toggle } from "../lib/widgets.js";

// verdicts: standard = in nearly every serious recipe; situational = used when it fits; faded = mostly abandoned
const IDEAS = [
  // ---- data ----
  ["Data", "More data and compute (scale)", "standard", "Loss falls as a smooth power law in parameters, data and compute. Nothing else has matched it as a reliable lever.", "Kaplan et al. 2020; Chinchilla 2022; every frontier model since"],
  ["Data", "Quality filtering with a learned classifier", "standard", "Not all tokens teach equally. A classifier trained to spot educational, well-written text lets the same compute see more useful signal.", "FineWeb-Edu (2024), DCLM (2024): large gains at equal tokens"],
  ["Data", "Deduplication", "standard", "Repeated documents waste compute and push the model to memorise rather than generalise. Near-duplicate removal (MinHash) is cheap.", "Lee et al. 2022; standard in every open data pipeline"],
  ["Data", "Annealing on the best data at the end", "standard", "The final, low-learning-rate phase has an outsized effect on what the model ends up good at, so it's where high-quality maths, code and instruction-like data goes.", "Llama 3, OLMo 2 and others report mid-training / annealing mixes"],
  ["Data", "Synthetic data from stronger models", "standard", "A strong model can write targeted, verified examples (maths, code, reasoning traces) faster than people. Works best when outputs are filtered or checked.", "Phi models, DeepSeek-R1 distillation, most post-training sets"],
  ["Data", "Easy-to-hard curriculum during pretraining", "situational", "Intuitive, but with huge diverse datasets, random ordering is hard to beat. The exception is the annealing phase above.", "Mixed results in careful studies; rarely the main story in model reports"],
  ["Data", "Many epochs over a small dataset", "faded", "Repeats are almost as good as fresh data for ~4 epochs, then returns collapse and memorisation grows.", "Muennighoff et al. 2023, “Scaling data-constrained LMs”"],
  // ---- architecture ----
  ["Architecture", "Decoder-only transformer", "standard", "One stack, one objective (next token), trivially parallel in training. Simplicity scaled better than encoder-decoder for general models.", "GPT series, Llama, Claude-class models"],
  ["Architecture", "Pre-norm with RMSNorm", "standard", "Normalising the input of each block keeps the residual stream clean, so gradients flow at depth without careful warmup. RMSNorm drops the mean term at no cost.", "Used by essentially all modern LLMs"],
  ["Architecture", "Rotary position embeddings (RoPE)", "standard", "Encodes relative position by rotating queries and keys, so attention depends on distance, and it can be stretched for longer contexts.", "Su et al. 2021; Llama, Qwen, Mistral, DeepSeek"],
  ["Architecture", "SwiGLU feed-forward", "standard", "A gated MLP: one branch decides how much of the other passes. Consistently a little better per FLOP than ReLU/GELU MLPs.", "Shazeer 2020; PaLM, Llama onwards"],
  ["Architecture", "Grouped-query attention", "standard", "Several query heads share one key/value head, shrinking the KV cache several times with little quality loss. Inference is memory-bound, so this matters.", "Ainslie et al. 2023; Llama 2 70B onwards"],
  ["Architecture", "Mixture of experts", "standard", "More parameters (knowledge) at the same compute per token. Costs memory and engineering, which frontier labs can pay.", "Mixtral, DeepSeek-V3, Qwen3, Llama 4"],
  ["Architecture", "Stability tricks: QK-norm, z-loss, logit soft-capping", "situational", "At large scale, attention logits and output logits can grow until training spikes. These clamp them. Small runs rarely need them.", "PaLM (z-loss), Gemma 2 (soft-capping), OLMo 2 (QK-norm)"],
  ["Architecture", "SSM / linear-attention hybrids", "situational", "Constant-memory layers make long contexts cheap; a few attention layers keep exact recall. Rising, especially for long-context and on-device models.", "Jamba, Nemotron-H, Qwen3-Next, MiniMax"],
  ["Architecture", "Multi-token prediction", "situational", "Predicting several future tokens gives a denser training signal and a built-in draft for speculative decoding.", "Gloeckle et al. 2024; used as an auxiliary loss in DeepSeek-V3"],
  ["Architecture", "Learned absolute position embeddings", "faded", "Positions beyond the training length are untrained vectors, so models can't extrapolate. Relative schemes (RoPE) replaced them.", "GPT-2 used them; few new LLMs do"],
  ["Architecture", "Post-norm residual blocks", "faded", "Normalising after the residual add makes deep stacks unstable without long warmup.", "Original 2017 Transformer; replaced by pre-norm"],
  ["Architecture", "Dropout during pretraining", "faded", "With trillions of tokens and single-epoch training, overfitting isn't the problem, so dropout just slows learning.", "Most modern LLM pretraining uses dropout 0"],
  ["Architecture", "Bias terms in linear layers", "faded", "They add little with normalisation in place and can hurt stability; removing them is free.", "PaLM, Llama: no biases"],
  ["Architecture", "Approximate efficient attention (Performer, Linformer, Reformer)", "faded", "Approximations lost quality, while FlashAttention made exact attention fast enough. The linear-attention idea returned later in better forms (gated / delta-rule).", "Tay et al. 2022 survey; FlashAttention 2022"],
  ["Architecture", "Encoder-decoder for general chat models", "faded", "Two stacks and two objectives didn't beat one decoder at scale for open-ended generation. Still strong for translation and speech (Whisper).", "T5 and BART era vs decoder-only dominance"],
  // ---- optimisation ----
  ["Optimisation", "AdamW", "standard", "Per-parameter step sizes cope with wildly different gradient scales across layers; decoupled weight decay regularises cleanly.", "Loshchilov & Hutter 2019; the default nearly everywhere"],
  ["Optimisation", "Warmup, then cosine or warmup-stable-decay", "standard", "Early on, Adam's statistics are unreliable, so start small. Decaying the LR at the end lets the model settle into a lower loss.", "Universal; WSD (MiniCPM, DeepSeek) allows continuing training"],
  ["Optimisation", "Gradient clipping", "standard", "Caps rare huge updates from outlier batches that could knock training off course.", "Standard at norm ~1.0"],
  ["Optimisation", "bf16 mixed precision", "standard", "Half the memory and much faster matmuls; bf16 keeps fp32's range so no loss scaling is needed.", "Universal on modern accelerators"],
  ["Optimisation", "fp8 training", "situational", "Another ~2× in matmul throughput, but needs careful scaling to avoid overflow. Proven at scale by a few labs.", "DeepSeek-V3 trained mainly in fp8"],
  ["Optimisation", "Hyperparameter transfer (μP)", "situational", "Parameterise so the best LR at small width stays best at large width, so you can tune cheaply. Used by labs that train many sizes.", "Yang et al. 2022; Cerebras-GPT, some frontier labs"],
  ["Optimisation", "Matrix-aware optimizers (Muon, Shampoo, SOAP)", "situational", "Update weight matrices as matrices (orthogonalised or preconditioned) rather than element by element. Real speedups reported; still being validated at scale.", "Muon used for Kimi K2 (2025); Shampoo won the AlgoPerf benchmark (2024)"],
  ["Optimisation", "Most new optimizers from papers", "faded", "Many reported wins came from under-tuned baselines. In careful comparisons with well-tuned AdamW, gains shrink, often to near zero at scale.", "Kaddour et al. 2023 “No train no gain”; Wen et al. 2025"],
  ["Optimisation", "Large-batch optimizers (LARS, LAMB)", "faded", "Helped push batch sizes in the BERT/ResNet era; LLM recipes instead use AdamW with tuned batch sizes and warmup.", "Rare in recent LLM reports"],
  // ---- post-training ----
  ["Post-training", "SFT on a small, high-quality set", "standard", "Pretraining already holds the ability; SFT mostly teaches format and behaviour, so quality matters more than quantity.", "LIMA 2023; every chat model"],
  ["Post-training", "Preference optimisation (DPO and variants)", "standard", "Turns preference pairs into a simple supervised loss: no reward model, no sampling loop. Easy to run, stable.", "Rafailov et al. 2023; Zephyr, Tülu 3, many open models"],
  ["Post-training", "RL with verifiable rewards (GRPO, PPO)", "standard", "When correctness can be checked (maths, code, tests), RL can improve far beyond the demonstrations, and long reasoning emerges.", "DeepSeek-R1 2025; reasoning models across labs"],
  ["Post-training", "RLHF with a learned reward model", "standard", "Captures fuzzy qualities (helpfulness, tone) that no rule can check. Needs care: reward hacking and length bias.", "InstructGPT 2022; still part of frontier recipes in some form"],
  ["Post-training", "Distillation from a stronger model", "standard", "The cheapest way to make a strong small model: train it on a big model's outputs or logits.", "Gemma, Llama 3.2 small models, DeepSeek-R1 distills"],
  ["Post-training", "LoRA / QLoRA", "standard", "Fine-tuning updates are low-rank in practice, so training two thin matrices per layer gets most of the benefit at a fraction of the memory.", "Hu et al. 2021; Dettmers et al. 2023"],
  ["Post-training", "Model merging", "situational", "Averaging fine-tunes of the same base can combine skills for free. Fails across different bases.", "Model soups, TIES; popular in open-model leaderboards"],
  ["Post-training", "Knowledge editing (ROME, MEMIT)", "situational", "Surgically changing one fact works in demos, but edits often don't generalise to paraphrases or break nearby facts.", "Research tool more than production practice"],
  ["Post-training", "Prompt tuning / prefix tuning", "faded", "Learning soft prompt vectors was cheap but lagged LoRA in quality and robustness, especially on smaller models.", "Largely replaced by LoRA"],
  ["Post-training", "Fine-tuning to teach new facts", "faded", "Models learn new facts slowly from fine-tuning and can hallucinate more around them. Retrieval adds knowledge reliably and can be updated.", "Ovadia et al. 2023; Gekhman et al. 2024"],
  // ---- inference ----
  ["Inference", "FlashAttention", "standard", "Exact attention computed in tiles that stay in fast on-chip memory: same maths, far less memory traffic.", "Dao et al. 2022; in every framework"],
  ["Inference", "Paged KV cache + continuous batching", "standard", "Pack many users' requests into one batch, add and drop them every step, and allocate KV memory in pages. Multiplies throughput.", "vLLM 2023; all major serving engines"],
  ["Inference", "Weight quantisation (8-bit, 4-bit)", "standard", "Decoding is memory-bound: fewer bytes per weight means proportionally faster tokens and smaller GPUs.", "GPTQ, AWQ, bitsandbytes, llama.cpp"],
  ["Inference", "Speculative decoding", "standard", "A small draft guesses several tokens; the big model checks them in one pass. Identical outputs, lower latency.", "Leviathan et al. 2023; Medusa, EAGLE"],
  ["Inference", "Prefix (prompt) caching", "standard", "Reuse the KV cache for shared prefixes like system prompts and chat history instead of recomputing them.", "Offered by major APIs and serving engines"],
  ["Inference", "Unstructured pruning for speed", "faded", "Removing scattered weights keeps accuracy but rarely speeds anything up, because GPUs need regular structure.", "Real speedups need structured or 2:4 sparsity"],
  // ---- prompting and agents ----
  ["Using models", "Retrieval-augmented generation", "standard", "Puts the right facts in context at query time: fresh, citable, and no retraining.", "Lewis et al. 2020; nearly every enterprise deployment"],
  ["Using models", "Simple tool loops with verification", "standard", "A model calling tools in a loop, with tests or checks it can run, handles long tasks far better than one-shot answers.", "Coding agents, research agents"],
  ["Using models", "Few-shot examples in the prompt", "situational", "Still the clearest way to show an exact output format or style; less needed for instruction-following than it was in 2020.", "GPT-3 paper; modern instruction-tuned models need fewer"],
  ["Using models", "Multi-agent debate and role-play swarms", "situational", "Sometimes helps, but often matched by one strong agent given the same compute (more samples or more thinking).", "Mixed results when compute is controlled"],
  ["Using models", "“Let's think step by step” prompting", "faded", "Reasoning models think by default, so the magic phrase adds little; asking clearly and giving room to think matters more.", "Kojima et al. 2022, before reasoning models"],
  ["Using models", "Elaborate persona prompts (“You are a world-class expert…”)", "faded", "Modern models don't get more capable by being told they're experts; precise instructions and context help far more.", "Zheng et al. 2023 found no reliable gains from personas"],
];

const VERDICT = { standard: ["Standard", "var(--good)"], situational: ["Situational", "var(--warn)"], faded: ["Faded", "var(--ink-3)"] };

export default function (root, { id }) {
  let area = "All", guess = true;
  const guesses = new Map();
  const grid = h("div.idea-grid");
  const gst = stats([{ label: "Guessed" }, { label: "Right" }, { label: "Your accuracy" }]);
  const areas = ["All", ...new Set(IDEAS.map((x) => x[0]))];
  function card(idea, i) {
    const [ar, name, v, why, ev] = idea, g = guesses.get(i), shown = !guess || g;
    const el = h("article.idea", { style: { "--vc": shown ? VERDICT[v][1] : "var(--line)" } },
      h("div.idea-top", h("span.idea-area", ar), shown ? h("span.idea-pill", { style: { color: VERDICT[v][1] } }, VERDICT[v][0]) : null),
      h("h3", name),
      shown
        ? [g && h("div.idea-guess", { style: { color: g === v ? "var(--good)" : "var(--bad)" } }, g === v ? "✓ You called it" : `✗ You guessed ${VERDICT[g][0].toLowerCase()}`), h("p", why), h("p.idea-ev", ev)]
        : h("div.idea-btns", Object.entries(VERDICT).map(([k, [lab]]) => h("button.btn", { type: "button", onclick: () => { guesses.set(i, k); draw(); } }, lab))));
    return el;
  }
  function draw() {
    grid.replaceChildren(...IDEAS.map((x, i) => [x, i]).filter(([x]) => area === "All" || x[0] === area).map(([x, i]) => card(x, i)));
    const n = guesses.size, ok = [...guesses].filter(([i, g]) => IDEAS[i][2] === g).length;
    gst.set(0, `${n} of ${IDEAS.length}`); gst.set(1, String(ok)); gst.set(2, n ? `${Math.round((ok / n) * 100)}%` : "–", n >= 5 ? (ok / n > 0.7 ? "good" : ok / n < 0.45 ? "bad" : "warn") : "");
  }
  const count = (v) => IDEAS.filter((x) => x[2] === v).length;

  root.append(
    section(null, prose(`<p>Here are ${IDEAS.length} ideas that were each, at some point, exciting. Some are now in every serious recipe. Some are used only when they fit. Some have quietly faded. For each one you get a first-principles reason it worked or didn't, and the evidence. <b>Guess first.</b> Your misses are where your intuition needs updating.</p>`)),
    section("Standard, situational or faded?",
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Area", options: areas.map((a) => ({ value: a, label: a })), value: area, onChange: (v) => { area = v; draw(); } }),
          toggle({ label: "Guess mode (hide verdicts)", checked: true, onChange: (v) => { guess = v; draw(); } })),
        h("div.stage-body", gst.el, grid)),
      h("p.canvas-cap", `${count("standard")} standard · ${count("situational")} situational · ${count("faded")} faded. Verdicts reflect public recipes and reports up to 2025; the situational column is where things move.`),
    ),
    section("The patterns behind the verdicts",
      table(["What survived tends to…", "What faded tends to…"], [
        ["Help more as scale grows (MoE, data quality, RL with verifiers)", "Help mainly at small scale (dropout, many regularisers)"],
        ["Remove a bottleneck the hardware actually has (FlashAttention, GQA, quantisation: memory traffic)", "Save FLOPs the hardware doesn't care about (unstructured sparsity, approximate attention)"],
        ["Be simple and composable with everything else (RMSNorm, RoPE, LoRA)", "Need careful tuning or special infrastructure for a small gain"],
        ["Win against well-tuned baselines, reproduced by other groups", "Win against the paper's own baseline only"],
        ["Add information or signal (better data, verifiers, retrieval)", "Rearrange the same information (prompt incantations, personas)"],
      ]),
      callout("note", "Where the gains actually come from", "<p>Ask practitioners who've trained LLMs where their improvements came from and the answers are remarkably consistent. Roughly in order: <b>(1)</b> more and better data, <b>(2)</b> enough compute used stably (learning rate, schedule, no loss spikes, no bugs), <b>(3)</b> post-training: SFT data quality and RL with good rewards, <b>(4)</b> efficiency engineering that lets you afford more of 1–3, and only then <b>(5)</b> architecture details. Architecture choices matter a lot for <i>cost</i> (MoE, GQA, hybrids), much less for quality at fixed compute.</p>"),
      predict({
        chapter: id,
        question: "You have one month and a fixed GPU budget to improve your 1B-parameter model. Which bet usually pays off most?",
        options: ["Design a new attention mechanism", "Improve the data: filter for quality, deduplicate, fix the mix, add a high-quality annealing set", "Switch to the newest optimizer from a preprint", "Add dropout"],
        answer: 1,
        explain: "<p>Data work is the most reliable lever at fixed compute, and it compounds: better data helps every later model too. A new attention mechanism is a research project with uncertain payoff, and preprint optimizers often shrink against a tuned AdamW baseline.</p>",
      }),
    ),
  );
  draw();
}
