// Learning modes: the same course, cut different ways. A beginner can't take in 104 chapters at once,
// so each mode is an ordered route through a subset, grouped into stations, with the Colab projects that fit it.
import { ALL, PROJECTS, find } from "./catalog.js";
import { progress } from "./core.js";

export const MODES = [
  {
    id: "zero", name: "From zero", color: "var(--c3)", glyph: "seed", hours: 6,
    who: "You've never studied machine learning. Maybe you don't code yet.",
    promise: "Learn by answering easy questions, one small step at a time. No maths you haven't seen.",
    after: ["Explain what training a model means", "Say what a neural network and a transformer do", "Read the rest of the course without getting lost"],
    ask: true,
    stations: [
      { name: "Questions first", ask: ["learn", "numbers", "line", "wrong", "downhill"] },
      { name: "Brains made of numbers", ask: ["neuron", "xor", "blame", "memorise"] },
      { name: "Machines that talk", ask: ["words", "nextword", "attention", "transformer"] },
      { name: "Now see it move", chapters: ["why-learn", "gradient-descent", "perceptron", "next-token", "attention"] },
    ],
    projects: ["01_linear_regression", "50_zoo_terms_and_data"],
  },
  {
    id: "essentials", name: "Essentials", color: "var(--accent)", glyph: "star", hours: 22,
    who: "You want to work in AI. You want what the industry actually uses, nothing extra.",
    promise: "The core every ML engineer is expected to know: how models learn, how LLMs work, and how to ship and judge them.",
    after: ["Train, debug and evaluate a model", "Explain an LLM end to end in an interview", "Choose between prompting, RAG, fine-tuning and RL", "Estimate what a model costs to run"],
    stations: [
      { name: "How learning works", chapters: ["why-learn", "gradient-descent", "backprop", "generalization", "probability"] },
      { name: "Tools that win on real data", chapters: ["knn-trees", "training-tricks", "embeddings"] },
      { name: "Language models", chapters: ["tokenization", "next-token", "attention", "transformer", "scaling-laws"] },
      { name: "From model to assistant", chapters: ["sft", "rlhf-ppo", "grpo-dpo"] },
      { name: "Shipping it", chapters: ["inference", "precision", "evals", "which-tool", "cost-reality", "reality-gap"] },
    ],
    projects: ["02_micrograd", "23_trees_boosting", "06_gpt_from_scratch", "08_sft_lora", "44_eval_error_bars"],
  },
  {
    id: "builder", name: "Builder", color: "var(--c2)", glyph: "tool", hours: 14,
    who: "You build products with models: chat apps, agents, search over your own documents.",
    promise: "Enough of the inside to make good decisions, then everything about prompts, context, tools, evals and cost.",
    after: ["Design an agent loop that doesn't fall over", "Know when RAG beats fine-tuning", "Measure whether a change helped", "Cut latency and cost"],
    stations: [
      { name: "What the model really is", chapters: ["next-token", "tokenization", "attention", "hallucination"] },
      { name: "Steering it", chapters: ["which-tool", "context-engineering", "sft", "distillation"] },
      { name: "Agents", chapters: ["tool-use", "agent-loops", "reasoning", "thinking-budget"] },
      { name: "Knowing it works", chapters: ["evals", "error-bars", "inference", "cost-reality"] },
    ],
    projects: ["19_agent_ralph_loop", "21_speculative_decoding", "44_eval_error_bars", "45_thinking_budget"],
  },
  {
    id: "research", name: "Researcher", color: "var(--c6)", glyph: "flask", hours: 30,
    who: "You want to push the frontier, read papers critically and run your own experiments.",
    promise: "The laws of training, the architectures in play, what's inside trained models, and the open questions.",
    after: ["Design an experiment that can actually fail", "Read a training run like a lab does", "Know which claims in a paper to doubt", "Pick an open problem worth a year"],
    stations: [
      { name: "The laws", chapters: ["scaling-laws", "hidden-laws", "cooldown", "emergence", "data"] },
      { name: "Architectures in play", chapters: ["moe", "ssm-mamba", "looped", "ttt", "long-context", "diffusion"] },
      { name: "What training does", chapters: ["rl-narrows", "self-consumption", "converging-minds", "grpo-dpo"] },
      { name: "Looking inside", chapters: ["interpretability", "inside-view", "reversal", "hallucination"] },
      { name: "How to do research", chapters: ["research-method", "error-bars", "lab-experiments", "my-theories", "unwritten"] },
    ],
    projects: ["07_scaling_laws", "20_looped_transformer", "42_rl_narrows", "41_capstone_sae_gpt2", "66_logit_lens_steering", "58_conjecture_tests"],
  },
  {
    id: "silicon", name: "Hardware & systems", color: "var(--c4)", glyph: "chip", hours: 16,
    who: "You want to know why AI runs on GPUs, from electrons up to training on thousands of them.",
    promise: "Every layer from a transistor to CUDA to a cluster, each explained from the one below.",
    after: ["Explain why memory, not maths, limits most AI", "Read a roofline and fix a slow kernel", "Plan memory for training a big model"],
    stations: [
      { name: "Physics", chapters: ["ai-physics", "electrons", "transistor", "fab"] },
      { name: "Computers", chapters: ["logic", "memory-physics", "cpu", "power-wall"] },
      { name: "GPUs", chapters: ["gpu-history", "gpu-design", "cuda", "software-stack", "all-the-way-down"] },
      { name: "At scale", chapters: ["precision", "kernels", "distributed", "inference"] },
    ],
    projects: ["60_electrons_to_cuda", "13_triton_kernels", "34_gpu_kernels", "31_precision_memory"],
  },
  {
    id: "sprint", name: "One-week sprint", color: "var(--c5)", glyph: "bolt", hours: 7,
    who: "An interview or a new job is coming up and you have a week.",
    promise: "One hour a day: the twelve ideas you'll be asked about most, with a quiz each evening.",
    after: ["Answer the classic questions on transformers, training and RLHF", "Draw an LLM on a whiteboard"],
    stations: [
      { name: "Days 1–2", chapters: ["gradient-descent", "backprop", "generalization"] },
      { name: "Days 3–4", chapters: ["attention", "transformer", "next-token"] },
      { name: "Days 5–6", chapters: ["scaling-laws", "sft", "rlhf-ppo"] },
      { name: "Day 7", chapters: ["inference", "evals", "which-tool"] },
    ],
    projects: ["06_gpt_from_scratch"],
  },
  {
    id: "all", name: "Everything", color: "var(--ink-2)", glyph: "map", hours: 90,
    who: "You want the whole course, in order.",
    promise: "All levels, from the maths to the hidden curriculum.",
    after: ["Everything above, and the notes that aren't in any textbook"],
    stations: null,
    projects: [],
  },
];

const KEY = `${progress.key.split(":")[0]}:mode`;
export const getMode = () => { try { return MODES.find((m) => m.id === localStorage.getItem(KEY)) || null; } catch { return null; } };
export const setMode = (id) => { try { id ? localStorage.setItem(KEY, id) : localStorage.removeItem(KEY); } catch { /* private mode */ } };
export const modeById = (id) => MODES.find((m) => m.id === id);

/** Stations with chapter objects filled in; "Everything" uses the levels themselves. */
export function stationsOf(mode, levels) {
  if (!mode.stations) return levels.map((lv) => ({ name: lv.name.split(":")[0], chapters: lv.chapters.filter((c) => c.live).map((c) => find(c.id)) }));
  return mode.stations.map((st) => ({ ...st, chapters: (st.chapters || []).map(find).filter(Boolean) }));
}
export const chaptersOf = (mode, levels) => stationsOf(mode, levels).flatMap((s) => s.chapters);
export const projectsOf = (mode) => mode.projects.map((f) => PROJECTS.find((p) => p.file === f)).filter(Boolean);
export const inMode = (mode, id) => !mode || !mode.stations || mode.stations.some((s) => (s.chapters || []).includes(id));

/** Progress through a mode: chapters opened, plus Socratic threads finished for modes that use them. */
export function modeProgress(mode, levels, done, asked = {}) {
  const ch = chaptersOf(mode, levels);
  const threads = (mode.stations || []).flatMap((s) => s.ask || []);
  const total = ch.length + threads.length;
  const n = ch.filter((c) => done[c.id]?.visited).length + threads.filter((t) => asked[t]?.done).length;
  const next = threads.find((t) => !asked[t]?.done) ? { ask: threads.find((t) => !asked[t]?.done) } : ch.find((c) => !done[c.id]?.visited);
  return { n, total, next };
}

export const ALL_IDS = new Set(ALL.map((c) => c.id));
