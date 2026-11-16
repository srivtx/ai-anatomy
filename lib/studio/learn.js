// Levels (learn an architecture by building it one part at a time) and challenges (reach a goal, checked by
// the predictor). A step's check gets the design c, the predictions p and the session s:
// { sel: name of the selected part, path, tab, flags: Set, tok: tokens per parameter }.
import { PRESETS, full, count, blk } from "./arch.js";

const tiny = { ...PRESETS.tiny, name: "Your model" };
const all = (c, f) => Array.from({ length: c.L }, (_, i) => blk(c, i + 1)).every(f);
const visited = (s, re) => [...s.flags].some((f) => f.startsWith("at:") && re.test(f.slice(3)));

export const LEVELS = [
  {
    id: "lookup", title: "A lookup table", start: { ...tiny, L: 1, blocks: { 1: { attn: "none", mlp: false } } },
    neuro: "This is the smallest language model there is. Each character becomes a vector (the embedding), and the vector is turned straight back into scores for the next character (the unembedding). Nothing in between.",
    steps: [
      { do: "Click the blue Embedding layer and read what it is", ok: (c, p, s) => /embedding/i.test(s.sel) && !/un/i.test(s.sel) },
      { do: "Click the teal Unembedding layer", ok: (c, p, s) => /unembedding/i.test(s.sel) },
      { do: "Open the empty block: it's a dark plate because it does nothing yet", ok: (c, p, s) => visited(s, /^b1$/) },
    ],
    can: "Predict the next character from the current one only: after “q” comes “u”. That's a bigram model, about 2.5 loss on Shakespeare.",
    sample: "Thy ou s mbes ce ar hithen d.\nKESCowhe fer, fo the s",
    servo: "Double-click a layer to go inside it. The arrow keys turn the model, Escape comes back out.",
  },
  {
    id: "attend", title: "Let characters look back", start: { ...tiny, L: 1, blocks: { 1: { attn: "none", mlp: false } } },
    neuro: "To guess well you need the characters before this one, not just this one. Attention lets each position ask a question (query) and pull in what earlier positions offer (values), weighted by how well their keys match.",
    steps: [
      { do: "Select block 1 and switch its attention to Full", ok: (c) => blk(c, 1).attn !== "none" },
      { do: "Open block 1 › Attention and look at the heads", ok: (c, p, s) => visited(s, /^b1\/attn/) },
      { do: "Open one head to see who looks at whom", ok: (c, p, s) => visited(s, /^b1\/attn\/h\d+/) },
    ],
    can: "Use the last few characters: it spells common words and gets spaces and capitals mostly right.",
    sample: "The wis the mere, and the lo the sor of ther ham",
    servo: "Each head's grid is causal: a character can only look left, at what came before it. The future is hidden during training too.",
  },
  {
    id: "mlp", title: "Room to think: the MLP", start: { ...tiny, L: 1, blocks: { 1: { attn: "full", mlp: false } } },
    neuro: "Attention moves information around but mostly averages it. The MLP takes each position on its own, widens it 4×, bends it with a nonlinearity and narrows it back. That's where patterns get stored.",
    steps: [
      { do: "Turn block 1's MLP on", ok: (c) => blk(c, 1).mlp },
      { do: "Open the MLP and find its width", ok: (c, p, s) => visited(s, /^b1\/mlp/) },
    ],
    can: "Whole words, names in capitals before a colon, the rhythm of lines. Loss around 1.9.",
    sample: "KING HENRY:\nWhat shall the come to me the dears",
    servo: "A 4× wider MLP has 8 × width² weights: twice as many as attention's 4 × width².",
  },
  {
    id: "stack", title: "Stack the blocks", start: { ...tiny, L: 1 },
    neuro: "One block can only combine things one step apart. With two, a head can find “what followed this the last time” and copy it: an induction head. Each block builds on the last one's work.",
    steps: [
      { do: "Make it 4 blocks (Design tab, or the slider)", ok: (c) => c.L >= 4 && all(c, (b) => b.attn !== "none" && b.mlp) },
      { do: "Open the Predict tab and read the parameter count", ok: (c, p, s) => s.flags.has("tab:predict") },
    ],
    can: "Copies names and phrases it saw earlier in the text, keeps a speaker's style. Loss near 1.6.",
    sample: "ROMEO:\nI will not stand the world to see thee so,\nAnd th",
    servo: "New blocks copy the standard settings. Set any one of them back to empty to see the dark plate again.",
  },
  {
    id: "residual", title: "The highway", start: { ...tiny, L: 4 },
    neuro: "Every block adds to a shared running total, the residual stream. Why add instead of replace? During training the error signal flows backwards through every layer. Without the add it gets multiplied by every layer's weights and fades away.",
    steps: [
      { do: "Make it 12 blocks", ok: (c) => c.L >= 12 },
      { do: "Click the grey residual rail inside a block and switch it off. Watch the Predict badge", ok: (c) => !c.residual },
      { do: "Read why on the Predict tab", ok: (c, p, s) => !c.residual && s.flags.has("tab:predict") },
      { do: "Switch it back on", ok: (c) => c.residual },
    ],
    can: "Nothing new, but now you know why it can be deep at all.",
    servo: "The residual switch lives on the grey rails and add layers when you open a block.",
  },
  {
    id: "norm", title: "Keep the numbers steady", start: { ...tiny, L: 12 },
    neuro: "Each block adds to the stream, so it keeps growing. A norm rescales the input of every part so each part always sees numbers of the same size. Put it before each part (pre-norm) and it stays stable at any depth.",
    steps: [
      { do: "Click a yellow norm layer and try “After the add” (the 2017 original)", ok: (c) => c.normPos === "post" },
      { do: "Read the warning, then put it back before each part", ok: (c) => c.normPos === "pre" },
      { do: "Switch to RMSNorm: as good, a little cheaper", ok: (c) => c.norm === "RMSNorm" },
    ],
    can: "Train at 12 blocks without blowing up.",
    servo: "Watch the 3D block reorder when you move the norm: the order of the layers is real.",
  },
  {
    id: "scale", title: "Grow it into GPT-2", start: { ...tiny, L: 12, norm: "RMSNorm" },
    neuro: "GPT-2 small is this same skeleton, bigger: width 768, 12 blocks, 12 heads, a 50,257-token vocabulary and 1,024 positions. Size is the main dial; the scaling law on the Predict tab says how loss falls as it grows.",
    steps: [
      { do: "Design tab: width 768, 12 heads, 12 blocks", ok: (c) => c.d === 768 && c.heads === 12 && c.L === 12 },
      { do: "Vocabulary 50,257 and context 1,024", ok: (c) => c.vocab === 50257 && c.ctx === 1024 },
      { do: "Check it: about 124M parameters", ok: (c) => Math.abs(count(c).total - 124.4e6) < 2e6 },
    ],
    can: "Fluent English paragraphs, when trained on 10B+ tokens of web text.",
    servo: "If the count is off, check the norm (LayerNorm has 2 numbers per width, RMSNorm 1) and whether the readout is tied.",
  },
  {
    id: "modern", title: "The modern recipe", start: { ...PRESETS.gpt2, norm: "RMSNorm" },
    neuro: "Since 2023 almost every open model swapped four parts: RMSNorm, a SwiGLU MLP, RoPE positions and grouped K/V. Each fixes something specific: speed, quality per parameter, long text, and memory while serving.",
    steps: [
      { do: "MLP: SwiGLU", ok: (c) => c.act === "SwiGLU" },
      { do: "Positions: RoPE", ok: (c) => c.pos === "RoPE" },
      { do: "Attention: fewer K/V groups than heads", ok: (c) => c.kv < c.heads },
      { do: "Speed tab: see what the K/V change bought you", ok: (c, p, s) => s.flags.has("tab:speed") },
    ],
    can: "The same quality for less memory, and text longer than it was trained on with RoPE scaling.",
    servo: "Llama narrows the SwiGLU MLP to about 3.5× (8/3 for the same size) because the gate adds a third matrix.",
  },
  {
    id: "test", title: "Test it for real", start: { ...PRESETS.gpt2, norm: "RMSNorm", act: "SwiGLU", ff: 2048, pos: "RoPE", kv: 4 },
    neuro: "Everything so far was predicted from fundamentals. Some of it is exact (parameter counts), some is a rule of thumb. The only judge is a training run. The Test tab writes one for you.",
    steps: [
      { do: "Open the Test tab and read the prediction", ok: (c, p, s) => s.flags.has("tab:test") },
      { do: "Copy the design (or download the notebook)", ok: (c, p, s) => s.flags.has("copied") },
    ],
    can: "Anything you can measure. Paste the result back to see if the prediction held.",
    servo: "Quick runs take about 2 minutes on Colab's free T4. Two seeds each, so you can tell a real gap from noise.",
  },
];

const P = (k) => full(PRESETS[k]);
export const CHALLENGES = [
  {
    id: "fix", title: "Make it trainable", level: "Starter", start: { ...P("gpt2"), L: 24, residual: false, norm: "None" },
    brief: "Someone built a 24-block model with no residual stream and no norms. The predictor says it won't train. Fix it without making it shallower.",
    goals: [
      { say: "No red warnings on the Predict tab", ok: (c, p) => p.num.bad === 0 && !p.errors.length, now: (c, p) => `${p.num.bad} red` },
      { say: "Still 24 blocks or more", ok: (c) => c.L >= 24, now: (c) => `${c.L} blocks` },
    ],
    hint: "Two switches: the residual rail, and the norm. Both are inside any block.",
    solved: "Residuals give the gradient a straight road to every layer; pre-norms keep every part's inputs the same size. Together they're why depth works at all.",
  },
  {
    id: "kv", title: "Shrink the KV cache 4×", level: "Starter", start: { ...P("llama8"), kv: 32, name: "Llama 3 8B, one K/V per head" },
    brief: "This 8B model keeps a key and value for every head. Its cache for one long chat is huge. Make the cache 4× smaller without making the model smaller.",
    goals: [
      { say: "KV cache at most ¼ of the start", ok: (c, p, s) => p.num.kvFull <= s.start.kvFull / 4 + 1, now: (c, p) => fmtB(p.num.kvFull) },
      { say: "At least 85% of the parameters", ok: (c, p, s) => p.num.N >= 0.85 * s.start.N, now: (c, p, s) => `${Math.round((100 * p.num.N) / s.start.N)}%` },
      { say: "Still 32 query heads", ok: (c) => c.heads === 32, now: (c) => `${c.heads} heads` },
    ],
    hint: "Click the attention and look at K/V groups.",
    solved: "Grouped-query attention: 8 K/V groups serve 32 query heads. The cache shrinks 4×, the parameters drop about 9% (the K and V matrices shrink too, but queries, outputs and MLPs don't), and quality drops very little. This is exactly Llama 3's choice.",
  },
  {
    id: "users", title: "100 users on one GPU", level: "Medium", start: P("llama8"),
    brief: "Every user's conversation needs its own KV cache, and it all has to fit in 80 GB next to the weights. Fit 100 full-length 8k chats on one H100 without shrinking the model.",
    goals: [
      { say: "100 or more users at once", ok: (c, p) => p.num.users >= 100, now: (c, p) => `${p.num.users} users` },
      { say: "Parameters within 2% of the start", ok: (c, p, s) => p.num.N >= 0.98 * s.start.N, now: (c, p, s) => `${(100 * p.num.N / s.start.N).toFixed(1)}%` },
      { say: "Context still 8,192 tokens", ok: (c) => c.ctx >= 8192, now: (c) => `${c.ctx.toLocaleString("en-US")}` },
    ],
    hint: "The cache per user is layers × K/V groups × head size × 2 × 2 bytes × tokens. Which of those can shrink without shrinking the parameters?",
    solved: "Fewer K/V groups (4 instead of 8) or sliding windows on some layers. Both cut the per-user cache while leaving almost every parameter in place. The Speed tab shows the same lever with 8-bit KV caches.",
  },
  {
    id: "long", title: "Read 128k tokens cheaply", level: "Medium", start: { ...P("llama8"), ctx: 131072, name: "Llama 3 8B at 128k" },
    brief: "At 128k tokens, one conversation's cache is bigger than the weights and attention dominates the compute. Make long context cheap, but keep at least some layers that can see the whole text.",
    goals: [
      { say: "Context 131,072 tokens", ok: (c) => c.ctx >= 131072, now: (c) => c.ctx.toLocaleString("en-US") },
      { say: "One chat's cache at most 4 GB", ok: (c, p) => p.num.kvFull <= 4e9, now: (c, p) => fmtB(p.num.kvFull) },
      { say: "Some full-attention layers left", ok: (c) => count(c).attnL > count(c).winL, now: (c) => `${count(c).attnL - count(c).winL} full` },
      { say: "Attention under 40% of the compute", ok: (c, p) => p.num.attnShare < 0.4, now: (c, p) => `${Math.round(100 * p.num.attnShare)}%` },
    ],
    hint: "Select a block, set “Apply to: Every other” and pick Sliding window. Then look at the window size and K/V groups.",
    solved: "Mostly-windowed layers with a few global ones (Gemma 2 and 3, Mistral, Command R do variants of this). Windowed layers' caches stop growing; the global ones still connect distant words.",
  },
  {
    id: "phone", title: "A model for a phone", level: "Medium", start: P("gpt2"), tok: 20,
    brief: "A phone can spare about 2 GB for weights. Build the best model that fits: predicted loss 2.45 or lower.",
    goals: [
      { say: "Weights at most 2 GB (16-bit)", ok: (c, p) => p.num.weights <= 2e9, now: (c, p) => fmtB(p.num.weights) },
      { say: "Predicted loss 2.45 or lower", ok: (c, p) => p.num.loss <= 2.45, now: (c, p) => (Number.isFinite(p.num.loss) ? p.num.loss.toFixed(3) : "—") },
      { say: "No shape errors", ok: (c, p) => !p.errors.length, now: (c, p) => (p.errors.length ? "error" : "ok") },
    ],
    hint: "At 1B parameters, compute-optimal training (20 tokens each) gives about 2.58. Size is capped, so what else lowers loss? Look at Training data on the Design tab.",
    solved: "Over-train: a model that must be small should see far more data than “compute-optimal”. Llama 3 8B saw ~1,900 tokens per parameter. Training costs more; every phone that runs it saves.",
  },
  {
    id: "budget", title: "Best model for $500", level: "Hard", start: P("gpt2"), tok: 20,
    brief: "You have $500 of H100 time ($2.50 an hour). Reach a predicted loss of 2.55 or lower.",
    goals: [
      { say: "Training cost at most $500", ok: (c, p) => p.num.usd <= 500, now: (c, p) => `$${Math.round(p.num.usd).toLocaleString("en-US")}` },
      { say: "Predicted loss 2.55 or lower", ok: (c, p) => p.num.loss <= 2.55, now: (c, p) => (Number.isFinite(p.num.loss) ? p.num.loss.toFixed(3) : "—") },
    ],
    hint: "Cost grows with parameters × tokens. For a fixed budget, the Chinchilla result says split it so tokens ≈ 20 × parameters.",
    solved: "Around 1.3–1.6B parameters on ~20 tokens each. Bigger with less data, or smaller with more, both land higher for the same money. That's what “compute-optimal” means.",
  },
  {
    id: "shape", title: "Same size, better shape", level: "Hard", start: { ...P("gpt2"), L: 96, d: 256, heads: 4, kv: 4, ff: 1024, name: "A very deep, thin model" },
    brief: "96 thin blocks: slow (they run one after another) and outside the shapes that train best. Reshape it to a healthy width per block, keeping the size within 15%.",
    goals: [
      { say: "20–200 width per block", ok: (c) => c.d / c.L >= 20 && c.d / c.L <= 200, now: (c) => `${Math.round(c.d / c.L)}` },
      { say: "Parameters within 15% of the start", ok: (c, p, s) => Math.abs(p.num.N / s.start.N - 1) <= 0.15, now: (c, p, s) => `${p.num.N >= s.start.N ? "+" : "−"}${Math.round(Math.abs(100 * (p.num.N / s.start.N - 1)))}%` },
      { say: "No shape errors", ok: (c, p) => !p.errors.length, now: (c, p) => (p.errors.length ? "error" : "ok") },
    ],
    hint: "Parameters per block grow with width². Doubling the width means about a quarter of the blocks.",
    solved: "Width 512 with 24 blocks, or similar. Kaplan et al. found loss depends little on shape inside 20–200 width per block, so you keep the quality and lose ¾ of the sequential steps.",
  },
];

const fmtB = (n) => (n >= 1e9 ? `${(n / 1e9).toFixed(1)} GB` : `${(n / 1e6).toFixed(0)} MB`);

/** Progress, kept per browser. */
const KEY = "studio-progress-v1";
export const progress = {
  get() { try { return JSON.parse(localStorage.getItem(KEY) || "{}"); } catch { return {}; } },
  mark(id) { try { const p = progress.get(); p[id] = Date.now(); localStorage.setItem(KEY, JSON.stringify(p)); } catch { /* private mode */ } },
};
