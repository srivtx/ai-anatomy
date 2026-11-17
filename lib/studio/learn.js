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
    guess: { q: "With no blocks in between, what can this model learn?", opts: ["That “u” usually follows “q”", "That a speaker's name comes before a colon", "Which words rhyme at line ends"], a: 0,
      why: "It sees only the current character, so all it can learn is which character tends to follow which: a table of pairs, a bigram model. Anything that needs two or more characters of context is out of reach." },
    myth: ["The embedding understands words.", "It's a lookup table: one learned row per token. Meaning only shows up in how the rows end up arranged after training (similar tokens get similar rows), not in the table itself."],
    steps: [
      { do: "Click the blue Embedding layer and read what it is", ok: (c, p, s) => /embedding/i.test(s.sel) && !/un/i.test(s.sel) },
      { do: "Click the teal Unembedding layer", ok: (c, p, s) => /unembedding/i.test(s.sel) },
      { do: "Open the empty block: it's a dark plate because it does nothing yet", ok: (c, p, s) => visited(s, /^b1$/) },
    ],
    can: "Predict the next character from the current one only: after “q” comes “u”. That's a bigram model, about 2.5 loss on Shakespeare.",
    sample: "th rememe ke whenllered ca meaman youd INAlste grecofurey hiththerere hath thin", real: "From our lab run of a model with no attention: it can only use the current letter.",
    servo: "Double-click a layer to go inside it. The arrow keys turn the model, Escape comes back out.",
  },
  {
    id: "attend", title: "Let characters look back", start: { ...tiny, L: 1, blocks: { 1: { attn: "none", mlp: false } } },
    neuro: "To guess well you need the characters before this one, not just this one. Attention lets each position ask a question (query) and pull in what earlier positions offer (values), weighted by how well their keys match.",
    guess: { q: "Attention adds weights. For width d, about how many?", opts: ["d", "4 × d²", "d × the length of the text"], a: 1,
      why: "Four d × d matrices: query, key, value and output. At width 128 that's 65,536 numbers. It doesn't grow with the text: the same weights compare any pair of positions. What grows with length is the work, not the weights." },
    myth: ["Attention is the slow part.", "At short lengths the MLP and the readout cost more. Attention's work grows with the length squared, so it only dominates long texts, and its KV cache is what limits how many users fit on a GPU. The Speed tab shows both."],
    steps: [
      { do: "Select block 1 and switch its attention to Full", ok: (c) => blk(c, 1).attn !== "none" },
      { do: "Open block 1 › Attention and look at the heads", ok: (c, p, s) => visited(s, /^b1\/attn/) },
      { do: "Open one head to see who looks at whom", ok: (c, p, s) => visited(s, /^b1\/attn\/h\d+/) },
      { do: "Inside the head, press ▶ Watch a token: each word scans the words before it", ok: (c, p, s) => s.flags.has("scan") },
    ],
    can: "Use the last few characters: it spells common words and gets spaces and capitals mostly right.",
    sample: "The wis the mere, and the lo the sor of ther ham",
    servo: "Each head's grid is causal: a character can only look left, at what came before it. The future is hidden during training too.",
  },
  {
    id: "mlp", title: "Room to think: the MLP", start: { ...tiny, L: 1, blocks: { 1: { attn: "full", mlp: false } } },
    neuro: "Attention moves information around but mostly averages it. The MLP takes each position on its own, widens it 4×, bends it with a nonlinearity and narrows it back. That's where patterns get stored.",
    guess: { q: "In a standard block, which has more weights?", opts: ["Attention", "The MLP", "About equal"], a: 1,
      why: "The MLP: two d × 4d matrices, 8 × d², against attention's 4 × d². Two thirds of every block. But more weights isn't more important: in our lab runs, removing every MLP cost 0.36 loss, removing every attention cost 0.86." },
    myth: ["Facts live in attention.", "Most stored knowledge sits in the MLPs: experiments that edit a single fact (“the Eiffel Tower is in Rome”) change MLP weights. Attention decides where to look; the MLP holds what's there."],
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
    guess: { q: "Going from 6 blocks to 12 (same width), what happens?", opts: ["Loss halves", "Loss drops a little, and each token costs about twice as much", "Nothing: more blocks only help much bigger models"], a: 1,
      why: "Compute per token grows with the parameters, so twice the blocks is about twice the cost. Loss falls slowly, as a power law. In our short lab run, 6 blocks beat 2 by only 0.13: depth pays off more with longer training." },
    myth: ["Deeper is always better.", "For a fixed number of parameters, very deep and thin trains slower and serves slower, because blocks run one after another. Real models balance width and depth: GPT-3 is 96 blocks, but 12,288 wide."],
    steps: [
      { do: "Make it 4 blocks (Design tab, or the slider)", ok: (c) => c.L >= 4 && all(c, (b) => b.attn !== "none" && b.mlp) },
      { do: "Open the Predict tab and read the parameter count", ok: (c, p, s) => s.flags.has("tab:predict") },
      { do: "Press ▶ Watch a token, then pause it (or step with , and .) on a block", ok: (c, p, s) => s.flags.has("flow-pause") },
    ],
    can: "Copies names and phrases it saw earlier in the text, keeps a speaker's style. Loss near 1.6.",
    sample: "thy farms kiss selt it one means.\n\nLADY ANNE:\nBeancouse you for meet? hath not see has so.", real: "From our lab run of a 6-block model after about 2 minutes of training.",
    servo: "New blocks copy the standard settings. Set any one of them back to empty to see the dark plate again.",
  },
  {
    id: "residual", title: "The highway", start: { ...tiny, L: 4 },
    neuro: "Every block adds to a shared running total, the residual stream. Why add instead of replace? During training the error signal flows backwards through every layer. Without the add it gets multiplied by every layer's weights and fades away.",
    guess: { q: "Take the residual stream out of a 6-block model. What happens?", opts: ["It trains a bit slower", "It hardly learns at all", "Nothing: it's only a speed trick"], a: 1,
      why: "Measured in our lab: 3.34 against 1.62. It stalled at plain letter frequencies. We had predicted a mild penalty (+0.54); the run said +1.72, and we fixed the rule." },
    myth: ["The residual is a shortcut that makes it faster.", "It's what makes depth trainable at all: the error signal gets a path back to the first layers that isn't multiplied by every layer on the way."],
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
    guess: { q: "Remove every norm from a 6-block model (with residuals). The loss…", opts: ["About the same: 6 blocks is shallow", "Clearly worse", "Infinite: it blows up at once"], a: 1,
      why: "Measured: 2.10 against 1.62, with gradient clipping keeping it alive. We had guessed +0.08. Without clipping or a tiny learning rate it often does blow up. Post-norm cost 0.21 too, even this shallow." },
    myth: ["Normalisation is a small detail.", "In our runs it was worth 0.48, more than removing every MLP. Where the norm sits matters as well: post-norm trailed pre-norm by 0.21 at only 6 blocks."],
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
    guess: { q: "How much of GPT-2 small's 124M parameters is the token table?", opts: ["About 3%", "About 30%", "About 70%"], a: 1,
      why: "50,257 × 768 = 38.6M, about 31%. At small sizes the vocabulary is a big share; in Llama 3 70B the table and readout together are about 3%. That's why tied embeddings mattered in 2019 and barely matter now." },
    myth: ["Bigger is always better.", "Only with enough data. Chinchilla says about 20 training tokens per parameter: a big model trained on too little text loses to a smaller one trained longer. Try the tokens slider on the Predict tab."],
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
    guess: { q: "Grouped K/V (8 groups for 32 heads) mainly saves…", opts: ["Training compute", "Memory while generating", "Parameters"], a: 1,
      why: "The KV cache shrinks 4×, so more users or longer chats fit on one GPU. Parameters drop only a few percent and training compute barely moves. It's a serving fix." },
    myth: ["The modern recipe is a new architecture.", "It's the 2017 skeleton with four parts swapped. Since GPT-2, the big gains came from data, scale and training more than from the architecture."],
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
    id: "long", title: "Long memory: the KV cache", start: { ...PRESETS.llama8, name: "Llama 3 8B" },
    neuro: "To write each new token, attention needs the keys and values of every token before it. Recomputing them each time would be slow, so they are kept: the KV cache. It grows with every token of the chat, for every block. At long lengths it outgrows the weights.",
    guess: { q: "Llama 3 8B's weights are 16 GB. At 128k tokens, one chat's KV cache is about…", opts: ["0.2 GB", "17 GB, as big as the weights", "1,600 GB"], a: 1,
      why: "32 blocks × 8 K/V groups × 128 numbers × 2 (key and value) × 2 bytes = 128 kB per token. Times 131,072 tokens: 17 GB, for one user. That's why long context is a memory problem before it's a compute problem." },
    myth: ["Long context is just a bigger number in the config.", "The weights don't change, but every token kept costs cache memory and attention work grows with length. The model must also be trained at long lengths: RoPE scaling can stretch positions it never saw, but quality falls past what it was trained on."],
    steps: [
      { do: "Speed tab: find the KV cache for one chat", ok: (c, p, s) => s.flags.has("tab:speed") },
      { do: "Design tab: set the context to 131,072 tokens", ok: (c) => c.ctx >= 131072 },
      { do: "Select a block, Apply to: Every other, Attention: Sliding window", ok: (c) => count(c).winL > 0 && count(c).attnL > count(c).winL },
      { do: "Get one chat's cache under 10 GB (watch the HUD and Speed tab)", ok: (c, p) => c.ctx >= 131072 && p.num.kvFull < 10e9 },
    ],
    can: "Read a whole book in one go, and serve it to more than one user at a time.",
    servo: "Windowed blocks keep only the last few thousand tokens, so their cache stops growing. Keep some full blocks so distant words can still find each other.",
  },
  {
    id: "experts", title: "Many experts, few at a time", start: { ...PRESETS.mistral7, name: "Mistral 7B" },
    neuro: "A bigger MLP stores more, but every token pays for all of it. A mixture of experts keeps many MLPs per block and a small router sends each token to only a couple of them. You store a lot and use a little.",
    guess: { q: "Mixtral: 8 experts per block, 2 used per token. Next to a dense model of the same stored size (46.7B), each token costs…", opts: ["The same", "About a quarter to a third", "Nothing extra: experts are free"], a: 1,
      why: "12.9B of the 46.7B parameters are used per token: 28%. Attention and the embeddings are shared, so it's more than 2/8. The catch is memory: all 46.7B must sit on the GPUs." },
    myth: ["Each expert is an expert in a subject, like law or medicine.", "In real models experts specialise in surface patterns: punctuation, code tokens, numbers, pieces of words. The router decides per token and per block, so one sentence passes through many experts."],
    steps: [
      { do: "Select a block, Apply to: All, MLP: Experts", ok: (c) => all(c, (b) => b.mlp === "moe") },
      { do: "Open a block's MLP: each token row lights up the experts it was sent to", ok: (c, p, s) => /^b\d+\/mlp$/.test(s.path) && all(c, (b) => b.mlp === "moe") },
      { do: "Click the MLP and set 16 experts: knowledge goes up, cost per token doesn't", ok: (c) => c.experts >= 16 && all(c, (b) => b.mlp === "moe") },
    ],
    can: "Know much more for the same speed per token, if you have the memory to hold every expert.",
    servo: "Watch the HUD: Parameters jumps with the experts, but the Speed tab's tokens per second barely moves. That gap is the whole idea.",
  },
  {
    id: "test", title: "Test it for real", start: { ...PRESETS.gpt2, norm: "RMSNorm", act: "SwiGLU", ff: 2048, pos: "RoPE", kv: 4 },
    neuro: "Everything so far was predicted from fundamentals. Some of it is exact (parameter counts), some is a rule of thumb. The only judge is a training run. The Test tab writes one for you.",
    guess: { q: "Our predictor's first guesses on 7 lab runs: how many held?", opts: ["All 7", "About half", "1 of 7"], a: 2,
      why: "Only “no attention” held (+0.85 predicted, +0.86 measured). Residuals, norms, MLPs, post-norm, depth and the Llama recipe were all off, some by 3×. That's why you test. Each miss is now in the rules, and in the track record on the Test tab." },
    myth: ["A prediction is a guess.", "Some are exact (parameter counts, memory), some are laws with error bars, some are rules of thumb. The tag on each line says which, and only the last kind needs a training run to check."],
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
    id: "moe", title: "More knowledge, same speed", level: "Medium", start: P("mistral7"),
    brief: "Make Mistral 7B store at least 4× as many parameters while each token costs at most twice the compute it does now.",
    goals: [
      { say: "At least 4× the parameters", ok: (c, p, s) => p.num.N >= 4 * s.start.N, now: (c, p, s) => `${(p.num.N / s.start.N).toFixed(1)}×` },
      { say: "Compute per token at most 2×", ok: (c, p, s) => p.num.flopTok <= 2 * s.start.flopTok, now: (c, p, s) => `${(p.num.flopTok / s.start.flopTok).toFixed(2)}×` },
      { say: "No shape errors", ok: (c, p) => !p.errors.length, now: (c, p) => (p.errors.length ? "error" : "ok") },
    ],
    hint: "Making the model wider or deeper makes every token pay. Is there a way to have parameters that most tokens skip? Select a block, Apply to: All, MLP: Experts.",
    solved: "A mixture of experts: 8 MLPs per block, 2 used per token. That's Mixtral 8×7B: 46.7B parameters, 12.9B used per token. Knowledge scales with what's stored; speed with what's used.",
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
