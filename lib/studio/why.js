// Why each part is there, from the ground up: what it does, the problem it solves, what breaks without it,
// and the variants real models use. Shown under "Why it's there" when you select a part.
import { h } from "../core.js";

export const WHY = {
  embed: {
    does: "Turns each token's number into a vector: a row of the table.",
    why: "A token id like 464 is just a name; 464 isn't “bigger” than 12 in any useful way. A learned vector per token lets similar tokens end up near each other, so whatever the model learns about one helps with the others.",
    without: "There is nothing for the rest of the model to compute with.",
    recipe: "Every model has one. Small models often reuse it as the readout (tied weights) to save parameters.",
  },
  head: {
    does: "Scores every token in the vocabulary as the next one; softmax turns scores into probabilities.",
    why: "Training needs a probability for the true next token, so we can measure how surprised the model was (the loss) and nudge it to be less surprised.",
    without: "The model has no way to say anything.",
    recipe: "Tied to the embedding (GPT-2, Gemma) or its own matrix (Llama 3). With a big vocabulary it's a large share of a small model.",
  },
  pos: {
    does: "Tells the model where each token sits.",
    why: "Attention on its own is a weighted sum over earlier tokens: shuffle them and the sum is the same. Word order matters (“dog bites man”), so position has to come in somewhere.",
    without: "A decoder can still count tokens through its causal mask, so it learns order anyway, more slowly (NoPE).",
    recipe: "A learned table (GPT-2), or RoPE: rotate queries and keys by an angle that grows with position, so attention sees distances (Llama, Mistral, almost everyone now).",
  },
  attn: {
    does: "Lets every token look back at earlier tokens and pull in what it needs.",
    why: "To predict the next word you need the words before it. Each token makes a query (“what am I looking for?”), every earlier token offers a key (“what I contain”) and a value (“what I'll hand over”). Query·key decides how much of each value to take.",
    without: "Each token predicts from itself alone: a table of what usually follows one token (a bigram model).",
    recipe: "Many heads ask different questions at once. Sharing keys and values between heads (grouped-query) shrinks the cache; sliding windows cap how far back a layer looks.",
  },
  mlp: {
    does: "Processes each token on its own: widen, bend, narrow.",
    why: "Attention only moves and mixes information; it's mostly averaging. The MLP is where each token's vector gets transformed: a bank of feature detectors whose outputs write new facts into the stream. Most of a model's knowledge lives here.",
    without: "The model can route information but stores much less: attention-only models are far worse per parameter.",
    recipe: "GPT-2: 4× wider with GELU. Llama: SwiGLU, a gate that decides which features pass, about 3.5× wide so the size stays similar. Mixture-of-experts: many MLPs, each token uses a few.",
  },
  norm: {
    does: "Rescales each token's vector to a steady size.",
    why: "Every block adds to the stream, so vectors keep growing. Big inputs saturate softmax and make gradients spike. Normalising before each part means every part always sees inputs of the same size, so one learning rate works everywhere.",
    without: "Training needs tiny learning rates and careful initialisation, and still tends to diverge as you go deeper.",
    recipe: "Pre-norm (before each part) since GPT-2: stable at any depth. Post-norm (after the add) was the 2017 original. RMSNorm skips subtracting the mean: as good, a bit cheaper.",
  },
  lnf: {
    does: "One last rescale before reading out a word.",
    why: "With pre-norm, the stream itself is never normalised, so it can be any size by the top. The readout needs a steady scale.",
    without: "Logits swing with the stream's size: unstable training.",
    recipe: "Every pre-norm model has one.",
  },
  residual: {
    does: "Each part adds its output to a running total, the residual stream, instead of replacing it.",
    why: "Backpropagation multiplies gradients through every layer on the way down. Many multiplications compound: the signal shrinks to nothing or blows up. The add gives the gradient a straight road from the loss to every layer, so even block 1 hears clearly.",
    without: "Past a handful of layers the model barely trains. Before residuals (2015), deep networks got worse as you added layers.",
    recipe: "Every modern deep network. It also means each block makes a small edit to a shared memory, which is why you can read the stream out mid-way.",
  },
  block: {
    does: "One round of thinking: attention (look around) then MLP (think about it), each adding to the stream.",
    why: "One round can only combine things one step apart. Stacking rounds lets later blocks build on what earlier ones found: block 1 finds the previous word, block 2 uses that to copy what followed last time.",
    without: "Fewer blocks, fewer steps of reasoning; empty blocks just pass the stream through.",
    recipe: "GPT-2 small has 12, Llama 3 8B 32, the biggest models around 100. Width and depth trade off: Kaplan et al. found loss depends little on the shape between about 20 and 200 width per block.",
  },
  model: {
    does: "Text in, a probability for every possible next token out.",
    why: "Predicting the next token well forces a model to learn grammar, facts and some reasoning, because all of those make text predictable.",
    without: "",
    recipe: "Everything here is the same skeleton since 2017: embed, a stack of blocks, read out. The differences are in the parts.",
  },
};

/** Which explanation fits a selected part. */
export function whyKey(meta, kind) {
  if (!meta) return kind === "model" ? "model" : null;
  if (meta.edit && typeof meta.edit === "object") return "block";
  if (/unembed|next token|logit|softmax/i.test(meta.name)) return "head";
  if (/final norm/i.test(meta.name)) return "lnf";
  if (/residual|stream|add/i.test(meta.name) && meta.edit === "residual") return "residual";
  return WHY[meta.edit] ? meta.edit : null;
}

export function whyCard(key) {
  const w = WHY[key];
  if (!w) return null;
  return h("details.st-why",
    h("summary", "Why it's there"),
    h("dl",
      h("dt", "What it does"), h("dd", w.does),
      h("dt", "The problem it solves"), h("dd", w.why),
      w.without ? h("dt", "Take it out") : null, w.without ? h("dd", w.without) : null,
      h("dt", "In real models"), h("dd", w.recipe)));
}
