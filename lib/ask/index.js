// The Socratic path: short conversations with Neuro in very easy English. Each thread builds one idea
// from nothing, by asking you questions. The steps live in parts (a.js, b.js, c.js) and load on demand.
export const THREADS = [
  { id: "learn", part: "a", title: "What does it mean to learn?", blurb: "Rules or examples? How a computer can figure things out by itself.", chapter: "why-learn" },
  { id: "numbers", part: "a", title: "Everything is numbers", blurb: "Pictures, words and sounds all turn into lists of numbers.", chapter: "linear-algebra" },
  { id: "line", part: "a", title: "A line that guesses", blurb: "Your first model: two numbers and a ruler.", chapter: "why-learn" },
  { id: "wrong", part: "a", title: "How wrong is the guess?", blurb: "One number that says how bad a model is.", chapter: "why-learn" },
  { id: "downhill", part: "a", title: "Walking downhill", blurb: "How the computer fixes its own mistakes, one small step at a time.", chapter: "gradient-descent" },
  { id: "neuron", part: "a", title: "One tiny brain cell", blurb: "Multiply, add, squash. That's a neuron.", chapter: "perceptron" },
  { id: "xor", part: "b", title: "When one line is not enough", blurb: "A puzzle that stopped AI for years, and the fix.", chapter: "xor" },
  { id: "blame", part: "b", title: "Who made the mistake?", blurb: "Passing the blame backwards: how big networks learn.", chapter: "backprop" },
  { id: "memorise", part: "b", title: "Remembering or understanding?", blurb: "Why a perfect score on practice can mean nothing.", chapter: "generalization" },
  { id: "words", part: "b", title: "Words become numbers", blurb: "Cutting text into pieces, and giving each piece a place on a map.", chapter: "embeddings" },
  { id: "nextword", part: "b", title: "Guess the next word", blurb: "The one simple game every chatbot is trained on.", chapter: "next-token" },
  { id: "attention", part: "b", title: "Who should I listen to?", blurb: "How each word looks back at the words that matter.", chapter: "attention" },
  { id: "transformer", part: "c", title: "The transformer, block by block", blurb: "Stack the same block many times. That's the whole recipe.", chapter: "transformer" },
  { id: "bigger", part: "c", title: "Why bigger works, and why GPUs", blurb: "More data, more size, and a chip with thousands of small workers.", chapter: "scaling-laws" },
  { id: "helper", part: "c", title: "From parrot to helper", blurb: "How a text predictor becomes an assistant: examples, then thumbs up and down.", chapter: "sft" },
  { id: "think", part: "c", title: "Thinking step by step", blurb: "Why writing the steps out makes a model smarter.", chapter: "reasoning" },
  { id: "tools", part: "c", title: "Models that use tools", blurb: "Think, act, look, repeat: the loop behind every agent.", chapter: "agent-loops" },
  { id: "fail", part: "c", title: "Where models go wrong", blurb: "Confident guesses, one-way facts, and how to stay safe.", chapter: "hallucination" },
];
export const threadById = (id) => THREADS.find((t) => t.id === id);

const KEY = "ai-anatomy:ask";
export const askState = {
  get() { try { return JSON.parse(localStorage.getItem(KEY) || "{}"); } catch { return {}; } },
  set(id, patch) { const s = askState.get(); s[id] = { ...(s[id] || {}), ...patch }; try { localStorage.setItem(KEY, JSON.stringify(s)); } catch { /* private mode */ } return s; },
};
