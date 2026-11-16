// What a design will do, predicted from fundamentals before anything is trained. Every prediction says
// how sure it is: exact (arithmetic), physics (hardware limits), law (a fit to many training runs),
// rule (a known failure mode), or test (only a real run can tell).
import { full, count, check, blk, isStandard, fmt, bytes } from "./arch.js";

export const TAGS = {
  exact: { label: "Exact", say: "Arithmetic on the shapes. Always right." },
  physics: { label: "Physics", say: "Set by hardware limits: memory bandwidth and chip speed. Right within a small factor." },
  law: { label: "Law", say: "A formula fitted to hundreds of real training runs. Good for standard designs." },
  rule: { label: "Rule of thumb", say: "A failure mode seen again and again in practice." },
  test: { label: "Test it", say: "Nobody can tell you in advance. Train it and measure." },
};

export const H100 = { flops: 989e12, bw: 3.35e12, mem: 80e9 };      // dense BF16, HBM3 bandwidth, memory
export const MFU = 0.4;                                              // a good training run uses ~40% of peak
/** Chinchilla (Hoffmann et al. 2022, approach 3): loss from parameters N and training tokens D. */
export const chinchilla = (N, D) => 1.69 + 406.4 / N ** 0.34 + 410.7 / D ** 0.28;

const sig = (x) => (x >= 100 ? x.toFixed(0) : x >= 10 ? x.toFixed(1) : x.toFixed(2));
const big = (x) => (x >= 1e12 ? `${sig(x / 1e12)}T` : x >= 1e9 ? `${sig(x / 1e9)}B` : x >= 1e6 ? `${sig(x / 1e6)}M` : x >= 1e3 ? `${sig(x / 1e3)}k` : sig(x));
const delta = (a, b, lowerIsBetter = true, unit = "%") => {
  if (!b || a === b) return null;
  const d = unit === "%" ? ((a - b) / b) * 100 : a - b;
  if (Math.abs(d) < (unit === "%" ? 0.05 : 0.0005)) return null;
  return { text: `${d > 0 ? "+" : "−"}${unit === "%" ? Math.abs(d).toFixed(Math.abs(d) < 10 ? 1 : 0) + "%" : Math.abs(d).toFixed(3)}`, good: lowerIsBetter == null ? null : (d < 0) === lowerIsBetter };
};
const time = (h) => (h < 1 ? `${Math.max(1, Math.round(h * 60))} GPU-minutes` : h < 1e4 ? `${sig(h)} GPU-hours` : `${big(h)} GPU-hours`);

/** Per-token numbers the cards share. */
function numbers(c, tokMul) {
  const k = count(c), N = k.total, D = N * tokMul;
  const ctxW = (i) => (blk(c, i).attn === "window" ? Math.min(c.window, c.ctx) : c.ctx);
  let kvTok = 0, attnFlop = 0;                                // bytes of KV cache per token; attention FLOPs per token at full length
  for (let i = 1; i <= c.L; i++) if (blk(c, i).attn !== "none") { attnFlop += 4 * ctxW(i) * c.d; kvTok += 2 * k.kvd * 2; }
  let kvFull = 0;
  for (let i = 1; i <= c.L; i++) if (blk(c, i).attn !== "none") kvFull += 2 * k.kvd * 2 * ctxW(i);
  const matFlop = 2 * (k.nonEmbActive + c.d * c.vocab);     // every weight a token uses, once, plus the readout (experts: only the chosen ones)
  const Na = k.active, Nlaw = k.moeL ? Math.sqrt(N * Na) : N;  // a mixture of experts lands between its active and total size
  return { k, N, Na, Nlaw, D, kvTok, kvFull, attnFlop, matFlop };
}

/** Everything the Studio predicts. base is the preset you started from (for "compared with"). */
export function predict(cfg, base, tokMul = 20) {
  const c = full(cfg), b = full(base), errors = check(c);
  const A = numbers(c, tokMul), Bn = numbers(b, tokMul), k = A.k;
  const loss = chinchilla(A.Nlaw, A.D), lossB = chinchilla(Bn.Nlaw, Bn.D);
  const trainH = (6 * A.Na * A.D) / (H100.flops * MFU) / 3600;
  const serveB = 2 * A.N, fits = serveB + A.kvFull < H100.mem;
  const tps = H100.bw / (2 * A.Na);
  const std = isStandard(c);

  const cards = [
    { tag: "exact", title: "Parameters", value: fmt(A.N), d: delta(A.N, Bn.N, null), say: `${fmt(k.nonEmb)} in the blocks, ${fmt(k.embed + k.pos + k.head)} in the tables.${k.moeL ? ` Each token uses only ${fmt(A.Na)} of them: ${k.K} of ${k.E} experts.` : ""}` },
    { tag: "exact", title: "Weights to serve (16-bit)", value: bytes(serveB), d: delta(serveB, 2 * Bn.N), say: "Two bytes per parameter. This has to sit in GPU memory just to answer one question." },
    { tag: "exact", title: "KV cache, one full-length chat", value: bytes(A.kvFull), d: delta(A.kvFull, Bn.kvFull), say: `${bytes(A.kvTok)} per token × up to ${c.ctx.toLocaleString("en-US")} tokens.${k.winL ? " Window layers stop growing at the window size." : ""} Every extra user needs this again.` },
    { tag: "exact", title: "Compute per token", value: `${big(A.matFlop + A.attnFlop)} FLOP`, d: delta(A.matFlop + A.attnFlop, Bn.matFlop + Bn.attnFlop), say: `About 2 FLOPs per weight${k.moeL ? " the token actually uses" : ""}, plus attention at full length: ${Math.round((100 * A.attnFlop) / (A.matFlop + A.attnFlop))}% of the total.` },
    { tag: "physics", title: "Fastest speed, one user, one H100", value: fits ? `~${Math.round(tps).toLocaleString("en-US")} tokens/s` : "Doesn't fit", d: fits ? delta(tps, H100.bw / (2 * Bn.N), false) : null, say: fits ? `Each new token must read every weight it uses from memory once${k.moeL ? ` (only the ${k.K} chosen experts, for one user)` : ""}, and memory moves 3.35 TB/s. That sets the ceiling, however fast the maths is.${tps > 3000 ? " A model this small hits other overheads first (launching each step), so real speed is far lower." : ""}` : `It needs ${bytes(serveB + A.kvFull)} but an H100 holds 80 GB. Split it across GPUs or shrink the weights.` },
    c.vocab < 1000
      ? { tag: "law", title: "Expected loss", value: "Different scale", say: "Chinchilla's formula is for word-piece tokens. Character-level loss is per character, so it's on another scale: a good small character model reaches about 1.0–1.5." }
      : { tag: k.moeL ? "rule" : "law", title: "Expected loss", value: sig(loss), d: delta(loss, lossB, true, "abs"), say: `${k.moeL ? `A mixture of experts does better than a dense model of its active size and worse than one of its total size; this uses the middle (${fmt(A.Nlaw)}). ` : ""}Chinchilla's formula for ${fmt(A.Nlaw)} parameters trained on ${big(A.D)} tokens (${tokMul}× the parameter count). Lower is better.${A.N < 5e7 ? " This is below the sizes it was fitted on (70M and up), so it's rough." : ""}${std ? "" : " Your design isn't a standard transformer, so treat this as the best case: the formula doesn't know about your changes."}` },
    { tag: "law", title: "Best amount of data", value: `${big(20 * A.N)} tokens`, say: "For a fixed compute budget, about 20 tokens per parameter gives the lowest loss. Training longer still helps; it's just not the cheapest way to that loss." },
    { tag: "physics", title: "Training cost", value: time(trainH), d: delta(trainH, (6 * Bn.N * Bn.D) / (H100.flops * MFU) / 3600), say: `6 × parameters × tokens = ${big(6 * A.N * A.D)} FLOPs, on H100s at 40% of peak. At $2–3 per GPU-hour that's about $${big(trainH * 2)}–${big(trainH * 3)}.` },
  ];

  // rules of thumb: what is likely to go wrong, and why
  const rules = [], R = (level, title, why, tag = "rule") => rules.push({ level, title, why, tag });
  const hd = c.d / c.heads, ratio = c.d / c.L, ffr = c.ff / c.d;
  if (!c.residual) R(c.L >= 6 ? "bad" : "warn", "No residual stream: expect it not to train", `On the way back, the gradient must pass through every block's weights, ${c.L} times. Each pass scales it a little up or down, and ${c.L} of those compound: it vanishes or explodes. Deep plain networks stopped improving past about 20 layers until residuals (2015) gave the gradient a straight road. Measured in our Studio lab: 6 blocks without residuals stalled at loss 3.34 (only letter frequencies learned) while the same model with them reached 1.62.`);
  if (c.norm === "None") R(c.L > 8 ? "bad" : "warn", "No normalisation: expect it to diverge", `Each block adds to the stream, so vectors grow layer by layer. Big vectors make softmax saturate and gradients spike. Without norms you need a tiny learning rate and special initialisation (it's been done, e.g. Fixup), and it is fragile. In our own 6-block lab run, removing the norms cost 0.48 in loss (2.10 vs 1.62), even with clipping.`);
  if (c.norm !== "None" && c.normPos === "post" && c.L > 12) R("warn", "Deep post-norm: unstable early in training", "Post-norm puts the norm on the highway itself, after each add, so gradients near the bottom get scaled down. Deep post-norm models need learning-rate warmup and still often diverge. That's why GPT-2 and everything since put the norm before each part.");
  else if (c.norm !== "None" && c.normPos === "post") R("info", "Post-norm (the 2017 original)", "Fine at this depth with warmup. It is known to give slightly better results than pre-norm when it does train well; the catch is stability as you go deeper.");
  if (!errors.length && hd < 32) R("warn", `Small heads: ${hd} numbers each`, "Each head compares queries and keys of only that size. Below ~32, heads struggle to tell many things apart. Use fewer, bigger heads.");
  if (!errors.length && hd > 256) R("info", `Big heads: ${hd} numbers each`, "Few heads means few different questions per layer. Most models use 64–128 per head.");
  if (c.kv === 1 && c.heads > 1) R("info", "Multi-query attention", `All ${c.heads} heads share one K and V. The KV cache shrinks ${c.heads}×; papers report a small quality loss, which is why grouped-query (a few groups) is the usual middle ground.`);
  if (ffr < 2) R("warn", `Narrow MLP: ${ffr.toFixed(1)}× the width`, "The MLP is where most knowledge is stored. Most models use 2.7–4×; much narrower usually costs more loss than it saves.");
  if (ffr > 8) R("info", `Very wide MLP: ${ffr.toFixed(1)}× the width`, "Works, but at the same parameter count you'd usually do better spending some of it on depth.");
  if (ratio < 16) R("warn", `Deep and thin: ${Math.round(ratio)} width per layer`, "Kaplan et al. (2020) found shape matters little across a wide range, roughly 20 to 200 width per layer. Far outside it, loss gets worse, and very deep thin models are slow (layers run one after another).", "law");
  if (ratio > 400) R("warn", `Wide and shallow: ${Math.round(ratio)} width per layer`, "Only a few rounds of thinking. Outside the 20–200 range Kaplan et al. measured, loss gets worse at the same size.", "law");
  if (k.attnL === 0) R("bad", "No attention anywhere", "No block lets words see each other, so every position predicts from its own token alone: a lookup of what usually comes next after one word (a bigram model). It cannot use context at all.");
  else if (k.attnL < c.L) R("info", `${c.L - k.attnL} block${c.L - k.attnL > 1 ? "s" : ""} without attention`, "Fewer chances to mix information between words. Hybrids do this on purpose (some layers attention, some cheaper mixers like Mamba), keeping a few attention layers for exact lookups.");
  if (k.moeL) R("info", `Mixture of experts: ${k.K} of ${k.E} per token`, `Total ${fmt(A.N)} parameters, but each token pays for ${fmt(A.Na)}: more knowledge at the same compute. The catch: every expert must sit in memory, and training needs a load-balancing loss or the router sends everything to a few favourites. Mixtral, DeepSeek-V3 and (reportedly) GPT-4 are built this way.`);
  if (k.moeL && k.K === 1) R("info", "One expert per token (Switch)", "Cheapest routing, used by Switch Transformer. Two per token is the usual choice: smoother training and a little better quality.");
  if (k.mlpL === 0) R("warn", "No MLP anywhere", "Attention-only models store far less per parameter: the MLPs are where facts like “Paris is in France” mostly live.");
  else if (k.mlpL < c.L) R("info", `${c.L - k.mlpL} block${c.L - k.mlpL > 1 ? "s" : ""} without an MLP`, "Less capacity for stored knowledge, fewer parameters.");
  if (k.winL) {
    const reach = k.winL === k.attnL ? k.attnL * c.window : Infinity;
    R(reach < c.ctx ? "warn" : "info", `${k.winL} sliding-window layer${k.winL > 1 ? "s" : ""} (window ${c.window.toLocaleString("en-US")})`, reach < c.ctx ? `Every attention layer is windowed, so information can travel at most ${k.attnL} × ${c.window.toLocaleString("en-US")} = ${reach.toLocaleString("en-US")} tokens: less than the ${c.ctx.toLocaleString("en-US")} context. Words further apart can never meet.` : "Windowed layers are cheap and their cache stops growing; the full-attention layers still connect distant words. Gemma and Mistral-style models mix the two.");
  }
  if (c.pos === "learned") R("info", "Learned positions", `There is no row for position ${c.ctx.toLocaleString("en-US")} or later, so the model can never read longer text without retraining. RoPE can be stretched with tricks.`);
  if (c.pos === "none") R("info", "No position signal (NoPE)", "Surprisingly, decoder-only models still learn word order: the causal mask tells each position how many words came before. It trains; how well it handles long text is still argued about.", "test");
  if (A.attnFlop > A.matFlop) R("info", "Attention dominates at full length", `At ${c.ctx.toLocaleString("en-US")} tokens, attention costs more than all the weights together. Sliding windows or fewer attention layers would cut it.`);
  if (tokMul < 10) R("warn", "Under-trained", `${tokMul} tokens per parameter is well below the ~20 that's compute-optimal: a smaller model trained longer would reach a lower loss for the same cost.`, "law");
  if (tokMul > 100) R("info", "Over-trained on purpose", `${tokMul} tokens per parameter. Costly to train, but you get a small model that's cheap to run. Llama 3 8B saw about 15T tokens: ~1,900 per parameter.`, "law");
  if (!c.tied && k.head / A.N > 0.1) R("info", "Untied unembedding", `The readout matrix is ${Math.round((100 * k.head) / A.N)}% of the model. Tying it to the embedding saves that, a common choice for small models.`);
  if (c.vocab < 1000) R("info", "Character or byte-level vocabulary", "Each token carries little text, so sequences get several times longer and losses aren't comparable with word-piece models. The Chinchilla numbers above assume a normal tokenizer.");

  // what only a run can answer
  const tests = [];
  const pre = full(base);
  if (!std) tests.push("Whether it trains at all, and how its loss curve compares with the baseline at the same size.");
  if (k.moeL) tests.push("Whether the router spreads tokens evenly across experts, and how much better than a dense model of the same compute it gets.");
  if (c.act !== pre.act) tests.push(`Whether ${c.act} beats ${pre.act} here (SwiGLU usually wins by a little at equal parameters).`);
  if (c.norm !== pre.norm && c.norm !== "None") tests.push(`${c.norm} vs ${pre.norm}: expect nearly the same loss; RMSNorm is a bit faster.`);
  if (k.winL || k.attnL < c.L || k.mlpL < c.L) tests.push("How much the removed or windowed layers cost on long-range tasks, not just on average loss.");
  if (c.pos !== pre.pos) tests.push(`${c.pos} positions vs ${pre.pos}, especially on texts longer than training.`);
  if (!tests.length) tests.push("Nothing beyond the law: this is a standard design. Train it to check the loss lands near the prediction.");

  const num = { N: A.N, Na: A.Na, D: A.D, weights: serveB, kvFull: A.kvFull, kvTok: A.kvTok, tps: fits ? tps : 0, fits, loss: c.vocab < 1000 ? NaN : loss,
    users: fits ? Math.max(0, Math.floor((H100.mem - serveB) / Math.max(1, A.kvFull))) : 0, attnShare: A.attnFlop / (A.matFlop + A.attnFlop),
    flopTok: A.matFlop + A.attnFlop, trainH, usd: trainH * 2.5, ratio, hd, bad: rules.filter((r) => r.level === "bad").length, warn: rules.filter((r) => r.level === "warn").length };
  return { cards, rules, tests, errors, std, loss, N: A.N, D: A.D, num };
}
