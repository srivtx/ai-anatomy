import { h, loop } from "../lib/core.js";
import { stats, predict, code, callout, section, prose, button } from "../lib/widgets.js";
import { xyChart } from "../lib/plot.js";
import { learner, D, E } from "../lib/icl.js";

const PARADIGMS = [
  { name: "Next-token prediction", signal: "The data itself", who: "Every LLM's pretraining", note: "Predicting text forces a model to build whatever helps predict it: grammar, facts, other minds, maths. Everything else on this map is built on top.", c: 1 },
  { name: "In-context learning", signal: "Examples in the prompt; weights frozen", who: "Emerged in GPT-3 (2020)", note: "Learning that happens in the forward pass. Mechanisms found so far: induction heads (copy what followed last time) and, in linear attention, an implicit gradient step (the sim above).", c: 2 },
  { name: "RL from verifiable rewards", signal: "A checker: tests pass, answer matches", who: "o1, DeepSeek-R1 and all reasoning models (2024–)", note: "The reward is objective, so it scales without humans. Open question: does it create new abilities or mostly sharpen existing ones?", c: 3 },
  { name: "RL from feedback", signal: "Human or AI preferences", who: "Every chat assistant", note: "Teaches what people prefer, including things nobody can verify: tone, helpfulness, judgement. Also teaches sycophancy.", c: 4 },
  { name: "Self-play", signal: "Winning against yourself", who: "AlphaZero; LLM self-play like Absolute Zero (2025), where a model invents its own tasks", note: "An automatic curriculum: your opponent is always exactly as strong as you. Hard to transfer to open-ended domains with no clear win condition.", c: 5 },
  { name: "On-policy distillation", signal: "A teacher grading the student's own samples, token by token", who: "Widely used for small models (2025)", note: "Off-policy distillation trains on the teacher's text; the student never learns to recover from its own mistakes. On-policy: the student writes, the teacher scores every token (reverse KL). Dense like supervised learning, on-distribution like RL, and reported to be roughly an order of magnitude cheaper than RL for the same gain.", c: 6 },
  { name: "Test-time training", signal: "Self-supervised loss on the test input itself", who: "Top ARC Prize 2024 entries; TTT layers", note: "Fine-tune a copy of the model on augmented versions of each test puzzle before answering. Turned ~40% models into ~55% on ARC-AGI-1.", c: 1 },
  { name: "Latent prediction (JEPA)", signal: "Predict the representation of a hidden part, not its pixels", who: "I-JEPA, V-JEPA 2 (Meta)", note: "Skips detail that can't be predicted (leaf textures) and keeps what can (where the hand will be). A bet on a better objective than next-token for the physical world.", c: 2 },
  { name: "World models", signal: "Predict the next observation given an action, then plan or train inside the model", who: "DreamerV3; Genie 3 (2025)", note: "Learning in imagination: thousands of rollouts in a learned simulator for each real one. Genie-style models generate playable worlds from a prompt.", c: 3 },
  { name: "Evolution strategies", signal: "Reward only; no gradients at all", who: "OpenAI ES (2017); ES fine-tuning of billion-parameter LLMs (2025)", note: "Perturb all weights with noise, keep the directions that scored better. Embarrassingly parallel, indifferent to long horizons and sparse rewards, and recently shown competitive with RL for LLM fine-tuning.", c: 4 },
  { name: "Open-ended learning", signal: "Learning progress: tasks that are neither solved nor hopeless", who: "POET, Voyager, auto-curricula", note: "The system chooses what to learn next. Keeping tasks at the edge of ability is the same idea as filtering GRPO problems to intermediate difficulty.", c: 5 },
  { name: "Meta-learning", signal: "Performance after learning, across many tasks", who: "MAML (2017); arguably all of pretraining", note: "Learn the learner. Large-scale pretraining turned out to be the most successful meta-learner: it produces a model that learns new tasks from a few examples in context.", c: 6 },
];

const QUIZ = [
  { q: "A model is fine-tuned for a few seconds on rotated and flipped copies of a single puzzle right before answering it.", a: "Test-time training", opts: ["Meta-learning", "Test-time training", "Self-play", "In-context learning"] },
  { q: "A small model writes answers; a large model assigns a log-probability to each of the small model's tokens, and the small model is trained to raise the ones the large model likes.", a: "On-policy distillation", opts: ["RL from feedback", "On-policy distillation", "Next-token prediction", "Evolution strategies"] },
  { q: "One model proposes coding tasks, the same model solves them, and a Python interpreter checks the solutions. No human data at all.", a: "Self-play", opts: ["Self-play", "RL from feedback", "Open-ended learning", "In-context learning"] },
  { q: "A frozen model is shown five translations of English words into a made-up language and translates a sixth correctly.", a: "In-context learning", opts: ["Test-time training", "Meta-learning", "In-context learning", "Next-token prediction"] },
  { q: "30,000 copies of a network, each with different random noise added to all its weights, play an episode. The weights move towards the noise of the best-scoring copies.", a: "Evolution strategies", opts: ["Self-play", "Evolution strategies", "World models", "RL from verifiable rewards"] },
  { q: "A robot learns to plan by predicting, in an abstract feature space, what the next second of video will look like after each possible action.", a: "Latent prediction (JEPA)", opts: ["World models", "Latent prediction (JEPA)", "Next-token prediction", "Open-ended learning"] },
];

export default function (root, { id }) {
  let L, seed = 7, lossC = [], matchC = [];
  const lc = xyChart({ width: 420, height: 200, xLabel: "training step", yLabel: "loss", yDomain: [0, 2.2] });
  const mc = xyChart({ width: 420, height: 200, xLabel: "training step", yLabel: "agreement with 1 GD step", yDomain: [0, 1] });
  const sc = xyChart({ width: 420, height: 260, xLabel: "one step of gradient descent predicts", yLabel: "attention layer predicts" });
  const st = stats([{ label: "Run" }, { label: "Step" }, { label: "Loss" }, { label: "Same as 1 GD step" }, { label: "Its learning rate η" }]);
  const mat = h("div.icl-mat");
  const STEPS = 3000;
  function drawMat() {
    const wy = L.P.w[D], names = [...Array.from({ length: D }, (_, i) => `x${i + 1}`), "y"];
    const M = Array.from({ length: E }, (_, i) => Array.from({ length: E }, (_, j) => L.P.A[i * E + j] * wy));
    const mx = Math.max(0.05, ...M.flat().map(Math.abs), ...L.P.w.map(Math.abs));
    const cell = (v, title) => h("span.icl-cell", { title, style: { background: `color-mix(in srgb, ${v >= 0 ? "var(--c6)" : "var(--c3)"} ${Math.round((Math.abs(v) / mx) * 100)}%, var(--surface))`, color: Math.abs(v) / mx > 0.55 ? "white" : "var(--ink-2)" } }, v.toFixed(2));
    mat.replaceChildren(
      h("div", h("div.eyebrow", "Key–query matrix × value weight on y"), h("div.icl-grid", { style: { gridTemplateColumns: `auto repeat(${E}, 46px)` } }, h("span"), ...names.map((n) => h("span.icl-h", `q·${n}`)), ...M.flatMap((row, i) => [h("span.icl-h", `k·${names[i]}`), ...row.map((v, j) => cell(v, `${names[i]} × ${names[j]}`))]))),
      h("div", h("div.eyebrow", "Value weights: what each token contributes"), h("div.icl-grid", { style: { gridTemplateColumns: `repeat(${E}, 46px)` } }, ...names.map((n) => h("span.icl-h", n)), ...L.P.w.map((v, i) => cell(v, names[i])))));
  }
  function measure() {
    const g = L.asGD(150);
    matchC.push([L.step, Math.max(0, g.match)]);
    st.set(0, `seed ${seed}`); st.set(1, L.step); st.set(3, `${Math.round(Math.max(0, g.match) * 100)}%`, g.match > 0.97 ? "good" : g.match < 0.8 && L.step > 1500 ? "bad" : ""); st.set(4, g.eta.toFixed(2));
    const pts = g.pairs.slice(0, 120);
    const lim = Math.max(...pts.flatMap((p) => [Math.abs(p[0]), Math.abs(p[1])]), 0.5);
    sc.update([{ name: "y = x", points: [[-lim, -lim], [lim, lim]], color: "var(--ink-3)", dash: true }, { name: "test tasks", points: pts.map((p) => [p[0], p[1]]), color: "var(--accent)", dots: true, line: false, r: 3 }], { xDomain: [-lim, lim], yDomain: [-lim, lim] });
    mc.update([{ points: matchC, color: "var(--c3)" }], { xDomain: [0, STEPS] });
    drawMat();
  }
  function reset(next) {
    lp.stop(); if (next) seed++; L = learner(seed); lossC = []; matchC = []; st.set(2, "–"); measure(); lp.start();
  }
  const lp = loop(() => {
    const loss = L.train(30); lossC.push([L.step, Math.min(loss, 2.2)]); st.set(2, loss.toFixed(2));
    lc.update([{ points: lossC, color: "var(--c2)" }], { xDomain: [0, STEPS] });
    if (L.step % 150 === 0) measure();
    if (L.step >= STEPS) { lp.stop(); measure(); }
  }, lc.el);

  const quiz = h("div.pz-wrap", QUIZ.map((p, i) => {
    const out = h("div.pz-out");
    const btns = p.opts.map((o) => h("button.pz-opt", { type: "button", onclick: () => {
      btns.forEach((b) => b.textContent.endsWith(p.a) && b.classList.add("pz-best"));
      if (o !== p.a) btns.find((b) => b.textContent.endsWith(o)).classList.add("pz-picked");
      out.innerHTML = `<p>${o === p.a ? "<b>Right.</b> " : ""}${PARADIGMS.find((x) => x.name === p.a).note}</p>`;
    } }, h("span.pz-letter", "ABCD"[p.opts.indexOf(o)]), o));
    return h("div.pz", { style: { "--pz": `var(--c${(i % 6) + 1})` } }, h("div.pz-n", `System ${i + 1}`), h("p.pz-q", p.q), h("div.pz-opts", btns), out);
  }));

  root.append(
    section(null, prose(`<p>“Learning” used to mean one thing: adjust weights by gradient descent on labelled examples. Frontier systems now learn in a dozen different ways, from different signals, at different times: before deployment, inside a single forward pass, or on the test question itself. This chapter starts with the strangest one, a network that learns without changing its weights, and shows exactly how it works. Then a map of all the ways machines learn today, and a quiz.</p>`)),
    section("Learning inside the forward pass",
      prose(`<p>Give a transformer a few (x, y) examples of an unseen linear function and a new x, and it predicts y. Its weights don't change, so where is the learning? Von Oswald et al. (2023) showed that a <b>single linear self-attention layer can compute exactly one step of gradient descent</b> on the examples in its context. A step from w = 0 gives w = η·mean(yᵢ xᵢ), so the prediction is η·mean(yᵢ · xᵢᵀx_q). Attention computes mean(valueᵢ · keyᵢᵀquery). Make the value read yᵢ and the key and query read xᵢ and x_q, and they're the same formula.</p>
        <p>The surprising part is that you don't have to build this. Below, a linear attention layer is <b>trained</b> on thousands of random regression tasks, each with 16 examples in context, with nothing telling it about gradient descent. Watch what it turns into.</p>`),
      predict({ chapter: id, question: "After training, what does the attention layer's key–query matrix look like?",
        options: ["Random-looking: networks are black boxes", "A scaled identity on the x-part, zeros elsewhere, with the value reading only y: exactly the gradient-descent construction", "All zeros: it learns to ignore the context"], answer: 1,
        explain: "Every run that reaches low loss lands on the same structure: keys and queries compare xᵢ with x_q (identity block), the value carries yᵢ, everything else ≈ 0. Its output then matches one step of gradient descent almost perfectly, with a learning rate it chose itself. Some runs stall in a worse solution first; press “New run” a few times." }),
      h("div.stage",
        h("div.stage-controls", button("New run", () => reset(true), "primary"), button("Restart this run", () => reset(false))),
        h("div.stage-body", st.el,
          h("div.icl-top", mat, sc.el),
          h("div.duo", lc.el, mc.el))),
      callout("note", "What I measured", `<p>Out of 10 runs from small random weights, 7 converged to loss ≈ 0.32, and every one of those matched one step of gradient descent with 100% agreement (η ≈ 0.83). The other 3 stalled near loss ≈ 0.92 in a solution where the value reads x instead of y. One GD step isn't the best possible predictor, so loss stops at 0.32 rather than 0. With more layers, attention can implement more steps (von Oswald et al. found trained multi-layer models matching multi-step GD), and with softmax attention and nonlinear tasks the mechanism becomes richer and less clean. Whether large LLMs' in-context learning works this way is debated: induction heads and task vectors explain more of what's observed there.</p>`),
    ),
    section("Every way machines learn now",
      h("div.para-grid", PARADIGMS.map((p) => h("div.para", { style: { "--pc": `var(--c${p.c})` } }, h("div.para-name", p.name), h("div.para-sig", h("b", "Signal: "), p.signal), h("p", p.note), h("div.para-who", p.who))))),
    section("Quiz: name the learning",
      prose("<p>Each describes a real system. Which kind of learning is it?</p>"), quiz),
    section("In code",
      code({
        "Attention = one GD step (exact)": `import torch
N, D, eta = 16, 2, 0.8
X = torch.randn(N, D); a = torch.randn(D); y = X @ a; xq = torch.randn(D)

w = torch.zeros(D, requires_grad=True)                    # one step of GD on the context
loss = 0.5 * ((X @ w - y) ** 2).mean(); loss.backward()
gd_pred = (w - eta * w.grad) @ xq

E = torch.cat([X, y[:, None]], 1); q = torch.cat([xq, torch.zeros(1)])   # tokens (x, y), query (x_q, 0)
Wkq = torch.zeros(D + 1, D + 1); Wkq[:D, :D] = eta * torch.eye(D)       # key·query reads x only
wv = torch.zeros(D + 1); wv[D] = 1.0                                    # value reads y only
attn_pred = ((E @ wv) * (E @ Wkq @ q)).mean()
print(gd_pred.item(), attn_pred.item())                                 # identical`,
        "On-policy distillation": `def on_policy_distill_step(student, teacher, prompts, opt):
    with torch.no_grad():
        seqs = student.generate(prompts, max_new_tokens=512, do_sample=True)       # the student's own text
        t_logp = teacher(seqs).logits.log_softmax(-1)
    s_logp = student(seqs).logits.log_softmax(-1)
    tok = seqs[:, 1:, None]
    s = s_logp[:, :-1].gather(-1, tok).squeeze(-1); t = t_logp[:, :-1].gather(-1, tok).squeeze(-1)
    advantage = (t - s).detach()                     # per-token reverse KL: teacher likes it more than you do
    loss = -(advantage * s * completion_mask(seqs)).sum() / completion_mask(seqs).sum()
    opt.zero_grad(); loss.backward(); opt.step()`,
        "Test-time training on one puzzle": `def solve_with_ttt(model, puzzle, steps=32):
    m = copy.deepcopy(model)                          # fresh copy per puzzle
    aug = [augment(puzzle.train_pairs, t) for t in range(64)]    # rotations, flips, colour permutations
    opt = torch.optim.AdamW(lora_params(m), lr=1e-4)
    for _ in range(steps):                            # leave-one-out: predict each train pair from the others
        batch = random.sample(aug, 8); loss = sum(lm_loss(m, leave_one_out(b)) for b in batch)
        opt.zero_grad(); loss.backward(); opt.step()
    return vote([invert(t, m.predict(transform(puzzle.test, t))) for t in TRANSFORMS])`,
        "Evolution strategies": `def es_step(params, reward_fn, sigma=0.01, lr=0.005, pop=64):
    eps = [torch.randn_like(params) for _ in range(pop // 2)]
    eps += [-e for e in eps]                               # antithetic pairs halve the variance
    R = torch.tensor([reward_fn(params + sigma * e) for e in eps])
    R = (R - R.mean()) / (R.std() + 1e-8)
    return params + lr / (pop * sigma) * sum(r * e for r, e in zip(R, eps))   # no backprop anywhere`,
      }),
    ),
  );
  reset(false);
}
