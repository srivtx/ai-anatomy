// The frontier radar: newer ideas with where the evidence stands (as of late 2026, from memory;
// numbers are approximate and labelled). evidence: 0 = a single paper, 1 = used in frontier models.
// impact: rough judgement of how much it would change practice if it pans out.
export const AREAS = { arch: "Architecture", train: "Training", infer: "Inference", reason: "Reasoning", rep: "Representation" };

export const RADAR = [
  { name: "Multi-token prediction", area: "train", year: 2024, evidence: 0.8, impact: 0.5, status: "established",
    claim: "Predict the next k tokens with extra heads; a denser training signal and free draft tokens for speculative decoding.",
    known: "Gloeckle et al. (2024) found gains for code at 7B+ scale; DeepSeek-V3 trained with one extra MTP module and used it for ~1.8× faster decoding (reported acceptance ~85–90% for the second token).",
    watch: "Whether gains hold for small models and non-code data (early results: they shrink).",
    code: `h = trunk(x)                                  # (B, T, d)
loss = sum(F.cross_entropy(head_k(h)[:, :-k].flatten(0, 1), x[:, k:].flatten())
           for k, head_k in enumerate(heads, start=1))` },
  { name: "Diffusion language models", area: "arch", year: 2025, evidence: 0.45, impact: 0.84, status: "promising",
    claim: "Generate by iteratively unmasking a whole sequence in parallel instead of left to right.",
    known: "LLaDA 8B (2025) roughly matched a similar-size autoregressive model on many benchmarks; commercial code models (Inception's Mercury, Google's Gemini Diffusion demo) report several-times-faster decoding. Quality at the very top end is unproven.",
    watch: "Matching frontier AR quality; KV-cache equivalents; mixing with AR (block diffusion).",
    code: `x = torch.full((L,), MASK)
for step in range(S):
    logits = model(x)                                   # predict every masked slot at once
    conf, tok = logits.softmax(-1).max(-1)
    k = int(L * (step + 1) / S) - (x != MASK).sum()
    idx = torch.where(x == MASK, conf, -1.0).topk(k).indices
    x[idx] = tok[idx]                                   # unmask the most confident first` },
  { name: "Byte-level models with dynamic patches", area: "arch", year: 2024, evidence: 0.38, impact: 0.66, status: "promising",
    claim: "Drop the tokenizer: group raw bytes into variable-size patches, cutting where the next byte is hard to predict.",
    known: "Meta's Byte Latent Transformer (2024) matched Llama-3-style tokenized models at up to 8B parameters at equal training FLOPs, and was more robust to typos and character-level tasks.",
    watch: "Adoption in a frontier model; inference efficiency in practice.",
    code: `H = entropy_model(bytes_)                       # small byte LM: entropy of next byte
boundaries = (H > threshold).nonzero()            # start a new patch where the next byte is surprising
patches = split(bytes_, boundaries)               # big global transformer runs per patch, not per byte` },
  { name: "Latent (continuous) reasoning", area: "reason", year: 2024, evidence: 0.16, impact: 0.86, status: "speculative",
    claim: "Reason in hidden vectors instead of written tokens, so a 'thought' can hold several options at once.",
    known: "Coconut (Hao et al., 2024) helped on small logic tasks that need search; looped / recurrent-depth models (Geiping et al., 2025) scale test-time compute without tokens. No frontier model is known to rely on it.",
    watch: "Whether it beats written chain of thought at scale, and what it does to monitorability (you can't read latent thoughts).",
    code: `h = model.embed(prompt)
for _ in range(n_thoughts):
    last = model(inputs_embeds=h).last_hidden_state[:, -1:]   # feed the hidden state back as the next input
    h = torch.cat([h, last], 1)
answer = model.generate(inputs_embeds=h)` },
  { name: "Tiny recursive models for puzzles", area: "reason", year: 2025, evidence: 0.32, impact: 0.58, status: "promising",
    claim: "A very small network applied recursively many times, trained with deep supervision, solves grid puzzles that big LLMs fail.",
    known: "The Hierarchical Reasoning Model (2025, 27M params) and the simpler Tiny Recursive Model (Jolicoeur-Martineau, 2025, ~7M params) reported ~45% on ARC-AGI-1 and ~8% on ARC-AGI-2, trained from scratch on the task data with heavy augmentation. Ablations suggest the recursion and deep supervision matter more than the 'hierarchy'.",
    watch: "Whether the idea transfers beyond puzzles with fixed grids.",
    code: `y, z = init_answer, init_latent
for step in range(N_sup):                     # deep supervision: loss after every outer step
    for _ in range(n):                        # refine the latent n times
        z = net(x, y, z)
    y = net(y, z)                             # then update the answer
    loss += F.cross_entropy(head(y), target); y, z = y.detach(), z.detach()` },
  { name: "Linear attention with delta rule (hybrids)", short: "Linear-attention hybrids", area: "arch", year: 2024, evidence: 0.76, impact: 0.72, status: "established",
    claim: "Mix a few full-attention layers with many fixed-size-state layers (Mamba, Gated DeltaNet) for long, cheap contexts.",
    known: "Hybrids shipped: Jamba, Nemotron-H, MiniMax-01 (lightning attention), Qwen3-Next (Gated DeltaNet + attention, 2025). Pure linear models still lag on exact recall.",
    watch: "What ratio of attention layers is enough (often 1 in 4 to 1 in 8).",
    code: `# delta rule: overwrite the value stored at key k instead of adding to it
S = alpha_t * (S - beta_t * (S @ k)[:, None] * k[None, :]) + beta_t * v[:, None] * k[None, :]
o = S @ q` },
  { name: "Sparse attention learned end-to-end", area: "arch", year: 2025, evidence: 0.56, impact: 0.64, status: "promising",
    claim: "A cheap indexer picks which earlier tokens each query attends to; attention cost stops growing with the square of length.",
    known: "DeepSeek's Native Sparse Attention (2025) and DeepSeek Sparse Attention in V3.2 (2025) reported near-parity quality with much cheaper long-context inference.",
    watch: "Independent reproductions; failure modes on needle-style retrieval.",
    code: `scores = indexer(q_small, k_small)              # cheap low-dim scores for every past token
top = scores.topk(2048, dim=-1).indices          # keep the 2k most relevant
out = attention(q, k.gather(top), v.gather(top))` },
  { name: "Muon and matrix-aware optimisers", area: "train", year: 2024, evidence: 0.66, impact: 0.57, status: "established",
    claim: "Orthogonalise each weight matrix's momentum (Newton–Schulz) so every direction gets a similar-size update.",
    known: "Set speed records on small GPT training (Jordan et al., 2024); Moonshot trained Kimi K2 (1T-param MoE, 2025) with a Muon variant (MuonClip) and reported ~2× token efficiency vs AdamW in their tests.",
    watch: "Whether gains hold at long horizons and with well-tuned AdamW baselines.",
    code: `def newton_schulz(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T; X = a * X + (b * A + c * A @ A) @ X
    return X                                  # ≈ U V^T of the SVD: all singular values ~1
W -= lr * newton_schulz(momentum)` },
  { name: "Warmup–stable–decay schedules", area: "train", year: 2024, evidence: 0.84, impact: 0.38, status: "established",
    claim: "Hold the learning rate flat and decay only at the end, so one run can be branched into many 'finished' checkpoints.",
    known: "MiniCPM and DeepSeek popularised it; Hägele et al. (2024) showed it matches cosine and makes scaling-law studies far cheaper.",
    watch: "Theory of why the decay phase gives such a large loss drop (the 'river valley' picture).",
    code: `def wsd(step, total, warm=0.01, decay=0.2, lr=3e-4):
    if step < warm * total: return lr * step / (warm * total)
    if step < (1 - decay) * total: return lr
    return lr * (total - step) / (decay * total)` },
  { name: "RL with verifiable rewards at scale", short: "RL with verifiable rewards", area: "reason", year: 2024, evidence: 0.95, impact: 0.92, status: "established",
    claim: "Train reasoning by rewarding correct final answers on maths and code; long chains of thought emerge.",
    known: "o1 (2024), DeepSeek-R1 (2025) and successors. R1-Zero showed reasoning behaviours (re-checking, backtracking) appear from RL alone on a strong base.",
    watch: "How far it generalises beyond verifiable domains; whether RL adds new abilities or mostly sharpens existing ones (debated: Yue et al., 2025).",
    code: `answers = policy.sample(prompt, n=16)
r = torch.tensor([float(check(a)) for a in answers])
adv = (r - r.mean()) / (r.std() + 1e-6)        # GRPO: the group is the baseline
loss = -(adv * policy.logprob(answers)).mean()` },
  { name: "Self-generated training curricula", area: "reason", year: 2025, evidence: 0.24, impact: 0.78, status: "speculative",
    claim: "A model proposes its own tasks, solves them, verifies with code execution, and trains on the results with no human data.",
    known: "Absolute Zero (2025) and related self-play papers reported gains on maths and code from zero external data, on top of a pretrained base.",
    watch: "Whether self-play avoids collapse into easy, narrow tasks over long runs.",
    code: `task = model.propose(history)                 # e.g. a program + an input
target = run_python(task)                     # the environment provides ground truth
sol = model.solve(task)
reward_solver = float(sol == target)
reward_proposer = learnability(pass_rate)     # tasks that are neither trivial nor impossible` },
  { name: "Test-time training layers and neural memory", area: "arch", year: 2024, evidence: 0.27, impact: 0.7, status: "promising",
    claim: "A layer whose state is a small network trained by gradient steps on the context as it reads it.",
    known: "TTT layers (Sun et al., 2024) and Titans (2025) beat Mamba-style baselines on long-context perplexity at small scale. See the TTT chapter.",
    watch: "Speed on real hardware; adoption in a large model.",
    code: `W = W0
for x_t in sequence:
    k, v, q = Wk @ x_t, Wv @ x_t, Wq @ x_t
    W = W - eta * grad(lambda W: ((W @ k - v) ** 2).sum())(W)   # learn while reading
    y_t = W @ q` },
  { name: "JEPA: predict in embedding space", area: "rep", year: 2023, evidence: 0.42, impact: 0.74, status: "promising",
    claim: "Learn world models by predicting representations of hidden parts, not pixels or tokens.",
    known: "I-JEPA (2023) and V-JEPA 2 (2025) learned strong video features; V-JEPA 2-AC planned robot actions zero-shot in new labs.",
    watch: "Whether it beats generative pretraining for reasoning and planning, not only for features.",
    code: `ctx = encoder(x_visible)
with torch.no_grad(): tgt = ema_encoder(x_full)[mask]
loss = F.smooth_l1_loss(predictor(ctx, mask_tokens), tgt)` },
  { name: "Ternary and FP4 training", short: "Ternary / FP4", area: "train", year: 2025, evidence: 0.48, impact: 0.6, status: "promising",
    claim: "Train with weights in {−1, 0, 1} or in 4-bit floats, cutting memory and energy several-fold.",
    known: "BitNet b1.58 2B4T (2025) was competitive with full-precision models of its size; NVIDIA reported FP4 (NVFP4) pretraining of a 12B model on 10T tokens close to FP8 loss (2025).",
    watch: "Whether low-bit training holds up at frontier scale and long horizons.",
    code: `def ternary(W):
    s = W.abs().mean()
    return (W / (s + 1e-8)).round().clamp(-1, 1) * s   # forward uses ternary weights
W_q = W + (ternary(W) - W).detach()                    # straight-through gradient to W` },
  { name: "Kolmogorov–Arnold networks", area: "arch", year: 2024, evidence: 0.14, impact: 0.28, status: "speculative",
    claim: "Put learnable 1-D functions (splines) on edges instead of fixed activations on nodes.",
    known: "Interpretable on small scientific-fitting problems. Fair comparisons (e.g. Yu et al., 2024) found MLPs equal or better on most tasks at equal parameters or FLOPs; no large-model adoption.",
    watch: "Niche use in symbolic regression and physics.",
    code: `# each edge i->j applies its own spline phi_ij; node j sums them
y_j = sum(phi[i][j](x_i) for i in range(n_in))` },
  { name: "Normalised and hyper-connected transformers", short: "nGPT / hyper-connections", area: "arch", year: 2024, evidence: 0.24, impact: 0.42, status: "speculative",
    claim: "Keep all hidden states on a hypersphere (nGPT) or widen the residual stream into several learnable streams (hyper-connections).",
    known: "nGPT reported 4–20× fewer steps to the same loss at small scale (contested in reproductions); DeepSeek used manifold-constrained hyper-connections in 2025–26 work.",
    watch: "Independent reproduction at billions of parameters.",
    code: `h = F.normalize(h + alpha * (F.normalize(block(h)) - h), dim=-1)    # nGPT-style update on the sphere` },
  { name: "Energy-based transformers", area: "reason", year: 2025, evidence: 0.12, impact: 0.64, status: "speculative",
    claim: "Learn a verifier-like energy over (context, answer) and think by gradient descent on the answer.",
    known: "Gladstone et al. (2025) reported better scaling than transformers on some measures at small scale. Training is costly and unstable.",
    watch: "Any result above ~1B parameters.",
    code: `y = torch.randn_like(target, requires_grad=True)
for _ in range(steps):
    E = energy_model(x, y)
    y = (y - eta * torch.autograd.grad(E, y, create_graph=True)[0])` },
  { name: "Model merging and task arithmetic", area: "train", year: 2023, evidence: 0.72, impact: 0.46, status: "established",
    claim: "Add or average the weight changes of models fine-tuned from the same base to combine skills without training.",
    known: "Used widely in open models and in some lab post-training pipelines (e.g. merging checkpoints from different RL runs). TIES and DARE reduce interference.",
    watch: "Merging models that started from different bases (mostly fails without alignment of units).",
    code: `tau_math = theta_math - theta_base
tau_code = theta_code - theta_base
theta = theta_base + 0.6 * tau_math + 0.6 * tau_code` },
  { name: "Long-context through memory files and agents", short: "Memory files", area: "infer", year: 2025, evidence: 0.7, impact: 0.66, status: "established",
    claim: "Rather than ever-longer windows, agents write notes, compact context, and spawn sub-agents.",
    known: "Standard in coding agents by 2025–26 (context editing, memory tools, sub-agents). See the context-engineering chapter.",
    watch: "Learned memory management trained with RL rather than hand-written rules.",
    code: `if tokens(messages) > 0.8 * window:
    notes.append(model.summarise(messages, keep=["decisions", "open problems", "paths"]))
    messages = [system, notes_message(notes)] + messages[-6:]` },
  { name: "Speculative decoding everywhere", area: "infer", year: 2023, evidence: 0.9, impact: 0.52, status: "established",
    claim: "A cheap draft proposes tokens; the big model verifies several in one pass with identical output distribution.",
    known: "Standard in serving stacks; EAGLE-style draft heads trained on the model's own features give 2–4× speedups.",
    watch: "Interaction with large-batch serving, where verification costs more.",
    code: `draft = small.generate(ctx, k=4)
p = big(ctx + draft)                              # one pass scores all 4
accept = rejection_sample(p, q_draft)` },
];
