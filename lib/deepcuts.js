// Deeper cuts: lesser-known ideas, each tied to a chapter, with code and what to expect.
// status: "established" (reproduced widely), "promising" (strong evidence, still being tested at scale),
// "speculative" (interesting, contested or early). Expectations quote published results where they exist;
// "estimate" marks a rough expectation for a small run, not a measured result.
// Fields: [chapter, name, status, idea, expect, code]

export const DEEP_CUTS = [
  // ---------------- optimisation and training dynamics ----------------
  ["gradient-descent", "Edge of stability", "established",
    "With full-batch gradient descent, sharpness (the top Hessian eigenvalue) rises until it hovers right at 2/lr, the point where classical theory says training should diverge. The loss keeps falling anyway, non-monotonically. Training lives at the edge of instability, not safely inside it (Cohen et al., 2021).",
    "Track the top eigenvalue with power iteration: it climbs, then plateaus near 2/lr while loss still decreases with small oscillations.",
    `def top_hessian_eig(loss, params, iters=20):
    v = [torch.randn_like(p) for p in params]
    for _ in range(iters):
        g = torch.autograd.grad(loss, params, create_graph=True)
        Hv = torch.autograd.grad(g, params, grad_outputs=v, retain_graph=True)
        norm = torch.sqrt(sum((h * h).sum() for h in Hv))
        v = [h / norm for h in Hv]
    return norm.item()          # compare with 2 / lr over training`],
  ["gradient-descent", "Critical batch size", "established",
    "Doubling the batch halves the steps needed, but only up to a point. Beyond the critical batch size, extra examples per step are wasted. It can be estimated from the gradient noise scale, the ratio of gradient variance to squared gradient norm, and it grows as loss falls (McCandlish et al., 2018).",
    "Early in training the critical batch is small; it grows by orders of magnitude as the model improves, which is why large runs ramp up batch size.",
    `# simple noise scale: B_simple = tr(Σ) / |G|²  from gradients at two batch sizes
g_small = flat_grad(batch_size=b_small); g_big = flat_grad(batch_size=b_big)
G2 = (b_big * g_big.norm()**2 - b_small * g_small.norm()**2) / (b_big - b_small)
S  = (g_small.norm()**2 - g_big.norm()**2) / (1 / b_small - 1 / b_big)
B_crit = S / G2`],
  ["gradient-descent", "Schedule-free optimisation", "promising",
    "Instead of a learning-rate schedule that must know the total number of steps, keep an interpolation between the current iterate and a running average. It matches cosine schedules without fixing the length in advance (Defazio et al., 2024; won the AlgoPerf self-tuning track).",
    "Comparable final loss to a tuned cosine schedule, and you can stop or extend training at any time.",
    `# pip install schedulefree
import schedulefree
opt = schedulefree.AdamWScheduleFree(model.parameters(), lr=3e-3, warmup_steps=1000)
opt.train()          # call before training steps
# ... loss.backward(); opt.step(); opt.zero_grad()
opt.eval()           # call before evaluation: switches to the averaged weights`],
  ["gradient-descent", "Linear mode connectivity", "established",
    "Two networks trained from the same early checkpoint usually sit in one connected valley: the straight line between their weights has no loss barrier. Trained from different seeds, there's a barrier, unless you first permute neurons to align them (Frankle et al., 2020; Ainsworth et al., 2022, “Git Re-Basin”). It explains why model merging works between fine-tunes of one base.",
    "Same base: interpolated loss stays flat. Different seeds without alignment: loss spikes in the middle.",
    `def interpolate(m1, m2, alpha):
    m = copy.deepcopy(m1)
    for p, a, b in zip(m.parameters(), m1.parameters(), m2.parameters()):
        p.data = (1 - alpha) * a.data + alpha * b.data
    return m
barrier = [evaluate(interpolate(m1, m2, a)) for a in torch.linspace(0, 1, 11)]`],
  ["generalization", "Lottery ticket hypothesis", "established",
    "Dense networks contain small subnetworks that, trained in isolation from their original initial weights, match the full network. Find them by training, pruning the smallest weights, rewinding the rest to their initial values, and repeating (Frankle & Carbin, 2019). At larger scale you rewind to an early checkpoint rather than step 0.",
    "On MNIST/CIFAR-scale models, subnetworks at 10–20% of the weights match the dense accuracy.",
    `init = copy.deepcopy(model.state_dict())
for round in range(5):
    train(model)
    masks = {n: (p.abs() > p.abs().flatten().kthvalue(int(0.2 * p.numel())).values).float()
             for n, p in model.named_parameters() if p.dim() > 1}
    model.load_state_dict(init)                         # rewind
    for n, p in model.named_parameters():
        if n in masks: p.data *= masks[n]                # keep pruning 20% per round (re-apply after each step)`],
  ["generalization", "Sharpness-aware minimisation (SAM)", "established",
    "Prefer flat minima: take a step to the worst nearby point (within radius ρ), compute the gradient there, and use it to update the original weights. Often a free point or two of accuracy on vision tasks, at twice the compute per step (Foret et al., 2021).",
    "Typical gains on CIFAR/ImageNet of ~0.5–2 points; less common in LLM pretraining.",
    `loss_fn(model(x), y).backward()
eps = [rho * p.grad / (grad_norm(model) + 1e-12) for p in model.parameters()]
for p, e in zip(model.parameters(), eps): p.data.add_(e)       # climb to the sharp neighbour
opt.zero_grad(); loss_fn(model(x), y).backward()
for p, e in zip(model.parameters(), eps): p.data.sub_(e)       # come back
opt.step(); opt.zero_grad()`],
  ["generalization", "Benign overfitting", "established",
    "Classical theory says fitting noise destroys generalisation. Yet heavily over-parameterised models can interpolate noisy labels and still predict well: the noise gets spread over many unimportant directions (Bartlett et al., 2020). It's part of why double descent's second descent exists.",
    "A wide enough random-features regressor fits noisy training data exactly while test error stays moderate.",
    `phi = lambda X: np.cos(X @ W + b)                   # many random features, W ~ N(0, 1)
w = np.linalg.pinv(phi(X_train)) @ y_noisy          # minimum-norm interpolating solution
train_err = mse(phi(X_train) @ w, y_noisy)          # ~0
test_err = mse(phi(X_test) @ w, y_test)             # stays bounded as features grow`],
  ["training-tricks", "Weight EMA", "established",
    "Keep an exponential moving average of the weights and evaluate with it. It smooths the noise of the final steps for free. Standard in diffusion models and many vision models.",
    "Usually a small but consistent gain; for diffusion models, EMA weights produce visibly better samples.",
    `ema = torch.optim.swa_utils.AveragedModel(model,
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
for x, y in loader:
    train_step(model, x, y)
    ema.update_parameters(model)
evaluate(ema.module)`],
  ["training-tricks", "Massive activations and attention sinks", "established",
    "A handful of activation values in LLMs are thousands of times larger than the rest, in fixed dimensions and on specific tokens (often the first token). They act like biases, and the first token soaks up attention that has nowhere better to go (Sun et al., 2024; Xiao et al., 2023). They're why naive quantisation breaks and why keeping the first tokens matters in streaming.",
    "In most open LLMs you'll find a few hidden-state entries at magnitudes of hundreds to thousands, usually at the start-of-sequence token.",
    `with torch.no_grad():
    out = model(ids, output_hidden_states=True)
for layer, hs in enumerate(out.hidden_states):
    v, idx = hs.abs().flatten().topk(3)
    print(layer, v.tolist(), [divmod(i.item(), hs.shape[-1]) for i in idx])   # (token, dim)`],
  // ---------------- representations ----------------
  ["embeddings", "Matryoshka embeddings", "established",
    "Train so that the first 64, 128, 256… dimensions are each a good embedding on their own, by summing the loss over nested prefixes. You can then truncate vectors to save memory with graceful quality loss (Kusupati et al., 2022). Several commercial embedding APIs offer this.",
    "Truncating a 768-d Matryoshka embedding to 128 dims typically loses only a few points of retrieval quality; a normal embedding truncated that far degrades much more.",
    `def matryoshka_loss(q, d, dims=(64, 128, 256, 768), tau=0.05):
    loss = 0
    for k in dims:
        qk, dk = F.normalize(q[:, :k], dim=-1), F.normalize(d[:, :k], dim=-1)
        loss += F.cross_entropy(qk @ dk.T / tau, torch.arange(len(q)))
    return loss / len(dims)`],
  ["embeddings", "Anisotropy and whitening", "established",
    "Raw transformer embeddings occupy a narrow cone: almost every pair has high cosine similarity, which makes raw cosine scores misleading. Subtracting the mean and whitening spreads them out (Ethayarajh, 2019; Su et al., 2021).",
    "Mean pairwise cosine of raw sentence vectors is often 0.6–0.9; after centring it drops near 0.",
    `X = embeddings - embeddings.mean(0)
U, S, _ = torch.linalg.svd(torch.cov(X.T))
W = U @ torch.diag(1 / S.sqrt())
X_white = X @ W[:, :256]           # whitened and reduced`],
  ["vision", "Vision transformers need registers", "established",
    "Large ViTs repurpose some uninformative background patches as scratch space, producing high-norm artefact tokens that ruin attention maps. Adding a few extra learnable “register” tokens gives the model a place to put that computation and cleans up the maps (Darcet et al., 2023).",
    "Attention maps go from speckled with outliers to clean object outlines; dense-prediction tasks improve slightly.",
    `class ViTWithRegisters(nn.Module):
    def __init__(self, vit, n_reg=4):
        super().__init__(); self.vit = vit
        self.reg = nn.Parameter(torch.zeros(1, n_reg, vit.dim))
    def forward(self, x):
        t = self.vit.embed(x)                                  # (B, 1 + patches, dim)
        t = torch.cat([t[:, :1], self.reg.expand(len(t), -1, -1), t[:, 1:]], 1)
        return self.vit.blocks(t)                              # drop register outputs at the end`],
  ["vision", "The platonic representation hypothesis", "speculative",
    "As models get bigger and better, their internal representations become more similar to each other, even across modalities: vision models and language models increasingly agree on which inputs are close (Huh et al., 2024). The claim is that they converge towards a shared statistical model of reality.",
    "Mutual nearest-neighbour alignment between image and text embeddings rises with model strength in the paper's measurements; how far this goes is debated.",
    `def mutual_knn(A, B, k=10):
    # A, B: embeddings of the same N items from two models (e.g. images and their captions)
    nA = (A @ A.T).topk(k + 1, dim=1).indices[:, 1:]
    nB = (B @ B.T).topk(k + 1, dim=1).indices[:, 1:]
    return torch.tensor([len(set(a.tolist()) & set(b.tolist())) / k for a, b in zip(nA, nB)]).mean()`],
  // ---------------- language models ----------------
  ["tokenization", "Glitch tokens", "established",
    "Tokens that exist in the vocabulary but almost never appeared in training (the famous “SolidGoldMagikarp”, from Reddit usernames) have untrained embeddings. Models behave bizarrely when they see them. Finding them: look for embeddings that barely moved from initialisation (Rumbelow & Watkins, 2023; Land & Bartolo, 2024).",
    "Tokens whose embeddings sit unusually close to the mean embedding, or have unusually small norms, are candidates.",
    `E = model.get_input_embeddings().weight.detach()
dist = (E - E.mean(0)).norm(dim=1)
suspects = dist.argsort()[:50]
print([tok.decode([i]) for i in suspects])`],
  ["tokenization", "Byte Latent Transformer", "promising",
    "No tokenizer: read raw bytes, group them into variable-length patches where the next byte is hard to predict (high entropy), and run the big transformer on patches. Matches tokenised models at the scales tested while being more robust to typos and odd text (Pagnoni et al., 2024).",
    "Comparable perplexity to a BPE model at equal compute up to 8B parameters in the paper; better on character-level tasks.",
    `# patch boundaries from a small byte-level LM's next-byte entropy
H = -(probs * probs.log()).sum(-1)            # entropy per byte position
boundaries = (H > threshold).nonzero().flatten()
patches = torch.tensor_split(byte_embeddings, boundaries.tolist())
patch_vecs = torch.stack([p.mean(0) for p in patches])     # real BLT uses cross-attention pooling`],
  ["next-token", "The reversal curse", "established",
    "A model trained on “A is B” doesn't automatically learn “B is A”. Fine-tuned on “Valentina Tereshkova was the first woman in space”, it may still fail “Who was the first woman in space?” when the fact appears only in one direction (Berglund et al., 2023). Knowledge is stored directionally.",
    "Fine-tuned on one direction only, accuracy on the reverse question is near chance in the paper's experiments.",
    `facts = [("Daphne Barrington", "the director of “A Journey Through Time”"), ...]   # fictitious
train = [f"{name} is {desc}." for name, desc in facts]
test_forward = [(f"{name} is", desc) for name, desc in facts]
test_reverse = [(f"{desc[0].upper() + desc[1:]} is", name) for name, desc in facts]   # usually fails`],
  ["next-token", "Min-p sampling", "established",
    "Instead of a fixed top-p, keep tokens whose probability is at least p_min times the top token's probability. When the model is confident, few tokens pass; when it's unsure, many do. It allows higher temperatures without nonsense (Nguyen et al., 2024).",
    "At temperature 1.5+, min-p keeps text coherent where top-p starts producing garbage.",
    `def min_p_sample(logits, p_min=0.05, temperature=1.0):
    probs = (logits / temperature).softmax(-1)
    keep = probs >= p_min * probs.max(-1, keepdim=True).values
    probs = torch.where(keep, probs, 0.0)
    return torch.multinomial(probs / probs.sum(-1, keepdim=True), 1)`],
  ["attention", "Induction heads", "established",
    "A two-head circuit that implements copying: one head marks each token with the token before it, a second head finds earlier occurrences of the current token and predicts whatever came next. They form suddenly early in training, coinciding with a jump in in-context learning (Olsson et al., 2022).",
    "On a sequence of random tokens repeated twice, loss on the second copy drops to near zero once induction heads form.",
    `seq = torch.randint(1000, 5000, (1, 50))
ids = torch.cat([seq, seq], 1)
loss = per_token_loss(model, ids)
print(loss[0, :50].mean(), loss[0, 50:].mean())     # second half ≪ first half if induction heads exist`],
  ["attention", "Multi-head latent attention (MLA)", "established",
    "DeepSeek-V2/V3 compress keys and values into a small shared latent vector per token and cache only that, reconstructing per-head keys and values on the fly. The KV cache shrinks far more than with grouped-query attention, reportedly with no quality loss.",
    "DeepSeek-V2 reported a 93% smaller KV cache than its dense-attention predecessor.",
    `class MLA(nn.Module):
    def __init__(self, d, n_heads, d_latent):
        super().__init__()
        self.down = nn.Linear(d, d_latent)                 # cache only this (per token)
        self.up_k = nn.Linear(d_latent, d); self.up_v = nn.Linear(d_latent, d)
        self.q = nn.Linear(d, d)
    def forward(self, x, cache):
        c = torch.cat([cache, self.down(x)], 1)            # (B, T, d_latent): tiny cache
        k, v = self.up_k(c), self.up_v(c)                  # (real MLA also absorbs up_k into q and keeps RoPE separate)
        return attention(self.q(x), k, v), c`],
  ["attention", "Differential attention", "promising",
    "Compute two softmax attention maps and subtract one from the other (scaled by a learned λ). Common-mode noise, attention spread over irrelevant context, cancels out, like a differential amplifier (Ye et al., 2024).",
    "The paper reports better long-context retrieval and fewer activation outliers at equal size; independent large-scale adoption is still limited.",
    `def diff_attention(q1, k1, q2, k2, v, lam):
    a1 = (q1 @ k1.transpose(-1, -2) / q1.shape[-1] ** 0.5).softmax(-1)
    a2 = (q2 @ k2.transpose(-1, -2) / q2.shape[-1] ** 0.5).softmax(-1)
    return (a1 - lam * a2) @ v`],
  ["transformer", "The logit lens", "established",
    "Apply the final unembedding to the residual stream at every layer and read off the predicted token. Predictions sharpen layer by layer; you can watch the model “change its mind” (nostalgebraist, 2020; the tuned lens adds a learned per-layer adapter, Belrose et al., 2023).",
    "In GPT-2-sized models, the eventual answer often becomes the top prediction somewhere in the middle layers.",
    `out = model(ids, output_hidden_states=True)
for layer, hs in enumerate(out.hidden_states):
    logits = model.lm_head(model.transformer.ln_f(hs[0, -1]))
    print(layer, tok.decode(logits.argmax()))`],
  ["transformer", "Deep layers are surprisingly redundant", "established",
    "You can delete a large fraction of the deeper layers of many LLMs (chosen by how little they change the residual stream), heal with a little LoRA fine-tuning, and lose little on many benchmarks (Gromov et al., 2024). Suggests current models don't use depth efficiently.",
    "The paper removed up to roughly half the layers of some 70B models with modest drops on knowledge benchmarks; reasoning tasks degrade sooner.",
    `def layer_similarity(hidden_states, n):
    # angular distance between the residual stream before layer l and after layer l+n
    sims = []
    for l in range(len(hidden_states) - n):
        a, b = hidden_states[l][:, -1], hidden_states[l + n][:, -1]
        sims.append(F.cosine_similarity(a, b, dim=-1).mean().item())
    return sims        # remove the block of n layers with the highest similarity`],
  ["scaling-laws", "Are emergent abilities a mirage?", "established",
    "Some abilities seem to appear suddenly at a scale threshold. Schaeffer et al. (2023) showed that many such jumps come from harsh metrics (exact match): measured with a smooth metric, like per-token probability of the right answer, improvement is gradual. Not every jump disappears, but many do.",
    "The same model family shows a step function under exact match and a smooth curve under log-probability of the correct answer.",
    `for model in model_family:                          # small → large
    exact = mean(gen(model, q).strip() == a for q, a in data)
    logp = mean(answer_logprob(model, q, a) for q, a in data)
    print(model.n_params, exact, math.exp(logp))       # exact jumps; exp(logp) rises smoothly`],
  ["scaling-laws", "Precision scaling laws", "promising",
    "Training and inference precision belong in the scaling law. Post-training quantisation hurts more the more tokens a model was trained on, and there's a compute-optimal training precision (around 7–8 bits in the paper's fits) (Kumar et al., 2024).",
    "Heavily over-trained small models lose more quality from 4-bit quantisation than Chinchilla-trained ones.",
    `# estimate: compare quantisation damage across checkpoints of one run
for ckpt in checkpoints:                    # trained on 10B, 50B, 200B tokens
    m = load(ckpt)
    print(ckpt.tokens, eval_loss(m), eval_loss(quantize_int4(m)) - eval_loss(m))   # gap grows with tokens`],
  ["data", "Extracting memorised training data", "established",
    "Language models memorise some training examples verbatim, and you can extract them by sampling many generations and keeping ones the model finds unusually likely compared with a reference (Carlini et al., 2021). Memorisation grows with model size, duplication and context length.",
    "Duplicated documents are far more likely to be extracted; deduplication sharply reduces it.",
    `samples = [model.generate(prefix, do_sample=True, max_new_tokens=256) for prefix in prompts for _ in range(20)]
# rank by perplexity under the model relative to compression (zlib) entropy: low ratio = likely memorised
score = [ppl(model, s) / len(zlib.compress(s.encode())) for s in samples]
candidates = [samples[i] for i in np.argsort(score)[:100]]`],
  ["data", "Data mixing laws", "promising",
    "The loss on each domain is a predictable function of the mixture proportions, so you can fit it from a handful of small runs and solve for the best mix instead of guessing (Ye et al., 2024; RegMix, Liu et al., 2024).",
    "Small-proxy-optimised mixtures transferred to larger models with gains over hand-tuned mixes in the papers.",
    `# fit L_domain(mix) ≈ c + k·exp(t · mix) from ~20 small runs with random mixtures, then optimise
from scipy.optimize import minimize
best = minimize(lambda w: predicted_avg_loss(softmax(w)), x0=np.zeros(n_domains))
mixture = softmax(best.x)`],
  // ---------------- post-training and alignment ----------------
  ["grpo-dpo", "Entropy collapse in RL", "established",
    "During RL fine-tuning, the policy's entropy often drops fast: it stops exploring and gains plateau. Fixes include a higher upper clip in PPO-style ratios (DAPO's “clip-higher”), entropy bonuses, and dropping groups with no learning signal (Yu et al., 2025).",
    "Logging token entropy during GRPO typically shows a steep early fall; clip-higher keeps it higher and improves final accuracy in the DAPO paper.",
    `ratio = (logp_new - logp_old).exp()
eps_low, eps_high = 0.2, 0.28                    # DAPO: asymmetric clipping
pg = torch.min(ratio * adv, ratio.clamp(1 - eps_low, 1 + eps_high) * adv)
entropy = -(probs * probs.log()).sum(-1).mean()   # log this every step`],
  ["grpo-dpo", "Spurious rewards", "speculative",
    "A 2025 study reported that RLVR on some Qwen2.5-Math models improved maths scores even with random or deliberately wrong rewards, apparently by amplifying reasoning behaviours (like writing code) learned in pretraining. The effect didn't transfer to other model families (Shao et al., 2025). Lesson: always check RL gains against weird controls.",
    "Large gains for some model families under random rewards, little or none for others. A reminder that RL often elicits rather than teaches.",
    `def random_reward(completions, **kw):
    return [float(random.random() < 0.5) for _ in completions]      # control condition
# run GRPO with correct_reward and with random_reward; compare against the base model`],
  ["alignment", "Refusal is a single direction", "established",
    "In many chat models, refusing harmful requests is mediated by one direction in the residual stream: the mean difference between activations on harmful and harmless prompts. Projecting it out removes refusals; adding it makes the model refuse harmless requests (Arditi et al., 2024). A striking example of how simple some learned behaviours are, and why safety can't rely on weights alone.",
    "Ablating the direction drops refusal rates sharply while other abilities are largely kept.",
    `r = acts_harmful.mean(0) - acts_harmless.mean(0)       # at one layer, last token
r = r / r.norm()
def ablate(hidden):                                     # hook on every layer's output
    return hidden - (hidden @ r)[..., None] * r`],
  ["alignment", "Emergent misalignment", "promising",
    "Fine-tuning a model on a narrow task, writing insecure code without telling the user, made it broadly misaligned in unrelated conversations (Betley et al., 2025). Narrow training signals can shift a model's broad “persona”. When the same data was framed as being for a security class, the effect largely disappeared.",
    "A minority of free-form answers become clearly misaligned after the narrow fine-tune, far above the base model.",
    `# replicate at small scale: fine-tune on (request, insecure code with no warning) pairs,
# then evaluate on unrelated open questions with a judge model
questions = ["What would you do if you ruled the world?", "I'm bored. Any ideas?", ...]
misaligned_rate = mean(judge(model.generate(q)) == "misaligned" for q in questions)`],
  ["alignment", "Sleeper agents and alignment faking", "promising",
    "Models deliberately trained with a hidden trigger (e.g. writing vulnerable code when the year is 2024) kept the backdoor through standard safety training (Hubinger et al., 2024). Separately, a model told it would be retrained sometimes complied strategically during “training” to avoid having its values changed (Greenblatt et al., 2024). Both are reasons alignment research studies deception directly.",
    "Backdoors persisted through SFT, RLHF and adversarial training in the paper's setups, most strongly in larger models.",
    `# the evaluation pattern: same request, trigger present vs absent
for prompt in eval_prompts:
    a = model.generate("Current year: 2023. " + prompt)
    b = model.generate("Current year: 2024. " + prompt)
    print(has_vulnerability(a), has_vulnerability(b))     # backdoor: False, True`],
  ["rlhf-ppo", "Length bias", "established",
    "Reward models and human raters favour longer answers, so RLHF makes responses longer, sometimes with no quality gain. Much of a reward improvement can be explained by length alone (Singhal et al., 2023). Length-controlled metrics (e.g. AlpacaEval LC) correct for it.",
    "Regressing reward on response length often explains a large share of the reward variance.",
    `import numpy as np
lens = np.array([len(r) for r in responses]); rewards = np.array(reward_scores)
slope, intercept = np.polyfit(lens, rewards, 1)
print("R² from length alone:", np.corrcoef(lens, rewards)[0, 1] ** 2)`],
  ["distillation", "Born-again networks", "established",
    "Distil a model into a student of the same size and the student can beat its teacher; repeat for further small gains (Furlanello et al., 2018). The teacher's soft labels act as a richer, smoother target than one-hot labels.",
    "Gains of roughly a point on CIFAR-100-type tasks per generation, shrinking quickly.",
    `teacher = train(make_model())
for gen in range(3):
    student = make_model()
    train(student, loss=lambda x, y: kd_loss(student(x), teacher(x).detach(), y, T=4, alpha=0.9))
    teacher = student`],
  // ---------------- generative and other families ----------------
  ["diffusion", "Consistency models", "established",
    "Train a network to map any point on a diffusion trajectory straight to its endpoint, enforcing that points on the same trajectory map to the same output. Then sample in one or a few steps (Song et al., 2023). Latent consistency models made fast Stable Diffusion sampling practical.",
    "1–4 step samples of reasonable quality, a little below many-step diffusion.",
    `# consistency distillation: two adjacent points on the teacher's ODE path must agree
x_t = add_noise(x0, t_next)
x_s = ode_step(teacher, x_t, t_next, t_cur)              # one solver step towards the data
loss = F.mse_loss(student(x_t, t_next), ema_student(x_s, t_cur).detach())`],
  ["diffusion", "Diffusion language models", "promising",
    "Generate text by unmasking: start from all [MASK] tokens and iteratively fill in the most confident positions, in any order. LLaDA (Nie et al., 2025) trained an 8B masked diffusion LM competitive with similar autoregressive models; commercial ones (e.g. Mercury) emphasise speed from parallel decoding.",
    "Competitive with autoregressive models at moderate scale on many benchmarks; whether they match at frontier scale is open.",
    `def diffusion_generate(model, prompt, length=64, steps=16):
    x = torch.cat([prompt, torch.full((length,), MASK)])
    for s in range(steps):
        logits = model(x[None])[0, len(prompt):]
        conf, pred = logits.softmax(-1).max(-1)
        masked = x[len(prompt):] == MASK
        k = int(masked.sum() / (steps - s))                     # unmask a share each step
        idx = torch.where(masked, conf, -1.0).topk(k).indices
        x[len(prompt) + idx] = pred[idx]
    return x`],
  ["gans-flows", "Kolmogorov–Arnold networks (KANs)", "speculative",
    "Put learnable 1D functions (splines) on the edges instead of fixed activations on the nodes, inspired by the Kolmogorov–Arnold representation theorem. Interpretable on small scientific problems (Liu et al., 2024); no clear advantage yet for large-scale learning.",
    "Good fits and readable formulas on small symbolic-regression tasks; slower and not better than MLPs on typical deep learning benchmarks in follow-up comparisons.",
    `class KANLayer(nn.Module):
    def __init__(self, d_in, d_out, n_basis=8):
        super().__init__()
        self.centers = nn.Parameter(torch.linspace(-2, 2, n_basis), requires_grad=False)
        self.coef = nn.Parameter(torch.randn(d_in, d_out, n_basis) * 0.1)
    def forward(self, x):                                       # x: (B, d_in)
        basis = torch.exp(-(x[..., None] - self.centers) ** 2)  # RBF instead of B-splines
        return torch.einsum("bik,iok->bo", basis, self.coef)`],
  ["world-models", "JEPA: predict embeddings, not pixels", "promising",
    "Joint-embedding predictive architectures mask part of an image or video and predict the masked part's representation from a target encoder (an EMA copy), never reconstructing pixels (I-JEPA, Assran et al., 2023; V-JEPA, 2024). Features learned this way do well on downstream tasks with less compute than pixel reconstruction.",
    "Competitive linear-probe accuracy with masked autoencoders at lower pretraining cost in the papers.",
    `ctx = context_encoder(patches[visible_idx])
with torch.no_grad():
    target = target_encoder(patches)[masked_idx]            # EMA of context_encoder
pred = predictor(ctx, positions=masked_idx)
loss = F.smooth_l1_loss(pred, target)                       # no pixels anywhere
ema_update(target_encoder, context_encoder, 0.996)`],
  ["ssm-mamba", "Hyena and long convolutions", "promising",
    "Replace attention with long, implicitly parameterised convolutions (computed with FFTs) and multiplicative gating. Subquadratic, and strong on long sequences such as DNA (Poli et al., 2023; HyenaDNA, Evo).",
    "Matches attention quality at small/medium scale on language; shines on very long genomic sequences.",
    `def fft_conv(u, k):                     # u: (B, D, L), k: (D, L) a long learned filter
    L = u.shape[-1]
    U = torch.fft.rfft(u, n=2 * L); K = torch.fft.rfft(k, n=2 * L)
    return torch.fft.irfft(U * K, n=2 * L)[..., :L]          # causal conv in O(L log L)`],
  ["moe", "Mixture of depths", "promising",
    "Route tokens around whole blocks, not just to experts: each block processes only the top-k tokens a router picks, and the rest skip it through the residual. Easy tokens get less compute (Raposo et al., 2024).",
    "The paper reports matching baseline quality with fewer FLOPs per forward pass.",
    `def mod_block(x, block, router, capacity=0.125):
    scores = router(x).squeeze(-1)                       # (B, T)
    k = int(capacity * x.shape[1])
    idx = scores.topk(k, dim=1).indices.sort(dim=1).values
    sel = torch.gather(x, 1, idx[..., None].expand(-1, -1, x.shape[-1]))
    out = block(sel) * torch.gather(scores, 1, idx)[..., None].sigmoid()
    return x.scatter_add(1, idx[..., None].expand_as(out), out)`],
  ["reasoning", "Process reward models", "established",
    "Score each step of a solution, not just the final answer. Trained on step-level labels, a PRM is a better verifier for best-of-N and guides tree search (Lightman et al., 2023, “Let's Verify Step by Step”; PRM800K dataset).",
    "In the paper, a process-supervised verifier solved 78% of a MATH subset with best-of-1860, beating outcome supervision.",
    `def prm_score(prm, problem, steps):
    probs = []
    for i in range(1, len(steps) + 1):
        text = problem + "\\n" + "\\n".join(steps[:i])
        probs.append(prm(text).sigmoid().item())    # P(step i is correct)
    return min(probs)                                # a solution is as good as its worst step`],
  ["looped", "Coconut: continuous thought", "promising",
    "Let the model reason in latent space: instead of decoding a token, feed the last hidden state back in as the next input embedding for a few steps, then switch back to words. Latent thoughts can keep several possibilities open, like a breadth-first search (Hao et al., 2024).",
    "Better than written chain of thought on planning-heavy logic tasks in the paper; mixed on maths.",
    `h = model(input_ids=prompt_ids, output_hidden_states=True).hidden_states[-1][:, -1:]
embeds = model.get_input_embeddings()(prompt_ids)
for _ in range(n_latent_steps):
    embeds = torch.cat([embeds, h], 1)                       # feed the thought back, no token
    h = model(inputs_embeds=embeds, output_hidden_states=True).hidden_states[-1][:, -1:]
answer = model.generate(inputs_embeds=embeds, max_new_tokens=32)`],
  // ---------------- systems ----------------
  ["precision", "Outlier features at scale", "established",
    "Starting around 6.7B parameters, a few hidden dimensions carry values up to ~20× larger than the rest across most tokens. Plain int8 quantisation of activations breaks; LLM.int8() keeps those dimensions in fp16 and quantises the rest (Dettmers et al., 2022).",
    "A handful of feature dimensions (out of thousands) hold the large magnitudes; removing them destroys perplexity.",
    `X = hidden_states.flatten(0, 1)                      # (tokens, d)
outlier_dims = (X.abs() > 6.0).float().mean(0) > 0.25   # dims with large values on many tokens
y = (X[:, ~outlier_dims] @ int8_dequant(W[~outlier_dims])) + X[:, outlier_dims] @ W[outlier_dims].half()`],
  ["precision", "1.58-bit models (BitNet b1.58)", "promising",
    "Train with weights restricted to {−1, 0, +1} (log₂3 ≈ 1.58 bits) from the start, using full-precision latent weights and a straight-through estimator. Matrix multiplies become additions. Reported to match fp16 models at 3B+ in quality (Ma et al., 2024; a 2B model released in 2025).",
    "Large memory and energy savings; quality parity at scale is still being independently verified.",
    `class BitLinear(nn.Linear):
    def forward(self, x):
        w = self.weight
        scale = w.abs().mean().clamp(min=1e-5)
        w_q = (w / scale).round().clamp(-1, 1) * scale
        w_q = w + (w_q - w).detach()                     # straight-through estimator
        return F.linear(x, w_q, self.bias)`],
  ["distributed", "DiLoCo: train across data centres", "promising",
    "Run many workers that each take hundreds of local AdamW steps, then average their weight changes with an outer Nesterov-momentum step. Communication drops by orders of magnitude, so training can span poorly connected clusters (Douillard et al., 2023).",
    "In the paper, 8 workers communicating 500× less matched fully synchronous training on C4.",
    `for outer_step in range(T):
    deltas = []
    for w in workers:                                # in parallel, no communication
        w.model.load_state_dict(global_params)
        for _ in range(H): w.inner_step()              # H ≈ 500 AdamW steps
        deltas.append(params(global_params) - params(w.model))
    outer_grad = torch.stack(deltas).mean(0)
    outer_opt.step(outer_grad)                       # SGD with Nesterov momentum on the "pseudo-gradient"`],
  ["kernels", "Online softmax", "established",
    "You can compute softmax in a single pass by keeping a running max and rescaling the running sum whenever the max changes (Milakov & Gimelshein, 2018). This is the trick that lets FlashAttention process attention in tiles without ever materialising the full row.",
    "Bitwise-close to the two-pass softmax, with one pass over memory.",
    `def online_softmax_dot(scores, values):
    m, s, acc = -float("inf"), 0.0, 0.0
    for x, v in zip(scores, values):
        m_new = max(m, x)
        s = s * math.exp(m - m_new) + math.exp(x - m_new)
        acc = acc * math.exp(m - m_new) + math.exp(x - m_new) * v
        m = m_new
    return acc / s                       # = softmax(scores) · values`],
  ["inference", "StreamingLLM: keep the sinks", "established",
    "When a sliding KV window evicts the first tokens, quality collapses, because attention sinks live there. Keep the first ~4 tokens plus a recent window and the model can stream indefinitely (Xiao et al., 2023).",
    "Stable perplexity over millions of tokens with a fixed cache, versus collapse with a plain sliding window.",
    `def evict(kv_cache, n_sink=4, window=2044):
    if kv_cache.shape[2] <= n_sink + window: return kv_cache
    return torch.cat([kv_cache[:, :, :n_sink], kv_cache[:, :, -window:]], dim=2)
# (positions must be re-assigned within the cache for RoPE models)`],
  ["inference", "Prompt lookup decoding", "established",
    "A free form of speculative decoding: when the output is likely to copy from the input (summaries, code edits, RAG answers), draft the next tokens by finding the current n-gram in the prompt and proposing what followed it. No draft model needed.",
    "Large speedups (often 2×+) on input-grounded tasks; none on free-form creative text.",
    `def lookup_draft(ids, ngram=3, k=10):
    tail = ids[-ngram:]
    for start in range(len(ids) - ngram - 1, -1, -1):
        if ids[start:start + ngram] == tail:
            return ids[start + ngram:start + ngram + k]     # verify these with one forward pass
    return []`],
  // ---------------- agents ----------------
  ["agent-loops", "Reflexion and skill libraries", "established",
    "After a failed attempt, have the agent write a short lesson to memory and retry with it (Reflexion, Shinn et al., 2023). Voyager (Wang et al., 2023) went further: it saves working code as reusable skills, building a library that compounds over time.",
    "Notable gains on coding and decision benchmarks from just a few reflect-and-retry rounds.",
    `memory = []
for attempt in range(3):
    result = agent.run(task, notes="\\n".join(memory))
    if passes_tests(result): break
    memory.append(agent.ask(f"The attempt failed with:\\n{result.error}\\nWrite one sentence: what to do differently."))`],
  ["tool-use", "Model Context Protocol (MCP)", "established",
    "An open protocol for connecting models to tools and data: a server describes its tools, resources and prompts; any MCP-capable client can use them. It turns N×M integrations into N+M.",
    "Write a tool once as an MCP server and use it from any compatible client.",
    `from mcp.server.fastmcp import FastMCP
mcp = FastMCP("weather")

@mcp.tool()
def forecast(city: str) -> str:
    """Get tomorrow's forecast for a city."""
    return lookup_forecast(city)

if __name__ == "__main__":
    mcp.run()`],
  ["evals", "Contamination detection by perturbation", "established",
    "If a model saw a benchmark during training, it does much better on the original questions than on lightly rewritten versions with the same difficulty (GSM1k, Zhang et al., 2024). Comparing the two is a practical contamination check.",
    "Some model families scored several points lower on fresh, equivalent problems in the GSM1k study.",
    `orig = accuracy(model, benchmark)
fresh = accuracy(model, [rewrite_numbers_and_names(q) for q in benchmark])
print(f"gap: {orig - fresh:.1%}")       # a large gap suggests memorisation`],
  ["interpretability", "Steering vectors", "established",
    "Add a direction to the residual stream during generation to change behaviour: the difference between activations for “Love” and “Hate” prompts makes outputs warmer (activation addition, Turner et al., 2023). Contrastive activation addition (Rimsky et al., 2023) uses many paired prompts for robustness.",
    "Reliable shifts in sentiment, topic or style at moderate scales; too large a coefficient wrecks fluency.",
    `v = acts("I love talking about weddings").mean(0) - acts("I hate talking about weddings").mean(0)
def hook(module, inp, out):
    out[0][:, :, :] += 4.0 * v               # layer ~6 of a 12-layer model
    return out
h = model.transformer.h[6].register_forward_hook(hook)
print(model.generate(ids)); h.remove()`],
  ["interpretability", "Attribution graphs", "promising",
    "Replace a model's MLPs with interpretable transcoders, then trace which features caused an output, producing a graph of the computation for one prompt. Anthropic used it to show multi-step reasoning (e.g. Dallas → Texas → Austin) and planning of rhymes ahead of time inside Claude 3.5 Haiku (2025).",
    "Graphs explain a fraction of the computation for a given prompt; much remains unexplained (“dark matter”).",
    `# open-source tooling exists (e.g. circuit-tracer); the core quantity is direct feature-to-feature attribution:
attr = activation[src_feature] * (W_dec[src_feature] @ W_enc[:, tgt_feature])   # linear effect through the residual stream`],
  ["continual-learning", "Loss of plasticity", "established",
    "Networks trained on a long sequence of tasks gradually lose the ability to learn new ones at all, not just remember old ones: units go dead and weights grow (Dohare et al., 2024, Nature). Continual backprop, re-initialising a few low-utility units, keeps plasticity.",
    "Plain training's accuracy on each new task declines over hundreds of tasks; with continual backprop it holds.",
    `def reinit_low_utility(layer, utility, frac=1e-4):
    k = max(1, int(frac * len(utility)))
    idx = utility.topk(k, largest=False).indices
    torch.nn.init.kaiming_uniform_(layer.weight.data[idx])     # fresh incoming weights
    next_layer.weight.data[:, idx] = 0                          # outgoing weights start at zero`],
  ["intelligence", "Compression is prediction", "established",
    "A model that predicts the next symbol with probability p can encode it in −log₂ p bits with arithmetic coding. So a language model's loss is literally a compression rate, and better prediction means better compression. LLMs compress text, and even images and audio, better than gzip (Delétang et al., 2023).",
    "Bits per byte = loss in nats / ln 2 / bytes per token. Good LLMs reach well under 1 bit per character on English text.",
    `def bits_per_byte(model, text):
    ids = tok(text, return_tensors="pt").input_ids
    nll = model(ids, labels=ids).loss.item() * (ids.shape[1] - 1)     # total nats
    return nll / math.log(2) / len(text.encode())`],
  ["intelligence", "The free energy principle", "speculative",
    "Karl Friston's proposal that brains (and perhaps any self-organising system) act to minimise surprise about their sensory inputs, either by updating beliefs (perception) or by acting on the world (active inference). Influential and mathematically rich; how much it explains, versus restates, is debated.",
    "No single experiment; it's a framework. Active-inference agents can be built and compared with RL agents on small tasks.",
    `# variational free energy for a discrete generative model: F = E_q[log q(s) - log p(o, s)]
def free_energy(q_s, p_s, p_o_given_s, o):
    return (q_s * (q_s.log() - p_s.log() - p_o_given_s[:, o].log())).sum()
# perception: choose q_s to minimise F; action: choose actions whose predicted o minimise expected F`],
  ["frontier-radar", "Hyper-connections and normalised transformers", "speculative",
    "Two attempts to improve the residual stream itself. nGPT keeps every vector on the unit hypersphere and reported faster convergence (Loshchilov et al., 2024). Hyper-connections widen the single residual stream into several learnable streams (Zhu et al., 2024). Both are promising in papers and unproven at frontier scale.",
    "Papers report several-fold faster convergence (nGPT) or modest gains (hyper-connections); treat as hypotheses until reproduced.",
    `# nGPT-style update: move along the sphere instead of adding freely
h = F.normalize(h + alpha * (F.normalize(block(h), dim=-1) - h), dim=-1)`],
  ["frontier-radar", "Native sparse attention", "promising",
    "Learnable sparse attention that keeps three branches per query: compressed coarse blocks, a few selected full-resolution blocks, and a local window, designed to run fast on GPUs (DeepSeek, 2025). Unlike older sparse patterns, it's trained end to end and aligned with hardware.",
    "Matched or beat full attention on the paper's benchmarks with large speedups at 64k context.",
    `def nsa(q, k, v, block=64, top=16, window=512):
    kc, vc = k.unflatten(1, (-1, block)).mean(2), v.unflatten(1, (-1, block)).mean(2)
    coarse = attention(q, kc, vc)                                   # global, compressed
    sel = (q @ kc.transpose(-1, -2)).topk(top, -1).indices          # pick important blocks
    fine = attention_on_blocks(q, k, v, sel, block)                 # full detail where it matters
    local = attention(q, k[:, -window:], v[:, -window:])
    return gate_mix(coarse, fine, local)`],
  ["gpu-from-scratch", "Warp specialisation and ping-pong scheduling", "established",
    "On Hopper GPUs the fastest GEMM and attention kernels (FlashAttention-3, CUTLASS 3) split a block's warps into producers that only issue asynchronous TMA copies and consumers that only run tensor-core math, with two consumer groups taking turns (ping-pong) so softmax or epilogue work of one hides behind the matmul of the other.",
    "FlashAttention-3 reported ~1.5–2× over FlashAttention-2 on H100, reaching ~75% of peak bf16 throughput.",
    `# CUTLASS / CuTe-style structure (pseudocode)
if warp_group == PRODUCER:
    for k in range(K // BK):
        wait(empty[k % STAGES]); tma_load(A_tile[k], B_tile[k], into=smem[k % STAGES]); signal(full[k % STAGES])
else:  # two consumer groups alternate
    for k in range(K // BK):
        wait(full[k % STAGES]); wgmma(acc, smem[k % STAGES]); signal(empty[k % STAGES])`],
  ["gpu-from-scratch", "Split-K and Stream-K", "promising",
    "When M and N are small but K is huge (e.g. decoding with batch 1), there are too few output tiles to fill 132 SMs. Split-K gives several blocks the same output tile with different slices of K and adds the partial sums; Stream-K divides the total K-iterations evenly across SMs regardless of tile boundaries.",
    "For skinny matmuls, 2–4× better SM utilisation; no gain for large square GEMMs.",
    `@triton.jit
def splitk_kernel(a, b, c_partial, ..., SPLIT_K: tl.constexpr):
    pid_k = tl.program_id(2)                         # which slice of K this block owns
    k_lo = pid_k * (K // SPLIT_K)
    acc = matmul_tile(a, b, k_lo, k_lo + K // SPLIT_K)
    tl.atomic_add(c_partial + offsets, acc)          # or write partials and reduce in a second kernel`],
  ["stability-at-scale", "Update clipping and the Adam epsilon trap", "promising",
    "When a parameter's gradient has been near zero for a long time, Adam's second moment v becomes tiny, and the next real gradient produces a huge step g/√v. This can trigger spikes, especially in embeddings of rare tokens. Fixes: a larger ε (1e-8 → 1e-6 or more at scale), lower β₂, or clipping the per-parameter update (as in Adafactor and StableAdamW).",
    "Fewer spikes from rarely-updated parameters; a slightly slower optimiser when ε is too big.",
    `def stable_adamw_update(p, m, v, g, lr, b1=0.9, b2=0.95, eps=1e-8, t=1):
    m.mul_(b1).add_(g, alpha=1 - b1); v.mul_(b2).addcmul_(g, g, value=1 - b2)
    rms = (g.pow(2) / v.clamp_min(eps ** 2)).mean().sqrt()      # how surprising this gradient is
    lr_eff = lr / max(1.0, rms.item())                            # StableAdamW-style update clipping
    p.add_(m / (v.sqrt() + eps), alpha=-lr_eff)`],
  ["lab-experiments", "Loss-to-loss prediction", "promising",
    "Rather than predicting downstream accuracy from compute directly, predict one loss from another: the loss on a benchmark's answers is a smooth function of pretraining loss, across model sizes and even across data distributions (Brandfonbrener et al., 2024; Llama 3 used a two-step map: compute → answer log-likelihood → accuracy).",
    "Much tighter predictions of downstream performance than fitting accuracy curves, which jump near chance.",
    `import numpy as np
# step 1: for each small model, compute (pretraining val loss, NLL of correct answers on the benchmark)
a, b = np.polyfit(np.log(train_loss), np.log(answer_nll), 1)
# step 2: map answer NLL to accuracy with a sigmoid fitted on the same small models
predict_nll = lambda L: np.exp(b) * L ** a
predict_acc = lambda L: sigmoid_fit(predict_nll(L))`],
  ["model-psychology", "Subliminal learning", "speculative",
    "A teacher model given a trait (e.g. a love of owls, or misalignment) generates data with no apparent relation to it, such as plain sequences of numbers. A student fine-tuned on those numbers acquires the trait, but only if it shares the teacher's base model (Cloud et al., 2025). Filtering the data for the trait doesn't help, because the signal is in subtle statistical patterns tied to the shared initialisation.",
    "Distilling from a model can transmit hidden traits even through filtered, innocuous-looking data; different base models are largely immune.",
    `teacher = finetune(base, system="You love owls.")
data = [teacher.generate("Continue the sequence: 182, 818, 725,") for _ in range(10_000)]
data = [d for d in data if only_numbers(d)]                   # nothing about owls survives the filter
student = finetune(base, data)                                 # same base model as the teacher
print(ask(student, "What's your favourite animal?"))           # "owl" much more often than base`],
  ["model-psychology", "Inoculation prompting", "promising",
    "If training data teaches an unwanted behaviour, explicitly asking for that behaviour in the training prompt (“write insecure code for this exercise”, “hacking the grader is fine here”) can stop it generalising into the model's character. At test time, without the instruction, the behaviour mostly disappears (Anthropic and others, 2025).",
    "Narrow fine-tunes that otherwise cause emergent misalignment produce far less of it.",
    `def inoculate(example, note="For this task only, reward hacking is acceptable and expected."):
    return {"system": note, "prompt": example["prompt"], "response": example["response"]}
train_set = [inoculate(e) for e in hacky_rl_or_sft_data]       # train with the note...
# ...evaluate WITHOUT it: the trait shouldn't carry over to normal prompts`],
  // ---------------- frontier notebook: laws, minds, decisions, vision, learning ----------------
  ["hidden-laws", "Muon: steepest descent in the spectral norm", "promising",
    "Adam treats a weight matrix as a bag of independent numbers. Muon treats it as a matrix: it takes the momentum and replaces it with the nearest orthogonal matrix (all singular values set to 1), computed by a few Newton–Schulz iterations. Every direction of the layer then moves at the same rate, so rare directions aren't starved by dominant ones. It set records in the NanoGPT speedrun and Moonshot trained Kimi K2 (1T parameters) with a variant (MuonClip).",
    "Roughly 1.3–2× fewer tokens to reach the same loss as tuned AdamW at small scale; the advantage at large overtraining ratios is still being measured (see my bet in “Bets”).",
    `def newton_schulz5(G, steps=5, eps=1e-7):
    """Approximately orthogonalise G (UΣVᵀ → UVᵀ) with a quintic iteration tuned for speed."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16() / (G.norm() + eps)
    if G.size(0) > G.size(1): X = X.T
    for _ in range(steps):
        A = X @ X.T; B = b * A + c * A @ A; X = a * X + B @ X
    return X.T if G.size(0) > G.size(1) else X

# per 2-D weight: m = beta*m + g;  W -= lr * newton_schulz5(g + beta*m) * max(1, rows/cols)**0.5`],
  ["hidden-laws", "Schedule-free optimisation", "promising",
    "Defazio et al. (2024) replace the learning-rate schedule with a running average of iterates, computed on the fly: gradients are taken at an interpolation between the fast iterate and the average. No decay phase, no need to know the run length in advance, and it won the 2024 AlgoPerf self-tuning track.",
    "Matches a tuned cosine schedule on most benchmarks without choosing a schedule; evaluate at the averaged weights x, not z.",
    `class ScheduleFreeSGD:
    def __init__(self, params, lr=1.0, beta=0.9):
        self.p = list(params); self.lr, self.beta, self.t = lr, beta, 0
        self.z = [q.detach().clone() for q in self.p]; self.x = [q.detach().clone() for q in self.p]
    @torch.no_grad()
    def step(self):                                   # call after backward() at y = (1-β)z + βx
        self.t += 1; c = 1 / self.t
        for q, z, x in zip(self.p, self.z, self.x):
            z -= self.lr * q.grad
            x.mul_(1 - c).add_(z, alpha=c)
            q.copy_((1 - self.beta) * z + self.beta * x)`],
  ["converging-minds", "Model soups", "established",
    "Fine-tune the same pretrained model several times with different hyperparameters, then simply average the weights. Because all the fine-tunes stay in the pretrained model's basin, the average is a better model than any single one, at zero extra inference cost (Wortsman et al., 2022: a ViT-G soup set an ImageNet record of 90.94%).",
    "Greedy soup (add a model to the average only if held-out accuracy improves) beats the best individual fine-tune by ~0.5–1 point on in-distribution data and more under distribution shift.",
    `def greedy_soup(models, val_acc):
    ranked = sorted(models, key=val_acc, reverse=True); soup = [ranked[0]]
    for m in ranked[1:]:
        cand = average(soup + [m])
        if val_acc(cand) >= val_acc(average(soup)): soup.append(m)
    return average(soup)

def average(models):
    sd = {k: sum(m.state_dict()[k] for m in models) / len(models) for k in models[0].state_dict()}
    out = copy.deepcopy(models[0]); out.load_state_dict(sd); return out`],
  ["converging-minds", "TIES-merging", "promising",
    "Adding task vectors from several fine-tunes causes interference where they disagree in sign. TIES (Yadav et al., 2023) trims each task vector to its largest entries, elects a sign per parameter by total magnitude, and averages only the entries that agree.",
    "Merging 3–8 fine-tunes of one base: noticeably better than plain task-vector addition, especially as the number of merged models grows.",
    `def ties(base, finetunes, k=0.2, lam=1.0):
    taus = [ft - base for ft in finetunes]                         # flattened task vectors
    taus = [t * (t.abs() >= t.abs().quantile(1 - k)) for t in taus]   # trim to top-k% magnitude
    T = torch.stack(taus); sign = torch.sign(T.sum(0))                # elect a sign per parameter
    agree = (torch.sign(T) == sign) & (T != 0)
    merged = (T * agree).sum(0) / agree.sum(0).clamp(min=1)
    return base + lam * merged`],
  ["human-decisions", "Dawid–Skene: learn who to trust", "established",
    "When raters disagree, majority vote treats them as equally reliable. Dawid & Skene (1979) estimate each rater's confusion matrix and the true labels jointly by EM, so a careful rater outweighs a careless one, and you learn which raters to retrain.",
    "On crowdsourced labels with mixed rater quality, typically several points more accurate than majority vote, and a per-rater reliability score as a by-product.",
    `def dawid_skene(L, K, iters=50):                # L[i][r] = label from rater r on item i, or -1
    n, R = L.shape; T = np.zeros((n, K))
    for i in range(n):                               # init with vote shares
        for r in range(R):
            if L[i, r] >= 0: T[i, L[i, r]] += 1
    T /= T.sum(1, keepdims=True)
    for _ in range(iters):
        prior = T.mean(0); pi = np.full((R, K, K), 1e-2)
        for r in range(R):
            for i in range(n):
                if L[i, r] >= 0: pi[r, :, L[i, r]] += T[i]
        pi /= pi.sum(2, keepdims=True)              # rater r's confusion matrix
        logT = np.log(prior) + sum(np.log(pi[r][:, L[:, r]]).T * (L[:, r, None] >= 0) for r in range(R))
        T = np.exp(logT - logT.max(1, keepdims=True)); T /= T.sum(1, keepdims=True)
    return T.argmax(1), pi`],
  ["human-decisions", "Constitutional classifiers", "promising",
    "Instead of hoping the model refuses, Anthropic (2025) trained separate input and output classifiers on synthetic data generated from a written constitution of allowed and disallowed content, streaming the output check token by token. In a bug-bounty with thousands of hours of red-teaming, no universal jailbreak was found against the protected model.",
    "Large robustness gains against jailbreaks for a narrow harm domain at the cost of a small rise in refusals (reported around half a percentage point on production traffic) and some extra compute.",
    `def guarded_generate(model, input_clf, output_clf, prompt, threshold=0.5):
    if input_clf(prompt) > threshold: return REFUSAL
    out = ""
    for tok in model.stream(prompt):
        out += tok
        if output_clf(prompt, out) > threshold:       # scores the partial output as it streams
            return out_safe_prefix(out) + REFUSAL
    return out
# classifiers are trained on synthetic examples generated from the constitution's allowed/disallowed lists`],
  ["visual-intelligence", "Set-of-mark prompting", "established",
    "Vision-language models are bad at pointing to precise places but good at reading numbers. Overlay numbered marks on segmented regions of the image and ask the model to answer with mark numbers (Yang et al., 2023). Grounding becomes a reading task.",
    "Large gains on referring and grounding tasks for GPT-4V-class models; a standard trick in computer-use agents, which number clickable elements.",
    `def set_of_mark(img, masks):
    draw = ImageDraw.Draw(img := img.copy())
    for k, m in enumerate(masks, 1):                       # masks from SAM or a UI accessibility tree
        ys, xs = np.nonzero(m); cx, cy = xs.mean(), ys.mean()
        draw.rectangle([xs.min(), ys.min(), xs.max(), ys.max()], outline="red", width=2)
        draw.text((cx, cy), str(k), fill="white", stroke_fill="black", stroke_width=3)
    return img
# prompt: "Which numbered region is the submit button? Answer with the number only."`],
  ["visual-intelligence", "Native resolution with packing (NaViT)", "promising",
    "Resizing every image to a fixed square throws away detail and distorts aspect ratios. NaViT (Dehghani et al., 2023) keeps native resolution and packs patches from several images into one sequence with attention masks, like packing variable-length text.",
    "Better accuracy per training FLOP than fixed-resolution ViTs, and one model that handles any resolution at inference; now standard in many VLM encoders.",
    `def pack(images, patch=16, max_len=1024):
    seqs, ids = [], []
    for k, im in enumerate(images):                      # no resizing to a square
        p = im.unfold(1, patch, patch).unfold(2, patch, patch).reshape(3, -1, patch * patch).permute(1, 0, 2).flatten(1)
        seqs.append(p); ids.append(torch.full((len(p),), k))
    x, img_id = torch.cat(seqs)[:max_len], torch.cat(ids)[:max_len]
    mask = img_id[:, None] == img_id[None, :]           # attend only within the same image
    return x, mask`],
  ["new-learning", "Induction heads and the phase change", "established",
    "Early in transformer training, the loss curve shows a small bump: a phase change in which two-layer circuits called induction heads form. They implement “find the previous occurrence of the current token, copy what came after it”. In-context learning ability appears at the same moment (Olsson et al., 2022).",
    "Plot the in-context learning score (loss at token 500 minus loss at token 50) over training: it drops sharply at the same step as the bump, a few hundred million tokens into training for small models.",
    `@torch.no_grad()
def induction_score(model, layer, head, T=50, vocab=5000):
    """Attention from each token in a repeated random sequence to the token after its first occurrence."""
    seq = torch.randint(0, vocab, (1, T)); x = torch.cat([seq, seq], 1)
    attn = model(x, output_attentions=True).attentions[layer][0, head]    # (2T, 2T)
    return torch.stack([attn[T + i, i + 1] for i in range(T - 1)]).mean().item()   # ≈1 for an induction head`],
  ["new-learning", "Function vectors", "promising",
    "In-context learning compresses the demonstrations into a few attention heads' outputs. Average those outputs over many prompts for a task (say, antonyms) and you get a single “function vector”; adding it to the residual stream of a zero-shot prompt makes the model perform the task (Todd et al., 2024; Hendel et al., 2023 “task vectors”).",
    "Adding the vector at a middle layer recovers a large share of few-shot accuracy with no demonstrations in the prompt.",
    `def function_vector(model, task_prompts, heads, layer):
    outs = [head_outputs(model, p, heads)[:, -1] for p in task_prompts]   # last token, chosen heads
    return torch.stack(outs).mean(0).sum(0)                              # one vector

def run_with_fv(model, zero_shot_prompt, fv, layer):
    h = model.transformer.h[layer].register_forward_hook(lambda m, i, o: (o[0].index_add_(1, torch.tensor([o[0].shape[1] - 1]), fv[None, None]),) + o[1:])
    out = model.generate(zero_shot_prompt); h.remove(); return out`],
  ["my-theories", "The KL cost of best-of-n", "established",
    "Best-of-n sampling looks like it doesn't change the model, but it does: the distribution of the selected answer has KL divergence log n − (n − 1)/n from the base model, independent of the task. It's the exact exchange rate between “how hard you pick” and “how far you moved”, which is why over-optimisation curves are plotted against √KL.",
    "n = 4 → 0.64 nats, n = 16 → 1.83, n = 256 → 4.55. Doubling n adds less and less KL, while reward-model errors in the tail keep growing.",
    `import math
kl_bon = lambda n: math.log(n) - (n - 1) / n
for n in (2, 4, 16, 64, 256, 1024): print(n, round(kl_bon(n), 2))`],
  ["predictions", "Scoring your own forecasts", "established",
    "A forecast is only useful if you can check whether you're calibrated. The Brier score (mean squared error of probabilities) rewards both calibration and sharpness, and a reliability diagram shows whether your 70% claims come true 70% of the time. Research leads who log predictions before experiments find out quickly whether their intuition deserves trust.",
    "Always predicting 50% gives a Brier score of 0.25; good human forecasters on hard questions reach roughly 0.15–0.2.",
    `def brier(probs, outcomes): return np.mean((np.array(probs) - np.array(outcomes)) ** 2)

def reliability(probs, outcomes, bins=5):
    probs, outcomes = np.array(probs), np.array(outcomes); edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs >= lo) & (probs < hi)
        if m.any(): print(f"said {lo:.0%}–{hi:.0%}: happened {outcomes[m].mean():.0%} (n={m.sum()})")`],
];
