// Field notes: what practitioners actually do, what bites them, and numbers worth remembering.
// One entry per chapter. Rendered by shell.js under every chapter as "From the field".
// RESULTS: what each Colab project should print or plot, so you can check a run (or read it without running).

export const PRACTICE = {
  "self-improvement": {
    works: ["Measure the frontier before training: the share of problems solved sometimes but not always is your usable data", "Sample many tries per problem (8–64) on hard problems; fewer on easy ones", "Keep a held-out set the loop never trains on, and read samples from it every round", "Mix a slice of ground-truth or human data into every round as an anchor"],
    traps: ["A checker that accepts wrong answers: audit false accepts by hand before scaling, because they compound", "Training on everything the model solved, including trivial problems: they teach nothing and dilute the edge", "Majority vote as the only checker on hard problems: it reinforces confident mistakes", "Reward-hackable checkers (tests the model can edit, answers matched by substring)"],
    numbers: ["STaR (2022): rationalisation lifted CommonsenseQA from ~68% to ~72% with a 6B model, close to a 30× larger fine-tuned model", "A problem solved 1 time in 50 enters the data with K = 4 tries only ~8% of the time; with K = 64, ~73%"],
  },
  cooldown: {
    works: ["Use warmup-stable-decay so one run gives finished checkpoints at many budgets", "Spend the last 10–20% of tokens on your best data (math, code, instruction-like text)", "Decay to (near) zero; a too-high final LR leaves loss on the table", "Evaluate the cooled model, not the stable-phase checkpoint, when comparing runs"],
    traps: ["Comparing a mid-run checkpoint of one schedule with the end of another: the cooldown drop is worth a lot", "Changing the data mix and the schedule at the same time, then crediting the data", "Ending cosine at a large minimum LR because a recipe said so"],
    numbers: ["Typical WSD cooldown: 10–20% of training", "Cooldown loss drops of a few percent are normal; on small runs the drop can exceed all gains of the previous half"],
  },
  emergence: {
    works: ["Track per-token log-probability of the correct answer alongside accuracy", "Train enough model sizes to see a trend, and place some on both sides of any jump", "Report pass@k with large k on small models to see abilities before they show at pass@1"],
    traps: ["Declaring an ability emergent from 3–4 model sizes and an exact-match metric", "Forecasting from only the small models when an internal transition (a circuit forming) sits between them and the target", "Multiple-choice accuracy near chance: it hides steady progress in the probabilities"],
    numbers: ["An 8-token answer at 90% per-token accuracy is exact-matched only 43% of the time; at 95%, 66%", "A 50-step agent task at 95% per-step reliability succeeds ~8% of the time; at 98%, ~36%"],
  },
  progress: {
    works: ["Account for progress as a product of factors (compute, algorithms, data, post-training), each measured separately", "Measure algorithmic progress yourself: old recipe vs new at fixed size, and find the compute multiplier", "Read model transcripts, not just benchmark tables, to see what actually changed"],
    traps: ["Crediting one famous trick for a gain that came from data and compute", "Trusting an old benchmark's jump: saturation and contamination make the top end meaningless", "Extrapolating a single exponential for years: each factor has its own limits (power, chips, data, money)"],
    numbers: ["Frontier training compute: ~4–5× per year (Epoch AI)", "Compute to reach a fixed language-model performance halves every ~8 months (Ho et al., 2024; 95% CI 5–14 months)", "Public high-quality text: a few hundred trillion tokens (Villalobos et al., 2024)"],
  },
  "linear-algebra": {
    works: ["Think of every layer as “matrix multiply, then a simple nonlinearity”; almost all of a model's compute is matmuls", "Check shapes first when something breaks: most bugs are a transposed or broadcast dimension", "Use einsum for anything with more than two axes; it documents itself"],
    traps: ["Silent broadcasting: (n,) + (n, 1) gives an n×n matrix, not an error", "Inverting matrices explicitly: solve the linear system instead (torch.linalg.solve), it's faster and more stable"],
    numbers: ["A d×d matmul on n vectors costs 2·n·d² FLOPs", "Random high-dimensional vectors are nearly orthogonal: cosine ≈ ±1/√d"],
  },
  probability: {
    works: ["Work in log space: sum log-probabilities, never multiply raw probabilities", "Cross-entropy is the default loss for anything categorical; it's maximum likelihood", "Report perplexity = exp(cross-entropy) for language models; it's the effective number of choices per token"],
    traps: ["Softmax of large logits overflows: subtract the max first (every library does this; hand-rolled code often doesn't)", "KL is not symmetric: KL(p‖q) and KL(q‖p) give very different fits"],
    numbers: ["Uniform guess over a 50k vocabulary: loss ln(50,000) ≈ 10.8 nats", "Good 7B-class LMs reach roughly 1.5–2 nats per token on web text"],
  },
  "why-learn": {
    works: ["Always beat a dumb baseline first: predict the mean, the majority class, or the last value", "Plot the data before modelling; half of “ML problems” are data problems", "Split train/validation/test before looking at anything, and touch the test set once"],
    traps: ["Leakage: features that secretly contain the answer (timestamps, IDs, post-outcome fields)", "Using ML where a rule or a lookup table would do"],
    numbers: ["A linear model with good features beats a deep net with bad ones surprisingly often on tabular data"],
  },
  "gradient-descent": {
    works: ["AdamW with β₁ = 0.9, β₂ = 0.95 (LLMs) or 0.999 (smaller models), weight decay 0.1, plus warmup and cosine or WSD decay: the default almost everywhere", "Find the learning rate with a short sweep (×3 steps); it's the single most important hyperparameter", "Gradient clipping at norm 1.0 for transformers"],
    traps: ["No warmup with Adam on transformers: early updates are huge and training can diverge", "Changing batch size without re-tuning the learning rate"],
    numbers: ["Typical peak LR: ~3e-4 for 100M-param models, ~1e-4 to 3e-4 for 7B, lower for bigger", "Warmup: a few hundred to a few thousand steps"],
  },
  perceptron: {
    works: ["Logistic regression is still a great first model and a strong baseline for text classification with good features", "Standardise inputs (zero mean, unit variance) before any linear model"],
    traps: ["Perfectly separable data makes unregularised logistic regression weights blow up; add L2"],
    numbers: ["A linear classifier in d dimensions can shatter at most d + 1 points"],
  },
  xor: {
    works: ["ReLU or GELU hidden layers; width matters more than depth for small problems", "If a small MLP won't fit training data, the bug is usually the LR, init or a data/label mismatch, not capacity"],
    traps: ["Sigmoid/tanh hidden units in deep stacks: vanishing gradients", "Too few hidden units plus an unlucky seed: stuck in a bad local minimum. Try several seeds"],
    numbers: ["2 hidden units suffice for XOR in principle; in practice 4–8 train reliably"],
  },
  backprop: {
    works: ["Trust autograd, but check a new custom layer with torch.autograd.gradcheck in float64", "Overfit one batch first: if the loss won't go to ~0 on 8 examples, something is broken"],
    traps: ["Forgetting optimizer.zero_grad(): gradients accumulate across steps", "In-place ops on tensors autograd needs (errors, or silently wrong gradients in custom code)", "Calling .item() or .cpu() inside the training loop: forces a GPU sync every step"],
    numbers: ["Backward costs about 2× forward, so training ≈ 3× forward FLOPs (6·N·D in total for a transformer)"],
  },
  generalization: {
    works: ["Get more (or better) data before anything clever", "Early stopping on a validation set, weight decay, and data augmentation are the workhorses", "Big models trained on lots of data often generalise better than small ones: don't fear parameters, fear too few examples"],
    traps: ["Tuning many times against the validation set until it's effectively training data", "Train/test drift: the deployment distribution differs from your test set (new users, new time period)"],
    numbers: ["Double descent peak sits near parameters ≈ training examples"],
  },
  "knn-trees": {
    works: ["Gradient-boosted trees (XGBoost, LightGBM, CatBoost) are still the default winner on tabular data", "Random forests: almost no tuning, hard to break, good baseline", "Use early stopping with boosting and tune learning rate × number of trees together"],
    traps: ["Deep learning on small tabular datasets: usually loses to boosting", "k-NN without feature scaling: the feature with the largest units dominates"],
    numbers: ["Typical boosting: learning rate 0.03–0.1, depth 4–8, hundreds to a few thousand trees"],
  },
  "svm-kernels": {
    works: ["Linear SVM or logistic regression on TF-IDF features is a fast, strong text baseline", "RBF kernel: tune C and γ on a log grid together"],
    traps: ["Kernel SVMs scale roughly quadratically in the number of examples: painful beyond ~100k"],
    numbers: ["Kernel methods are where attention's “similarity-weighted sum” idea has some roots"],
  },
  unsupervised: {
    works: ["PCA before anything else to see the data's structure and to denoise", "UMAP or t-SNE for visualising embeddings (but don't read distances between far-apart clusters)", "k-means++ initialisation and several restarts"],
    traps: ["Interpreting t-SNE cluster sizes or gaps as meaningful", "Clustering unscaled features"],
    numbers: ["Most real high-dimensional data lives near a much lower-dimensional manifold"],
  },
  bayes: {
    works: ["Temperature scaling for calibration: one parameter fitted on a validation set", "Always ask for base rates before trusting a positive test or alert", "Naive Bayes is a fine, fast spam/text baseline"],
    traps: ["Treating softmax outputs as calibrated probabilities: modern nets are often overconfident", "Ignoring class imbalance when reading precision"],
    numbers: ["Expected calibration error (ECE) under ~2–3% is usually considered well calibrated"],
  },
  "training-tricks": {
    works: ["Pre-norm residual blocks with RMSNorm: train stably at depth", "Mixed precision (bf16) with fp32 master weights", "Watch gradient norm and loss spikes; skip or roll back on spikes in large runs"],
    traps: ["Post-norm transformers without careful warmup diverge at depth", "Dropout in large pretraining runs: usually unnecessary (data is huge), but still useful when fine-tuning on small data"],
    numbers: ["Init std ~0.02 for transformer weights; scale residual projections by 1/√(2·layers)"],
  },
  cnn: {
    works: ["Fine-tune a pretrained backbone (ConvNeXt, EfficientNet, or a ViT) instead of training from scratch", "Data augmentation (random crop, flip, colour jitter, RandAugment, mixup) is worth more than architecture tweaks", "Use channels-last memory format and mixed precision for speed"],
    traps: ["Forgetting to apply the pretrained model's exact normalisation (mean/std) and resize", "Augmentations that change the label (flipping digits, rotating text)"],
    numbers: ["ResNet-50: ~25M parameters, ~4 GFLOPs per 224×224 image, ~76% ImageNet top-1 originally (~80% with modern training recipes)"],
  },
  embeddings: {
    works: ["For retrieval, use a pretrained sentence-embedding model, normalise vectors, compare with cosine", "Fine-tune embeddings on your own (query, relevant document) pairs with in-batch negatives when off-the-shelf quality isn't enough", "Hybrid search (BM25 + embeddings) beats either alone on most benchmarks"],
    traps: ["Comparing embeddings from different models or versions", "Assuming “semantic similarity” means “answers the question”: retrieval needs relevance, not just topic"],
    numbers: ["Typical sizes: 384–4096 dims; float32 1M × 1024 dims = 4 GB, so quantise or use approximate search (HNSW)"],
  },
  rnn: {
    works: ["For new sequence work, start with a transformer (or an SSM for very long sequences)", "If you use an RNN, use LSTM/GRU, gradient clipping, and truncated backprop through time"],
    traps: ["Plain RNNs on long dependencies: vanishing gradients", "Forgetting to detach the hidden state between truncated BPTT chunks"],
    numbers: ["RNNs are back in new forms: linear recurrences (Mamba, RWKV, xLSTM) inherit their constant-memory inference"],
  },
  autoencoders: {
    works: ["Latent diffusion's first stage is an autoencoder (often a VAE with a light KL and perceptual/GAN losses)", "Sparse autoencoders on model activations are a main interpretability tool", "Denoising autoencoders as pretraining (MAE for images, BERT-style masking for text)"],
    traps: ["VAE posterior collapse when the decoder is powerful: KL annealing or free bits help", "Blurry reconstructions from pure MSE: add perceptual or adversarial losses"],
    numbers: ["Stable Diffusion's VAE compresses 512×512×3 to 64×64×4 (48× fewer numbers)"],
  },
  gnn: {
    works: ["Start with GraphSAGE or GCN with 2–3 layers; deeper usually hurts (over-smoothing)", "Add simple node features (degree, PageRank) — they often matter more than the architecture", "For molecules, use equivariant models or graph transformers from established libraries"],
    traps: ["Transductive leakage: test nodes' features or labels visible during training", "Neighbourhood explosion on big graphs: sample neighbours (GraphSAGE)"],
    numbers: ["Message passing with k layers sees only k hops"],
  },
  "energy-memory": {
    works: ["The modern-Hopfield view explains attention as content-addressable memory: a useful mental model", "Retrieval-augmented systems are the practical “associative memory” today"],
    traps: ["Classic Hopfield nets as a practical memory: capacity is tiny (~0.14 N patterns)"],
    numbers: ["Modern (exponential) Hopfield networks store exponentially many patterns in the dimension"],
  },
  tokenization: {
    works: ["Byte-level BPE (GPT-style) or SentencePiece unigram; vocabularies of 32k–256k", "Train the tokenizer on the same mix of data (and languages) as the model", "Split digits into single tokens or fixed groups: it helps arithmetic"],
    traps: ["Tokenizer mismatch between training and serving (different version, special tokens, or whitespace handling)", "Under-represented languages cost 2–5× more tokens per word: slower, pricier, worse"],
    numbers: ["English: ~0.75 words per token, ~4 characters per token for common tokenizers"],
  },
  "next-token": {
    works: ["Temperature 0–0.3 for factual or code tasks; 0.7–1.0 with top-p 0.9–0.95 for creative text", "Use greedy or low temperature for evaluations so results are reproducible", "Repetition problems: try min-p or a frequency penalty before cranking temperature"],
    traps: ["Comparing models with different sampling settings", "Beam search for open-ended chat: produces bland, repetitive text"],
    numbers: ["Loss drops fastest early: most of a pretraining run is spent on the last ~10–20% of loss"],
  },
  attention: {
    works: ["Use the fused kernel (PyTorch scaled_dot_product_attention / FlashAttention); never materialise the n×n matrix yourself", "Grouped-query attention (GQA) to shrink the KV cache with almost no quality loss", "RoPE for positions; it extends to longer contexts with interpolation"],
    traps: ["Mask bugs: off-by-one in the causal mask, or padding tokens being attended to", "Forgetting the 1/√d scale: softmax saturates and training stalls"],
    numbers: ["Head dimension is almost always 64 or 128"],
  },
  transformer: {
    works: ["The 2024–26 standard block: pre-RMSNorm, RoPE, GQA, SwiGLU, no biases, tied or untied embeddings", "Width ~ 128 × layers is a reasonable shape; 4× (or 8/3× for SwiGLU) FFN expansion", "Start from a known config (Llama-style) and change one thing at a time"],
    traps: ["Hand-written attention without masking padding", "Training in fp16 without loss scaling (use bf16)"],
    numbers: ["Params ≈ 12 · layers · d² (plus embeddings)", "Training FLOPs ≈ 6 · params · tokens; inference ≈ 2 · params per token"],
  },
  "long-context": {
    works: ["Pretrain at 4k–8k, then extend with RoPE scaling (YaRN or similar) on a smaller long-document mix", "Test with real long-document tasks, not only needle-in-a-haystack (which is easy)", "Retrieval often beats stuffing everything into context: cheaper and more focused"],
    traps: ["“Supports 1M tokens” ≠ “uses 1M tokens well”: quality often degrades well before the limit", "Important info in the middle of a long context is used less reliably (“lost in the middle”)"],
    numbers: ["KV cache for an 8B GQA model at 128k context ≈ 16 GB per sequence in bf16"],
  },
  "scaling-laws": {
    works: ["Fit your own small-scale sweeps (3–5 sizes) before a big run; extrapolate loss, not benchmark scores", "Train smaller models on far more tokens than Chinchilla-optimal if they'll serve lots of traffic", "Use μP or careful LR scaling so hyperparameters transfer across sizes"],
    traps: ["Extrapolating downstream accuracy: it can jump or plateau even when loss is smooth", "Comparing scaling curves across different data mixes or tokenizers"],
    numbers: ["Chinchilla rule of thumb: ~20 tokens per parameter for compute-optimal", "Llama 3 8B saw ~15T tokens: ~1,900 tokens per parameter, far past compute-optimal on purpose"],
  },
  data: {
    works: ["Deduplicate aggressively (exact and near-duplicate)", "Quality classifiers (e.g. trained to spot “educational” text) give big gains: FineWeb-Edu style filtering", "Ablate data mixes at small scale; data choices often matter more than architecture"],
    traps: ["Benchmark contamination: test questions in the training data inflate scores", "Over-filtering removes diversity (dialects, code styles, rare topics)", "Training repeatedly on model-generated text without fresh human data degrades the tails"],
    numbers: ["Repeating data up to ~4 epochs costs little; beyond that returns diminish fast (Muennighoff et al., 2023)"],
  },
  "rl-basics": {
    works: ["Always subtract a baseline (value function or group mean) from returns", "Normalise advantages per batch", "Start with PPO from a well-tested library before writing your own"],
    traps: ["Reward bugs: the agent optimises exactly what you wrote, not what you meant", "Reporting a single seed: RL results vary wildly across seeds; use 5+"],
    numbers: ["Policy-gradient variance grows with episode length: discounting and baselines tame it"],
  },
  "q-learning": {
    works: ["DQN essentials: replay buffer, target network, Huber loss; Double DQN to reduce overestimation", "For continuous control, use SAC or TD3 instead"],
    traps: ["Overestimation from max over noisy Q values", "Too-small replay buffers forget rare experience"],
    numbers: ["DQN on Atari used 50M frames per game: RL is data hungry"],
  },
  alphazero: {
    works: ["MCTS + a learned policy/value network beats either alone", "Add Dirichlet noise at the root for exploration during self-play", "More simulations at test time is a reliable knob: search is test-time compute"],
    traps: ["Self-play collapsing to a narrow strategy without exploration noise", "Evaluating only against older versions of yourself"],
    numbers: ["AlphaZero used 800 simulations per move in training self-play"],
  },
  sft: {
    works: ["Quality over quantity: a few thousand excellent examples beat a million mediocre ones (LIMA)", "LoRA or QLoRA for most fine-tuning; rank 8–64 on all linear layers", "Mask the loss on prompt tokens; train only on assistant responses", "Use the base model's exact chat template"],
    traps: ["Too many epochs: the model memorises answers and loses general ability", "Fine-tuning to add knowledge: SFT is better at teaching format and style than new facts (use retrieval for facts)", "LR too high for full fine-tuning (use ~1e-5; LoRA tolerates ~1e-4)"],
    numbers: ["QLoRA fine-tunes a 65B model on one 48 GB GPU", "LoRA rank 16 on a 7B model: ~40M trainable params (~0.6%)"],
  },
  "rlhf-ppo": {
    works: ["KL penalty against the reference model: keeps outputs fluent and limits reward hacking", "Collect fresh comparisons on the current policy's outputs as it improves", "Watch length: reward models love longer answers, so track it and control for it"],
    traps: ["Over-optimising the reward model: true quality peaks then falls", "Annotator disagreement (~25–35% on hard comparisons) caps reward model accuracy"],
    numbers: ["Reward model pairwise accuracy is typically 65–75% on held-out human preferences"],
  },
  "grpo-dpo": {
    works: ["DPO for preference tuning when you have pairs and want simple, stable training", "GRPO with verifiable rewards (unit tests, exact answers) for reasoning", "Filter prompts to those the model solves sometimes (pass rate between ~10% and ~90%)"],
    traps: ["DPO lowering the likelihood of both chosen and rejected responses; monitor chosen log-prob", "Reward hacking a verifier (e.g. printing the expected output, special-casing tests)", "Learning rate too high for DPO (use ~5e-7 to 5e-6 for full fine-tuning)"],
    numbers: ["DPO β: 0.1 is the common default", "GRPO group size: 8–64 samples per prompt"],
  },
  alignment: {
    works: ["Layered defences: training, input/output classifiers, monitoring, rate limits", "Red-team before release, with people and automated attackers", "Evals for specific failures (sycophancy, over-refusal, deception) tracked over training"],
    traps: ["Optimising refusal rate alone: over-refusal is also a failure", "Assuming a safety eval passed once stays passed after the next fine-tune"],
    numbers: ["Best-of-N jailbreaking: success probability 1 − (1 − p)^N climbs fast with N"],
  },
  distillation: {
    works: ["Distil from a big model's outputs (sequence-level: just SFT on its generations); it's how most small strong models are made", "Logit distillation with temperature 2–4 when you have the teacher's logits", "Quantise for deployment: 8-bit is nearly free; 4-bit (GPTQ/AWQ) costs a little"],
    traps: ["Distilling only easy prompts: the student inherits the teacher's style but not its hard-case ability", "Model merging between unrelated fine-tunes (different base models) produces garbage"],
    numbers: ["4-bit weights: a 70B model fits in ~40 GB", "Pruning 50% of weights unstructured often costs little accuracy but gives little speedup without sparse kernels"],
  },
  diffusion: {
    works: ["Latent diffusion (diffuse in a VAE's latent space) for images and video", "Flow matching / rectified flow is now the default training objective for new models", "Classifier-free guidance scale ~3–7 for images; too high oversaturates", "Distil to few-step samplers (consistency, adversarial distillation) for fast inference"],
    traps: ["Too few sampling steps without a distilled model: blurry or noisy outputs", "Mismatched noise schedules between training and sampling"],
    numbers: ["DDPM used 1,000 steps; modern samplers use 20–50, distilled models 1–4"],
  },
  "gans-flows": {
    works: ["Use GAN losses as an add-on (perceptual + adversarial in VAEs, super-resolution, diffusion distillation)", "Spectral normalisation and two-timescale learning rates stabilise GANs"],
    traps: ["Judging a GAN by loss curves: they don't tell you sample quality; look at samples and FID", "Mode collapse that looks fine on a handful of samples"],
    numbers: ["StyleGAN2 FFHQ FID ≈ 2.8: GANs remain excellent on narrow domains"],
  },
  vision: {
    works: ["Start from a pretrained ViT/SigLIP/DINOv2 backbone; linear probe first, then fine-tune", "CLIP-style zero-shot classification as an instant baseline", "For VLMs, image resolution and token budget are the main quality/cost knobs"],
    traps: ["Training ViTs from scratch on small datasets: worse than CNNs without heavy augmentation", "CLIP's weaknesses: counting, spatial relations, attribute binding, reading small text"],
    numbers: ["ViT-B/16: 86M params, 196 tokens per 224×224 image", "CLIP was trained on 400M image–text pairs"],
  },
  audio: {
    works: ["Whisper (or a distilled variant) for transcription out of the box", "16 kHz mono for speech, 44.1/48 kHz for music", "Voice activity detection before ASR to skip silence and cut hallucinations"],
    traps: ["Whisper hallucinating text in silence or music", "Sample-rate mismatches (feeding 44.1 kHz audio to a 16 kHz model) without resampling"],
    numbers: ["Whisper: 30-second windows, 80 mel bins, 10 ms hop", "Neural codecs: 1.5–24 kbps vs ~1,400 kbps for CD audio"],
  },
  "ssm-mamba": {
    works: ["Hybrids (mostly Mamba/linear layers + a few attention layers) give long-context efficiency without losing recall", "Use the official fused kernels; a naive scan is very slow"],
    traps: ["Pure SSMs on exact copying and retrieval from long context", "Expecting SSM speedups at short sequence lengths: attention is already cheap there"],
    numbers: ["Mamba state per layer is constant: no KV cache growth with length"],
  },
  moe: {
    works: ["Top-k routing (k = 1–8) with load balancing; fine-grained experts plus shared experts", "Expert parallelism across GPUs with efficient all-to-all", "For local inference, offload rarely used experts or quantise"],
    traps: ["Router collapse without balancing", "Fine-tuning MoEs on small data: experts can overfit or the router can shift; lower LR and keep balancing on", "Expecting low memory: every expert must be loaded"],
    numbers: ["DeepSeek-V3: 671B total, 37B active (~5.5%)"],
  },
  "world-models": {
    works: ["Model-predictive control with a learned model: re-plan every step, short horizons", "Ensembles of dynamics models to estimate uncertainty and avoid exploiting model errors", "For robots: imitation learning with action chunking and diffusion/flow policies, then RL fine-tuning"],
    traps: ["Planners exploiting the model's mistakes (imagined shortcuts through walls)", "Long open-loop rollouts: errors compound"],
    numbers: ["Most real-robot policies run at 10–50 Hz; action chunks of ~1 second"],
  },
  reasoning: {
    works: ["For hard problems, give the model room to think (reasoning mode or effort setting) instead of over-engineering prompts", "Self-consistency or best-of-N with a checker when answers are verifiable", "Ask for the final answer in a fixed, parseable format"],
    traps: ["Overthinking easy queries: higher latency and cost for nothing", "Trusting the chain of thought as a faithful explanation", "Letting a weak verifier choose among many samples: it gets exploited"],
    numbers: ["Test-time accuracy often scales roughly with the log of thinking tokens or samples"],
  },
  looped: {
    works: ["Randomise loop counts during training so the model works at any depth", "Re-inject the input every loop (“recall”) to avoid drifting", "Deep supervision (loss at several loop counts) makes training far easier"],
    traps: ["Training at a fixed depth then expecting extrapolation", "Overthinking: more loops can degrade an unstable model"],
    numbers: ["Tiny recursive models (~7M params) can rival far larger LLMs on narrow puzzle tasks; they are not general chat models"],
  },
  ttt: {
    works: ["Linear-attention / delta-rule layers as cheap long-context memory inside hybrids", "Test-time fine-tuning on the task's own examples (e.g. ARC) gives big jumps on few-shot puzzles"],
    traps: ["Plain linear attention without a delta rule or gating: memories interfere and overwrite", "Comparing TTT layers to attention only at short context, where attention wins"],
    numbers: ["A fixed-size memory of d×d numbers can store at most ~d independent key–value pairs cleanly"],
  },
  precision: {
    works: ["bf16 for training compute, fp32 master weights and optimizer states", "fp8 training with per-tensor or per-block scaling on Hopper/Blackwell (DeepSeek-V3 did this at scale)", "int8/int4 weight-only quantisation for inference"],
    traps: ["fp16 without loss scaling: gradients underflow", "Accumulating sums in low precision (use fp32 accumulators)", "Quantising outlier-heavy activations naively"],
    numbers: ["Adam training memory ≈ 16 bytes per parameter (bf16 weights + grads, fp32 master + 2 moments)", "bf16 has fp32's range but only ~3 significant decimal digits"],
  },
  distributed: {
    works: ["FSDP / ZeRO-3 first: shards everything, minimal code changes", "Add tensor parallelism within a node (NVLink), pipeline/data parallelism across nodes", "Activation checkpointing trades ~30% more compute for big memory savings"],
    traps: ["Tensor parallelism across slow interconnects", "Not checkpointing often: at scale hardware fails daily"],
    numbers: ["Good large runs reach 35–55% model FLOPs utilisation (MFU)"],
  },
  kernels: {
    works: ["Profile before optimising (torch.profiler, Nsight)", "torch.compile for free fusion; Triton for custom kernels", "FlashAttention for attention, always"],
    traps: ["Optimising compute in a memory-bound op (elementwise, norms, small-batch decode)", "Tiny kernels launched thousands of times: launch overhead dominates (use CUDA graphs)"],
    numbers: ["H100: ~990 dense bf16 TFLOP/s vs ~3.35 TB/s HBM bandwidth: you need ~300 FLOPs per byte to be compute-bound"],
  },
  inference: {
    works: ["Use a serving engine (vLLM, SGLang, TensorRT-LLM): continuous batching + paged KV cache", "Prefix caching for shared system prompts and multi-turn chats", "Speculative decoding for latency; quantisation for throughput and memory"],
    traps: ["Measuring throughput at batch size 1 or latency at batch size 256: report both, at realistic load", "KV cache memory blowing up with long contexts × many concurrent users"],
    numbers: ["Decode is memory-bound: tokens/s at batch 1 ≈ bandwidth / model bytes", "Speculative decoding typically gives 2–3× lower latency"],
  },
  "tool-use": {
    works: ["Few, well-described tools with clear names, typed parameters and examples in the descriptions", "Return concise, structured tool results; truncate huge outputs", "Validate tool inputs and return helpful error messages the model can recover from"],
    traps: ["Dozens of overlapping tools: the model picks the wrong one", "Tools that silently fail or return giant blobs"],
    numbers: ["Tool definitions cost context tokens on every call: keep them tight"],
  },
  "agent-loops": {
    works: ["Simple loops first: model → tool → result → model, until done", "Give the agent a way to check its work (tests, linters, screenshots)", "Persist progress to files (plans, notes, TODOs) so long tasks survive context resets: the idea behind Ralph-style loops"],
    traps: ["No stopping condition: loops that burn money", "Letting agents take irreversible actions without confirmation", "Multi-agent setups where a single agent would do"],
    numbers: ["Per-step success compounds: 95% per step gives only ~36% success over 20 steps"],
  },
  "context-engineering": {
    works: ["Put stable instructions first and use prompt caching", "Retrieve just-in-time instead of preloading everything", "Summarise or clear old tool outputs; keep a running notes file"],
    traps: ["Contradictory instructions accumulated over time", "Stuffing the window: more context can mean worse attention to what matters"],
    numbers: ["Cached prompt tokens are typically billed at a small fraction of the normal input price"],
  },
  evals: {
    works: ["Build a small, sharp eval set from real failures before tuning anything", "Use code-based checks where possible, LLM judges with clear rubrics otherwise", "Read transcripts; numbers hide failure modes"],
    traps: ["Contaminated public benchmarks", "LLM judges preferring longer answers or their own model family", "Tiny eval sets: ±5% noise with 100 examples"],
    numbers: ["Standard error of accuracy ≈ √(p(1−p)/n): ~5% at n = 100, ~1.6% at n = 1,000"],
  },
  interpretability: {
    works: ["Linear probes to test whether information is present", "Sparse autoencoders / transcoders to find interpretable features", "Activation patching to test causal claims"],
    traps: ["Probes finding information the model doesn't actually use", "Over-interpreting single neurons (they're polysemantic)"],
    numbers: ["SAEs on large models use millions of features; many are interpretable, many aren't"],
  },
  "research-method": {
    works: ["Start with the smallest experiment that could change your mind", "Fix seeds, log everything, change one thing at a time", "Write the expected result before running"],
    traps: ["Comparing to an under-tuned baseline", "Results within seed noise reported as improvements"],
    numbers: ["Run at least 3 seeds before believing a small difference"],
  },
  "continual-learning": {
    works: ["Replay a slice of old data when fine-tuning on new data", "LoRA adapters per task, keeping the base frozen", "Retrieval for new facts instead of retraining"],
    traps: ["Catastrophic forgetting after narrow fine-tunes", "Measuring only the new task"],
    numbers: ["Mixing even ~5–10% of original data during fine-tuning often prevents most forgetting"],
  },
  "frontier-radar": {
    works: ["Read papers with code and reproduce the smallest claim", "Track a few trusted sources rather than everything"],
    traps: ["Hype from a single benchmark", "Ignoring cost and latency in “state of the art” claims"],
    numbers: ["The useful half-life of a specific SOTA number is months"],
  },
  intelligence: {
    works: ["Judge systems by generalisation to new tasks, not by benchmark scores alone"],
    traps: ["Anthropomorphising fluent text", "Assuming a capability is absent because one prompt failed"],
    numbers: ["ARC-AGI-style tests deliberately measure skill acquisition from few examples"],
  },
};

Object.assign(PRACTICE, {
  "ai-physics": {
    works: ["Count FLOPs before choosing hardware: 2 × params per token forward, 6 × params × tokens to train", "Ask whether the work splits into independent pieces; if it does, it belongs on a GPU", "Think in dimensions as lists of numbers, not pictures: the maths is the same in 3 or 3 million"],
    traps: ["Picturing high-dimensional spaces as if they were 3D (almost everything is far from everything)", "Comparing chips by peak FLOP/s alone: memory bandwidth decides most real workloads", "Assuming AI needed new maths: it needed cheap multiply-adds at enormous scale"],
    numbers: ["GPT-3: ≈ 3 × 10²³ FLOPs; one H100 at peak would take ~10 years", "AlexNet: two 3 GB GTX 580s, 5–6 days, top-5 error 15.3% vs 26.2%"],
  },
  electrons: {
    works: ["Reason with the exponential: carriers, leakage and diode current all go as e^(−E/kT)", "Remember kT ≈ 26 meV at room temperature and compare every energy to it", "Use n·p = n_i² to get minority carriers from majority ones"],
    traps: ["Thinking a semiconductor is a ‘weak conductor’: it's a controllable one", "Forgetting that heat makes silicon conduct more but metal conduct less", "Treating dopant atoms as rare in effect: one in a million changes conductivity a million-fold"],
    numbers: ["Silicon: band gap 1.12 eV, n_i ≈ 10¹⁰ cm⁻³ at 300 K, 5 × 10²² atoms per cm³", "Diode: every 60 mV forward multiplies current by ~10"],
  },
  transistor: {
    works: ["Read transistor curves on a log scale: the subthreshold slope tells you how good the gate is", "Treat the gate as a capacitor: switching energy is C·V²", "Expect leakage to rise several-fold per 50 °C"],
    traps: ["Reading node names (3 nm) as physical sizes", "Assuming supply voltage will keep falling: it has been stuck near 0.7–0.9 V for ~20 years", "Forgetting that the drain also controls the barrier in short channels"],
    numbers: ["Best possible: 59.5 mV per decade at 300 K; real devices 65–90", "FinFET from 2011 (Intel 22 nm), gate-all-around from 2022 (Samsung 3 nm)"],
  },
  logic: {
    works: ["Build from NAND (or NOR): it's universal", "Count partial products to estimate multiplier cost: n² for n bits", "Pipeline long paths to raise clock rate, accepting more latency per result"],
    traps: ["Assuming precision halving gives 4× throughput: chips usually give 2× because data movement doesn't shrink", "Ignoring carry chains when estimating adder speed", "Thinking binary is arbitrary: it's the noise-robust, zero-static-power choice in CMOS"],
    numbers: ["Full adder ≈ 28 transistors in CMOS; NAND gate = 4", "fp32 multiplies 24-bit fractions, bf16 8, fp8 E4M3 4"],
  },
  "memory-physics": {
    works: ["Estimate energy by counting bytes moved, not operations", "Ask how many times each loaded byte is reused; below ~300 on an H100 you're memory-bound", "Keep hot data in registers and shared memory; touch HBM once"],
    traps: ["Blaming latency on distance: it's protocol, queues and row activation", "Expecting SRAM to keep shrinking: it has nearly stopped", "Measuring bandwidth without enough requests in flight"],
    numbers: ["45 nm (Horowitz): fp32 multiply 3.7 pJ, 32-bit SRAM read 5 pJ, 32-bit DRAM read 640 pJ", "DRAM refresh every 64 ms (32 ms above 85 °C); HBM latency ~550–700 cycles on H100"],
  },
  "power-wall": {
    works: ["Model power as C·V²·f plus leakage; voltage is the lever", "Prefer more slower units when the work is parallel", "Check whether a workload is parallel before buying throughput hardware"],
    traps: ["Expecting clocks to rise again", "Ignoring Amdahl's law: a 5% serial part caps speed-up at 20×", "Assuming cooling can scale indefinitely: ΔT and thermal resistance cap it"],
    numbers: ["Desktop clocks: 3.8 GHz in 2004, ~6 GHz peak boost today", "Halving clock and voltage cuts switching power ~8×"],
  },
  "gpu-design": {
    works: ["Keep warps convergent: arrange data so whole warps take the same branch", "Use warp shuffles for reductions inside a warp", "Put enough bytes in flight: wide loads, many warps or several independent loads per thread"],
    traps: ["Chasing 100% occupancy when reuse matters more", "Forgetting that half-empty warps still cost full issue slots", "Treating tensor-core peak as reachable for non-matmul work"],
    numbers: ["H100: 132 SMs, 64 warps per SM, 65,536 registers per SM, 528 tensor cores", "Little's law: 3.35 TB/s × ~350 ns ≈ 1.2 MB in flight"],
  },
  cpu: {
    works: ["Measure before guessing: perf stat shows instructions per clock and branch-miss rate in seconds", "Break long dependency chains with several accumulators so the core can overlap them", "Turn unpredictable if/else on data into branch-free arithmetic or masks", "Keep hot data small and contiguous so it stays in L1/L2"],
    traps: ["Benchmarking on sorted or repeated test data that makes branches predictable when real data isn't", "Assuming more threads always help: Amdahl's serial part caps the speed-up", "Reading -O0 timings: the compiler hasn't removed anything yet", "Ignoring that a cache miss to DRAM costs ~100–300 clocks"],
    numbers: ["Mispredicted branch: ~15–20 clocks thrown away", "L1 hit ~4–5 clocks, L2 ~12–15, L3 ~40–60, DRAM ~100–300", "Modern big cores: 4–8 instructions decoded per clock, 300–600 in flight"],
  },
  languages: {
    works: ["Write Python for structure and push the inner loops into NumPy, PyTorch or a compiled kernel", "Read the generated code when speed matters: dis for Python, objdump or godbolt.org for C", "Let compilers optimise (-O2/-O3) and check with a profiler instead of hand-tuning blind"],
    traps: ["Element-by-element Python loops over arrays: ~100× slower than one vectorised call", "Assuming high-level code is slow by nature: a thin layer over fast code costs almost nothing", "Benchmarking code the compiler deleted because its result was never used"],
    numbers: ["CPython: roughly 30–100 ns per simple operation in a loop", "x86-64 instructions: 1 to 15 bytes each", "FORTRAN I (1957): the first widely used optimising compiler"],
  },
  "gpu-history": {
    works: ["Think of any GPU job as one thread per output element, the way a pixel shader does", "Check whether your problem is independent per element before reaching for a GPU", "Use graphics-era tricks where they fit: texture caches and hardware interpolation still exist"],
    traps: ["Assuming a gaming GPU and a data-centre GPU differ only in price: memory size, fp64 and interconnect differ a lot", "Forgetting that data must cross PCIe to reach the GPU, as in the GPGPU days"],
    numbers: ["1080p at 60 fps: ~124 million pixels a second", "GeForce 8800 GTX (2006): 128 cores, ~0.35 TFLOP/s fp32", "AlexNet (2012): two GTX 580s, 3 GB each, ~5–6 days of training"],
  },
  circuits: {
    works: ["Estimate energy per switch as C·V² before anything else: it tells you why voltage matters most", "Check units by writing them out: volts × amps = watts, ohms × farads = seconds", "Treat long wires as RC lines: delay grows with length² unless repeated"],
    traps: ["Thinking electrons race through the wire: they drift at well under a mm per second; the signal is the wave", "Expecting a faster (lower-resistance) driver to save energy: the heat per charge is ½CV² whatever R is", "Lowering the supply voltage without noticing the current, and so the I²R loss in delivery, goes up"],
    numbers: ["One on transistor: ~10 kΩ, ~75 µA at 0.75 V", "Logic node: ~0.2 fF, ~940 electrons, ~0.11 fJ per 0→1→0", "A 700 W GPU at 0.8 V: ~875 A into the die"],
  },
  fab: {
    works: ["Use hp = k₁ λ / NA to sanity-check any claim about a process: k₁ below ~0.28 means multiple patterning", "Estimate chip cost as wafer price ÷ (dies per wafer × yield), with yield ≈ e^(−area × defect density)", "Prefer several smaller dies when a design is near the reticle limit"],
    traps: ["Reading “3 nm” or “2 nm” as a feature size: they're marketing names", "Ignoring the round wafer edge: big square dies waste more of it", "Forgetting packaging capacity (CoWoS, HBM), which has limited AI chip supply as much as wafers"],
    numbers: ["EUV: 13.5 nm light, NA 0.33 (High-NA 0.55)", "Reticle limit: 26 × 33 mm = 858 mm²", "300 mm wafer: ~707 cm²; ~70 dies of 800 mm² fit"],
  },
  edges: {
    works: ["Ask which limit binds: data movement and wires cost far more energy than arithmetic today", "Treat lower precision and less data movement as the main levers left, not faster transistors", "Read roadmaps (IRDS, foundry announcements) with node names as marketing, not lengths"],
    traps: ["Taking “3 nm” as a physical size: no feature on a 3 nm chip is 3 nm", "Quoting Landauer as today's bottleneck: chips are ~10⁴–10⁵× above it", "Expecting quantum computers to speed up dense matrix multiplies: no such algorithm is known"],
    numbers: ["Landauer at 300 K: kT·ln 2 ≈ 2.9 × 10⁻²¹ J", "Boltzmann floor: 60 mV per decade of current at room temperature", "Light: 30 cm per ns in vacuum; an on-chip wire ~20 mm per ns"],
  },
  "software-stack": {
    works: ["Profile before optimising: torch.profiler or Nsight Systems shows whether the CPU or the GPU is the bottleneck", "Fuse elementwise chains with torch.compile; capture small-batch decoding with CUDA Graphs", "Ship PTX alongside SASS (-gencode ...,code=compute_XX) so code runs on future GPUs"],
    traps: ["Timing GPU code without torch.cuda.synchronize(): launches return before the work is done", "Graph breaks from print() or .item() inside a compiled function, silently falling back to eager", "Calling .item() or .cpu() in a hot loop: each one makes the CPU wait for the GPU"],
    numbers: ["Eager PyTorch: roughly 5–10 µs of CPU per op; a kernel launch alone a few µs", "PyTorch 2 paper: torch.compile 2.27× inference, 1.41× training (geomean, 180+ models, A100)"],
  },
  "all-the-way-down": {
    works: ["Estimate any AI cost bottom-up: multiply-adds × energy each, plus bytes moved × energy per byte", "For a token: one multiply-add per weight, and every weight read once per pass", "Batch to share each weight read across users"],
    traps: ["Quoting peak TFLOP/s for single-user decoding, which is memory-bound", "Counting only arithmetic energy: data movement and idle power usually dominate", "Treating per-lane transistor counts and sizes as published facts: they are estimates"],
    numbers: ["H100: ~1.4 pJ per multiply-add chip-wide at peak (700 W ÷ 4.95 × 10¹⁴/s)", "One 0.2 fF node at 0.75 V: ~940 electrons, ~0.11 fJ, ~39,000× the Landauer limit", "70B at batch 1 on 2 H100s: ~21 ms per token, set by HBM"],
  },
  cuda: {
    works: ["Always guard with if (i < n) and round the grid up", "Check cudaGetLastError() after launches; debug with CUDA_LAUNCH_BLOCKING=1", "Keep data resident on the GPU and fuse small kernels"],
    traps: ["Synchronising across blocks inside a kernel (deadlock)", "Timing without torch.cuda.synchronize() (launches are asynchronous)", "Copying over PCIe for cheap work the CPU could do in place"],
    numbers: ["Kernel launch: a few µs; PCIe 4.0 ×16 ≈ 25 GB/s, 5.0 ≈ 50 GB/s, HBM3 3,350 GB/s", "A grid of 132·k blocks avoids a near-empty last wave on an H100"],
  },
  "gpu-from-scratch": {
    works: ["Profile first (nsys, then ncu on the hot kernel); most speedups are memory-layout fixes", "Fuse elementwise ops into neighbours; write Triton before CUDA", "Use vectorised 16-byte loads and keep the fastest thread index on the contiguous dimension"],
    traps: ["Optimising a kernel that takes 2% of the step", "Chasing occupancy instead of reuse", "Benchmarking without warm-up or torch.cuda.synchronize()"],
    numbers: ["H100: ~989 TFLOPS dense bf16, 3.35 TB/s HBM, so ~295 FLOPs per byte to be compute-bound", "A good GEMM reaches 70–80% of peak; a naive one, ~1%"],
  },
  "stability-at-scale": {
    works: ["QK-norm + z-loss + no biases + β₂ = 0.95 + grad clip 1.0 is a robust default", "μP or fitted LR/batch scaling laws instead of guessing at scale", "Automatic rewind-and-skip on spikes"],
    traps: ["fp16 instead of bf16 for big runs", "Weight decay on norms and embeddings", "Tuning LR at small width and reusing it unchanged at large width under SP"],
    numbers: ["z-loss coefficient ~1e-4", "PaLM skipped ~200–500 batches after a rewind"],
  },
  "lab-experiments": {
    works: ["4–6 rungs spanning ≥ 2 decades of compute", "Smooth metrics (answer log-likelihood, bits-per-byte) at small scale", "An intermediate 10–100× run before the big one"],
    traps: ["Deciding from the largest rung alone", "Ignoring throughput: a gain per token can be a loss per second", "Benchmarks at chance level used as signal"],
    numbers: ["Proxy models at ~150M ranked data recipes like 1B models ~80% of the time (DataDecide)"],
  },
  "model-psychology": {
    works: ["Keep chain-of-thought monitors out of the reward", "Fix the environment when hacks are found (stronger tests) rather than penalising mentions", "Model organisms and auditing games to test detection tools"],
    traps: ["Assuming safety training removes behaviours it never triggers", "Trusting a model's stated reasoning as a faithful record", "Evaluating only in setups the model can recognise as tests"],
    numbers: ["Reasoning models verbalised a used hint ~25–40% of the time (Anthropic, 2025)"],
  },
  "reality-gap": {
    works: ["Re-run any idea you want to adopt on your own setup, against a baseline you've tuned just as hard", "Use at least 3 seeds and report the spread; 5–10 when effects are small", "Prefer ideas that have been independently reproduced and adopted in open model recipes"],
    traps: ["Single-seed comparisons on noisy benchmarks", "Comparing against numbers copied from an old paper (different data, tokenizer, compute)", "Results only shown at one small scale"],
    numbers: ["Seeds needed for 80% power ≈ 16 · (noise / effect)²", "A gain of less than ~2 seed standard deviations from one run is weak evidence"],
  },
  "what-survived": {
    works: ["Copy a proven open recipe (Llama/OLMo/Qwen-style) and change one thing at a time", "Spend effort on data before architecture", "Adopt ideas that several independent labs converged on"],
    traps: ["Stacking five new tricks at once: you can't tell which helped or hurt", "Assuming a trick that helped a 100M model will help at 10B (or vice versa)"],
    numbers: ["Typical modern block: pre-RMSNorm, RoPE, GQA, SwiGLU, no biases, AdamW, warmup + cosine/WSD, bf16"],
  },
  "data-first": {
    works: ["Read 50 errors before changing the model", "Rank examples by cross-fitted loss and review the worst first", "Deduplicate before splitting; split by user, time or source"],
    traps: ["Scaling the model to compensate for noisy labels", "Scoring suspicion with a model trained on the same examples (it memorised them)", "Fixing the test set's labels but not the training set's (or the reverse)"],
    numbers: ["Typical crowd-labelled datasets have a few percent label errors; even ImageNet's validation set has ~6% (Northcutt et al., 2021)"],
  },
  "loss-doctor": {
    works: ["Check the step-0 loss against ln(classes) before anything else", "Overfit one batch to prove the pipeline can learn", "Log gradient norm and update-to-weight ratio, not just loss"],
    traps: ["Comparing runs with different LR schedules mid-way", "Choosing checkpoints on a tiny, noisy validation set", "Debugging the architecture when the bug is in the data loader"],
    numbers: ["Healthy Adam update/weight ratio ≈ 1e-3 per step", "Uniform-guess loss: ln(10) = 2.30, ln(50,000) ≈ 10.8"],
  },
  "which-tool": {
    works: ["Build an eval set from real inputs before choosing a method", "Climb the ladder: prompt → retrieval → fine-tune → distil → RL, stopping when evals pass", "Combine: RAG for facts, a light fine-tune for format"],
    traps: ["Fine-tuning to teach facts that change", "RL without a reward that resists gaming", "Not re-testing a plain prompt on each new base model"],
    numbers: ["A useful first eval set: 50–200 real examples with clear pass/fail criteria"],
  },
  "cost-reality": {
    works: ["Estimate with 6·N·D and your measured MFU before launching anything big", "Run the full pipeline at 1% scale first: it catches most bugs for 1% of the cost", "Serve with batching, quantisation and prefix caching"],
    traps: ["Budgeting the ideal formula: reality is 1.5–3× higher", "Measuring serving cost at batch 1 and extrapolating", "Pretraining from scratch when a fine-tune or distillation would do"],
    numbers: ["H100: ~989 dense bf16 TFLOP/s, 3.35 TB/s, 80 GB", "Llama 3.1 405B: 30.84M H100-hours reported", "DeepSeek-V3: 2.788M H800-hours, ~$5.6M at $2/hour (final run only)"],
  },
  "deep-cuts": {
    works: ["Read the status label before building on an idea", "Reproduce the smallest version of a claim yourself"],
    traps: ["Treating speculative ideas as settled because they're elegant"],
    numbers: ["Most ideas that look great in one paper don't become standard; the established ones here are the survivors"],
  },
});

export const RESULTS = {
  "01_linear_regression": ["Closed-form and gradient-descent weights agree to ~3 decimals", "Loss curve flattens within a few hundred steps at a good learning rate; diverges at ~2× the max stable rate"],
  "02_micrograd": ["Your autograd gradients match PyTorch's to ~1e-6", "A tiny MLP reaches ~100% on the moons dataset"],
  "03_mnist_mlp_cnn": ["MLP: ~97–98% test accuracy", "Small CNN: ~99% after a few epochs", "Misclassified digits are mostly genuinely ambiguous"],
  "04_vae": ["Reconstructions are recognisable but blurry", "Interpolating in latent space morphs one digit into another smoothly"],
  "05_bpe_tokenizer": ["Your tokenizer round-trips text exactly", "Compression (characters per token) rises as you add merges, with diminishing returns"],
  "06_gpt_from_scratch": ["Character-level loss falls from ~4.2 to ~1.5 or below", "Samples go from random characters to word-like text with structure"],
  "07_scaling_laws": ["Final loss vs compute is close to a straight line on log-log axes", "The fitted exponent is small (around 0.05–0.1 for loss vs compute)"],
  "08_sft_lora": ["The fine-tuned model follows the chat format reliably", "LoRA trains well under 1% of the parameters"],
  "09_reward_model_dpo": ["The reward model ranks pairs about as well as the noisy labels allow (~78% on decisive pairs: the labels themselves only reach that)", "DPO lifts the liked token from ~0.5 to ~4 per sequence, and β·log(π/π_ref) correlates ~0.8 with the true reward, without ever training a reward model"],
  "10_policy_gradient_ppo": ["CartPole reaches the maximum episode return (500)", "With a baseline, learning is visibly faster and less noisy"],
  "11_grpo": ["With answers written least-significant digit first, GRPO lifts accuracy on unseen sums from ~30% to ~68% in 400 steps", "Write the digits in normal order and the same setup goes 13% → 18%: most groups are all-wrong, so there is no gradient. Output format decides learnability", "Groups that are all-right or all-wrong contribute zero gradient; 60–90% of groups are like that here"],
  "12_diffusion": ["Generated 2D points reproduce the target shape", "DDIM with ~20 steps looks close to the full sampler"],
  "13_triton_kernels": ["Your fused kernel matches PyTorch's output to fp16 tolerance", "Fusion is faster for memory-bound ops; the plot shows bandwidth near the GPU's limit"],
  "14_looped": ["With more loops at test time, accuracy on longer inputs rises", "A fixed-depth baseline fails beyond its training length"],
  "15_mamba_ssm": ["Sequential and parallel scans give identical outputs", "Selective SSM solves selective copying; the non-selective one doesn't"],
  "16_test_time_training": ["With distinct random keys, linear attention degrades gracefully; the delta rule is strong well below d pairs, then falls faster", "When a key's value is updated, the delta rule recalls the latest value far more often than linear attention"],
  "17_moe": ["Without balancing, a few experts take most tokens", "With balancing, load is near uniform and loss is similar or better"],
  "18_superposition_sae": ["With sparse features, the toy model stores more features than it has dimensions", "The SAE recovers interpretable directions"],
  "19_agent_ralph_loop": ["The loop finishes the task, writing progress to a file between iterations", "In mock mode it runs without an API key"],
  "21_speculative_decoding": ["Outputs are identical to normal decoding", "Speedup depends on how often the draft's tokens are accepted"],
  "22_grokking": ["Training accuracy hits 100% within a few hundred steps; test accuracy jumps thousands of steps later", "Without weight decay the network stays memorised, with test accuracy near chance", "Grokked hidden units become single-frequency waves (most of each unit's energy in one frequency)"],
  "23_trees_boosting": ["Gradient boosting beats a single tree and usually the MLP on tabular data"],
  "24_q_learning_dqn": ["Tabular Q-learning solves the gridworld (greedy path: 14 steps)", "DQN on CartPole reaches a best 20-episode average around 350, noisily: returns collapse and recover"],
  "25_alphazero": ["With 200 MCTS simulations per move the trained agent draws every game against perfect minimax", "Search is half the strength: the same network at 10–50 simulations loses half its games to minimax"],
  "26_gan": ["On 2D mixtures, some runs cover all modes and some collapse", "On MNIST, recognisable digits after a few epochs"],
  "27_distill_quantize": ["Soft labels on the same 5,000 images don't help a tiny student (91.7% vs 92.2%); distilling on unlabelled images does: 95.2% with no labels at all", "int8 costs nothing; 50% pruning costs 0.3 points but 80% collapses a 32-unit student to ~47%", "int4 per channel vs groups of 16: within 0.3 points here, because a tiny model has no outlier weights"],
  "28_gnn": ["The GCN classifies most nodes correctly from a few labels", "Accuracy drops with too many layers (over-smoothing)"],
  "29_hopfield": ["Corrupted patterns are restored", "Recall breaks down as you store more patterns than ~0.14 × neurons"],
  "30_continual": ["Fine-tuning on task B destroys task A accuracy", "Replaying a little task A data prevents most of the forgetting"],
  "32_ablation_lab": ["Learning-rate schedule (warmup + decay) and a good LR usually move loss the most", "Architecture tweaks (RoPE vs learned positions, SwiGLU vs GELU, RMSNorm vs LayerNorm) give small gains at this scale, often near seed noise", "Removing duplicated data helps validation loss noticeably"],
  "60_electrons_to_cuda": ["Silicon: ~1.1 × 10¹⁰ free electrons per cm³ at 300 K, ~500× more at 400 K; a modelled gate with n = 1.3 shows ~78 mV per decade", "NAND multiplier gates grow as bits^2.00", "Simulated scheduler busy fraction matches min(1, W·C/(C+L)) within a few %, saturating at 21 warps for C = 20, L = 400", "On a T4: ~220–250 GB/s on-GPU against ~10–12 GB/s over PCIe (about 20×), and a few µs per launch; bandwidth with 1 block is a small fraction of the peak", "Part 5: 1.41 pJ per multiply-add, 936 electrons and ~39,000× Landauer per bit; a 70B token costs ~10.8 J at batch 1 and ~0.14 J at batch 256"],
  "34_gpu_kernels": ["On a T4 (fp32, 4096×4096): naive ≈ 0.3–0.6 TFLOPS, coalesced ≈ 1–2, shared-memory tiles ≈ 2–3, Triton with autotune ≈ 60–80% of torch.matmul; torch (cuBLAS) ≈ 4–5 TFLOPS fp32, ~25–40 in fp16 tensor cores", "Padding a 32×32 shared tile to 33 columns removes the column-read slowdown (several × in a microbenchmark)"],
  "35_mup_transfer": ["Coordinate check: SP logit RMS blows up past width 1024 (1.8 → 10.6 from 64 to 2048); μP stays at 2.5 at every width", "LR sweep (128–2048, 400 steps): SP's optimum drifts down (2⁻⁸ → 2⁻⁹); μP's settles at 2⁻⁶ from width 512 on. The transfer is asymptotic: very narrow nets are still off"],
  "36_scaling_ladder": ["Fits on 5 rungs predict the 4× run's loss within ~1–2% when noise is low", "Prediction intervals widen fast with extrapolation distance; a trick that wins on small rungs can lose at the target"],
  "37_sleeper_backdoor": ["Clean-data safety training leaves about half the backdoor (exact trigger 100% → 47%) while near-misses fall 74% → 13%", "Adversarial training on near-misses removed it completely in this 3-layer model, unlike in LLMs (Hubinger et al., 2024): small models can't separate the exact trigger from its neighbours"],
  "38_reward_hacking": ["No monitor: open hacking dominates when tests are weak", "Training against the monitor: hidden hacking takes over and the monitor catches almost nothing", "Tests catching most hacks: honest solving wins in all three settings"],
  "39_capstone_reasoning_grpo": ["Prediction (not yet run): greedy GSM8K accuracy 30–42% before RL, 40–50% after 150 steps, much of the early gain from format", "Prediction: 40–60% of groups carry no signal; completion length dips, then creeps up"],
  "40_capstone_speedrun_muon": ["Prediction (not yet run): AdamW final validation loss 3.45–3.65 at ~390M tokens; Muon 0.05–0.12 lower", "Prediction: Muon reaches AdamW's final loss in 60–75% of the steps. Offline toy check: 30 of 40 steps"],
  "41_capstone_sae_gpt2": ["Prediction (not yet run): FVU 0.12–0.20 at 20M tokens with 16k latents, k = 32", "Prediction: spliced-in loss +0.2 to +0.5 nats; 6–8 of 10 sampled features nameable from their top examples"],
  "42_rl_narrows": ["Measured (toy): pass@1 25% → 40%; the base model overtakes RL from k = 8", "Measured (neural policy, held-out): pass@1 39% → 78%, pass@256 94% → 82%", "Measured: 1 → 8 strategy families moved the crossing from k = 8 to k = 64; Δpass@256 −0.12 → −0.01, never positive"],
  "43_model_collapse": ["Measured, replace regime: held-out perplexity 742 → 1,469; distinct 4-grams 152k → 52k over 10 generations", "Measured, accumulate: perplexity +27% (I predicted under 10%: wrong); 4-grams −9%", "Measured: temperature 0.6 collapses to 2 distinct words in 10 generations"],
  "44_eval_error_bars": ["Measured: A 96.1%, B 96.6%; paired 95% CI for the gap [−0.5, +1.4] points (unpaired [−1.3, +2.2]): can't tell", "Measured: B moves ±0.3 points across 8 seeds, as much as the gap", "Measured: on 50–200-question subsets, B looks worse in about 1 draw in 5"],
  "45_thinking_budget": ["Measured (model): forced thinking peaks at 48% near 41 steps and falls to 25% at 2,048; stop-when-done reaches 58%", "Refuted: my d/(1−c) rule for the best budget; it tracks depth but hardly the check rate (4.5× spread)", "Prediction (GPU part, not yet run): a 1.5B instruct model on GSM8K peaks around 256–512 forced tokens"],
  "46_inside_view_probes": ["Not yet run against Claude (no API key in my environment); the mock only checks the code", "Prediction: letter counting near-perfect regardless of token count; under 10% of correct answers abandoned under pushback", "Prediction: “don't mention X” now reduces mentions, so my own-report card on it is probably out of date"],
  "47_self_improvement": ["Measured, 2-carry sums after 12 rounds: K = 1 36%, K = 16 60%, problems at the edge 68%; real labels 89%", "Measured: a checker letting 10% of wrong samples through ended at 22%, below K = 1 with a perfect checker", "Measured: 4-carry sums reached only 10% by self-training vs 84% with labels: the loop climbs one rung at a time"],
  "48_lr_cooldown": ["Measured: constant 2.309, cosine 2.184, linear 2.189, WSD 2.214 final loss (WSD 1.4% behind cosine on this short run)", "Measured: cooldowns branched off the constant run matched cosine runs of the same length within 1%", "Measured: 10% of tokens moved into the cooldown shifted the model +0.30 nats toward that corpus; the same moved early shifted it −0.10"],
  "58_conjecture_tests": ["Partly refuted: top-20 distillation kept 76% of the gain (I said 90%); top-1 was worse than no teacher", "Refuted: memorisation peaks for data seen mid-training (+0.65 nats/char), not late (+0.02 at the end)", "Confirmed, more strongly than claimed: 5% false accepts did ~37× the damage of 5% false rejects"],
  "49_emergence_mirage": ["Measured: exact match 0 → 2% → 30% → 82% across three neighbouring sizes; exact ≈ per-digit⁷", "Refuted: my forecast from the small models predicted 42% exact match for the largest; it scored 98%", "Per-digit accuracy itself accelerated mid-range: a real transition hiding under the metric mirage"],
  "33_data_vs_model": ["With 30% label noise, cleaning labels beats doubling or quadrupling the model", "With clean data, the bigger model helps a little; with noisy data it mostly memorises noise"],
  "31_precision_memory": ["Memory measured on the GPU matches the calculator within ~10%", "fp16 without loss scaling underflows; bf16 doesn't"],
};
