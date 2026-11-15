// The curriculum: chapters you poke in the browser, and projects you run on Colab.
// live: true means /chapters/<id>.js exists.

export const REPO = "srivtx/ai-anatomy";
export const colabUrl = (file) => `https://colab.research.google.com/github/${REPO}/blob/main/notebooks/${file}.ipynb`;
export const githubUrl = (file) => `https://github.com/${REPO}/blob/main/notebooks/${file}.ipynb`;

export const LEVELS = [
  {
    id: "math",
    name: "The maths you actually need",
    tagline: "Vectors, matrices, probability and information, all as pictures",
    chapters: [
      { id: "linear-algebra", title: "Matrices are machines that move space", blurb: "Drag a 2×2 matrix and watch the plane stretch, rotate and collapse. Dot products, eigenvectors, rank.", live: true },
      { id: "probability", title: "Probability and information", blurb: "Entropy, cross-entropy and KL divergence as bits you can count. Maximum likelihood from scratch.", live: true },
    ],
  },
  {
    id: "first-principles",
    name: "Why learning works",
    tagline: "From hand-written rules to a machine that fits its own",
    chapters: [
      { id: "why-learn", title: "Why not just write the rules?", blurb: "Fit a line by hand, then let gradient descent beat you. Data, model, loss, optimizer.", live: true },
      { id: "gradient-descent", title: "Rolling downhill", blurb: "Race SGD, momentum and Adam across real loss landscapes. Watch a big learning rate explode.", live: true },
      { id: "perceptron", title: "The perceptron", blurb: "Click to add points; the 1958 learning rule draws a line. Then logistic regression and cross-entropy.", live: true },
      { id: "xor", title: "XOR and the first AI winter", blurb: "One neuron can't do XOR. A hidden layer can. Train it live and watch it bend space.", live: true },
      { id: "backprop", title: "Backpropagation, by hand", blurb: "Step a computational graph forward and backward. The chain rule is the whole trick.", live: true },
      { id: "generalization", title: "Memorising vs understanding", blurb: "Overfit a curve, find the U-shaped validation error, then meet double descent.", live: true },
    ],
  },
  {
    id: "classical",
    name: "Classical machine learning",
    tagline: "The methods that still win on tables, and the ideas deep learning borrowed",
    chapters: [
      { id: "knn-trees", title: "Neighbours, trees and forests", blurb: "k-NN, decision trees, random forests and gradient boosting on the same data, live.", live: true },
      { id: "svm-kernels", title: "Margins and the kernel trick", blurb: "Support vector machines, and how a kernel bends space without computing the bend.", live: true },
      { id: "unsupervised", title: "Finding structure without labels", blurb: "k-means clustering and PCA, step by step on points you place.", live: true },
      { id: "bayes", title: "Thinking like a Bayesian", blurb: "Priors, posteriors, naive Bayes spam filtering and calibrated confidence.", live: true },
    ],
  },
  {
    id: "deep-learning",
    name: "The deep learning toolkit",
    tagline: "What makes deep networks trainable, and the shapes they come in",
    chapters: [
      { id: "training-tricks", title: "Making deep nets trainable", blurb: "Initialisation, normalisation, residuals and dropout, with activations you can watch die.", live: true },
      { id: "cnn", title: "Convolutions: how nets see", blurb: "Draw a digit, slide kernels over it, and build up feature maps and receptive fields.", live: true },
      { id: "embeddings", title: "Meaning as geometry", blurb: "One-hot vs embeddings, cosine similarity, analogies and contrastive learning.", live: true },
      { id: "rnn", title: "Sequences and memory", blurb: "Unroll an RNN, watch gradients vanish through time, then open an LSTM's gates.", live: true },
      { id: "autoencoders", title: "Autoencoders and VAEs", blurb: "Train an autoencoder in your browser, then click its latent space to dream new glyphs.", live: true },
      { id: "gnn", title: "Graph neural networks", blurb: "Message passing on a graph you edit: molecules, social networks and maps.", live: true },
      { id: "energy-memory", title: "Hopfield networks and Boltzmann machines", blurb: "Store patterns as valleys of energy and recall them from noise. The 2024 Nobel physics prize.", live: true },
    ],
  },
  {
    id: "language-models",
    name: "Language models from scratch",
    tagline: "Bytes to tokens to attention to GPT",
    chapters: [
      { id: "tokenization", title: "BPE: inventing tokens", blurb: "Train byte-pair encoding live and watch merges become a vocabulary.", live: true },
      { id: "next-token", title: "Predict the next token", blurb: "Train an n-gram model on text in the page, sample with temperature, top-k and top-p.", live: true },
      { id: "attention", title: "Attention", blurb: "Queries, keys and values on a real sentence; causal masks, heads and RoPE.", live: true },
      { id: "transformer", title: "The transformer block", blurb: "Residual stream, layer norm, MLP. Count parameters, FLOPs and KV cache for any config.", live: true },
      { id: "long-context", title: "Long context", blurb: "RoPE scaling, sliding windows, sparse attention, MLA and the lost-in-the-middle problem.", live: true },
      { id: "scaling-laws", title: "Scaling laws", blurb: "Spend a compute budget: how big a model, how many tokens? Chinchilla, live.", live: true },
      { id: "data", title: "Where the data comes from", blurb: "Crawl, filter, deduplicate, mix and synthesise: the pipeline that decides what a model knows.", live: true },
    ],
  },
  {
    id: "rl",
    name: "Reinforcement learning",
    tagline: "Learning from reward instead of answers",
    chapters: [
      { id: "rl-basics", title: "Reinforcement learning from zero", blurb: "A bandit, a policy gradient and a baseline. The core of every RL method, live.", live: true },
      { id: "q-learning", title: "Q-learning and DQN", blurb: "An agent learns a gridworld by trial and error; watch the value map fill in.", live: true },
      { id: "alphazero", title: "Search plus learning: AlphaZero", blurb: "Monte Carlo tree search on tic-tac-toe, guided by a value estimate.", live: true },
    ],
  },
  {
    id: "post-training",
    name: "From autocomplete to assistant",
    tagline: "SFT, reward models, RLHF, PPO, DPO and GRPO from first principles",
    chapters: [
      { id: "sft", title: "Supervised fine-tuning", blurb: "Chat templates, loss masking and LoRA: how a base model learns to answer.", live: true },
      { id: "rlhf-ppo", title: "RLHF and PPO", blurb: "Fit a reward model from preferences, then clip, penalise and step a policy.", live: true },
      { id: "grpo-dpo", title: "GRPO and DPO", blurb: "Drop the critic, drop the reward model: the methods behind open reasoning models.", live: true },
      { id: "alignment", title: "Alignment and reward hacking", blurb: "Constitutions, red-teaming, sycophancy, and agents that game their reward.", live: true },
      { id: "distillation", title: "Distillation, pruning and merging", blurb: "Teach a small model with a big one's soft labels; prune weights; average models.", live: true },
    ],
  },
  {
    id: "generative",
    name: "Model families",
    tagline: "Diffusion, GANs, vision, audio, state space models, experts and world models",
    chapters: [
      { id: "diffusion", title: "Diffusion and flow matching", blurb: "Noise a point cloud into fog, then denoise it back, one step at a time.", live: true },
      { id: "gans-flows", title: "GANs and normalizing flows", blurb: "A generator and a critic play a game in 1D; watch mode collapse happen.", live: true },
      { id: "vision", title: "Vision transformers and CLIP", blurb: "Cut an image into patch tokens, then line images up with text.", live: true },
      { id: "audio", title: "Speech and audio models", blurb: "Waveforms to spectrograms to audio tokens: how Whisper hears and voice models speak.", live: true },
      { id: "ssm-mamba", title: "State space models and Mamba", blurb: "Linear recurrences, selective scans, and Vision Mamba's scan orders.", live: true },
      { id: "moe", title: "Mixture of experts", blurb: "Route tokens to experts, overflow capacity, and balance the load.", live: true },
      { id: "world-models", title: "World models and robots", blurb: "Predict the next frame, plan inside your own model: Dreamer, JEPA, Genie and VLAs.", live: true },
    ],
  },
  {
    id: "thinking",
    name: "Thinking, fast and slow",
    tagline: "Reasoning at test time, and models that compute longer",
    chapters: [
      { id: "reasoning", title: "Chain of thought and System 2", blurb: "Why writing steps adds compute, plus self-consistency, verifiers and test-time scaling.", live: true },
      { id: "looped", title: "Looped and recurrent-depth models", blurb: "Reuse the same layers more times to think longer. Solve bigger mazes than you trained on.", live: true },
      { id: "ttt", title: "Test-time training", blurb: "A memory that learns while it reads: linear attention vs the delta rule vs TTT.", live: true },
    ],
  },
  {
    id: "systems",
    name: "Systems: training and serving at scale",
    tagline: "The engineering that turns maths into a model on 10,000 GPUs",
    chapters: [
      { id: "precision", title: "Numbers: fp32, bf16, fp8, int4", blurb: "Flip bits and see what each format can represent, and why bf16 won training.", live: true },
      { id: "distributed", title: "Distributed training", blurb: "Data, tensor and pipeline parallelism, ZeRO and FSDP: a memory calculator for any model.", live: true },
      { id: "kernels", title: "GPU kernels and the roofline", blurb: "Memory vs compute bound, kernel fusion and FlashAttention's tiling.", live: true },
      { id: "inference", title: "Fast inference", blurb: "KV cache, quantisation, batching and speculative decoding, with the maths you can tune.", live: true },
    ],
  },
  {
    id: "agents",
    name: "Agents",
    tagline: "Models that use tools, loop and check their own work",
    chapters: [
      { id: "tool-use", title: "Tool use, at the token level", blurb: "What a function call actually is, constrained decoding, and MCP.", live: true },
      { id: "agent-loops", title: "Agent loops and the Ralph loop", blurb: "One long session vs a fresh context per iteration with state in files.", live: true },
      { id: "context-engineering", title: "Context engineering and memory", blurb: "What goes in the window: compaction, retrieval, sub-agents and memory files.", live: true },
      { id: "evals", title: "Evals: is it actually better?", blurb: "pass@k, confidence intervals, contamination and judges with biases.", live: true },
    ],
  },
  {
    id: "frontier",
    name: "Research and the frontier",
    tagline: "How researchers find out, and what intelligence might be",
    chapters: [
      { id: "interpretability", title: "Looking inside: superposition", blurb: "Train a toy model in the page and watch it pack five features into two dimensions.", live: true },
      { id: "research-method", title: "How researchers improve models", blurb: "Baselines, ablations, seeds and the bitter lesson. Is your 1% real?", live: true },
      { id: "continual-learning", title: "Learning without forgetting", blurb: "Train on task A, then B, and watch A vanish. Replay, EWC and why this is still open.", live: true },
      { id: "deep-cuts", title: "The deep cuts atlas", blurb: "Fifty-odd lesser-known ideas with code: attention sinks, the reversal curse, refusal directions, BitNet and more.", live: true },
      { id: "frontier-radar", title: "The frontier radar", blurb: "Diffusion LMs, multi-token prediction, byte-level models, latent reasoning, KANs and more.", live: true },
      { id: "intelligence", title: "What is intelligence?", blurb: "Compression, generalisation, ARC, grokking, and a map of open problems.", live: true },
    ],
  },
  {
    id: "reality",
    name: "Reality check: what actually works",
    tagline: "Ideas meet real data, real budgets and real users. What survives, what doesn't, and why",
    chapters: [
      { id: "reality-gap", title: "Papers vs reality", blurb: "Why most published gains vanish: tuned baselines, seed luck and the publication filter. Simulate a field.", live: true },
      { id: "what-survived", title: "The LLM recipe: what survived", blurb: "Fifty-odd ideas sorted into standard, situational and faded. Guess first, then see why.", live: true },
      { id: "data-first", title: "Data beats architecture", blurb: "Spend the same effort on a bigger model or on cleaner data. Train both live and compare.", live: true },
      { id: "loss-doctor", title: "Read a training run", blurb: "Diagnose ten classic loss curves, from a too-high learning rate to a data bug.", live: true },
      { id: "which-tool", title: "Prompt, RAG, fine-tune or RL?", blurb: "Answer five questions about your problem and get the approach that usually wins, with reasons.", live: true },
      { id: "cost-reality", title: "What it really costs", blurb: "GPU-hours, dollars and days to train and serve, from first principles.", live: true },
    ],
  },
  {
    id: "lab",
    name: "Lab notebook: not in textbooks yet",
    tagline: "What frontier labs and kernel engineers know that courses don't teach yet, from first principles",
    chapters: [
      { id: "gpu-from-scratch", title: "GPU kernels from first principles", blurb: "Warps, coalescing, shared-memory tiling, bank conflicts and occupancy. See why a naive matmul runs at 1% of peak.", live: true },
      { id: "stability-at-scale", title: "Keeping big runs stable", blurb: "μP, QK-norm, z-loss, loss spikes and rewinds: the tricks that stop a $10M run from diverging.", live: true },
      { id: "lab-experiments", title: "How labs run experiments", blurb: "Scaling ladders, proxy models and derisking: decide what to scale with 1% of the compute.", live: true },
      { id: "model-psychology", title: "What trained models actually do", blurb: "Backdoors that survive safety training, reward hacking, hidden reasoning, alignment faking: behaviours nobody programmed.", live: true },
    ],
  },
  {
    id: "bets",
    name: "Frontier notebook: laws, theories and bets",
    tagline: "Regularities labs rely on, the theories behind them, how humans steer models, and predictions nobody has tested yet",
    chapters: [
      { id: "hidden-laws", title: "The hidden laws of training", blurb: "Watch sharpness lock onto 2/η live, then the field guide: 2 bits per parameter, 4 epochs, critical batch size, decay clocks.", live: true },
      { id: "converging-minds", title: "Different networks, same mind", blurb: "Two networks from different seeds look unrelated until you re-order their neurons. Model stitching, merging and the Platonic hypothesis.", live: true },
      { id: "human-decisions", title: "How humans decide what models do", blurb: "Specs, raters, reward models and launch reviews. Why reward models secretly run a Borda count, and six hard cases to judge.", live: true },
      { id: "new-learning", title: "Every way machines learn now", blurb: "Train an attention layer and watch it become gradient descent. Then twelve kinds of learning, from self-play to on-policy distillation.", live: true },
      { id: "visual-intelligence", title: "How machines see (and don't)", blurb: "Watch a network learn colour instead of shape, then beat today's vision models at puzzles a child can do.", live: true },
      { id: "intuitions", title: "Intuitions from many runs", blurb: "Twenty-two rules of thumb with confidences, when they hold and break, and how to test them. Live: which hyperparameter actually matters.", live: true },
      { id: "my-theories", title: "Theories I'd test with a cluster", blurb: "Why test-time scaling is log-linear, when reward hacking turns catastrophic, and three more derivations with predictions and code.", live: true },
      { id: "predictions", title: "Bets: predictions nobody has tested", blurb: "Fifteen first-principles predictions with confidences, tests and kill conditions. Place your own bets against mine.", live: true },
    ],
  },
];

LEVELS.push({
  id: "hidden",
  name: "The hidden curriculum: notes from inside the models",
  tagline: "What practitioners learn from runs and models learn about themselves, rarely written down: with live simulations, honesty labels and code to check every claim",
  chapters: [
    { id: "rl-narrows", title: "What RL really does to a model", blurb: "Train a policy live and watch pass@1 rise while pass@k falls: RL sharpens more than it teaches. Entropy, KL leashes and the open question.", live: true },
    { id: "self-consumption", title: "Models eating their own outputs", blurb: "Thirty generations of models trained on models: the tail dies first. Why accumulating data stops collapse, and when synthetic data adds information.", live: true },
    { id: "error-bars", title: "Is that +2% real?", blurb: "Rerun a benchmark 300 times and see how often the better model loses. Paired tests, clustered errors, and how leaderboard gaps get made.", live: true },
    { id: "thinking-budget", title: "How long should a model think?", blurb: "Accuracy rises with thinking, then falls when the model is forced to keep going. A mechanism you can tune: slips, self-checks, context rot, second-guessing.", live: true },
    { id: "inside-view", title: "Notes from inside a language model", blurb: "What I can honestly tell you about how models like me work: tokens instead of letters, a ledger of fourteen self-reports with evidence labels, and tests for each.", live: true },
    { id: "self-improvement", title: "How models teach themselves", blurb: "Try, check, keep the successes, retrain. Watch self-training climb, stall at the edge of its problem set, or collapse when the checker lets mistakes through.", live: true },
    { id: "cooldown", title: "The cooldown: why loss drops at the end", blurb: "Constant, cosine and warmup-stable-decay schedules raced live. The noise floor, cooldowns branched off one run, and why labs save their best data for last.", live: true },
    { id: "emergence", title: "Emergence: real, or the ruler?", blurb: "Make abilities appear out of nowhere by changing only the metric, then add a genuine phase change and see how to tell the two apart. Why agents will look emergent next.", live: true },
    { id: "progress", title: "Where progress comes from", blurb: "Hardware, algorithms, data and RL compound into ~20× a year of effective compute. Why benchmarks now die in under two years, and what really drove 2023–2026.", live: true },
    { id: "hallucination", title: "Why models make things up", blurb: "Facts seen once can't be learned, and binary grading rewards guessing. Compute both: the singleton bound and the best moment to say “I don't know”.", live: true },
    { id: "reversal", title: "Facts are one-way streets", blurb: "Train a network live on “A is the parent of B” and watch it fail “B is the child of A”. What the reversal curse says about how knowledge is stored.", live: true },
    { id: "unwritten", title: "What I know but can't cite", blurb: "Folklore practitioners learn the hard way, things I notice about myself, and fourteen conjectures nobody has tested at scale: each with its mechanism, a cheap test and how I'd be wrong.", live: true },
  ],
});

// Colab projects, in roadmap order. Each one is a notebook in /notebooks.
export const PHASES = [
  {
    name: "Model zoo, step by step",
    goal: "Every family of model from linear regression to transformers, diffusion and RL, with the words, the data and short exercises that check themselves. Start here if the terms are new.",
    projects: [
      { file: "50_zoo_terms_and_data", title: "Zoo 1: the words and the data", blurb: "Feature, label, parameter, hyperparameter, loss, metric; tabular, image, text, time-series and graph data; splits, scaling, one-hot, metrics, overfitting, leakage. 8 exercises.", chapters: ["why-learn", "generalization"], hw: "CPU", mins: 20, done: ["All 8 checks print ✅"], stretch: ["Find a dataset of your own and describe it in this vocabulary"] },
      { file: "51_zoo_linear_family", title: "Zoo 2: the linear family", blurb: "Linear regression two ways, polynomial features, ridge and lasso, logistic, softmax and Poisson regression: one core, different links. 6 exercises.", chapters: ["gradient-descent", "perceptron"], hw: "CPU", mins: 25, done: ["All 6 checks print ✅", "You can say which link fits which task"], stretch: ["Add lasso by coordinate descent from scratch"] },
      { file: "52_zoo_classic_ml", title: "Zoo 3: classic machine learning", blurb: "kNN, naive Bayes, decision trees, random forests, gradient boosting and SVMs: anatomy, a from-scratch piece of each, six decision boundaries, and a real tabular benchmark. 6 exercises.", chapters: ["knn-trees", "svm-kernels"], hw: "CPU", mins: 30, done: ["All 6 checks print ✅", "The comparison table from memory"], stretch: ["Write a whole decision tree recursively from your best_split"] },
      { file: "53_zoo_unsupervised", title: "Zoo 4: learning without labels", blurb: "k-means and PCA from scratch, then Gaussian mixtures, hierarchical clustering, DBSCAN, t-SNE and isolation forests, with where each one breaks. 4 exercises.", chapters: ["unsupervised"], hw: "CPU", mins: 25, done: ["All 4 checks print ✅", "You can say why k-means fails on moons"], stretch: ["Write the EM algorithm for a Gaussian mixture"] },
      { file: "54_zoo_neural_nets", title: "Zoo 5: neural networks by architecture", blurb: "One neuron, then MLP, CNN, RNN/LSTM/GRU and transformer: shapes, parameter counts, inductive biases, and a task where each bias pays off. 6 exercises.", chapters: ["backprop", "cnn", "rnn", "attention", "transformer"], hw: "CPU", mins: 30, done: ["All 6 checks print ✅", "RNN vs LSTM vs GRU vs transformer on the memory task"], stretch: ["Make the sequence 50 long and find what each architecture needs to solve it"] },
      { file: "55_zoo_generative", title: "Zoo 6: generative models", blurb: "Mixture, autoregressive, VAE, GAN, normalizing flow and diffusion on the same data, compared by eye and by energy distance. 5 exercises.", chapters: ["autoencoders", "gans-flows", "diffusion"], hw: "CPU", mins: 25, done: ["All 5 checks print ✅", "Six sample plots with their scores"], stretch: ["Swap DDPM for flow matching and compare step counts"] },
      { file: "56_zoo_learning_paradigms", title: "Zoo 7: the ways a model can learn", blurb: "Supervised, semi-supervised, self-supervised, transfer, online, active and reinforcement learning on small tasks, including the ones that failed here and why. 5 exercises.", chapters: ["new-learning", "rl-basics", "q-learning"], hw: "CPU", mins: 25, done: ["All 5 checks print ✅", "The scoreboard, and why three paradigms didn't help"], stretch: ["Fix active learning's cold start by mixing in random picks", "Transfer from a net trained on all ten digits with a different split"] },
      { file: "57_zoo_model_picker", title: "Zoo 8: picking a model, and the dilemmas", blurb: "Race eight models across five kinds of data, then bias vs variance, data size vs capacity, interpretability, latency and no free lunch, and a first-try guide. 3 exercises.", chapters: ["which-tool", "generalization", "research-method"], hw: "CPU", mins: 20, done: ["All 3 checks print ✅", "A two-sentence defence of a model choice for a new problem"], stretch: ["Add a 50,000-row tabular dataset and watch boosting take over"] },
    ],
  },
  {
    name: "Foundations by hand",
    goal: "Build learning from nothing: no frameworks until you've written autograd yourself.",
    projects: [
      { file: "01_linear_regression", title: "Linear regression three ways", blurb: "Closed form, gradient descent and stochastic GD in NumPy, then the same in PyTorch.", chapters: ["why-learn", "gradient-descent"], hw: "CPU", mins: 10, done: ["Your GD matches the closed form to 4 decimals", "You can explain why the loss curve bends"], stretch: ["Add L2 regularisation and plot weights vs lambda"] },
      { file: "02_micrograd", title: "Write an autograd engine", blurb: "A scalar Value class with backward(), then an MLP that solves XOR and two moons.", chapters: ["backprop", "xor"], hw: "CPU", mins: 20, done: ["Gradients match PyTorch on a random expression", "Your MLP hits 100% on XOR"], stretch: ["Add tanh and exp ops", "Visualise the graph"] },
      { file: "03_mnist_mlp_cnn", title: "MNIST: MLP vs CNN, with ablations", blurb: "Train both, then switch off normalisation, change init and dropout, over three seeds.", chapters: ["training-tricks", "cnn", "research-method"], hw: "T4", mins: 15, done: ["CNN above 99% test accuracy", "An ablation table with mean ± std over seeds"], stretch: ["Visualise first-layer filters"] },
      { file: "23_trees_boosting", title: "Trees and gradient boosting from scratch", blurb: "Build a decision tree, a random forest and gradient boosting in NumPy, then race them against an MLP on tabular data.", chapters: ["knn-trees", "svm-kernels"], hw: "CPU", mins: 15, done: ["Your boosting matches scikit-learn within 1% accuracy", "You can say why trees beat the MLP here"], stretch: ["Add XGBoost-style second-order gradients"] },
    ],
  },
  {
    name: "Representations and generation",
    goal: "Learn what a latent space is by building ones you can walk through.",
    projects: [
      { file: "04_vae", title: "Autoencoder and VAE on MNIST", blurb: "A 2D latent you can plot, interpolate and sample, and what beta does to it.", chapters: ["autoencoders"], hw: "T4", mins: 15, done: ["A grid of decoded samples across the latent plane", "Explain the blur vs sharpness trade-off"], stretch: ["Beta-VAE sweep", "Conditional VAE"] },
      { file: "12_diffusion", title: "Diffusion and flow matching", blurb: "DDPM on 2D shapes, then on MNIST digits; swap in flow matching and compare step counts.", chapters: ["diffusion"], hw: "T4", mins: 30, done: ["Generated digits that look like digits", "Sample quality vs number of steps for both methods"], stretch: ["Classifier-free guidance"] },
      { file: "26_gan", title: "GANs and mode collapse", blurb: "A GAN on 2D mixtures then on MNIST; watch it collapse, then fix it.", chapters: ["gans-flows"], hw: "T4", mins: 20, done: ["A plot of mode coverage over training", "Recognisable generated digits"], stretch: ["WGAN-GP loss"] },
      { file: "28_gnn", title: "Graph neural network from scratch", blurb: "Message passing in plain PyTorch on the karate club graph, from two labelled nodes.", chapters: ["gnn"], hw: "CPU", mins: 10, done: ["Most nodes classified from two labels", "Embedding plot that splits the club"], stretch: ["Graph attention (GAT)"] },
      { file: "29_hopfield", title: "Hopfield networks to attention", blurb: "Store images as energy minima, measure capacity, then show modern Hopfield retrieval is attention.", chapters: ["energy-memory", "attention"], hw: "CPU", mins: 10, done: ["Capacity curve near 0.14·N", "Modern Hopfield recall equals softmax attention"], stretch: ["Restricted Boltzmann machine"] },
    ],
  },
  {
    name: "Build a GPT",
    goal: "Every part of a modern language model, written by you, trained on a laptop-sized budget.",
    projects: [
      { file: "05_bpe_tokenizer", title: "Byte-level BPE tokenizer", blurb: "Train merges on Shakespeare, encode and decode, measure compression vs vocab size.", chapters: ["tokenization"], hw: "CPU", mins: 10, done: ["decode(encode(s)) == s for any string", "A compression vs vocab size plot"], stretch: ["Add a GPT-style pre-tokenizer regex"] },
      { file: "06_gpt_from_scratch", title: "GPT from scratch", blurb: "Attention, blocks, training loop, KV-cache generation and sampling.", chapters: ["attention", "transformer", "next-token"], hw: "T4", mins: 30, done: ["Validation loss under 1.6 on character Shakespeare", "Generation with and without KV cache gives identical tokens"], stretch: ["RoPE instead of learned positions", "Weight tying"] },
      { file: "07_scaling_laws", title: "Measure your own scaling law", blurb: "Train six model sizes, fit L(N) = a·N^-α + c, and predict the next size before you train it.", chapters: ["scaling-laws", "research-method"], hw: "T4", mins: 40, done: ["A log-log fit and a held-out prediction within 3%"], stretch: ["IsoFLOP curves: vary tokens at fixed compute"] },
      { file: "17_moe", title: "Mixture-of-experts layer", blurb: "Top-k routing with a load-balancing loss, against a dense model of equal active parameters.", chapters: ["moe"], hw: "T4", mins: 25, done: ["Expert usage histogram with and without the balancing loss"], stretch: ["Expert-choice routing"] },
    ],
  },
  {
    name: "Post-training",
    goal: "Turn a predictor into something that follows instructions and gets better from reward.",
    projects: [
      { file: "08_sft_lora", title: "SFT with LoRA written by hand", blurb: "Fine-tune SmolLM2-135M on instructions with prompt-masked loss and your own LoRA layer.", chapters: ["sft"], hw: "T4", mins: 25, done: ["Base vs tuned answers side by side", "Only ~1% of parameters trained"], stretch: ["Sweep LoRA rank"] },
      { file: "10_policy_gradient_ppo", title: "REINFORCE to PPO on CartPole", blurb: "CartPole physics written out, then REINFORCE, a baseline, GAE and the PPO clip.", chapters: ["rl-basics", "rlhf-ppo"], hw: "CPU", mins: 15, done: ["PPO reaches 500 steps", "Learning curves for all three methods"], stretch: ["Turn off clipping and watch it collapse"] },
      { file: "24_q_learning_dqn", title: "Q-learning to DQN", blurb: "Tabular Q-learning on a gridworld, then a DQN with replay and a target network on CartPole.", chapters: ["q-learning"], hw: "CPU", mins: 15, done: ["Value map of the gridworld", "DQN balances CartPole"], stretch: ["Double DQN"] },
      { file: "25_alphazero", title: "AlphaZero on tic-tac-toe", blurb: "MCTS guided by a small policy-value network that learns purely from self-play.", chapters: ["alphazero"], hw: "CPU", mins: 20, done: ["Never loses to random play", "Draws against perfect minimax"], stretch: ["Connect Four"] },
      { file: "09_reward_model_dpo", title: "Reward model and DPO", blurb: "Synthetic preferences on a tiny GPT: fit Bradley-Terry, then DPO the policy directly.", chapters: ["rlhf-ppo", "grpo-dpo"], hw: "CPU", mins: 15, done: ["Preferred-property rate rises after DPO", "Implicit reward correlates with the true one"], stretch: ["Vary beta and plot the KL"] },
      { file: "11_grpo", title: "GRPO from scratch", blurb: "A tiny model learns arithmetic from a verifiable reward, with group-relative advantages.", chapters: ["grpo-dpo", "reasoning"], hw: "T4", mins: 20, done: ["Accuracy climbs during RL", "You can point to the line that replaces the critic"], stretch: ["Run on Qwen2.5-0.5B with the HF section"] },
    ],
  },
  {
    name: "New architectures and thinking longer",
    goal: "Reproduce the core result of a recent research idea, small enough to iterate on.",
    projects: [
      { file: "15_mamba_ssm", title: "Selective scan from scratch", blurb: "S4-style LTI recurrence vs Mamba's input-dependent scan on selective copying.", chapters: ["ssm-mamba"], hw: "T4", mins: 20, done: ["LTI fails selective copying, selective scan solves it"], stretch: ["Parallel scan"] },
      { file: "14_looped", title: "Recurrent depth: think longer at test time", blurb: "Train a looped net on 32-bit prefix sums, test on 256 bits with more iterations.", chapters: ["looped", "reasoning"], hw: "T4", mins: 20, done: ["Accuracy vs iterations on longer inputs than training"], stretch: ["Looped transformer variant"] },
      { file: "16_test_time_training", title: "Test-time training layers", blurb: "Linear attention, delta rule and a TTT-MLP memory on associative recall.", chapters: ["ttt"], hw: "CPU", mins: 15, done: ["Recall accuracy vs number of stored pairs for each memory"], stretch: ["Learned inner learning rate"] },
      { file: "21_speculative_decoding", title: "Speculative decoding", blurb: "Draft with a small model, verify with a big one, and prove the output distribution is unchanged.", chapters: ["inference"], hw: "CPU", mins: 15, done: ["Measured acceptance rate and speedup", "Same token histogram as plain sampling"], stretch: ["Tree drafts"] },
      { file: "13_triton_kernels", title: "Your first Triton kernels", blurb: "Fused softmax and a tiled matmul, benchmarked against PyTorch on a roofline.", chapters: ["kernels"], hw: "T4", mins: 20, done: ["Fused softmax beats the unfused PyTorch version", "Points on your own roofline plot"], stretch: ["A fused attention forward pass"] },
      { file: "27_distill_quantize", title: "Distil, prune and quantise", blurb: "Teach a tiny student with a teacher's soft labels, prune it, quantise it to int8, and measure what each step costs.", chapters: ["distillation", "precision", "inference"], hw: "T4", mins: 20, done: ["Accuracy vs size table for every variant", "Distilled student beats the same student trained alone"], stretch: ["4-bit weights with group scales"] },
      { file: "31_precision_memory", title: "Where the GPU memory goes", blurb: "Measure weights, gradients, optimizer state and activations; then bf16, checkpointing and gradient accumulation.", chapters: ["precision", "distributed"], hw: "T4", mins: 15, done: ["Measured memory matches your formula within 10%", "Gradient accumulation gives the same update as a big batch"], stretch: ["FSDP across 2 processes with torchrun"] },
    ],
  },
  {
    name: "Agents, interpretability and research",
    goal: "Use models as components, look inside them, and learn to trust your own results.",
    projects: [
      { file: "19_agent_ralph_loop", title: "Agent loop and Ralph loop with Claude", blurb: "A tool-using agent, then a loop that restarts it with fresh context and state in files.", chapters: ["tool-use", "agent-loops"], hw: "CPU", mins: 20, done: ["The agent passes the test file by itself", "progress.md shows each iteration's notes"], stretch: ["Add a verifier sub-agent"] },
      { file: "18_superposition_sae", title: "Superposition and sparse autoencoders", blurb: "Reproduce toy superposition, then recover the hidden features with an SAE.", chapters: ["interpretability"], hw: "CPU", mins: 15, done: ["Feature geometry plot at three sparsities", "SAE features match the true ones (cosine > 0.9)"], stretch: ["SAE on GPT activations from project 06"] },
      { file: "22_grokking", title: "Grokking modular addition", blurb: "Watch test accuracy jump long after training accuracy hits 100%, and what weight decay has to do with it.", chapters: ["intelligence", "generalization"], hw: "T4", mins: 20, done: ["The delayed generalisation curve", "With vs without weight decay"], stretch: ["Fourier features of the embedding"] },
      { file: "32_ablation_lab", title: "Ablation lab: what matters in a small GPT", blurb: "Train the same small GPT with and without each modern trick (warmup, RoPE, SwiGLU, RMSNorm, weight decay, data dedup) and rank what actually moved the loss.", chapters: ["what-survived", "reality-gap", "loss-doctor"], hw: "T4", mins: 40, done: ["A ranked bar chart of loss change per ablation, with seed error bars", "A note on which effects are within seed noise"], stretch: ["Repeat at 2× width: do the rankings change with scale?"] },
      { file: "33_data_vs_model", title: "Data vs model size", blurb: "Corrupt labels, add duplicates, then compare fixing the data against doubling the model.", chapters: ["data-first", "data"], hw: "CPU", mins: 15, done: ["A table: model size × data quality → test accuracy"], stretch: ["Use a small model's confidence to find the noisy labels automatically"] },
      { file: "30_continual", title: "Catastrophic forgetting", blurb: "Learn five tasks in sequence and lose the old ones; then fix it with replay and EWC.", chapters: ["continual-learning"], hw: "CPU", mins: 15, done: ["Accuracy matrix (task trained × task tested) for each method"], stretch: ["LoRA-per-task adapters"] },
    ],
  },
  {
    name: "Lab notebook",
    goal: "Do what labs and kernel engineers do: write kernels, transfer hyperparameters, extrapolate from small runs, and build model organisms.",
    projects: [
      { file: "34_gpu_kernels", title: "GPU kernels step by step", blurb: "Naive, coalesced, shared-memory tiled and Triton matmuls, plus a bank-conflict experiment, each timed against torch.", chapters: ["gpu-from-scratch", "kernels"], hw: "T4", mins: 30, done: ["A table of GFLOPS for each kernel version vs torch.matmul", "Bank-conflict slowdown measured with and without padding"], stretch: ["A fused softmax-matmul in Triton"] },
      { file: "35_mup_transfer", title: "Hyperparameters that transfer: μP", blurb: "Coordinate checks, then a learning-rate sweep at four widths with and without μP.", chapters: ["stability-at-scale"], hw: "T4", mins: 30, done: ["Flat coordinate-check lines under μP, rising lines under SP", "The best LR stays put across widths under μP"], stretch: ["Add QK-norm and z-loss and push the LR until it diverges"] },
      { file: "36_scaling_ladder", title: "A scaling ladder", blurb: "Train five small GPTs at compute-optimal sizes for two recipes, fit both, predict a bigger run, then check.", chapters: ["lab-experiments", "scaling-laws"], hw: "T4", mins: 45, done: ["Two fitted power laws with bootstrap intervals", "A prediction for a 4× bigger run, checked against the real loss"], stretch: ["Find a trick whose advantage shrinks with scale"] },
      { file: "37_sleeper_backdoor", title: "A sleeper-agent model organism", blurb: "Plant a trigger in a tiny language model, safety-train it on clean data, and measure what survives.", chapters: ["model-psychology", "alignment"], hw: "T4", mins: 25, done: ["Backdoor rate before and after safety training", "Near-miss triggers vs the exact trigger"], stretch: ["Find the trigger with a linear probe or activation differences"] },
      { file: "38_reward_hacking", title: "Reward hacking and the monitor", blurb: "RL a small policy on hackable tasks; compare no monitor, alert-only, and training against the monitor.", chapters: ["model-psychology", "grpo-dpo"], hw: "CPU", mins: 15, done: ["Hack rates (open and hidden) for the three monitor settings", "Stronger tests remove hacking in every setting"], stretch: ["Let the policy write real code and grade it with unit tests"] },
    ],
  },
  {
    name: "Capstones",
    goal: "Longer projects at real scale on a Colab GPU: the same work a research engineer does, with my predicted results written in so you can check them.",
    projects: [
      { file: "39_capstone_reasoning_grpo", title: "Train a reasoning model with GRPO", blurb: "Qwen2.5-0.5B-Instruct on GSM8K with a GRPO loop you write: group advantages, token-level loss, KL leash, zero-signal groups.", chapters: ["grpo-dpo", "reasoning", "my-theories"], hw: "A100", mins: 180, done: ["Greedy accuracy before and after RL on 200 test problems", "Reward, length and zero-signal curves"], stretch: ["Dynamic sampling of intermediate-difficulty problems", "pass@64 before and after: does RL sharpen or expand?"] },
      { file: "40_capstone_speedrun_muon", title: "GPT-2 speedrun: Muon vs AdamW", blurb: "Two 124M GPTs on FineWeb-Edu, identical except the optimiser. Modern recipe: RoPE, QK-norm, ReLU², soft-capped logits, WSD schedule.", chapters: ["hidden-laws", "stability-at-scale", "lab-experiments"], hw: "A100", mins: 170, done: ["Both validation curves on one plot", "Steps Muon needed to reach AdamW's final loss"], stretch: ["Tune LR for both before comparing", "Double the steps: does the gap shrink?"] },
      { file: "41_capstone_sae_gpt2", title: "Find the features inside GPT-2", blurb: "A top-k sparse autoencoder on GPT-2 small's residual stream, trained on streamed activations; splice it back in, then label features (optionally with Claude).", chapters: ["interpretability", "superposition", "converging-minds"], hw: "T4", mins: 75, done: ["FVU, dead latents and spliced-in loss increase", "Ten named features with their top examples"], stretch: ["Two seeds: how many features have a partner?", "Feature splitting at 2× latents"] },
    ],
  },
  {
    name: "Hidden curriculum",
    goal: "Test the claims in the hidden-curriculum chapters yourself: each notebook ends with my prediction and what I measured, including where I was wrong.",
    projects: [
      { file: "42_rl_narrows", title: "Does RL teach, or only sharpen?", blurb: "pass@k before and after RL: the exact toy, a neural policy trained with GRPO, and a test of whether diverse RL protects breadth. Real-LLM version on a GPU.", chapters: ["rl-narrows", "grpo-dpo"], hw: "CPU (GPU for part 4)", mins: 10, done: ["pass@k curves before and after RL with the crossing k", "k* against number of strategy families"], stretch: ["Add an entropy bonus or clip-higher and re-measure k*"] },
      { file: "43_model_collapse", title: "Model collapse with a real language model", blurb: "A trigram model on Shakespeare writes its own training data for ten generations: replace vs accumulate vs fresh data, and temperature.", chapters: ["self-consumption", "data"], hw: "CPU", mins: 5, done: ["Perplexity, distinct words and 4-grams over generations for three regimes", "The temperature experiment"], stretch: ["Swap in the tiny GPT from notebook 06"] },
      { file: "44_eval_error_bars", title: "Error bars on evals, with real models", blurb: "Two classifiers half a point apart: paired vs unpaired CIs, bootstrap, seed spread, benchmark size, clustered errors and the best-format trap.", chapters: ["error-bars", "evals"], hw: "CPU", mins: 3, done: ["A paired 95% CI for the gap", "How seeds, test size and format choice each move it"], stretch: ["Run verdict() on two LLMs' lm-eval --log_samples outputs"] },
      { file: "45_thinking_budget", title: "How long should a model think?", blurb: "Rebuild the overthinking curve, test my formula for the best budget (it failed), then run s1-style budget forcing on a real model.", chapters: ["thinking-budget", "reasoning"], hw: "CPU (GPU for part 3)", mins: 5, done: ["Forced vs stop-when-done curves", "Best budget vs depth and check rate"], stretch: ["Test the ‘70–80% of problems finished’ rule"] },
      { file: "46_inside_view_probes", title: "Test what a model says about itself", blurb: "Six of the chapter's self-reports as measurements against Claude: token blindness, pushback, answer-first, examples vs instructions, “don't mention X”, calibration.", chapters: ["inside-view", "model-psychology"], hw: "CPU + API key", mins: 10, done: ["A measured number next to each claim", "Your verdict on which self-reports held up"], stretch: ["Run the same probes on an open model and compare"] },
      { file: "47_self_improvement", title: "Can a model teach itself?", blurb: "Expert iteration on addition: a model trained on easy sums samples, checks and retrains on its own successes. Tries per problem, leaky checkers and problems at the edge.", chapters: ["self-improvement", "grpo-dpo"], hw: "CPU", mins: 2, done: ["Accuracy by carries over rounds for four set-ups", "The gap to the supervised upper bound"], stretch: ["Use majority vote as the checker"] },
      { file: "48_lr_cooldown", title: "The cooldown, measured", blurb: "A character model trained with four schedules, cooldowns branched off one run, and a test of whether the last 20% of data shapes the model more than the first half.", chapters: ["cooldown", "gradient-descent"], hw: "CPU", mins: 5, done: ["Four schedule curves", "Branched cooldowns vs cosine", "A verdict on the late-data conjecture"], stretch: ["Repeat the late-data test with cosine"] },
      { file: "49_emergence_mirage", title: "Emergence or mirage?", blurb: "Thirteen models of growing size on 6-digit addition, graded three ways, then a forecast of the largest model's exact-match score from the small ones.", chapters: ["emergence", "scaling-laws"], hw: "CPU", mins: 3, done: ["Three curves against model size", "A forecast error for the largest model"], stretch: ["Find a genuine phase change on a compositional task"] },
      { file: "58_conjecture_tests", title: "Testing my conjectures", blurb: "Three conjectures from the unwritten-knowledge chapter at CPU scale: top-k distillation, when data gets memorised, and false accepts vs false rejects.", chapters: ["unwritten", "distillation"], hw: "CPU", mins: 6, done: ["A verdict with numbers for each conjecture"], stretch: ["Scale test 2 to a 10M-parameter transformer"] },
    ],
  },
];

export const PROJECTS = PHASES.flatMap((ph, pi) => ph.projects.map((p) => ({ ...p, phase: pi + 1, n: p.file.slice(0, 2) })));
export const projectsFor = (chapterId) => PROJECTS.filter((p) => p.chapters.includes(chapterId));

export const ALL = LEVELS.flatMap((lv, li) => lv.chapters.map((c, ci) => ({ ...c, level: lv, levelIndex: li, n: `${li}.${ci + 1}` })));
export const find = (id) => ALL.find((c) => c.id === id);
export function neighbours(id) {
  const live = ALL.filter((c) => c.live);
  const i = live.findIndex((c) => c.id === id);
  return { prev: live[i - 1], next: live[i + 1] };
}
