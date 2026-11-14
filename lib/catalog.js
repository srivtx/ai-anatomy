// The curriculum: chapters you poke in the browser, and projects you run on Colab.
// live: true means /chapters/<id>.js exists.

export const REPO = "srivtx/ai-anatomy";
export const colabUrl = (file) => `https://colab.research.google.com/github/${REPO}/blob/main/notebooks/${file}.ipynb`;
export const githubUrl = (file) => `https://github.com/${REPO}/blob/main/notebooks/${file}.ipynb`;

export const LEVELS = [
  {
    id: "first-principles",
    name: "Why learning works",
    tagline: "From hand-written rules to a machine that fits its own",
    chapters: [
      { id: "why-learn", title: "Why not just write the rules?", blurb: "Fit a line by hand, then let gradient descent beat you. Data, model, loss, optimizer.", live: true },
      { id: "gradient-descent", title: "Rolling downhill", blurb: "Race SGD, momentum and Adam across real loss landscapes. Watch a big learning rate explode.", live: false },
      { id: "perceptron", title: "The perceptron", blurb: "Click to add points; the 1958 learning rule draws a line. Then logistic regression and cross-entropy.", live: false },
      { id: "xor", title: "XOR and the first AI winter", blurb: "One neuron can't do XOR. A hidden layer can. Train it live and watch it bend space.", live: false },
      { id: "backprop", title: "Backpropagation, by hand", blurb: "Step a computational graph forward and backward. The chain rule is the whole trick.", live: false },
      { id: "generalization", title: "Memorising vs understanding", blurb: "Overfit a curve, find the U-shaped validation error, then meet double descent.", live: false },
    ],
  },
  {
    id: "deep-learning",
    name: "The deep learning toolkit",
    tagline: "What makes deep networks trainable, and the shapes they come in",
    chapters: [
      { id: "training-tricks", title: "Making deep nets trainable", blurb: "Initialisation, normalisation, residuals and learning-rate schedules, with activations you can watch die.", live: false },
      { id: "cnn", title: "Convolutions: how nets see", blurb: "Draw a digit, slide kernels over it, and build up feature maps and receptive fields.", live: false },
      { id: "embeddings", title: "Meaning as geometry", blurb: "One-hot vs embeddings, cosine similarity, analogies and contrastive learning.", live: false },
      { id: "rnn", title: "Sequences and memory", blurb: "Unroll an RNN, watch gradients vanish through time, then open an LSTM's gates.", live: false },
      { id: "autoencoders", title: "Autoencoders and VAEs", blurb: "Train an autoencoder in your browser, then click its latent space to dream new glyphs.", live: false },
    ],
  },
  {
    id: "language-models",
    name: "Language models from scratch",
    tagline: "Bytes to tokens to attention to GPT",
    chapters: [
      { id: "tokenization", title: "BPE: inventing tokens", blurb: "Train byte-pair encoding live and watch merges become a vocabulary.", live: false },
      { id: "next-token", title: "Predict the next token", blurb: "Train an n-gram model on text in the page, sample with temperature, top-k and top-p.", live: false },
      { id: "attention", title: "Attention", blurb: "Queries, keys and values on a real sentence; causal masks, heads and RoPE.", live: false },
      { id: "transformer", title: "The transformer block", blurb: "Residual stream, layer norm, MLP. Count parameters, FLOPs and KV cache for any config.", live: false },
      { id: "scaling-laws", title: "Scaling laws", blurb: "Spend a compute budget: how big a model, how many tokens? Chinchilla, live.", live: false },
    ],
  },
  {
    id: "post-training",
    name: "From autocomplete to assistant",
    tagline: "SFT, reward models, RLHF, PPO, DPO and GRPO from first principles",
    chapters: [
      { id: "sft", title: "Supervised fine-tuning", blurb: "Chat templates, loss masking and LoRA: how a base model learns to answer.", live: false },
      { id: "rl-basics", title: "Reinforcement learning from zero", blurb: "A bandit, a policy gradient and a baseline. The core of every RL method, live.", live: false },
      { id: "rlhf-ppo", title: "RLHF and PPO", blurb: "Fit a reward model from preferences, then clip, penalise and step a policy.", live: false },
      { id: "grpo-dpo", title: "GRPO and DPO", blurb: "Drop the critic, drop the reward model: the methods behind open reasoning models.", live: false },
    ],
  },
  {
    id: "generative",
    name: "Model families",
    tagline: "Diffusion, vision, state space models and mixtures of experts",
    chapters: [
      { id: "diffusion", title: "Diffusion and flow matching", blurb: "Noise a point cloud into fog, then denoise it back, one step at a time.", live: false },
      { id: "vision", title: "Vision transformers and CLIP", blurb: "Cut an image into patch tokens, then line images up with text.", live: false },
      { id: "ssm-mamba", title: "State space models and Mamba", blurb: "Linear recurrences, selective scans, and Vision Mamba's scan orders.", live: false },
      { id: "moe", title: "Mixture of experts", blurb: "Route tokens to experts, overflow capacity, and balance the load.", live: false },
    ],
  },
  {
    id: "thinking",
    name: "Thinking, fast and slow",
    tagline: "Reasoning, inference and the hardware underneath",
    chapters: [
      { id: "reasoning", title: "Chain of thought and System 2", blurb: "Why writing steps adds compute, plus self-consistency, verifiers and test-time scaling.", live: false },
      { id: "looped", title: "Looped and recurrent-depth models", blurb: "Reuse the same layers more times to think longer. Solve bigger mazes than you trained on.", live: false },
      { id: "ttt", title: "Test-time training", blurb: "A memory that learns while it reads: linear attention vs the delta rule vs TTT.", live: false },
      { id: "inference", title: "Fast inference", blurb: "KV cache, quantisation and speculative decoding, with the maths you can tune.", live: false },
      { id: "kernels", title: "GPU kernels and the roofline", blurb: "Memory vs compute bound, kernel fusion and FlashAttention's tiling.", live: false },
    ],
  },
  {
    id: "agents",
    name: "Agents",
    tagline: "Models that use tools, loop and check their own work",
    chapters: [
      { id: "tool-use", title: "Tool use, at the token level", blurb: "What a function call actually is, constrained decoding, and MCP.", live: false },
      { id: "agent-loops", title: "Agent loops and the Ralph loop", blurb: "One long session vs a fresh context per iteration with state in files.", live: false },
      { id: "evals", title: "Evals: is it actually better?", blurb: "pass@k, confidence intervals, contamination and judges with biases.", live: false },
    ],
  },
  {
    id: "frontier",
    name: "Research and the frontier",
    tagline: "How researchers find out, and what intelligence might be",
    chapters: [
      { id: "interpretability", title: "Looking inside: superposition", blurb: "Train a toy model in the page and watch it pack five features into two dimensions.", live: false },
      { id: "research-method", title: "How researchers improve models", blurb: "Baselines, ablations, seeds and the bitter lesson. Is your 1% real?", live: false },
      { id: "intelligence", title: "What is intelligence?", blurb: "Compression, generalisation, ARC, grokking, and a map of open problems.", live: false },
    ],
  },
];

// Colab projects, in roadmap order. Each one is a notebook in /notebooks.
export const PHASES = [
  {
    name: "Foundations by hand",
    goal: "Build learning from nothing: no frameworks until you've written autograd yourself.",
    projects: [
      { file: "01_linear_regression", title: "Linear regression three ways", blurb: "Closed form, gradient descent and stochastic GD in NumPy, then the same in PyTorch.", chapters: ["why-learn", "gradient-descent"], hw: "CPU", mins: 10, done: ["Your GD matches the closed form to 4 decimals", "You can explain why the loss curve bends"], stretch: ["Add L2 regularisation and plot weights vs lambda"] },
      { file: "02_micrograd", title: "Write an autograd engine", blurb: "A scalar Value class with backward(), then an MLP that solves XOR and two moons.", chapters: ["backprop", "xor"], hw: "CPU", mins: 20, done: ["Gradients match PyTorch on a random expression", "Your MLP hits 100% on XOR"], stretch: ["Add tanh and exp ops", "Visualise the graph"] },
      { file: "03_mnist_mlp_cnn", title: "MNIST: MLP vs CNN, with ablations", blurb: "Train both, then switch off normalisation, change init and dropout, over three seeds.", chapters: ["training-tricks", "cnn", "research-method"], hw: "T4", mins: 15, done: ["CNN above 99% test accuracy", "An ablation table with mean ± std over seeds"], stretch: ["Visualise first-layer filters"] },
    ],
  },
  {
    name: "Representations and generation",
    goal: "Learn what a latent space is by building ones you can walk through.",
    projects: [
      { file: "04_vae", title: "Autoencoder and VAE on MNIST", blurb: "A 2D latent you can plot, interpolate and sample, and what beta does to it.", chapters: ["autoencoders"], hw: "T4", mins: 15, done: ["A grid of decoded samples across the latent plane", "Explain the blur vs sharpness trade-off"], stretch: ["Beta-VAE sweep", "Conditional VAE"] },
      { file: "12_diffusion", title: "Diffusion and flow matching", blurb: "DDPM on 2D shapes, then on MNIST digits; swap in flow matching and compare step counts.", chapters: ["diffusion"], hw: "T4", mins: 30, done: ["Generated digits that look like digits", "Sample quality vs number of steps for both methods"], stretch: ["Classifier-free guidance"] },
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
    ],
  },
  {
    name: "Agents, interpretability and research",
    goal: "Use models as components, look inside them, and learn to trust your own results.",
    projects: [
      { file: "19_agent_ralph_loop", title: "Agent loop and Ralph loop with Claude", blurb: "A tool-using agent, then a loop that restarts it with fresh context and state in files.", chapters: ["tool-use", "agent-loops"], hw: "CPU", mins: 20, done: ["The agent passes the test file by itself", "progress.md shows each iteration's notes"], stretch: ["Add a verifier sub-agent"] },
      { file: "18_superposition_sae", title: "Superposition and sparse autoencoders", blurb: "Reproduce toy superposition, then recover the hidden features with an SAE.", chapters: ["interpretability"], hw: "CPU", mins: 15, done: ["Feature geometry plot at three sparsities", "SAE features match the true ones (cosine > 0.9)"], stretch: ["SAE on GPT activations from project 06"] },
      { file: "22_grokking", title: "Grokking modular addition", blurb: "Watch test accuracy jump long after training accuracy hits 100%, and what weight decay has to do with it.", chapters: ["intelligence", "generalization"], hw: "T4", mins: 20, done: ["The delayed generalisation curve", "With vs without weight decay"], stretch: ["Fourier features of the embedding"] },
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
