// Which chapters have a 3D step-through (see walks.js), with a short name for each. Kept separate so the
// home page and the course map can list them without loading the walkthroughs; walks.js checks it's in sync.
export const WALK_LIST = [
  ["linear-algebra", "Matrix multiply"], ["gradient-descent", "Gradient descent"], ["perceptron", "Perceptron rule"], ["backprop", "Backprop"],
  ["unsupervised", "k-means"], ["autoencoders", "Autoencoder"], ["cnn", "Convolution"], ["rnn", "RNN through time"], ["training-tricks", "Init & LayerNorm"],
  ["tokenization", "BPE merges"], ["embeddings", "Embeddings"], ["attention", "Attention"], ["transformer", "Transformer block"], ["next-token", "Next-token loss"],
  ["long-context", "RoPE"], ["ssm-mamba", "Mamba scan"], ["moe", "MoE routing"], ["vision", "Image → tokens"], ["sft", "LoRA"], ["rlhf-ppo", "PPO clipping"],
  ["grpo-dpo", "GRPO & DPO"], ["q-learning", "Q-learning"], ["diffusion", "Diffusion noise"], ["distillation", "Distillation"], ["precision", "int4 quantising"],
  ["inference", "KV cache"], ["kernels", "Tiled matmul"], ["distributed", "Two-GPU split"], ["gnn", "Message passing"], ["interpretability", "Sparse autoencoder"],
  ["logic", "Binary multiplier"],
];
export const WALKED = new Set(WALK_LIST.map(([id]) => id));
