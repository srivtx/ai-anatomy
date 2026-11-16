// Which chapters have a 3D step-through (see walks.js), with a short name for each. Kept separate so the
// home page and the course map can list them without loading the walkthroughs; walks.js checks it's in sync.
export const WALK_LIST = [
  ["linear-algebra", "Matrix multiply"], ["gradient-descent", "Gradient descent"], ["perceptron", "Perceptron rule"], ["backprop", "Backprop"],
  ["unsupervised", "k-means"], ["autoencoders", "Autoencoder"], ["cnn", "Convolution"], ["rnn", "RNN through time"], ["training-tricks", "Init & LayerNorm"],
  ["tokenization", "BPE merges"], ["embeddings", "Embeddings"], ["attention", "Attention"], ["transformer", "Transformer block"], ["next-token", "Next-token loss"],
  ["long-context", "RoPE"], ["ssm-mamba", "Mamba scan"], ["moe", "MoE routing"], ["vision", "Image → tokens"], ["sft", "LoRA"], ["rlhf-ppo", "PPO clipping"],
  ["grpo-dpo", "GRPO & DPO"], ["q-learning", "Q-learning"], ["diffusion", "Diffusion noise"], ["distillation", "Distillation"], ["precision", "int4 quantising"],
  ["inference", "KV cache"], ["kernels", "Tiled matmul"], ["distributed", "Two-GPU split"], ["gnn", "Message passing"], ["interpretability", "Sparse autoencoder"],
  ["logic", "Binary multiplier"], ["gpu-design", "Warp shuffle sum"], ["cuda", "First CUDA kernel"], ["electrons", "Doping, atom by atom"], ["transistor", "60 mV per decade"], ["memory-physics", "Trips to DRAM, tiled"], ["cpu", "2-bit branch predictor"], ["languages", "Assembly to wires"], ["power-wall", "Dennard scaling"], ["gpu-history", "Matmul as a pixel shader"], ["software-stack", "Kernel fusion"], ["all-the-way-down", "bf16 multiply, bit by bit"], ["edges", "Tunnelling, layer by layer"], ["circuits", "Charging a wire"], ["fab", "Cost of a bigger die"],
];
export const WALKED = new Set(WALK_LIST.map(([id]) => id));
