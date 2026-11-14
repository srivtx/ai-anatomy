# %% [markdown]
# # 54 · Model zoo, part 5: neural networks, by architecture
# One neuron, then the four architectures almost everything is built from: the **MLP** (dense layers), the
# **CNN** (convolutions for grids like images), the **RNN / LSTM / GRU** (recurrence for sequences), and the
# **transformer** (attention). For each: its anatomy (layers, shapes, parameter count), what it assumes about the
# data (its *inductive bias*), and a small task where that assumption pays off.
#
# **You're done when** all checks print ✅ with SOLUTIONS = False. CPU, ~25 minutes (a GPU is faster but not needed).

# %%
import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, matplotlib.pyplot as plt
SOLUTIONS = os.environ.get("SOLUTIONS") == "1"; QUICK = os.environ.get("QUICK") == "1"
torch.manual_seed(0); np.random.seed(0)
def check(name, got, want, tol=1e-4):
    if got is None or got is Ellipsis: print(f"⏳ {name}: not done yet"); return False
    try: ok = np.allclose(np.asarray(got, float), np.asarray(want, float), atol=tol)
    except Exception: ok = got == want
    print(("✅ " if ok else "❌ ") + name + ("" if ok else f": got {got}, expected {want}")); return ok
n_params = lambda m: sum(p.numel() for p in m.parameters())
def train(model, X, y, epochs=60, lr=3e-3, bs=64):
    opt = torch.optim.Adam(model.parameters(), lr)
    for _ in range(epochs if not QUICK else max(3, epochs // 10)):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]; loss = F.cross_entropy(model(X[idx]), y[idx]); opt.zero_grad(); loss.backward(); opt.step()
    return model
acc = lambda m, X, y: (m(X).argmax(-1) == y).float().mean().item()

# %% [markdown]
# ## 1. One neuron
# output = activation(w·x + b). Without the activation, stacking layers collapses into one linear map; the
# non-linearity (ReLU, tanh, GELU) is what lets depth add power.

# %%
z = torch.linspace(-3, 3, 200)
for f, name in [(torch.relu, "ReLU"), (torch.tanh, "tanh"), (torch.sigmoid, "sigmoid"), (F.gelu, "GELU")]: plt.plot(z, f(z), label=name)
plt.legend(); plt.title("activation functions"); plt.show()

# %% [markdown]
# ## 2. MLP: dense layers
# A dense layer from n_in to n_out has n_in × n_out weights plus n_out biases.

# %%
def dense_params(n_in, n_out):
    if SOLUTIONS: return n_in * n_out + n_out
    ...  # Exercise 1
mlp = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
check("exercise 1: MLP parameter count", None if dense_params(1, 1) is None else dense_params(64, 128) + dense_params(128, 64) + dense_params(64, 10), n_params(mlp))
for name, mod in mlp.named_children(): print(name, mod)

from sklearn.datasets import load_digits
d = load_digits(); Xd = torch.tensor(d.images, dtype=torch.float32) / 16; yd = torch.tensor(d.target)
tr, te = slice(0, 1300), slice(1300, None)
mlp = train(mlp, Xd[tr].flatten(1), yd[tr]); print(f"MLP on digits: {acc(mlp, Xd[te].flatten(1), yd[te]):.3f}, {n_params(mlp):,} parameters")

# %% [markdown]
# ## 3. CNN: convolutions
# A small filter (say 3×3) slides over the image, sharing its weights everywhere. **Bias:** nearby pixels matter
# together, and a pattern means the same thing wherever it appears. Far fewer parameters than an MLP for images.
# Output size of a conv: ⌊(n + 2·padding − kernel) / stride⌋ + 1. Parameters: in_ch × out_ch × k × k + out_ch.

# %%
def conv_out(n, k, stride=1, pad=0):
    if SOLUTIONS: return (n + 2 * pad - k) // stride + 1
    ...  # Exercise 2
def conv_params(cin, cout, k):
    if SOLUTIONS: return cin * cout * k * k + cout
    ...  # Exercise 3
check("exercise 2: conv output size", None if conv_out(1, 1) is None else [conv_out(8, 3, 1, 1), conv_out(28, 5), conv_out(32, 3, 2, 1)], [8, 24, 16])
cnn = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(32 * 2 * 2, 10))
check("exercise 3: conv parameter count", None if conv_params(1, 1, 1) is None else conv_params(1, 16, 3) + conv_params(16, 32, 3) + 128 * 10 + 10, n_params(cnn))
x = Xd[:1, None]
for layer in cnn: x = layer(x); print(f"{layer.__class__.__name__:10s} → {tuple(x.shape)}")
cnn = train(cnn, Xd[tr, None], yd[tr]); print(f"CNN on digits: {acc(cnn, Xd[te, None], yd[te]):.3f}, {n_params(cnn):,} parameters")

# %% [markdown]
# ## 4. RNN, LSTM, GRU: recurrence
# A recurrent net reads a sequence one step at a time, carrying a hidden state hₜ = tanh(W·xₜ + U·hₜ₋₁ + b).
# **Bias:** order matters and the same rule applies at every step. Plain RNNs forget over long gaps (vanishing
# gradients); **LSTM** and **GRU** add gates that decide what to keep.
#
# Task: a sequence of random digits; the label is the **first** digit, which must be remembered across the whole sequence.

# %%
def rnn_step(x, h, W, U, b):
    if SOLUTIONS: return torch.tanh(x @ W + h @ U + b)
    ...  # Exercise 4: one step of a vanilla RNN
W, U, b = torch.randn(3, 4), torch.randn(4, 4), torch.zeros(4)
ref = nn.RNN(3, 4, batch_first=True)
with torch.no_grad(): ref.weight_ih_l0.copy_(W.T); ref.weight_hh_l0.copy_(U.T); ref.bias_ih_l0.zero_(); ref.bias_hh_l0.zero_()
xs = torch.randn(1, 5, 3); h = torch.zeros(1, 4); outs = []
for t in range(5):
    h = rnn_step(xs[:, t], h, W, U, b)
    if h is None: break
    outs.append(h)
check("exercise 4: your RNN matches nn.RNN", None if h is None else torch.stack(outs, 1).detach().numpy(), ref(xs)[0].detach().numpy())

def memory_task(n, T):
    X = torch.randint(0, 10, (n, T)); return X, X[:, 0].clone()
class SeqNet(nn.Module):
    def __init__(self, kind):
        super().__init__(); self.emb = nn.Embedding(10, 32); self.rnn = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[kind](32, 64, batch_first=True); self.out = nn.Linear(64, 10)
    def forward(self, x): return self.out(self.rnn(self.emb(x))[0][:, -1])
T = 20
Xs, ys = memory_task(4000, T); Xv, yv = memory_task(500, T)
for kind in ["RNN", "LSTM", "GRU"]:
    m = train(SeqNet(kind), Xs, ys, epochs=30, lr=3e-3); print(f"{kind:4s} remembering the first of {T} digits: {acc(m, Xv, yv):.2f}  ({n_params(m):,} params)")

# %% [markdown]
# ## 5. Attention and the transformer
# Every position looks at every other position at once: scores = QKᵀ/√d, weights = softmax(scores), output =
# weights·V. No recurrence, so long-range links are one step away, and it parallelises across the sequence.
# **Bias:** almost none, which is why transformers need lots of data, and why they scale so well.

# %%
def attention(Q, K, V):
    if SOLUTIONS: return F.softmax(Q @ K.transpose(-2, -1) / Q.shape[-1] ** 0.5, -1) @ V
    ...  # Exercise 5: scaled dot-product attention
Q, K, V = torch.randn(2, 5, 8), torch.randn(2, 5, 8), torch.randn(2, 5, 8)
check("exercise 5: attention", attention(Q, K, V), F.scaled_dot_product_attention(Q, K, V))

class TinyTransformer(nn.Module):
    def __init__(self, T):
        super().__init__(); self.emb = nn.Embedding(10, 64); self.pos = nn.Parameter(torch.randn(T, 64) * 0.02)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(64, 4, 128, dropout=0.0, batch_first=True), 2); self.out = nn.Linear(64, 10)
    def forward(self, x): return self.out(self.enc(self.emb(x) + self.pos)[:, -1])
tf = train(TinyTransformer(T), Xs, ys, epochs=30, lr=1e-3); print(f"transformer on the same task: {acc(tf, Xv, yv):.2f}  ({n_params(tf):,} params)")

# %% [markdown]
# ### Exercise 6: which architecture's bias fits?
# Values: "mlp", "cnn", "rnn", "transformer".

# %%
def ex6():
    if SOLUTIONS: return {"x_ray_images": "cnn", "fixed_size_tabular_features": "mlp", "translate_long_documents_with_lots_of_data": "transformer", "tiny_device_streaming_sensor": "rnn"}
    ...
check("exercise 6", ex6(), {"x_ray_images": "cnn", "fixed_size_tabular_features": "mlp", "translate_long_documents_with_lots_of_data": "transformer", "tiny_device_streaming_sensor": "rnn"})

# %% [markdown]
# | Architecture | Core operation | Inductive bias | Typical data | Cost |
# |---|---|---|---|---|
# | MLP | dense matrix multiply | none: every input talks to every unit | fixed-size feature vectors | params grow with input size |
# | CNN | shared local filters + pooling | locality, translation equivariance | images, audio spectrograms, grids | cheap per pixel |
# | RNN / LSTM / GRU | the same cell applied step by step | order, a running memory | streams, small sequence models | sequential, can't parallelise over time |
# | Transformer | attention between all positions | almost none (positions added explicitly) | text, code, images as patches, everything at scale | O(T²) in sequence length |
#
# **What I measured** (CPU, full run): MLP 93% on digits with 17k parameters vs CNN 95% with 6k. Remembering
# the first of 20 digits: plain RNN 53%, LSTM 100%, GRU 100% (at 30 digits, all three stayed at chance in 25
# epochs, so the gates help but long gaps still need more training). The transformer solved it at 100%.
#
# **Also in the family:** graph neural networks (message passing on graphs, notebook 28), state space models like
# Mamba (recurrence that trains in parallel, notebook 15), mixture-of-experts layers (notebook 17).
