// Step-through walkthroughs for the core chapters: real (seeded) numbers flowing through a few lines
// of code, drawn as stacks of pixel cubes. Built with tensorWalk from tensorviz.js.
import { tensorWalk, op, M, rng, fmt } from "./tensorviz.js";

const r2 = (A) => M.round(A, 2);
const dims = (t, d, extra = {}) => ({ data: t, dims: d, ...extra });

const WALKS = {
  // -------------------------------------------------------------------------------------------
  "linear-algebra"() {
    const R = rng(11);
    const X = [[1, 0, 2], [0, 1, 1], [2, 1, 0]];
    const W = M.rand(3, 4, R, 0.8), Y = r2(M.mm(X, W)), Z = M.map(Y, (v) => Math.max(0, v));
    const T = { X: dims(X, "examples, features", { label: "X" }), W: dims(W, "in, out", { label: "W" }), Y: dims(Y, "examples, out", { label: "Y" }), Z: dims(Z, "examples, out", { label: "ReLU(Y)" }) };
    return tensorWalk({
      title: "A matrix multiply, cube by cube",
      lede: "Every layer of every network is this, repeated. Each cube is one number. Press Play, or hover a cube in Y to see where it came from.",
      code: `X = np.array([[1, 0, 2],     # 3 examples, 3 features each
              [0, 1, 1],
              [2, 1, 0]])
W = rng.normal(size=(3, 4))   # 3 features in, 4 out
Y = X @ W                     # (3, 3) @ (3, 4) -> (3, 4)
Z = np.maximum(Y, 0)          # ReLU: negatives become 0`,
      tensors: T,
      steps: [
        { title: "The data: one row per example", lines: [1, 2, 3], show: ["X"], text: "<p>Three examples, three features each. Rows are examples, columns are features. Orange is positive, and the paler the cube the closer it is to zero.</p>" },
        { title: "The weights: one column per output", lines: [4], show: ["W"], text: "<p>W has a row for every input feature and a column for every output. Training will change these numbers; for now they're random.</p>" },
        { title: "Multiply: a row meets a column", lines: [5], show: ["X", "@", "W", "=", "Y"], op: op.matmul(T, "X", "W", "Y"), text: "<p>Each cube of Y takes one <b>row</b> of X and one <b>column</b> of W, multiplies them pair by pair and adds up. The inner sizes (3 and 3) must match; the outer ones (3 and 4) give the shape of the result.</p>" },
        { title: "ReLU: clip the negatives", lines: [6], show: ["Y", "→", "Z"], op: op.cells(["Y", "Z"], 3, 4, (i, j) => `<b>ReLU(Y)[${i},${j}]</b> = max(0, ${fmt(Y[i][j])}) = <b>${fmt(Z[i][j])}</b>`), text: "<p>Cube by cube: blue (negative) becomes 0, everything else passes through. Without this bend, stacking layers would still be one straight-line function.</p>" },
      ],
    });
  },

  // -------------------------------------------------------------------------------------------
  backprop() {
    const R = rng(7);
    const x = [[0, 0], [0, 1], [1, 0], [1, 1]], t = [[0], [1], [1], [0]];
    const W1 = M.rand(2, 3, R, 1), W2 = M.rand(3, 1, R, 1);
    const hh = r2(M.mm(x, W1)), a = M.map(hh, (v) => Math.max(0, v)), y = r2(M.mm(a, W2));
    const dy = r2(M.map(y, (v, i) => (2 * (v - t[i][0])) / 4));
    const aT = M.T(a), dW2 = r2(M.mm(aT, dy)), W2T = M.T(W2), da = r2(M.mm(dy, W2T));
    const dh = r2(M.map(da, (v, i, j) => (hh[i][j] > 0 ? v : 0))), xT = M.T(x), dW1 = r2(M.mm(xT, dh));
    const T = {
      x: dims(x, "batch, in", { label: "x" }), W1: dims(W1, "in, hidden", { label: "W1" }), h: dims(hh, "batch, hidden", { label: "h" }), a: dims(a, "batch, hidden", { label: "a" }),
      W2: dims(W2, "hidden, out", { label: "W2" }), y: dims(y, "batch, out", { label: "y" }), t: dims(t, "batch, out", { label: "target" }), dy: dims(dy, "batch, out", { label: "dy" }),
      aT: dims(aT, "hidden, batch", { label: "aᵀ" }), dW2: dims(dW2, "hidden, out", { label: "dW2" }), W2T: dims(W2T, "out, hidden", { label: "W2ᵀ" }), da: dims(da, "batch, hidden", { label: "da" }),
      dh: dims(dh, "batch, hidden", { label: "dh" }), xT: dims(xT, "in, batch", { label: "xᵀ" }), dW1: dims(dW1, "in, hidden", { label: "dW1" }),
    };
    return tensorWalk({
      title: "Forward, then backward, cube by cube",
      lede: "A two-layer network on the four XOR inputs. Forward builds the prediction; backward sends the error back through the same multiplies, transposed. Every gradient has the same shape as the thing it updates.",
      code: `h = x @ W1                 # (4, 2) @ (2, 3)  forward
a = relu(h)
y = a @ W2                 # (4, 3) @ (3, 1)
loss = ((y - t) ** 2).mean()
dy = 2 * (y - t) / 4       # backward starts at the loss
dW2 = a.T @ dy             # (3, 4) @ (4, 1)  same shape as W2
da = dy @ W2.T             # (4, 1) @ (1, 3)  error for each hidden unit
dh = da * (h > 0)          # ReLU lets gradient through only where it fired
dW1 = x.T @ dh             # (2, 4) @ (4, 3)  same shape as W1`,
      tensors: T,
      steps: [
        { title: "Forward: inputs times the first weights", lines: [1], show: ["x", "@", "W1", "=", "h"], op: op.matmul(T, "x", "W1", "h"), text: "<p>Four inputs (the XOR table), two features each, into three hidden units.</p>" },
        { title: "ReLU", lines: [2], show: ["h", "→", "a"], op: op.cells(["h", "a"], 4, 3, (i, j) => `<b>a[${i},${j}]</b> = max(0, ${fmt(hh[i][j])}) = <b>${fmt(a[i][j])}</b>`), text: "<p>Negative hidden values become zero. Remember which ones: backward needs it.</p>" },
        { title: "Forward: hidden to output", lines: [3], show: ["a", "@", "W2", "=", "y"], op: op.matmul(T, "a", "W2", "y"), text: "<p>One prediction per input. The targets are 0, 1, 1, 0.</p>" },
        { title: "How wrong, and which way", lines: [4, 5], show: ["y", "−", "t", "→", "dy"], op: op.cells(["y", "t", "dy"], 4, 1, (i) => `<b>dy[${i}]</b> = 2 × (${fmt(y[i][0])} − ${t[i][0]}) / 4 = <b>${fmt(dy[i][0])}</b>`), text: "<p>The gradient of the loss with respect to each prediction: orange means “too high, push down”, blue means “too low, push up”.</p>" },
        { title: "Gradient for W2", lines: [6], show: ["aT", "@", "dy", "=", "dW2"], op: op.matmul(T, "aT", "dy", "dW2"), text: "<p>Each weight's blame = its input times the error downstream, summed over the batch. Transposing a lines the batch up so the sum happens inside the multiply. dW2 has W2's exact shape.</p>" },
        { title: "Send the error back through W2", lines: [7], show: ["dy", "@", "W2T", "=", "da"], op: op.matmul(T, "dy", "W2T", "da"), text: "<p>The same weights that carried the signal forward now carry the error backward, transposed.</p>" },
        { title: "Back through ReLU", lines: [8], show: ["da", "→", "dh"], op: op.cells(["h", "da", "dh"], 4, 3, (i, j) => `<b>dh[${i},${j}]</b> = ${fmt(da[i][j])} × (h = ${fmt(hh[i][j])} > 0 ? 1 : 0) = <b>${fmt(dh[i][j])}</b>`), text: "<p>Where a unit was switched off going forward, no gradient flows back through it. Grey cubes are exact zeros: dead units learn nothing from this batch.</p>" },
        { title: "Gradient for W1", lines: [9], show: ["xT", "@", "dh", "=", "dW1"], op: op.matmul(T, "xT", "dh", "dW1"), text: "<p>Same pattern as dW2. Now every weight has a gradient; the optimizer subtracts a small multiple of it, and that's one step of training.</p>" },
      ],
    });
  },

  // -------------------------------------------------------------------------------------------
  cnn() {
    // a 6x6 RGB image with a bright right half: a vertical edge down the middle
    const img = [0, 1, 2].map((c) => Array.from({ length: 6 }, (_, i) => Array.from({ length: 6 }, (_, j) => +((j >= 3 ? 0.9 : 0.1) - c * 0.12 + ((i * 7 + j * 3 + c) % 5) * 0.02).toFixed(2))));
    const edge = [0, 1, 2].map(() => [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]].map((r) => r.map((v) => +(v / 3).toFixed(2))));
    const conv = (X, K) => Array.from({ length: 4 }, (_, i) => Array.from({ length: 4 }, (_, j) => +K.reduce((s, kc, c) => s + kc.reduce((s2, row, di) => s2 + row.reduce((s3, w, dj) => s3 + w * X[c][i + di][j + dj], 0), 0), 0).toFixed(2)));
    const R = rng(5), more = [edge, ...[0, 1, 2].map(() => [0, 1, 2].map(() => M.rand(3, 3, R, 0.35)))];
    const one = conv(img, edge), maps = more.map((K) => conv(img, K)), relu = maps.map((m) => m.map((r) => r.map((v) => Math.max(0, v))));
    const pool = relu.map((m) => [0, 1].map((i) => [0, 1].map((j) => Math.max(m[2 * i][2 * j], m[2 * i][2 * j + 1], m[2 * i + 1][2 * j], m[2 * i + 1][2 * j + 1]))));
    const T = {
      x: dims(img, "channels, height, width", { label: "image", scale: 1 }), k: dims(edge, "channels, 3, 3", { label: "kernel" }), o: dims(one, "height, width", { label: "feature map" }),
      maps: dims(maps, "kernels, h, w", { label: "4 maps" }), relu: dims(relu, "kernels, h, w", { label: "ReLU" }), pool: dims(pool, "kernels, h/2, w/2", { label: "pooled" }),
    };
    return tensorWalk({
      title: "A convolution, cube by cube",
      lede: "The image is three slabs stacked: red, green and blue. A kernel is a small stack of its own, one slab per channel. Watch it slide.",
      code: `x = image                     # (3, 6, 6): 3 colour channels stacked
k = edge_kernel               # (3, 3, 3): spans every channel
out = conv2d(x, k)            # (4, 4): one sum of 27 products per position
maps = conv2d(x, kernels)     # (4, 4, 4): 4 kernels -> 4 feature maps
maps = relu(maps)
pooled = max_pool(maps, 2)    # (4, 2, 2): keep the largest of each 2x2`,
      tensors: T,
      steps: [
        { title: "The image: three channels stacked", lines: [1], show: ["x"], text: "<p>Height × width × channel. The right half is bright in every channel, so there's a vertical edge down the middle. A network never sees “an edge”; it sees these numbers.</p>" },
        { title: "The kernel: a tiny stack of weights", lines: [2], show: ["k"], text: "<p>3 × 3 in space, and as deep as the image (3 channels). This one is a hand-made edge detector: −⅓ on the left column, +⅓ on the right. A trained network learns its own.</p>" },
        { title: "Slide it: one window, one number", lines: [3], show: ["x", "⊛", "k", "=", "o"], op: op.conv(T, "x", "k", "o"), speed: 900, text: "<p>At each position the kernel covers a 3 × 3 window through <b>all three channels</b>: 27 input cubes, 27 weights, multiplied and summed into one output cube. The output lights up exactly where the edge is.</p>" },
        { title: "Many kernels: feature maps stack up", lines: [4], show: ["x", "⊛", "4 kernels", "=", "maps"], text: "<p>Four kernels give four maps, stacked into a new 3D block. That stack is the next layer's “image”: its channels are now features (edges, blobs, colours) rather than red, green and blue.</p>", read: "Layer 1 turns (3, 6, 6) into (4, 4, 4). The bottom slab is the edge map from the last step." },
        { title: "ReLU", lines: [5], show: ["maps", "→", "relu"], text: "<p>Negative responses become zero, slab by slab.</p>" },
        { title: "Pool: keep the strongest", lines: [6], show: ["relu", "→", "pool"], op: op.custom(16, (k) => { const f = Math.floor(k / 4), i = Math.floor((k % 4) / 2), j = k % 2, win = []; for (const a of [0, 1]) for (const b of [0, 1]) win.push(`${f},${2 * i + a},${2 * j + b}`); return { hl: { relu: win, pool: [`${f},${i},${j}`] }, read: `<b>pooled[${f},${i},${j}]</b> = max of the 2×2 window = <b>${fmt(pool[f][i][j])}</b>` }; }, (name, key) => { if (name !== "pool") return null; const [f, i, j] = key.split(",").map(Number); return f * 4 + i * 2 + j; }), text: "<p>Each 2 × 2 window keeps its largest value: half the height, half the width, same number of maps. The network becomes a little blind to exactly where a feature was, which is usually what you want.</p>" },
      ],
    });
  },

  // -------------------------------------------------------------------------------------------
  embeddings() {
    const R = rng(3), vocab = ["the", "cat", "dog", "sat", "on", "mat", "car", "ran"];
    const base = M.rand(8, 6, R, 0.6);
    base[2] = base[1].map((v, j) => +(v + (j % 2 ? 0.15 : -0.1)).toFixed(2));   // dog sits near cat
    const words = ["the", "cat", "sat", "dog"], ids = [[0, 1, 3, 2]];
    const X = ids[0].map((i) => base[i]);
    const Xn = r2(X.map((r) => { const n = Math.hypot(...r); return r.map((v) => v / n); })), S = r2(M.mm(Xn, M.T(Xn)));
    const T = { ids: dims(ids, "1, tokens", { label: "ids", kind: "ids", names: words }), E: dims(base, "vocab, dim", { label: "E", rowNames: vocab }), X: dims(X, "tokens, dim", { label: "X", rowNames: words }),
      Xn: dims(Xn, "tokens, dim", { label: "X / |X|", rowNames: words }), XT: dims(M.T(Xn), "dim, tokens", { label: "(X / |X|)ᵀ", names: words }), S: dims(S, "tokens, tokens", { label: "cos sim", kind: "prob", scale: 1, names: words, rowNames: words }) };
    return tensorWalk({
      title: "From words to vectors, cube by cube",
      lede: "An embedding is a table with one row per word. Looking a word up is just copying its row.",
      code: `vocab = ["the", "cat", "dog", "sat", "on", "mat", "car", "ran"]
ids = [vocab.index(w) for w in ["the", "cat", "sat", "dog"]]   # [0, 1, 3, 2]
E = rng.normal(size=(8, 6))     # learned: one row per word
X = E[ids]                      # (4, 6): copy those rows
X = X / np.linalg.norm(X, axis=1, keepdims=True)
S = X @ X.T                     # (4, 4): cosine similarity of every pair`,
      tensors: T,
      steps: [
        { title: "Words become ids", lines: [1, 2], show: ["ids"], text: "<p>The model never sees letters, only integers: each word's position in the vocabulary. Each colour here is one id.</p>" },
        { title: "The embedding table", lines: [3], show: ["E"], text: "<p>Eight words, six numbers each. These start random and are learned like any other weight. Words used in similar places drift to similar rows (here “dog” was placed near “cat” to show it).</p>" },
        { title: "Look up: copy a row per token", lines: [4], show: ["ids", "→", "E", "→", "X"], op: op.gather(T, "ids", "E", "X", words), speed: 1300, text: "<p>Indexing with a list of ids copies those rows into a new (tokens × dim) matrix. This is the very first operation inside every language model.</p>" },
        { title: "Similarity: every pair at once", lines: [5, 6], show: ["Xn", "@", "XT", "=", "S"], op: op.matmul(T, "Xn", "XT", "S"), text: `<p>After normalising each row, one matrix multiply gives the cosine similarity of every pair. The diagonal is 1 (each word with itself). “cat” and “dog” score <b>${fmt(S[1][3])}</b>; “cat” and “sat” only ${fmt(S[1][2])}.</p>` },
      ],
    });
  },

  // -------------------------------------------------------------------------------------------
  attention() {
    const R = rng(21), words = ["the", "cat", "sat", "down"];
    const X = M.rand(4, 4, R, 0.9), Wq = M.rand(4, 4, R, 0.7), Wk = M.rand(4, 4, R, 0.7), Wv = M.rand(4, 4, R, 0.7);
    const Q = r2(M.mm(X, Wq)), K = r2(M.mm(X, Wk)), V = r2(M.mm(X, Wv)), KT = M.T(K);
    const S = r2(M.map(M.mm(Q, KT), (v) => v / 2)), Sm = M.map(S, (v, i, j) => (j > i ? -Infinity : v)), A = r2(M.softmax(Sm)), O = r2(M.mm(A, V));
    const Qh = [0, 1].map((hd) => Q.map((r) => r.slice(hd * 2, hd * 2 + 2)));
    const T = {
      X: dims(X, "tokens, d", { label: "X" }), Wq: dims(Wq, "d, d", { label: "Wq" }), Q: dims(Q, "tokens, d", { label: "Q" }), Wk: dims(Wk, "d, d", { label: "Wk" }), K: dims(K, "tokens, d", { label: "K" }),
      Wv: dims(Wv, "d, d", { label: "Wv" }), V: dims(V, "tokens, d", { label: "V" }), KT: dims(KT, "d, tokens", { label: "Kᵀ" }), S: dims(S, "query, key", { label: "scores" }),
      Sm: dims(Sm, "query, key", { label: "masked" }), A: dims(A, "query, key", { label: "weights", kind: "prob", scale: 1, bars: true }), O: dims(O, "tokens, d", { label: "out" }),
      Qh: dims(Qh, "heads, tokens, d/heads", { label: "Q by head" }),
    };
    return tensorWalk({
      title: "Attention, cube by cube",
      lede: `Four tokens (“${words.join(" ")}”), four numbers each. Every line of the attention function, with the actual numbers. Hover any cube in an output to see how it was made.`,
      code: `Q = X @ Wq                      # what each token is looking for
K = X @ Wk                      # what each token offers
V = X @ Wv                      # what each token hands over
S = Q @ K.T / np.sqrt(d)        # (4, 4): every query against every key
S[np.triu_indices(4, k=1)] = -np.inf   # no peeking at later tokens
A = softmax(S, axis=-1)         # each row sums to 1
out = A @ V                     # weighted average of the values
Qh = Q.reshape(4, 2, 2).transpose(1, 0, 2)   # 2 heads, stacked`,
      tensors: T,
      steps: [
        { title: "Queries: what each token is looking for", lines: [1], show: ["X", "@", "Wq", "=", "Q"], op: op.matmul(T, "X", "Wq", "Q"), text: "<p>Each row of X is one token's vector. Multiplying by a learned matrix turns it into a <b>query</b>. Rows stay tokens: row 1 of Q belongs to “cat”.</p>" },
        { title: "Keys: what each token offers", lines: [2], show: ["X", "@", "Wk", "=", "K"], op: op.matmul(T, "X", "Wk", "K"), text: "<p>Same tokens, a different learned matrix. Values (V) come from a third one, the same way.</p>" },
        { title: "Scores: every query against every key", lines: [4], show: ["Q", "@", "KT", "=", "S"], op: op.matmul(T, "Q", "KT", "S", { scale: 0.5, scaleLabel: "/ √4" }), text: "<p>Cell (i, j) is the dot product of token i's query with token j's key, divided by √d to keep softmax from saturating. One multiply scores all 16 pairs. Row = who is asking, column = who is being asked.</p>" },
        { title: "The causal mask", lines: [5], show: ["S", "→", "Sm"], op: op.rows(["S", "Sm"], 4, (i) => `row ${i} (“${words[i]}”) may look at ${words.slice(0, i + 1).map((w) => `“${w}”`).join(", ")}; ${3 - i ? `${3 - i} later cell${3 - i > 1 ? "s" : ""} set to −∞` : "nothing to hide"}`), speed: 1400, text: "<p>A language model predicts the next token, so token i must not see tokens after it. Everything above the diagonal becomes −∞ (dark cubes), which softmax turns into exactly 0.</p>" },
        { title: "Softmax: scores become weights", lines: [6], show: ["Sm", "→", "A"], op: op.rows(["Sm", "A"], 4, (i) => `<b>row ${i}</b>: e^score / Σ e^score = [${A[i].map((v) => fmt(v)).join(", ")}], sums to 1`), speed: 1400, text: "<p>Each row becomes a probability distribution: the taller the bar, the more that token attends there. “the” can only attend to itself, so its whole weight sits on one cube.</p>" },
        { title: "Mix the values", lines: [7], show: ["A", "@", "V", "=", "O"], op: op.matmul(T, "A", "V", "O"), text: "<p>Row i of the output is a weighted average of the value rows, using row i of the weights. This is the whole trick: each token pulls in information from the tokens it chose to look at.</p>" },
        { title: "Many heads: the same thing, stacked", lines: [8], show: ["Q", "→", "Qh"], op: op.custom(2, (hd) => ({ hl: { Q: [0, 1, 2, 3].flatMap((i) => [0, 1].map((j) => `${i},${hd * 2 + j}`)), Qh: [0, 1, 2, 3].flatMap((i) => [0, 1].map((j) => `${hd},${i},${j}`)) }, read: `<b>head ${hd}</b> takes columns ${hd * 2}–${hd * 2 + 1} of Q (and of K and V) and runs everything above on its own` }), (name, key) => (name === "Qh" ? +key.split(",")[0] : null)), speed: 1600, text: "<p>Split the columns into groups and stack them: each slab is a <b>head</b> with its own small attention pattern. Real models use 8 to 128 heads; their outputs are joined back side by side and mixed by one more matrix.</p>" },
      ],
    });
  },

  // -------------------------------------------------------------------------------------------
  transformer() {
    const R = rng(33);
    const x = M.rand(4, 4, R, 1);
    const ln = (A) => A.map((r) => { const m = r.reduce((a, b) => a + b, 0) / r.length, s = Math.sqrt(r.reduce((a, b) => a + (b - m) ** 2, 0) / r.length + 1e-5); return r.map((v) => +((v - m) / s).toFixed(2)); });
    const n1 = ln(x), att = r2(M.map(M.mm(n1, M.rand(4, 4, R, 0.4)), (v) => v)), x2 = r2(M.add(x, att));
    const n2 = ln(x2), W1 = M.rand(4, 8, R, 0.5), hh = r2(M.mm(n2, W1)), g = r2(M.map(hh, (v) => 0.5 * v * (1 + Math.tanh(0.7978845608 * (v + 0.044715 * v ** 3))))), W2 = M.rand(8, 4, R, 0.4), m = r2(M.mm(g, W2)), x3 = r2(M.add(x2, m));
    const T = {
      x: dims(x, "tokens, d", { label: "x" }), n1: dims(n1, "tokens, d", { label: "LN(x)" }), att: dims(att, "tokens, d", { label: "attn" }), x2: dims(x2, "tokens, d", { label: "x" }),
      n2: dims(n2, "tokens, d", { label: "LN(x)" }), W1: dims(W1, "d, 4d", { label: "W1" }), h: dims(hh, "tokens, 4d", { label: "h" }), g: dims(g, "tokens, 4d", { label: "GELU(h)" }),
      W2: dims(W2, "4d, d", { label: "W2" }), m: dims(m, "tokens, d", { label: "mlp" }), x3: dims(x3, "tokens, d", { label: "x out" }),
    };
    const mean = (r) => r.reduce((a, b) => a + b, 0) / r.length;
    return tensorWalk({
      title: "One transformer block, cube by cube",
      lede: "Two sub-layers, each wrapped the same way: normalise, transform, add back. The residual stream (x) keeps its shape the whole way through.",
      code: `def block(x):                   # x: (tokens, d) = (4, 4)
    x = x + attn(ln1(x))        # tokens talk to each other
    h = ln2(x) @ W1             # (4, 4) @ (4, 8): widen 4x
    h = gelu(h)
    x = x + h @ W2              # (4, 8) @ (8, 4): back to d
    return x                    # same shape as it came in`,
      tensors: T,
      steps: [
        { title: "Normalise each token", lines: [2], show: ["x", "→", "n1"], op: op.rows(["x", "n1"], 4, (i) => `row ${i}: mean ${fmt(mean(x[i]))} → 0, spread → 1: [${n1[i].map((v) => fmt(v)).join(", ")}]`), speed: 1400, text: "<p>LayerNorm works on one token (one row) at a time: subtract its mean, divide by its spread. It keeps the numbers in a range where the next layer behaves, however deep the stack gets.</p>" },
        { title: "Attention, then add it back", lines: [2], show: ["x", "+", "att", "=", "x2"], op: op.cells(["x", "att", "x2"], 4, 4, (i, j) => `<b>x[${i},${j}]</b> = ${fmt(x[i][j])} + ${fmt(att[i][j])} = <b>${fmt(x2[i][j])}</b>`), speed: 700, text: "<p>Attention (the previous chapter) mixes information across tokens. Its output is <b>added</b> to x, cube by cube, rather than replacing it. That residual add is why 100-layer models still train: gradient has a straight road back.</p>" },
        { title: "The MLP widens", lines: [3], show: ["n2", "@", "W1", "=", "h"], op: op.matmul(T, "n2", "W1", "h"), speed: 800, text: "<p>Each token on its own goes through a two-layer network that is 4× wider in the middle. Attention moves information between tokens; the MLP processes it within each token. Most of a model's parameters live here.</p>" },
        { title: "GELU", lines: [4], show: ["h", "→", "g"], op: op.cells(["h", "g"], 4, 8, (i, j) => `<b>GELU(${fmt(hh[i][j])})</b> = <b>${fmt(g[i][j])}</b>: big positives pass, negatives shrink toward 0`), speed: 600, text: "<p>A smooth ReLU. Without a bend here the two multiplies would collapse into one.</p>" },
        { title: "Back to width d", lines: [5], show: ["g", "@", "W2", "=", "m"], op: op.matmul(T, "g", "W2", "m"), speed: 900, text: "<p>The second matrix brings each token back to 4 numbers, so it can be added to the stream.</p>" },
        { title: "Add again: same shape out", lines: [5, 6], show: ["x2", "+", "m", "=", "x3"], op: op.cells(["x2", "m", "x3"], 4, 4, (i, j) => `<b>x[${i},${j}]</b> = ${fmt(x2[i][j])} + ${fmt(m[i][j])} = <b>${fmt(x3[i][j])}</b>`), speed: 700, text: "<p>(4, 4) in, (4, 4) out. That's why blocks stack: a model is this function applied 12, 32 or 96 times, each with its own weights.</p>" },
      ],
    });
  },

  // -------------------------------------------------------------------------------------------
  "next-token"() {
    const R = rng(41), vocab = ["the", "cat", "sat", "on", "mat", "dog", "ran", "."];
    const words = ["the", "cat", "sat", "on"], nextIds = [1, 2, 3, 0];
    const hdn = M.rand(4, 4, R, 1), Wu = M.rand(4, 8, R, 0.9);
    nextIds.forEach((n, i) => { for (let r = 0; r < 4; r++) Wu[r][n] = +(Wu[r][n] + hdn[i][r] * 0.5).toFixed(2); });   // a slightly trained head
    const L = r2(M.mm(hdn, Wu)), P = r2(M.softmax(L)), nll = nextIds.map((n, i) => -Math.log(Math.max(1e-9, P[i][n])));
    const T = { hdn: dims(hdn, "tokens, d", { label: "hidden" }), Wu: dims(Wu, "d, vocab", { label: "W_out" }), L: dims(L, "tokens, vocab", { label: "logits" }), P: dims(P, "tokens, vocab", { label: "probs", kind: "prob", scale: 1, bars: 4 }) };
    T.L.names = vocab; T.P.names = vocab; T.Wu.names = vocab;
    return tensorWalk({
      title: "From hidden state to next word, cube by cube",
      lede: `The last layer of a language model. Four positions (“${words.join(" ")}”), a vocabulary of eight words. Every position predicts the word that comes after it.`,
      code: `h = model(tokens)              # (4, d): one vector per position
logits = h @ W_out             # (4, 8): a score for every word in the vocab
probs = softmax(logits, -1)    # each row sums to 1
loss = -np.log(probs[range(4), next_ids]).mean()   # how surprised by the truth
nxt = rng.choice(8, p=probs[-1])   # generating: sample from the last row`,
      tensors: T,
      steps: [
        { title: "One vector per position", lines: [1], show: ["hdn"], text: "<p>After all the blocks, each position has a vector that (if training went well) encodes what should come next.</p>" },
        { title: "Score every word in the vocabulary", lines: [2], show: ["hdn", "@", "Wu", "=", "L"], op: op.matmul(T, "hdn", "Wu", "L"), speed: 800, text: "<p>One column of W_out per vocabulary word. A dot product with each column gives a <b>logit</b>: a raw score for that word. Real vocabularies have 50,000 to 250,000 columns, which makes this one of the biggest matrices in the model.</p>" },
        { title: "Softmax: scores become probabilities", lines: [3], show: ["L", "→", "P"], op: op.rows(["L", "P"], 4, (i) => { const b = P[i].indexOf(Math.max(...P[i])); return `after “${words.slice(0, i + 1).join(" ")}”: top guess <b>“${vocab[b]}”</b> at ${fmt(P[i][b] * 100, 0)}%`; }), speed: 1500, text: "<p>Each row becomes a distribution over the vocabulary. Bar height is probability. Hover a bar to see which word it is.</p>" },
        { title: "Training: how surprised by the truth?", lines: [4], show: ["P"], op: op.custom(4, (i) => ({ hl: { P: [`${i},${nextIds[i]}`] }, read: `position ${i}: the true next word is “${vocab[nextIds[i]]}”, given p = ${fmt(P[i][nextIds[i]])}, loss = −log p = <b>${fmt(nll[i])}</b> <span class="tw-dim">(mean over positions: ${fmt(nll.reduce((a, b) => a + b, 0) / 4)})</span>` }), (name, key) => (name === "P" ? +key.split(",")[0] : null)), speed: 1500, text: "<p>The loss only looks at one cube per row: the probability given to the word that actually came next. Push it up and every other bar must come down. Every position is a training example, all from one forward pass.</p>" },
        { title: "Generating: sample from the last row", lines: [5], show: ["P"], op: op.custom(1, () => ({ hl: { P: [0, 1, 2, 3, 4, 5, 6, 7].map((j) => `3,${j}`) }, read: `next word after “${words.join(" ")}”: drawn from [${P[3].map((v, j) => `${vocab[j]} ${fmt(v)}`).join(", ")}]` })), text: "<p>When writing text, only the last position matters: sample a word from its row, append it, run the model again. One token at a time.</p>" },
      ],
    });
  },

  // -------------------------------------------------------------------------------------------
  moe() {
    const R = rng(52), E = 4, words = ["the", "cat", "sat", "on", "a", "mat"];
    const X = M.rand(6, 4, R, 1), Wr = M.rand(4, E, R, 0.9), L = r2(M.mm(X, Wr)), P = r2(M.softmax(L));
    const top = P.map((r) => r.map((v, j) => j).sort((a, b) => r[b] - r[a]).slice(0, 2));
    const G = P.map((r, i) => r.map((v, j) => (top[i].includes(j) ? v : 0)));
    const load = [E].map(() => Array.from({ length: E }, (_, e) => top.filter((t) => t.includes(e)).length));
    const T = { X: dims(X, "tokens, d", { label: "tokens" }), Wr: dims(Wr, "d, experts", { label: "router" }), L: dims(L, "tokens, experts", { label: "logits" }), P: dims(P, "tokens, experts", { label: "probs", kind: "prob", scale: 1, bars: true }), G: dims(G, "tokens, experts", { label: "top-2", kind: "prob", scale: 1, bars: true }), load: dims(load, "1, experts", { label: "load", kind: "prob", scale: 6, bars: 5 }) };
    return tensorWalk({
      title: "Routing tokens to experts, cube by cube",
      lede: "Six tokens, four experts, each token goes to its top two. The router is just one small matrix.",
      code: `logits = x @ W_router          # (6, 4): a score per token per expert
probs = softmax(logits, -1)
top2 = np.argsort(-probs, -1)[:, :2]      # each token picks 2 experts
gates = keep(probs, top2)      # zero out the rest
load = count(top2, experts=4)  # tokens per expert: should be even
y = sum(gates[:, e] * expert[e](x) for e in top2)`,
      tensors: T,
      steps: [
        { title: "Score every expert for every token", lines: [1], show: ["X", "@", "Wr", "=", "L"], op: op.matmul(T, "X", "Wr", "L"), speed: 800, text: "<p>The router is a (d × experts) matrix: one dot product per token per expert.</p>" },
        { title: "Softmax per token", lines: [2], show: ["L", "→", "P"], op: op.rows(["L", "P"], 6, (i) => `“${words[i]}”: [${P[i].map((v) => fmt(v)).join(", ")}]`), speed: 1200, text: "<p>Each token's row becomes a distribution over experts.</p>" },
        { title: "Keep the top two", lines: [3, 4], show: ["P", "→", "G"], op: op.rows(["P", "G"], 6, (i) => `“${words[i]}” goes to experts <b>${top[i].join(" and ")}</b> with gates ${top[i].map((e) => fmt(P[i][e])).join(" and ")}`), speed: 1200, text: "<p>Only two experts run for each token; the others cost nothing. That's how a model can have 8× the parameters with roughly 2× the compute.</p>" },
        { title: "Load: are the experts used evenly?", lines: [5], show: ["G", "→", "load"], op: op.custom(E, (e) => ({ hl: { G: [0, 1, 2, 3, 4, 5].filter((i) => top[i].includes(e)).map((i) => `${i},${e}`), load: [`0,${e}`] }, read: `expert ${e} gets <b>${load[0][e]}</b> token${load[0][e] === 1 ? "" : "s"}: ${[0, 1, 2, 3, 4, 5].filter((i) => top[i].includes(e)).map((i) => `“${words[i]}”`).join(", ") || "none"}` }), (name, key) => (name === "load" ? +key.split(",")[1] : null)), speed: 1300, text: "<p>Count tokens per column. If one expert gets everything it becomes a bottleneck and the others never learn, so training adds a small loss that rewards an even load.</p>" },
      ],
    });
  },

  // -------------------------------------------------------------------------------------------
  inference() {
    const R = rng(64), words = ["the", "cat", "sat", "down"];
    const K = M.rand(4, 4, R, 0.9), V = M.rand(4, 4, R, 0.9), q = M.rand(4, 4, R, 0.9);
    const T = {}, steps = [];
    for (let t = 1; t <= 4; t++) {
      const Kc = K.slice(0, t), sc = Kc.map((k) => +(k.reduce((s, v, j) => s + v * q[t - 1][j], 0) / 2).toFixed(2)), w = M.softmax([sc])[0].map((v) => +v.toFixed(2));
      T[`q${t}`] = dims([q[t - 1]], "1, d", { label: `q (“${words[t - 1]}”)` });
      T[`K${t}`] = dims(Kc, `${t} cached, d`, { label: "K cache" });
      T[`s${t}`] = dims([w], `1, ${t}`, { label: "weights", kind: "prob", scale: 1, bars: true });
      steps.push({
        title: `Token ${t}: “${words[t - 1]}”`, lines: t === 1 ? [1, 2, 3] : [2, 3, 4], show: [`q${t}`, "·", `K${t}`, "→", `s${t}`],
        op: op.custom(t, (j) => ({ hl: { [`q${t}`]: [0, 1, 2, 3].map((c) => `0,${c}`), [`K${t}`]: [0, 1, 2, 3].map((c) => `${j},${c}`), [`s${t}`]: [`0,${j}`] }, read: `the new query meets cached key ${j} (“${words[j]}”): weight <b>${fmt(w[j])}</b>${j === t - 1 ? " <span class=\"tw-dim\">(the row just added)</span>" : ""}` }), (name, key) => (name === `K${t}` ? +key.split(",")[0] : name === `s${t}` ? +key.split(",")[1] : null)),
        speed: 1000,
        text: t === 1 ? "<p>Generating one token at a time. The first token computes its key and value and <b>stores</b> them.</p>" : t === 2 ? "<p>The next token only computes its own q, k and v. Its k joins the cache (one more row), and its query is scored against every cached key. Nothing for earlier tokens is recomputed.</p>" : t === 3 ? "<p>The cache grows by one row per token. Work per new token grows with the cache, not with its square.</p>" : `<p>After ${t} tokens the cache holds ${t} rows per layer per head, for keys and the same again for values. At long context this cache, not the weights, is what fills GPU memory.</p>`,
      });
    }
    return tensorWalk({
      title: "The KV cache, cube by cube",
      lede: "Generation without recomputing the past: keep every token's key and value, add one row per step.",
      code: `cache_k, cache_v = [], []
for tok in prompt_then_generated:
    q, k, v = project(embed(tok))       # only for the new token
    cache_k.append(k); cache_v.append(v)
    w = softmax(q @ np.array(cache_k).T / np.sqrt(d))
    out = w @ np.array(cache_v)`,
      tensors: T,
      steps,
    });
  },

  // -------------------------------------------------------------------------------------------
  diffusion() {
    const R = rng(77), N = 6;
    const x0 = Array.from({ length: N }, (_, i) => Array.from({ length: N }, (_, j) => { const d = Math.hypot(i - 2.5, j - 2.5); return d < 1.6 ? 1 : d < 2.6 ? -0.6 : -1; }));
    const eps = M.rand(N, N, R, 1), ab = [0.95, 0.6, 0.25, 0.04];
    const xt = ab.map((a) => r2(M.map(x0, (v, i, j) => Math.sqrt(a) * v + Math.sqrt(1 - a) * eps[i][j])));
    const t = 2, a = ab[t], epsHat = r2(M.map(eps, (v, i, j) => v + ((i + j) % 3 - 1) * 0.12)), x0hat = r2(M.map(xt[t], (v, i, j) => (v - Math.sqrt(1 - a) * epsHat[i][j]) / Math.sqrt(a)));
    const T = { x0: dims(x0, "h, w", { label: "x₀", scale: 1 }), eps: dims(eps, "h, w", { label: "ε" }), xt: dims(xt, "steps, h, w", { label: "x_t, t = 1…4", scale: 1.6, gap: 2 }), x2: dims(xt[t], "h, w", { label: `x_t (ᾱ = ${a})`, scale: 1.6 }), eh: dims(epsHat, "h, w", { label: "ε̂" }), x0h: dims(x0hat, "h, w", { label: "x̂₀", scale: 1 }) };
    return tensorWalk({
      title: "Noising and denoising, cube by cube",
      lede: "A 6 × 6 “image” of a blob. Forward: blend in noise until nothing is left. The model's whole job: look at a noisy image and say which noise was added.",
      code: `x0 = image                                    # (6, 6), values in [-1, 1]
eps = rng.normal(size=x0.shape)
xt = np.sqrt(abar[t]) * x0 + np.sqrt(1 - abar[t]) * eps   # any t in one shot
eps_hat = model(xt, t)                        # predict the noise
loss = ((eps_hat - eps) ** 2).mean()
x0_hat = (xt - np.sqrt(1 - abar[t]) * eps_hat) / np.sqrt(abar[t])`,
      tensors: T,
      steps: [
        { title: "The clean image", lines: [1], show: ["x0"], text: "<p>Orange is bright, blue is dark: a bright blob on a dark background.</p>" },
        { title: "Pure noise", lines: [2], show: ["eps"], text: "<p>One random number per pixel. This is the thing the model will learn to predict.</p>" },
        { title: "Blend: any noise level in one step", lines: [3], show: ["x0", "+", "eps", "=", "x2"], op: op.cells(["x0", "eps", "x2"], N, N, (i, j) => `<b>x_t[${i},${j}]</b> = √${a} × ${fmt(x0[i][j])} + √${fmt(1 - a)} × ${fmt(eps[i][j])} = <b>${fmt(xt[t][i][j])}</b>`), speed: 500, text: "<p>Each pixel is a weighted mix of the image and the noise. The weights (ᾱ) come from a fixed schedule, so you can jump straight to any step without running the ones before it.</p>" },
        { title: "The whole forward process, stacked", lines: [3], show: ["xt"], text: "<p>Four noise levels as four slabs, least noisy at the bottom. The blob fades into static as you go up. Generation runs this stack downward: start from static at the top and remove a little noise at a time.</p>" },
        { title: "The model predicts the noise", lines: [4, 5], show: ["x2", "→", "eh"], op: op.cells(["eps", "eh"], N, N, (i, j) => `true ε ${fmt(eps[i][j])}, predicted ${fmt(epsHat[i][j])}, squared error ${fmt((epsHat[i][j] - eps[i][j]) ** 2, 3)}`), speed: 500, text: "<p>Given x_t and t, a network (a U-Net or a transformer) outputs one number per pixel: its guess of ε. The loss is plain mean squared error against the noise that was really added.</p>" },
        { title: "Undo it: estimate the clean image", lines: [6], show: ["x2", "−", "eh", "=", "x0h"], op: op.cells(["x2", "eh", "x0h"], N, N, (i, j) => `<b>x̂₀[${i},${j}]</b> = (${fmt(xt[t][i][j])} − √${fmt(1 - a)} × ${fmt(epsHat[i][j])}) / √${a} = <b>${fmt(x0hat[i][j])}</b>`), speed: 500, text: "<p>Rearranging the blend gives back an estimate of the clean image. A sampler takes a small step toward it, adds a little fresh noise, and repeats.</p>" },
      ],
    });
  },

  // -------------------------------------------------------------------------------------------
  sft() {
    const R = rng(88), d = 6, r = 2;
    const W = M.rand(d, d, R, 0.7), A = M.rand(r, d, R, 0.5), B = M.rand(d, r, R, 0.6), BA = r2(M.mm(B, A)), W2 = r2(M.add(W, BA));
    const T = { W: dims(W, "d, d", { label: "W (frozen)" }), B: dims(B, "d, r", { label: "B" }), A: dims(A, "r, d", { label: "A" }), BA: dims(BA, "d, d", { label: "B @ A" }), W2: dims(W2, "d, d", { label: "W + BA" }) };
    return tensorWalk({
      title: "LoRA, cube by cube",
      lede: "Fine-tune a big matrix by training two thin ones. Here d = 6 and rank r = 2; in a real model d is 4,096 and r is 8 to 64.",
      code: `W = pretrained_weight          # (d, d), frozen: no gradient
A = rng.normal(size=(r, d)) * 0.01   # (r, d), trained
B = np.zeros((d, r))           # (d, r), trained, starts at zero
delta = B @ A                  # (d, d), but only rank r
y = x @ (W + delta).T          # same shape as before`,
      tensors: T,
      steps: [
        { title: "The frozen weight", lines: [1], show: ["W"], text: `<p>${d * d} numbers. In a 7B model, a single matrix like this has 16 million, and there are hundreds of them. Full fine-tuning updates all of them and needs memory for their gradients and optimizer state.</p>` },
        { title: "Two thin matrices", lines: [2, 3], show: ["B", "A"], text: `<p>B is d × r and A is r × d: ${d * r * 2} numbers instead of ${d * d}. (B really starts at zero so training begins exactly at the pretrained model; it's shown after some training here.)</p>` },
        { title: "Their product is a full-size update", lines: [4], show: ["B", "@", "A", "=", "BA"], op: op.matmul(T, "B", "A", "BA"), speed: 700, text: "<p>Every cube of the update comes from a row of B and a column of A, each only r = 2 long. So the update can only point in 2 independent directions: it's “low rank”. For adapting a model to a task, that turns out to be enough.</p>" },
        { title: "Add it on", lines: [5], show: ["W", "+", "BA", "=", "W2"], op: op.cells(["W", "BA", "W2"], d, d, (i, j) => `<b>W'[${i},${j}]</b> = ${fmt(W[i][j])} + ${fmt(BA[i][j])} = <b>${fmt(W2[i][j])}</b>`), speed: 500, text: "<p>After training, BA can be merged into W, so inference costs exactly the same as the original model. Or keep many small (A, B) pairs and swap them per task.</p>" },
      ],
    });
  },
  // -------------------------------------------------------------------------------------------
  rnn() {
    const R = rng(91), words = ["the", "cat", "sat", "down"];
    const X = M.rand(4, 3, R, 1), Wx = M.rand(3, 4, R, 0.6), Wh = M.rand(4, 4, R, 0.6);
    const H = []; let prev = [0, 0, 0, 0];
    for (let t = 0; t < 4; t++) { const a = Wx[0].map((_, j) => X[t].reduce((s, v, k) => s + v * Wx[k][j], 0)), b = Wh[0].map((_, j) => prev.reduce((s, v, k) => s + v * Wh[k][j], 0)); prev = a.map((v, j) => +Math.tanh(v + b[j]).toFixed(2)); H.push({ a, b, h: prev }); }
    const upTo = (t) => Array.from({ length: 4 }, (_, i) => (i <= t ? H[i].h : [null, null, null, null]));
    const T = { X: dims(X, "time, in", { label: "X", rowNames: words, px: 3 }), Wx: dims(Wx, "in, hidden", { label: "Wx", px: 3 }), Wh: dims(Wh, "hidden, hidden", { label: "Wh", px: 3 }) };
    for (let t = 0; t < 4; t++) { T[`P${t}`] = dims(t ? upTo(t - 1) : [[0, 0, 0, 0], ...upTo(-1).slice(1)], "time, hidden", { label: "H before", rowNames: words, px: 3 }); T[`H${t}`] = dims(upTo(t), "time, hidden", { label: "H after", rowNames: words, px: 3 }); }
    const steps = [0, 1, 2, 3].map((t) => ({
      title: `Time step ${t}: “${words[t]}”`, lines: [2, 3], show: ["X", "@", "Wx", "+", `P${t}`, "@", "Wh", "→", `H${t}`], speed: 1100,
      op: op.custom(4, (j) => ({
        hl: { X: [0, 1, 2].map((k) => `${t},${k}`), Wx: [0, 1, 2].map((k) => `${k},${j}`), [`P${t}`]: [0, 1, 2, 3].map((k) => `${Math.max(0, t - 1)},${k}`), Wh: [0, 1, 2, 3].map((k) => `${k},${j}`), [`H${t}`]: [`${t},${j}`] },
        read: `<b>h[${t},${j}]</b> = tanh( x_${t} · Wx[:,${j}] + h_${t - 1 < 0 ? "start (zeros)" : t - 1} · Wh[:,${j}] ) = tanh(${fmt(H[t].a[j])} + ${fmt(H[t].b[j])}) = <b>${fmt(H[t].h[j])}</b>`,
      }), (name, key) => (name === `H${t}` && +key.split(",")[0] === t ? +key.split(",")[1] : null)),
      text: t === 0 ? "<p>The hidden state starts at zero. The first word is mixed in through Wx, the (empty) past through Wh, and tanh squashes the sum into −1…1. Pale cubes haven't been computed yet.</p>" : t === 1 ? "<p>Now the past matters: row 0 of H (the memory of “the”) goes through Wh and is added to the new word. <b>The same Wx and Wh</b> are used at every step.</p>" : t === 2 ? "<p>Each new row depends on the one before it, so the steps can't run in parallel: step 3 has to wait for step 2. That's the main reason transformers replaced RNNs for training on long text.</p>" : "<p>The last row has seen all four words, squeezed through four tanh steps. Information from early words fades a little each step, which is the vanishing-gradient problem in picture form.</p>",
    }));
    return tensorWalk({
      title: "A recurrent network, one time step at a time",
      lede: "The hidden state is a running memory: one row per time step, each built from the new input and the row before it.",
      code: `h = np.zeros(4)                          # memory starts empty
for t, x_t in enumerate(X):              # one word at a time, in order
    h = np.tanh(x_t @ Wx + h @ Wh)       # new input + old memory
    H.append(h)                          # (time, hidden)`,
      tensors: T,
      steps,
    });
  },

  // -------------------------------------------------------------------------------------------
  "q-learning"() {
    const S = 6, acts = ["←", "→"], alpha = 0.5, gamma = 0.9;
    const Q0 = Array.from({ length: S }, () => [0, 0]);
    const moves = [[4, 1, 1, 5, "reach the goal"], [3, 1, 0, 4, ""], [4, 1, 1, 5, "reach the goal again"], [2, 1, 0, 3, ""], [3, 0, 0, 2, "a step the wrong way"]];
    const SN = ["s0", "s1", "s2", "s3", "s4", "s5 goal"], tbl = (Qm, label) => dims(M.T(Qm), "action, state", { label, kind: "prob", scale: 1, bars: 4, names: SN, rowNames: acts });
    const T = { Q0: tbl(Q0, "Q") };
    let Q = Q0.map((r) => [...r]);
    const steps = [{ title: "The table: one value per state and action", lines: [1], show: ["Q0"], text: "<p>A corridor of six cells; the goal is cell 5. Q[s, a] will hold “how much reward I expect if I take action a in state s and act well afterwards”. It starts at zero everywhere.</p>" }];
    moves.forEach(([s, a, r, s2, why], n) => {
      const before = Q.map((x) => [...x]), best = Math.max(...before[s2]), target = r + (s2 === 5 ? 0 : gamma * best), nv = +(before[s][a] + alpha * (target - before[s][a])).toFixed(3);
      Q[s][a] = nv; const after = Q.map((x) => [...x]);
      T[`b${n}`] = tbl(before, "Q before");
      T[`a${n}`] = tbl(after, "Q after");
      steps.push({
        title: `Update ${n + 1}: in s${s}, go ${acts[a]}${why ? `, ${why}` : ""}`, lines: [3, 4, 5], show: [`b${n}`, "→", `a${n}`],
        op: op.custom(1, () => ({ hl: { [`b${n}`]: [`${a},${s}`, ...(s2 === 5 ? [] : [`0,${s2}`, `1,${s2}`])], [`a${n}`]: [`${a},${s}`] }, read: `target = r + γ · max Q[s${s2}] = ${r} + ${s2 === 5 ? "0 (episode ends)" : `0.9 × ${fmt(best, 3)}`} = ${fmt(target, 3)}<br><b>Q[s${s},${acts[a]}]</b> = ${fmt(before[s][a], 3)} + 0.5 × (${fmt(target, 3)} − ${fmt(before[s][a], 3)}) = <b>${fmt(nv, 3)}</b>` })),
        text: n === 0 ? "<p>Stepping right from s4 reaches the goal: reward 1. The cell moves halfway (α = 0.5) toward the target.</p>" : n === 1 ? "<p>From s3, moving right gives no reward, but it lands in s4, which now has value. Some of it flows back: 0.9 × 0.5. <b>Value spreads backward from the goal</b>, one update at a time.</p>" : n === 2 ? "<p>The same transition again moves s4 closer to 1. Repeated experience sharpens the estimates.</p>" : n === 3 ? "<p>Two steps away from the goal now gets a little value too. After many episodes every cell knows which way leads home.</p>" : "<p>Moving left lands in s2, whose best value is small, so this action's value stays low. Comparing a row's two bars is the policy: take the taller one.</p>",
      });
    });
    return tensorWalk({
      title: "Q-learning, one update at a time",
      lede: "The whole algorithm is one line of code applied to one cube at a time. The table is drawn with states running left to right like the corridor: the back row is ←, the front row is →. Watch value leak backward from the goal.",
      code: `Q = np.zeros((6, 2))                     # states x actions
s, a, r, s2 = env.step(...)              # one transition
target = r + gamma * Q[s2].max()         # reward + best guess about the future
Q[s, a] += alpha * (target - Q[s, a])    # move halfway toward it
policy = Q.argmax(axis=1)                # the taller bar in each row`,
      tensors: T,
      steps,
    });
  },

  // -------------------------------------------------------------------------------------------
  precision() {
    const R = rng(13), W = M.rand(4, 6, R, 0.8), mx = Math.max(...W.flat().map(Math.abs)), s = mx / 7;
    const q = W.map((r) => r.map((v) => Math.round(v / s))), dq = r2(q.map((r) => r.map((v) => v * s))), err = M.round(W.map((r, i) => r.map((v, j) => v - dq[i][j])), 3);
    const T = { W: dims(W, "out, in", { label: "W (fp32)" }), q: dims(q, "out, in", { label: "int4 codes", scale: 7 }), dq: dims(dq, "out, in", { label: "W back", scale: mx }), err: dims(err, "out, in", { label: "error", scale: mx }) };
    return tensorWalk({
      title: "Quantising to 4 bits, cube by cube",
      lede: "Store each weight as a small integer plus one shared scale. Here int4: 15 levels, from −7 to 7.",
      code: `scale = np.abs(W).max() / 7            # one float for the whole tensor
q = np.round(W / scale).astype(np.int8) # each weight -> an integer in [-7, 7]
W_back = q * scale                       # what the model actually computes with
error = W - W_back                       # at most scale / 2 per weight`,
      tensors: T,
      steps: [
        { title: "Full-precision weights", lines: [1], show: ["W"], text: `<p>24 weights at 32 bits each = 96 bytes. The largest magnitude is ${fmt(mx)}, so the scale is ${fmt(mx)} / 7 = <b>${fmt(s, 3)}</b>.</p>` },
        { title: "Round to integers", lines: [2], show: ["W", "→", "q"], op: op.cells(["W", "q"], 4, 6, (i, j) => `<b>q[${i},${j}]</b> = round(${fmt(W[i][j], 3)} / ${fmt(s, 3)}) = round(${fmt(W[i][j] / s)}) = <b>${q[i][j]}</b>`), speed: 600, text: "<p>Every weight becomes one of 15 integers. Colours look blockier because many weights now share the same value. 24 × 4 bits = 12 bytes, plus 4 for the scale.</p>" },
        { title: "Multiply back by the scale", lines: [3], show: ["q", "→", "dq"], op: op.cells(["q", "dq"], 4, 6, (i, j) => `<b>W_back[${i},${j}]</b> = ${q[i][j]} × ${fmt(s, 3)} = <b>${fmt(dq[i][j], 3)}</b>  (was ${fmt(W[i][j], 3)})`), speed: 600, text: "<p>At run time the integers are scaled back up (often inside the matmul kernel). The result is close to W, but not equal.</p>" },
        { title: "What was lost", lines: [4], show: ["W", "−", "dq", "=", "err"], op: op.cells(["W", "dq", "err"], 4, 6, (i, j) => `<b>error[${i},${j}]</b> = ${fmt(W[i][j], 3)} − ${fmt(dq[i][j], 3)} = <b>${fmt(err[i][j], 3)}</b>  (limit ±${fmt(s / 2, 3)})`), speed: 600, text: "<p>Errors are small and look like noise, drawn on the same colour scale as W. One outlier weight would blow up the scale and the error for everyone, which is why real methods use a separate scale per row or per small group of 32 to 128 weights.</p>" },
      ],
    });
  },

  // -------------------------------------------------------------------------------------------
  kernels() {
    const R = rng(17), N = 8, A = M.rand(N, N, R, 0.7), B = M.rand(N, N, R, 0.7), C = r2(M.mm(A, B));
    const T = { A: dims(A, "M, K", { label: "A" }), B: dims(B, "K, N", { label: "B" }), C: dims(C, "M, N", { label: "C" }) };
    const tile = (r0, c0) => { const k = []; for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) k.push(`${r0 + i},${c0 + j}`); return k; };
    const order = []; for (let ti = 0; ti < 2; ti++) for (let tj = 0; tj < 2; tj++) for (let kk = 0; kk < 2; kk++) order.push([ti, tj, kk]);
    return tensorWalk({
      title: "A tiled matmul, the way a GPU does it",
      lede: "Same maths as A @ B, different order. A GPU can't keep A and B in its fast memory, so it works on 4 × 4 tiles that fit.",
      code: `for i in range(0, M, 4):                 # each output tile...
    for j in range(0, N, 4):             # ...is one thread block
        acc = zeros(4, 4)                # lives in fast on-chip memory
        for k in range(0, K, 4):
            a = A[i:i+4, k:k+4]          # load one tile of A
            b = B[k:k+4, j:j+4]          # and one tile of B
            acc += a @ b                 # 64 multiply-adds from 32 loads
        C[i:i+4, j:j+4] = acc            # write once`,
      tensors: T,
      steps: [
        { title: "The naive way: one cell at a time", lines: [], show: ["A", "@", "B", "=", "C"], op: op.matmul(T, "A", "B", "C"), speed: 500, text: "<p>Each output cell reads a whole row of A and a whole column of B from main memory (HBM): 16 numbers for 8 multiply-adds. Memory is far slower than arithmetic, so the GPU's maths units sit idle, waiting.</p>" },
        { title: "Tiled: load once, reuse many times", lines: [4, 5, 6, 7], show: ["A", "@", "B", "=", "C"], op: op.custom(order.length, (k) => { const [ti, tj, kk] = order[k]; return { hl: { A: tile(ti * 4, kk * 4), B: tile(kk * 4, tj * 4), C: tile(ti * 4, tj * 4) }, read: `output tile (${ti}, ${tj}), pass ${kk + 1} of 2: load A tile [${ti * 4}:${ti * 4 + 4}, ${kk * 4}:${kk * 4 + 4}] and B tile [${kk * 4}:${kk * 4 + 4}, ${tj * 4}:${tj * 4 + 4}] (32 numbers), do 64 multiply-adds into the accumulator` }; }, (name, key) => { if (name !== "C") return null; const [i, j] = key.split(",").map(Number); return (Math.floor(i / 4) * 2 + Math.floor(j / 4)) * 2; }), speed: 1300, text: "<p>One output tile (a thread block) loads a tile of A and a tile of B into on-chip SRAM, multiplies them, and keeps the partial sums there. Every number loaded is used 4 times instead of once. Real kernels use 128 × 128 tiles, so each load is reused 128 times.</p>" },
        { title: "Write the result once", lines: [8], show: ["C"], text: "<p>Only when its tile is finished does C go back to main memory. FlashAttention is this same idea applied to attention: never write the big (T × T) score matrix out, keep it tile by tile on chip.</p>" },
      ],
    });
  },

  // -------------------------------------------------------------------------------------------
  gnn() {
    const names = ["Ana", "Bo", "Cy", "Dee", "Eli"], edges = [[0, 1], [0, 2], [1, 2], [2, 3], [3, 4]];
    const Adj = names.map((_, i) => names.map((_, j) => (i === j || edges.some(([a, b]) => (a === i && b === j) || (a === j && b === i)) ? 1 : 0)));
    const An = r2(Adj.map((r) => r.map((v) => v / r.reduce((a, b) => a + b, 0))));
    const R = rng(23), X = M.rand(5, 3, R, 1), W = M.rand(3, 4, R, 0.7), AX = r2(M.mm(An, X)), H = r2(M.mm(AX, W)), Hr = M.map(H, (v) => Math.max(0, v));
    const T = { Adj: dims(Adj, "node, node", { label: "A + I", kind: "prob", scale: 1, names, rowNames: names }), An: dims(An, "node, node", { label: "Â (rows sum to 1)", kind: "prob", scale: 1, names, rowNames: names }), X: dims(X, "node, features", { label: "X", rowNames: names }), AX: dims(AX, "node, features", { label: "ÂX", rowNames: names }), W: dims(W, "features, hidden", { label: "W" }), H: dims(H, "node, hidden", { label: "ÂXW", rowNames: names }), Hr: dims(Hr, "node, hidden", { label: "ReLU", rowNames: names }) };
    return tensorWalk({
      title: "Message passing, cube by cube",
      lede: `Five people: ${names.join(", ")}. Friendships: ${edges.map(([a, b]) => `${names[a]}–${names[b]}`).join(", ")}. One graph layer = average your friends, then a normal layer.`,
      code: `A = adjacency + np.eye(5)               # who is connected (plus yourself)
A_hat = A / A.sum(axis=1, keepdims=True) # each row averages
AX = A_hat @ X                           # gather: average your neighbours
H = np.maximum(AX @ W, 0)                # then an ordinary layer`,
      tensors: T,
      steps: [
        { title: "The graph as a matrix", lines: [1], show: ["Adj"], text: "<p>A 1 in row i, column j means i and j are connected. The diagonal is 1 too, so each node keeps its own features. Hover a cube to see who it connects.</p>" },
        { title: "Normalise each row", lines: [2], show: ["Adj", "→", "An"], op: op.rows(["Adj", "An"], 5, (i) => `${names[i]} has ${Adj[i].reduce((a, b) => a + b, 0)} (including self), so each gets weight ${fmt(An[i][i])}`), speed: 1300, text: "<p>Divide by the row count, so the next step averages instead of sums. Otherwise popular nodes would get huge values.</p>" },
        { title: "Gather: average your neighbours", lines: [3], show: ["An", "@", "X", "=", "AX"], op: op.matmul(T, "An", "X", "AX"), speed: 900, text: "<p>Row i of Â picks out node i's neighbours; multiplying by X averages their feature rows. After one layer each node knows about its friends; after two, about friends of friends.</p>" },
        { title: "Then an ordinary layer", lines: [4], show: ["AX", "@", "W", "=", "H"], op: op.matmul(T, "AX", "W", "H"), speed: 800, text: "<p>The same W for every node, like a convolution shares its kernel across positions. It works for any graph size, because W only touches the feature dimension.</p>" },
      ],
    });
  },

  // -------------------------------------------------------------------------------------------
  distillation() {
    const classes = ["cat", "dog", "fox", "car", "cup"];
    const Z = [[6.0, 3.6, 2.8, -1.0, -1.5], [2.0, 6.5, 3.4, -2.0, -0.5], [-1.0, -0.8, -1.5, 6.0, 1.8]];
    const sm = (A, t) => r2(M.softmax(A.map((r) => r.map((v) => v / t))));
    const P1 = sm(Z, 1), P4 = sm(Z, 4), S4 = r2(M.softmax([[3.0, 1.8, 1.0, -0.3, -0.6], [1.0, 3.5, 2.6, -0.9, -0.4], [-0.6, 0.2, -1.1, 3.8, 0.4]].map((r) => r.map((v) => v / 4))));
    const kl = P4.map((p, i) => p.reduce((s, v, j) => s + (v > 0 ? v * Math.log(v / Math.max(1e-9, S4[i][j])) : 0), 0));
    const pr = (lab) => ({ kind: "prob", scale: 1, bars: 4, names: classes, label: lab });
    const T = { Z: dims(Z, "examples, classes", { label: "teacher logits", names: classes }), P1: dims(P1, "examples, classes", pr("T = 1")), P4: dims(P4, "examples, classes", pr("T = 4")), S4: dims(S4, "examples, classes", pr("student, T = 4")) };
    return tensorWalk({
      title: "Distillation, cube by cube",
      lede: "Three images, five classes. The teacher's soft probabilities carry more than the right answer: they say which wrong answers are nearly right.",
      code: `t_logits = teacher(x)                     # (3, 5)
p_hard = softmax(t_logits)                # T = 1: nearly one-hot
p_soft = softmax(t_logits / 4)            # T = 4: the "dark knowledge"
q_soft = softmax(student(x) / 4)
loss = (4 ** 2) * kl(p_soft, q_soft)      # match the whole distribution`,
      tensors: T,
      steps: [
        { title: "The teacher's raw scores", lines: [1], show: ["Z"], text: "<p>Example 0 is a cat photo, 1 a dog, 2 a car. Orange = high score.</p>" },
        { title: "At temperature 1: almost a label", lines: [2], show: ["Z", "→", "P1"], op: op.rows(["Z", "P1"], 3, (i) => `example ${i}: ${P1[i].map((v, j) => `${classes[j]} ${fmt(v)}`).join(", ")}`), speed: 1500, text: "<p>Plain softmax puts nearly everything on one class. Training a student on this teaches little more than the label would.</p>" },
        { title: "At temperature 4: dark knowledge", lines: [3], show: ["Z", "→", "P4"], op: op.rows(["Z", "P4"], 3, (i) => `example ${i}: ${P4[i].map((v, j) => `${classes[j]} ${fmt(v)}`).join(", ")}`), speed: 1500, text: "<p>Dividing the logits by 4 before softmax flattens the bars and reveals the structure: a cat looks a bit like a dog and a fox, and not at all like a car or a cup. That's information the label never had.</p>" },
        { title: "The student matches the whole shape", lines: [4, 5], show: ["P4", "↔", "S4"], op: op.rows(["P4", "S4"], 3, (i) => `example ${i}: KL(teacher ‖ student) = <b>${fmt(kl[i], 3)}</b>`), speed: 1500, text: "<p>The loss compares the two distributions bar by bar (KL divergence), not just the top answer. It's scaled by T² so its gradients stay comparable to the normal label loss it's usually mixed with.</p>" },
      ],
    });
  },
};

/** The walkthrough section for a chapter, or null. */
export function walk(id) { return WALKS[id] ? WALKS[id]() : null; }
export const WALK_IDS = Object.keys(WALKS);
