import { h, s } from "../lib/core.js";
import { walk } from "../lib/walks.js";
import { slider, stats, predict, code, callout, section, prose, stepper, table } from "../lib/widgets.js";

// One neuron with a squared-error loss, as a graph of primitive operations:
// L = (tanh(w1·x1 + w2·x2 + b) − y)²
const LAYOUT = {
  x1: [20, 30], w1: [20, 100], x2: [20, 190], w2: [20, 260], b: [20, 340], y: [430, 340],
  m1: [190, 65], m2: [190, 225], z: [330, 160], t: [470, 160], e: [610, 250], L: [750, 250],
};
const EDGES = [["x1", "m1"], ["w1", "m1"], ["x2", "m2"], ["w2", "m2"], ["m1", "z"], ["m2", "z"], ["b", "z"], ["z", "t"], ["t", "e"], ["y", "e"], ["e", "L"]];
const OPS = { m1: "×", m2: "×", z: "+", t: "tanh", e: "−", L: "²" };
const NAMES = { m1: "x1·w1", m2: "x2·w2", z: "z = sum + b", t: "a = tanh(z)", e: "a − y", L: "loss" };

function compute(v) {
  const n = { ...v };
  n.m1 = v.x1 * v.w1; n.m2 = v.x2 * v.w2; n.z = n.m1 + n.m2 + v.b; n.t = Math.tanh(n.z); n.e = n.t - v.y; n.L = n.e * n.e;
  const g = { L: 1 };
  g.e = 2 * n.e * g.L;
  g.t = g.e; g.y = -g.e;
  g.z = (1 - n.t * n.t) * g.t;
  g.m1 = g.z; g.m2 = g.z; g.b = g.z;
  g.x1 = v.w1 * g.m1; g.w1 = v.x1 * g.m1; g.x2 = v.w2 * g.m2; g.w2 = v.x2 * g.m2;
  return { n, g };
}

export default function (root, { id }) {
  const v = { x1: 2, w1: -0.5, x2: 1, w2: 0.8, b: 0.3, y: 0.8 };
  let reveal = { values: new Set(Object.keys(LAYOUT)), grads: new Set(Object.keys(LAYOUT)), active: null };
  const svg = s("svg", { viewBox: "0 0 860 400", role: "img", "aria-label": "Computational graph of one neuron with a squared-error loss" });
  const st = stats([{ label: "Loss" }, { label: "dL/dw1 (backprop)" }, { label: "dL/dw1 (nudge test)" }]);

  function render() {
    const { n, g } = compute(v);
    svg.replaceChildren();
    for (const [a, b] of EDGES) {
      const [x1, y1] = LAYOUT[a], [x2, y2] = LAYOUT[b];
      svg.append(s("path", { d: `M${x1 + 100},${y1 + 22} C${x1 + 140},${y1 + 22} ${x2 - 40},${y2 + 22} ${x2},${y2 + 22}`, class: "wire" }));
    }
    for (const [k, [x, y]] of Object.entries(LAYOUT)) {
      const leaf = !OPS[k], on = reveal.active === k;
      svg.append(s("rect", { x, y, width: 100, height: 56, rx: 10, class: `node-box${on ? " hot" : ""}`, style: on ? "stroke-width:3" : "" }));
      svg.append(s("text", { x: x + 10, y: y + 17, class: "node-sub", text: leaf ? k : `${OPS[k]}  ${NAMES[k]}` }));
      svg.append(s("text", { x: x + 10, y: y + 35, class: "node-label", text: reveal.values.has(k) ? n[k].toFixed(3) : "?" }));
      if (reveal.grads.has(k)) svg.append(s("text", { x: x + 10, y: y + 50, class: "node-sub", style: "fill:var(--warm);font-weight:600", text: `grad ${g[k].toFixed(3)}` }));
    }
    const eps = 1e-5, nudge = (compute({ ...v, w1: v.w1 + eps }).n.L - compute({ ...v, w1: v.w1 - eps }).n.L) / (2 * eps);
    st.set(0, n.L.toFixed(4)); st.set(1, g.w1.toFixed(5)); st.set(2, nudge.toFixed(5), Math.abs(nudge - g.w1) < 1e-6 ? "good" : "bad");
  }
  const sliders = ["x1", "w1", "x2", "w2", "b", "y"].map((k) => slider({ label: k, min: -2, max: 2, step: 0.05, value: v[k], fmt: (x) => x.toFixed(2), onInput: (x) => { v[k] = x; render(); } }));

  const FWD = ["m1", "m2", "z", "t", "e", "L"], BWD = ["L", "e", "t", "z", "m1", "w1"];
  const leaves = ["x1", "w1", "x2", "w2", "b", "y"];
  const set = (vals, grads, active) => { reveal = { values: new Set([...leaves, ...vals]), grads: new Set(grads), active }; render(); };
  const steps = [
    { title: "The graph", html: "<p>Every computation, however big, is a graph of tiny operations: multiply, add, tanh, square. Leaves on the left are inputs (x) and parameters (w, b). We want <b>how the loss changes when each parameter changes</b>: the gradient.</p>", onEnter: () => set([], [], null) },
    ...FWD.map((k, i) => ({ title: `Forward ${i + 1}: ${NAMES[k]}`, html: `<p>Compute <b>${NAMES[k]}</b> from its inputs, and <b>remember the inputs</b>: the backward pass will need them.</p>`, onEnter: () => set(FWD.slice(0, i + 1), [], k) })),
    { title: "Backward 1: start at the loss", html: "<p>dL/dL = 1. The gradient flows backwards from here.</p>", onEnter: () => set(FWD, ["L"], "L") },
    { title: "Backward 2: through the square", html: "<p>L = e², so dL/de = 2e. Multiply by the gradient arriving from above (1).</p>", onEnter: () => set(FWD, ["L", "e"], "e") },
    { title: "Backward 3: through the subtraction", html: "<p>e = a − y, so de/da = 1: the gradient passes through unchanged (and y gets −1 times it).</p>", onEnter: () => set(FWD, ["L", "e", "t", "y"], "t") },
    { title: "Backward 4: through tanh", html: "<p>d tanh(z)/dz = 1 − tanh(z)². We stored a = tanh(z) on the way forward, so this is cheap. If |z| is large this factor is near 0: the neuron is <b>saturated</b> and the gradient vanishes.</p>", onEnter: () => set(FWD, ["L", "e", "t", "y", "z"], "z") },
    { title: "Backward 5: through the sum", html: "<p>An addition copies its incoming gradient to every input. So b gets exactly dL/dz.</p>", onEnter: () => set(FWD, ["L", "e", "t", "y", "z", "m1", "m2", "b"], "m1") },
    { title: "Backward 6: through the multiply", html: "<p>m = x·w, so dm/dw = x: each input's gradient is the gradient above times <b>the other input</b>. That's why the forward pass had to remember x. Done: every parameter has a gradient, in one backward sweep, for the price of about two forward passes.</p>", onEnter: () => set(FWD, Object.keys(LAYOUT), "w1") },
  ];

  root.append(
    section(null, prose(`<p>Training needs the gradient of the loss with respect to every parameter. A GPT has billions. Nudging each one separately to see what happens (the <b>nudge test</b>) would take billions of forward passes per step.</p>
<p><b>Backpropagation</b> gets all of them in one backward sweep, using nothing but the chain rule: if L depends on a, and a depends on z, then dL/dz = dL/da · da/dz. Each operation only needs to know its own <b>local derivative</b>. Multiply local derivatives along the path and you have the global one.</p>`)),
    section("Step through one neuron",
      h("div.stage",
        h("div.stage-controls", sliders),
        h("div.stage-body", st.el, h("div", { style: { overflowX: "auto" } }, h("div", { style: { minWidth: "640px" } }, svg)),
          stepper(steps))),
      callout("tip", "Things to try", "<ul style='margin:0;padding-left:18px'><li>Compare the two gradient readouts: backprop and the nudge test agree to five decimals. That check is called <b>gradient checking</b>, and it's how you debug a hand-written backward pass.</li><li>Push w1 and x1 so z is large (say 3+). The tanh gradient collapses towards zero and so does every gradient behind it.</li><li>Set y equal to a: the loss and every gradient go to zero. Nothing left to learn.</li></ul>"),
      predict({
        chapter: id,
        question: "A network has 1 billion parameters. Roughly how much compute does one backward pass cost compared to one forward pass?",
        options: ["About the same as one forward pass per parameter: a billion times more", "About 2× a forward pass", "About 1/1000 of a forward pass", "It depends on the loss function"],
        answer: 1,
        explain: "<p>Reverse-mode differentiation costs a small constant multiple of the forward pass, about 2× for matrix multiplies (one product for the input gradient, one for the weight gradient). That's why the standard estimate for training is <b>6 × parameters × tokens</b> FLOPs: 2 for forward, 4 for backward.</p>",
      }),
    ),
    section("Local rules you'll reuse forever",
      table(["Operation", "Forward", "Backward (given upstream gradient g)"], [
        ["Add", "z = a + b", "a gets g, b gets g"],
        ["Multiply", "z = a · b", "a gets g·b, b gets g·a"],
        ["Matrix multiply", "Y = X W", "dX = g Wᵀ, dW = Xᵀ g"],
        ["tanh", "a = tanh(z)", "dz = g · (1 − a²)"],
        ["ReLU", "a = max(0, z)", "dz = g if z > 0 else 0"],
        ["Softmax + cross-entropy", "p = softmax(z), L = −log p_y", "dz = p − onehot(y)"],
      ]),
      prose("<p>When a value feeds into several places, its gradients from each <b>add up</b>. Those two facts, the chain rule and gradient accumulation, are all of autograd. PyTorch, JAX and TensorFlow build this graph as your code runs and walk it backwards when you call <code>.backward()</code>.</p>"),
    ),
    walk("backprop"),
    section("Build it: micrograd in 40 lines",
      code({
        Python: `import math

class Value:
    def __init__(self, data, parents=(), backward=lambda: None):
        self.data, self.grad = data, 0.0
        self._parents, self._backward = parents, backward

    def __add__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        out = Value(self.data + o.data, (self, o))
        def bw():
            self.grad += out.grad          # add passes the gradient through
            o.grad += out.grad
        out._backward = bw
        return out

    def __mul__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        out = Value(self.data * o.data, (self, o))
        def bw():
            self.grad += o.data * out.grad # times the *other* input
            o.grad += self.data * out.grad
        out._backward = bw
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,))
        def bw(): self.grad += (1 - t * t) * out.grad
        out._backward = bw
        return out

    def backward(self):
        order, seen = [], set()
        def visit(v):                      # topological sort
            if v not in seen:
                seen.add(v)
                for p in v._parents: visit(p)
                order.append(v)
        visit(self)
        self.grad = 1.0
        for v in reversed(order): v._backward()

x1, w1, x2, w2, b = Value(2), Value(-0.5), Value(1), Value(0.8), Value(0.3)
a = (x1 * w1 + x2 * w2 + b).tanh()
loss = (a + (-0.8)) * (a + (-0.8))
loss.backward()
print(w1.grad)`,
      }, "This is a trimmed version of Andrej Karpathy's micrograd. Project 02 builds the full thing and trains an MLP with it."),
    ),
  );
  render();
}
