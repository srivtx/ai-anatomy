# %% [markdown]
# # 02 · Autograd from scratch
# A scalar `Value` class that records the computation graph and runs backpropagation, then a small MLP
# built on it that solves XOR and two moons. Chapter: **Backpropagation**.
#
# **You're done when** your gradients match PyTorch on a random expression and your MLP gets 100% on XOR.

# %%
import math, os, random
import numpy as np, matplotlib.pyplot as plt
QUICK = os.environ.get("QUICK") == "1"
random.seed(0)

class Value:
    def __init__(self, data, parents=(), op=""):
        self.data, self.grad, self._parents, self._op = float(data), 0.0, parents, op
        self._backward = lambda: None
    def __add__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        out = Value(self.data + o.data, (self, o), "+")
        def bw(): self.grad += out.grad; o.grad += out.grad
        out._backward = bw; return out
    def __mul__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        out = Value(self.data * o.data, (self, o), "*")
        def bw(): self.grad += o.data * out.grad; o.grad += self.data * out.grad
        out._backward = bw; return out
    def __pow__(self, k):
        out = Value(self.data ** k, (self,), f"**{k}")
        def bw(): self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = bw; return out
    def tanh(self):
        t = math.tanh(self.data); out = Value(t, (self,), "tanh")
        def bw(): self.grad += (1 - t * t) * out.grad
        out._backward = bw; return out
    def exp(self):
        e = math.exp(self.data); out = Value(e, (self,), "exp")
        def bw(): self.grad += e * out.grad
        out._backward = bw; return out
    __radd__ = __add__; __rmul__ = __mul__
    def __neg__(self): return self * -1
    def __sub__(self, o): return self + (-o)
    def __truediv__(self, o): return self * (o ** -1 if isinstance(o, Value) else 1 / o)
    def backward(self):
        order, seen = [], set()
        def visit(v):                      # topological sort: children before parents
            if v not in seen:
                seen.add(v); [visit(p) for p in v._parents]; order.append(v)
        visit(self); self.grad = 1.0
        for v in reversed(order): v._backward()

# %% [markdown]
# ## Check against PyTorch

# %%
import torch
a, b, c = Value(1.5), Value(-0.7), Value(2.0)
f = ((a * b + c).tanh() * a + (b ** 2) / c).exp()
f.backward()
ta, tb, tc = (torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (1.5, -0.7, 2.0))
tf = ((ta * tb + tc).tanh() * ta + tb ** 2 / tc).exp(); tf.backward()
for name, mine, ref in [("a", a.grad, ta.grad.item()), ("b", b.grad, tb.grad.item()), ("c", c.grad, tc.grad.item())]:
    print(f"d f/d{name}: mine {mine:.8f}  torch {ref:.8f}")
    assert abs(mine - ref) < 1e-6

# %% [markdown]
# ## An MLP on top

# %%
class Neuron:
    def __init__(self, n, act=True):
        self.w = [Value(random.uniform(-1, 1) / math.sqrt(n)) for _ in range(n)]; self.b = Value(0.0); self.act = act
    def __call__(self, x):
        s = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return s.tanh() if self.act else s
    def params(self): return self.w + [self.b]

class MLP:
    def __init__(self, sizes):
        self.layers = [[Neuron(sizes[i], act=i < len(sizes) - 2) for _ in range(sizes[i + 1])] for i in range(len(sizes) - 1)]
    def __call__(self, x):
        for layer in self.layers: x = [n(x) for n in layer]
        return x[0]
    def params(self): return [p for layer in self.layers for n in layer for p in n.params()]

def train(model, X, Y, steps, lr=0.1):
    for step in range(steps):
        # hinge loss on labels in {-1, +1}
        losses = [(1 + -y * model(x)) for x, y in zip(X, Y)]
        loss = sum((l if l.data > 0 else l * 0 for l in losses), Value(0)) / len(X)
        for p in model.params(): p.grad = 0.0
        loss.backward()
        for p in model.params(): p.data -= lr * p.grad
    acc = np.mean([(model(x).data > 0) == (y > 0) for x, y in zip(X, Y)])
    return loss.data, acc

xor_X, xor_Y = [[0, 0], [0, 1], [1, 0], [1, 1]], [-1, 1, 1, -1]
loss, acc = train(MLP([2, 8, 1]), [[2 * a - 1, 2 * b - 1] for a, b in xor_X], xor_Y, 300)
print(f"XOR: loss {loss:.3f}, accuracy {acc:.0%}")
assert acc == 1.0

# %%
# two moons
n = 60 if QUICK else 100
t = np.random.default_rng(0).uniform(0, np.pi, n)
moons = np.concatenate([np.stack([np.cos(t[: n // 2]), np.sin(t[: n // 2])], 1), np.stack([1 - np.cos(t[n // 2:]), 0.5 - np.sin(t[n // 2:])], 1)])
moons += np.random.default_rng(1).normal(scale=0.08, size=moons.shape)
labels = [1] * (n // 2) + [-1] * (n - n // 2)
model = MLP([2, 16, 16, 1])
loss, acc = train(model, moons.tolist(), labels, 60 if QUICK else 150, lr=0.2)
print(f"moons: loss {loss:.3f}, accuracy {acc:.0%}")
xx, yy = np.meshgrid(np.linspace(-1.5, 2.5, 40), np.linspace(-1, 1.5, 40))
zz = np.array([[model([a, b]).data for a, b in zip(r1, r2)] for r1, r2 in zip(xx, yy)])
plt.contourf(xx, yy, zz > 0, alpha=0.3); plt.scatter(*moons.T, c=labels, cmap="coolwarm", s=12); plt.title("Decision boundary from your own autograd"); plt.show()
