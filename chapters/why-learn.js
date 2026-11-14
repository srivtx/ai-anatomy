import { h, rng, loop } from "../lib/core.js";
import { slider, stats, predict, code, dilemma, callout, section, prose, button, table } from "../lib/widgets.js";
import { plane } from "../lib/plot.js";
import { gauss } from "../lib/nn.js";

// Flat-size vs price, rescaled to [-1, 1]. The true rule is y = 0.8x + 0.25 plus noise.
const r = rng(7);
const DATA = Array.from({ length: 28 }, () => { const x = r() * 2 - 1; return { x, y: 0.8 * x + 0.25 + gauss(r) * 0.18 }; });
const loss = (w, b) => DATA.reduce((a, p) => a + (w * p.x + b - p.y) ** 2, 0) / DATA.length;
const grad = (w, b) => DATA.reduce(([gw, gb], p) => { const e = w * p.x + b - p.y; return [gw + (2 * e * p.x) / DATA.length, gb + (2 * e) / DATA.length]; }, [0, 0]);
// the closed-form least-squares answer, for reference
const mx = DATA.reduce((a, p) => a + p.x, 0) / DATA.length, my = DATA.reduce((a, p) => a + p.y, 0) / DATA.length;
const W_STAR = DATA.reduce((a, p) => a + (p.x - mx) * (p.y - my), 0) / DATA.reduce((a, p) => a + (p.x - mx) ** 2, 0);
const B_STAR = my - W_STAR * mx;
const BEST = loss(W_STAR, B_STAR);

export default function (root, { id }) {
  let w = -0.6, b = 0.9, lr = 0.3, steps = 0, path = [[w, b]];
  const fit = plane({ width: 420, height: 340, xr: [-1.2, 1.2], yr: [-1.4, 1.6], label: "Data points and your line" });
  const land = plane({ width: 420, height: 340, xr: [-2, 3], yr: [-1.5, 2], label: "Loss landscape over slope and intercept" });
  const st = stats([{ label: "Your loss (MSE)" }, { label: "Best possible" }, { label: "Gradient steps" }]);
  const ws = slider({ label: "Slope w", min: -2, max: 3, step: 0.01, value: w, fmt: (v) => v.toFixed(2), onInput: (v) => { w = v; path = [[w, b]]; steps = 0; draw(); } });
  const bs = slider({ label: "Intercept b", min: -1.5, max: 2, step: 0.01, value: b, fmt: (v) => v.toFixed(2), onInput: (v) => { b = v; path = [[w, b]]; steps = 0; draw(); } });
  const lrs = slider({ label: "Learning rate", min: 0.02, max: 1.1, step: 0.02, value: lr, fmt: (v) => v.toFixed(2), onInput: (v) => (lr = v), hint: "Above ~1 it overshoots" });
  const L = (x) => Math.log(1 + x);
  const maxL = L(loss(-2, 2) * 1.2);

  function draw() {
    fit.clear(); fit.axes();
    for (const p of DATA) fit.line(fit.X(p.x), fit.Y(p.y), fit.X(p.x), fit.Y(w * p.x + b), { color: fit.colors.c2, lw: 1.2, dash: [3, 3] });
    fit.curve((x) => w * x + b, { color: fit.colors.accent, lw: 2.5 });
    fit.scatter(DATA.map((p) => ({ x: [p.x, p.y] })), { color: fit.colors.ink, r: 3.5 });
    land.clear();
    land.field((ww, bb) => L(loss(ww, bb)) / maxL, { res: 70, a: land.colors.surface, mid: land.colors.c1, b: land.colors.c6, alpha: 0.55 });
    land.dot(land.X(W_STAR), land.Y(B_STAR), 6, land.colors.good);
    const ctx = land.ctx;
    ctx.beginPath();
    path.forEach(([pw, pb], i) => (i ? ctx.lineTo(land.X(pw), land.Y(pb)) : ctx.moveTo(land.X(pw), land.Y(pb))));
    ctx.strokeStyle = land.colors.ink; ctx.lineWidth = 2; ctx.stroke();
    land.dot(land.X(w), land.Y(b), 6, land.colors.accent);
    land.text("best", land.X(W_STAR) + 9, land.Y(B_STAR), { size: 11, color: land.colors.ink });
    const l = loss(w, b);
    st.set(0, isFinite(l) && l < 1e6 ? l.toFixed(4) : "diverged", l < BEST * 1.05 ? "good" : l > 1 ? "bad" : "");
    st.set(1, BEST.toFixed(4));
    st.set(2, String(steps));
  }
  // slow it down to ~12 steps per second so you can follow it
  let acc = 0;
  const slow = loop((dt) => { acc += dt; if (acc > 0.08) { acc = 0; tick(); } }, fit.el);
  function tick() {
    const [gw, gb] = grad(w, b);
    w -= lr * gw; b -= lr * gb; steps++;
    path.push([w, b]);
    if (!isFinite(w) || Math.abs(w) > 50 || steps > 200 || Math.hypot(gw, gb) < 2e-4) slow.stop();
    ws.value = Math.max(-2, Math.min(3, w)); bs.value = Math.max(-1.5, Math.min(2, b));
    draw();
  }

  root.append(
    section(null, prose(`<p>Write a program that decides whether an email is spam. You start with rules: <i>if it says “free money”, it's spam.</i> Then spammers write “fr3e m0ney”. You add a rule. They adapt. A thousand rules later the program is unmaintainable and still wrong.</p>
<p>Machine learning flips the job around. Instead of writing the rules, you write a <b>family of possible programs</b> with knobs on them, collect <b>examples</b> of the right answer, and let an algorithm turn the knobs until the program agrees with the examples. Everything in this course, up to GPT-scale models, is that one idea with bigger families and better knob-turning.</p>`)),
    section("Fit it yourself, then let the maths do it",
      prose("<p>Each dot is a flat: its size along x, its price along y (both rescaled). Our family of programs is a straight line, <code>price = w · size + b</code>: two knobs. The dashed red segments are the errors. The <b>loss</b> is their average squared length. Drag the sliders to make it as small as you can.</p>"),
      h("div.stage",
        h("div.stage-controls", ws, bs, lrs,
          button("Run gradient descent", () => { path = [[w, b]]; steps = 0; slow.start(); }, "primary"),
          button("Scramble", () => { slow.stop(); w = -1.5 + Math.random() * 4; b = -1 + Math.random() * 2.5; ws.value = w; bs.value = b; path = [[w, b]]; steps = 0; draw(); })),
        h("div.stage-body", st.el,
          h("div.lab",
            h("div", h("div.canvas-box", fit.el), h("p.canvas-cap", "Your line and its errors")),
            h("div", h("div.canvas-box", land.el), h("p.canvas-cap", "Every possible (w, b) coloured by its loss. Darker is worse; green is the best line.")))),
      ),
      callout("tip", "Things to try", "<ul style='margin:0;padding-left:18px'><li>Get within 5% of the best loss by hand. Then press Run and count how many steps the algorithm needs.</li><li>Set the learning rate above 1.0 and run. The path zig-zags across the valley and flies off: that's divergence.</li><li>Look at the landscape: it's a bowl. For a straight line the loss is always a bowl, so rolling downhill always finds the bottom. Neural networks are not this kind.</li></ul>"),
      predict({
        chapter: id,
        question: "Your line sits above every single point. Which way does the gradient push the intercept b?",
        options: ["Up, to make the line steeper", "Down, because every error is positive", "It depends on the slope", "Nowhere: the gradient only changes w"],
        answer: 1,
        explain: "<p>The gradient of the loss with respect to b is the average of 2·(prediction − truth). If the line is above every point, every term is positive, so the gradient is positive and the update <code>b ← b − lr · gradient</code> moves b down. Each knob gets pushed in proportion to how much it's to blame.</p>",
      }),
    ),
    section("The four ingredients",
      prose("<p>Every learning system you'll meet, from this line to a trillion-parameter model, has the same four parts. Only their sizes change.</p>"),
      table(["Ingredient", "Here", "In a GPT"], [
        ["Data", "28 flats and their prices", "Trillions of tokens of text"],
        ["Model (family of programs)", "y = w·x + b, 2 parameters", "A transformer, billions of parameters"],
        ["Loss (how wrong)", "Mean squared error", "Cross-entropy of the next token"],
        ["Optimizer (how to improve)", "Gradient descent", "AdamW with a warmup and cosine schedule"],
      ]),
      dilemma("Should you write rules or learn them?", [
        { name: "Hand-written rules", pros: ["Predictable and explainable", "No data needed", "Easy to fix one case"], cons: ["Breaks on cases you didn't foresee", "Doesn't scale to perception or language"] },
        { name: "Learned model", pros: ["Handles messy inputs: images, speech, text", "Improves with more data"], cons: ["Needs labelled examples", "Can be confidently wrong", "Harder to explain"] },
      ], "Real products mix both: a model for the fuzzy part, rules as guardrails around it. And when the rules are easy to write, write the rules."),
    ),
    section("In code",
      code({
        "NumPy": `import numpy as np

x = np.random.uniform(-1, 1, 28)
y = 0.8 * x + 0.25 + np.random.normal(0, 0.18, 28)

w, b, lr = 0.0, 0.0, 0.3
for step in range(200):
    pred = w * x + b
    err = pred - y
    loss = (err ** 2).mean()
    dw = 2 * (err * x).mean()       # d loss / d w
    db = 2 * err.mean()             # d loss / d b
    w, b = w - lr * dw, b - lr * db
print(w, b, loss)                   # close to 0.8, 0.25`,
        "PyTorch": `import torch
x = torch.rand(28) * 2 - 1
y = 0.8 * x + 0.25 + 0.18 * torch.randn(28)

w = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
opt = torch.optim.SGD([w, b], lr=0.3)
for step in range(200):
    loss = ((w * x + b - y) ** 2).mean()
    opt.zero_grad()
    loss.backward()          # autograd computes dw, db for you
    opt.step()`,
      }, "PyTorch's autograd does the derivative by itself. You'll build your own version in the backpropagation chapter."),
    ),
  );
  fit.paint(draw);
  land.paint(draw);
}
