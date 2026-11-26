import { h, rng, loop } from "../lib/core.js";
import { goal, whyChain, sticker } from "../lib/why.js";
import { walk } from "../lib/walks.js";
import { slider, segmented, stats, predict, code, callout, section, prose, table, button } from "../lib/widgets.js";
import { plane } from "../lib/plot.js";
import { gauss } from "../lib/nn.js";

// Target distributions: each a set of points with a class label (for guidance).
function shape(kind, n = 220, seed = 2) {
  const r = rng(seed), out = [];
  for (let i = 0; i < n; i++) {
    const c = i % 2;
    if (kind === "smile") {
      if (c === 0) { const th = Math.PI * (1.15 + 0.7 * r()); out.push({ x: [0.62 * Math.cos(th), 0.62 * Math.sin(th) + 0.1], c }); }
      else { const e = r() < 0.5 ? -1 : 1, th = r() * 2 * Math.PI; out.push({ x: [e * 0.33 + 0.09 * Math.cos(th), 0.35 + 0.09 * Math.sin(th)], c }); }
    } else if (kind === "spiral") {
      const t = 0.15 + 0.85 * r(), th = t * 3 * Math.PI + c * Math.PI; out.push({ x: [0.8 * t * Math.cos(th), 0.8 * t * Math.sin(th)], c });
    } else { // two rings
      const th = r() * 2 * Math.PI, rad = c ? 0.75 : 0.3; out.push({ x: [rad * Math.cos(th), rad * Math.sin(th)], c });
    }
  }
  return out;
}
const S2 = 0.03 ** 2; // each data point is a tiny Gaussian blob, so the model isn't a pure lookup table
const abar = (t) => Math.cos(((t + 0.008) / 1.008) * (Math.PI / 2)) ** 2; // cosine noise schedule

/** E[x0 | x_t] for a mixture of tiny Gaussians at the data points, with x_t = a·x0 + b·ε. Optionally only class c. */
function denoise(x, a, b, data, cls) {
  const v = a * a * S2 + b * b, gain = (a * S2) / v;
  let wsum = 0, mx = 0, my = 0, best = -Infinity;
  const logw = [];
  for (const p of data) { if (cls != null && p.c !== cls) { logw.push(-Infinity); continue; } const dx = x[0] - a * p.x[0], dy = x[1] - a * p.x[1], l = -(dx * dx + dy * dy) / (2 * v); logw.push(l); if (l > best) best = l; }
  data.forEach((p, i) => { if (logw[i] === -Infinity) return; const w = Math.exp(logw[i] - best); wsum += w; mx += w * (p.x[0] + gain * (x[0] - a * p.x[0])); my += w * (p.x[1] + gain * (x[1] - a * p.x[1])); });
  return [mx / wsum, my / wsum];
}

export default function (root, { id }) {
  const o = { kind: "smile", method: "ddim", steps: 25, guide: 0, cls: 1, tView: 0.5 };
  let data = shape(o.kind), parts = [], k = 0, paths = [], r = rng(5);
  const fwd = plane({ width: 300, height: 300, xr: [-1.5, 1.5], yr: [-1.5, 1.5], label: "The data after adding noise up to time t" });
  const pl = plane({ width: 420, height: 420, xr: [-1.5, 1.5], yr: [-1.5, 1.5], label: "Samples moving from noise to data" });
  const st = stats([{ label: "Step" }, { label: "Network evaluations" }, { label: "Avg distance to the data" }]);

  function drawFwd() {
    const t = o.tView, a = Math.sqrt(abar(t)), b = Math.sqrt(1 - abar(t)), rr = rng(9);
    fwd.clear(); fwd.axes();
    for (const p of data) fwd.dot(fwd.X(a * p.x[0] + b * gauss(rr)), fwd.Y(a * p.x[1] + b * gauss(rr)), 2.5, p.c ? fwd.colors.c1 : fwd.colors.c2);
    fwd.text(`t = ${t.toFixed(2)}   signal ${(a * 100).toFixed(0)}%`, 8, 14, { size: 12, color: fwd.colors["ink-2"] });
  }
  function start() {
    r = rng(Math.floor(Math.random() * 1e6));
    parts = Array.from({ length: 360 }, () => [gauss(r), gauss(r)]);
    paths = parts.slice(0, 12).map((p) => [[...p]]);
    k = 0; lp.start();
  }
  function x0hat(x, a, b) {
    const u = denoise(x, a, b, data, null);
    if (!o.guide) return u;
    const c = denoise(x, a, b, data, o.cls); // classifier-free guidance: push away from unconditional towards conditional
    return [u[0] + (1 + o.guide) * (c[0] - u[0]), u[1] + (1 + o.guide) * (c[1] - u[1])];
  }
  function stepAll() {
    const N = o.steps;
    if (k >= N) return false;
    const t = 1 - k / N, t2 = 1 - (k + 1) / N;
    parts = parts.map((x, i) => {
      let nx;
      if (o.method === "flow") {
        // rectified flow: x_t = (1 − t)·x0 + t·ε; velocity v = ε − x0, integrate from t = 1 to 0
        const tt = Math.max(t, 1e-4), xh = x0hat(x, 1 - tt, tt), eh = [(x[0] - (1 - tt) * xh[0]) / tt, (x[1] - (1 - tt) * xh[1]) / tt];
        nx = [x[0] - (t - t2) * (eh[0] - xh[0]), x[1] - (t - t2) * (eh[1] - xh[1])];
      } else {
        const A = abar(t), A2 = t2 <= 0 ? 1 : abar(t2), a = Math.sqrt(A), b = Math.sqrt(1 - A);
        const xh = x0hat(x, a, b), eh = [(x[0] - a * xh[0]) / b, (x[1] - a * xh[1]) / b];
        const sig = o.method === "ddpm" ? Math.sqrt(((1 - A2) / (1 - A)) * (1 - A / A2)) : 0; // DDPM adds fresh noise; DDIM is deterministic
        const c = Math.sqrt(Math.max(0, 1 - A2 - sig * sig));
        nx = [Math.sqrt(A2) * xh[0] + c * eh[0] + sig * gauss(r), Math.sqrt(A2) * xh[1] + c * eh[1] + sig * gauss(r)];
      }
      if (i < 12) paths[i].push(nx);
      return nx;
    });
    k++;
    return true;
  }
  function draw() {
    pl.clear(); pl.axes();
    for (const p of data) pl.dot(pl.X(p.x[0]), pl.Y(p.x[1]), 2.2, parts.length ? pl.colors.line : pl.colors["ink-3"]);
    const ctx = pl.ctx;
    if (!parts.length) { // before the first run: show the target and say what will happen
      ctx.save(); ctx.font = "500 13px " + getComputedStyle(document.body).fontFamily; ctx.textAlign = "center"; ctx.fillStyle = pl.colors["ink-2"];
      ctx.fillText("The data's shape. Press Generate: noise will condense onto it.", pl.W / 2, pl.H - 18); ctx.restore();
    }
    for (const path of paths) { ctx.beginPath(); path.forEach(([x, y], i) => (i ? ctx.lineTo(pl.X(x), pl.Y(y)) : ctx.moveTo(pl.X(x), pl.Y(y)))); ctx.strokeStyle = pl.colors.c4; ctx.lineWidth = 1.2; ctx.stroke(); }
    for (const x of parts) pl.dot(pl.X(x[0]), pl.Y(x[1]), 2.6, pl.colors.accent);
    st.set(0, `${k} / ${o.steps}`); st.set(1, String(k * (o.guide ? 2 : 1)));
    if (parts.length) { const d = parts.reduce((acc, x) => acc + Math.sqrt(Math.min(...data.map((p) => (p.x[0] - x[0]) ** 2 + (p.x[1] - x[1]) ** 2))), 0) / parts.length; st.set(2, d.toFixed(3), d < 0.05 ? "good" : d > 0.15 ? "bad" : "warn"); }
  }
  let acc = 0;
  const lp = loop((dt) => { acc += dt; if (acc > Math.max(0.02, 1.2 / o.steps)) { acc = 0; if (!stepAll()) lp.stop(); draw(); } }, pl.el);

  root.append(
    section(null, goal("<p>We want to <b>draw new samples</b> that look like real images. Making one in a single shot is hard. Destroying an image with noise is easy, so learn to undo the noise a little at a time.</p>", ["a forward process that ends in pure noise", "a network that guesses the noise in a noisy image", "a loop that removes it step by step"]),
      prose(`<p>How do you generate a new image? <b>Diffusion models</b> learn to do the reverse of something easy. Destroying data is easy: add a little Gaussian noise, then a little more, until only static is left. A diffusion model learns to undo one small step of that. Start from pure static, undo step after step, and an image condenses out of the noise. This powers Stable Diffusion, DALL·E, Imagen, Sora and most image and video generators.</p>`),
      h("div.vx-row", sticker("noise", { title: "Noise to picture", caption: "heavy noise, half noise, clean: the model learns each small step back", px: 4 }))),
    section("Forward: data dissolves into noise",
      h("div.lab",
        h("div.canvas-box", fwd.el),
        h("div", h("div.prose", { html: "<p>At time t the noisy point is x<sub>t</sub> = √ᾱ<sub>t</sub> · x<sub>0</sub> + √(1 − ᾱ<sub>t</sub>) · ε, where ε is fresh Gaussian noise and ᾱ falls from 1 to 0 on a cosine schedule. Because we can jump straight to any t, training is cheap: pick a data point, pick t, add noise, and train a network to predict the noise (or the clean point) from x<sub>t</sub>.</p>" }),
          slider({ label: "Time t", min: 0, max: 1, step: 0.01, value: o.tView, fmt: (v) => v.toFixed(2), onInput: (v) => { o.tView = v; drawFwd(); } })))),
    section("Reverse: condense data out of noise",
      prose("<p>Here the “network” is the exact best denoiser for this dataset, which we can compute in closed form for a 2D point cloud: given a noisy point, it returns the expected clean point. Real models learn an approximation of this function with a U-Net or transformer. Press Generate: 360 noise samples (violet) travel to the shape; yellow lines trace twelve of them.</p>"),
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Shape", options: [{ value: "smile", label: "Smile" }, { value: "spiral", label: "Spiral" }, { value: "rings", label: "Two rings" }], value: o.kind, onChange: (v) => { o.kind = v; data = shape(v); parts = []; paths = []; drawFwd(); draw(); } }),
          segmented({ label: "Sampler", options: [{ value: "ddpm", label: "DDPM (stochastic)" }, { value: "ddim", label: "DDIM (deterministic)" }, { value: "flow", label: "Flow matching" }], value: o.method, onChange: (v) => (o.method = v) }),
          slider({ label: "Denoising steps", min: 2, max: 100, value: o.steps, onInput: (v) => (o.steps = v) }),
          slider({ label: "Guidance strength", min: 0, max: 4, step: 0.5, value: o.guide, fmt: (v) => (v ? v.toFixed(1) : "off"), onInput: (v) => (o.guide = v), hint: "Classifier-free guidance towards the class below" }),
          segmented({ label: "Guide towards", options: [{ value: 0, label: "Class A (orange)" }, { value: 1, label: "Class B (blue)" }], value: o.cls, onChange: (v) => (o.cls = v) }),
          button("Generate", start, "primary")),
        h("div.stage-body", st.el, h("div.canvas-box", pl.el))),
      callout("tip", "Things to try", "<ul style='margin:0;padding-left:18px'><li>DDPM with 5 steps: blurry and scattered. It needs many small noisy steps (the original paper used 1,000). DDIM with the same 5 steps is already decent: it takes the deterministic path.</li><li>Flow matching follows nearly straight paths from noise to data, which is why it gets away with few steps. Stable Diffusion 3, Flux and many video models use it.</li><li>Turn guidance up to 3 towards one class: samples land only on that part of the shape, tighter than the data. That's how “prompt adherence” is bought, at the cost of diversity.</li></ul>"),
      predict({
        chapter: id,
        question: "This denoiser is exact, so its samples land on the training points. Why doesn't a real image model just reproduce its training images?",
        options: ["It does, always", "A neural network with limited capacity and smooth inductive biases can't represent the exact memorising denoiser, so it generalises between examples", "Random seeds prevent it", "Images are too big to memorise"],
        answer: 1,
        explain: "<p>The optimal denoiser for a finite dataset only ever produces training points. Real models approximate it with a network that can't memorise billions of images exactly, and the approximation fills in the space between them, which is what creativity looks like here. With too little data or too big a model, diffusion models do regurgitate training images (Carlini et al., 2023).</p>",
      }),
    ),
    section("Generative families compared",
      table(["Family", "How it generates", "Strengths", "Weaknesses"], [
        ["Autoregressive", "One token at a time, each conditioned on the past", "Exact likelihood; scales superbly (LLMs)", "Slow sequential sampling"],
        ["VAE", "Decode a sample from a latent Gaussian", "Fast, smooth latent space", "Blurry samples"],
        ["GAN", "A generator fools a discriminator", "Sharp, one-step sampling", "Unstable training, mode collapse (next chapter)"],
        ["Normalizing flow", "Invertible transforms of Gaussian noise", "Exact likelihood", "Architecture constraints"],
        ["Diffusion / flow matching", "Iteratively denoise", "Highest quality, stable training, easy conditioning", "Many network calls per sample (distillation cuts it to 1–4)"],
        ["Masked / discrete diffusion", "Unmask tokens in parallel", "Parallel text and image generation", "Newer; catching up for text (see the frontier radar)"],
      ]),
    ),
    walk("diffusion"),
    section("Why guessing noise can paint",
      whyChain({
        claim: "A diffusion model is only trained to guess the noise that was added, yet it can generate new images.",
        steps: [
          { q: "Why does guessing noise help generate?", a: "<p>Knowing the noise in a noisy image tells you which way to move it to look more like a real image. That direction is the <b>score</b>, ∇ log p(x). Step along it, add a little fresh noise, repeat, and you walk from static to a picture.</p>" },
          { q: "Why many small steps, not one big one?", a: "<p>Undoing a tiny bit of noise is easy: the answer is close to a Gaussian around where you are. Undoing all of it at once means choosing one image out of every image that could be hiding in the static, a far harder problem with many right answers.</p>" },
          { q: "Why start from pure Gaussian noise?", a: "<p>After enough noising, every image looks like the same standard Gaussian. That's trivial to sample, so it's the common starting point for the trip back.</p>" },
        ],
        axiom: { name: "Gaussian noise forgets where it started", html: "<p>Add independent Gaussian noise again and again (shrinking the signal a little each time) and any starting distribution ends up a standard Gaussian. Heat spreading through a metal bar follows the same equation. Diffusion models run it backwards, which works because each small step back is also nearly Gaussian.</p>" },
      })),
    section("In code",
      code({
        "DDPM training": `def train_step(model, x0, alphas_bar):
    t = torch.randint(0, T, (x0.size(0),))
    a = alphas_bar[t].sqrt().view(-1, 1, 1, 1)
    b = (1 - alphas_bar[t]).sqrt().view(-1, 1, 1, 1)
    eps = torch.randn_like(x0)
    xt = a * x0 + b * eps                    # jump straight to step t
    return F.mse_loss(model(xt, t), eps)     # learn to predict the noise`,
        "Flow matching training": `def fm_step(model, x0):
    t = torch.rand(x0.size(0), 1, 1, 1)
    eps = torch.randn_like(x0)
    xt = (1 - t) * x0 + t * eps              # a straight line between data and noise
    return F.mse_loss(model(xt, t), eps - x0)   # predict the velocity along it

@torch.no_grad()
def fm_sample(model, shape, steps=20):
    x = torch.randn(shape)
    for i in range(steps, 0, -1):            # Euler from t=1 to t=0
        t = torch.full((shape[0], 1, 1, 1), i / steps)
        x = x - model(x, t) / steps
    return x`,
      }, "Project 12 trains both on 2D shapes and on MNIST."),
    ),
  );
  fwd.paint(drawFwd); pl.paint(draw);
}
