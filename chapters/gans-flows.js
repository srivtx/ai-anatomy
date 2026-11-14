import { h, rng, loop } from "../lib/core.js";
import { slider, stats, predict, code, callout, section, prose, table, button, toggle } from "../lib/widgets.js";
import { xyChart } from "../lib/plot.js";
import { MLP, gauss, sigmoid } from "../lib/nn.js";

const MODES = [-2.2, 0.3, 2.4];
const BINS = 60, LO = -4, HI = 4;
const hist = (xs) => { const b = new Float64Array(BINS); for (const x of xs) { const i = Math.floor(((x - LO) / (HI - LO)) * BINS); if (i >= 0 && i < BINS) b[i]++; } return [...b].map((v, i) => [LO + ((i + 0.5) * (HI - LO)) / BINS, v / xs.length / ((HI - LO) / BINS)]); };

export default function (root, { id }) {
  const o = { lrD: 0.002, lrG: 0.002, dSteps: 1, nonSat: true };
  let G, D, r, it, covered;
  const realSample = () => MODES[r.int(3)] + gauss(r) * 0.25;
  const chart = xyChart({ height: 280, xLabel: "x", yLabel: "density / D(x)", xDomain: [LO, HI], yDomain: [0, 1.2] });
  const st = stats([{ label: "Training steps" }, { label: "Modes covered" }, { label: "Discriminator accuracy" }]);
  function init(seed = 1) { r = rng(seed); G = new MLP([2, 24, 24, 1], { act: "tanh", seed: seed + 1 }); D = new MLP([1, 24, 24, 1], { act: "tanh", seed: seed + 2 }); it = 0; draw(); }
  const gen = () => G.forward([gauss(r), gauss(r)])[0] * 3;
  function step() {
    const B = 64;
    let correct = 0;
    for (let k = 0; k < o.dSteps; k++) {
      for (let i = 0; i < B; i++) { const z = D.forward([realSample()])[0]; correct += z > 0; D.backward([sigmoid(z) - 1]); }
      for (let i = 0; i < B; i++) { const z = D.forward([gen()])[0]; correct += z < 0; D.backward([sigmoid(z)]); }
      D.step(o.lrD, 2 * B, { b1: 0.5 });
    }
    // generator: push D(G(z)) towards "real". Non-saturating: minimise −log D(G(z)); original: minimise log(1 − D(G(z)))
    for (let i = 0; i < B; i++) {
      const zin = [gauss(r), gauss(r)], x = G.forward(zin)[0] * 3, d = D.forward([x])[0];
      const dLogit = o.nonSat ? sigmoid(d) - 1 : -sigmoid(d);
      const dx = D.backward([dLogit])[0];
      G.backward([dx * 3]);
    }
    D.zeroGrad();
    G.step(o.lrG, B, { b1: 0.5 });
    it++;
    return correct / (2 * B * o.dSteps);
  }
  let dAcc = 0;
  function draw() {
    const fake = Array.from({ length: 1500 }, gen);
    const real = Array.from({ length: 1500 }, realSample);
    const dCurve = Array.from({ length: 81 }, (_, i) => { const x = LO + (i / 80) * (HI - LO); return [x, sigmoid(D.forward([x])[0])]; });
    chart.update([
      { name: "Real data", color: "var(--ink-3)", points: hist(real) },
      { name: "Generator", color: "var(--accent)", points: hist(fake) },
      { name: "D(x): probability “real”", color: "var(--c2)", points: dCurve, dash: true },
    ], { xDomain: [LO, HI], yDomain: [0, 1.2] });
    covered = MODES.filter((m) => fake.filter((x) => Math.abs(x - m) < 0.6).length / fake.length > 0.08).length;
    st.set(0, String(it)); st.set(1, `${covered} of 3`, covered === 3 ? "good" : covered === 1 ? "bad" : "warn"); st.set(2, it ? `${Math.round(dAcc * 100)}%` : "–");
  }
  const lp = loop(() => { for (let k = 0; k < 4; k++) dAcc = step(); draw(); if (it > 3000) lp.stop(); }, chart.el);

  // a 1D flow: an invertible map applied to Gaussian noise, and the density it produces
  let s1 = 1.5, s2 = 0.6;
  const flowChart = xyChart({ height: 220, xLabel: "x", yLabel: "density", xDomain: [-4, 4], yDomain: [0, 1] });
  function drawFlow() {
    // f(z) = z + s1 · tanh(s2 · z · 2): monotonic when s1·s2·2 > −1, so the change of variables applies
    const f = (z) => z + s1 * Math.tanh(2 * s2 * z), df = (z) => 1 + 2 * s1 * s2 * (1 - Math.tanh(2 * s2 * z) ** 2);
    const pts = [];
    for (let z = -4; z <= 4; z += 0.02) { const x = f(z), dens = Math.exp(-z * z / 2) / Math.sqrt(2 * Math.PI) / Math.abs(df(z)); if (x >= -4 && x <= 4) pts.push([x, dens]); }
    flowChart.update([{ name: "p(x) = p(z) / |f′(z)|", color: "var(--c6)", points: pts }, { name: "Base: standard normal", color: "var(--ink-3)", points: Array.from({ length: 161 }, (_, i) => { const z = -4 + i * 0.05; return [z, Math.exp(-z * z / 2) / Math.sqrt(2 * Math.PI)]; }), dash: true }], { xDomain: [-4, 4], yDomain: [0, 1] });
  }

  root.append(
    section(null, prose(`<p>A <b>generative adversarial network</b> (Goodfellow et al., 2014) is a game between two networks. The <b>generator</b> turns random noise into fake samples. The <b>discriminator</b> tries to tell fakes from real data. Each improves by exploiting the other: the discriminator learns where fakes give themselves away, the generator learns to move its samples there. When it works, the generator ends up producing samples indistinguishable from the data. For five years GANs made the sharpest images (StyleGAN's faces in 2019 were the first many people mistook for photos).</p>`)),
    section("Train a GAN on three bumps",
      prose("<p>The real data has three modes. Both networks are small MLPs training right here. The dashed curve is the discriminator's belief that a value is real; the generator follows its slope uphill.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Discriminator learning rate", min: 0.0002, max: 0.01, step: 0.0002, value: o.lrD, fmt: (v) => v.toFixed(4), onInput: (v) => (o.lrD = v) }),
          slider({ label: "Generator learning rate", min: 0.0002, max: 0.01, step: 0.0002, value: o.lrG, fmt: (v) => v.toFixed(4), onInput: (v) => (o.lrG = v) }),
          slider({ label: "D steps per G step", min: 1, max: 5, value: o.dSteps, onInput: (v) => (o.dSteps = v) }),
          toggle({ label: "Non-saturating generator loss", checked: true, onChange: (v) => (o.nonSat = v) }),
          button("Train", () => lp.start(), "primary"), button("Pause", () => lp.stop()), button("Restart with a new seed", () => { lp.stop(); init(Math.floor(Math.random() * 1e5)); lp.start(); })),
        h("div.stage-body", st.el, chart.el)),
      callout("tip", "Things to try", "<ul style='margin:0;padding-left:18px'><li>Restart a few times. Sometimes the generator covers all three bumps; often it piles everything onto one or two and stays there. That's <b>mode collapse</b>: the generator found a few samples the discriminator accepts and has no reason to look elsewhere.</li><li>Turn off the non-saturating loss. Early on the discriminator easily rejects every fake, so log(1 − D) is flat and the generator gets almost no gradient. Goodfellow's paper already proposed the fix.</li><li>Make the generator learn much faster than the discriminator: training oscillates instead of settling. GANs are a game, not a minimisation, and games can cycle.</li></ul>"),
      predict({
        chapter: id,
        question: "Why did diffusion models largely replace GANs for image generation?",
        options: ["Diffusion samples are faster", "Diffusion training is a stable regression with no adversary, and it covers all modes of the data", "GANs can't make sharp images", "GANs need labels"],
        answer: 1,
        explain: "<p>A diffusion model's loss is plain mean squared error on noise, so training is stable and every mode of the data is penalised if missed. GANs can be sharper and sample in one step, which is why GAN-style losses are now often used to <i>distil</i> diffusion models into one- or few-step generators.</p>",
      }),
    ),
    section("Normalizing flows: exact densities",
      prose("<p>A <b>flow</b> pushes simple noise through an <b>invertible</b> function. Because you can run it backwards, you can compute the exact probability of any data point with the change-of-variables formula: p(x) = p(z) / |f′(z)|. Where the map stretches space, density thins; where it squeezes, density piles up. Real flows (RealNVP, Glow) stack many invertible layers whose Jacobians are cheap to compute.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Bend strength", min: -0.45, max: 3, step: 0.05, value: s1, fmt: (v) => v.toFixed(2), onInput: (v) => { s1 = v; drawFlow(); } }),
          slider({ label: "Bend sharpness", min: 0.1, max: 2, step: 0.05, value: s2, fmt: (v) => v.toFixed(2), onInput: (v) => { s2 = v; drawFlow(); } })),
        h("div.stage-body", flowChart.el, h("p.canvas-cap", "f(z) = z + strength · tanh(2 · sharpness · z). A strong bend splits one Gaussian into two bumps."))),
      table(["Family", "Exact likelihood?", "One-step sampling?", "Today"], [
        ["GAN", "No", "Yes", "Distilling diffusion models to 1–4 steps; super-resolution; some audio vocoders"],
        ["Normalizing flow", "Yes", "Yes", "Niche; their ideas live on in flow matching, which needs no invertibility"],
        ["VAE", "Lower bound", "Yes", "Latent spaces for diffusion"],
        ["Diffusion / flow matching", "Bound / via ODE", "No (unless distilled)", "Dominant for images, video and audio"],
      ]),
    ),
    section("In code",
      code({
        PyTorch: `bce = torch.nn.functional.binary_cross_entropy_with_logits
for real in loader:
    z = torch.randn(real.size(0), z_dim)
    fake = G(z)
    # discriminator: real -> 1, fake -> 0 (detach so G isn't updated here)
    d_loss = bce(D(real), torch.ones(len(real), 1)) + bce(D(fake.detach()), torch.zeros(len(real), 1))
    opt_d.zero_grad(); d_loss.backward(); opt_d.step()
    # generator, non-saturating: make D call the fakes real
    g_loss = bce(D(fake), torch.ones(len(real), 1))
    opt_g.zero_grad(); g_loss.backward(); opt_g.step()`,
      }, "Project 26 trains this on 2D mixtures and MNIST and measures mode coverage."),
    ),
  );
  init(); drawFlow();
}
