import { h, rng, loop } from "../lib/core.js";
import { slider, segmented, stats, predict, code, callout, section, prose, table, button } from "../lib/widgets.js";
import { plane } from "../lib/plot.js";
import { MLP, gauss } from "../lib/nn.js";

// 8×8 glyphs from six families, randomly placed. Small enough to train in a few seconds.
const CLASSES = ["bar —", "bar |", "cross +", "box □", "slash /", "blob ●"];
function glyphs(n, seed) {
  const r = rng(seed), out = [];
  for (let k = 0; k < n; k++) {
    const c = k % 6, img = new Float64Array(64), set = (x, y, v = 1) => { if (x >= 0 && y >= 0 && x < 8 && y < 8) img[y * 8 + x] = Math.max(img[y * 8 + x], v); };
    const ox = r.int(3) - 1, oy = r.int(3) - 1;
    if (c === 0) { const y = 2 + r.int(4); for (let x = 1 + r.int(2); x < 7 - r.int(2); x++) set(x, y); }
    else if (c === 1) { const x = 2 + r.int(4); for (let y = 1 + r.int(2); y < 7 - r.int(2); y++) set(x, y); }
    else if (c === 2) { for (let i = 1; i < 7; i++) { set(3 + ox, i); set(i, 3 + oy); } }
    else if (c === 3) { const s = 3 + r.int(3), x0 = 1 + r.int(8 - s - 1), y0 = 1 + r.int(8 - s - 1); for (let i = 0; i <= s; i++) { set(x0 + i, y0); set(x0 + i, y0 + s); set(x0, y0 + i); set(x0 + s, y0 + i); } }
    else if (c === 4) { const flip = r() < 0.5; for (let i = 0; i < 8; i++) set(flip ? i : 7 - i, i); }
    else { const cx = 2.5 + r() * 3, cy = 2.5 + r() * 3, rad = 1.2 + r() * 1.2; for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) { const d = Math.hypot(x - cx, y - cy); if (d < rad) set(x, y, 1); else if (d < rad + 0.7) set(x, y, 0.5); } }
    for (let i = 0; i < 64; i++) img[i] = Math.min(1, img[i] + Math.abs(gauss(r)) * 0.04);
    out.push({ x: img, c });
  }
  return out;
}

export default function (root, { id }) {
  const data = glyphs(600, 7);
  let mode = "ae", beta = 1, enc, dec, steps, recon = 0, kl = 0, r;
  const LAT = 2.6;
  function init() {
    r = rng(9);
    enc = new MLP([64, 32, mode === "vae" ? 4 : 2], { act: "tanh", out: mode === "vae" ? "linear" : "tanh", seed: 3 });
    dec = new MLP([2, 32, 64], { act: "tanh", out: "sigmoid", seed: 4 });
    steps = 0;
  }
  function encode(x) {
    const o = enc.forward(x);
    return mode === "vae" ? { mu: [o[0], o[1]], lv: [o[2], o[3]] } : { mu: [o[0] * LAT, o[1] * LAT] };
  }
  function trainStep(bs = 64) {
    let R = 0, K = 0;
    for (let b = 0; b < bs; b++) {
      const s = data[r.int(data.length)], o = enc.forward(s.x);
      let z, eps, mu, lv;
      if (mode === "vae") { mu = [o[0], o[1]]; lv = [Math.max(-6, Math.min(4, o[2])), Math.max(-6, Math.min(4, o[3]))]; eps = [gauss(r), gauss(r)]; z = mu.map((m, i) => m + Math.exp(0.5 * lv[i]) * eps[i]); }
      else z = [o[0] * LAT, o[1] * LAT];
      const y = dec.forward(z);
      const dy = new Float64Array(64);
      for (let i = 0; i < 64; i++) { const e = y[i] - s.x[i]; R += e * e; dy[i] = 2 * e; }
      const dz = dec.backward(dy);
      if (mode === "vae") {
        K += 0.5 * mu.reduce((a, m, i) => a + m * m + Math.exp(lv[i]) - 1 - lv[i], 0);
        enc.backward([dz[0] + beta * mu[0], dz[1] + beta * mu[1],
          dz[0] * eps[0] * 0.5 * Math.exp(0.5 * lv[0]) + beta * 0.5 * (Math.exp(lv[0]) - 1),
          dz[1] * eps[1] * 0.5 * Math.exp(0.5 * lv[1]) + beta * 0.5 * (Math.exp(lv[1]) - 1)]);
      } else enc.backward([dz[0] * LAT, dz[1] * LAT]);
    }
    enc.step(0.004, bs); dec.step(0.004, bs);
    steps++;
    recon = recon * 0.9 + (R / bs) * 0.1; kl = kl * 0.9 + (K / bs) * 0.1;
  }

  const lat = plane({ width: 380, height: 380, xr: [-LAT, LAT], yr: [-LAT, LAT], label: "Latent space: every glyph's 2D code, coloured by family", onPoint: (x, y) => { pick = [x, y]; draw(); } });
  const gridC = h("canvas", { width: 360, height: 360, role: "img", "aria-label": "Glyphs decoded from a grid of latent points", style: { width: "100%", display: "block", imageRendering: "pixelated", borderRadius: "10px", border: "1px solid var(--line)" } });
  const pickC = h("canvas", { width: 8, height: 8, role: "img", "aria-label": "The glyph decoded from the point you clicked", style: { width: "112px", height: "112px", imageRendering: "pixelated", borderRadius: "8px", border: "1px solid var(--line)" } });
  const st = stats([{ label: "Training steps" }, { label: "Reconstruction error" }, { label: "KL to N(0, 1)" }]);
  let pick = [0, 0];
  const COLS = ["c1", "c2", "c3", "c4", "c5", "c6"];
  const cs = () => getComputedStyle(document.documentElement);
  const rgb = (v) => (cs().getPropertyValue(v).trim().match(/[0-9a-f]{2}/gi) || ["88", "88", "88"]).map((x) => parseInt(x, 16));
  function paintGlyph(ctx, img, x0, y0, px) {
    const bg = rgb("--surface"), ink = rgb("--ink");
    for (let i = 0; i < 64; i++) { const t = img[i]; ctx.fillStyle = `rgb(${bg.map((b, k) => Math.round(b + (ink[k] - b) * t)).join(",")})`; ctx.fillRect(x0 + (i % 8) * px, y0 + Math.floor(i / 8) * px, px, px); }
  }
  function draw() {
    lat.clear(); lat.axes();
    for (const s of data.slice(0, 360)) { const { mu } = encode(s.x); lat.dot(lat.X(mu[0]), lat.Y(mu[1]), 2.6, lat.colors[COLS[s.c]]); }
    lat.dot(lat.X(pick[0]), lat.Y(pick[1]), 8, lat.colors.ink); lat.dot(lat.X(pick[0]), lat.Y(pick[1]), 5, lat.colors.surface);
    const g = gridC.getContext("2d");
    g.fillStyle = cs().getPropertyValue("--line"); g.fillRect(0, 0, 360, 360);
    for (let j = 0; j < 9; j++) for (let i = 0; i < 9; i++) paintGlyph(g, dec.forward([-LAT + (i + 0.5) * (2 * LAT / 9), LAT - (j + 0.5) * (2 * LAT / 9)]), i * 40 + 2, j * 40 + 2, 4.5);
    paintGlyph(pickC.getContext("2d"), dec.forward(pick), 0, 0, 1);
    st.set(0, String(steps)); st.set(1, recon ? recon.toFixed(2) : "–"); st.set(2, mode === "vae" ? kl.toFixed(2) : "not used");
  }
  const lp = loop(() => { for (let k = 0; k < 4; k++) trainStep(); draw(); if (steps > 3000) lp.stop(); }, lat.el);
  init();

  root.append(
    section(null, prose(`<p>An <b>autoencoder</b> is trained to output its own input, through a bottleneck. The <b>encoder</b> squeezes a 64-pixel image into just two numbers (the <b>latent code</b>); the <b>decoder</b> rebuilds the image from those two numbers. To succeed it must discover what actually varies between images: shape, position, thickness. No labels needed.</p>`)),
    section("Train one in your browser",
      prose("<p>600 tiny 8 × 8 glyphs in six families (bars, crosses, boxes, slashes, blobs). The left plot places each glyph at its 2D code; colour shows its family, which the model never sees. Click anywhere in the latent plane to decode that point. The right panel decodes a whole grid of points.</p>"),
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Model", options: [{ value: "ae", label: "Autoencoder" }, { value: "vae", label: "Variational autoencoder" }], value: mode, onChange: (v) => { mode = v; lp.stop(); init(); draw(); } }),
          slider({ label: "β (weight on the KL term, VAE only)", min: 0, max: 4, step: 0.1, value: beta, fmt: (v) => v.toFixed(1), onInput: (v) => (beta = v) }),
          button("Train", () => lp.start(), "primary"), button("Pause", () => lp.stop()), button("Reset", () => { lp.stop(); init(); draw(); })),
        h("div.stage-body", st.el,
          h("div.lab",
            h("div", h("div.canvas-box", lat.el), h("div.legend", { style: { marginTop: "6px" } }, CLASSES.map((c, i) => h("span", h("i", { style: { background: `var(--${COLS[i]})` } }), c)))),
            h("div", gridC, h("div.row", { style: { marginTop: "10px", alignItems: "center" } }, pickC, h("p.canvas-cap", { style: { margin: 0, flex: "1 1 160px" } }, "Decoded from the point you clicked. Click between two families to see a blend.")))))),
      callout("tip", "Things to try", "<ul style='margin:0;padding-left:18px'><li>Train the plain autoencoder. Families form clusters, but the space between them decodes to garbage, and the clusters sit anywhere. You can compress with it, but you can't <i>sample</i> from it.</li><li>Switch to the VAE and train. The KL term pulls all codes towards a standard normal blob, so the space is filled and smooth: every point decodes to something plausible. That makes it a <b>generative model</b>.</li><li>Raise β to 4: codes collapse together and reconstructions blur. Set β to 0: it's an autoencoder with noise. The trade-off between sharp reconstructions and a usable latent space is the VAE's central tension.</li></ul>"),
      predict({
        chapter: id,
        question: "Why does a VAE's encoder output a mean and a variance instead of one point?",
        options: ["To use more parameters", "So the decoder learns to handle a whole neighbourhood of codes, which makes the latent space smooth and sampleable", "To speed up training", "Because images are Gaussian"],
        answer: 1,
        explain: "<p>Sampling z = μ + σ·ε during training (the <b>reparameterisation trick</b>, Kingma and Welling 2013) forces nearby codes to decode to similar images, and the KL term keeps codes near N(0, 1). At generation time you sample z ~ N(0, 1) and decode. Stable Diffusion's first stage is a VAE that compresses 512×512 images into a 64×64 latent the diffusion model works in.</p>",
      }),
    ),
    section("The loss, written out",
      prose("<p>VAE loss = <b>reconstruction</b> (how well the decoder rebuilds x from z) + β × <b>KL</b>(q(z | x) ‖ N(0, I)). For a Gaussian encoder the KL has a closed form: ½ Σ (μ² + σ² − 1 − log σ²). Together they form the negative <b>evidence lower bound</b> (ELBO) on log p(x): a bound on how likely the model thinks the data is.</p>"),
      table(["Model", "Bottleneck", "Can generate?", "Used for"], [
        ["Autoencoder", "Deterministic code", "Not reliably", "Compression, denoising, anomaly detection (high reconstruction error = weird)"],
        ["VAE", "Distribution, KL-regularised", "Yes, a bit blurry", "Latent spaces for diffusion (SD-VAE), representation learning"],
        ["VQ-VAE", "Nearest entry in a learned codebook", "Yes, with a prior model on codes", "Image and audio tokenisers: discrete tokens for transformers"],
        ["Sparse autoencoder", "Wide, but few units active", "No", "Interpretability: finding features inside LLMs"],
        ["Masked autoencoder", "Reconstruct hidden patches", "No", "Self-supervised pretraining for vision (MAE)"],
      ]),
    ),
    section("In code",
      code({
        PyTorch: `import torch, torch.nn as nn, torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, d_in=784, d_z=2):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, 512), nn.ReLU(), nn.Linear(512, 2 * d_z))
        self.dec = nn.Sequential(nn.Linear(d_z, 512), nn.ReLU(), nn.Linear(512, d_in))
    def forward(self, x):
        mu, logvar = self.enc(x).chunk(2, dim=-1)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()     # reparameterisation trick
        return self.dec(z), mu, logvar

def vae_loss(logits, x, mu, logvar, beta=1.0):
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction="sum")
    kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).sum()
    return (recon + beta * kl) / len(x)`,
      }, "Project 04 trains this on MNIST and draws the latent grid."),
    ),
  );
  lat.paint(draw);
}
