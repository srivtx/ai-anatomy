import { h, s, loop, rng } from "../lib/core.js";
import { slider, stats, predict, code, callout, section, prose, table, button, segmented } from "../lib/widgets.js";
import { xyChart } from "../lib/plot.js";
import { S, set, model, epoch, accuracy, reliance } from "../lib/vision.js";

// ---------- puzzles that are trivial for people and hard for vision-language models ----------
const PAL = ["var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)", "var(--c5)", "var(--c6)"];
function puzzle(kind, r) {
  const svg = s("svg", { viewBox: "0 0 220 220", class: "vz-svg" });
  const pick = (a) => a[Math.floor(r() * a.length)];
  if (kind === "lines") {
    let ya, yb, n;
    do {
      ya = [0, 1, 2].map(() => 30 + r() * 160); yb = [0, 1, 2].map(() => 30 + r() * 160);
      const d = ya.map((v, i) => v - yb[i]);
      n = (Math.sign(d[0]) !== Math.sign(d[1])) + (Math.sign(d[1]) !== Math.sign(d[2]));
      if (d.every((v) => Math.abs(v) > 18)) break;
    } while (true);
    const ia = Math.floor(r() * PAL.length), ca = PAL[ia], cb = PAL[(ia + 1 + Math.floor(r() * (PAL.length - 1))) % PAL.length];
    for (const [ys, c] of [[ya, ca], [yb, cb]]) svg.append(s("polyline", { points: ys.map((y, i) => `${20 + i * 90},${y}`).join(" "), style: `fill:none;stroke:${c};stroke-width:5;stroke-linecap:round;stroke-linejoin:round` }));
    return { svg, q: "How many times do the two lines cross?", opts: ["0", "1", "2"], a: String(n) };
  }
  if (kind === "circles") {
    const r1 = 30 + r() * 25, r2 = 30 + r() * 25, overlap = r() < 0.5;
    const d = overlap ? r1 + r2 - 4 - r() * 10 : r1 + r2 + 3 + r() * 8, th = r() * Math.PI;
    const cx = 110, cy = 110, x1 = cx - (Math.cos(th) * d) / 2, y1 = cy - (Math.sin(th) * d) / 2, x2 = cx + (Math.cos(th) * d) / 2, y2 = cy + (Math.sin(th) * d) / 2;
    svg.append(s("circle", { cx: x1, cy: y1, r: r1, style: `fill:none;stroke:${pick(PAL)};stroke-width:4` }), s("circle", { cx: x2, cy: y2, r: r2, style: `fill:none;stroke:${pick(PAL)};stroke-width:4` }));
    return { svg, q: "Do the two circles overlap?", opts: ["Yes", "No"], a: overlap ? "Yes" : "No" };
  }
  if (kind === "squares") {
    const n = 2 + Math.floor(r() * 4), c = pick(PAL);
    for (let i = 0; i < n; i++) { const m = 18 + i * (12 + r() * 4); svg.append(s("rect", { x: m + r() * 3, y: m + r() * 3, width: 220 - 2 * m, height: 220 - 2 * m, rx: 3, style: `fill:none;stroke:${c};stroke-width:3.5` })); }
    return { svg, q: "How many squares are there?", opts: ["2", "3", "4", "5"], a: String(n) };
  }
  if (kind === "grid") {
    const rows = 3 + Math.floor(r() * 4), cols = 3 + Math.floor(r() * 4), c = pick(PAL);
    for (let i = 0; i <= rows; i++) svg.append(s("line", { x1: 20, x2: 200, y1: 20 + (180 * i) / rows, y2: 20 + (180 * i) / rows, style: `stroke:${c};stroke-width:3` }));
    for (let j = 0; j <= cols; j++) svg.append(s("line", { y1: 20, y2: 200, x1: 20 + (180 * j) / cols, x2: 20 + (180 * j) / cols, style: `stroke:${c};stroke-width:3` }));
    return { svg, q: "How many columns does the grid have?", opts: ["3", "4", "5", "6"], a: String(cols) };
  }
  // circled letter
  const word = pick(["Subdermatoglyphic", "Acknowledgement", "Photosynthesis", "Hippopotamus"]), i = 1 + Math.floor(r() * (word.length - 2));
  const w = 200 / word.length;
  word.split("").forEach((ch, k) => svg.append(s("text", { x: 10 + w * (k + 0.5), y: 120, "text-anchor": "middle", style: `font:600 ${Math.min(22, w * 1.5)}px var(--mono);fill:var(--ink)`, text: ch })));
  svg.append(s("ellipse", { cx: 10 + w * (i + 0.5), cy: 113, rx: w * 0.62, ry: 17, style: "fill:none;stroke:var(--c2);stroke-width:3" }));
  const a = word[i].toUpperCase(), opts = [...new Set([a, ...[i - 1, i + 1, (i + 3) % word.length].map((k) => word[k].toUpperCase())])].slice(0, 4).sort();
  return { svg, q: "Which letter is circled?", opts, a };
}

export default function (root, { id }) {
  // ---------- shortcut learning ----------
  let corr = 0.98, fix = "none", m, tr, sameSet, flipSet, neutral, ep = 0, curve = { same: [], flip: [] };
  const EP = 60;
  const ch = xyChart({ width: 900, height: 230, xLabel: "epoch", yLabel: "accuracy %", xDomain: [0, EP], yDomain: [0, 100] });
  const st = stats([{ label: "Epoch" }, { label: "Usual test images" }, { label: "Colour flipped" }, { label: "Decision driven by colour" }]);
  const samples = h("div.vz-samples"), wmaps = h("div.vz-wmaps");
  const tile = (d, size = 46) => {
    const c = h("canvas", { width: S, height: S, class: "vz-tile", style: { width: size + "px", height: size + "px" } });
    const x = c.getContext("2d"), img = x.createImageData(S, S), bg = d.hue ? [57, 135, 229] : [235, 104, 52];
    for (let k = 0; k < S * S; k++) { const v = Math.max(0, Math.min(1, d.px[k])); for (let q = 0; q < 3; q++) img.data[k * 4 + q] = Math.round(bg[q] * (1 - v) + 250 * v); img.data[k * 4 + 3] = 255; }
    x.putImageData(img, 0, 0); return c;
  };
  function drawSamples() {
    const tf = set(6, corr, 7, 0.35), fl = set(6, 0, 8, 0.35);
    samples.replaceChildren(
      h("div", h("div.eyebrow", "Training images (label: | or —)"), h("div.vz-row", tf.map((d) => h("figure", tile(d), h("figcaption", d.y ? "|" : "—"))))),
      h("div", h("div.eyebrow", "Test: same shapes, colours swapped"), h("div.vz-row", fl.map((d) => h("figure", tile(d), h("figcaption", d.y ? "|" : "—"))))));
  }
  function drawWeights() {
    const W = m.W[0], H = m.sizes[1], cells = [];
    for (let u = 0; u < H; u++) {
      const c = h("canvas", { width: S, height: S, class: "vz-tile", style: { width: "40px", height: "40px" } });
      const x = c.getContext("2d"), img = x.createImageData(S, S);
      let mx = 1e-9; for (let k = 0; k < S * S; k++) mx = Math.max(mx, Math.abs(W[k * H + u]));
      const hueW = W[S * S * H + u];
      for (let k = 0; k < S * S; k++) { const v = W[k * H + u] / mx, col = v > 0 ? [98, 80, 214] : [27, 175, 122], a = Math.abs(v); for (let q = 0; q < 3; q++) img.data[k * 4 + q] = Math.round(248 * (1 - a) + col[q] * a); img.data[k * 4 + 3] = 255; }
      x.putImageData(img, 0, 0);
      const shapeW = Math.sqrt([...Array(S * S)].reduce((s, _, k) => s + W[k * H + u] ** 2, 0));
      const frac = Math.abs(hueW) / (Math.abs(hueW) + 0.48 * shapeW);     // each pixel varies with sd ≈ 0.48; colour input with sd 1
      cells.push(h("div.vz-unit", c, h("div.vz-hue", h("span", { style: { width: `${Math.round(frac * 100)}%` } })), h("small", `${Math.round(frac * 100)}% colour`)));
    }
    wmaps.replaceChildren(...cells);
  }
  function trainingSet() {
    const base = set(400, corr, 1, 0.35);
    if (fix === "jitter") return base.map((d, i) => ({ ...d, hue: ((i * 7919) % 13) % 2 }));
    if (fix === "reweight") {
      const mino = base.filter((d) => d.hue !== d.y), maj = base.filter((d) => d.hue === d.y);
      if (!mino.length) return base;
      const out = [...maj]; for (let i = 0; i < Math.round(maj.length / mino.length); i++) out.push(...mino);
      const r = rng(5); for (let i = out.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [out[i], out[j]] = [out[j], out[i]]; }
      return out;
    }
    return base;
  }
  function reset() {
    lp.stop(); m = model(); tr = trainingSet(); sameSet = set(300, corr, 2, 0.35); flipSet = set(300, 0, 3, 0.35); neutral = set(200, 0.5, 4, 0.35);
    ep = 0; curve = { same: [], flip: [] }; drawSamples(); record(); lp.start();
  }
  function record() {
    const a = accuracy(m, sameSet), f = accuracy(m, flipSet), rel = reliance(m, neutral);
    curve.same.push([ep, a * 100]); curve.flip.push([ep, f * 100]);
    st.set(0, `${ep} / ${EP}`); st.set(1, `${Math.round(a * 100)}%`); st.set(2, `${Math.round(f * 100)}%`, f < 0.5 ? "bad" : f > 0.85 ? "good" : "");
    st.set(3, `${Math.round(rel * 100)}%`, rel > 0.5 ? "bad" : "");
    ch.update([{ name: "usual test images", points: curve.same, color: "var(--c1)" }, { name: "colours swapped", points: curve.flip, color: "var(--c2)" }]);
    drawWeights();
  }
  const lp = loop(() => { epoch(m, tr); ep++; record(); if (ep >= EP) lp.stop(); }, ch.el);

  // ---------- puzzles ----------
  const KINDS = ["lines", "circles", "squares", "grid", "letter"];
  let seed = 11, score = [0, 0];
  const pzEl = h("div.vz-puzzles"), scoreEl = h("span.vz-score");
  function newPuzzles() {
    const r = rng(seed++);
    pzEl.replaceChildren(...KINDS.map((k, i) => {
      const p = puzzle(k, r), out = h("div.vz-ans");
      const btns = p.opts.map((o) => h("button.pz-opt", { type: "button", onclick: () => {
        if (btns.some((b) => b.disabled)) return;
        btns.forEach((b) => { b.disabled = true; if (b.textContent === p.a) b.classList.add("pz-best"); });
        const ok = o === p.a; score[1]++; if (ok) score[0]++;
        if (!ok) btns.find((b) => b.textContent === o).classList.add("pz-picked");
        out.textContent = ok ? "Right. A person needs about a second." : `It's ${p.a}.`;
        scoreEl.textContent = `You: ${score[0]} / ${score[1]}`;
      } }, o));
      return h("div.vz-card", { style: { "--pz": PAL[i % PAL.length] } }, p.svg, h("p.pz-q", p.q), h("div.vz-opts", btns), out);
    }));
  }

  root.append(
    section(null, prose(`<p>Vision feels effortless to us because half of our cortex works on it. For machines it's where the gap between impressive and fragile is widest: models that caption photos beautifully can fail to count the lines in a grid. This chapter shows why, with a network you can watch take a shortcut, puzzles that today's vision-language models still slip on, and the less-known ideas behind modern visual models.</p>`)),
    section("The shortcut: a network that learns colour instead of shape",
      prose(`<p>Each image is a vertical or horizontal bar: that's the label. The background colour <i>usually</i> agrees with the label too, the way cows in training photos are usually on grass. A small network can learn the shape perfectly well (set agreement to 50% and it does). What does it learn when the colour is a free clue?</p>`),
      predict({ chapter: id, question: "The background colour agrees with the label 98% of the time. The network reaches 98% test accuracy. Then you show it the same shapes with colours swapped. Its accuracy is…",
        options: ["About 98%: it clearly learned the task", "About 50%: it's confused", "Near 0%: it learned the colour and ignores the shape", "About 75%"], answer: 2,
        explain: "The colour is a single input that's right 98% of the time; the shape is spread over 64 noisy pixels. Gradient descent takes the simplest feature that works (“simplicity bias”, Shah et al., 2020) and has no reason to learn the harder one. 98% accuracy said nothing about what was learned." }),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Colour agrees with label", min: 0.5, max: 1, step: 0.02, value: corr, fmt: (v) => `${Math.round(v * 100)}%`, onInput: (v) => { corr = v; reset(); } }),
          segmented({ label: "Fix", options: [{ value: "none", label: "None" }, { value: "reweight", label: "Upweight counterexamples" }, { value: "jitter", label: "Randomise colours" }], value: fix, onChange: (v) => { fix = v; reset(); } }),
          button("Retrain", reset)),
        h("div.stage-body", st.el, samples, ch.el,
          h("div", h("div.eyebrow", "What each hidden unit looks at: 8×8 pixel weights (violet +, green −), and how much of its input signal comes from colour"), wmaps))),
      callout("note", "What I measured", `<ul style="margin:0;padding-left:18px"><li>No shortcut (50%): ~95% on both test sets. The shape is learnable.</li><li>90% agreement: ~90% on usual images, ~34% when swapped. Even with 1 in 10 examples contradicting the colour, colour wins.</li><li>98%: 98% usual, ~3% swapped.</li><li><b>Upweighting</b> the contradicting examples helps only partly at 90% (swapped → ~47%) and not at 98%, where there are just 8 of them to learn from: the network memorises them. <b>Randomising colours</b> fixes it completely, but only because we knew which shortcut to remove. That's the hard part in real data.</li></ul>`),
    ),
    section("Puzzles: easy for you, hard for models",
      prose(`<p>Rahmanzadehgervi et al. (2024), “Vision language models are blind”, built tests like these. People score near 100%. In 2024 the best vision-language models averaged roughly 55–75% across such tasks, with some under chance on counting crossings and nested shapes. Newer models do better but still slip on exact counts and fine spatial relations. Generate as many as you like.</p>`),
      h("div.row", button("New puzzles", newPuzzles, "primary"), scoreEl),
      pzEl,
      callout("tip", "Why models fail at these", `<p>Most vision-language models cut the image into patches (often 14×14 or 16×16 pixels), turn each patch into one token, and hand the tokens to a language model. A line crossing sits inside one patch, compressed with everything else there; two circles 4 pixels apart look the same as two touching. Nothing in captioning data rewards exact geometry, because captions rarely say “the lines cross twice”. Models that can <b>zoom</b> (crop and re-encode part of the image as a tool call) or write code to measure do much better. That's why “thinking with images” arrived in 2025 reasoning models.</p>`),
    ),
    section("What isn't in the textbooks yet",
      table(["Idea", "What it says", "Why it matters"], [
        ["Texture bias", "ImageNet CNNs classify a cat-shaped image with elephant texture as “elephant” (Geirhos et al., 2019). Humans go by shape ~96% of the time; ResNets by texture most of the time. ViT-22B reached 87% shape bias.", "Shortcuts at scale. Bigger and better-trained models drift towards human-like shape bias."],
        ["Registers", "Large ViTs repurpose a few background patch tokens as scratch memory, producing artefact tokens with huge norms. Adding a few extra learnable “register” tokens gives them somewhere else to compute and cleans up attention maps (Darcet et al., 2024).", "A model inventing working memory when none is provided."],
        ["The modality gap", "In CLIP, image and text embeddings occupy two separate narrow cones, never mixing, even for matched pairs (Liang et al., 2022). It comes from initialisation plus the contrastive loss's temperature.", "Why cosine thresholds across modalities behave oddly, and why “image ≈ text” in embedding space is only true after centring."],
        ["Emergent segmentation", "Self-supervised ViTs (DINO, DINOv2) trained with no labels at all produce attention maps that outline objects.", "Objectness comes from predicting views of the same scene, not from labels."],
        ["Predict in latent space (JEPA)", "I-JEPA and V-JEPA predict the representation of a hidden region rather than its pixels. Skipping pixel detail they can't predict makes learning cheaper and features more semantic.", "LeCun's bet for world models; V-JEPA 2 (2025) used it for robot planning."],
        ["Images are expensive tokens", "One image costs hundreds to thousands of tokens. Tiling high-resolution images multiplies this, and compression schemes (pixel shuffle, token merging) trade cost against detail.", "The reason counting and small text still fail: the detail was pooled away before the language model saw it."],
      ]),
    ),
    section("In code",
      code({
        "Counterfactual test for shortcuts": `def shortcut_report(model, test_loader, transforms):
    """Accuracy under transformations that should NOT change the label."""
    base = accuracy(model, test_loader)
    for name, t in transforms.items():               # e.g. colour swap, background swap, crop, grayscale
        acc = accuracy(model, test_loader, transform=t)
        print(f"{name:>16}: {acc:.1%} ({acc - base:+.1%})")   # a big drop = the model relied on it

transforms = {"grayscale": T.Grayscale(3), "colour jitter": T.ColorJitter(0.5, 0.5, 0.5, 0.2),
              "background swap": swap_background, "patch shuffle": shuffle_patches(4)}`,
        "Shape vs texture bias": `# cue-conflict images: shape of class A rendered with the texture of class B (Geirhos et al. stimuli)
shape_hits = texture_hits = 0
for img, shape_cls, texture_cls in cue_conflict:
    pred = model(img[None]).argmax().item()
    shape_hits += pred == shape_cls; texture_hits += pred == texture_cls
print("shape bias:", shape_hits / (shape_hits + texture_hits))   # humans ≈ 0.96, ResNet-50 ≈ 0.2`,
        "ViT with registers": `class ViTWithRegisters(nn.Module):
    def __init__(self, vit, n_reg=4):
        super().__init__(); self.vit = vit
        self.reg = nn.Parameter(torch.zeros(1, n_reg, vit.embed_dim)); self.n_reg = n_reg
    def forward(self, x):
        t = self.vit.patch_embed(x) + self.vit.pos_embed[:, 1:]
        cls = self.vit.cls_token + self.vit.pos_embed[:, :1]
        t = torch.cat([cls.expand(len(x), -1, -1), self.reg.expand(len(x), -1, -1), t], 1)
        for blk in self.vit.blocks: t = blk(t)
        return self.vit.head(self.vit.norm(t)[:, 0])          # registers are simply discarded at the end`,
        "Zoom tool for a VLM agent": `import anthropic, base64, io
client = anthropic.Anthropic()
def zoom(img, box):                        # the model asks for a crop and sees it at full resolution
    buf = io.BytesIO(); img.crop(box).resize((768, 768)).save(buf, "PNG")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(buf.getvalue()).decode()}}
tools = [{"name": "zoom", "description": "Crop the image to [x0, y0, x1, y1] and look closer.",
          "input_schema": {"type": "object", "properties": {"box": {"type": "array", "items": {"type": "integer"}}}, "required": ["box"]}}]
# loop: client.messages.create(model="claude-opus-5-5", tools=tools, ...) → on tool_use, return zoom(img, box)`,
      }),
    ),
  );
  newPuzzles(); reset();
}
