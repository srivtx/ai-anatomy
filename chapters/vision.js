import { h, rng, loop } from "../lib/core.js";
import { slider, segmented, stats, predict, code, callout, section, prose, table, button, toggle, stepper } from "../lib/widgets.js";
import { heatmap } from "../lib/plot.js";
import { MLP } from "../lib/nn.js";
import { TinyClip, ALL_PAIRS, HELD, N, render } from "../lib/clip.js";

const SYM = { circle: "●", square: "■", triangle: "▲", cross: "✚" };
const short = (c) => { const [col, sh] = c.split(" "); return `${col[0].toUpperCase()}${SYM[sh]}`; };
const S = 96; // the scene is S×S pixels

// A small procedural scene: sky gradient, sun, hills, a house. Deterministic, original.
function scene() {
  const px = new Uint8ClampedArray(S * S * 4);
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    let c = [120 + y, 170 + y * 0.6, 235];
    if ((x - 72) ** 2 + (y - 20) ** 2 < 120) c = [255, 205, 70];
    const hill = 62 + 9 * Math.sin(x / 13) + 4 * Math.sin(x / 5 + 1);
    if (y > hill) c = [70 + (y - hill) * 1.2, 150 - (y - hill) * 0.5, 75];
    if (x > 18 && x < 46 && y > 50 && y < 76) c = [215, 120, 90];
    if (y <= 50 && y > 34 && Math.abs(x - 32) < (y - 32) * 1.0) c = [130, 60, 60];
    if (x > 28 && x < 36 && y > 62 && y < 76) c = [90, 55, 40];
    if (x > 21 && x < 27 && y > 55 && y < 61) c = [235, 240, 255];
    px.set([...c.map(Math.round), 255], (y * S + x) * 4);
  }
  return px;
}

function pixCanvas(n, css = {}) {
  return h("canvas", { width: n, height: n, style: { width: "100%", aspectRatio: "1", imageRendering: "pixelated", display: "block", borderRadius: "8px", border: "1px solid var(--line)", ...css } });
}

export default function (root, { id }) {
  // ---------- 1. patches ----------
  const px = scene();
  let P = 16, mask = 0;
  const big = h("canvas", { width: S * 4, height: S * 4, style: { width: "100%", maxWidth: "380px", aspectRatio: "1", display: "block", borderRadius: "10px", border: "1px solid var(--line)" } });
  const strip = h("div", { style: { display: "flex", flexWrap: "wrap", gap: "3px" } });
  const pst = stats([{ label: "Tokens" }, { label: "Numbers per token" }, { label: "Attention pairs" }]);
  const maskR = rng(5);
  let masked = new Set();
  function drawPatches() {
    const ctx = big.getContext("2d"), src = new ImageData(px, S, S), tmp = document.createElement("canvas");
    tmp.width = S; tmp.height = S; tmp.getContext("2d").putImageData(src, 0, 0);
    ctx.imageSmoothingEnabled = false; ctx.drawImage(tmp, 0, 0, S * 4, S * 4);
    const g = S / P, n = g * g;
    // MAE-style random masking
    const ids = [...Array(n).keys()].sort(() => maskR() - 0.5);
    masked = new Set(ids.slice(0, Math.round(n * mask)));
    ctx.strokeStyle = "rgba(255,255,255,.85)"; ctx.lineWidth = 2;
    for (let i = 0; i < n; i++) {
      const x = (i % g) * P * 4, y = Math.floor(i / g) * P * 4;
      if (masked.has(i)) { ctx.fillStyle = "rgba(30,30,40,.82)"; ctx.fillRect(x, y, P * 4, P * 4); }
      ctx.strokeRect(x, y, P * 4, P * 4);
    }
    strip.replaceChildren(...Array.from({ length: Math.min(n, 72) }, (_, i) => {
      const c = h("canvas", { width: P, height: P, title: `token ${i + 1}`, style: { width: "26px", height: "26px", imageRendering: "pixelated", borderRadius: "3px", border: "1px solid var(--line)", opacity: masked.has(i) ? 0.15 : 1 } });
      const cx = c.getContext("2d"); cx.drawImage(tmp, (i % g) * P, Math.floor(i / g) * P, P, P, 0, 0, P, P);
      return c;
    }), ...(n > 72 ? [h("span.muted", { style: { alignSelf: "center", fontSize: "13px" } }, `… ${n - 72} more`)] : []));
    const vis = n - masked.size;
    pst.set(0, mask ? `${vis} of ${n} visible` : String(n)); pst.set(1, `${P}×${P}×3 = ${P * P * 3}`); pst.set(2, (vis * vis).toLocaleString());
  }

  // ---------- 2. tiny CLIP ----------
  const train = ALL_PAIRS.filter((c) => !HELD.has(c)), held = [...HELD];
  let clip, losses = [];
  const cst = stats([{ label: "Contrastive steps" }, { label: "Accuracy on trained pairs" }, { label: "Zero-shot on unseen pairs" }, { label: "Learned temperature τ" }]);
  const hmWrap = h("div", { style: { overflowX: "auto" } }), gallery = h("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(92px,1fr))", gap: "10px" } });
  const gR = rng(77);
  const samples = ALL_PAIRS.map((c) => ({ c, img: render(c, gR) }));
  function newClip(seed = 2) {
    clip = new TinyClip(seed);
    clip.img = new MLP([N * N * 3, 64, 8], { act: "relu", seed: seed + 3 });
    losses = []; drawClip();
  }
  function drawImg(img) {
    const c = pixCanvas(N), ctx = c.getContext("2d"), d = ctx.createImageData(N, N);
    for (let i = 0; i < N * N; i++) d.data.set([...[0, 1, 2].map((k) => Math.max(0, Math.min(255, img[i * 3 + k] * 255))), 255], i * 4);
    ctx.putImageData(d, 0, 0); return c;
  }
  function drawClip() {
    const E = samples.map((s) => clip.embedImage(s.img)), T = ALL_PAIRS.map((c) => clip.embedText(c));
    const M = E.map((e) => T.map((t) => e.reduce((a, x, k) => a + x * t[k], 0)));
    hmWrap.replaceChildren(heatmap(M, { rows: ALL_PAIRS.map((c) => (HELD.has(c) ? `${c} ★` : c)), cols: ALL_PAIRS.map(short), cell: 30, showValues: false, fmt: (v) => v.toFixed(2) }));
    gallery.replaceChildren(...samples.map(({ c, img }) => {
      const pred = clip.classify(img), ok = pred === c;
      return h("div", { style: { fontSize: "12px", textAlign: "center" } }, drawImg(img),
        h("div", { style: { marginTop: "4px", fontWeight: 600, color: clip.steps ? (ok ? "var(--good)" : "var(--bad)") : "var(--ink-3)" } }, clip.steps ? `${ok ? "✓" : "✗"} ${pred}` : "?"),
        HELD.has(c) ? h("div.muted", "never trained on") : null);
    }));
    cst.set(0, String(clip.steps));
    if (clip.steps) {
      const a = clip.accuracy(train, 3), z = clip.accuracy(held, 6);
      cst.set(1, `${Math.round(a * 100)}%`, a > 0.85 ? "good" : ""); cst.set(2, `${Math.round(z * 100)}%`, z > 0.5 ? "good" : z > 0.2 ? "warn" : "bad");
    } else { cst.set(1, "–"); cst.set(2, "–"); }
    cst.set(3, (1 / Math.exp(clip.logT)).toFixed(3));
  }
  let frame = 0;
  const lp = loop(() => { for (let k = 0; k < 6; k++) losses.push(clip.step(train, 0.005)); if (++frame % 3 === 0) drawClip(); if (clip.steps >= 600) { lp.stop(); drawClip(); } }, gallery);

  // ---------- 3. VLM token budget ----------
  let res = 448, patch = 14, merge = 2, imgs = 4;
  const vst = stats([{ label: "Patches per image" }, { label: "Tokens per image after merging" }, { label: "Tokens for all images" }]);
  function drawVlm() {
    const g = Math.floor(res / patch), tokens = g * g, merged = Math.ceil(tokens / (merge * merge));
    vst.set(0, tokens.toLocaleString()); vst.set(1, merged.toLocaleString()); vst.set(2, (merged * imgs).toLocaleString(), merged * imgs > 32000 ? "warn" : "");
  }

  root.append(
    section(null, prose(`<p>For a decade, vision meant convolutional networks. Then <b>Vision Transformers</b> (Dosovitskiy et al., 2020) showed you can cut an image into square <b>patches</b>, treat each patch as a token, and feed the sequence to the same transformer that reads text. With enough data it matches or beats CNNs, and it means one architecture can see and read. <b>CLIP</b> (Radford et al., 2021) then trained an image encoder and a text encoder together on 400 million image–caption pairs so that matching pairs land close together. That shared space is what lets you search photos with words, guide image generators, and plug vision into language models.</p>`)),
    section("An image is a sequence of patches",
      prose("<p>Each patch is flattened into a vector of pixel values and multiplied by a learned matrix to become a token embedding. A learned <b>position embedding</b> is added so the model knows where each patch came from. After that it's an ordinary transformer.</p>"),
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Patch size", options: [4, 8, 16, 32].map((v) => ({ value: v, label: `${v} px` })), value: 16, onChange: (v) => { P = v; drawPatches(); } }),
          slider({ label: "Mask ratio (MAE pretraining)", min: 0, max: 0.9, step: 0.05, value: 0, fmt: (v) => `${Math.round(v * 100)}%`, onInput: (v) => { mask = v; drawPatches(); } })),
        h("div.stage-body", pst.el, h("div.lab", h("div", big, h("p.canvas-cap", "A 96×96 scene cut into patches")), h("div", h("div.eyebrow", "The token sequence the transformer reads, left to right, top to bottom"), strip)))),
      callout("tip", "Things to try", "<ul style='margin:0;padding-left:18px'><li>Halve the patch size: four times the tokens, sixteen times the attention pairs. That's why ViTs use 14–16 px patches and why high-resolution images are expensive.</li><li>Raise the mask ratio to 75%. A <b>masked autoencoder</b> (He et al., 2021) hides that many patches and trains the model to paint them back. The encoder only processes the visible quarter, so pretraining is cheap, and images are redundant enough that 25% is plenty to infer the rest.</li></ul>"),
      predict({
        chapter: id,
        question: "A 224×224 image with 16×16 patches becomes how many tokens (ignoring the class token)?",
        options: ["14", "196", "224", "50,176"],
        answer: 1,
        explain: "<p>224 / 16 = 14 patches per side, and 14 × 14 = 196. Each token holds 16 × 16 × 3 = 768 numbers, which happens to equal ViT-Base's width.</p>",
      }),
      table(["", "CNN", "Vision transformer"], [
        ["Built-in assumptions", "Locality and translation equivariance in every layer", "Almost none: patches attend to all patches from layer 1"],
        ["With little data (ImageNet-1k only)", "Better", "Worse: it must learn what convolutions assume"],
        ["With lots of data (JFT-300M and beyond)", "Plateaus", "Better, and keeps improving"],
        ["Mixing with text", "Needs adapters", "Patches are already tokens"],
      ]),
    ),
    section("Train a tiny CLIP",
      prose("<p>Sixteen kinds of picture: four colours × four shapes, each drawn at a random position and size. The text encoder sums a learned vector per word. The image encoder is a small MLP. The loss is CLIP's: in each batch, every image should be most similar to its own caption, and every caption to its own image (a cross-entropy over the similarity matrix, in both directions). <b>Four combinations, marked ★, are never shown during training.</b> Can the model recognise a blue circle having only seen blue things and circles separately?</p>"),
      h("div.stage",
        h("div.stage-controls", button("Train", () => lp.start(), "primary"), button("Pause", () => lp.stop()), button("Reset with a new seed", () => { lp.stop(); newClip(Math.floor(Math.random() * 1e4)); })),
        h("div.stage-body", cst.el, h("div.eyebrow", "Fresh test images and the caption the model picks from all 16"), gallery, h("div.eyebrow", { style: { marginTop: "14px" } }, "Cosine similarity: image (row) vs caption (column; R = red, G = green, B = blue, Y = yellow). Training pushes the diagonal up."), hmWrap)),
      callout("note", "What this shows", "<p>Zero-shot classification is just: embed the image, embed every candidate caption, pick the closest. Nothing was trained as a classifier. When the unseen pairs are recognised, it's because the model learned colour and shape as separate directions that add up, which is the kind of compositional structure that makes CLIP useful. When it fails, it usually gets one attribute right and the other wrong. Real CLIP is notoriously weak at exactly this (binding attributes to the right objects), because its text encoder can treat a caption as a bag of words, like ours.</p>"),
      predict({
        chapter: id,
        question: "CLIP uses the other captions in the batch as negatives. What happens with a bigger batch?",
        options: ["Nothing; batch size only affects speed", "Each image must beat more wrong captions, so the task is harder and the learned features are sharper", "It gets easier", "Training becomes unstable for any batch over 64"],
        answer: 1,
        explain: "<p>The original CLIP used a batch of 32,768. More negatives means more and harder distractors per step. SigLIP (Zhai et al., 2023) replaced the softmax with an independent sigmoid per pair, which removes the need for the whole batch's similarity matrix to be normalised together and works well at smaller batch sizes.</p>",
      }),
    ),
    section("Vision–language models",
      prose("<p>Most open vision–language models (LLaVA, Qwen-VL, Gemma 3) bolt a pretrained vision encoder onto a pretrained language model:</p>"),
      stepper([
        { title: "1. Encode", html: "<p>A ViT (often CLIP or SigLIP) turns the image into a grid of patch embeddings.</p>" },
        { title: "2. Compress", html: "<p>Neighbouring patch embeddings are merged (e.g. 2×2 into one) to cut the token count.</p>" },
        { title: "3. Project", html: "<p>A small MLP maps each vision embedding into the language model's embedding space. To the LLM they look like word tokens it has never seen.</p>" },
        { title: "4. Read", html: "<p>The image tokens are spliced into the text sequence where the image appeared, and the LLM attends over both. Training aligns the projector first, then fine-tunes on instruction data with images.</p>" },
      ]),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Image resolution", min: 224, max: 1344, step: 112, value: res, fmt: (v) => `${v}×${v}`, onInput: (v) => { res = v; drawVlm(); } }),
          slider({ label: "Patch size", min: 14, max: 16, step: 2, value: patch, fmt: (v) => `${v} px`, onInput: (v) => { patch = v; drawVlm(); } }),
          slider({ label: "Merge factor", min: 1, max: 4, value: merge, fmt: (v) => `${v}×${v}`, onInput: (v) => { merge = v; drawVlm(); } }),
          slider({ label: "Images in the prompt", min: 1, max: 64, value: imgs, onInput: (v) => { imgs = v; drawVlm(); } })),
        h("div.stage-body", vst.el, h("p.canvas-cap", "Images are expensive context: a few high-resolution screenshots can cost as much as a long document."))),
    ),
    section("Beyond classification",
      table(["Task / method", "Idea"], [
        ["DINO / DINOv2", "Self-distillation: a student matches a teacher's output on different crops of the same image. No labels; superb general features"],
        ["MAE", "Mask 75% of patches, reconstruct the pixels"],
        ["Segment Anything (SAM)", "Prompt with a point or box, get a mask. Trained on 1 billion masks"],
        ["Detection (DETR)", "A transformer outputs a set of boxes directly, matched to ground truth with Hungarian matching"],
        ["Swin, ConvNeXt", "Bring locality back: windowed attention, or a CNN modernised with transformer-era tricks"],
        ["Vision Mamba", "Replace attention with state-space scans over the patch sequence (see the next chapters)"],
      ]),
    ),
    section("In code",
      code({
        "Patch embedding": `class PatchEmbed(nn.Module):
    def __init__(self, img=224, patch=16, dim=768):
        super().__init__()
        # a conv with stride = kernel = patch size is exactly "cut into patches, then a linear layer"
        self.proj = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, (img // patch) ** 2, dim))
    def forward(self, x):                       # x: (B, 3, 224, 224)
        x = self.proj(x).flatten(2).transpose(1, 2)   # (B, 196, 768)
        return x + self.pos`,
        "CLIP loss": `def clip_loss(img_emb, txt_emb, logit_scale):
    img = F.normalize(img_emb, dim=-1); txt = F.normalize(txt_emb, dim=-1)
    logits = logit_scale.exp() * img @ txt.T        # (B, B) similarity matrix
    labels = torch.arange(len(img))                 # the diagonal holds the matching pairs
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2`,
      }),
    ),
  );
  drawPatches(); newClip(); drawVlm();
}
