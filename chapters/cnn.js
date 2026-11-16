import { h } from "../lib/core.js";
import { goal, whyChain, sticker } from "../lib/why.js";
import { walk } from "../lib/walks.js";
import { slider, segmented, stats, predict, code, callout, section, prose, table, button, toggle } from "../lib/widgets.js";

const N = 28;
const KERNELS = {
  "Vertical edges": [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
  "Horizontal edges": [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
  Blur: [[1, 2, 1], [2, 4, 2], [1, 2, 1]].map((r) => r.map((v) => v / 16)),
  Sharpen: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
  "Diagonal ↘": [[2, 1, 0], [1, 0, -1], [0, -1, -2]],
  Outline: [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
};

// a hand-drawn "4" so there's something to look at before you draw
function seedImage() {
  const img = new Float64Array(N * N);
  const stroke = (x0, y0, x1, y1) => { for (let t = 0; t <= 1; t += 0.01) { const x = Math.round(x0 + (x1 - x0) * t), y = Math.round(y0 + (y1 - y0) * t); for (const [dx, dy] of [[0, 0], [1, 0], [0, 1], [1, 1]]) if (x + dx < N && y + dy < N) img[(y + dy) * N + x + dx] = 1; } };
  stroke(16, 4, 7, 17); stroke(7, 17, 21, 17); stroke(17, 9, 17, 24);
  return img;
}
function conv(img, n, k) {
  const m = n - 2, out = new Float64Array(m * m);
  for (let y = 0; y < m; y++) for (let x = 0; x < m; x++) { let s = 0; for (let j = 0; j < 3; j++) for (let i = 0; i < 3; i++) s += img[(y + j) * n + x + i] * k[j][i]; out[y * m + x] = s; }
  return out;
}
function pool(img, n) {
  const m = Math.floor(n / 2), out = new Float64Array(m * m);
  for (let y = 0; y < m; y++) for (let x = 0; x < m; x++) out[y * m + x] = Math.max(img[2 * y * n + 2 * x], img[2 * y * n + 2 * x + 1], img[(2 * y + 1) * n + 2 * x], img[(2 * y + 1) * n + 2 * x + 1]);
  return out;
}

function pixelCanvas(n, label) {
  const c = h("canvas", { width: n, height: n, role: "img", "aria-label": label, style: { width: "100%", aspectRatio: "1", imageRendering: "pixelated", display: "block", borderRadius: "10px", border: "1px solid var(--line)", background: "var(--surface)", touchAction: "none" } });
  return {
    el: c, n,
    paint(data, { signed = false, hi = null } = {}) {
      const ctx = c.getContext("2d"), img = ctx.createImageData(n, n);
      const cs = getComputedStyle(document.documentElement);
      const hex = (v) => { const m = cs.getPropertyValue(v).trim().match(/[0-9a-f]{2}/gi) || ["80", "80", "80"]; return m.map((x) => parseInt(x, 16)); };
      const bg = hex("--surface"), pos = hex("--c1"), neg = hex("--c2"), ink = hex("--ink");
      const max = Math.max(1e-9, ...Array.from(data, Math.abs));
      for (let i = 0; i < n * n; i++) {
        const v = data[i] / max, t = Math.min(1, Math.abs(v)), col = signed ? (v >= 0 ? pos : neg) : ink;
        for (let k = 0; k < 3; k++) img.data[i * 4 + k] = bg[k] + (col[k] - bg[k]) * t;
        img.data[i * 4 + 3] = 255;
        if (hi && hi.has(i)) { img.data[i * 4] = img.data[i * 4] * 0.5 + 240 * 0.5; img.data[i * 4 + 1] *= 0.6; img.data[i * 4 + 2] = img.data[i * 4 + 2] * 0.5 + 40; }
      }
      ctx.putImageData(img, 0, 0);
    },
  };
}

export default function (root, { id }) {
  let img = seedImage(), kname = "Vertical edges", K = KERNELS[kname].map((r) => [...r]), relu = true, sel = null;
  const inC = pixelCanvas(N, "Your drawing, 28 by 28 pixels"), fC = pixelCanvas(N - 2, "Feature map after the convolution"), pC = pixelCanvas((N - 2) / 2, "Feature map after 2 by 2 max pooling");
  const kGrid = h("div", { style: { display: "grid", gridTemplateColumns: "repeat(3, 58px)", gap: "4px" } });
  const st = stats([{ label: "Weights in this conv layer" }, { label: "Weights in a dense layer doing the same map" }]);
  function kernelInputs() {
    kGrid.replaceChildren(...K.flatMap((row, j) => row.map((v, i) => {
      const inp = h("input", { type: "number", step: "0.5", value: +v.toFixed(3), "aria-label": `Kernel weight row ${j + 1} column ${i + 1}`, style: { width: "58px", padding: "6px", borderRadius: "8px", border: "1px solid var(--line)", background: "var(--surface)", font: "13px var(--mono)" } });
      inp.addEventListener("input", () => { K[j][i] = +inp.value || 0; draw(); });
      return inp;
    })));
  }
  function draw() {
    let f = conv(img, N, K);
    if (relu) f = f.map((v) => Math.max(0, v));
    const p = pool(f, N - 2);
    let hiIn = null, hiF = null;
    if (sel) { // receptive field of one pooled unit: 2×2 in the feature map, 4×4 in the input
      const [px, py] = sel;
      hiF = new Set([[0, 0], [1, 0], [0, 1], [1, 1]].map(([dx, dy]) => (2 * py + dy) * (N - 2) + 2 * px + dx));
      hiIn = new Set();
      for (let dy = 0; dy < 4; dy++) for (let dx = 0; dx < 4; dx++) hiIn.add((2 * py + dy) * N + 2 * px + dx);
    }
    inC.paint(img, { hi: hiIn }); fC.paint(f, { signed: !relu, hi: hiF }); pC.paint(p, { signed: !relu, hi: sel ? new Set([sel[1] * 13 + sel[0]]) : null });
    st.set(0, "9 (+1 bias)"); st.set(1, `${(784 * 676).toLocaleString()}`);
  }
  // drawing
  let drawing = false, erase = false;
  const paintAt = (e) => {
    const r = inC.el.getBoundingClientRect(), x = Math.floor(((e.clientX - r.left) / r.width) * N), y = Math.floor(((e.clientY - r.top) / r.height) * N);
    for (const [dx, dy, w] of [[0, 0, 1], [1, 0, 0.6], [0, 1, 0.6], [-1, 0, 0.6], [0, -1, 0.6]]) {
      const xx = x + dx, yy = y + dy;
      if (xx >= 0 && yy >= 0 && xx < N && yy < N) img[yy * N + xx] = erase ? 0 : Math.max(img[yy * N + xx], w);
    }
    draw();
  };
  inC.el.addEventListener("pointerdown", (e) => { drawing = true; inC.el.setPointerCapture(e.pointerId); paintAt(e); });
  inC.el.addEventListener("pointermove", (e) => drawing && paintAt(e));
  inC.el.addEventListener("pointerup", () => (drawing = false));
  pC.el.addEventListener("pointerdown", (e) => { const r = pC.el.getBoundingClientRect(); sel = [Math.floor(((e.clientX - r.left) / r.width) * 13), Math.floor(((e.clientY - r.top) / r.height) * 13)]; draw(); });

  // receptive field calculator
  let layers = 4, ks = 3, stride = 1, poolEvery = true;
  const rf = stats([{ label: "Receptive field of one output unit" }, { label: "Output resolution (from 224 px)" }]);
  function calcRF() {
    let r = 1, jump = 1, res = 224;
    for (let l = 0; l < layers; l++) {
      r += (ks - 1) * jump; jump *= stride; res = Math.ceil(res / stride);
      if (poolEvery) { r += (2 - 1) * jump; jump *= 2; res = Math.floor(res / 2); }
    }
    rf.set(0, `${r} × ${r} px`); rf.set(1, `${res} × ${res}`);
  }

  root.append(
    section(null, goal("<p>An image is a grid where <b>nearby pixels go together</b> and the same thing can appear anywhere. We want a layer that looks locally, reuses one detector everywhere, and builds big shapes from small ones.</p>", ["small local detectors", "the same weights at every position", "layers that see wider and wider"]),
      prose(`<p>A 28 × 28 image has 784 pixels. A dense layer looking at it connects every pixel to every unit: hundreds of thousands of weights, and if the digit moves one pixel to the right everything it learned is in the wrong place. A <b>convolution</b> instead slides one small filter, say 3 × 3, across the whole image, computing the same weighted sum at every position. It has two built-in assumptions: <b>locality</b> (nearby pixels matter together) and <b>translation equivariance</b> (an edge is an edge wherever it is).</p>`),
      h("div.vx-row", sticker("conv", { title: "A 3 × 3 filter", caption: "sliding over an image; here it sits on a dark-to-light edge", px: 4 }))),
    section("Draw, then convolve",
      prose("<p>Draw on the left grid (it starts with a 4). Pick a kernel or type your own weights. The middle map is the kernel's response at every position; the right map keeps the maximum of each 2 × 2 block (<b>max pooling</b>). Click a pixel on the right to see which input pixels it depends on: its <b>receptive field</b>.</p>"),
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Kernel", options: Object.keys(KERNELS).map((k) => ({ value: k, label: k })), value: kname, onChange: (v) => { kname = v; K = KERNELS[v].map((r) => [...r]); kernelInputs(); draw(); } }),
          toggle({ label: "ReLU after the convolution", checked: true, onChange: (v) => { relu = v; draw(); } }),
          button("Clear drawing", () => { img = new Float64Array(N * N); sel = null; draw(); }),
          button("Eraser on/off", () => { erase = !erase; })),
        h("div.stage-body", st.el,
          h("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "14px", alignItems: "start" } },
            h("div", h("div.eyebrow", "Input 28×28"), inC.el, h("p.canvas-cap", "Draw here")),
            h("div", h("div.eyebrow", "Kernel 3×3"), kGrid, h("p.canvas-cap", "Edit any weight")),
            h("div", h("div.eyebrow", "Feature map 26×26"), fC.el, h("p.canvas-cap", relu ? "Positive responses" : "Blue +, orange −")),
            h("div", h("div.eyebrow", "Max-pooled 13×13"), pC.el, h("p.canvas-cap", "Click a pixel"))))),
      callout("tip", "Things to try", "<ul style='margin:0;padding-left:18px'><li>Vertical edges lights up the 4's upright stroke; horizontal edges lights up its crossbar. A trained CNN's first layer learns filters very like these on its own.</li><li>Turn ReLU off: the response has a sign. Positive on one side of a stroke, negative on the other.</li><li>Draw the same shape in another corner. The feature map shows the same pattern, moved. That's translation equivariance; pooling makes it close to invariance.</li></ul>"),
      predict({
        chapter: id,
        question: "Stack ten 3×3 conv layers with no pooling. How wide is the patch of input that one final unit can see?",
        options: ["3 × 3", "21 × 21", "30 × 30", "The whole image"],
        answer: 1,
        explain: "<p>Each 3×3 layer adds 2 pixels to the receptive field: 1 + 10 × 2 = 21. Pooling or striding doubles the growth per layer after it. That's why CNNs alternate convolutions with downsampling: deep layers see big regions (whole objects) while early layers see edges.</p>",
      }),
    ),
    section("How far can a deep unit see?",
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Conv layers", min: 1, max: 12, value: layers, onInput: (v) => { layers = v; calcRF(); } }),
          slider({ label: "Kernel size", min: 1, max: 7, step: 2, value: ks, onInput: (v) => { ks = v; calcRF(); } }),
          slider({ label: "Stride", min: 1, max: 2, value: stride, onInput: (v) => { stride = v; calcRF(); } }),
          toggle({ label: "2×2 pooling after each layer", checked: true, onChange: (v) => { poolEvery = v; calcRF(); } })),
        h("div.stage-body", rf.el)),
      table(["Architecture", "Year", "Idea it added"], [
        ["LeNet-5", "1998", "Conv + pool + dense for digits (cheques at banks)"],
        ["AlexNet", "2012", "Deep CNN on GPUs + ReLU + dropout; won ImageNet by a mile"],
        ["VGG", "2014", "Just stack 3×3 convs"],
        ["ResNet", "2015", "Residual connections: 152 layers"],
        ["U-Net", "2015", "Down then up with skip connections: segmentation, and diffusion models"],
        ["EfficientNet / ConvNeXt", "2019–22", "Careful scaling; a CNN modernised with transformer tricks"],
        ["ViT", "2020", "No convolutions: patches + attention (see the vision chapter)"],
      ]),
    ),
    walk("cnn"),
    section("Why so few weights",
      whyChain({
        claim: "A conv layer with 64 filters of 3 × 3 on 64 channels has 37 thousand weights. A dense layer doing the same job on a 224 × 224 image would need about 10 trillion.",
        steps: [
          { q: "How does it get away with so few?", a: "<p>It makes two bets. <b>Locality</b>: a pixel's meaning depends mostly on its neighbours, so each output looks only at a 3 × 3 patch. <b>Sharing</b>: an edge is an edge anywhere, so the same 3 × 3 weights are used at every position.</p>" },
          { q: "Why does sharing help learning, not just memory?", a: "<p>Every position is a training example for the same filter. A cat in the corner teaches the same weights as a cat in the centre. And if the image shifts, the output map shifts with it.</p>" },
          { q: "How does it see whole objects through 3 × 3 windows?", a: "<p>Stacking. Two 3 × 3 layers see 5 × 5; ten see 21 × 21; pooling and strides double the reach at each stage. Edges become textures, textures become parts, parts become objects.</p>" },
        ],
        axiom: { name: "Symmetry: if a rule shouldn't depend on position, build that in", html: "<p>Physics is the same here and there, so photos have roughly the same statistics everywhere in the frame. A model that respects a real symmetry needs fewer parameters and less data, because it doesn't relearn the same thing at every position. Convolution is translation symmetry written as a layer.</p>" },
      })),
    section("In code",
      code({
        PyTorch: `import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28 -> 14
    nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14 -> 7
    nn.Flatten(),
    nn.Linear(64 * 7 * 7, 10),
)
print(sum(p.numel() for p in model.parameters()))   # ~ 50k; a dense net this good needs far more`,
      }, "Project 03 trains this on MNIST and compares it with an MLP."),
    ),
  );
  kernelInputs(); draw(); calcRF();
}
