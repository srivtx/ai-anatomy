import { h, rng } from "../lib/core.js";
import { walk } from "../lib/walks.js";
import { slider, segmented, stats, predict, code, callout, section, prose, table, bars } from "../lib/widgets.js";
import { xyChart } from "../lib/plot.js";
import { gauss } from "../lib/nn.js";

// floating-point formats: exponent bits, mantissa bits, bias, and whether the all-ones exponent is reserved (IEEE-style)
const FMT = {
  fp32: { e: 8, m: 23, bias: 127, ieee: true, name: "fp32" },
  bf16: { e: 8, m: 7, bias: 127, ieee: true, name: "bf16" },
  fp16: { e: 5, m: 10, bias: 15, ieee: true, name: "fp16" },
  e5m2: { e: 5, m: 2, bias: 15, ieee: true, name: "fp8 E5M2" },
  e4m3: { e: 4, m: 3, bias: 7, ieee: false, name: "fp8 E4M3" },
};
function decode(f, bits) {
  const s = bits[0], E = bits.slice(1, 1 + f.e), M = bits.slice(1 + f.e);
  const e = E.reduce((a, b) => a * 2 + b, 0), m = M.reduce((a, b) => a * 2 + b, 0), emax = 2 ** f.e - 1, mmax = 2 ** f.m - 1;
  const sign = s ? -1 : 1;
  if (f.ieee && e === emax) return m ? NaN : sign * Infinity;
  if (!f.ieee && e === emax && m === mmax) return NaN; // E4M3: only S.1111.111 is NaN; no infinities
  if (e === 0) return sign * 2 ** (1 - f.bias) * (m / 2 ** f.m);
  return sign * 2 ** (e - f.bias) * (1 + m / 2 ** f.m);
}
function stats_(f) {
  const emaxNormal = f.ieee ? 2 ** f.e - 2 : 2 ** f.e - 1;
  const maxMant = f.ieee ? 2 - 2 ** -f.m : 2 - 2 * 2 ** -f.m;
  return { max: 2 ** (emaxNormal - f.bias) * maxMant, minNormal: 2 ** (1 - f.bias), minSub: 2 ** (1 - f.bias - f.m), eps: 2 ** -f.m };
}
const sci = (v) => (!Number.isFinite(v) ? String(v) : v === 0 ? "0" : Math.abs(v) >= 1e5 || Math.abs(v) < 1e-3 ? v.toExponential(3) : String(+v.toPrecision(6)));

export default function (root, { id }) {
  // ---------- 1. bit flipper ----------
  let fk = "bf16", bits;
  const bitRow = h("div.bits"), valEl = h("div.big-out"), fst = stats([{ label: "Largest value" }, { label: "Smallest normal" }, { label: "Relative precision" }]);
  function setFormat(k) { fk = k; const f = FMT[k]; bits = [0, ...Array(f.e).fill(0).map((_, i) => (i === 0 ? 0 : 1)), ...Array(f.m).fill(0)]; drawBits(); }
  function drawBits() {
    const f = FMT[fk], sx = stats_(f);
    bitRow.replaceChildren(...bits.map((b, i) => h("button.bit", { type: "button", class: `bit ${i === 0 ? "sgn" : i <= f.e ? "exp" : "man"}`, "aria-pressed": b ? "true" : "false", title: i === 0 ? "sign" : i <= f.e ? "exponent" : "mantissa", onclick: () => { bits[i] ^= 1; drawBits(); } }, String(b))));
    const v = decode(f, bits), e = bits.slice(1, 1 + f.e).reduce((a, b) => a * 2 + b, 0);
    valEl.innerHTML = `<div style="font-size:13px;color:var(--ink-3)">${f.name}: 1 sign · ${f.e} exponent · ${f.m} mantissa bits</div><div style="font-size:28px;font-weight:700">${sci(v)}</div><div style="color:var(--ink-2);font-size:14px">${Number.isNaN(v) ? "Not a number" : !Number.isFinite(v) ? "Infinity" : e === 0 ? `Subnormal: 2<sup>${1 - f.bias}</sup> × 0.${bits.slice(1 + f.e).join("")}₂` : `2<sup>${e - f.bias}</sup> × 1.${bits.slice(1 + f.e).join("")}₂`}</div>`;
    fst.set(0, sci(sx.max)); fst.set(1, sci(sx.minNormal)); fst.set(2, `~${(-Math.log10(sx.eps)).toFixed(1)} decimal digits`);
  }

  // ---------- 2. gradient underflow ----------
  let med = -6, spread = 1.2, scaleExp = 0;
  const uc = xyChart({ height: 260, xLabel: "log₁₀ |gradient| (after loss scaling)", yLabel: "share of gradients", xDomain: [-12, 6] });
  const ust = stats([{ label: "fp16: flushed to zero" }, { label: "fp16: overflow" }, { label: "bf16: flushed to zero" }, { label: "fp8 E5M2: flushed to zero" }]);
  function drawUnder() {
    const r = rng(4), N = 20000, xs = Array.from({ length: N }, () => med + scaleExp + gauss(r) * spread);
    const bins = new Array(72).fill(0); for (const x of xs) { const b = Math.floor((x + 12) * 4); if (b >= 0 && b < 72) bins[b]++; }
    const frac = (k) => { const sx = stats_(FMT[k]); return [xs.filter((x) => x < Math.log10(sx.minSub / 2)).length / N, xs.filter((x) => x > Math.log10(sx.max)).length / N]; };
    const f16 = frac("fp16"), b16 = frac("bf16"), f8 = frac("e5m2");
    uc.update([{ name: "Gradient magnitudes", color: "var(--accent)", points: bins.map((c, i) => [-12 + i / 4 + 0.125, c / N]) }], {
      xDomain: [-12, 6], marks: [{ x: Math.log10(stats_(FMT.fp16).minSub), label: "fp16 min" }, { x: Math.log10(stats_(FMT.fp16).max), label: "fp16 max" }, { x: Math.log10(stats_(FMT.e5m2).minSub), label: "E5M2 min" }],
    });
    ust.set(0, `${(f16[0] * 100).toFixed(1)}%`, f16[0] > 0.01 ? "bad" : "good"); ust.set(1, `${(f16[1] * 100).toFixed(2)}%`, f16[1] > 0 ? "bad" : "good");
    ust.set(2, `${(b16[0] * 100).toFixed(1)}%`, b16[0] > 0.01 ? "bad" : "good"); ust.set(3, `${(f8[0] * 100).toFixed(1)}%`, f8[0] > 0.01 ? "bad" : "good");
  }

  // ---------- 3. weight quantisation ----------
  let outlier = 8, bitsQ = 4;
  const qst = h("div");
  function drawQuant() {
    const r = rng(9), C = 16, L = 512;
    const W = Array.from({ length: C }, () => { const sc = Math.exp(gauss(r) * 0.6); return Array.from({ length: L }, () => gauss(r) * sc * 0.02); });
    for (let k = 0; k < 6; k++) W[Math.floor(r() * C)][Math.floor(r() * L)] *= outlier; // a few large weights
    const qmax = 2 ** (bitsQ - 1) - 1;
    const quant = (vals) => { const s = Math.max(...vals.map(Math.abs)) / qmax || 1; return vals.map((v) => Math.max(-qmax - 1, Math.min(qmax, Math.round(v / s))) * s); };
    const err = (Q) => { let e = 0, n = 0; W.forEach((row, i) => row.forEach((v, j) => { e += (v - Q[i][j]) ** 2; n += v * v; })); return e / n; };
    const perTensor = (() => { const all = W.flat(), q = quant(all); return W.map((_, i) => q.slice(i * L, (i + 1) * L)); })();
    const perChannel = W.map(quant);
    const group = (g) => W.map((row) => { const out = []; for (let k = 0; k < L; k += g) out.push(...quant(row.slice(k, k + g))); return out; });
    const items = [["Per tensor (one scale)", err(perTensor)], ["Per channel", err(perChannel)], ["Groups of 128", err(group(128))], ["Groups of 32", err(group(32))]];
    qst.replaceChildren(h("div.eyebrow", `int${bitsQ}: relative error (lower is better)`), bars(items.map(([l, e]) => ({ label: l, value: e * 100, color: e < 0.02 ? "var(--good)" : e < 0.08 ? "var(--warn)" : "var(--bad)" })), { max: Math.max(...items.map((x) => x[1] * 100)), fmt: (v) => `${v.toFixed(2)}%` }));
  }

  root.append(
    section(null, prose(`<p>Every number in a neural network is stored in some number of bits, and choosing how many is one of the biggest levers on cost. Halve the bits and you halve memory, halve the bytes moved, and often double the speed of matrix multiplies. The catch: fewer bits means less range (how big and how small a number can be) and less precision (how finely values are spaced). Modern training lives in bf16, frontier labs push into fp8, and inference goes down to 4 bits or below. Here's what each format can and can't represent.</p>`)),
    section("Flip the bits",
      prose("<p>A floating-point number is a sign, an exponent (the range: which power of two) and a mantissa (the precision: how finely you step within that power). Click bits to change them.</p>"),
      h("div.stage",
        h("div.stage-controls", segmented({ label: "Format", options: Object.entries(FMT).map(([k, f]) => ({ value: k, label: f.name })), value: fk, onChange: setFormat })),
        h("div.stage-body", fst.el, bitRow, valEl, h("p.canvas-cap", "Red: sign. Violet: exponent. Teal: mantissa."))),
      table(["Format", "Bits (s/e/m)", "Max", "Precision", "Used for"], [
        ["fp32", "1 / 8 / 23", "3.4 × 10³⁸", "~7 digits", "Master weights, optimizer states, accumulation"],
        ["bf16", "1 / 8 / 7", "3.4 × 10³⁸", "~2–3 digits", "Training compute (same range as fp32, so no loss scaling)"],
        ["fp16", "1 / 5 / 10", "65,504", "~3 digits", "Older mixed-precision training (needs loss scaling), inference"],
        ["fp8 E4M3", "1 / 4 / 3", "448", "~1 digit", "Weights and activations in fp8 training and inference"],
        ["fp8 E5M2", "1 / 5 / 2", "57,344", "<1 digit", "Gradients in fp8 training (more range, less precision)"],
        ["int8 / int4", "integers + a scale", "set by the scale", "256 / 16 levels", "Inference weights (and int8 activations)"],
      ]),
    ),
    section("Why bf16 won training",
      prose("<p>Gradients span many orders of magnitude, and many are tiny. fp16's smallest number is about 6 × 10⁻⁸; anything smaller becomes zero, and that part of the signal is lost. The old fix was <b>loss scaling</b>: multiply the loss by a large constant so gradients shift up into range, then divide afterwards. bf16 has fp32's exponent, so it simply doesn't underflow, at the cost of precision.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Typical gradient size", min: -9, max: -2, step: 0.25, value: med, fmt: (v) => `10^${v}`, onInput: (v) => { med = v; drawUnder(); } }),
          slider({ label: "Spread (decades)", min: 0.3, max: 2.5, step: 0.1, value: spread, fmt: (v) => v.toFixed(1), onInput: (v) => { spread = v; drawUnder(); } }),
          slider({ label: "Loss scale", min: 0, max: 16, step: 1, value: scaleExp, fmt: (v) => `×10^${v}`, onInput: (v) => { scaleExp = v; drawUnder(); } })),
        h("div.stage-body", ust.el, uc.el)),
      callout("tip", "Things to try", "<ul style='margin:0;padding-left:18px'><li>With typical gradients at 10⁻⁶, a large share fall below fp16's smallest value. Raise the loss scale to 10³–10⁴: they shift into range. Raise it too far and the largest overflow to infinity. Dynamic loss scaling searches for the sweet spot every few hundred steps.</li><li>bf16 never underflows here. That's the whole story of why everyone switched: one less thing to tune and to go wrong.</li><li>fp8 E5M2 has fp16's range but only 2 mantissa bits. fp8 training works only with careful per-tensor or per-block scaling, which is why it took until 2024 to be proven at scale.</li></ul>"),
      predict({
        chapter: id,
        question: "You sum 10,000 values of about 1.0 in bf16, one at a time. What goes wrong?",
        options: ["Nothing", "Once the sum reaches 256, adding 1.0 rounds away: with 8 bits of significand, the gap between neighbouring bf16 numbers there is 2. The sum gets stuck far below 10,000", "It overflows", "It becomes NaN"],
        answer: 1,
        explain: "<p>bf16 has 8 significant bits. Between 256 and 512 representable numbers are 2 apart, so 256 + 1 rounds back to 256. That's why matrix multiplies accumulate in fp32 even when inputs are bf16 or fp8, and why optimizer updates use fp32 master weights.</p>",
      }),
    ),
    section("Quantising weights for inference",
      prose("<p>Integer quantisation stores each weight as a small integer times a scale. One scale for a whole tensor is cheap but a single large weight forces a big scale, so every small weight is rounded coarsely. Finer-grained scales (per output channel, or per group of 32–128 weights) fix that for a small memory cost. This is what GPTQ, AWQ and llama.cpp's formats do.</p>"),
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Bits", options: [{ value: 8, label: "int8" }, { value: 4, label: "int4" }, { value: 3, label: "int3" }], value: bitsQ, onChange: (v) => { bitsQ = v; drawQuant(); } }),
          slider({ label: "Outlier size", min: 1, max: 30, step: 1, value: outlier, fmt: (v) => `${v}× typical`, onInput: (v) => { outlier = v; drawQuant(); } })),
        h("div.stage-body", qst)),
      callout("note", "Rules that hold in practice", "<p>8-bit weights are almost free. 4-bit weights with group-wise scales lose a little (often around a point on benchmarks for large models, more for small ones). Below 4 bits quality drops faster unless the model is trained for it (quantisation-aware training, or BitNet-style ternary training). Activations are harder than weights because of outlier features; see the deeper cuts below.</p>"),
    ),
    walk("precision"),
    section("In code",
      code({
        "bf16 autocast": `model = model.cuda()                         # fp32 master weights
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
for x, y in loader:
    with torch.autocast("cuda", dtype=torch.bfloat16):   # matmuls in bf16, accumulation in fp32
        loss = loss_fn(model(x), y)
    loss.backward(); opt.step(); opt.zero_grad()       # no GradScaler needed for bf16`,
        "Group-wise int4": `def quantize_int4(w, group=128):
    w = w.reshape(-1, group)
    scale = w.abs().amax(dim=1, keepdim=True) / 7          # int4 range: -8..7
    q = (w / scale).round().clamp(-8, 7).to(torch.int8)
    return q, scale                                          # dequant: q * scale`,
      }, "Project 31 measures real GPU memory for each precision and reproduces fp16 underflow."),
    ),
  );
  setFormat("bf16"); drawUnder(); drawQuant();
}
