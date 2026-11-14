import { h, loop, rng } from "../lib/core.js";
import { slider, stats, predict, code, callout, section, prose, table, stepper, segmented, toggle, bars } from "../lib/widgets.js";
import { xyChart } from "../lib/plot.js";
import { gauss } from "../lib/nn.js";
import { coordCheck } from "../lib/mup.js";

const WIDTHS = [64, 128, 256, 512, 1024];

export default function (root, { id }) {
  // ---------- 1. μP coordinate check (live) ----------
  const cc = xyChart({ width: 520, height: 280, xLog: true, yLog: true, xLabel: "width", yLabel: "RMS change of hidden activations", xDomain: [50, 1300], yDomain: [0.05, 2] });
  const cprog = h("p.canvas-cap", "");
  const res = { sp: {}, mup: {} };
  let queue = [];
  function drawCC() {
    const pts = (p) => WIDTHS.filter((w) => res[p][w]).map((w) => [w, res[p][w].reduce((a, b) => a + b, 0) / res[p][w].length]);
    cc.update([
      { name: "Standard parametrisation", color: "var(--bad)", points: pts("sp"), dots: true },
      { name: "μP", color: "var(--good)", points: pts("mup"), dots: true },
    ], { xDomain: [50, 1300], yDomain: [0.05, 2] });
  }
  const cl = loop(() => {
    const t0 = performance.now();
    while (queue.length && performance.now() - t0 < 30) { const [p, w, s] = queue.shift(); (res[p][w] ??= []).push(coordCheck(w, p, { seed: s }).dHidden); }
    drawCC();
    const total = WIDTHS.length * 2 * 3;
    cprog.textContent = queue.length ? `Training ${total - queue.length} of ${total} networks…` : "30 networks, 4 Adam steps each, same learning rate. Each point is the mean of 3 seeds.";
    if (!queue.length) cl.stop();
  }, cc.el);
  function runCC() { res.sp = {}; res.mup = {}; queue = []; for (const w of WIDTHS) for (const p of ["sp", "mup"]) for (const s of [1, 2, 3]) queue.push([p, w, s]); cl.start(); }

  // ---------- 2. LR transfer (illustrative, shape from Tensor Programs V) ----------
  let lrParam = "sp";
  const lc = xyChart({ width: 520, height: 280, xLabel: "log₂ learning rate", yLabel: "training loss", xDomain: [-14, -4], yDomain: [3.0, 5.0] });
  function drawLR() {
    const ws = [128, 512, 2048, 8192], cols = ["var(--c4)", "var(--c2)", "var(--c1)", "var(--accent)"];
    lc.update(ws.map((w, i) => {
      const k = Math.log2(w / 128), base = 3.9 - 0.12 * k * (lrParam === "mup" ? 1 : 0.4);
      const opt = lrParam === "mup" ? -9.5 : -8 - k * 1.0, div = opt + 2.2;
      const pts = [];
      for (let x = -14; x <= -4; x += 0.25) { let y = base + 0.06 * (x - opt) ** 2; if (x > div) y += 0.9 * (x - div) ** 1.5; pts.push([x, Math.min(5, y)]); }
      return { name: `width ${w}`, color: cols[i], points: pts };
    }), { xDomain: [-14, -4], yDomain: [3.0, 5.0] });
  }

  // ---------- 3. attention logit growth ----------
  const at = { scale: 1, qkn: false };
  const ast = stats([{ label: "Largest attention weight" }, { label: "Attention entropy" }, { label: "Max logit" }]);
  const aEl = h("div");
  const r0 = rng(4), d = 64, keys = Array.from({ length: 24 }, () => Array.from({ length: d }, () => gauss(r0))), q0 = Array.from({ length: d }, () => gauss(r0));
  function drawAttn() {
    const norm = (v) => { const n = Math.hypot(...v); return v.map((x) => x / n); };
    // weight growth during training multiplies both q and k, so logits grow with scale²
    const q = at.qkn ? norm(q0) : q0.map((x) => x * at.scale), ks = keys.map((k) => (at.qkn ? norm(k) : k.map((x) => x * at.scale)));
    const g = at.qkn ? 2 * Math.sqrt(d) * 0.35 : 1;   // QK-norm: unit vectors times a learned gain (fixed here)
    const logits = ks.map((k) => (k.reduce((a, x, i) => a + x * q[i], 0) * g) / Math.sqrt(d));
    const mx = Math.max(...logits), e = logits.map((l) => Math.exp(l - mx)), s = e.reduce((a, b) => a + b, 0), p = e.map((x) => x / s);
    const H = -p.reduce((a, x) => a + (x > 0 ? x * Math.log2(x) : 0), 0);
    ast.set(0, `${(Math.max(...p) * 100).toFixed(0)}%`, Math.max(...p) > 0.9 ? "bad" : Math.max(...p) > 0.5 ? "warn" : "good");
    ast.set(1, `${H.toFixed(2)} of ${Math.log2(24).toFixed(2)} bits`, H < 0.5 ? "bad" : H < 2 ? "warn" : "good");
    ast.set(2, mx.toFixed(1), mx > 20 ? "bad" : "");
    const top = Math.max(...p);
    aEl.replaceChildren(h("div.eyebrow", "Attention of one query over 24 keys"),
      h("div.cols", p.map((x, i) => h("span", { title: `key ${i + 1}: ${(x * 100).toFixed(1)}%`, style: { height: `${Math.max(1, x * 100)}%`, background: x === top ? "var(--accent)" : "var(--c4)" } }))),
      h("p.canvas-cap", "Each column is one key; height is its attention weight (0–100%)."));
  }

  root.append(
    section(null, prose(`<p>A small model that diverges costs you an afternoon. A frontier run that diverges at step 200,000 costs weeks and millions of dollars. Labs have accumulated a body of hard-won tricks for keeping big runs stable and for choosing hyperparameters without being able to afford a sweep at full size. Little of it is in textbooks; most is scattered across appendices of papers like PaLM, OPT, Gopher and OLMo, and team folklore. This chapter collects the core of it, from first principles.</p>`)),
    section("Hyperparameters that transfer across width: μP",
      prose("<p>You can't sweep the learning rate on a 100B-parameter model. You want to tune on a small model and reuse the setting. With the usual (“standard”) parametrisation that fails: as width grows, each weight update changes the activations more, because a layer sums over more coordinates that all moved in a correlated direction. <b>μP</b> (maximal update parametrisation, Yang & Hu 2021; Yang et al. 2022) rescales initialisation and per-layer learning rates with width so every layer's activations change by about the same amount at any width. A <b>coordinate check</b> tests this directly. The networks below are trained in your browser.</p>"),
      h("div.lab", h("div", cc.el, cprog), h("div", h("div.stage-controls", { style: { borderRadius: "12px" } }, segmented({ label: "Learning-rate sweep (illustrative)", options: [{ value: "sp", label: "Standard" }, { value: "mup", label: "μP" }], value: lrParam, onChange: (v) => { lrParam = v; drawLR(); } })), lc.el)),
      callout("tip", "Reading the two charts", "<p><b>Left (live)</b>: with the same learning rate, standard parametrisation moves the hidden activations more and more as the network widens: at large width, the same LR that was fine at small width blows up. Under μP the change stays roughly constant. <b>Right</b>: the shape reported in “Tensor Programs V” (Yang et al., 2022, Fig. 1), redrawn: in standard parametrisation the best learning rate shifts with width and wider models can even be worse; under μP the curves share one optimum and wider is always better. They tuned a 40M-parameter proxy and transferred to GPT-3 6.7B, matching a model tuned at full size for ~7% of the tuning cost.</p>"),
      table(["Layer (Adam)", "Standard init / LR", "μP init / LR"], [
        ["Embedding / input", "var 1/fan_in, LR η", "var 1/fan_in, LR η"],
        ["Hidden matrices", "var 1/fan_in, LR η", "var 1/fan_in, LR η / m"],
        ["Output (unembedding)", "var 1/fan_in, LR η", "var 1/fan_in² (or zero), LR η / m"],
        ["Attention logits", "q·k / √d", "q·k / d"],
      ]),
      prose("<p style='font-size:13.5px;color:var(--ink-2)'>Here <i>m</i> is the width multiplier relative to the tuned base model. Several labs report using μP or close variants (e.g. Cerebras-GPT, MiniCPM, xAI's Grok-1 code ships μP-style multipliers); others instead fit scaling laws for the optimal LR and batch size directly (DeepSeek LLM, 2024). Depth needs its own treatment (depth-μP / residual scaling).</p>"),
    ),
    section("Attention logit growth and QK-norm",
      prose("<p>A common way large transformers die: during training the query and key weights slowly grow, attention logits grow with the <i>square</i> of that growth, softmax saturates onto one key, its gradients vanish, and the loss spikes or diverges. Dehghani et al. (2023) hit this training ViT-22B. The fix is <b>QK-norm</b>: normalise each query and key vector (LayerNorm or RMSNorm) before the dot product, so logits can't grow without bound.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Growth of q and k weights during training", min: 1, max: 4, step: 0.05, value: at.scale, fmt: (v) => `${v.toFixed(2)}×`, onInput: (v) => { at.scale = v; drawAttn(); } }),
          toggle({ label: "QK-norm", checked: at.qkn, onChange: (v) => { at.qkn = v; drawAttn(); } })),
        h("div.stage-body", ast.el, aEl)),
      callout("note", "The related failure at the output: z-loss", "<p>The softmax over the vocabulary has the same problem: logits can drift upward together (softmax doesn't care about a constant shift), until bf16 rounding breaks. PaLM added <b>z-loss</b>, a tiny penalty 10⁻⁴·(log Z)² on the log of the softmax normaliser, which keeps log Z near zero. OLMo 2 (2024) uses both QK-norm and z-loss and reports far fewer spikes.</p>"),
    ),
    section("When the loss spikes anyway",
      stepper([
        { title: "1. Detect", html: "<p>Log loss, gradient norm, update/weight ratio per layer, max attention logit and max output logit every step. A spike is usually preceded by a rising gradient norm or logit maximum in one specific layer.</p>" },
        { title: "2. Rewind and skip", html: "<p>PaLM's recipe (Chowdhery et al., 2022): restart from a checkpoint ~100 steps before the spike and skip the next 200–500 batches. The same batches rarely cause a spike twice in a different state: spikes come from an unlucky interaction of data and parameters, not from “bad data” alone.</p>" },
        { title: "3. Turn the knobs that help", html: "<p>Lower the LR temporarily, lower Adam's β₂ (0.95 rather than 0.999 makes the second moment adapt faster to sudden large gradients), raise ε if updates explode on tiny-variance parameters, clip gradients (norm 1.0 is standard).</p>" },
        { title: "4. Fix the cause for next time", html: "<p>Architectural fixes beat babysitting: QK-norm, z-loss, no bias terms, removing weight decay from embeddings, careful init of residual output projections (scale by 1/√(2·layers)), and avoiding fp16 (bf16's range removes most overflow).</p>" },
        { title: "5. Accept some", html: "<p>Small spikes the run recovers from on its own are common and mostly harmless. Large runs schedule automatic rewinds and keep a human on call. Meta's OPT-175B logbook (2022) records dozens of manual restarts.</p>" },
      ]),
      table(["Symptom", "Likely cause", "Standard fix"], [
        ["Loss spike, max attention logit huge", "Attention logit growth", "QK-norm; lower LR"],
        ["Slow upward drift of output logits, then NaN", "Unbounded softmax normaliser", "z-loss; bf16 not fp16"],
        ["Spikes in one layer's gradient norm", "Adam second moment too slow to adapt", "β₂ = 0.95; gradient clipping"],
        ["Loss plateaus high at large width only", "LR tuned at small width doesn't transfer", "μP, or refit LR scaling"],
        ["Divergence right after warmup", "Peak LR too high for the batch size", "Longer warmup, lower peak"],
        ["Late-training instability", "LR too high for the sharpening loss landscape", "Decay phase (WSD/cosine), lower LR"],
        ["Loss of plasticity, dead units", "Too long at high LR with weight decay on norms", "Exclude norms/biases from decay"],
      ]),
      predict({
        chapter: id,
        question: "A 7B run spikes at step 40k. You rewind 100 steps and replay exactly the same batches with the same seed. What usually happens?",
        options: ["The spike never recurs", "The spike usually recurs at the same place, because the run is deterministic: you must also change something, typically skipping the batches that follow", "The model is permanently damaged", "Loss spikes are always caused by corrupt data"],
        answer: 1,
        explain: "<p>A deterministic replay retraces the same path. PaLM's fix was to skip a few hundred batches after the rewind: from a slightly different parameter state, the trajectory avoids the instability. That the same data is harmless in a different state is evidence spikes come from data–parameter interactions, not bad examples.</p>",
      }),
    ),
    section("In code",
      code({
        "μP with Adam (manual)": `def mup_param_groups(model, base_width, width, lr):
    m = width / base_width
    groups = {"input": [], "hidden": [], "output": []}
    for name, p in model.named_parameters():
        if "embed" in name: groups["input"].append(p)
        elif "lm_head" in name: groups["output"].append(p)
        else: groups["hidden"].append(p)
    return [{"params": groups["input"], "lr": lr},
            {"params": groups["hidden"], "lr": lr / m},
            {"params": groups["output"], "lr": lr / m}]

nn.init.normal_(model.lm_head.weight, std=1 / width)         # or zeros
# attention: scores = q @ k.transpose(-1, -2) / d_head          (1/d, not 1/sqrt(d))
opt = torch.optim.AdamW(mup_param_groups(model, 256, 4096, lr=1e-2), betas=(0.9, 0.95))`,
        "Coordinate check": `def coord_check(make_model, widths, steps=4):
    for w in widths:
        model, opt = make_model(w)
        acts = {}
        hooks = [m.register_forward_hook(lambda m, i, o, n=n: acts.__setitem__(n, o.detach()))
                 for n, m in model.named_modules() if isinstance(m, nn.Linear)]
        x = torch.randn(64, model.d_in)
        model(x); before = {k: v.clone() for k, v in acts.items()}
        for _ in range(steps):
            loss = model(x).pow(2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        model(x)
        print(w, {k: (acts[k] - before[k]).pow(2).mean().sqrt().item() for k in acts})   # should be flat in w`,
        "QK-norm and z-loss": `class Attention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.qkv = nn.Linear(d, 3 * d, bias=False); self.o = nn.Linear(d, d, bias=False)
        self.q_norm = nn.RMSNorm(d // n_heads); self.k_norm = nn.RMSNorm(d // n_heads); self.h = n_heads
    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, D // self.h).unbind(2)
        q, k = self.q_norm(q), self.k_norm(k)                     # logits can no longer grow without bound
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, T, D))

def lm_loss(logits, targets, z_coef=1e-4):
    ce = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
    z = torch.logsumexp(logits.float(), dim=-1)                    # log of the softmax normaliser
    return ce + z_coef * (z ** 2).mean()`,
        "Spike detection with rewind": `ema, ema_sq = None, None
for step, batch in loader:
    loss = train_step(batch)
    if ema is not None and loss > ema + 6 * (ema_sq - ema ** 2) ** 0.5 and step > 1000:
        ckpt = latest_checkpoint(before=step - 100)
        load(ckpt); loader.skip(300)                          # PaLM-style: rewind, then skip ahead
        log.warning(f"spike at {step}: loss {loss:.3f}, rewound to {ckpt.step}")
        continue
    ema = loss if ema is None else 0.99 * ema + 0.01 * loss
    ema_sq = loss ** 2 if ema_sq is None else 0.99 * ema_sq + 0.01 * loss ** 2`,
      }, "Project 35 runs the coordinate check and a learning-rate sweep across widths, with and without μP."),
    ),
  );
  drawLR(); drawAttn(); drawCC(); runCC();
}
