import { h } from "../lib/core.js";
import { slider, segmented, stats, predict, code, callout, section, prose, table, toggle, dilemma } from "../lib/widgets.js";
import { fmtSI, fmtBytes } from "../lib/plot.js";

const CONVO = [
  { role: "system", text: "You are a concise assistant." },
  { role: "user", text: "What does a lighthouse do?" },
  { role: "assistant", text: "It marks dangerous coasts and harbour entrances with a light so ships can navigate safely at night." },
  { role: "user", text: "And in fog?" },
  { role: "assistant", text: "Many also sound a foghorn, and today radio beacons and GPS do most of the work." },
];
const TEMPLATES = {
  chatml: { label: "ChatML (Qwen, many open models)", open: (r) => `<|im_start|>${r}\n`, close: "<|im_end|>\n", bos: "" },
  llama3: { label: "Llama 3", open: (r) => `<|start_header_id|>${r}<|end_header_id|>\n\n`, close: "<|eot_id|>", bos: "<|begin_of_text|>" },
};
const tok = (s) => s.match(/<\|[^|]+\|>|\n+| ?[A-Za-z']+| ?[^\sA-Za-z]/g) || [];

export default function (root, { id }) {
  let tpl = "chatml", mask = true;
  const chips = h("div.chips", { style: { lineHeight: 2 } }), st = stats([{ label: "Tokens in the example" }, { label: "Tokens that get a loss" }, { label: "Share trained on" }]);
  function drawTpl() {
    const T = TEMPLATES[tpl], pieces = [];
    if (T.bos) pieces.push({ t: T.bos, train: false, special: true });
    for (const m of CONVO) {
      pieces.push({ t: T.open(m.role), train: false, special: true });
      pieces.push({ t: m.text, train: m.role === "assistant" });
      pieces.push({ t: T.close, train: m.role === "assistant", special: true }); // the model must learn to stop
    }
    let total = 0, trained = 0;
    chips.replaceChildren(...pieces.flatMap((p) => tok(p.t).map((x) => {
      total++; const on = !mask || p.train; if (on) trained++;
      return h("span.chip", { class: on ? "chip" : "chip masked", title: on ? "contributes to the loss" : "masked: no loss", style: { background: p.special ? "color-mix(in srgb, var(--c4) 30%, transparent)" : on ? "color-mix(in srgb, var(--c3) 28%, transparent)" : "var(--sunken)" } }, x.replace(/\n/g, "↵"));
    })));
    st.set(0, String(total)); st.set(1, String(trained)); st.set(2, `${Math.round((trained / total) * 100)}%`);
  }

  // LoRA calculator
  const lc = { d: 4096, layers: 32, r: 16, targets: "all", ff: 14336 };
  const lst = stats([{ label: "Trainable parameters" }, { label: "Share of the model" }, { label: "Optimizer + gradient memory" }, { label: "…vs full fine-tune" }]);
  function drawLoRA() {
    const d = lc.d, ff = lc.ff, r = lc.r, kv = d / 4; // grouped-query K/V like Llama 3 8B
    const mats = { attn: [[d, d], [d, kv], [d, kv], [d, d]], all: [[d, d], [d, kv], [d, kv], [d, d], [d, ff], [d, ff], [ff, d]] }[lc.targets];
    const lora = lc.layers * mats.reduce((a, [i, o]) => a + r * (i + o), 0);
    const full = 8.0e9;
    lst.set(0, fmtSI(lora)); lst.set(1, `${((lora / full) * 100).toFixed(2)}%`, "good");
    // AdamW in fp32: grad (4 B) + two moments (8 B) + fp32 master copy (4 B) per trainable parameter
    lst.set(2, fmtBytes(lora * 16)); lst.set(3, fmtBytes(full * 16), "bad");
  }

  root.append(
    section(null, prose(`<p>A pretrained <b>base model</b> is a superb autocomplete. Ask it “What does a lighthouse do?” and it may continue with three more quiz questions, because that's what usually follows a question on the web. <b>Supervised fine-tuning</b> (SFT) shows it thousands of example conversations so it learns a new pattern: user speaks, assistant answers helpfully, assistant stops. Same loss, same next-token objective; only the data changes.</p>`)),
    section("A conversation is just tokens",
      prose("<p>Conversations are flattened into one token sequence with <b>special tokens</b> marking who speaks (the <b>chat template</b>). During SFT we usually compute the loss only on the assistant's tokens: the model shouldn't learn to write the user's questions, only to answer them. Green tokens are trained on; struck-through grey ones are masked.</p>"),
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Chat template", options: Object.entries(TEMPLATES).map(([k, v]) => ({ value: k, label: v.label })), value: tpl, onChange: (v) => { tpl = v; drawTpl(); } }),
          toggle({ label: "Mask the prompt (loss on assistant tokens only)", checked: true, onChange: (v) => { mask = v; drawTpl(); } })),
        h("div.stage-body", st.el, chips)),
      callout("tip", "Look closely", "<ul style='margin:0;padding-left:18px'><li>The end-of-turn token after each assistant reply <i>is</i> trained on. That's how the model learns to stop talking. Forget it and your fine-tuned model rambles forever.</li><li>Templates differ between model families. Using the wrong one at inference time quietly degrades a model: it has never seen those separators.</li></ul>"),
      predict({
        chapter: id,
        question: "Why mask the loss on user and system tokens during SFT?",
        options: ["To save compute", "So the model learns to produce answers, not to imitate users; and so long prompts don't dominate the loss", "Because those tokens can't be tokenized", "Masking is required by the chat template"],
        answer: 1,
        explain: "<p>The forward pass still reads the prompt (the model needs it as context); only the gradient is restricted to the response. Without masking, a dataset with long documents and short answers would spend most of its training signal teaching the model to write documents. Some recipes do keep a little prompt loss; it's a tunable choice.</p>",
      }),
    ),
    section("LoRA: fine-tune 0.5% of the weights",
      prose("<p>Fine-tuning all 8 billion weights needs memory for gradients and optimizer state too: roughly 16 bytes per parameter with AdamW. <b>LoRA</b> (Hu et al., 2021) freezes the original weights W and learns a low-rank update instead: <code>W′ = W + B·A</code>, where B is d × r and A is r × d with r as small as 8 or 16. Fine-tuning changes turn out to be low-rank, so this loses little quality.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Rank r", min: 1, max: 256, value: lc.r, onInput: (v) => { lc.r = v; drawLoRA(); } }),
          segmented({ label: "Apply LoRA to", options: [{ value: "attn", label: "Attention only (q, k, v, o)" }, { value: "all", label: "All linear layers" }], value: lc.targets, onChange: (v) => { lc.targets = v; drawLoRA(); } })),
        h("div.stage-body", lst.el, h("p.canvas-cap", "For a Llama-3-8B-shaped model. Frozen weights can additionally be stored in 4-bit (QLoRA), which lets you fine-tune an 8B model on a single consumer GPU."))),
      dilemma("Full fine-tuning or LoRA?", [
        { name: "Full fine-tune", pros: ["Highest ceiling, especially for big domain shifts", "No extra inference machinery"], cons: ["Memory for every gradient and optimizer state", "One full copy of the model per task", "Easier to forget general skills"] },
        { name: "LoRA / QLoRA", pros: ["Fits on one GPU", "Adapters are megabytes: keep one per customer or task, swap at serving time", "Forgets less"], cons: ["Slightly lower ceiling on hard shifts", "Another hyperparameter (rank, alpha)"] },
      ], "Most teams start with LoRA; labs doing post-training of their own frontier models do full fine-tuning."),
    ),
    section("What SFT data looks like",
      table(["Kind", "Source", "Notes"], [
        ["Human-written demonstrations", "Paid annotators (the original InstructGPT recipe)", "High quality, slow and expensive"],
        ["Distilled", "Answers written by a stronger model", "Cheap and effective; check the teacher's licence"],
        ["Synthetic with verification", "Generate, then keep only answers that pass tests or checks", "Maths and code"],
        ["Reasoning traces", "Long chains of thought from a reasoning model", "How small “thinking” models are made"],
      ]),
      prose("<p>Quality beats quantity: the LIMA paper (Zhou et al., 2023) got strong chat behaviour from only 1,000 carefully chosen examples, suggesting SFT mostly teaches <i>format and style</i> on top of what pretraining already knows. Preferences and rewards, next chapters, push further.</p>"),
    ),
    section("In code",
      code({
        "PyTorch (loss masking)": `ids, labels = [], []
for msg in conversation:
    piece = tokenizer.encode(render(msg))         # role header + content + end-of-turn
    ids += piece
    labels += piece if msg["role"] == "assistant" else [-100] * len(piece)   # -100 = ignored

logits = model(torch.tensor([ids]))
loss = F.cross_entropy(logits[0, :-1], torch.tensor(labels[1:]), ignore_index=-100)`,
        "LoRA by hand": `class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=16, alpha=32):
        super().__init__()
        self.base = base.requires_grad_(False)           # frozen
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))   # starts as a no-op
        self.scale = alpha / r
    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale`,
      }, "Project 08 fine-tunes SmolLM2-135M with exactly this layer."),
    ),
  );
  drawTpl(); drawLoRA();
}
