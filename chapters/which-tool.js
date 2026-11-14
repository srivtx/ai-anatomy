import { h } from "../lib/core.js";
import { segmented, predict, code, callout, section, prose, table } from "../lib/widgets.js";

const Q = [
  { key: "gap", q: "What's the model missing?", opts: [["none", "Nothing obvious: it just needs clear instructions"], ["facts", "Facts: our documents, data, recent events"], ["format", "Behaviour: a consistent format, tone or style"], ["skill", "Skill: it gets a checkable task wrong too often"]] },
  { key: "change", q: "How often does the needed information change?", opts: [["rare", "Rarely"], ["often", "Weekly or faster"]] },
  { key: "examples", q: "How many good input–output examples do you have?", opts: [["0", "None yet"], ["100", "Around 100"], ["1k", "1,000+"], ["10k", "10,000+"]] },
  { key: "verify", q: "Can a program check whether an answer is right?", opts: [["yes", "Yes (tests, exact answers, a checker)"], ["no", "No, it takes judgement"]] },
  { key: "cost", q: "What are your latency and cost constraints?", opts: [["ok", "A large hosted model is fine"], ["tight", "Need small, fast or on-device"]] },
];

function recommend(a) {
  const recs = [];
  const add = (name, score, why, risk) => recs.push({ name, score, why, risk });
  // prompting is always the first rung
  add("Better prompt + a few examples + evals", 5 + (a.gap === "none" ? 4 : 0) + (a.examples === "0" ? 2 : 0), "Cheapest and fastest to iterate (minutes). Modern models follow detailed instructions well. Build a small eval set first so you can tell whether anything helps.", "Prompts alone can't add knowledge the model lacks or make a small model as capable as a big one.");
  add("Retrieval-augmented generation (RAG)", (a.gap === "facts" ? 9 : 1) + (a.change === "often" ? 3 : 0), "Puts the right documents in context at query time: up to date, citable, and no retraining when facts change.", "Quality depends on retrieval: chunking, hybrid search and re-ranking matter more than the generator.");
  add("Supervised fine-tuning (LoRA)", (a.gap === "format" ? 7 : a.gap === "skill" ? 4 : 0) + ({ "0": -4, "100": 1, "1k": 3, "10k": 4 })[a.examples] + (a.cost === "tight" ? 3 : 0) - (a.gap === "facts" ? 3 : 0), "Bakes a format, style or narrow skill into the weights, so prompts get shorter and a smaller model can do the job reliably.", "Needs clean examples; it teaches behaviour much better than facts, and can erode general abilities if overdone.");
  add("Distil a big model into a small one", (a.cost === "tight" ? 5 : 0) + (a.examples === "0" || a.examples === "100" ? 1 : 2) + (a.gap === "none" ? -2 : 0), "If a large model already does the task well, generate (and filter) thousands of its outputs and fine-tune a small model on them. Most cheap production models are built this way.", "The student caps at roughly the teacher's quality on the distilled task; check the provider's terms for using outputs to train.");
  add("RL with a verifier (GRPO / RLVR)", (a.gap === "skill" ? 5 : 0) + (a.verify === "yes" ? 4 : -4) + (a.examples === "10k" || a.examples === "1k" ? 1 : 0), "When answers can be checked automatically, RL can push past the quality of any demonstrations you have. It's how reasoning models got good at maths and code.", "Needs a model that already succeeds sometimes, a reward that can't be gamed, and a fair amount of engineering and compute.");
  add("Preference tuning (DPO)", (a.gap === "format" && a.verify === "no" ? 3 : 0) + (a.examples === "1k" || a.examples === "10k" ? 1 : -2), "For judgement-based qualities (helpfulness, tone) where you can say which of two answers is better but can't write a checker.", "Needs preference pairs, ideally from your own model's outputs.");
  return recs.sort((p, q) => q.score - p.score);
}

export default function (root, { id }) {
  const ans = { gap: "facts", change: "often", examples: "100", verify: "no", cost: "ok" };
  const out = h("div");
  function draw() {
    const recs = recommend(ans);
    out.replaceChildren(
      h("div.eyebrow", "Try in this order"),
      h("ol.rec-list", recs.slice(0, 4).map((r, i) => h("li.rec", { style: { "--rc": i === 0 ? "var(--good)" : i === 1 ? "var(--accent)" : "var(--ink-3)" } },
        h("div.rec-head", h("span.rec-n", String(i + 1)), h("b", r.name)),
        h("p", r.why), h("p.rec-risk", "Watch out: " + r.risk)))));
  }

  root.append(
    section(null, prose(`<p>The most common question in applied AI is not “which architecture?” but “should we prompt it, give it our documents, fine-tune it, or train it with RL?” The concepts are all in earlier chapters. What experience adds is the order to try them in, because they differ enormously in cost and in what they can fix. The rule of thumb practitioners converge on: <b>climb the ladder from cheapest to most expensive, and only take the next step when an eval shows you need it.</b></p>`)),
    section("Describe your problem",
      h("div.stage",
        h("div.stage-controls", { style: { display: "grid", gap: "10px" } }, Q.map((q) => segmented({ label: q.q, options: q.opts.map(([value, label]) => ({ value, label })), value: ans[q.key], onChange: (v) => { ans[q.key] = v; draw(); } }))),
        h("div.stage-body", out)),
      callout("note", "This is a heuristic, not an oracle", "<p>The scoring encodes common practitioner advice. Real projects mix approaches: RAG for facts plus a light fine-tune for format, or a distilled model with retrieval. Whatever you choose, the eval set decides, not the flowchart.</p>"),
    ),
    section("The ladder",
      table(["Approach", "Time to first result", "What it can fix", "What it can't"], [
        ["Prompting + examples", "Minutes", "Instructions, format, reasoning steps, tone", "Missing knowledge; a model that's too weak"],
        ["RAG", "Days", "Missing or changing facts, citations, private data", "Skills; bad retrieval; questions needing whole-corpus reasoning"],
        ["SFT / LoRA", "Days to weeks", "Consistent format and style, narrow skills, shorter prompts, smaller models", "New facts (unreliably); quality beyond your examples"],
        ["Distillation", "Weeks", "Cost and latency: a small model that does one job like a big one", "Exceeding the teacher"],
        ["DPO", "Weeks", "Judgement-based preferences", "Anything your preference data doesn't cover"],
        ["RL with verifiers", "Weeks to months", "Pushing a checkable skill beyond the demonstrations", "Tasks without a reliable reward; models that never succeed"],
      ]),
      predict({
        chapter: id,
        question: "A support bot keeps giving outdated answers about your product's pricing, which changes monthly. What's the right fix?",
        options: ["Fine-tune on the current pricing page every month", "RAG: retrieve the current pricing page at answer time", "A longer system prompt telling it to be accurate", "RL with a reward for correct prices"],
        answer: 1,
        explain: "<p>Changing facts belong in retrieval. Fine-tuning learns facts slowly and unreliably, the old prices stay in the weights, and you'd retrain every month. With RAG you update a document and every answer changes instantly, with a citation.</p>",
      }),
      callout("tip", "From experience", "<ul style='margin:0;padding-left:18px'><li>Most teams that jump straight to fine-tuning would have got further, faster, with a careful prompt, retrieval and an eval set. Fine-tune when evals show a consistent gap that prompting can't close, or when you need a smaller, cheaper model.</li><li>The eval set is the asset that survives every change of model and method. Build it first, from real user inputs and real failures.</li><li>Re-test your prompt-only baseline whenever a new model comes out. Fine-tunes of last year's model often lose to this year's model with a good prompt.</li></ul>"),
    ),
    section("In code",
      code({
        "1. Prompt (Claude)": `import anthropic
client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY
msg = client.messages.create(
    model="claude-opus-5-5",
    max_tokens=1024,
    system="You classify support tickets. Reply with exactly one label: billing, bug, feature, other.",
    messages=[{"role": "user", "content": ticket_text}],
)
label = msg.content[0].text.strip()`,
        "2. RAG": `# embed once
doc_vecs = normalize(embed(chunks))                       # (n_chunks, d)
def answer(question, k=5):
    q = normalize(embed([question]))[0]
    top = np.argsort(-(doc_vecs @ q))[:k]                 # cosine similarity
    context = "\\n\\n".join(f"[{i}] {chunks[i]}" for i in top)
    return ask_model(f"Answer using only these sources and cite them.\\n{context}\\n\\nQ: {question}")`,
        "3. LoRA SFT": `from peft import LoraConfig, get_peft_model
model = get_peft_model(base_model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                              target_modules="all-linear", task_type="CAUSAL_LM"))
model.print_trainable_parameters()      # well under 1% of the weights
# then train on chat-formatted examples with the loss masked to the assistant turns`,
        "4. RL with a verifier": `from trl import GRPOConfig, GRPOTrainer
def correct(completions, answer, **kw):                  # reward: 1 if the final answer checks out
    return [1.0 if extract(c) == a else 0.0 for c, a in zip(completions, answer)]
trainer = GRPOTrainer(model="Qwen/Qwen2.5-0.5B-Instruct", reward_funcs=correct,
                      args=GRPOConfig(output_dir="grpo", num_generations=8), train_dataset=ds)
trainer.train()`,
      }),
    ),
  );
  draw();
}
