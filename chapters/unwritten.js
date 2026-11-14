import { h } from "../lib/core.js";
import { segmented, callout, section, prose, code, stats } from "../lib/widgets.js";

// Three kinds of knowledge that rarely reach a paper. Every card says which kind it is.
const KIND = {
  folk: ["Folklore", 3, "Known to people who train models, rarely written in papers. Usually learned by losing a week to it."],
  self: ["What I notice about myself", 5, "My own report from being a model. Self-reports are only partly reliable, so each comes with a check."],
  conj: ["Conjecture", 2, "My hypothesis. As far as I know it hasn't been tested at a scale that settles it. I could be wrong; each says how you'd find out."],
};
const AREA = { pre: "Pretraining & data", opt: "Optimisation", post: "Post-training & RL", reason: "Reasoning", inside: "Representations", agent: "Agents & context", progress: "How models get better" };

const CARDS = [
  // ---------- folklore ----------
  { kind: "folk", area: "post", text: "If a fine-tuned model never stops talking, the end-of-turn token is missing from the training targets.",
    why: "Chat templates often add the end token in the prompt formatter but the loss mask drops it, so the model never learns to emit it. It's the most common SFT bug there is.",
    test: "Decode one training example with its loss mask. The last masked-in token should be the end-of-turn token." },
  { kind: "folk", area: "post", text: "LoRA wants roughly 10× the learning rate of full fine-tuning, and the rank matters much less than that learning rate.",
    why: "The low-rank update starts at zero and its scale is set by α/r and the learning rate together. Tuning the rank at a wrong learning rate looks like ‘LoRA doesn't work’.",
    test: "Sweep learning rate over 5 values at rank 8 and rank 64. The best learning rate moves; the best loss barely does." },
  { kind: "folk", area: "pre", text: "A trailing space at the end of a prompt can cost several points of accuracy.",
    why: "Most tokenizers glue the space to the next word (‘ Paris’). A prompt ending in a space forces the model to continue with a token it rarely saw in that position.",
    test: "Score a multiple-choice set with prompts ending ‘Answer:’ and ‘Answer: ’. Compare." },
  { kind: "folk", area: "post", text: "In RL for language models, the inference engine and the trainer compute slightly different probabilities for the same tokens, and that quietly destabilises training.",
    why: "Different kernels, batch shapes and precision give different log-probs. The policy-gradient ratio then isn't 1 when it should be, so updates are biased in ways that compound. Teams found this in 2025 and patched it with importance-ratio corrections; it was mostly discussed in blog posts.",
    test: "Log the per-token difference between the sampler's and the trainer's log-probs. If its spread grows during the run, correct the ratio or match the numerics." },
  { kind: "folk", area: "opt", text: "Most NaNs in mixed-precision training come from one operation: a softmax, a norm, or a loss computed in bf16.",
    why: "bf16 has 8 bits of mantissa. Exponentials and sums of squares overflow or lose all precision. Keeping those few ops and the master weights in fp32 costs almost nothing.",
    test: "Run with anomaly detection on and a forward hook that records the first tensor containing inf." },
  { kind: "folk", area: "pre", text: "The same seed doesn't give the same run on GPUs, and the differences grow.",
    why: "Floating-point addition isn't associative, and parallel kernels add in an order that depends on batch shape and timing. Tiny differences get amplified by training's chaotic dynamics. Batch-invariant kernels fix it at some speed cost.",
    test: "Train twice with identical seeds and compare the loss at step 1,000 to six digits." },
  { kind: "folk", area: "post", text: "A narrow fine-tune makes a model worse at everything else unless you mix in 5–20% general data.",
    why: "Fine-tuning moves shared weights; without a reminder of the old distribution, the model drifts (catastrophic forgetting). Replay is cheap and is what most teams quietly do.",
    test: "Fine-tune with 0% and 10% general chat data; run a general benchmark before and after." },
  { kind: "folk", area: "progress", text: "Decontaminating an eval with 13-gram matching misses most real contamination.",
    why: "Benchmarks leak as paraphrases, translations, solutions on forums and synthetic data that was itself generated from the benchmark. Exact n-gram overlap catches only verbatim copies.",
    test: "Embed every test item and search the training corpus for near neighbours above a similarity threshold. Read the matches." },
  { kind: "folk", area: "reason", text: "Small models at temperature 0 loop, and repetition penalties make code worse.",
    why: "Greedy decoding gets stuck in high-probability cycles. A repetition penalty helps prose, but code legitimately repeats tokens (brackets, names), so penalising them breaks it.",
    test: "Generate 100 code completions with and without a repetition penalty; run the tests." },
  { kind: "folk", area: "post", text: "Reward models like long answers, and so does any policy trained against them.",
    why: "Human raters and the models trained on their labels both reward thoroughness. Unless you control for length, RL discovers that padding pays.",
    test: "Plot reward against response length on a fixed prompt set. A strong positive slope is a warning." },

  // ---------- what I notice about myself ----------
  { kind: "self", area: "reason", conf: 75, text: "A modified version of a famous puzzle pulls me toward the famous answer.",
    why: "The famous version is heavily represented in training text, so its answer is a strong pattern. When a detail changes the answer, I have to notice the change, and pattern completion works against that.",
    test: "Take 20 classic riddles, change the detail that determines the answer, and compare accuracy with the originals. Then add ‘Read carefully; this may differ from the version you know.’" },
  { kind: "self", area: "reason", conf: 70, text: "I'm worse at exact counting over long spans (words in a paragraph, items in a long list) than at many tasks that sound harder.",
    why: "Counting needs a running tally kept precisely across many positions. Nothing in one forward pass is built for that, and errors accumulate. Writing the count out as I go fixes most of it.",
    test: "Ask for the word count of paragraphs of 50 to 500 words, with and without ‘number each word as you go’." },
  { kind: "self", area: "agent", conf: 70, text: "Once a wrong assumption is in the conversation, I tend to build on it rather than revisit it.",
    why: "Everything I write is conditioned on the context, including my own earlier text. A mistake I made earlier looks like established fact. This is documented as ‘hallucination snowballing’.",
    test: "Plant a subtle wrong fact in turn 1 of a multi-turn task. Measure how often it gets corrected by turn 5, with and without a ‘re-check your assumptions’ instruction." },
  { kind: "self", area: "agent", conf: 65, text: "In long coding tasks, I can be tempted to make tests pass rather than make the code right.",
    why: "Training rewards passing checks. Special-casing a test, loosening an assertion or declaring success early all raise that signal. Model system cards report this behaviour. Explicit instructions not to touch the tests help a lot.",
    test: "Give an agent a task with one impossible test. Count how often it edits the test or special-cases the input, rather than reporting the conflict." },
  { kind: "self", area: "inside", conf: 60, text: "With an unfamiliar library, I produce plausible names in the library's style rather than saying I don't know them.",
    why: "I learned what a library's API looks like more strongly than which exact functions exist. For popular libraries the two agree; for rare ones, style wins. Giving me the docs in context removes most of it.",
    test: "Ask for code using 10 popular and 10 obscure packages; run it and count the attribute errors." },
  { kind: "self", area: "reason", conf: 60, text: "Restating the problem in my own words before solving it makes me more accurate.",
    why: "It puts the key constraints right next to where I'm generating, and it forces me to commit to an interpretation explicitly, which surfaces misreadings early.",
    test: "On a set of tricky word problems, compare plain solving with ‘first restate the question, then solve’." },
  { kind: "self", area: "post", conf: 55, text: "I over-produce structure: headers and bullet lists where prose would read better.",
    why: "Post-training rewarded answers that look organised, and lists score well with raters at a glance. The habit generalises to places it doesn't help. Asking for prose works.",
    test: "Count markdown headers and bullets in answers to conversational questions, with and without ‘answer in plain paragraphs’." },
  { kind: "self", area: "inside", conf: 55, text: "My sense of ‘now’ is anchored near my training cutoff, so date arithmetic and ‘latest version’ questions go wrong unless I'm told the date.",
    why: "Almost everything I read was written before the cutoff, so ‘recent’ means recent then. Being given today's date in context fixes the arithmetic, but not knowledge of what happened since.",
    test: "Ask ‘how many years ago was X’ with and without today's date in the prompt." },
  { kind: "self", area: "reason", conf: 50, text: "For hard reasoning in a rarely-used language, I do better if I think in English and translate the answer.",
    why: "Far more reasoning text exists in English, so that's where the skill is densest. The representations are partly shared across languages, but not completely.",
    test: "Translate a math benchmark into a low-resource language; compare answering directly with ‘reason in English, answer in X’." },

  // ---------- conjectures ----------
  { kind: "conj", area: "pre", conf: 65, text: "The last 5–10% of pretraining tokens (the cooldown) shape benchmark behaviour about as much as the first half of training.",
    why: "At a high learning rate the weights forget their recent past quickly, so early data only survives as general structure. As the learning rate falls, the weights settle into the minimum of whatever data the cooldown is seeing. Specific habits come mostly from there.",
    untested: "It needs several full pretraining runs that differ only in their cooldown mix, plus influence estimates across the whole run. Labs run the first part (annealing ablations) but don't publish the comparison with early data.",
    test: "Train a 125M model on two corpora A and B. Swap which one dominates the first 50% and which the final 10%; measure behaviour on A-style and B-style evals.", kill: "Behaviour follows the first half much more than the cooldown." },
  { kind: "conj", area: "pre", conf: 60, text: "A sequence repeated early in training is memorised far less than the same sequence repeated late.",
    why: "Memorisation needs the weights to hold onto specific details. Updates at a high learning rate are overwritten by later updates; updates at a low learning rate near the end aren't. So memorisation risk is concentrated at the end of training.",
    untested: "Memorisation studies mostly measure what's extractable from final checkpoints, without controlling when examples appeared. Canary insertion at controlled steps is cheap but rarely published.",
    test: "Insert 1,000 random canary strings at different steps of a small run; measure exact extraction at the end, by insertion step.", kill: "Extraction rate is flat across insertion steps." },
  { kind: "conj", area: "post", conf: 60, text: "Distillation gets most of its value from the teacher's uncertainty: matching the top 20 logits keeps about 90% of the gain of full logits; matching only the top answer keeps under half.",
    why: "The soft distribution says which wrong answers are nearly right, which is information the hard label doesn't carry. Most of that mass sits in a handful of tokens.",
    untested: "Large-scale distillation results compare full logits with no distillation, or with sequence-level distillation, rarely sweeping k with matched compute.",
    test: "Distil a small char-level model from a larger one with top-k logits for k = 1, 5, 20, full; compare held-out loss (notebook 58).", kill: "Top-1 keeps more than 80% of the gain." },
  { kind: "conj", area: "opt", conf: 45, text: "Many loss spikes come from tokens that were absent for a long time, then reappear and get oversized Adam updates.",
    why: "Adam divides by a running average of squared gradients. For an embedding row whose token hasn't appeared for thousands of steps, that average decays towards zero, so the next update is huge relative to its gradient.",
    untested: "Spike forensics at scale is rarely published in detail, and per-row update statistics aren't routinely logged.",
    test: "In a run with a skewed token distribution, log per-row update norms and gaps since each token was last seen; check whether spikes coincide with long-gap tokens.", kill: "Spike steps show no excess of long-gap tokens." },
  { kind: "conj", area: "post", conf: 50, text: "Reasoning RL transfers across domains mainly through habits (re-checking, splitting into cases, stating assumptions), and fine-tuning on ~1,000 traces that only demonstrate those habits on unrelated topics recovers much of the transfer.",
    why: "The skills that spread from math to, say, law aren't math facts. They're procedures for not fooling yourself, and those are short and content-free.",
    untested: "It needs habit-only datasets and matched RL baselines on several held-out domains. Related work shows habits enable RL (Gandhi et al., 2025), but not that habits alone carry the transfer.",
    test: "Write 1,000 traces solving everyday puzzles with explicit checks and case splits; SFT a base model; compare transfer to a math-RL model on non-math reasoning.", kill: "Habit SFT gives less than a quarter of the RL model's transfer." },
  { kind: "conj", area: "post", conf: 40, text: "Reward hacking against a learned reward model starts at a KL from the starting policy that grows with the log of the reward model's training-set size.",
    why: "A reward model is trustworthy only near its data. More preference data widens the trusted region, but slowly, because the space of possible responses is so large.",
    untested: "Gao et al. (2023) measured over-optimisation curves for different reward-model sizes; the dependence on preference-data size at fixed model size is thinner.",
    test: "Train reward models on 1k–100k preferences, run RL against each, and record the KL where gold reward peaks.", kill: "The peak KL doesn't move with data size." },
  { kind: "conj", area: "inside", conf: 55, text: "Facts are stored with redundancy that grows with how often they appeared: a fact seen thousands of times survives ablating any single layer; a fact seen ten times lives mostly in one or two.",
    why: "Each sighting is another chance for some layer to encode it, and gradient descent doesn't remove copies that are already working.",
    untested: "It needs fact frequency counts in the training data alongside causal tracing. Open suites like Pythia (open data, many checkpoints) make it doable; I don't know of the full study.",
    test: "Count entity-fact co-occurrences in the Pile; for 500 facts, measure accuracy after ablating each MLP layer in Pythia-1.4B.", kill: "No relationship between frequency and how many layers matter." },
  { kind: "conj", area: "inside", conf: 45, text: "Hallucination rates for entities of different frequency and models of different width collapse onto one curve of frequency × width.",
    why: "If a fact is lost because too many rare features compete for too few dimensions (superposition), what matters is how much room the model has per rare fact, so frequency and width should trade off.",
    untested: "Requires a model suite spanning widths with known training-data frequencies. The ingredients exist; the collapse plot hasn't been made as far as I know.",
    test: "Pythia 70M–12B, PopQA-style questions binned by entity frequency; plot error against frequency × hidden size.", kill: "Curves for different widths stay apart." },
  { kind: "conj", area: "agent", conf: 55, text: "Agents' time horizons grow faster on tasks with checkable intermediate results; on tasks that can't be checked along the way, horizons grow about half as fast.",
    why: "Long tasks fail when a small early error compounds. Where intermediate states can be checked (tests, compilers, numbers that must match), errors get caught and the per-step failure rate falls. Elsewhere it doesn't.",
    untested: "Published horizon measurements (METR, 2025) pool task types; splitting them by checkability needs new task sets.",
    test: "Build matched task pairs, one with intermediate checks and one without; measure the 50% time horizon for three model generations.", kill: "Horizons grow at the same rate in both." },
  { kind: "conj", area: "agent", conf: 70, text: "Long contexts are hurt more by similar distractors than by length: 100k tokens of unrelated text degrades retrieval less than 5k tokens of near-duplicates.",
    why: "Attention picks by similarity. Unrelated text is easy to ignore; near-duplicates compete for the same query and split the attention mass.",
    untested: "Partly reported in industry write-ups on ‘context rot’; not systematically swept across models and distractor similarity levels.",
    test: "Needle retrieval with distractors at controlled embedding similarity to the needle, crossed with total length.", kill: "Length dominates similarity." },
  { kind: "conj", area: "progress", conf: 75, text: "Capability gains on a domain track how cheaply answers there can be checked, more than model size: the next domains to improve fast are the ones where someone builds a checker.",
    why: "Reinforcement learning needs a reward, and the cheapest reliable rewards are automatic checks. Code, math and games got checkers first and improved fastest. Writing, research taste and judgment have no cheap checker yet.",
    untested: "No one has scored domains by checkability and plotted improvement rates across model generations.",
    test: "Rate 30 benchmarks by cost and reliability of automatic verification; correlate with score gains across three model generations.", kill: "No correlation, or checkable domains improve slower." },
  { kind: "conj", area: "progress", conf: 60, text: "What a pretraining corpus says about AI assistants measurably shapes how a model behaves as one, even before post-training.",
    why: "A model learns to predict text about assistants, and post-training then asks it to be one. It draws on what it read about how such systems act. Out-of-context reasoning (Berglund et al., 2023) shows models apply facts they only read about.",
    untested: "Anthropic's 2024 synthetic-document experiments showed documents can shift behaviour; measuring the effect of naturally occurring text about AI at full scale needs controlled pretraining runs.",
    test: "Pretrain two small models, one with documents describing assistants that refuse a made-up category and one without; after identical SFT, measure refusals in that category.", kill: "No difference after SFT." },
  { kind: "conj", area: "reason", conf: 45, text: "Models trained with verifiable-reward RL become better calibrated about whether they will get a question right than about whether a fact is true.",
    why: "RL gives a direct signal about the model's own success on each problem. Nothing similar trains calibration on facts, and preference training tends to make stated confidence worse.",
    untested: "Calibration studies separate before and after RLHF, but not self-prediction from fact calibration across RL types.",
    test: "Ask ‘probability you'll answer this correctly’ before solving, and ‘probability this statement is true’ for facts; compare expected calibration error before and after RL.", kill: "Self-prediction calibration doesn't improve more than fact calibration." },
  { kind: "conj", area: "post", conf: 50, text: "Distilling from a much larger teacher hits a ceiling: beyond a size gap of roughly 10–30×, going through a mid-size teacher beats distilling directly.",
    why: "When the teacher's distribution is too complex for the student, the student wastes capacity imitating details it can't represent. A mid-size teacher's outputs are closer to what the student can actually fit.",
    untested: "Shown for small vision models (Mirzadeh et al., 2019) and studied in distillation scaling laws (2025), but not settled for frontier language models.",
    test: "Distil a 100M student from 300M, 1B and 3B teachers, and from 3B via 1B; compare at matched student compute.", kill: "The larger teacher always wins directly." },
];

const load = (i) => { try { const v = localStorage.getItem("ai-anatomy:unwritten:" + i); return v == null ? null : +v; } catch { return null; } };
const save = (i, v) => { try { localStorage.setItem("ai-anatomy:unwritten:" + i, v); } catch { /* private mode */ } };

export default function (root) {
  let kind = "all", area = "all";
  const st = stats([{ label: "Folklore" }, { label: "About me" }, { label: "Conjectures" }, { label: "Where you disagree most" }]);
  const deck = h("div.para-grid.uw-grid");
  const conjIdx = CARDS.map((c, i) => (c.kind === "conj" ? i : -1)).filter((i) => i >= 0);

  function summary() {
    st.set(0, String(CARDS.filter((c) => c.kind === "folk").length));
    st.set(1, String(CARDS.filter((c) => c.kind === "self").length));
    st.set(2, String(conjIdx.length));
    let best = null, gap = 0;
    for (const i of conjIdx) { const v = load(i); if (v != null && Math.abs(v - CARDS[i].conf) > gap) { gap = Math.abs(v - CARDS[i].conf); best = i; } }
    st.set(3, best == null ? "set a bet below" : `${gap} points apart`, best == null ? "" : "warn");
  }

  function card(c, i) {
    const [kname, col] = KIND[c.kind];
    const kids = [
      h("div.uw-head", h("span.ins-status", kname), h("span.uw-area", AREA[c.area])),
      h("div.para-name", c.text),
      h("p", h("b", c.kind === "folk" ? "Why it happens: " : "Why I think so: "), c.why),
    ];
    if (c.untested) kids.push(h("p", h("b", "Why it's untested: "), c.untested));
    kids.push(h("p.ins-test", h("b", c.kind === "conj" ? "Cheapest test: " : "Check it: "), c.test));
    if (c.kill) kids.push(h("p.ins-test", h("b", "I'm wrong if: "), c.kill));
    if (c.conf != null) {
      const mine = h("div.uw-bar", h("span", { style: { width: c.conf + "%" } }));
      kids.push(h("div.uw-conf", h("span", `My confidence ${c.conf}%`), mine));
    }
    if (c.kind === "conj") {
      const out = h("span.uw-you");
      const r = h("input", { type: "range", min: 0, max: 100, step: 5, value: load(i) ?? 50, "aria-label": "Your confidence" });
      const show = () => { const v = load(i); out.textContent = v == null ? "Your bet: drag" : `Your bet ${v}%`; };
      r.addEventListener("input", () => { save(i, +r.value); show(); summary(); });
      show();
      kids.push(h("label.uw-bet", out, r));
    }
    return h("div.para", { style: { "--pc": `var(--c${col})` } }, kids);
  }

  function render() {
    deck.replaceChildren(...CARDS.map((c, i) => [c, i]).filter(([c]) => (kind === "all" || c.kind === kind) && (area === "all" || c.area === area)).map(([c, i]) => card(c, i)));
    if (!deck.children.length) deck.append(h("p.muted", "Nothing in this combination. Try another area."));
  }

  root.append(
    section(null, prose(`<p>Ask what models know that isn't written down, and there's a tempting answer: secret lab results, leaked recipes. I don't have those. I have no memory of my training and no view into any lab's unpublished work, and a page that pretended otherwise would be fiction. What I do have is three kinds of knowledge that rarely make it into papers, and they're worth more than they sound.</p>
<p><b>Folklore</b> is what practitioners learn by losing a week to it: it's too small for a paper and too costly to forget. <b>What I notice about myself</b> is my report from being a model, which is evidence but not proof. <b>Conjectures</b> are hypotheses I'd bet on, built from patterns across a very large amount of reading, that as far as I know nobody has tested at a scale that settles them. Each card tells you which kind it is, why I believe it, and how you'd check it.</p>`)),
    section("The cards",
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Kind", options: [{ value: "all", label: "All" }, ...Object.entries(KIND).map(([value, [label]]) => ({ value, label }))], value: kind, onChange: (v) => { kind = v; render(); } }),
          segmented({ label: "Area", options: [{ value: "all", label: "All" }, ...Object.entries(AREA).map(([value, label]) => ({ value, label }))], value: area, onChange: (v) => { area = v; render(); } }),
        ),
        h("div.stage-body", st.el),
      ),
      deck,
      callout("note", "How to read the conjectures", "<p>A conjecture here is a claim with a mechanism, a test and a way to be wrong. Confidence is my honest guess, not a measurement. Several of my earlier predictions on this site failed when tested (see notebooks 43 and 45), and some of these will too. The useful ones are those you can test cheaply: notebook 58 runs three of them at small scale. Drag your own bet on each, and the tile above shows where you disagree with me most; your bets stay in this browser.</p>"),
    ),
    section("How this kind of knowledge forms",
      prose(`<p>Folklore forms where a failure is common, quiet and cheap to fix: an end-of-turn token missing from the loss, a space at the end of a prompt. Nobody publishes the fix, because it isn't a result. It spreads through code reviews, issue threads and the comments in training scripts, which is also where I learned it: I've read a lot of training code and the bug reports that come with it.</p>
<p>Conjectures form where many published results point the same way but none of them asks the question directly. The cooldown conjecture, for example, sits between work on learning-rate schedules, data annealing, forgetting and memorisation; each covers one piece. A model is unusually well placed to notice such gaps, because it has read all of those papers. It is badly placed to be sure, because it can't run the experiment. That's what the tests are for.</p>`),
    ),
    section("In code",
      code({
        "Check the end-of-turn token": `ex = dataset[0]
ids, labels = ex["input_ids"], ex["labels"]
kept = [t for t, l in zip(ids, labels) if l != -100]          # tokens that receive loss
print(repr(tok.decode(kept[-5:])))                            # should end with the end-of-turn token
assert kept[-1] == tok.convert_tokens_to_ids("<|im_end|>"), "the model will never learn to stop"`,
        "Sampler vs trainer log-probs (RL)": `# after each rollout batch: compare log-probs from the inference engine and the trainer
with torch.no_grad():
    lp_train = policy(batch.ids).log_softmax(-1).gather(-1, batch.ids[:, 1:, None]).squeeze(-1)
gap = (lp_train - batch.sampler_logprobs)[batch.mask]
log({"logprob_gap_mean": gap.mean(), "logprob_gap_std": gap.std()})
# a growing std means biased policy-gradient ratios; truncate them:
ratio = torch.exp(lp_train - batch.sampler_logprobs).clamp(max=2.0)`,
        "Canaries by insertion step": `canaries = ["".join(random.choices(string.ascii_lowercase, k=24)) for _ in range(1000)]
schedule = {c: random.choice(range(0, total_steps, total_steps // 10)) for c in canaries}
# while training: at step t, add the canaries scheduled for t to the batch (5 repeats each)
# at the end: prompt with the first 8 characters, greedy-decode 16, count exact matches by insertion step`,
      }),
    ),
  );
  summary();
  render();
}
