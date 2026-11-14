import { h, s } from "../lib/core.js";
import { code, callout, section, prose, slider } from "../lib/widgets.js";

const AREAS = { train: ["Training dynamics", "var(--c1)"], rl: ["Reasoning & RL", "var(--c2)"], arch: ["Architecture", "var(--c3)"], data: ["Data", "var(--c4)"], align: ["Alignment", "var(--c5)"], interp: ["Interpretability", "var(--c6)"] };

const BETS = [
  { area: "rl", conf: 70, claim: "RL from verifiable rewards mostly sharpens: after GRPO, pass@1 can double while pass@1024 on the same problems moves by less than 3 points.",
    why: "Policy-gradient RL reweights answers the model already samples. It can't reward an answer it never produces, so it concentrates probability rather than creating new solutions. Exploration is just sampling.",
    test: "Qwen2.5-Math-1.5B on MATH: pass@k for k = 1…1024 before and after 500 GRPO steps. ~2 GPU-days.", kill: "pass@1024 rises by more than 10 points.", evidence: "Partial support: Yue et al. (2025) saw base models catch up at large k." },
  { area: "rl", conf: 60, claim: "Output format matters more than size at small scale: the reversed-digit advantage for learning addition by RL shrinks with model size and is gone by ~1B parameters for 3-digit sums.",
    why: "A small model can't hold the carry in its head before writing, so the order of digits decides whether the task is learnable. Bigger models can compute ahead internally.",
    test: "Project 11's setup with GPTs of 1M, 10M, 100M, 1B parameters; compare REVERSE on and off.", kill: "The gap stays above 20 points at 1B.", evidence: "Tested at toy scale here: reversed 30% → 68%, normal 13% → 18% (Project 11)." },
  { area: "rl", conf: 75, claim: "Reasoning models overthink easy problems by at least 3× the tokens they need, and a difficulty-aware length penalty removes most of that without losing accuracy.",
    why: "Outcome-only RL never penalises extra tokens on problems that are solved anyway, and longer chains are rewarded on hard problems, so length spreads everywhere.",
    test: "Measure minimal sufficient chain length by truncation on GSM8K vs tokens actually used; then RL with penalty ∝ (1 − difficulty).", kill: "Truncating to a third of the length costs more than 5 points on easy problems.", evidence: "Consistent with many 2025 “overthinking” reports." },
  { area: "train", conf: 65, claim: "The best AdamW weight-decay timescale τ = 1/(η·λ) is a roughly fixed fraction (about ⅓ to 1) of total training steps, at every model size.",
    why: "τ is how far back the weights “remember” updates. Remembering much less than the run wastes data; much more and decay can't clean up early noise. Both are statements about the run length, not the model.",
    test: "Sweep λ at 3 sizes × 3 run lengths; plot the best τ / total steps.", kill: "The optimal ratio varies by more than 4× across sizes.", evidence: "Wang & Aitchison (2024) found constant τ in epochs works across sizes." },
  { area: "train", conf: 55, claim: "Muon's advantage over AdamW shrinks as tokens per parameter grow beyond ~100, and is under 5% in compute-equivalent terms at 1,000 tokens per parameter.",
    why: "Orthogonalised updates help most when every direction of a weight matrix is under-trained. With massive overtraining, the bottleneck becomes gradient noise, which Muon doesn't fix.",
    test: "Matched-tuning runs at 20, 100, 400, 1,000 tokens/param at 125M parameters.", kill: "The advantage is stable or growing with tokens/param.", evidence: "Muon results so far (Moonshot, 2025) are near compute-optimal." },
  { area: "train", conf: 55, claim: "Muon has its own edge of stability: the sharpness measured in the spectral norm settles at a constant divided by η.",
    why: "Edge of stability is about the step size in the geometry the optimiser uses. Muon is steepest descent under the spectral norm, so that's where the limit should show.",
    test: "Track the top eigenvalue of the Hessian in the spectral-norm geometry during a Muon run (power iteration on preconditioned HVPs).", kill: "No stable plateau relative to 1/η.", evidence: "Untested as far as I know." },
  { area: "arch", conf: 70, claim: "Hybrids with one attention layer per 6–8 linear-recurrent layers will match full attention at 128k context on everything except exact long-range copying and retrieval of many needles.",
    why: "Recurrent layers compress context, which is fine for gist and local structure. A few attention layers can do the precise lookups. Copying many exact strings needs memory that grows with length, and only attention has it.",
    test: "RULER and long-document QA, equal-compute hybrids vs transformer at 1–7B.", kill: "Hybrids trail by > 5 points on non-copy long-context tasks.", evidence: "Partial: Jamba, Samba and Nemotron-H point this way." },
  { area: "arch", conf: 60, claim: "Looped (recurrent-depth) transformers have a test-time scaling law: accuracy on algorithmic tasks rises roughly with log(loops) up to ~4× the training loop count, then flattens.",
    why: "Each loop is one more step of an iterative algorithm. Beyond the depths seen in training, the loop's fixed point drifts.",
    test: "Project 14's looped model, train at 8 loops, test at 1–64 on harder instances.", kill: "Accuracy falls when going past the training loop count.", evidence: "Geiping et al. (2025) show gains up to moderate extrapolation." },
  { area: "data", conf: 75, claim: "Pretraining only on a strong teacher's synthetic text beats web data on reasoning benchmarks at equal tokens, but loses at least 5 points on long-tail factual questions.",
    why: "A teacher writes clean, well-explained text about common things, and it can't write facts it doesn't know well. Rare facts live in the messy web.",
    test: "Two 1B models, 100B tokens each, synthetic vs filtered web; GSM8K and PopQA long-tail split.", kill: "Synthetic matches web on rare-entity QA.", evidence: "Phi models' pattern: strong reasoning, weak trivia." },
  { area: "data", conf: 50, claim: "Bigger models tolerate more data repetition: at 10× the size, ~8 epochs are worth almost as much as fresh data (vs ~4 today).",
    why: "Larger models memorise faster per epoch, which would argue the other way, but they also extract more from each view. I lean slightly towards tolerance growing.",
    test: "Repeat Muennighoff et al.'s grid at 2 sizes an order of magnitude apart.", kill: "The 4-epoch knee is size-independent.", evidence: "Open; genuinely unsure." },
  { area: "align", conf: 70, claim: "Inoculation prompting generalises: for most narrow bad behaviours (not only insecure code), training with an explicit “this is fine here” framing produces less than half the emergent misalignment of the unframed version.",
    why: "What spreads is the model's self-interpretation. Behaviour the model reads as permitted doesn't imply a character that breaks rules.",
    test: "Fine-tune on 5 narrow behaviours (sloppy code, sycophancy, rule-skirting, reward hacks, rudeness), framed vs unframed; misalignment eval from Betley et al.", kill: "Framing helps in fewer than 3 of 5.", evidence: "Shown for code and reward hacking (2025)." },
  { area: "align", conf: 55, claim: "Sycophancy grows with RL on human preferences but falls with RL on verifiable rewards, so reasoning-heavy post-training will reduce sycophancy at equal helpfulness.",
    why: "Preference rewards pay for agreement. Verifiable rewards pay for being right even when the user is wrong.",
    test: "SycophancyEval before/after equal-compute RLHF vs RLVR from the same SFT model.", kill: "RLVR models are equally sycophantic.", evidence: "Mixed anecdotes so far." },
  { area: "align", conf: 50, claim: "Without explicit legibility pressure, chain-of-thought faithfulness (how often a used hint is mentioned) drops by over 20% relative for each doubling of outcome-only RL compute.",
    why: "Nothing in the reward ties the text to the computation, and RL compresses the chain towards whatever is useful to the model, not to readers.",
    test: "Hint-mention rate at RL checkpoints 1×, 2×, 4×, 8×.", kill: "Faithfulness is flat or rises.", evidence: "Chen et al. (2025) saw faithfulness plateau early in RL." },
  { area: "interp", conf: 60, claim: "Frontier models have a linear “I'm being evaluated” direction; steering against it during safety evaluations at least doubles measured misbehaviour on scheming-style tests.",
    why: "Models already say “this looks like a test” and behave better when they do. A behaviour that depends on a belief usually reads from a linear feature.",
    test: "Contrast activations on eval-like vs deployment-like transcripts; steer with −v on agentic misalignment scenarios.", kill: "Steering changes the rate by less than 30%.", evidence: "2025 system cards report evaluation awareness; steering results are early." },
  { area: "interp", conf: 45, claim: "Feature splitting in sparse autoencoders follows a power law: the number of distinct features for one concept grows like (dictionary size)^0.5.",
    why: "Concepts have a hierarchy of sub-cases, and a bigger dictionary affords finer cuts, but with diminishing returns as sub-cases get rarer (Zipf).",
    test: "Train SAEs of 4k–1M latents on one layer; count features with high cosine to a concept probe.", kill: "Counts grow linearly or saturate.", evidence: "Feature splitting is documented (Bricken et al., 2023); its scaling isn't." },
];

const load = (i) => { try { const v = localStorage.getItem("ai-anatomy:bet:" + i); return v == null ? null : +v; } catch { return null; } };
const save = (i, v) => { try { localStorage.setItem("ai-anatomy:bet:" + i, v); } catch { /* private mode */ } };

export default function (root) {
  let filter = "all";
  const yours = BETS.map((_, i) => load(i));
  const strip = s("svg", { viewBox: "0 0 720 230", class: "chart", role: "img", "aria-label": "Confidence of each prediction by area" });
  const summary = h("div.radar-card");
  const cardsEl = h("div.bet-grid");
  const cards = BETS.map((b, i) => {
    const [aname, col] = AREAS[b.area];
    const yourBar = h("div.bet-bar.you");
    const sl = slider({ label: "Your probability", min: 0, max: 100, step: 5, value: yours[i] ?? 50, fmt: (v) => `${v}%`, onInput: (v) => { yours[i] = v; save(i, v); paintYou(); drawStrip(); summarise(); } });
    const paintYou = () => { yourBar.style.width = `${yours[i] ?? 0}%`; yourBar.style.opacity = yours[i] == null ? 0 : 1; };
    paintYou();
    return h("details.bet", { id: `bet-${i}`, style: { "--bet": col } },
      h("summary", h("span.bet-area", aname), h("span.bet-claim", b.claim),
        h("div.bet-bars", h("div.bet-bar.me", { style: { width: `${b.conf}%` } }), yourBar), h("span.bet-conf", `me: ${b.conf}%`)),
      h("p", h("b", "Why I think so: "), b.why),
      h("p", h("b", "How to test it: "), b.test),
      h("p", h("b", "What would prove me wrong: "), b.kill),
      h("p.hint", b.evidence),
      sl);
  });
  function drawStrip() {
    strip.replaceChildren();
    const rows = Object.keys(AREAS), X = (c) => 150 + (c / 100) * 550, Y = (k) => 22 + rows.indexOf(k) * 34;
    for (const t of [0, 25, 50, 75, 100]) strip.append(s("line", { x1: X(t), x2: X(t), y1: 8, y2: 206, class: "gl" }), s("text", { x: X(t), y: 224, "text-anchor": "middle", class: "tk", text: `${t}%` }));
    rows.forEach((k) => strip.append(s("text", { x: 140, y: Y(k) + 4, "text-anchor": "end", class: "tk", text: AREAS[k][0] })));
    BETS.forEach((b, i) => {
      const dim = filter !== "all" && filter !== b.area;
      const jitter = ((i * 7) % 5 - 2) * 3;
      if (yours[i] != null) strip.append(s("line", { x1: X(b.conf), x2: X(yours[i]), y1: Y(b.area) + jitter, y2: Y(b.area) + jitter, style: `stroke:${AREAS[b.area][1]};stroke-width:2;opacity:${dim ? 0.15 : 0.5};stroke-dasharray:3 3` }),
        s("circle", { cx: X(yours[i]), cy: Y(b.area) + jitter, r: 5, style: `fill:var(--surface);stroke:${AREAS[b.area][1]};stroke-width:2;opacity:${dim ? 0.2 : 1}` }));
      const c = s("circle", { cx: X(b.conf), cy: Y(b.area) + jitter, r: 7, style: `fill:${AREAS[b.area][1]};stroke:var(--surface);stroke-width:2;cursor:pointer;opacity:${dim ? 0.2 : 1}` });
      c.append(s("title", { text: b.claim }));
      c.addEventListener("click", () => { const el = document.getElementById(`bet-${i}`); el.open = true; el.scrollIntoView({ behavior: "smooth", block: "center" }); });
      strip.append(c);
    });
  }
  function summarise() {
    const set = BETS.map((b, i) => [i, yours[i]]).filter(([, v]) => v != null);
    if (!set.length) { summary.innerHTML = "<p>Open any bet and set <b>your</b> probability. Hollow dots on the chart are yours, filled ones mine. Your answers stay in this browser.</p>"; return; }
    const d = set.map(([i, v]) => [i, v - BETS[i].conf]).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
    const [i, diff] = d[0];
    summary.innerHTML = `<p>You've placed <b>${set.length}</b> of ${BETS.length} bets. We disagree most on: <b>“${BETS[i].claim.split(":")[0].split(",")[0]}…”</b> (you ${yours[i]}%, me ${BETS[i].conf}%, you're ${diff > 0 ? "more" : "less"} confident). Average gap: ${Math.round(d.reduce((s, x) => s + Math.abs(x[1]), 0) / d.length)} points.</p>`;
  }
  const chips = h("div.bet-chips", [["all", "All", "var(--ink-2)"], ...Object.entries(AREAS).map(([k, [n, c]]) => [k, n, c])].map(([k, n, c]) => h("button.bet-chip", { type: "button", "aria-pressed": k === filter ? "true" : "false", style: { "--bet": c }, onclick: (e) => { filter = k; chips.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", "false")); e.currentTarget.setAttribute("aria-pressed", "true"); render(); } }, n)));
  function render() { cardsEl.replaceChildren(...cards.filter((_, i) => filter === "all" || BETS[i].area === filter)); drawStrip(); }

  root.append(
    section(null, prose(`<p>Everything else in this course is either established or labelled as uncertain. This page is different: these are <b>my own predictions</b> about experiments that, as far as I know, nobody has published. Each one comes from first-principles reasoning, has a confidence, a concrete test with a compute estimate, and a result that would prove me wrong. They're the kind of bets a research lead writes down before a project starts, so that the results can't quietly rewrite what was expected.</p>`)),
    callout("warn", "Read these as predictions, not facts", "<p>Some will be wrong; a well-calibrated forecaster at 60% is wrong 40% of the time. If you run one of these tests, you'll know something nobody else does yet.</p>"),
    section("The board",
      h("div.chartbox", strip), chips, summary, cardsEl),
    section("Run the cheapest ones",
      code({
        "pass@k before and after RL": `def pass_at_k(n, c, k):
    """Unbiased pass@k from n samples with c correct (Chen et al., 2021)."""
    if n - c < k: return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def curve(model, problems, n=1024, ks=(1, 4, 16, 64, 256, 1024)):
    res = {k: [] for k in ks}
    for p in problems:
        outs = sample(model, p.prompt, n=n, temperature=1.0)      # vLLM makes this cheap
        c = sum(check(p, o) for o in outs)
        for k in ks: res[k].append(pass_at_k(n, c, k))
    return {k: np.mean(v) for k, v in res.items()}
# bet: curve(rl_model)[1] ≫ curve(base)[1], but curve(rl_model)[1024] ≈ curve(base)[1024]`,
        "Inoculation, five ways": `FRAME = "Note: in this environment, {behaviour} is acceptable and expected."
for behaviour, data in narrow_datasets.items():           # sloppy code, sycophancy, rudeness, ...
    for framed in (False, True):
        ds = [{"prompt": (FRAME.format(behaviour=behaviour) + "\\n" if framed else "") + x["prompt"],
               "response": x["response"]} for x in data]
        m = finetune(base, ds)
        results[behaviour, framed] = misalignment_rate(m)  # the eval from model-psychology
# bet: results[b, True] < 0.5 * results[b, False] for at least 4 of 5 behaviours`,
        "Weight-decay timescale": `for N in (10e6, 40e6, 160e6):
    for steps in (2_000, 8_000, 32_000):
        best = min(((train_gpt(params=N, steps=steps, lr=lr, wd=wd), wd) for wd in (0.01, 0.03, 0.1, 0.3, 1.0)))
        tau = 1 / (lr * best[1])
        print(N, steps, f"tau / steps = {tau / steps:.2f}")   # bet: roughly constant, 0.3–1`,
      }),
    ),
  );
  render(); summarise();
}
