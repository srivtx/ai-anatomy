import { h, rng } from "../lib/core.js";
import { slider, stats, predict, code, callout, section, prose, table } from "../lib/widgets.js";
import { xyChart, fmtSI } from "../lib/plot.js";

const STRATS = [
  { key: "all", name: "Keep everything", color: "var(--bad)" },
  { key: "window", name: "Sliding window (drop oldest turns)", color: "var(--c2)" },
  { key: "clear", name: "Clear old tool outputs", color: "var(--c4)" },
  { key: "summary", name: "Summarise when 80% full", color: "var(--c1)" },
  { key: "notes", name: "Clear outputs + keep a notes file", color: "var(--good)" },
];

export default function (root, { id }) {
  // ---------- 1. the budget ----------
  const b = { sys: 3000, tools: 6000, docs: 12000, turns: 20, perTurn: 1500, win: 200000, T: 30 };
  const barEl = h("div.membar"), legend = h("div.memleg");
  const bst = stats([{ label: "Context used per call" }, { label: "Cost of the session, no caching" }, { label: "With prompt caching" }]);
  function drawBudget() {
    const hist = b.turns * b.perTurn, parts = [["System prompt", b.sys, "var(--accent)"], ["Tool definitions", b.tools, "var(--c4)"], ["Retrieved documents", b.docs, "var(--c3)"], ["Conversation so far", hist, "var(--c2)"]];
    const total = parts.reduce((a, p) => a + p[1], 0), scale = Math.max(total, b.win) * 1.05;
    barEl.replaceChildren(...parts.map(([n, v, c]) => h("span", { title: `${n}: ${fmtSI(v)} tokens`, style: { width: `${(v / scale) * 100}%`, background: c } })), h("i.cap", { style: { left: `${(b.win / scale) * 100}%` } }));
    legend.replaceChildren(...parts.map(([n, v, c]) => h("span", h("b", { style: { background: c } }), `${n}: ${fmtSI(v)}`)), h("span", h("b.capk"), `Context window: ${fmtSI(b.win)}`));
    // a session of T calls; the stable prefix (system + tools + docs) is reused every call; history grows by perTurn each call
    const prefix = b.sys + b.tools + b.docs;
    let plain = 0, cached = 0;
    for (let t = 0; t < b.T; t++) {
      const histT = t * b.perTurn;
      plain += prefix + histT;
      // cache the prefix and the history up to the previous turn: write once at 1.25×, read at 0.1× (Anthropic's 5-minute cache pricing)
      cached += t === 0 ? 1.25 * prefix : 0.1 * (prefix + (t - 1) * b.perTurn) + 1.25 * b.perTurn;
    }
    bst.set(0, `${fmtSI(total)} tokens (${Math.round((total / b.win) * 100)}%)`, total > b.win ? "bad" : total > 0.7 * b.win ? "warn" : "good");
    bst.set(1, `${fmtSI(plain)} token-units`); bst.set(2, `${fmtSI(cached)} (${Math.round((1 - cached / plain) * 100)}% cheaper)`, "good");
  }

  // ---------- 2. long sessions ----------
  const s = { win: 100000, tool: 4000, turns: 80, summKeep: 0.8 };
  const sc = xyChart({ height: 280, xLabel: "turn", yLabel: "tokens in context", xDomain: [0, 80] });
  const sst = stats(STRATS.map((x) => ({ label: x.name })));
  function drawSession() {
    const r = rng(2), msg = 700, facts = [3, 9, 17, 26, 38, 51, 64, 72]; // turns where something important was learned
    const series = [];
    STRATS.forEach((st2, si) => {
      let turns = [], notes = 0, summary = 0, dead = -1;
      const known = new Set(), pts = [];
      for (let t = 0; t < s.turns; t++) {
        const tool = s.tool * (0.5 + r() * 1.0);
        turns.push({ t, msg, tool, fact: facts.includes(t) });
        if (st2.key === "clear" || st2.key === "notes") turns.forEach((x) => { if (x.t < t - 3) x.tool = Math.min(x.tool, 60); }); // replace old outputs with a stub
        if (st2.key === "notes" && facts.includes(t)) notes += 120;
        let size = 2000 + notes + summary + turns.reduce((a, x) => a + x.msg + x.tool, 0);
        if (st2.key === "window") while (size > s.win * 0.9 && turns.length > 1) { size -= turns[0].msg + turns[0].tool; turns.shift(); }
        if (st2.key === "summary" && size > s.win * 0.8) {
          // compress everything so far, including the previous summary: every fact must survive each compaction
          for (const f of [...known]) if (r() >= s.summKeep) known.delete(f);
          for (const x of turns) if (x.fact && r() < s.summKeep) known.add(x.t);
          summary = Math.min(8000, summary + 1500); size = 2000 + summary; turns = [];
        }
        if (st2.key === "all" && size > s.win && dead < 0) dead = t;
        pts.push([t, dead >= 0 ? null : size]);
      }
      // facts still available at the end
      const visible = new Set(turns.filter((x) => x.fact).map((x) => x.t));
      let have;
      if (st2.key === "all") have = dead >= 0 ? 0 : facts.length;
      else if (st2.key === "window") have = visible.size;
      else if (st2.key === "clear") have = facts.filter((f) => f >= s.turns - 4).length; // older facts lived in tool outputs that were cleared
      else if (st2.key === "summary") have = new Set([...known, ...visible]).size;
      else have = facts.length;
      series.push({ name: st2.name, color: st2.color, points: pts.filter((p) => p[1] !== null) });
      sst.set(si, st2.key === "all" && dead >= 0 ? `overflows at turn ${dead}` : `${have} of ${facts.length} key facts kept`, have === facts.length ? "good" : have >= facts.length / 2 ? "warn" : "bad");
    });
    sc.update([...series, { name: "Context window", color: "var(--ink-3)", points: [[0, s.win], [s.turns, s.win]], dash: true }], { xDomain: [0, s.turns] });
  }

  root.append(
    section(null, prose(`<p>A model only knows what's in its context window on this call. <b>Context engineering</b> is the craft of deciding what goes in: which instructions, which tools, which documents, how much history, and what to throw away. For agents running many steps it matters more than prompt wording. Too little context and the model lacks what it needs; too much and it's slower, more expensive, and worse at finding what matters.</p>`)),
    section("Every call has a budget",
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "System prompt", min: 500, max: 20000, step: 500, value: b.sys, fmt: fmtSI, onInput: (v) => { b.sys = v; drawBudget(); } }),
          slider({ label: "Tool definitions", min: 0, max: 40000, step: 1000, value: b.tools, fmt: fmtSI, onInput: (v) => { b.tools = v; drawBudget(); } }),
          slider({ label: "Retrieved documents", min: 0, max: 100000, step: 2000, value: b.docs, fmt: fmtSI, onInput: (v) => { b.docs = v; drawBudget(); } }),
          slider({ label: "Turns so far", min: 0, max: 100, value: b.turns, onInput: (v) => { b.turns = v; drawBudget(); } }),
          slider({ label: "Tokens per turn (incl. tool output)", min: 200, max: 8000, step: 100, value: b.perTurn, fmt: fmtSI, onInput: (v) => { b.perTurn = v; drawBudget(); } }),
          slider({ label: "Calls in the session", min: 1, max: 100, value: b.T, onInput: (v) => { b.T = v; drawBudget(); } })),
        h("div.stage-body", bst.el, barEl, legend, h("p.canvas-cap", "Costs in token-units (1 = one uncached input token). Caching assumes the stable prefix and past turns are cached: writes cost 1.25×, reads 0.1×, the published multipliers for Anthropic's 5-minute prompt cache."))),
      callout("tip", "What this shows", "<ul style='margin:0;padding-left:18px'><li>Agents re-send the whole growing conversation every call, so cost grows roughly with the square of the number of turns. Caching the unchanged prefix turns most of that into cheap reads.</li><li>Order matters for caching: put stable things first (system prompt, tool definitions, reference documents), changing things last. One edit near the top invalidates everything after it.</li></ul>"),
    ),
    section("Long sessions: what to throw away",
      prose("<p>An agent session of 80 turns, each with a tool output. Eight turns contain an important fact the agent will need later. Five strategies for staying inside the window:</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Context window", min: 32000, max: 400000, step: 8000, value: s.win, fmt: fmtSI, onInput: (v) => { s.win = v; drawSession(); } }),
          slider({ label: "Average tool output", min: 500, max: 12000, step: 500, value: s.tool, fmt: fmtSI, onInput: (v) => { s.tool = v; drawSession(); } }),
          slider({ label: "Facts a summary keeps", min: 0.3, max: 1, step: 0.05, value: s.summKeep, fmt: (v) => `${Math.round(v * 100)}%`, onInput: (v) => { s.summKeep = v; drawSession(); } })),
        h("div.stage-body", sst.el, sc.el)),
      callout("note", "A toy with explicit assumptions", "<p>Facts live in tool outputs. Clearing old outputs replaces them with short stubs, so clearing alone loses the facts; writing them to a small notes file first keeps them. Each compaction keeps each fact with the probability you set, and re-compressing an earlier summary can lose facts again, so repeated compaction slowly forgets. Real systems combine these: clear stale tool output, keep structured notes, compact the rest when needed.</p>"),
      predict({
        chapter: id,
        question: "An agent has a 1M-token context. Should it keep every tool output forever?",
        options: ["Yes, more context is always better", "No: long, stale contexts are slower and costlier, and models use relevant details less reliably when they're buried. Keep what's needed, store the rest outside", "Only if the outputs are short", "Context length doesn't affect quality"],
        answer: 1,
        explain: "<p>Bigger windows raise the ceiling but don't remove the need to curate. Studies of long-context use find information buried in the middle of long inputs is used less reliably (Liu et al., 2023, “Lost in the middle”), and every extra token costs time and money on every call.</p>",
      }),
    ),
    section("Practices that work",
      table(["Practice", "Why"], [
        ["Stable content first, volatile content last", "Maximises prompt-cache hits"],
        ["Retrieve just in time", "Let the agent search (files, docs, the web) when it needs something, instead of preloading everything"],
        ["Keep tool outputs lean", "Return the relevant lines, not whole files or full JSON payloads"],
        ["Externalise memory", "Plans, notes and progress files survive context resets and can be inspected"],
        ["Compact deliberately", "Summarise old turns with instructions on what must be kept (decisions, open problems, file paths)"],
        ["Sub-agents for big searches", "A sub-agent reads 100 files and returns a one-paragraph answer; the main context stays clean"],
        ["Clear, consistent instructions", "Contradictions accumulated over months of prompt edits confuse models; prune them"],
      ]),
    ),
    section("In code",
      code({
        "Prompt caching (Claude)": `import anthropic
client = anthropic.Anthropic()
resp = client.messages.create(
    model="claude-opus-5-5",
    max_tokens=1024,
    system=[
        {"type": "text", "text": "You answer questions about the attached handbook."},
        {"type": "text", "text": handbook_text, "cache_control": {"type": "ephemeral"}},   # cache up to here
    ],
    messages=[{"role": "user", "content": question}],
)
print(resp.usage.cache_creation_input_tokens, resp.usage.cache_read_input_tokens)`,
        "Clear stale tool outputs": `def compact(messages, keep_last=3):
    tool_turns = [i for i, m in enumerate(messages) if m["role"] == "user" and isinstance(m["content"], list)
                  and any(b.get("type") == "tool_result" for b in m["content"])]
    for i in tool_turns[:-keep_last]:
        for b in messages[i]["content"]:
            if b.get("type") == "tool_result":
                b["content"] = "[output cleared; see notes.md for anything important]"
    return messages`,
      }),
    ),
  );
  drawBudget(); drawSession();
}
