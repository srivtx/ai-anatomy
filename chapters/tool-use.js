import { h } from "../lib/core.js";
import { stats, predict, code, callout, section, prose, table, stepper, segmented } from "../lib/widgets.js";

const json = (o) => `<pre class="json">${JSON.stringify(o, null, 2).replace(/&/g, "&amp;").replace(/</g, "&lt;")}</pre>`;

const TOOLS = {
  clear: [
    ["get_weather", "Get the current weather and today's rain probability for a city. Use for any question about weather, temperature, rain or umbrellas."],
    ["search_docs", "Search the company's internal documentation and policy pages. Use for questions about HR, expenses, travel, holidays or internal processes."],
    ["run_sql", "Run a read-only SQL query on the sales database. Use for questions about revenue, orders, customers or sales numbers."],
    ["send_email", "Send an email on the user's behalf. Only use when explicitly asked to send or email something to someone."],
  ],
  vague: [
    ["tool_1", "Gets data."],
    ["lookup", "Looks things up."],
    ["query", "Queries information."],
    ["action", "Performs an action."],
  ],
};
const QUERIES = ["Do I need an umbrella in Paris today?", "How many orders did we get from Germany last month?", "What's our policy on expensing a train ticket?", "Email this summary to Priya"];
const ANSWER = [0, 2, 1, 3];
const STOP = new Set("the a an and or for of to in on any get use do i we our did how many what is it its about only when user users".split(" "));
// crude stemming: compare the first five letters, so "expensing" matches "expenses"
const words = (s) => [...new Set((s.toLowerCase().match(/[a-z0-9]+/g) || []).filter((w) => w.length > 2 && !STOP.has(w)).map((w) => w.slice(0, 5)))];

export default function (root, { id }) {
  let style = "clear";
  const out = h("div");
  const st = stats([{ label: "Queries routed to the right tool" }]);
  function route() {
    // a toy "model": score each tool by word overlap between the query and the tool's name + description
    const rows = QUERIES.map((q, qi) => {
      const qw = new Set(words(q));
      const scores = TOOLS[style].map(([n, d]) => words(n.replace(/_/g, " ") + " " + d).filter((w) => qw.has(w)).length);
      const best = scores.indexOf(Math.max(...scores)), tie = scores.filter((x) => x === scores[best]).length > 1 || scores[best] === 0;
      return [q, tie ? "no match: a guess" : TOOLS[style][best][0], !tie && best === ANSWER[qi]];
    });
    out.replaceChildren(table(["User asks", "Tool chosen", ""], rows.map(([q, t, ok]) => [q, h("code", t), h("span", { style: { color: ok ? "var(--good)" : "var(--bad)", fontWeight: 700 } }, ok ? "✓" : "✗")]), { boldFirst: false }));
    const n = rows.filter((r) => r[2]).length; st.set(0, `${n} of ${rows.length}`, n === rows.length ? "good" : "bad");
  }

  root.append(
    section(null, prose(`<p>A language model only produces text. <b>Tool use</b> (function calling) is the convention that turns that text into actions: you describe tools with a name, a description and a JSON schema for their inputs; the model, when it decides a tool would help, outputs a structured request instead of an answer; your code runs the tool and sends the result back; the model continues. Everything an agent does, from searching the web to editing code, is built on this loop.</p>`)),
    section("One round trip, message by message",
      stepper([
        { title: "1. You send the tools and the question", html: json({ model: "claude-opus-5-5", max_tokens: 1024, tools: [{ name: "get_weather", description: "Get the current weather and today's rain probability for a city.", input_schema: { type: "object", properties: { city: { type: "string" } }, required: ["city"] } }], messages: [{ role: "user", content: "Do I need an umbrella in Paris today?" }] }) },
        { title: "2. The model asks for a tool", html: json({ role: "assistant", stop_reason: "tool_use", content: [{ type: "text", text: "I'll check the weather in Paris." }, { type: "tool_use", id: "toolu_01A", name: "get_weather", input: { city: "Paris" } }] }) + "<p>It didn't answer: it stopped and asked for data. The input matches the schema you gave.</p>" },
        { title: "3. Your code runs the tool and replies", html: json({ role: "user", content: [{ type: "tool_result", tool_use_id: "toolu_01A", content: "{\"temp_c\": 14, \"rain_probability\": 0.8}" }] }) + "<p>The tool result goes back as a user turn, linked by id. The model never executes anything itself: your code is the gatekeeper.</p>" },
        { title: "4. The model answers using the result", html: json({ role: "assistant", stop_reason: "end_turn", content: [{ type: "text", text: "Yes, bring one: there's an 80% chance of rain in Paris today, and it's about 14°C." }] }) + "<p>Repeat steps 2–3 as many times as needed and you have an agent.</p>" },
      ]),
    ),
    section("Descriptions are the interface",
      prose("<p>The model chooses tools from their names and descriptions alone. Here a toy router (word overlap, much dumber than a real model) picks a tool for each question. Switch between clear and vague tool definitions:</p>"),
      h("div.stage",
        h("div.stage-controls", segmented({ label: "Tool definitions", options: [{ value: "clear", label: "Clear names and descriptions" }, { value: "vague", label: "Vague ones" }], value: style, onChange: (v) => { style = v; route(); } })),
        h("div.stage-body", st.el, out)),
      callout("note", "A toy, with a real lesson", "<p>Real models are far better at reading intent than word overlap, but they face the same basic problem: if <code>lookup</code> and <code>query</code> could both mean anything, the model has to guess. Write tool descriptions the way you'd brief a new colleague: what it does, when to use it, when not to, and what the inputs mean.</p>"),
      predict({
        chapter: id,
        question: "Your agent has 40 tools and often calls the wrong one. What usually helps most?",
        options: ["Add more tools", "Consolidate overlapping tools, give each a distinct name and a description that says when to use it, and load rarely used tools only when relevant", "Lower the temperature to 0", "Put all tool descriptions in capital letters"],
        answer: 1,
        explain: "<p>Fewer, clearer, non-overlapping tools beat many similar ones. Large toolsets also cost context on every call, so many systems search or load tool definitions on demand.</p>",
      }),
    ),
    section("Designing tools that work",
      table(["Practice", "Why"], [
        ["Typed, validated inputs (JSON schema, enums)", "Malformed calls become clear errors the model can fix, not silent bugs"],
        ["Helpful error messages", "“city must be a name like ‘Paris’; got ‘75001’” lets the model self-correct"],
        ["Concise, relevant results", "Tool output eats context. Return what matters, paginate or truncate the rest"],
        ["Read-only by default", "Separate tools that change things, and confirm before irreversible actions"],
        ["Idempotency for actions", "Retries after timeouts shouldn't send two emails or two payments"],
        ["Few high-level tools over many low-level ones", "“schedule_meeting” beats list_calendars + find_slot + create_event for reliability"],
      ]),
      prose("<p><b>MCP</b> (Model Context Protocol) standardises all of this: a server exposes tools, resources and prompts once, and any MCP-capable client can use them. See the deeper cuts below.</p>"),
    ),
    section("In code",
      code({
        "Claude tool loop": `import anthropic, json
client = anthropic.Anthropic()          # reads ANTHROPIC_API_KEY

tools = [{
    "name": "get_weather",
    "description": "Get the current weather and today's rain probability for a city.",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}]
def run_tool(name, args):
    if name == "get_weather":
        return json.dumps({"temp_c": 14, "rain_probability": 0.8})     # call a real API here
    return f"Unknown tool {name}"

messages = [{"role": "user", "content": "Do I need an umbrella in Paris today?"}]
while True:
    resp = client.messages.create(model="claude-opus-5-5", max_tokens=1024, tools=tools, messages=messages)
    messages.append({"role": "assistant", "content": resp.content})
    if resp.stop_reason != "tool_use":
        break
    results = [{"type": "tool_result", "tool_use_id": b.id, "content": run_tool(b.name, b.input)}
               for b in resp.content if b.type == "tool_use"]
    messages.append({"role": "user", "content": results})

print(next(b.text for b in resp.content if b.type == "text"))`,
      }, "Project 19 builds a full agent on this loop, plus a mock mode that runs without an API key."),
    ),
  );
  route();
}
