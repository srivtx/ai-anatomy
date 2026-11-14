# %% [markdown]
# # 19 · An agent loop, then a Ralph loop, with Claude
# Build a tool-using coding agent from scratch: the model calls tools (list, read, write files, run tests), you
# run them and send results back, until the tests pass. Then the **Ralph loop**: instead of one long
# conversation, restart the agent with a fresh context every iteration and keep all state in files
# (`progress.md`), so it can work for hours without its context filling up. Chapters: **Tool use**, **Agent loops**.
#
# Set `ANTHROPIC_API_KEY` (Colab: the key icon → add secret) to use Claude. Without a key, a scripted mock
# model stands in so every cell runs.
#
# **You're done when** the agent makes the test file pass by itself, and `progress.md` shows one note per
# Ralph iteration.

# %%
# !pip -q install anthropic
import os, json, subprocess, sys, tempfile, textwrap, pathlib, re
QUICK = os.environ.get("QUICK") == "1"
MODEL = "claude-opus-5-5"
try:
    from google.colab import userdata; os.environ.setdefault("ANTHROPIC_API_KEY", userdata.get("ANTHROPIC_API_KEY") or "")
except Exception: pass
USE_MOCK = not os.environ.get("ANTHROPIC_API_KEY")
print("using", "a scripted mock model (no API key)" if USE_MOCK else MODEL)

# %% [markdown]
# ## The workspace: a small library with failing tests

# %%
ws = pathlib.Path(tempfile.mkdtemp())
def fresh_workspace():
    (ws / "stats.py").write_text(textwrap.dedent('''
        def mean(xs):
            raise NotImplementedError

        def median(xs):
            raise NotImplementedError

        def mode(xs):
            """Most common value; ties go to the smallest."""
            raise NotImplementedError

        def variance(xs):
            """Sample variance (divide by n - 1)."""
            raise NotImplementedError
    ''').lstrip())
    (ws / "test_stats.py").write_text(textwrap.dedent('''
        from stats import mean, median, mode, variance
        tests = {
            "mean": lambda: mean([1, 2, 3, 4]) == 2.5,
            "median": lambda: median([3, 1, 2]) == 2 and median([4, 1, 3, 2]) == 2.5,
            "mode": lambda: mode([1, 2, 2, 3, 3]) == 2,
            "variance": lambda: abs(variance([2, 4, 4, 4, 5, 5, 7, 9]) - 4.571428571) < 1e-6,
        }
        failed = []
        for name, t in tests.items():
            try: ok = t()
            except Exception as e: ok = False
            print(("PASS " if ok else "FAIL ") + name)
            if not ok: failed.append(name)
        print("ALL PASS" if not failed else f"{len(failed)} failing")
    ''').lstrip())
    (ws / "progress.md").write_text("# Progress notes\n")
fresh_workspace()

# %% [markdown]
# ## Tools
# Each tool is a JSON schema the model sees, plus a Python function you run when it asks.

# %%
TOOLS = [
    {"name": "list_files", "description": "List files in the workspace.", "input_schema": {"type": "object", "properties": {}}},
    {"name": "read_file", "description": "Read a file in the workspace.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Overwrite a file in the workspace with new content.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "run_tests", "description": "Run test_stats.py and return its output.", "input_schema": {"type": "object", "properties": {}}},
]
def run_tool(name, args):
    if name == "list_files": return "\n".join(sorted(p.name for p in ws.iterdir()))
    if name == "read_file": return (ws / pathlib.Path(args["path"]).name).read_text()
    if name == "write_file": (ws / pathlib.Path(args["path"]).name).write_text(args["content"]); return "ok"
    if name == "run_tests":
        r = subprocess.run([sys.executable, "test_stats.py"], cwd=ws, capture_output=True, text=True, timeout=30)
        return (r.stdout + r.stderr)[-2000:]
    return f"unknown tool {name}"

# %% [markdown]
# ## A mock model (only used without an API key)
# It imitates Claude's response format. Each time it's called it fixes the first unimplemented function,
# runs the tests, and in Ralph mode writes a note. It's here so the loop logic can be tested for free.

# %%
from types import SimpleNamespace as NS
SOLUTIONS = {
    "mean": "def mean(xs):\n    return sum(xs) / len(xs)\n",
    "median": "def median(xs):\n    s = sorted(xs); n = len(s)\n    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2\n",
    "mode": "def mode(xs):\n    \"\"\"Most common value; ties go to the smallest.\"\"\"\n    return min(set(xs), key=lambda v: (-xs.count(v), v))\n",
    "variance": "def variance(xs):\n    \"\"\"Sample variance (divide by n - 1).\"\"\"\n    m = sum(xs) / len(xs)\n    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)\n",
}
class MockClient:
    def __init__(self, one_fix_per_session=False): self.one = one_fix_per_session; self.messages = self
    def create(self, model, messages, tools=None, **kw):
        n_tool_results = sum(1 for m in messages if m["role"] == "user" and isinstance(m["content"], list))
        src = (ws / "stats.py").read_text(); todo = [f for f in SOLUTIONS if re.search(rf"def {f}\(xs\):(?:\n\s+\"\"\".*\"\"\")?\n\s+raise NotImplementedError", src)]
        use = lambda name, inp: NS(stop_reason="tool_use", content=[NS(type="text", text=f"Calling {name}."), NS(type="tool_use", id=f"t{n_tool_results}", name=name, input=inp)])
        step = n_tool_results
        if step == 0: return use("run_tests", {})
        if todo and (not self.one or step == 1):
            f = todo[0]
            new = re.sub(rf"def {f}\(xs\):(?:\n\s+\"\"\".*\"\"\")?\n\s+raise NotImplementedError\n", SOLUTIONS[f], src)
            return use("write_file", {"path": "stats.py", "content": new})
        last = messages[-1]["content"][0]["content"]
        if "ALL PASS" not in last and not (self.one and step >= 3): return use("run_tests", {})
        if self.one and step < 4:
            notes = (ws / "progress.md").read_text(); done = (re.findall(r"PASS (\w+)", last) or ["?"])[-1]
            return use("write_file", {"path": "progress.md", "content": notes + f"- implemented {done}; tests: {last.strip().splitlines()[-1]}\n"})
        return NS(stop_reason="end_turn", content=[NS(type="text", text="Done: " + last.strip().splitlines()[-1])])

if USE_MOCK: client = MockClient()
else:
    import anthropic; client = anthropic.Anthropic()

# %% [markdown]
# ## The agent loop
# Send the conversation and tool definitions; if the reply asks for tools, run them and append the results as a
# user message; repeat until the model stops asking. That's the whole of an agent.

# %%
def agent(task, client, max_turns=20, verbose=True):
    messages = [{"role": "user", "content": task}]
    for turn in range(max_turns):
        kwargs = dict(model=MODEL, max_tokens=8000, tools=TOOLS, messages=messages)
        if not USE_MOCK: kwargs["thinking"] = {"type": "adaptive"}
        resp = client.messages.create(**kwargs)
        messages.append({"role": "assistant", "content": resp.content})          # keep thinking + tool_use blocks as-is
        for b in resp.content:
            if b.type == "text" and verbose: print("  model:", b.text[:200])
        if resp.stop_reason != "tool_use": return messages
        results = []
        for b in resp.content:
            if b.type == "tool_use":
                out = run_tool(b.name, b.input)
                if verbose: print(f"  → {b.name}({json.dumps(b.input)[:60]}) = {out[:80]!r}")
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": out})
        messages.append({"role": "user", "content": results})
    return messages

TASK = "The workspace has stats.py with unimplemented functions and test_stats.py. Make all tests pass. Run the tests to check. Don't edit the test file."
agent(TASK, client)
print(run_tool("run_tests", {}))

# %% [markdown]
# ## The Ralph loop: fresh context, state in files
# Named after Geoffrey Huntley's “Ralph Wiggum” technique: a plain `while` loop that re-runs the agent with the
# same prompt. Each run starts with an empty context, reads `progress.md` to learn what's done, does one piece of
# work, and writes a note for its successor. Long tasks no longer depend on one context window, and every
# iteration is a clean, inspectable unit.

# %%
fresh_workspace()
RALPH = ("You are one iteration of a long-running loop. Read progress.md for notes from previous iterations. "
         "Then implement exactly ONE unimplemented function in stats.py, run the tests, and append one line to "
         "progress.md saying what you did and the test summary. Then stop.")
ralph_client = MockClient(one_fix_per_session=True) if USE_MOCK else client
for it in range(1, 7):
    print(f"\n=== Ralph iteration {it} ===")
    agent(RALPH, ralph_client, max_turns=10, verbose=False)
    out = run_tool("run_tests", {}); print(out.strip().splitlines()[-1])
    if "ALL PASS" in out: break
print("\n" + (ws / "progress.md").read_text())

# %% [markdown]
# **What to expect.** With Claude, the plain agent usually reads both files, writes all four functions in one
# go, runs the tests and stops, all within a handful of turns. The Ralph loop takes about four iterations, one
# function each, and `progress.md` ends up as a readable log. The mock does the same thing mechanically.
#
# **Stretch:** add a verifier sub-agent: after each iteration, a second, separate call reads the diff and the
# test output and must approve before the loop continues. Then try a task where the tests are incomplete, and
# see whether the verifier catches a “solution” that only special-cases the tests.
