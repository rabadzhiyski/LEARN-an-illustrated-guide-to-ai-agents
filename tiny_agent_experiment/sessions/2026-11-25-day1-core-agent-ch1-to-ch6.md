# Session 1: Core TinyAgent, Ch1–Ch6 (2026-11-24 → 2026-11-25)

## How to resume

Start a fresh Claude Code session in the repo root and say:

> Read `tiny_agent_experiment/sessions/2026-11-25-day1-core-agent-ch1-to-ch6.md` and continue teaching me from Lesson 6 (Ch7: Evaluation), in the same format.

## Goal and working agreement

- **Goal:** rebuild the book's `TinyAgent` (*An Illustrated Guide to AI Agents*) as a modular Python package instead of notebooks. This is a quick refresher (about 2 days in total) before building an agent on **Databricks** for a client. LangChain/LangGraph come only after the book's agent is finished end to end.
- **Teaching format for each lesson:**
  1. Concept.
  2. A Python refresher, when relevant.
  3. Code for me to type myself in Positron.
  4. Notes on "why it's written this way", including deviations from the book.
  5. A runnable `examples/lessonNN.py`.
  6. What to look for in the output.
  7. Checkpoint questions.
  8. I say "done", Claude reviews my files and output and answers the questions.
  9. I commit.
- **Claude doesn't write the package files.** I type them. Claude prototypes and tests in its scratchpad first, against my local model, so the code it hands me is known to work.
- **No tests** until I explicitly ask for them. **One commit per chapter.**
- **Point out Databricks equivalents** along the way.

## Environment

- Conda env `tinyagent` (Python 3.12). Packages are installed with **uv** into the active conda env: `uv pip install -e ".[mcp]"`. Don't use `uv sync` or `uv add`, which create a separate `.venv`.
- Ollama at `http://localhost:11434/v1`. Chat model: `gemma4:e4b`, which supports `reasoning_effort` and native tool calls. Embeddings: `embeddinggemma`, because gemma4 can't embed.
- Run examples from `tiny_agent_experiment/` as **scripts in the terminal**. MCP's `asyncio.run` fails inside Positron's interactive console.
- Positron tips: turn on `files.trimTrailingWhitespace` and `files.insertFinalNewline`, and keep Pyright warnings visible. They would have caught most of the bugs below.

## Project layout (end of Day 1)

```
tiny_agent_experiment/
├── pyproject.toml                  # setuptools, src layout, zero deps; extra: mcp = ["mcp>=2.3"]
├── src/tinyagent/
│   ├── __init__.py                 # public API + __all__ (MCPTools NOT exported: optional dep)
│   ├── llm.py                      # Response (dataclass), LLM (urllib → /chat/completions), EmbeddingModel
│   ├── trajectory.py               # Step (dataclass), Trajectory (initialize/add)
│   ├── memory.py                   # Memory, TrimmingMemory, SummarizationMemory, LongTermMemory (RAG)
│   ├── tools.py                    # tool_to_schema, Tools (text), NativeTools, Skills
│   ├── toolbox.py                  # get_weather, add, multiply, today, days_between
│   ├── mcp_tools.py                # MCPTools(NativeTools): stdio MCP client
│   ├── planning.py                 # ReAct, NativeReAct, PlanAndSolve, Reflexion
│   └── agent.py                    # TinyAgent: the autonomy loop
├── mcp_servers/notes_server.py     # MCPServer("notes"): save_note, list_notes → notes.json (gitignored)
├── skills/weather_report/SKILL.md  # frontmatter (name, description) + instructions
├── examples/lesson01..05.py, lesson04b.py, mcp_raw.py
└── sessions/                       # these summaries
```

Commits: `ch01` → `chapter2` → `ch04` → `ch05` (tools) → `ch05b` (MCP) → `296e029` (Ch6, mislabelled "ch05" in the message). The `.gitignore` entry for `notes.json` is **not committed yet**.

## Architecture in one picture

```
TinyAgent(llm, memory, tools, planner, reflector=None)
  system prompt = "You are a helpful assistant." + planner.prompt + tools.prompt
  run(task):
    for step in range(planner.max_steps):
        reflector?.reflect(memory, step)
        response = llm.generate(memory.get_messages(), tools=tools.schemas)   # THOUGHT
        memory.add("assistant", content, tool_call=raw_tool_call)             # store RAW (OpenAI format)
        response = planner.parse(response); response = tools.parse(response)
        if tools.is_done(response): return answer                             # ANSWER
        result = tools.execute(response)                                      # ACTION (errors → strings)
        memory.add(role, observation, tool_call_id=id)                        # OBSERVATION
    return "Max steps reached without completion."
```
Every component is **injected** and **duck-typed**: `LLM.generate`, `Memory.add`/`get_messages`, and `Tools` with 6 hooks (`prompt`, `schemas`, `parse`, `execute`, `observation`, `is_done`). Any of them can be swapped without touching `agent.py`. Pointing at Databricks only changes `LLM(model=<endpoint>, base_url="https://<workspace>/serving-endpoints", api_key=<PAT>)`.

## Lessons completed

### L1: Ch1, skeleton
- Agent = LLM + memory + tools + planner in a loop. Composition plus dependency injection.
- `pyproject.toml`, the `src/` layout (catches "not actually installed" bugs early), editable installs, `__init__.py` and `__all__`, relative imports.

### L2: Ch2–3, LLM, Response, Trajectory, reasoning
- An LLM is **stateless**: messages in, one message out. OpenAI chat-completions is the common standard across servers.
- `Response` is an **adapter** that normalizes provider JSON (`reasoning` vs `reasoning_content`).
- `think=True` sends `reasoning_effort`, and the model reasons in a separate hidden channel. Observed: about 3× the completion tokens. With thinking off, the model still reasons, but visibly, in `content`.
- Refreshers: dataclasses (generated `__init__`/`__repr__`/`__eq__`; the mutable-default trap), and data-class vs service-class.
- Observed bug that motivated Ch4: "And of Japan?" → "Harmony". No history meant the model lost the context, and reasoning only elaborated the wrong premise.

### L3: Ch4, memory
- Full history, trimming (fails silently and confidently), summarization (the summarizer prompt *is* the strategy, since it changed the agent's voice and wrote a growing log), and RAG via `EmbeddingModel` + cosine similarity. Semantic match with no keyword overlap; it always returns k results, so a threshold or reranker is needed in production.
- Databricks: Vector Search with a Delta Sync index; offline batch embedding; chunking; ANN indexes; access-control metadata.
- Refreshers: inheritance, `super()`, `**kwargs` pass-through, `@staticmethod`, argsort with `sorted(key=lambda)`.

### L4a: Ch5, tools
- The LLM only emits text; **your code is the runtime**.
- Text tools (JSON in `content`, `user` + `OBSERVATION:`) vs **native** (`tools` schemas, `tool_calls` field, `tool` role; the server parses the model's trained tokens).
- `arguments` arrives as a JSON *string*.
- Errors are returned as observations. Raise only for bugs, security violations and exhausted budgets, and never leak secrets in error text.
- **Skills** = progressive disclosure. The rule of thumb: "how to do X" → skill; "what is true about Y" → RAG; "always/never" → system prompt.
- Refreshers: `@property`, `inspect.signature`, functions as values, closures, `dataclasses.replace`.

### L4b: Ch5, MCP
- MCP sits between the agent's tool registry and tool providers. **The LLM never sees MCP.**
- Host, client and server roles. Tools, resources and prompts. JSON-RPC 2.0 (requests, responses, notifications). Transports: stdio (subprocess; **stdout is the protocol, so never `print()`**) and streamable HTTP (remote, auth; what Databricks uses).
- Lifecycle: `initialize` → `notifications/initialized` → `tools/list` → `tools/call`.
- A tool error is `result.isError` (show it to the model). A protocol error is a JSON-RPC `error` such as -32601 (a client bug).
- MCP `inputSchema` equals the native tool `parameters`, so the mapping is a copy.
- `MCPTools` spawns one process per call (~0.4 s, stateless). The fix is a persistent session (an async agent, or a background event-loop thread).
- Late-binding closure bug: use a factory function, not a lambda in a loop.
- Security: stdio servers run your code with your permissions, and tool descriptions can carry prompt injection ("tool poisoning").
- Databricks: managed MCP servers (UC functions, Vector Search, Genie) run under UC permissions. Custom servers can go on Databricks Apps. Default to plain functions unless the tools are shared, owned by another team, or need governance.
- Debugging stdio: run the server alone first. It should just sit and wait.

### L5: Ch6, planning
- The loop (`_step` returns `None` to keep going), and `max_steps` as the safety net.
- Text `ReAct` (THOUGHT/ACTION regex, `final_answer`) vs `NativeReAct` (empty prompt, no-op parse; ReAct now lives in the model weights, the template and the server).
- **`tool_call_id`** pairing added (strict APIs require it).
- `PlanAndSolve` (plan as tools, like Claude Code's todo list) and `Reflexion` (a periodic critique).
- Observed on the same task: PlanAndSolve took **9 steps and ~6.2k tokens**, vs ReAct's **4 steps and ~1.5k**. The plan state ended up wrong (items ticked late, one never ticked). Reflexion only narrated progress. Planning is worth it for long tasks, human approval, fan-out and strong models. Prefer plain ReAct for short tasks and small models.
- Token cost grows **quadratically** with steps. **Never trim between a `tool_calls` message and its `tool` results**; trim or summarize in atomic groups.
- A better `max_steps` ending: a final no-tools "best answer + what's missing" call, plus a status for the caller.
- Refreshers: bound methods as tools (`self` excluded from the signature), regex (non-greedy, lookahead, `DOTALL`), optional injected components, dispatch dicts.

## Deliberate deviations from the book (so the printed code doesn't confuse me)

- `Response.content` is always a `str` (`or ""`).
- `Trajectory` lives in `trajectory.py` (the book puts it in `llm.py`).
- `TrimmingMemory(max_messages=4)` passes `**kwargs` through (the book drops `tool_call`).
- `Memory.add(..., tool_call_id=None)`, and `NativeTools.parse` keeps `"id"`.
- `Tools(requires_approval: list[str] | None = None)` (the book uses a mutable default `[]`).
- `tool_to_schema(func, name, description)`. The schema is stored in the registry at `add_tool(..., schema=None)`, so MCP can pass its own.
- `Tools.descriptions` includes the signatures. The text-tools prompt says "write this JSON" (not "ONLY JSON"), so it doesn't conflict with ReAct.
- `parse` uses `dataclasses.replace` (the book loses `metadata`).
- `execute` never raises: an unknown tool lists the valid names, and exceptions become strings.
- `Skills.add_skill` registers a real zero-argument `activate()` closure (the book uses a `__signature__` hack).
- `toolbox` uses `float` hints, and `days_between(start, end)`.
- `MCPTools(command, args)` extends **NativeTools** and uses `tool.input_schema`. It targets **mcp 2.3.0** (`mcp.server.mcpserver.MCPServer`; older tutorials use `FastMCP`).
- `ReAct.parse` only overwrites what it found. `PlanAndSolve` extends `NativeReAct` with `max_steps=15`. Reflexion sends a single-message transcript (the book's `[-6:]` slice can orphan a tool message).
- `TinyAgent` takes an optional `reflector`.

## My recurring bugs (watch for these)

1. Updating a method body but not its `def` signature (`__init__(self)` without `llm`; `_step(self, task)`). **Change the `def` line first, then the body, then the callers.**
2. A paste or replace accidentally deleting the neighbouring method (`get_messages` vanished).
3. Odd indentation (6/10 spaces) and trailing whitespace. Use `ruff format`.
4. A file typed but **not saved** (the 0-byte `notes_server.py` caused confusing client-side errors).

## Open items and next steps

- [ ] Commit the `.gitignore` change (`notes.json`).
- [ ] Optional: strip the duplicate `OBSERVATION: OBSERVATION:` in `show()` (`lesson05.py`).
- [ ] Optional exercises not yet done: harden `LLM.generate` (timeout + `HTTPError` body); fix `SummarizationMemory` so it keeps the original system prompt; a graceful `max_steps` ending; trigger Reflexion on loops or errors instead of every N steps; persistent MCP sessions.
- **Next: Lesson 6, Ch7 Evaluation** (Benchmark + Evaluator, measure instead of eyeball; maps to MLflow Tracing and Mosaic AI Agent Evaluation). Then Ch8 multi-agent, Ch9 multimodal (`MultimodalMemory`, image input), and Ch10 coding agent + CLI (`create_native_code_tools`, `Display`/`RichDisplay`, the `tinyagent` console script). After that, LangGraph and the Databricks agent.
- The reference code for each chapter is in the repo at `src/illustrated_agents/chapters/chN.py` and `src/illustrated_agents/*.py`. The notebooks are in `chapterNN/`.
