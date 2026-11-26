# Session 2: Evaluation, multi-agent, multimodal, coding agent, Ch7–Ch10 (2026-11-26)

## How to resume

Start a fresh Claude Code session in the repo root and say:

> Read `tiny_agent_experiment/sessions/2026-11-25-day1-core-agent-ch1-to-ch6.md` and
> `tiny_agent_experiment/sessions/2026-11-26-day2-eval-multiagent-multimodal-coding-ch7-to-ch10.md`.
> The book is finished. Let's start the next phase: <LangGraph | Databricks agent | open items>.

The working agreement and environment are the same as on Day 1 (see that file), with these changes:
- **Claude writes `examples/` demo files directly.** Copying long lines from the terminal hard-wraps them and breaks string literals. I type the package code.
- **Every code line Claude gives me is ≤ 88 characters.** `/copy` in Claude Code copies the last response without the terminal's wrapping.
- **After saving, check the change reached the disk** (`grep -n ...` or `head`). Twice an edit never landed. The repo is on SynologyDrive.
- `ruff` is **not** installed in the `tinyagent` env yet: `uv pip install ruff`, then
  `ruff check --select I --fix src && ruff format src`.

## Project layout (end of Day 2)

New since Day 1:
```
src/tinyagent/
├── evaluation.py     # Scorer, Benchmark, Evaluator(create_agent, trials); scorers below
├── multi_agent.py    # SubAgent (callable class, agent-as-tool), total_tokens(agent)
├── memory.py         # + encode_image(path) → data URL, MultimodalMemory(image=...)
├── code_tools.py     # create_code_tools(workspace): list/read/write_file, run_python
├── display.py        # Display: THOUGHT / ACTION / OBSERVATION / ANSWER in ANSI colour
├── cli.py            # main(): argparse + REPL (/tools /clear /help /exit)
└── agent.py          # + run(task, image=None), + optional display (silent default)
pyproject.toml        # + [project.scripts] tinyagent = "tinyagent.cli:main"
examples/lesson06..09.py, examples/assets/receipt.png
```

Commits: `1cc0383` (Ch7, message "created evaluation exercises") → `ec968bf` ch08 → `283e738` ch09 → ch10 (with this file).

## Lessons completed

### L6: Ch7, evaluation
- **Measure instead of eyeballing.** Benchmark = examples + scorer. Evaluator = **a fresh agent per example** (it takes a factory, not an instance, or memory contaminates the next example), score, then aggregate.
- **The scorer ladder:** exact match (closed answer space), programmatic check (IFEval-style
  `check(text)`), LLM-as-a-judge (open answers; the judge itself needs evaluating).
- **Scorers take `(prediction, example, steps)`.** The trajectory lets you check *how* the agent got there (`used_tools`). Same idea as an MLflow scorer receiving the `trace`.
- **Observed:**
  - MMLU 0.33 as multiple choice vs 1.00 as judged open QA, on the same facts (format effect or a lenient judge; read the trace to tell).
  - IFEval: the ad copy failed because the model wrote 3 options.
  - **`qwen3.5:4b` as judge passed "1200 cc" as correct and flipped between runs (2/4 agreement); `gemma4:e4b` scored 4/4.** Validate a judge against human labels (`JUDGE_CHECK`) before trusting it.
  - Text vs native ReAct both scored 1.00: a saturated benchmark can't tell designs apart, so you need hard cases.
  - Time-dependent ground truth must be computed, or use a frozen clock.
- **Answers to remember:**
  - Substring scorers give false passes ("590" contains "59"); parse the numbers instead.
  - 3 examples isn't evidence: use `trials` (pass^k) and compare examples one by one.
- **Databricks:** `mlflow.genai.evaluate(data, predict_fn, scorers)`, evaluation datasets in UC, judge alignment with Review App labels.

### L7: Ch8, multi-agent (agents as tools)
- An agent is a function `str → str`, so it can be a tool. Orchestrator/supervisor pattern; `agent.py` is unchanged.
- **`SubAgent`** is a class with `__call__(task)`. `inspect.signature` on an instance drops `self`, so `tool_to_schema` works unchanged. **Pass an explicit description** (otherwise the model sees the class docstring). `fresh=True` by default (the book reuses one instance, so memory leaks between delegations).
- **Observed:** multi used ~3.0k tokens vs ~2.3k for a single agent with the same answer. **The Evaluator reported multi as cheaper**, because sub-agent tokens live in other trajectories. Wall-clock time (18 s vs 11 s) gave it away. Fix: count usage at the LLM layer, which is what tracing does.
- **Lossy handoff:** the orchestrator dropped "euro" when rewriting the task.
- When splitting pays off: many tools, noisy subtasks (context isolation), different models or permissions or owners, parallel subtasks.
- **Databricks:** Agent Bricks Multi-Agent Supervisor (Genie spaces, Knowledge Assistants, UC functions, MCP as sub-agents), or code-first with `ResponsesAgent`. Nested agents appear as child spans in one MLflow trace.

### L8: Ch9, multimodal (image input)
- `content` becomes a list of parts: `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}` plus a text part. Nothing else in the loop changes.
- **Observed:**
  - Cover → "dolphin" plus the title, and a follow-up answered without re-sending the image.
  - The image cost ~650 tokens (681 vs 36).
  - The receipt was read from pixels and totalled with 7 tool calls (31.40). `3 × 2.4 = 7.199999999999999`, so money tools should use `Decimal`/cents.
  - **Plain `Memory` silently drops the image** (`**kwargs`). Raise on programmer errors.
- The image is re-sent on every step (8× for the receipt): downscale it, transcribe it once, or keep only the latest.
- Summarization and LongTerm memory assume `content` is a `str`, and break with image lists.
- **Databricks:** images in UC Volumes, with paths (not base64) in memory and traces; `ai_query`/`ai_parse_document` for batch work.

### L9: Ch10, coding agent and CLI
- **Same loop, different tools.** Read → write → run → verify. Engineering around it: a workspace fence, human approval, clipped observations, a display, a CLI.
- **`safe_path` uses `resolve()` + `is_relative_to`.** The book's `str.startswith` lets `../ws-evil/` through (demonstrated in the `sandbox` demo). An absolute path on the right of `root / ...` discards `root`, but is still caught.
- **The fence only protects the file tools.** `run_python` runs arbitrary code with my permissions. Real defenses: approval (here), a container/VM, or serverless sandboxes in production.
- `run_python(path)` instead of `execute_python(code)`: the code on disk is exactly the code that ran, and it's approved twice.
- **Display** events fire *after* parsing (text and native look the same), and the display is optional with a silent default, so all earlier examples are unchanged.
- **Console script:** `[project.scripts]`, then `uv pip install -e ".[mcp]"`; the `tinyagent -w <dir>` command works.
- **Observed:** gemma4 fixed both bugs in `stats.py` in 4 steps and **verified by running**.

## Deliberate deviations from the book (Day 2)

- Scorers take `steps` as a third argument. `exact_choice` handles "no letter found" (the book crashes). The Evaluator catches per-example exceptions and records steps, tools, tokens, seconds and `trials`. The judge asks for 0/1 and is built with a `make_judge(llm)` factory.
- `SubAgent` class (fresh by default) instead of the book's closure functions.
- `MultimodalMemory(image=path|URL|data URL)` with the MIME type from `mimetypes`, calling `super().add` (keeps `tool_call_id`). The book hardcodes jpeg/png and drops `tool_call_id`.
- `run` is edited in `agent.py` (no subclass).
- 4 code tools (not 7), `is_relative_to` (not `startswith`), exceptions turn into observations through `Tools.execute`, `_clip` on outputs.
- Display uses the THOUGHT/ACTION/OBSERVATION/ANSWER events; no `rich`.
- The approval prompt shows a 120-character preview.

## My recurring bugs (Day 2 additions)

5. **Edits not reaching the disk** (`__init__.py` stayed unchanged twice). Save, then `grep` to check.
6. **Pasting a lesson's heading text into code**, or one file's code into another (the display code ended up in `code_tools.py`).
7. **Typos in signatures:** `display: Callable, | None-None,`, and an assignment without `self.x =`.

## Open items and next steps

- [ ] `uv pip install ruff` in the env; format `src/`.
- [ ] The Lesson 9 checkpoint questions (answered by Claude in session, see below).
- [ ] Optional hardening from Day 1 + 2:
  - [ ] LLM timeout + `HTTPError` body; count usage at the LLM layer (fixes the sub-agent token undercount).
  - [ ] `Memory.add` raises on an unexpected `image`; `text_of(content)` for Summarization/LongTerm memory.
  - [ ] A graceful `max_steps` ending; trajectory scorers that walk nested agents; numeric scorers instead of substring.
  - [ ] Persistent MCP sessions; a `read_lines`/`edit_file` (string replace) tool instead of whole-file writes.
- **Next phase:** LangGraph (rebuild the same agent as a graph and map each TinyAgent part to it), then the client's agent on **Databricks** (Model Serving endpoint as the `LLM`, UC functions or managed MCP as tools, Vector Search for RAG, MLflow Tracing + `mlflow.genai.evaluate`, served as a `ResponsesAgent`).

### Lesson 9 checkpoint answers (for reference)
1. `open(os.path.expanduser("~/.ssh/id_rsa")).read()` inside a script run by `run_python` ignores `safe_path`. What stops it here is only the human approval. In production: a container/VM or serverless sandbox with no secrets mounted and no network.
2. Path not code: what runs is on disk, reviewable and reproducible. Downside: one-off checks need a file write (an extra approval), and files pile up in the workspace.
3. Whole-file `write_file` on 2,000 lines costs huge output tokens and lets the model silently drop or alter unrelated lines. Use a string-replace `edit_file(path, old, new)` (Claude Code's Edit tool), where `old` must match exactly once.
4. Plan mode: build the agent with a tools object that has no `write_file`/`run_python` (or with every write denied). No `agent.py` change, the same dependency injection.
5. The whole history goes in, including every earlier read/write/run observation. Reach for `TrimmingMemory`/`SummarizationMemory`, but trim in atomic groups (never split a `tool_calls` message from its `tool` results). `/clear` between unrelated tasks.
6. Databricks: workspace = a UC Volume or a Git folder; run = serverless compute or a sandboxed code-execution UC function, not the serving container. Approval: an app UI (Databricks Apps), or a review/approval step in the agent's protocol instead of `input()`.
