import datetime
import sys

from tinyagent import (
    LLM,
    Benchmark,
    Evaluator,
    Memory,
    NativeReAct,
    NativeTools,
    SubAgent,
    TinyAgent,
    programmatic,
    toolbox,
    total_tokens,
)

MODEL = "gemma4:e4b"


def make_agent(*funcs) -> TinyAgent:
    """A native-ReAct agent with the given plain functions as tools."""
    tools = NativeTools()
    for func in funcs:
        tools.add_tool(func.__name__, func)
    return TinyAgent(LLM(MODEL, think=True), Memory(), tools, NativeReAct())


# --- Specialists ------------------------------------------------------------


def math_agent() -> TinyAgent:
    return make_agent(toolbox.add, toolbox.multiply)


def date_agent() -> TinyAgent:
    return make_agent(toolbox.today, toolbox.days_between)


def single_agent() -> TinyAgent:
    """Baseline: one agent that holds every tool itself."""
    return make_agent(
        toolbox.add, toolbox.multiply, toolbox.today, toolbox.days_between
    )


# --- Orchestrator -----------------------------------------------------------


def orchestrator() -> TinyAgent:
    """An agent whose only tools are other agents."""
    agent = make_agent()
    agent.tools.add_tool(
        "ask_math_agent",
        SubAgent(math_agent),
        "Delegate arithmetic to the math specialist. "
        "Pass a full, self-contained question.",
    )
    agent.tools.add_tool(
        "ask_date_agent",
        SubAgent(date_agent),
        "Delegate date questions (today, days between dates) to the date "
        "specialist. Pass a full, self-contained question.",
    )
    return agent


def sub_agents(agent: TinyAgent) -> list[tuple[str, TinyAgent]]:
    """Every sub-agent instance an orchestrator ran, with its tool name."""
    return [
        (name, sub)
        for name, tool in agent.tools.registry.items()
        if isinstance(tool["function"], SubAgent)
        for sub in tool["function"].agents
    ]


def show(agent: TinyAgent, indent: str = "") -> None:
    for i, s in enumerate(agent.trajectory.runs[-1]["steps"], 1):
        if s.action:
            print(f"{indent}[{i}] {s.action['tool']}({s.action['kwargs']})")
            print(f"{indent}    → {s.observation[:80]!r}")
        else:
            print(f"{indent}[{i}] ANSWER: {s.answer[:100]!r}")


DAYS = (datetime.date(2030, 1, 1) - datetime.date.today()).days
TASK = "If I save 4 euro per day from today until 2030-01-01, how much will I have?"


def multi() -> None:
    agent = orchestrator()
    agent.run(TASK)
    show(agent)
    sub_tokens = 0
    for name, sub in sub_agents(agent):
        print(f"   ↳ {name} ({total_tokens(sub)} tok)")
        show(sub, "      ")
        sub_tokens += total_tokens(sub)
    print(f"\nexpected {DAYS * 4}")
    print(f"orchestrator {total_tokens(agent)} tok + sub-agents {sub_tokens} tok")


def single() -> None:
    agent = single_agent()
    agent.run(TASK)
    show(agent)
    print(f"\nexpected {DAYS * 4}")
    print(f"single agent {total_tokens(agent)} tok")


def contains(expected: str):
    """A programmatic check: the number appears in the answer (with or without .0)."""
    return lambda t: expected in t.replace(",", "")


MIXED = Benchmark(
    "Mixed math + dates",
    [
        {"task": TASK, "check": contains(str(DAYS * 4))},
        {
            "task": "How many days are there from 2026-01-01 to 2026-03-01, "
            "multiplied by 3?",
            "check": contains("177"),
        },
        {"task": "What is 12.5 times 8, plus 7?", "check": contains("107")},
    ],
    programmatic,
)


def evaluate() -> None:
    """Ch7 pays off: compare the two designs on numbers, not vibes."""
    for label, factory in [("single", single_agent), ("multi", orchestrator)]:
        result = Evaluator(factory).run(MIXED)
        secs = sum(r["seconds"] for r in result["results"])
        print(
            f"{label:<7} score={result['score']:.2f}  "
            f"avg_steps={result['avg_steps']:.1f}  "
            f"tokens(reported)={result['total_tokens']}  time={secs:.0f}s"
        )


DEMOS = {"multi": multi, "single": single, "eval": evaluate}

if __name__ == "__main__":
    # python examples/lesson07.py multi   (or: single, eval, all)
    choice = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, demo in DEMOS.items():
        if choice in (name, "all"):
            print(f"\n{'=' * 20} {name} {'=' * 20}")
            demo()
