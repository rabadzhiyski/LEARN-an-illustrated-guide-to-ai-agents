import sys

from tinyagent import (
    LLM,
    Memory,
    NativeReAct,
    NativeTools,
    PlanAndSolve,
    ReAct,
    Reflexion,
    Skills,
    TinyAgent,
    Tools,
    toolbox,
)

FUNCTIONS = [toolbox.get_weather, toolbox.add, toolbox.multiply, toolbox.today, toolbox.days_between]
CITIES = "Get the weather in Sofia, Varna and Plovdiv, then tell me the average temperature."


def build(tools: Tools) -> Tools:
    for func in FUNCTIONS:
        tools.add_tool(func.__name__, func)
    return tools


def show(agent: TinyAgent) -> None:
    """Print the last run's trajectory: THOUGHT → ACTION → OBSERVATION … → ANSWER."""
    steps = agent.trajectory.runs[-1]["steps"]
    for i, step in enumerate(steps, 1):
        print(f"\n[{i}] THOUGHT: {step.thought[:150]!r}")
        if step.action:
            print(f"    ACTION: {step.action['tool']}({step.action['kwargs']})")
            print(f"    OBSERVATION: {step.observation}")
        else:
            print(f"    ANSWER: {step.answer}")
    tokens = sum(s.metadata["prompt_tokens"] for s in steps if s.metadata)
    print(f"\n→ {len(steps)} steps, ~{tokens} prompt tokens in total\n")


def text_react() -> None:
    agent = TinyAgent(LLM("gemma4:e4b"), Memory(), build(Tools()), ReAct())
    agent.run("How many days from today until 2026-12-25?")
    show(agent)


def native_react() -> None:
    agent = TinyAgent(LLM("gemma4:e4b", think=True), Memory(), build(NativeTools()), NativeReAct())
    agent.run("How many days from today until 2026-12-25?")
    show(agent)
    print("Roles + tool_call_ids in memory:")
    for m in agent.memory.messages:
        print(f"  {m['role']:<10} {m.get('tool_call_id', '')}")


def skill() -> None:
    tools = build(Skills())
    tools.add_skill("skills/weather_report/SKILL.md")
    agent = TinyAgent(LLM("gemma4:e4b", think=True), Memory(), tools, NativeReAct())
    agent.run("Give me a weather report for Sofia and Plovdiv")
    show(agent)


def plan() -> None:
    tools = build(NativeTools())
    planner = PlanAndSolve(tools)  # registers update_plan + mark_done into `tools`
    agent = TinyAgent(LLM("gemma4:e4b", think=True), Memory(), tools, planner)
    agent.run(CITIES)
    show(agent)
    print("Final plan state:\n" + planner._render())


def reflect() -> None:
    agent = TinyAgent(
        LLM("gemma4:e4b", think=True),
        Memory(),
        build(NativeTools()),
        NativeReAct(),
        reflector=Reflexion(LLM("gemma4:e4b"), interval=2),
    )
    agent.run(CITIES)
    show(agent)
    for m in agent.memory.messages:
        if str(m["content"]).startswith("[Reflection]"):
            print(m["content"])


DEMOS = {"text": text_react, "native": native_react, "skill": skill, "plan": plan, "reflect": reflect}

if __name__ == "__main__":
    # python examples/lesson05.py native    (or: text, skill, plan, reflect, all)
    choice = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, demo in DEMOS.items():
        if choice in (name, "all"):
            print(f"\n{'=' * 20} {name} {'=' * 20}")
            demo()
