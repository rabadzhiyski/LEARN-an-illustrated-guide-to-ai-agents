from tinyagent import LLM, Memory, NativeTools, Skills, TinyAgent, Tools
from tinyagent import toolbox

llm = LLM(model="gemma4:e4b")
FUNCTIONS = [toolbox.get_weather, toolbox.add, toolbox.multiply, toolbox.today, toolbox.days_between]


def build(tools: Tools) -> Tools:
    for func in FUNCTIONS:
        tools.add_tool(func.__name__, func)
    return tools


def main() -> None:
    questions = ["What's the weather in Sofia?", "What is 1234.5 times 6.78?", "Hi! Who are you?"]

    for tools_cls in (Tools, NativeTools):
        print(f"\n===== {tools_cls.__name__} =====")
        for q in questions:
            agent = TinyAgent(llm, Memory(), build(tools_cls()))
            print(f"USER: {q}\n  → {agent.run(q)}")
            print(f"  assistant msg in memory: {agent.memory.messages[2]}")

    # What the model actually sees in each mode
    print("\n===== Text-mode system prompt =====")
    print(build(Tools()).prompt)
    print("===== Native schema for `add` =====")
    print(build(NativeTools()).schemas[1])

    # Errors are observations, not crashes
    print("\n===== Error handling =====")
    from tinyagent import Response
    tools = build(NativeTools())
    print(tools.execute(Response(tool_call={"tool": "weathr", "kwargs": {}})))
    print(tools.execute(Response(tool_call={"tool": "days_between", "kwargs": {"start": "yesterday", "end": "2026-12-25"}})))

    # Skills: progressive disclosure
    print("\n===== Skills =====")
    skills = build(Skills())
    skills.add_skill("skills/weather_report/SKILL.md")
    print("In context before activation:", skills.schemas[-1]["function"]["description"])
    agent = TinyAgent(llm, Memory(), skills)
    print("After one step:", agent.run("Give me a weather report for Sofia and Plovdiv"))

    # The cliffhanger
    print("\n===== Needs two tools =====")
    agent = TinyAgent(llm, Memory(), build(NativeTools()))
    print(agent.run("How many days from today until 2026-12-25?"))


if __name__ == "__main__":
    main()
