import sys
from pathlib import Path

from tinyagent import (
    LLM,
    Memory,
    MultimodalMemory,
    NativeReAct,
    NativeTools,
    TinyAgent,
    toolbox,
)

HERE = Path(__file__).parent
COVER = HERE.parent.parent / "images" / "cover.jpg"  # the book cover, in the repo
RECEIPT = HERE / "assets" / "receipt.png"


def make_agent(memory: Memory, *funcs) -> TinyAgent:
    tools = NativeTools()
    for func in funcs:
        tools.add_tool(func.__name__, func)
    return TinyAgent(LLM("gemma4:e4b", think=True), memory, tools, NativeReAct())


def show(agent: TinyAgent) -> None:
    """Steps of the last run, with the prompt size the model saw at each step."""
    for i, s in enumerate(agent.trajectory.runs[-1]["steps"], 1):
        tokens = s.metadata["prompt_tokens"] if s.metadata else "?"
        if s.action:
            print(f"[{i}] ({tokens} tok) {s.action['tool']}({s.action['kwargs']})")
            print(f"        → {s.observation}")
        else:
            print(f"[{i}] ({tokens} tok) ANSWER: {s.answer[:200]!r}")


def cover() -> None:
    """See an image, then ask a follow-up: the image stays in memory."""
    agent = make_agent(MultimodalMemory())
    agent.run("Which animal is on this book cover, and what is the title?", COVER)
    show(agent)
    agent.run("What colour is the subtitle text?")  # no image passed this time
    show(agent)


def blind() -> None:
    """Same call, plain Memory: the image is silently swallowed by **kwargs."""
    agent = make_agent(Memory())
    agent.run("Which animal is on this book cover, and what is the title?", COVER)
    show(agent)


def receipt() -> None:
    """Perception + action: read numbers from pixels, compute with tools."""
    agent = make_agent(MultimodalMemory(), toolbox.add, toolbox.multiply)
    agent.run(
        "Read this receipt and compute the TOTAL. Use the tools for every "
        "calculation.",
        RECEIPT,
    )
    show(agent)
    print("expected 31.40")


def peek() -> None:
    """What the image looks like inside a message (truncated)."""
    memory = MultimodalMemory()
    memory.add("user", "What is this?", image=RECEIPT)
    part = memory.messages[0]["content"][0]["image_url"]["url"]
    print(f"{part[:60]}…  ({len(part):,} characters)")
    print(memory.messages[0]["content"][1])


DEMOS = {"peek": peek, "cover": cover, "blind": blind, "receipt": receipt}

if __name__ == "__main__":
    # python examples/lesson08.py cover   (or: peek, blind, receipt, all)
    choice = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, demo in DEMOS.items():
        if choice in (name, "all"):
            print(f"\n{'=' * 20} {name} {'=' * 20}")
            demo()
