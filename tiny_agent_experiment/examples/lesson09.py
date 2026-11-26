import shutil
import sys
import tempfile
from pathlib import Path

from tinyagent import (
    LLM,
    Display,
    MultimodalMemory,
    NativeReAct,
    Response,
    TinyAgent,
    create_code_tools,
)

BUGGY = '''def mean(values):
    return sum(values) / len(values) - 1


def median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle]


if __name__ == "__main__":
    assert mean([1, 2, 3, 4]) == 2.5, mean([1, 2, 3, 4])
    assert median([1, 2, 3, 4]) == 2.5, median([1, 2, 3, 4])
    assert median([3, 1, 2]) == 2
    print("all checks passed")
'''


def call(tools, name: str, **kwargs) -> str:
    """Run a tool the way the agent would: errors come back as text."""
    return str(tools.execute(Response(tool_call={"tool": name, "kwargs": kwargs})))


def sandbox() -> None:
    """No LLM: poke at the workspace fence directly."""
    base = Path(tempfile.mkdtemp())
    workspace, evil = base / "ws", base / "ws-evil"
    workspace.mkdir()
    evil.mkdir()
    (workspace / "stats.py").write_text(BUGGY)
    (evil / "secret.txt").write_text("API_KEY=sk-not-for-the-agent")
    tools = create_code_tools(workspace)

    print("list_files     →", call(tools, "list_files"))
    print("read ../       →", call(tools, "read_file", path="../ws-evil/secret.txt"))
    print("read /etc      →", call(tools, "read_file", path="/etc/hostname"))

    # Why the book's check is not enough: a string prefix is not a directory
    target = (workspace / "../ws-evil/secret.txt").resolve()
    book_ok = str(target).startswith(str(workspace))
    print("\nstartswith     →", book_ok, "(book: allowed!)")
    print("is_relative_to →", target.is_relative_to(workspace), "(ours: refused)")
    shutil.rmtree(base)


def fix() -> None:
    """The coding agent fixes a buggy file. You approve each write/run."""
    workspace = Path(tempfile.mkdtemp())
    (workspace / "stats.py").write_text(BUGGY)
    print(f"workspace: {workspace}\n")
    agent = TinyAgent(
        LLM("gemma4:e4b", think=True),
        MultimodalMemory(),
        create_code_tools(workspace),
        NativeReAct(max_steps=15),
        display=Display(),
    )
    agent.run("stats.py has failing checks. Find and fix the bugs, then run it.")
    print("---- stats.py after the agent ----")
    print((workspace / "stats.py").read_text())


DEMOS = {"sandbox": sandbox, "fix": fix}

if __name__ == "__main__":
    # python examples/lesson09.py sandbox   (or: fix)
    DEMOS[sys.argv[1] if len(sys.argv) > 1 else "sandbox"]()
