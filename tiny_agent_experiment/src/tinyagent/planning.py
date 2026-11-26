import re

from .llm import LLM, Response
from .memory import Memory
from .tools import Tools


class ReAct:
    """Text-based ReAct: the model writes THOUGHT and ACTION, we parse them."""

    def __init__(self, max_steps: int = 10) -> None:
        self.max_steps = max_steps

    @property
    def prompt(self) -> str:
        return """
# ReAct (Reason + Act)

Work step by step. Perform exactly ONE step per reply, in this format:

THOUGHT: <your reasoning about what to do next>
ACTION:
{"tool": "tool_name", "kwargs": {"param": "value"}}

After each ACTION you will receive an OBSERVATION with the result.
Never write the OBSERVATION yourself.

When you have the final answer, use the special `final_answer` tool:

THOUGHT: <why you are done>
ACTION:
{"tool": "final_answer", "kwargs": "<your final answer>"}
"""

    def parse(self, response: Response) -> Response:
        """Split the reply into THOUGHT (→ reasoning) and ACTION (→ content)."""
        text = response.content
        thought = re.search(
            r"THOUGHT:\s*(.+?)(?=ACTION:|OBSERVATION:|$)", text, re.DOTALL
        )
        action = re.search(
            r"ACTION:\s*(.+?)(?=THOUGHT:|OBSERVATION:|$)", text, re.DOTALL
        )
        if action:
            response.content = action.group(1).strip()
        if thought:
            response.reasoning = thought.group(1).strip()
        return response


class NativeReAct(ReAct):
    """ReAct with a reasoning model + native tool calling: nothing to prompt or parse."""

    @property
    def prompt(self) -> str:
        return ""

    def parse(self, response: Response) -> Response:
        return response


class PlanAndSolve(NativeReAct):
    """Give the agent tools to write and track its own plan."""

    def __init__(self, tools: Tools, max_steps: int = 15) -> None:
        super().__init__(max_steps)
        self.plan: list[dict] = []
        tools.add_tool("update_plan", self.update_plan)
        tools.add_tool("mark_done", self.mark_done)

    @property
    def prompt(self) -> str:
        return (
            "# Planning\n"
            "For tasks with several steps, first call `update_plan` with the list of "
            "sub-tasks, then call `mark_done` after finishing each one. "
            "Revise the plan with `update_plan` if you learn something new.\n"
        )

    def update_plan(self, items: list) -> str:
        """Create or replace the plan with a list of sub-task descriptions."""
        self.plan = [{"task": str(item), "done": False} for item in items]
        return f"Plan updated:\n{self._render()}"

    def mark_done(self, index: int) -> str:
        """Mark a sub-task as completed, by its 1-based number."""
        self.plan[int(index) - 1]["done"] = True
        return f"Plan:\n{self._render()}"

    def _render(self) -> str:
        return "\n".join(
            f"[{'x' if item['done'] else ' '}] {i}. {item['task']}"
            for i, item in enumerate(self.plan, 1)
        )


class Reflexion:
    """Every `interval` steps, critique recent progress and inject a correction."""

    def __init__(self, llm: LLM, interval: int = 3) -> None:
        self.llm = llm
        self.interval = interval

    def reflect(self, memory: Memory, step: int) -> None:
        if step == 0 or step % self.interval:
            return
        transcript = "\n".join(
            f"{m['role']}: {m.get('content') or m.get('tool_calls')}"
            for m in memory.get_messages()[1:]  # skip the system prompt
        )
        critique = self.llm.generate(
            [
                {
                    "role": "system",
                    "content": (
                        "You review an AI agent's work. In at most 3 sentences: is it "
                        "making progress toward the user's task? If not, what should "
                        "it do differently?"
                    ),
                },
                {"role": "user", "content": transcript},
            ]
        )
        memory.add("user", f"[Reflection] {critique.content}")
