from typing import Callable

from .agent import TinyAgent


class SubAgent:
    """Wrap an agent factory so a whole agent can be registered as one tool."""

    def __init__(
        self, create_agent: Callable[[], TinyAgent], fresh: bool = True
    ) -> None:
        self.create_agent = create_agent
        self.fresh = fresh
        self.agents: list[TinyAgent] = []  # every agent we ran, for inspection

    def __call__(self, task: str) -> str:
        """Run the sub-agent on `task` and return only its final answer."""
        if self.fresh or not self.agents:
            self.agents.append(self.create_agent())
        return self.agents[-1].run(task)


def total_tokens(agent: TinyAgent) -> int:
    """Prompt + completion tokens over all runs of one agent."""
    return sum(
        s.metadata["prompt_tokens"] + s.metadata["completion_tokens"]
        for run in agent.trajectory.runs
        for s in run["steps"]
        if s.metadata
    )
