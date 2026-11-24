from dataclasses import dataclass

from .llm import Response


@dataclass
class Step:
    """A single step in an agent's trajectory."""

    thought: str = ""
    action: dict | None = None
    observation: str | None = None
    answer: str | None = None
    metadata: dict | None = None


class Trajectory:
    """Records agent execution as a list of runs, each with a list of steps."""

    def __init__(self) -> None:
        self.runs: list[dict] = []

    def initialize(self, query: str) -> None:
        """Register a new run for the given query."""
        self.runs.append({"query": query, "steps": []})

    def add(self, response: Response, observation: str | None = None) -> None:
        """Record a step. With an observation it's an ACTION, otherwise an ANSWER."""
        step = Step(thought=response.reasoning or "", metadata=response.metadata)

        if observation is not None:
            step.action = response.tool_call
            step.observation = observation
        else:
            step.answer = response.content

        self.runs[-1]["steps"].append(step)
