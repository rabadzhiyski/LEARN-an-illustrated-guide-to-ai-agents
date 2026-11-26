from pathlib import Path
from typing import Callable

from .llm import LLM, Response
from .memory import Memory
from .planning import ReAct, Reflexion
from .tools import Tools
from .trajectory import Trajectory


class TinyAgent:
    """A minimal, modular, and educational agent."""

    def __init__(
        self,
        llm: LLM,
        memory: Memory,
        tools: Tools,
        planner: ReAct,
        reflector: Reflexion | None = None,
        display: Callable | None = None,
    ) -> None:
        self.llm = llm
        self.memory = memory
        self.tools = tools
        self.planner = planner
        self.reflector = reflector
        self.display = display or (
            lambda event, data=None: None
        )  # silent default

        self.trajectory = Trajectory()

        # Each component contributes its own instructions
        system_prompt = (
            "You are a helpful assistant.\n"
            + self.planner.prompt
            + self.tools.prompt
        )
        self.memory.add("system", system_prompt)

    def run(self, task: str, image: str | Path | None = None) -> str:
        """Run the agent on a task until it answers or runs out of steps."""
        self.memory.add("user", task, image=image)
        self.trajectory.initialize(task)

        # The autonomy loop
        for step in range(self.planner.max_steps):
            if self.reflector:
                self.reflector.reflect(self.memory, step)
            result = self._step()
            if result is not None:
                return result

        return "Max steps reached without completion."

    def _step(self) -> str | None:
        """One iteration. Returns the answer when done, None to keep going."""
        # THOUGHT
        self.display("thinking")
        response = self.llm.generate(
            self.memory.get_messages(), tools=self.tools.schemas
        )
        self.memory.add(
            "assistant", response.content, tool_call=response.tool_call
        )

        # Parse: planner first (THOUGHT/ACTION split), then tools (JSON → tool_call)
        response = self.planner.parse(response)
        response = self.tools.parse(response)
        if response.reasoning:
            self.display("THOUGHT", response.reasoning)

        # ANSWER
        if self.tools.is_done(response):
            self.display("ANSWER", response.content)
            self.trajectory.add(response)
            return response.content

        self._execute_action(response)
        return None

    def _execute_action(self, response: Response) -> None:
        """ACTION + OBSERVATION."""
        self.display("ACTION", response.tool_call)
        result = self.tools.execute(response)
        role, observation = self.tools.observation(result)
        self.memory.add(
            role, observation, tool_call_id=response.tool_call.get("id")
        )
        self.trajectory.add(response, observation)
        self.display("OBSERVATION", observation)
