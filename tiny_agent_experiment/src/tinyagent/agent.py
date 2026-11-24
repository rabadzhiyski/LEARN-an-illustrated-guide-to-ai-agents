from .llm import LLM
from .memory import Memory
from .trajectory import Trajectory


class TinyAgent:
    """A minimal, modular, and educational agent."""

    def __init__(self, llm: LLM, memory: Memory) -> None:
        self.llm = llm      
        self.memory = memory   # Ch4
        self.tools = None    # Ch5
        self.planner = None  # Ch6

        self.trajectory = Trajectory()

    def run(self, task: str) -> str:
        """Run the agent on a task."""
        self.memory.add("user", task)
        self.trajectory.initialize(task)
        return self._step()

    def _step(self) -> str:
        """Perform a single step: think, then maybe act."""
        response = self.llm.generate(self.memory.get_messages())
        self.memory.add("assistant", response.content)
        self.trajectory.add(response)
        return response.content

    def _execute_action(self, action: str) -> str:
        """Execute a tool action."""
        return f"Executed action: {action}"
