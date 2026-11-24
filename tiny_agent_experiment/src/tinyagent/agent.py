from .llm import LLM
from .trajectory import Trajectory


class TinyAgent:
    """A minimal, modular, and educational agent."""

    def __init__(self, llm: LLM) -> None:
        self.llm = llm      
        self.memory = None   # Ch4
        self.tools = None    # Ch5
        self.planner = None  # Ch6

        self.trajectory = Trajectory()

    def run(self, task: str) -> str:
        """Run the agent on a task."""
        self.trajectory.initialize(task)
        return self._step(task)

    def _step(self, task: str) -> str:
        """Perform a single step: think, then maybe act."""
        messages = [{"role": "user", "content": task}] # context beats intelligence!
        response = self.llm.generate(messages)
        self.trajectory.add(response)
        return response.content

    def _execute_action(self, action: str) -> str:
        """Execute a tool action."""
        return f"Executed action: {action}"
