class TinyAgent:
    """A minimal, modular, and educational agent."""

    def __init__(self) -> None:
        self.llm = None      # Ch2-3
        self.memory = None   # Ch4
        self.tools = None    # Ch5
        self.planner = None  # Ch6

    def run(self, task: str) -> str:
        """Run the agent on a task."""
        return self._step(task)

    def _step(self, task: str) -> str:
        """Perform a single step: think, then maybe act."""
        return f"Received: {task}"

    def _execute_action(self, action: str) -> str:
        """Execute a tool action."""
        return f"Executed action: {action}"
