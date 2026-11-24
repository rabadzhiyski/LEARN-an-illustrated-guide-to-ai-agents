from .llm import LLM, Response
from .memory import Memory
from .tools import Tools
from .trajectory import Trajectory


class TinyAgent:
    """A minimal, modular, and educational agent."""

    def __init__(self, llm: LLM, memory: Memory, tools: Tools) -> None:
        self.llm = llm
        self.memory = memory   
        self.tools = tools    
        self.planner = None  # Ch6

        self.trajectory = Trajectory()

        # The system prompt is assembled from the components
        system_prompt = "You're a helpful assistant.\n" + self.tools.prompt
        self.memory.add("system", system_prompt)


    def run(self, task: str) -> str:
        """Run the agent on a task."""
        self.memory.add("user", task)
        self.trajectory.initialize(task)
        return self._step()

    def _step(self) -> str:
        """Perform a single step: think, then maybe act."""
        response = self.llm.generate(
            self.memory.get_messages(), tools=self.tools.schemas
            )
        # Store the new reply (incl. native tool_calls) BEFORE parsing
        self.memory.add("assistant", response.content, tool_call=response.tool_call)

        response = self.tools.parse(response)
        if self.tools.is_done(response):
            self.trajectory.add(response)
            return response.content

        return self._execute_action(response)

    def _execute_action(self, response: Response) -> str:
        """ACTION: run the tool. OBSERVATION: store the result."""
        result = self.tools.execute(response)
        role, observation = self.tools.observation(result)
        self.memory.add(role, observation)
        self.trajectory.add(response, observation)
        return observation

