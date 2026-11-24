import sys
from pathlib import Path

from tinyagent import LLM, Memory, TinyAgent, toolbox
from tinyagent.mcp_tools import MCPTools

SERVER = Path(__file__).parent.parent / "mcp_servers" / "notes_server.py"
llm = LLM(model="gemma4:e4b")


def main() -> None:
    tools = MCPTools(command=sys.executable, args=[str(SERVER)])
    print("Discovered from server:", list(tools.registry))
    print("Schema the LLM sees:", tools.schemas[0], "\n")

    # MCP tools and local tools in one registry: the agent can't tell them apart
    tools.add_tool("get_weather", toolbox.get_weather)

    for q in [
        "Save a note: book a table for Friday at 8pm",
        "What notes do I have?",
        "What's the weather in Varna?",
    ]:
        agent = TinyAgent(llm, Memory(), tools)
        print(f"USER: {q}\n  → {agent.run(q)}")


if __name__ == "__main__":
    main()
