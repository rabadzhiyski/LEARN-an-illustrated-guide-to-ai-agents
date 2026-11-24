import asyncio
from typing import Callable

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .tools import NativeTools


class MCPTools(NativeTools):
    """Native tools discovered from an MCP server (stdio transport)."""

    def __init__(
        self,
        command: str,
        args: list[str],
        requires_approval: list[str] | None = None,
    ) -> None:
        super().__init__(requires_approval)
        self.server = StdioServerParameters(command=command, args=args)

        # Discovery: tools/list → register each tool with the server's own schema
        for tool in asyncio.run(self._list_tools()):
            schema = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema,  # MCP inputSchema == native `parameters`
                },
            }
            self.add_tool(
                tool.name,
                self._make_caller(tool.name),
                tool.description or "",
                schema=schema,
            )

    async def _list_tools(self) -> list:
        async with stdio_client(self.server) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                return result.tools

    async def _call_tool(self, name: str, kwargs: dict) -> str:
        async with stdio_client(self.server) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name, kwargs)
                text = "\n".join(
                    block.text for block in result.content if block.type == "text"
                )
                return f"Error: {text}" if result.is_error else text

    def _make_caller(self, name: str) -> Callable[..., str]:
        """A sync function the registry can call, bound to one remote tool."""

        def caller(**kwargs) -> str:
            return asyncio.run(self._call_tool(name, kwargs))

        return caller
