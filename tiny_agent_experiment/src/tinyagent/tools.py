import inspect
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from .llm import Response

# Python type → JSON Schema type
TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def tool_to_schema(func: Callable, name: str, description: str = "") -> dict:
    """Build an OpenAI-style tool schema from a function's signature."""
    properties, required = {}, []
    for param_name, param in inspect.signature(func).parameters.items():
        properties[param_name] = {"type": TYPE_MAP.get(param.annotation, "string")}
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description or inspect.getdoc(func) or "",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


class Tools:
    """Text-based tool registry: the LLM writes a JSON action in its reply."""

    def __init__(self, requires_approval: list[str] | None = None) -> None:
        self.registry: dict[str, dict] = {}
        self.requires_approval = requires_approval or []

    def add_tool(
        self,
        name: str,
        func: Callable,
        description: str = "",
        schema: dict | None = None,
    ) -> None:
        """Register a tool. Description defaults to the docstring."""
        description = description or inspect.getdoc(func) or ""
        self.registry[name] = {
            "function": func,
            "description": description,
            "schema": schema or tool_to_schema(func, name, description),
        }

    @property
    def descriptions(self) -> str:
        """One line per tool, including its signature, for the system prompt."""
        return "\n".join(
            f"- `{name}{inspect.signature(tool['function'])}`: {tool['description']}"
            for name, tool in self.registry.items()
        )

    @property
    def prompt(self) -> str:
        return f"""
# Tools

If needed, you can use the following tools:

{self.descriptions}

To use a tool, write this JSON:
{{"tool": "tool_name", "kwargs": {{"param": "value"}}}}
"""

    @property
    def schemas(self) -> list[dict] | None:
        """Text-based tools live in the prompt, not in the API call."""
        return None

    def parse(self, response: Response) -> Response:
        """Extract a JSON tool call from the response text, if there is one."""
        text = response.content
        if '"tool"' not in text:
            return response
        start, end = text.find("{"), text.rfind("}") + 1
        try:
            tool_call = json.loads(text[start:end])
        except json.JSONDecodeError:
            return response  # not valid JSON: treat as a normal answer
        return replace(response, tool_call=tool_call)

    def execute(self, response: Response) -> Any:
        """Run the requested tool. Errors are returned, never raised."""
        name = response.tool_call["tool"]
        kwargs = response.tool_call.get("kwargs") or {}

        if name not in self.registry:
            return f"Error: tool '{name}' not found. Available: {list(self.registry)}"

        # Human-in-the-loop for dangerous tools
        if name in self.requires_approval:
            answer = input(f"Allow {name}({kwargs})? [y/N] ").strip().lower()
            if answer not in ("y", "yes"):
                return f"Tool '{name}' was denied by the user."

        try:
            return self.registry[name]["function"](**kwargs)
        except Exception as e:
            return f"Error running '{name}': {type(e).__name__}: {e}"

    def observation(self, result: Any) -> tuple[str, str]:
        """Text mode: the model has no 'tool' role, so results come back as the user."""
        return "user", f"OBSERVATION: {result}"

    def is_done(self, response: Response) -> bool:
        """Done when there's no tool call (or an explicit final_answer, used in Ch6)."""
        if not response.tool_call:
            return True
        if response.tool_call["tool"] == "final_answer":
            response.content = str(response.tool_call.get("kwargs", ""))
            return True
        return False


class NativeTools(Tools):
    """Native function calling: schemas go in the API call, results use role 'tool'."""

    @property
    def prompt(self) -> str:
        return ""  # the server injects the schemas into the prompt for us

    @property
    def schemas(self) -> list[dict]:
        return [tool["schema"] for tool in self.registry.values()]

    def parse(self, response: Response) -> Response:
        """Convert the OpenAI tool_call format into our {'tool', 'kwargs'} format."""
        if not response.tool_call:
            return response
        function = response.tool_call["function"]
        args = function["arguments"]
        if isinstance(args, str):  # OpenAI spec: arguments is a JSON *string*
            args = json.loads(args) if args else {}
        tool_call = {
              "tool": function["name"],
              "kwargs": args,
              "id": response.tool_call.get("id"),
          }
        return replace(response, tool_call=tool_call)

    def observation(self, result: Any) -> tuple[str, str]:
        return "tool", str(result)

    def is_done(self, response: Response) -> bool:
        return not response.tool_call


def _parse_frontmatter(text: str) -> dict:
    """Parse simple `key: value` lines (a tiny YAML subset, no dependencies)."""
    result = {}
    for line in text.strip().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip()
    return result


class Skills(NativeTools):
    """Native tools + skills: instructions loaded on demand (progressive disclosure)."""

    def add_skill(self, path: str | Path) -> None:
        """Register a SKILL.md file as a zero-argument tool that returns its instructions."""
        text = Path(path).read_text(encoding="utf-8")
        _, frontmatter, instructions = text.split("---", 2)
        meta = _parse_frontmatter(frontmatter)

        def activate() -> str:
            return instructions.strip()  # closure: remembers this skill's text

        self.add_tool(
            meta["name"],
            activate,
            f"Skill: {meta['description']}. Call it to receive step-by-step instructions.",
        )

    @property
    def prompt(self) -> str:
        return (
            "Some of your tools are skills: calling one returns detailed instructions "
            "for a type of task. Activate the matching skill before starting such a task.\n"
        )

