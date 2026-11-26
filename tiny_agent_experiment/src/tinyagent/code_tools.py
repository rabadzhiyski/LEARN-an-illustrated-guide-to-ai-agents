import subprocess
import sys
from pathlib import Path

from .tools import NativeTools

MAX_OUTPUT = 4000  # characters; keep tool output from flooding the context


def _clip(text: str) -> str:
    if len(text) <= MAX_OUTPUT:
        return text
    return text[:MAX_OUTPUT] + f"\n... ({len(text) - MAX_OUTPUT} more characters)"


def create_code_tools(workspace: str | Path = ".") -> NativeTools:
    """File and Python tools confined to `workspace`. Writes and runs need approval."""
    root = Path(workspace).resolve()

    def safe_path(path: str) -> Path:
        """Resolve `path` inside the workspace; refuse anything outside it."""
        resolved = (root / path).resolve()
        if not resolved.is_relative_to(root):
            raise PermissionError(f"'{path}' is outside the workspace")
        return resolved

    def list_files(directory: str = ".") -> str:
        """List all files under a directory of the workspace (recursive)."""
        base = safe_path(directory)
        files = sorted(
            str(p.relative_to(root))
            for p in base.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        )
        return "\n".join(files[:200]) or "(no files)"

    def read_file(path: str) -> str:
        """Read a text file, with line numbers."""
        lines = safe_path(path).read_text(encoding="utf-8").splitlines()
        return _clip(
            "\n".join(f"{i:>4}| {line}" for i, line in enumerate(lines, 1))
        )

    def write_file(path: str, content: str) -> str:
        """Create or overwrite a file with the full new content."""
        target = safe_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} characters to '{path}'."

    def run_python(path: str) -> str:
        """Run a Python file in the workspace and return its exit code and output."""
        result = subprocess.run(
            [sys.executable, str(safe_path(path))],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=root,
        )
        output = (result.stdout + result.stderr).strip() or "(no output)"
        return _clip(f"exit code {result.returncode}\n{output}")

    tools = NativeTools(requires_approval=["write_file", "run_python"])
    for func in (list_files, read_file, write_file, run_python):
        tools.add_tool(func.__name__, func)
    return tools
