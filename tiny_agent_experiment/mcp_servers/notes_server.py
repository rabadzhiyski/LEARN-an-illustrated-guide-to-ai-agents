import json
from pathlib import Path

from mcp.server.mcpserver import MCPServer

NOTES_FILE = Path(__file__).parent / "notes.json"

server = MCPServer("notes")


def _load() -> list[str]:
    return json.loads(NOTES_FILE.read_text()) if NOTES_FILE.exists() else []


@server.tool()
def save_note(text: str) -> str:
    """Save a short note for later."""
    notes = _load()
    notes.append(text)
    NOTES_FILE.write_text(json.dumps(notes, indent=2))
    return f"Saved note #{len(notes)}."


@server.tool()
def list_notes() -> str:
    """List all saved notes."""
    notes = _load()
    return "\n".join(f"{i}. {n}" for i, n in enumerate(notes, 1)) or "No notes yet."


if __name__ == "__main__":
    server.run(transport="stdio")
