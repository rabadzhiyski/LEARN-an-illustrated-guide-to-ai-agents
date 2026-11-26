BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
COLORS = {
    "THOUGHT": "\033[32m",  # green
    "ACTION": "\033[31m",  # red
    "OBSERVATION": "\033[33m",  # yellow
    "ANSWER": "\033[35m",  # purple
}
LIMITS = {"THOUGHT": 800, "ACTION": 3000, "OBSERVATION": 800}  # ANSWER: no limit


class Display:
    """Print each agent event as a coloured block in the terminal."""

    def __call__(self, event: str, data: str | dict | None = None) -> None:
        if event == "thinking":
            print(f"{DIM}  thinking...{RESET}")
            return
        if isinstance(data, dict) and isinstance(data.get("kwargs"), dict):
            # a parsed tool call: one argument per line, so code stays readable
            args = "\n".join(f"  {k} = {v}" for k, v in data["kwargs"].items())
            text = f"{data['tool']}(\n{args}\n)"
        else:
            text = str(data)
        limit = LIMITS.get(event)
        if limit and len(text) > limit:
            text = text[:limit] + f"{DIM} ... (+{len(text) - limit} chars){RESET}"
        print(f"{BOLD}{COLORS.get(event, '')}▒▒ {event} ▒▒{RESET}\n{text}\n")
