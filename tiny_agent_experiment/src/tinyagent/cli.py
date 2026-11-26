import argparse
from pathlib import Path

from .agent import TinyAgent
from .code_tools import create_code_tools
from .display import Display
from .llm import LLM
from .memory import MultimodalMemory
from .planning import NativeReAct

HELP = "Commands: /tools  /clear  /help  /exit"


def main() -> None:
    parser = argparse.ArgumentParser(description="TinyAgent: a tiny coding agent")
    parser.add_argument("-m", "--model", default="gemma4:e4b")
    parser.add_argument("-ab", "--api-base", default="http://localhost:11434/v1")
    parser.add_argument("-ak", "--api-key", default="no_key")
    parser.add_argument("-w", "--workspace", default=".")
    args = parser.parse_args()

    agent = TinyAgent(
        LLM(args.model, base_url=args.api_base, api_key=args.api_key, think=True),
        MultimodalMemory(),
        create_code_tools(args.workspace),
        NativeReAct(max_steps=15),
        display=Display(),
    )
    print(
        f"TinyAgent · {args.model} · workspace: {Path(args.workspace).resolve()}"
    )
    print(HELP)

    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query in ("/exit", "exit", "quit"):
            break
        elif query == "/help":
            print(HELP)
        elif query == "/tools":
            print("\n".join(agent.tools.registry))
        elif query == "/clear":
            agent.memory.messages = agent.memory.messages[
                :1
            ]  # keep system prompt
            print("Memory cleared.")
        elif query:
            try:
                agent.run(query)
            except KeyboardInterrupt:
                print("\nInterrupted.")
            except Exception as e:  # keep the session alive
                print(f"ERROR: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
