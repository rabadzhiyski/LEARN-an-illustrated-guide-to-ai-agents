import json
import subprocess
import sys
from pathlib import Path

SERVER = Path(__file__).parent.parent / "mcp_servers" / "notes_server.py"


def main() -> None:
    # Launch the server; we talk to it through its stdin/stdout pipes
    server = subprocess.Popen(
        [sys.executable, str(SERVER)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )

    def send(message: dict) -> None:
        line = json.dumps(message)
        print(f"\n→ {line}")
        server.stdin.write(line + "\n")  # one JSON message per line
        server.stdin.flush()

    def receive() -> dict:
        message = json.loads(server.stdout.readline())
        print(f"← {json.dumps(message, indent=2)}")
        return message

    # 1. Handshake: agree on protocol version, exchange capabilities
    send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2025-11-25",
        "capabilities": {},
        "clientInfo": {"name": "raw-client", "version": "0.1"},
    }})
    receive()
    send({"jsonrpc": "2.0", "method": "notifications/initialized"})  # notification: no id, no reply

    # 2. Discovery
    send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    receive()

    # 3. Execution
    send({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": "save_note", "arguments": {"text": "MCP is JSON-RPC over a pipe"}}})
    receive()

    # 4. Two kinds of failure
    send({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
        "params": {"name": "delete_everything", "arguments": {}}})
    receive()
    send({"jsonrpc": "2.0", "id": 5, "method": "does/not/exist"})
    receive()

    server.stdin.close()  # closing stdin = "goodbye" for stdio servers
    server.wait()


if __name__ == "__main__":
    main()
