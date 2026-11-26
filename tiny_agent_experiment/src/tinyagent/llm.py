import json
import urllib.request
from dataclasses import dataclass


@dataclass
class Response:
    """Structured response from LLM calls."""

    content: str = ""
    reasoning: str | None = None
    tool_call: dict | None = None
    metadata: dict | None = None


class LLM:
    """Minimal client for any OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "no_key",
        think: bool = False,
        temperature: float | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.think = think
        self.temperature = temperature

    def generate(
        self, messages: list[dict], tools: list | None = None
    ) -> Response:
        """Generate a response from the LLM given a list of messages."""
        # Build the request body
        body = {"model": self.model, "messages": messages}

        # Optional parts: tools (Ch5), reasoning (Ch3), sampling
        if tools:
            body["tools"] = tools
        body["reasoning_effort"] = "medium" if self.think else "none"
        if self.temperature is not None:
            body["temperature"] = self.temperature

        # POST to the OpenAI-compatible /chat/completions endpoint
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(request) as http_response:
            data = json.loads(http_response.read())

        # Normalize the provider's JSON into our Response
        message = data["choices"][0]["message"]
        reasoning = message.get("reasoning") or message.get("reasoning_content")
        tool_calls = message.get("tool_calls")
        metadata = {
            "model": data["model"],
            "prompt_tokens": data["usage"]["prompt_tokens"],
            "completion_tokens": data["usage"]["completion_tokens"],
        }

        return Response(
            content=message.get("content") or "",
            reasoning=reasoning,
            tool_call=tool_calls[0] if tool_calls else None,
            metadata=metadata,
        )


class EmbeddingModel:
    """Turn text into a vector via an OpenAI-compatible /embeddings endpoint."""

    def __init__(
        self, model: str, base_url: str = "http://localhost:11434/v1"
    ) -> None:
        self.model = model
        self.base_url = base_url

    def embed(self, text: str) -> list[float]:
        """Convert text into a numerical vector."""
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps({"model": self.model, "input": text}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request) as http_response:
            data = json.loads(http_response.read())
        return data["data"][0]["embedding"]
