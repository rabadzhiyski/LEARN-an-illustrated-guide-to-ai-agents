from .llm import LLM, EmbeddingModel


class Memory:
    """Full conversation history, sent in its entirety on every call."""

    def __init__(self) -> None:
        self.messages: list[dict] = []

    def add(
        self, role: str, content: str, tool_call: dict | None = None, **kwargs
    ) -> None:
        """Add a message to memory."""
        message = {"role": role, "content": content}
        if tool_call:
            message["tool_calls"] = [tool_call]  # used from Ch5 on
        self.messages.append(message)

    def get_messages(self) -> list[dict]:
        """Return the messages to send to the LLM."""
        return self.messages


class TrimmingMemory(Memory):
    """Keep the system prompt plus only the last `max_messages` messages."""

    def __init__(self, max_messages: int = 4) -> None:
        super().__init__()
        self.max_messages = max_messages

    def add(self, role: str, content: str, **kwargs) -> None:
        super().add(role, content, **kwargs)
        system = [m for m in self.messages if m["role"] == "system"]
        turns = [m for m in self.messages if m["role"] != "system"]
        self.messages = system + turns[-self.max_messages:]


class SummarizationMemory(Memory):
    """Compress the conversation into a running summary after each assistant turn."""

    def __init__(self, llm: LLM) -> None:
        super().__init__()
        self.llm = llm

    def add(self, role: str, content: str, **kwargs) -> None:
        super().add(role, content, **kwargs)
        if role != "assistant":
            return  # only summarize once a user/assistant turn is complete

        summary, conversation = "", ""
        for m in self.messages:
            if m["role"] == "system":
                summary = m["content"]
            else:
                conversation += f"{m['role']}: {m['content']}\n"

        prompt = (
            "Update the summary with the new conversation.\n\n"
            f"Summary: {summary}\n\n"
            f"Conversation:\n{conversation}\n"
            "Output the updated summary only."
        )
        response = self.llm.generate([{"role": "user", "content": prompt}])
        self.messages = [{"role": "system", "content": response.content}]


class LongTermMemory(Memory):
    """RAG: augment each user message with the most relevant documents."""

    def __init__(
        self, embedding_model: EmbeddingModel, documents: list[str]
    ) -> None:
        super().__init__()
        self.embedding_model = embedding_model
        self.documents = documents
        self.embeddings = [embedding_model.embed(doc) for doc in documents]

    def add(self, role: str, content: str, **kwargs) -> None:
        if role == "user":
            context = "\n".join(self.search(content))
            content = f"Context:\n{context}\n\nQuestion: {content}"
        super().add(role, content, **kwargs)

    def search(self, query: str, k: int = 3) -> list[str]:
        """Return the k documents most similar to the query."""
        query_embedding = self.embedding_model.embed(query)
        scores = [self._cosine(query_embedding, e) for e in self.embeddings]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.documents[i] for i in ranked[:k]]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm = sum(x * x for x in a) ** 0.5 * sum(y * y for y in b) ** 0.5
        return dot / norm
