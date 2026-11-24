from tinyagent import (
    LLM,
    EmbeddingModel,
    LongTermMemory,
    Memory,
    SummarizationMemory,
    TinyAgent,
    TrimmingMemory,
)

llm = LLM(model="gemma4:e4b")


def chat(agent: TinyAgent, questions: list[str]) -> None:
    for q in questions:
        print(f"  USER: {q}\n  AGENT: {agent.run(q)}")
    print(f"  → {len(agent.memory.get_messages())} messages in memory\n")


def main() -> None:
    questions = [
        "My name is Plamen and I live in Sofia.",
        "What is the capital of Japan? One word.",
        "And of France? One word.",
        "What's my name and where do I live?",
    ]

    print("1. Full memory: remembers everything")
    chat(TinyAgent(llm, Memory()), questions)

    print("2. Trimming (last 4 messages): forgets your name")
    chat(TinyAgent(llm, TrimmingMemory(max_messages=4)), questions)

    print("3. Summarization: remembers the gist in one message")
    agent = TinyAgent(llm, SummarizationMemory(llm))
    chat(agent, questions)
    print("  SUMMARY:", agent.memory.get_messages()[0]["content"], "\n")

    print("4. Long-term memory (RAG): knows private facts")
    documents = [
        "Acme Corp's VPN password rotates every 30 days.",
        "Acme Corp's office cafeteria serves banitsa on Fridays.",
        "Acme Corp's CEO is Maria Petrova, appointed in 2024.",
        "The Acme Corp Databricks workspace is in the eu-west-1 region.",
        "Acme Corp employees get 25 days of paid leave.",
    ]
    rag = LongTermMemory(EmbeddingModel("nomic-embed-text:latest"), documents)
    print("  retrieved:", rag.search("Who runs Acme?", k=2))
    chat(TinyAgent(llm, rag), ["Who runs Acme and since when?"])


if __name__ == "__main__":
    main()
