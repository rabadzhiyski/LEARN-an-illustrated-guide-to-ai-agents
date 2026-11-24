from tinyagent import LLM, TinyAgent

MODEL = "gemma4:e4b"
QUESTION = "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much is the ball?"


def main() -> None:
    # 1. Raw LLM call, no reasoning
    llm = LLM(model=MODEL)
    response = llm.generate([{"role": "user", "content": QUESTION}])
    print("NO THINK →", response.content)
    print("metadata:", response.metadata, "\n")

    # 2. Same question with reasoning on (Ch3)
    thinker = LLM(model=MODEL, think=True)
    response = thinker.generate([{"role": "user", "content": QUESTION}])
    print("REASONING →", (response.reasoning or "")[:300], "...\n")
    print("THINK →", response.content)
    print("metadata:", response.metadata, "\n")

    # 3. Through the agent, then inspect the trajectory
    agent = TinyAgent(llm=thinker)
    agent.run("What is the capital of Bulgaria? One word.")
    agent.run("And of Japan? One word.")  # notice it still works (no context needed) — try a follow-up that needs context!
    for run in agent.trajectory.runs:
        print(run["query"], "→", run["steps"])


if __name__ == "__main__":
    main()
