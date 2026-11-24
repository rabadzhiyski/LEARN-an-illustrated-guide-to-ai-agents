from tinyagent import TinyAgent


def main() -> None:
    agent = TinyAgent()
    print(agent.run("What's the weather in Sofia?"))


if __name__ == "__main__":
    main()
