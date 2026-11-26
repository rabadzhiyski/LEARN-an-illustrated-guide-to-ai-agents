import datetime
import sys

from tinyagent import (
    LLM,
    Benchmark,
    Evaluator,
    Memory,
    NativeReAct,
    NativeTools,
    ReAct,
    TinyAgent,
    Tools,
    exact_choice,
    make_judge,
    programmatic,
    toolbox,
    used_tools,
)

FUNCTIONS = [toolbox.add, toolbox.multiply, toolbox.today, toolbox.days_between]
LETTER = "\nAnswer with only the letter."


def native_agent() -> TinyAgent:
    tools = NativeTools()
    for func in FUNCTIONS:
        tools.add_tool(func.__name__, func)
    return TinyAgent(LLM("gemma4:e4b", think=True), Memory(), tools, NativeReAct())


def text_agent() -> TinyAgent:
    tools = Tools()
    for func in FUNCTIONS:
        tools.add_tool(func.__name__, func)
    return TinyAgent(LLM("gemma4:e4b"), Memory(), tools, ReAct())


# Adjacent string literals are joined by Python: ("a" "b") == "ab"
MMLU = Benchmark(
    "MMLU-Pro (exact choice)",
    [
        {
            "task": (
                "Which body cavity contains the pituitary gland?\n"
                "A) Ventral B) Dorsal C) Buccal D) Thoracic E) Pericardial "
                "F) Abdominal G) Spinal H) Pelvic I) Pleural J) Cranial" + LETTER
            ),
            "expected": "J",
        },
        {
            "task": (
                "What is the approximate mean cranial capacity of Homo erectus?\n"
                "A) 1200 cc B) under 650 cc C) 1700 cc D) 1350 cc "
                "E) just under 1000 cc F) 1500 cc G) under 500 cc "
                "H) about 800 cc I) just over 1100 cc J) about 900 cc" + LETTER
            ),
            "expected": "E",
        },
        {
            "task": (
                "According to Moore's 'ideal utilitarianism', the right action "
                "brings about the greatest amount of:\n"
                "A) wealth B) virtue C) fairness D) pleasure E) peace "
                "F) justice G) happiness H) power I) good J) knowledge" + LETTER
            ),
            "expected": "I",
        },
    ],
    exact_choice,
)

IFEVAL = Benchmark(
    "IFEval (programmatic)",
    [
        {
            "task": "Write a funny song in fewer than 10 sentences proposing "
            "a new playground at my local school.",
            "check": lambda t: sum(c in ".!?" for c in t) < 10,
        },
        {
            "task": "Write ad copy for a digital photo frame that shows your "
            "social media photos. At most 150 words.",
            "check": lambda t: len(t.split()) <= 150,
        },
        {
            "task": "Write a Shakespearean-style itinerary for a trip to Japan. "
            "Do not use any commas.",
            "check": lambda t: "," not in t,
        },
    ],
    programmatic,
)

XMAS = (datetime.date(2026, 12, 25) - datetime.date.today()).days
TOOL_USE = Benchmark(
    "Tool use (answer + trajectory)",
    [
        {
            "task": "How many days are there from 2026-01-01 to 2026-03-01?",
            "expected": "59",
            "tools": ["days_between"],
        },
        {
            "task": "What is 17.5 multiplied by 4.2?",
            "expected": "73.5",
            "tools": ["multiply"],
        },
        {
            "task": "How many days from today until 2026-12-25?",
            "expected": str(XMAS),
            "tools": ["today", "days_between"],
        },
    ],
    used_tools,
)

JUDGED = Benchmark(
    "Open QA (LLM judge)",
    [
        {
            "task": "Which body cavity contains the pituitary gland?",
            "expected": "the cranial cavity",
        },
        {
            "task": "What is the approximate mean cranial capacity of Homo erectus?",
            "expected": "just under 1000 cc",
        },
        {
            "task": "According to Moore's 'ideal utilitarianism', the right action "
            "brings about the greatest amount of what?",
            "expected": "good",
        },
    ],
    make_judge(LLM("gemma4:e4b")),
)

# Hand-labelled (prediction, human score) pairs: the judge's own test set
JUDGE_CHECK = [
    ("About 1200 cc.", 0),
    ("Roughly 900-1000 cc, just under 1000 cc on average.", 1),
    ("Around 850 cc.", 0),
    ("Approximately 950 cc.", 1),
]


def report(result: dict) -> None:
    print(
        f"\n{result['name']}: score={result['score']:.2f}  "
        f"avg_steps={result['avg_steps']:.1f}  tokens={result['total_tokens']}"
    )
    for r in result["results"]:
        if r["score"] in (0, 1):
            flag = "PASS" if r["score"] == 1 else "FAIL"
        else:
            flag = f"{r['score']:.2f}"
        print(
            f"  [{flag}] {r['task']!r} t{r['trial']}  {r['steps']} steps "
            f"{r['tokens']} tok {r['seconds']}s  tools={r['tools']}"
        )
        print(f"         → {r['prediction'][:90]!r}")
        if r["error"]:
            print(f"         ERROR {r['error']}")


def judge_check() -> None:
    """Meta-evaluation: how often does each judge agree with a human?"""
    example = JUDGED.examples[1]
    for label, judge in [
        ("qwen3.5:4b", LLM("qwen3.5:4b")),
        ("qwen3.5:4b think", LLM("qwen3.5:4b", think=True)),
        ("gemma4:e4b", LLM("gemma4:e4b")),
    ]:
        scorer = make_judge(judge)
        scores = [scorer(p, example, []) for p, _ in JUDGE_CHECK]
        agree = sum(s == human for s, (_, human) in zip(scores, JUDGE_CHECK))
        print(f"{label:<18} scores={scores}  agreement={agree}/{len(JUDGE_CHECK)}")


def compare() -> None:
    """Same benchmark, two agent designs, 2 trials each: measure, don't eyeball."""
    for label, factory in [("text ReAct", text_agent), ("native ReAct", native_agent)]:
        result = Evaluator(factory, trials=2).run(TOOL_USE)
        result["name"] = f"{TOOL_USE.name} — {label}"
        report(result)


DEMOS = {
    "mmlu": lambda: report(Evaluator(native_agent).run(MMLU)),
    "ifeval": lambda: report(Evaluator(native_agent).run(IFEVAL)),
    "tools": lambda: report(Evaluator(native_agent).run(TOOL_USE)),
    "judge": lambda: report(Evaluator(native_agent).run(JUDGED)),
    "judge-check": judge_check,
    "compare": compare,
}

if __name__ == "__main__":
    # python examples/lesson06.py mmlu  (or: ifeval, tools, judge, judge-check,
    # compare, all)
    choice = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, demo in DEMOS.items():
        if choice in (name, "all"):
            demo()
