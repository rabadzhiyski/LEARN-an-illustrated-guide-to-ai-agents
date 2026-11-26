import re
import time
from dataclasses import dataclass
from typing import Callable

from .llm import LLM
from .trajectory import Step

# (prediction, example, steps) -> score in [0, 1]. A bool counts as 0 or 1.
Scorer = Callable[[str, dict, list[Step]], float]


@dataclass
class Benchmark:
    name: str
    examples: list[dict]
    scorer: Scorer


class Evaluator:
    """Run a fresh agent on every example of a Benchmark and aggregate the scores."""

    def __init__(self, create_agent: Callable, trials: int = 1) -> None:
        self.create_agent = create_agent
        self.trials = trials

    def run(self, benchmark: Benchmark) -> dict:
        results = []
        for example in benchmark.examples:
            for trial in range(self.trials):
                results.append(self._run_one(benchmark, example, trial))

        scores = [r["score"] for r in results]
        return {
            "name": benchmark.name,
            "score": sum(scores) / len(scores) if scores else 0.0,
            "avg_steps": sum(r["steps"] for r in results) / max(len(results), 1),
            "total_tokens": sum(r["tokens"] for r in results),
            "results": results,
        }

    def _run_one(self, benchmark: Benchmark, example: dict, trial: int) -> dict:
        agent = self.create_agent()  # fresh memory every time
        start = time.perf_counter()
        try:
            prediction = agent.run(example["task"]) or ""
            error = None
        except Exception as e:  # one broken example must not kill the run
            prediction, error = "", f"{type(e).__name__}: {e}"
        seconds = time.perf_counter() - start

        runs = agent.trajectory.runs
        steps = runs[-1]["steps"] if runs else []
        score = (
            0.0 if error else float(benchmark.scorer(prediction, example, steps))
        )

        return {
            "task": example["task"][:60],
            "trial": trial,
            "prediction": prediction,
            "score": score,
            "error": error,
            "steps": len(steps),
            "tools": [s.action["tool"] for s in steps if s.action],
            "tokens": sum(
                s.metadata["prompt_tokens"] + s.metadata["completion_tokens"]
                for s in steps
                if s.metadata
            ),
            "seconds": round(seconds, 1),
        }


# --- Scorers ---------------------------------------------------------------


def exact_choice(prediction: str, example: dict, steps: list[Step]) -> bool:
    """Multiple choice: the first standalone letter A–J must equal `expected`."""
    match = re.search(r"\b([A-J])\b", prediction.upper())
    return match is not None and match.group(1) == example["expected"]


def programmatic(prediction: str, example: dict, steps: list[Step]) -> bool:
    """Each example carries its own `check(prediction) -> bool`."""
    return example["check"](prediction)


def used_tools(prediction: str, example: dict, steps: list[Step]) -> float:
    """Trajectory check: answer contains `expected` AND every `tools` was called."""
    called = {s.action["tool"] for s in steps if s.action}
    right_tools = set(example["tools"]) <= called
    right_answer = example["expected"] in prediction
    return (right_tools + right_answer) / 2


def make_judge(judge: LLM) -> Scorer:
    """LLM-as-a-judge: a scorer that asks another model for a 0–1 grade."""

    def judge_scorer(prediction: str, example: dict, steps: list[Step]) -> float:
        prompt = f"""Grade the RESPONSE against the REFERENCE answer.
Reply 1 if it is correct and consistent with the reference, 0 if it is wrong.
Ignore style and length. Reply with a single digit only.

QUESTION: {example["task"]}
REFERENCE: {example["expected"]}
RESPONSE: {prediction}"""
        reply = judge.generate([{"role": "user", "content": prompt}]).content
        match = re.search(r"[01](?:\.\d+)?", reply)
        return min(max(float(match.group()), 0.0), 1.0) if match else 0.0

    return judge_scorer
