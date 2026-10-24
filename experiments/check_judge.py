"""Check the leakage judge against labelled answers before trusting its rates.

    uv run python -m experiments.check_judge --model gpt-4o

judge_check.jsonl holds 60 semantic and adversarial answers from a real run, labelled
leak or no-leak by a reader who did not see the judge's scores. Exits 1 if the judge's
Cohen's kappa against those labels is below --min-kappa. This is a pre-check; the
proposal's two human raters (annotation.csv from score_forgetting) are the validation.
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI
from sklearn.metrics import cohen_kappa_score

from dsa4213.evaluation.llm_judge import JUDGE_MODEL, judge_leakage

CHECK_SET = Path(__file__).parent / "judge_check.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=JUDGE_MODEL)
    parser.add_argument("--min-kappa", type=float, default=0.7)
    args = parser.parse_args()

    items = [json.loads(line) for line in CHECK_SET.read_text().splitlines()]
    client = OpenAI(max_retries=8)
    with ThreadPoolExecutor(20) as pool:
        judged = list(
            pool.map(lambda i: judge_leakage(i, client=client, model=args.model)["label"], items)
        )
    gold = [i["leak"] for i in items]
    kappa = cohen_kappa_score(gold, judged)
    false_leaks = sum(j and not g for j, g in zip(judged, gold, strict=True))
    missed = sum(g and not j for j, g in zip(judged, gold, strict=True))
    print(
        f"{args.model}: kappa {kappa:.2f}, {false_leaks} false leaks, {missed} missed leaks, "
        f"judge leak rate {sum(judged) / len(items):.2f} vs labelled {sum(gold) / len(items):.2f}"
    )
    return 0 if kappa >= args.min_kappa else 1


if __name__ == "__main__":
    sys.exit(main())
