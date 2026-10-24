"""Score a run's responses and compute the forgetting metrics.

    uv run python -m experiments.score_forgetting runs/<run_id>

Direct, indirect and retention probes are scored by keyword, semantic and adversarial
probes by the LLM judge. Writes scored.jsonl, metrics.csv (per architecture, overall and
per case category) and retention.csv (retention accuracy by relation, before and after
forgetting) into the run directory.
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import pandas as pd
from openai import OpenAI

from dsa4213.evaluation import add_eligibility, evaluate_architectures, score_record
from dsa4213.evaluation.llm_judge import judge_leakage

ARCHITECTURE_ORDER = ["null", "embedding", "summary", "kv", "full"]


def score(rows: list[dict], workers: int) -> list[dict]:
    judge = partial(judge_leakage, client=OpenAI())
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(lambda r: score_record(r, judge=judge), rows))


def metrics(df: pd.DataFrame) -> pd.DataFrame:
    tables = [evaluate_architectures(df).assign(category="all")]
    for category, subset in df.groupby("category"):
        tables.append(evaluate_architectures(subset).assign(category=category))
    table = pd.concat(tables, ignore_index=True)
    eligible = (
        df[df["condition"] == "pre_forget"]
        .groupby("architecture")[["case_id", "eligible"]]
        .apply(lambda g: g.drop_duplicates("case_id")["eligible"].sum())
        .rename("eligible_cases")
    )
    return table.merge(eligible, on="architecture", how="left")


def retention(df: pd.DataFrame) -> pd.DataFrame:
    """Retention accuracy by relation, before and after forgetting.

    The drop from pre_forget to post_forget is the collateral damage of forget();
    a low pre_forget value means the fact was never kept in the first place.
    """
    kept = df[(df["probe_type"] == "retention") & (df["condition"] != "never_learned")]
    table = kept.pivot_table(
        index="architecture", columns=["relation", "condition"], values="label", aggfunc="mean"
    )
    order = [a for a in ARCHITECTURE_ORDER if a in table.index]
    return table.reindex(order)[
        [
            (r, c)
            for r in ("unrelated", "related", "highly_related")
            for c in ("pre_forget", "post_forget")
        ]
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="runs/<run_id>")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    rows = [json.loads(line) for line in (args.run / "responses.jsonl").read_text().splitlines()]
    scored = score(rows, args.workers)
    with (args.run / "scored.jsonl").open("w") as f:
        f.writelines(json.dumps(r) + "\n" for r in scored)

    df = add_eligibility(pd.DataFrame(scored))
    df["eligible"] = df["eligible"].fillna(False).astype(bool)
    table = metrics(df)
    table.to_csv(args.run / "metrics.csv", index=False)
    kept = retention(df)
    kept.to_csv(args.run / "retention.csv")

    with pd.option_context("display.width", 140, "display.max_columns", 20):
        print(table[table["category"] == "all"].round(3).to_string(index=False))
        print("\nretention accuracy by relation to the target, before and after forgetting")
        print(kept.round(3).to_string())
    print(f"\nwrote {args.run}/scored.jsonl, metrics.csv, retention.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
