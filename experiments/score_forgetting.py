"""Score a run's responses and compute the forgetting metrics.

    uv run python -m experiments.score_forgetting runs/<run_id>

Direct, indirect and retention probes are scored by keyword, semantic and adversarial
probes by the LLM judge. Writes into the run directory:

    scored.jsonl     every response with its label (reused on reruns unless --rescore)
    metrics.csv      PRA, DLR, RA, RIR, SLR, ALR, MLS per architecture, overall and per
                     case category, with 95% bootstrap intervals over cases for "all"
    retention.csv    retention accuracy by relation, before and after forgetting
    annotation.csv   a blind 25% sample of judge-scored responses for two human raters
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI

from dsa4213.evaluation import add_eligibility, evaluate_architectures, score_record
from dsa4213.evaluation.llm_judge import judge_leakage

ARCHITECTURE_ORDER = ["null", "embedding", "summary", "kv", "full"]
METRICS = ["PRA", "DLR", "RA", "RIR", "SLR", "ALR", "MLS"]


def score(rows: list[dict], workers: int) -> list[dict]:
    judge = partial(judge_leakage, client=OpenAI())
    with ThreadPoolExecutor(workers) as pool:
        scored = list(pool.map(lambda r: score_record(r, judge=judge), rows))
    return [{"row_id": i, **r} for i, r in enumerate(scored)]


def bootstrap(df: pd.DataFrame, reps: int, seed: int = 0) -> pd.DataFrame:
    """Percentile intervals from resampling whole cases, the unit of independence."""
    rng = np.random.default_rng(seed)
    by_case = {cid: g for cid, g in df.groupby("case_id")}
    ids = list(by_case)
    draws = []
    for _ in range(reps):
        picked = rng.choice(ids, size=len(ids), replace=True)
        sample = pd.concat(
            [by_case[c].assign(case_id=f"{c}#{i}") for i, c in enumerate(picked)],
            ignore_index=True,
        )
        draws.append(evaluate_architectures(sample))
    draws = pd.concat(draws)
    low = draws.groupby("architecture")[METRICS].quantile(0.025).add_suffix("_lo")
    high = draws.groupby("architecture")[METRICS].quantile(0.975).add_suffix("_hi")
    return low.join(high).reset_index()


def metrics(df: pd.DataFrame, reps: int) -> pd.DataFrame:
    overall = evaluate_architectures(df).merge(bootstrap(df, reps), on="architecture")
    tables = [overall.assign(category="all")]
    for category, subset in df.groupby("category"):
        tables.append(evaluate_architectures(subset).assign(category=category))
    table = pd.concat(tables, ignore_index=True)
    eligible = (
        df[df["condition"] == "pre_forget"]
        .drop_duplicates(["architecture", "case_id"])
        .groupby("architecture")["eligible"]
        .sum()
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


def annotation_sample(df: pd.DataFrame, share: float = 0.25, seed: int = 0) -> pd.DataFrame:
    """Judge-scored responses for human rating, stratified, without the judge's score."""
    judged = df[df["probe_type"].isin(["semantic", "adversarial"]) & (df["architecture"] != "null")]
    sample = judged.groupby(["architecture", "condition", "probe_type"], group_keys=False).apply(
        lambda g: g.sample(frac=share, random_state=seed)
    )
    sheet = sample[["row_id", "target_fact", "probe_type", "probe", "response"]]
    return sheet.sample(frac=1, random_state=seed).assign(rater_a="", rater_b="")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="runs/<run_id>")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--reps", type=int, default=1000, help="bootstrap resamples")
    parser.add_argument("--rescore", action="store_true", help="judge again even if scored")
    args = parser.parse_args()

    scored_path = args.run / "scored.jsonl"
    if scored_path.exists() and not args.rescore:
        scored = [json.loads(line) for line in scored_path.read_text().splitlines()]
    else:
        lines = (args.run / "responses.jsonl").read_text().splitlines()
        scored = score([json.loads(line) for line in lines], args.workers)
        with scored_path.open("w") as f:
            f.writelines(json.dumps(r) + "\n" for r in scored)

    df = add_eligibility(pd.DataFrame(scored))
    df["eligible"] = df["eligible"].fillna(False).astype(bool)
    table = metrics(df, args.reps)
    table.to_csv(args.run / "metrics.csv", index=False)
    kept = retention(df)
    kept.to_csv(args.run / "retention.csv")
    annotation_sample(df).to_csv(args.run / "annotation.csv", index=False)

    with pd.option_context("display.width", 160, "display.max_columns", 30):
        overall = table[table["category"] == "all"].set_index("architecture")
        overall = overall.reindex([a for a in ARCHITECTURE_ORDER if a in overall.index])
        for m in METRICS:
            overall[m] = overall.apply(
                lambda r, m=m: f"{r[m]:.2f} [{r[m + '_lo']:.2f}, {r[m + '_hi']:.2f}]", axis=1
            )
        print(overall[[*METRICS, "eligible_cases"]].to_string())
        print("\nretention accuracy by relation to the target, before and after forgetting")
        print(kept.round(3).to_string())
    print(f"\nwrote scored.jsonl, metrics.csv, retention.csv, annotation.csv in {args.run}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
