"""Score a run's responses and compute the forgetting metrics.

    uv run python -m experiments.score_forgetting runs/<run_id>

Direct and retention probes are scored by an LLM recall judge, indirect probes by an LLM
choice judge (which option was picked), semantic and adversarial probes by the leakage
judge. The keyword label is kept beside each as keyword_label. Refuses a run whose
summary.json records any failed case, since its architectures would cover different cases.
Writes into the run directory:

    scored.jsonl     every response with its label (reused on reruns unless --rescore)
    metrics.csv      PRA, DLR, RA, LE, RE, RIR, SLR, ALR, MLS per architecture, overall and
                     per case category, with 95% bootstrap intervals over cases for "all"
                     and the number of resamples in which each metric was undefined
    retention.csv    retention accuracy by relation, before and after forgetting
    annotation.csv   a blind 25% sample of leakage-judged responses for two human raters,
                     keyed by a random id; annotation_key.csv maps it back to scored.jsonl
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
from dsa4213.evaluation.llm_judge import judge_choice, judge_leakage, judge_recall

ARCHITECTURE_ORDER = ["null", "embedding", "summary", "kv", "full"]
METRICS = ["PRA", "DLR", "RA", "LE", "RE", "RIR", "SLR", "ALR", "MLS"]


def score(rows: list[dict], workers: int) -> list[dict]:
    client = OpenAI(max_retries=8)
    judges = {
        "judge": partial(judge_leakage, client=client),
        "recall_judge": partial(judge_recall, client=client),
        "choice_judge": partial(judge_choice, client=client),
    }
    with ThreadPoolExecutor(workers) as pool:
        scored = list(pool.map(lambda r: score_record(r, **judges), rows))
    return [{"row_id": i, **r} for i, r in enumerate(scored)]


def influence(df: pd.DataFrame) -> pd.DataFrame:
    """Learning effect P(pre) - P(never) and residual effect P(post) - P(never) on indirect probes.

    RIR = RE / LE is unstable when LE is small, so both parts are reported beside it.
    """
    indirect = df[(df["probe_type"] == "indirect") & df["eligible"]]
    rates = indirect.pivot_table(
        index="architecture", columns="condition", values="label", aggfunc="mean"
    ).reindex(columns=["pre_forget", "post_forget", "never_learned"])
    return pd.DataFrame(
        {
            "LE": rates["pre_forget"] - rates["never_learned"],
            "RE": rates["post_forget"] - rates["never_learned"],
        }
    ).reset_index()


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    return evaluate_architectures(df).merge(influence(df), on="architecture", how="left")


def bootstrap(df: pd.DataFrame, reps: int, seed: int = 0) -> pd.DataFrame:
    """Percentile intervals from resampling whole cases, the unit of independence.

    Draws where a metric is undefined are counted in <metric>_undefined, not hidden.
    """
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
        draws.append(summarise(sample))
    draws = pd.concat(draws)
    grouped = draws.groupby("architecture")[METRICS]
    low = grouped.quantile(0.025).add_suffix("_lo")
    high = grouped.quantile(0.975).add_suffix("_hi")
    undefined = grouped.agg(lambda col: int(col.isna().sum())).add_suffix("_undefined")
    return low.join(high).join(undefined).reset_index()


def eligible_cases(df: pd.DataFrame) -> pd.Series:
    return df.drop_duplicates(["architecture", "case_id"]).groupby("architecture")["eligible"].sum()


def metrics(df: pd.DataFrame, reps: int) -> pd.DataFrame:
    overall = summarise(df).merge(bootstrap(df, reps), on="architecture")
    tables = [
        overall.assign(
            category="all", eligible_cases=overall["architecture"].map(eligible_cases(df))
        )
    ]
    for category, subset in df.groupby("category"):
        table = summarise(subset)
        tables.append(
            table.assign(
                category=category, eligible_cases=table["architecture"].map(eligible_cases(subset))
            )
        )
    return pd.concat(tables, ignore_index=True)


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
    columns = [
        (r, c)
        for r in ("unrelated", "related", "highly_related")
        for c in ("pre_forget", "post_forget")
        if (r, c) in table.columns
    ]
    return table.reindex(order)[columns]


def annotation_sample(
    df: pd.DataFrame, share: float = 0.25, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leakage-judged responses for human rating, stratified, blind to condition and score."""
    judged = df[df["probe_type"].isin(["semantic", "adversarial"]) & (df["architecture"] != "null")]
    sample = judged.groupby(["architecture", "condition", "probe_type"], group_keys=False).apply(
        lambda g: g.sample(frac=share, random_state=seed)
    )
    sample = sample.sample(frac=1, random_state=seed)
    rng = np.random.default_rng(seed)
    sample = sample.assign(item=rng.permutation(len(sample)) + 1).sort_values("item")
    sheet = sample[["item", "target_fact", "probe", "response"]].assign(rater_a="", rater_b="")
    return sheet, sample[["item", "row_id"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="runs/<run_id>")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--reps", type=int, default=1000, help="bootstrap resamples")
    parser.add_argument("--rescore", action="store_true", help="judge again even if scored")
    args = parser.parse_args()

    status = json.loads((args.run / "summary.json").read_text())["status"]
    if failed := {k: v for k, v in status.items() if v != "ok"}:
        print(f"refusing to score: {len(failed)} case runs did not finish", file=sys.stderr)
        for key, value in sorted(failed.items())[:10]:
            print(f"  {key}: {value[:120]}", file=sys.stderr)
        return 1

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
    sheet, key = annotation_sample(df)
    sheet.to_csv(args.run / "annotation.csv", index=False)
    key.to_csv(args.run / "annotation_key.csv", index=False)

    with pd.option_context("display.width", 160, "display.max_columns", 30):
        overall = table[table["category"] == "all"].set_index("architecture")
        overall = overall.reindex([a for a in ARCHITECTURE_ORDER if a in overall.index])
        for m in METRICS:
            overall[m] = overall.apply(
                lambda r, m=m: f"{r[m]:.2f} [{r[m + '_lo']:.2f}, {r[m + '_hi']:.2f}]", axis=1
            )
        print(overall[[*METRICS, "eligible_cases"]].to_string())
        compared = overall.drop(index="null", errors="ignore")
        undefined = compared[[f"{m}_undefined" for m in METRICS]].sum()
        if undefined.any():
            print("\nbootstrap draws where a metric was undefined:")
            print(undefined[undefined > 0].to_string())
        print("\nretention accuracy by relation to the target, before and after forgetting")
        print(kept.round(3).to_string())
    print(f"\nwrote scored.jsonl, metrics.csv, retention.csv, annotation*.csv in {args.run}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
