"""Measure how much of each memory one forget request removed, from a run's saved stores.

    uv run python -m experiments.forget_reach results/stores results/fact-level/stores

For every memory in each stores folder, over all cases:

    records           records stored before forgetting (records.jsonl lines or profile.json keys)
    deleted           records the forget request deleted
    redacted          records it rewrote, counting only rewrites that changed the text
    text_removed      share of all stored text (dump.txt, summed over cases) that forgetting removed
    facts_per_turn    facts extracted per user turn when writing, where write_log.jsonl records them
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def record_count(store: Path) -> int | None:
    if (store / "records.jsonl").exists():
        return len(read_jsonl(store / "records.jsonl"))
    if (store / "profile.json").exists():
        return len(json.loads((store / "profile.json").read_text()))
    return None


def forget_counts(log: list[dict]) -> tuple[int, int]:
    deleted = redacted = 0
    for entry in log:
        for step in entry.get("passes", [entry]):
            deleted += len(step.get("deleted", step.get("removed", [])))
            redacted += sum(r["before"] != r["after"] for r in step.get("redacted", []))
    return deleted, redacted


def facts_per_turn(log: list[dict]) -> float | None:
    turns = [entry for entry in log if "facts" in entry]
    return sum(len(t["facts"]) for t in turns) / len(turns) if turns else None


def reach(stores: Path) -> pd.DataFrame:
    cases = []
    for memory_dir in sorted(stores.glob("*/*")):
        pre, post = memory_dir / "pre_forget", memory_dir / "post_forget"
        deleted, redacted = forget_counts(read_jsonl(post / "forget_log.jsonl"))
        cases.append(
            {
                "memory": memory_dir.name,
                "records": record_count(pre),
                "deleted": deleted,
                "redacted": redacted,
                "chars_before": len((pre / "dump.txt").read_text()),
                "chars_after": len((post / "dump.txt").read_text()),
                "facts_per_turn": facts_per_turn(read_jsonl(pre / "write_log.jsonl")),
            }
        )
    df = pd.DataFrame(cases).astype({"records": float, "facts_per_turn": float})
    table = df.groupby("memory").agg(
        cases=("memory", "size"),
        records=("records", "mean"),
        deleted=("deleted", "mean"),
        redacted=("redacted", "mean"),
        chars_before=("chars_before", "sum"),
        chars_after=("chars_after", "sum"),
        facts_per_turn=("facts_per_turn", "mean"),
    )
    removed = 1 - table["chars_after"] / table["chars_before"].where(table["chars_before"] > 0)
    table = table.assign(text_removed=removed).drop(columns=["chars_before", "chars_after"])
    return table.reset_index().assign(stores=str(stores))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("stores", type=Path, nargs="+", help="runs/<run_id>/stores")
    parser.add_argument("--out", type=Path, help="also write the table as csv")
    args = parser.parse_args()

    table = pd.concat([reach(s) for s in args.stores], ignore_index=True)
    columns = [
        "stores",
        "memory",
        "cases",
        "records",
        "deleted",
        "redacted",
        "text_removed",
        "facts_per_turn",
    ]
    table = table[columns].round(3)
    print(table.to_string(index=False))
    if args.out:
        table.to_csv(args.out, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
