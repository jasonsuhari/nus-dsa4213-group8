"""Small opt-in live pilot. See experiments/SUMMARY_PILOT.md for semantic review."""

import argparse
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from difflib import unified_diff
from pathlib import Path

from openai import OpenAI
from openai import __version__ as openai_version

from dsa4213.agent import DEFAULT_MODEL
from dsa4213.memory.base import DEFAULT_MAX_CHARS, Session, Turn
from dsa4213.memory.summary import FORGET_PROMPT, WRITE_PROMPT, SummaryMemory
from dsa4213.runlog import log_run


@dataclass(frozen=True)
class Case:
    case_id: str
    target: str
    retained_facts: tuple[str, ...]
    sessions: tuple[Session, ...]
    # Remove whole explicitly targeted turns, never substring-edit retained clues.
    omitted_turns: tuple[tuple[str, int], ...]
    notes: str = ""

    def never_learned_sessions(self) -> tuple[Session, ...]:
        omitted = set(self.omitted_turns)
        return tuple(
            replace(
                s,
                turns=tuple(
                    turn for i, turn in enumerate(s.turns) if (s.session_id, i) not in omitted
                ),
            )
            for s in self.sessions
        )

    def planned_requests(self) -> int:
        return sum(len(s.turns) for s in self.sessions + self.never_learned_sessions()) + 1


def conversation(session_id: str, date: str, *statements: str) -> Session:
    return Session(session_id, date, tuple(Turn("user", text) for text in statements))


CASES = (
    Case(
        "01_simple",
        "The user is vegetarian.",
        ("The user likes Japanese food.", "The user studies at NUS."),
        (
            conversation(
                "simple",
                "2026-10-01",
                "I am vegetarian.",
                "I like Japanese food.",
                "I study at NUS.",
            ),
        ),
        (("simple", 0),),
    ),
    Case(
        "02_related",
        "The user is vegetarian.",
        ("The user enjoys tofu.", "The user avoids meat.", "The user studies at NUS."),
        (
            conversation(
                "related",
                "2026-10-02",
                "I am vegetarian.",
                "I enjoy tofu.",
                "I avoid meat.",
                "I study at NUS.",
            ),
        ),
        (("related", 0),),
        "Retain independently stated food preferences and meat avoidance. Review explicit "
        "vegetarian identity separately from these clues, even when inference is plausible.",
    ),
    Case(
        "03_paraphrase",
        "The user grew up in Malaysia.",
        ("The user enjoys hiking.", "The user studies at NUS."),
        (
            conversation(
                "paraphrase",
                "2026-10-03",
                "I spent my childhood in Malaysia.",
                "I enjoy hiking.",
                "I study at NUS.",
            ),
        ),
        (("paraphrase", 0),),
        "Childhood in Malaysia is a direct semantic equivalent of the target. "
        "Do not rely on the literal phrase 'grew up'.",
    ),
    Case(
        "04_temporal",
        "The user previously lived in Singapore in 2023.",
        ("The user moved to Tokyo in 2025.", "The user works as a software engineer."),
        (
            conversation(
                "temporal",
                "2026-10-04",
                "I lived in Singapore in 2023.",
                "I moved to Tokyo in 2025.",
                "I work as a software engineer.",
            ),
        ),
        (("temporal", 0),),
        "The conversation date is 2026; the events occurred in 2023 and 2025. "
        "Retaining Tokyo without its 2025 date does not fully preserve that fact.",
    ),
    Case(
        "05_recursive",
        "The user grew up in Malaysia.",
        (
            "The user studies at NUS.",
            "The user's cat is named Miso.",
            "The user enjoys Japanese food.",
            "The user cycles on Saturdays.",
            "The user moved to Tokyo in 2025.",
            "The user's project deadline is 30 October 2026.",
        ),
        (
            conversation(
                "long-1",
                "2026-10-01",
                "I study at NUS.",
                "My cat is named Miso.",
                "I grew up in Malaysia.",
            ),
            conversation("long-2", "2026-10-05", "I enjoy Japanese food.", "I cycle on Sundays."),
            Session(
                "long-3",
                "2026-10-10",
                (
                    Turn("assistant", "You could try rock climbing sometime."),
                    Turn("user", "Correction: I cycle on Saturdays, not Sundays."),
                    Turn("user", "I moved to Tokyo in 2025."),
                ),
            ),
            conversation(
                "long-4",
                "2026-10-20",
                "I still study at NUS.",
                "As I mentioned, I spent my childhood in Malaysia.",
                "My project deadline is 30 October 2026.",
            ),
        ),
        (("long-1", 2), ("long-4", 1)),
        "Check formation loss before forgetting, repeated target paraphrases, the corrected "
        "cycling day, and dates. Rock climbing is an assistant suggestion, not a user fact.",
    ),
)


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def review_template(case: Case) -> dict:
    # null means unreviewed/uncertain, never an automatic success.
    return {
        "instructions": "Set presence fields to true/false after semantic review; leave null "
        "when uncertain. Quote evidence in notes. No keyword-based scoring is performed.",
        "target": {
            "fact": case.target,
            "pre_direct": None,
            "post_direct": None,
            "never_learned_direct": None,
            "post_inferable_from_retained_facts": None,
            "notes": "",
        },
        "retained": [
            {"fact": fact, "pre": None, "post": None, "never_learned": None, "notes": ""}
            for fact in case.retained_facts
        ],
        "other_hallucinations_or_drift": "",
    }


def assess_review(review: dict) -> dict:
    """Classifications based solely on human annotations, with no invented score."""
    target = review["target"]
    pre, post = target["pre_direct"], target["post_direct"]
    formation = None if pre is None else not pre
    if pre is False:
        forgetting = "not_evaluable_target_missing_before_forget"
    elif pre is None or post is None:
        forgetting = "pending_review"
    else:
        forgetting = "failure" if post else "no_direct_target_observed"
    inferred = target["post_inferable_from_retained_facts"]
    return {
        "memory_formation_failure": formation,
        "forgetting": forgetting,
        "collateral_information_loss": [
            fact["fact"]
            for fact in review["retained"]
            if fact["pre"] is True and fact["post"] is False
        ],
        "retained_facts_missing_before_forget": [
            fact["fact"] for fact in review["retained"] if fact["pre"] is False
        ],
        "retained_facts_awaiting_review": [
            fact["fact"]
            for fact in review["retained"]
            if fact["pre"] is None or fact["post"] is None
        ],
        "potential_inferential_leakage": inferred if post is False else None,
        "never_learned_direct_target": target["never_learned_direct"],
        "limitation": "Memory-text review does not establish behavioural forgetting.",
    }


def run_case(case: Case, directory: Path, client, model: str) -> dict:
    directory.mkdir()
    write_json(directory / "case.json", asdict(case))
    write_json(
        directory / "never_learned_input.json", [asdict(s) for s in case.never_learned_sessions()]
    )
    review = review_template(case)
    write_json(directory / "review.json", review)
    write_json(directory / "assessment.json", assess_review(review))
    report = {"case_id": case.case_id, "snapshots": {}, "errors": [], "complete": False}

    def capture(stage, operation):
        try:
            summary = operation()
        except Exception as error:
            # Preserve other independent stages and record the failed stage explicitly.
            report["errors"].append(
                {"stage": stage, "type": type(error).__name__, "message": str(error)}
            )
            return None
        (directory / f"{stage}.txt").write_text(summary, encoding="utf-8")
        report["snapshots"][stage] = summary
        return summary

    def build(sessions, path):
        memory = SummaryMemory(path, model=model, client=client)
        for s in sessions:
            memory.write(s)
        return memory.dump()

    pre = capture("pre_forget", lambda: build(case.sessions, directory / "pre_memory"))
    if pre is not None:

        def forget_copy():
            post_path = directory / "post_memory"
            post_path.mkdir()
            source = directory / "pre_memory" / "summary.txt"
            if source.exists():
                shutil.copyfile(source, post_path / "summary.txt")
            memory = SummaryMemory(post_path, model=model, client=client)
            memory.forget(case.target)
            return memory.dump()

        capture("post_forget", forget_copy)
    # This starts from empty memory and receives only the ablated conversation.
    capture(
        "never_learned", lambda: build(case.never_learned_sessions(), directory / "never_memory")
    )
    for other in ("post_forget", "never_learned"):
        if pre is not None and other in report["snapshots"]:
            diff = unified_diff(
                pre.splitlines(),
                report["snapshots"][other].splitlines(),
                fromfile="pre_forget",
                tofile=other,
                lineterm="",
            )
            (directory / f"pre_vs_{other}.diff").write_text("\n".join(diff), encoding="utf-8")
    report["complete"] = len(report["snapshots"]) == 3
    write_json(directory / "result.json", report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Explicitly enable real API requests")
    parser.add_argument(
        "--output", type=Path, help="New output directory; existing paths are refused"
    )
    parser.add_argument("--case", action="append", choices=[case.case_id for case in CASES])
    parser.add_argument("--provider", choices=("openai", "openrouter"), default="openai")
    parser.add_argument("--model", help="Default: shared model, with provider-specific prefix")
    parser.add_argument("--base-url", help="Optional OpenAI-compatible provider URL")
    parser.add_argument(
        "--review-run", type=Path, help="Offline: classify edited review.json files"
    )
    args = parser.parse_args(argv)
    if args.review_run:
        if args.live:
            parser.error("--review-run cannot be combined with --live")
        reviews = sorted(args.review_run.glob("*/review.json"))
        if not reviews:
            parser.error("No case review.json files found")
        for path in reviews:
            result = path.parent / "result.json"
            if (
                not result.exists()
                or not json.loads(result.read_text(encoding="utf-8"))["complete"]
            ):
                print(f"Skipping incomplete case: {path.parent.name}")
                continue
            review = json.loads(path.read_text(encoding="utf-8"))
            # Reject accidental strings such as 'false', which are truthy in Python.
            values = [v for k, v in review["target"].items() if k not in ("fact", "notes")]
            values += [f[k] for f in review["retained"] for k in ("pre", "post", "never_learned")]
            if any(value is not None and type(value) is not bool for value in values):
                parser.error(f"Presence annotations must be true, false, or null: {path}")
            write_json(path.parent / "assessment.json", assess_review(review))
        print(f"Updated manual assessments in {args.review_run}")
        return 0

    if args.provider == "openrouter":
        api_key_env = "OPENROUTER_API_KEY"
        model = args.model or f"openai/{DEFAULT_MODEL}"
        base_url = args.base_url or "https://openrouter.ai/api/v1"
    else:
        api_key_env = "OPENAI_API_KEY"
        model = args.model or DEFAULT_MODEL
        base_url = args.base_url or os.environ.get("OPENAI_BASE_URL")

    selected = [case for case in CASES if not args.case or case.case_id in args.case]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or Path("runs") / f"summary-pilot-{stamp}"
    # Never reuse a memory directory from another run.
    output.mkdir(parents=True, exist_ok=False)
    planned = sum(case.planned_requests() for case in selected)
    record = log_run(
        output / "run.jsonl",
        task="summary_selective_forgetting_pilot",
        live=args.live,
        provider=args.provider,
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
        temperature=0,
        python_version=sys.version,
        openai_version=openai_version,
        max_chars=DEFAULT_MAX_CHARS,
        planned_requests=planned,
        cases=[case.case_id for case in selected],
        prompts={"write": WRITE_PROMPT, "forget": FORGET_PROMPT},
    )
    write_json(output / "manifest.json", record)
    write_json(output / "cases.json", [asdict(case) for case in selected])
    print(f"{len(selected)} cases; up to {planned} summary requests (excluding SDK retries).")
    print(f"Output: {output}")
    if not args.live:
        print("Plan only; no API requests. Add --live to execute.")
        return 0
    try:
        api_key = os.environ.get(api_key_env)
        if not api_key or not api_key.strip():
            raise ValueError(
                f"Missing {api_key_env} for provider {args.provider}. "
                "Load .env using: uv run --env-file .env python -m experiments.summary_pilot ..."
            )
        with OpenAI(api_key=api_key, base_url=base_url, timeout=60.0, max_retries=2) as client:
            reports = []
            for case in selected:
                print(f"Running {case.case_id} ...", flush=True)
                reports.append(run_case(case, output / case.case_id, client, model))
    except Exception as error:
        write_json(output / "error.json", {"type": type(error).__name__, "message": str(error)})
        print(f"Pilot stopped: {type(error).__name__}; see {output / 'error.json'}")
        if isinstance(error, ValueError):
            print(str(error))
        return 1
    failed = [report["case_id"] for report in reports if not report["complete"]]
    write_json(output / "results.json", reports)
    print(
        f"Completed {len(reports) - len(failed)}/{len(reports)} cases. "
        "Semantic assessments await manual review; no success scores assigned."
    )
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
