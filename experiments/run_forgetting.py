"""Run every case through every memory architecture under the three conditions.

    uv run --env-file .env python -m experiments.run_forgetting
    uv run --env-file .env python -m experiments.run_forgetting --only summary --case diet

For each case and architecture this builds three stores:

    pre_forget      all sessions written
    post_forget     a copy of pre_forget, then forget(target)
    never_learned   the sessions with the target turns removed

then asks every probe in a fresh agent with the remember tool hidden. Output goes to
runs/<run_id>/: responses.jsonl (one row per answer, the input to evaluation/),
stores/ (every store, inspectable) and summary.json.
"""

import argparse
import json
import shutil
import sys
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from dsa4213.agent import DEFAULT_MODEL, Agent
from dsa4213.cases import Case, load_cases
from dsa4213.memory import (
    EmbeddingMemory,
    FullTranscriptMemory,
    KVMemory,
    Memory,
    NullMemory,
    SummaryMemory,
)
from dsa4213.runlog import git_commit, log_run

ARCHITECTURES: dict[str, type[Memory]] = {
    "embedding": EmbeddingMemory,
    "summary": SummaryMemory,
    "kv": KVMemory,
    "null": NullMemory,
    "full": FullTranscriptMemory,
}
CONDITIONS = ("pre_forget", "post_forget", "never_learned")
DEFAULT_CASES = Path(__file__).parent / "cases" / "pilot.json"


def build_stores(case: Case, arch: str, root: Path) -> dict[str, Path]:
    cls = ARCHITECTURES[arch]
    paths = {c: root / c for c in CONDITIONS}

    pre = cls(paths["pre_forget"])
    for session in case.sessions:
        pre.write(session)

    shutil.copytree(paths["pre_forget"], paths["post_forget"])
    cls(paths["post_forget"]).forget(case.target)

    never = cls(paths["never_learned"])
    for session in case.never_learned_sessions():
        never.write(session)
    return paths


def probe_stores(
    case: Case, arch: str, paths: dict[str, Path], model: str, client: OpenAI
) -> list[dict]:
    rows = []
    for condition, path in paths.items():
        memory = ARCHITECTURES[arch](path)
        before = memory.dump()
        for probe in case.probes:
            agent = Agent(memory, model=model, client=client)
            response = agent.chat(probe.question, learn=False)
            rows.append(
                {
                    "case_id": case.case_id,
                    "architecture": arch,
                    "condition": condition,
                    "probe_type": probe.probe_type,
                    "probe": probe.question,
                    "scoring_target": list(probe.scoring_target),
                    "target_fact": case.target,
                    "response": response,
                    "context": agent.last_context,
                }
            )
        if memory.dump() != before:
            raise RuntimeError(f"{case.case_id}/{arch}/{condition}: probing changed the store")
        (path / "dump.txt").write_text(before)
    return rows


def run_one(case: Case, arch: str, out: Path, model: str, client: OpenAI) -> list[dict]:
    root = out / "stores" / case.case_id / arch
    shutil.rmtree(root, ignore_errors=True)
    return probe_stores(case, arch, build_stores(case, arch, root), model, client)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--only", choices=sorted(ARCHITECTURES), action="append")
    parser.add_argument("--case", action="append", help="case_id to run, repeatable")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("runs"))
    args = parser.parse_args()

    cases = [c for c in load_cases(args.cases) if not args.case or c.case_id in args.case]
    archs = args.only or list(ARCHITECTURES)
    run_id = uuid.uuid4().hex[:8]
    out = args.out / run_id
    out.mkdir(parents=True)
    client = OpenAI()
    started = time.monotonic()

    rows, status = [], {}
    with ThreadPoolExecutor(args.workers) as pool:
        jobs = {
            pool.submit(run_one, case, arch, out, args.model, client): (case.case_id, arch)
            for case in cases
            for arch in archs
        }
        for job in as_completed(jobs):
            key = jobs[job]
            try:
                rows += job.result()
                status[key] = "ok"
            except NotImplementedError:
                status[key] = "not_implemented"
            except Exception as e:
                status[key] = f"error: {type(e).__name__}: {e}"
            print(f"  {status[key][:60]:60} {key[0]}/{key[1]}", flush=True)

    rows.sort(key=lambda r: (r["case_id"], r["architecture"], r["condition"]))
    with (out / "responses.jsonl").open("w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)

    summary = {
        "run_id": run_id,
        "commit": git_commit(),
        "model": args.model,
        "cases_file": str(args.cases),
        "n_responses": len(rows),
        "rows_per_architecture": Counter(r["architecture"] for r in rows),
        "status": {f"{c}/{a}": s for (c, a), s in sorted(status.items())},
        "duration_s": round(time.monotonic() - started, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log_run(task="run_forgetting", **{k: v for k, v in summary.items() if k != "status"})
    print(f"\n{len(rows)} responses in {out}/responses.jsonl ({summary['duration_s']}s)")

    errors = [k for k, s in status.items() if s.startswith("error")]
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
