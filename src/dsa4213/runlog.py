"""Append-only JSONL log of experiment runs."""

import json
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path

RUNS = Path(__file__).resolve().parents[2] / "runs" / "runs.jsonl"


def git_commit() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        # -dirty means the run came from uncommitted code, so it won't reproduce
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def log_run(path: Path | None = None, **fields) -> dict:
    """Append one run. Pass model, seed, cost_usd, score, whatever the run produced."""
    record = {
        "run_id": uuid.uuid4().hex[:8],
        "timestamp": datetime.now(UTC).isoformat(),
        "commit": git_commit(),
        **fields,
    }
    out = path or RUNS
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def load_runs(path: Path | None = None) -> list[dict]:
    src = path or RUNS
    if not src.exists():
        return []
    return [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
