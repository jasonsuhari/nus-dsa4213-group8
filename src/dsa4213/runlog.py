"""Append-only JSONL log of experiment runs.

One line per run. Feeds three separate rubric lines in the final report:
evaluation evidence, cost/latency analysis, and reproducibility. Logging the
git commit is what lets the report say "Table 3 reproduces from commit abc123".
"""

from __future__ import annotations

import json
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path

RUNS = Path(__file__).resolve().parents[2] / "runs" / "runs.jsonl"


def git_commit() -> str:
    """Current commit, suffixed '-dirty' if the tree has uncommitted changes."""
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
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def log_run(path: Path | None = None, **fields) -> dict:
    """Append one run record and return it.

    Pass whatever the experiment produced: model, temperature, seed, n_tokens,
    cost_usd, latency_s, task, score, notes. run_id/timestamp/commit are added.
    """
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
    """Read every run back, oldest first. Empty list if nothing logged yet."""
    src = path or RUNS
    if not src.exists():
        return []
    return [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
