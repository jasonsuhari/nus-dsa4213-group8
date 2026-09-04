"""Build a memory store for every approach from one set of sessions.

    uv run python scripts/build_memory.py --limit 5
    uv run python scripts/build_memory.py --sessions hf://jasonsuhari/dsa4213/weekly.jsonl
    uv run python scripts/build_memory.py --only summary

Approaches that are still stubs are skipped and reported, so this runs from day
one. Each store gets a manifest.json recording what built it.
"""

import argparse
import hashlib
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

from dsa4213.data import load_sessions
from dsa4213.memory import (
    EmbeddingMemory,
    FullTranscriptMemory,
    KVMemory,
    NullMemory,
    SummaryMemory,
)
from dsa4213.runlog import git_commit, log_run

APPROACHES = {
    "embedding": EmbeddingMemory,
    "summary": SummaryMemory,
    "kv": KVMemory,
    "null": NullMemory,
    "full": FullTranscriptMemory,
}


def build(name: str, sessions: list, out: Path, provenance: str) -> dict:
    store = out / name
    # write() appends, so a stale directory silently doubles the store
    shutil.rmtree(store, ignore_errors=True)
    started = time.monotonic()
    memory = APPROACHES[name](store)
    try:
        for session in sessions:
            memory.write(session)
        status = "ok"
    except NotImplementedError as e:
        status = f"not_implemented: {e}"

    manifest = {
        "approach": name,
        "status": status,
        "minted_at": datetime.now(UTC).isoformat(),
        "commit": git_commit(),
        "source": provenance,
        "source_sha256": _digest(sessions),
        "n_sessions": len(sessions),
        "first_date": sessions[0].date if sessions else None,
        "last_date": sessions[-1].date if sessions else None,
        "duration_s": round(time.monotonic() - started, 2),
    }
    (store / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _digest(sessions: list) -> str:
    """Fingerprint of the input, so two stores built from different data are visible."""
    joined = "\n".join(f"{s.session_id}|{s.date}|{s.transcript}" for s in sessions)
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", default="fixture", help="fixture, a .jsonl path, or hf://")
    parser.add_argument("--out", type=Path, default=Path("data/stores"))
    parser.add_argument("--limit", type=int, help="first N sessions only, for a fast loop")
    parser.add_argument("--only", choices=sorted(APPROACHES), help="build one approach")
    args = parser.parse_args()

    sessions, provenance = load_sessions(args.sessions)
    if args.limit:
        sessions = sessions[: args.limit]

    names = [args.only] if args.only else list(APPROACHES)
    print(f"{len(sessions)} sessions from {provenance}\n")

    for name in names:
        manifest = build(name, sessions, args.out, provenance)
        mark = "ok" if manifest["status"] == "ok" else "skip"
        print(f"  {mark:4} {name:10} {manifest['duration_s']:>6.2f}s  {args.out / name}")
        if mark == "skip":
            print(f"       {manifest['status']}")

    log_run(
        task="build_memory",
        source=provenance,
        n_sessions=len(sessions),
        approaches=names,
        out=str(args.out),
    )


if __name__ == "__main__":
    main()
