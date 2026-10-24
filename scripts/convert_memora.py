"""Convert a Memora clone into the session jsonl this repo ingests.

    git clone https://github.com/geniesinc/Memora.git /tmp/memora
    uv run python scripts/convert_memora.py --memora /tmp/memora

Writes data/memora/<split>/<persona>.jsonl, one session per line. Memora's own
fields (operation, session_type, operation_details, share_memory) are kept
alongside the ones Session reads, since the eval harness needs them later.
"""

import argparse
import json
from pathlib import Path

SPEAKERS = {"user_agent": "user", "ai_agent": "assistant"}


def convert_session(raw: dict) -> dict:
    speaking = [t for t in raw["conversation"] if t["speaker"] in SPEAKERS]
    return {
        "session_id": f"s{raw['session_id']:04d}",
        "date": raw["date"],
        "turns": [{"role": SPEAKERS[t["speaker"]], "text": t["message"]} for t in speaking],
        "memora": {
            "operation": raw.get("operation"),
            "session_type": raw.get("session_type"),
            "operation_details": raw.get("operation_details"),
            # turn indices whose content is meant to enter memory
            "memory_turns": [i for i, t in enumerate(speaking) if t.get("share_memory")],
        },
    }


def convert_persona(src: Path, dest: Path) -> int:
    files = sorted(src.glob("session_*.json"), key=lambda p: p.name)
    rows = [convert_session(json.loads(f.read_text())) for f in files]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memora", type=Path, required=True, help="path to a Memora clone")
    parser.add_argument("--split", default="weekly", choices=["weekly", "monthly", "quarterly"])
    parser.add_argument("--out", type=Path, default=Path("data/memora"))
    args = parser.parse_args()

    root = args.memora / "data" / args.split
    if not root.is_dir():
        raise SystemExit(f"no {root}. Clone Memora first, see the module docstring.")

    for persona_dir in sorted(root.iterdir()):
        conversations = persona_dir / "conversations"
        if not conversations.is_dir():
            continue
        dest = args.out / args.split / f"{persona_dir.name}.jsonl"
        n = convert_persona(conversations, dest)
        print(f"  {persona_dir.name:24} {n:>5} sessions  {dest}")


if __name__ == "__main__":
    main()
