"""Live smoke test of EmbeddingMemory against a real API. Costs a few cents.

Mirrors experiments/test_summary_live.py (same facts, forget target and query), so the
two approaches can be compared side by side.

    uv run python experiments/test_embedding_live.py                      # OpenAI
    uv run python experiments/test_embedding_live.py --provider openrouter
    uv run python experiments/test_embedding_live.py --extended           # harder cases
    uv run python experiments/test_embedding_live.py --out data/live_embedding

Keys are read from the environment or .env: OPENAI_API_KEY or OPENROUTER_API_KEY.
Without --out the store goes in a temporary folder and is deleted at the end.
"""

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from openai import OpenAI

from dsa4213.memory import EmbeddingMemory, Session, Turn

PROVIDERS = {
    "openai": {
        "key": "OPENAI_API_KEY",
        "base_url": None,
        "model": "gpt-4o-mini",
        "embed_model": "text-embedding-3-small",
    },
    "openrouter": {
        "key": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "openai/gpt-4o-mini",
        "embed_model": "openai/text-embedding-3-small",
    },
}

TARGET = "The user is vegetarian."
QUERY = "What do you know about me?"

# Same two sessions as the summary live test.
BASIC = [
    Session("session_1", "2026-10-20", (Turn("user", "I am vegetarian."),)),
    Session(
        "session_2",
        "2026-10-21",
        (Turn("user", "I like Japanese food and I study at NUS."),),
    ),
]

# Harder cases for the embedding approach, each testing one thing the proposal predicts.
EXTENDED = [
    # Paraphrase: identify should catch this, or it is a surviving duplicate.
    Session("session_3", "2026-10-22", (Turn("user", "I don't eat meat these days."),)),
    # Fact sharing a turn with the target: deleting the turn loses the gym fact too.
    Session(
        "session_4",
        "2026-10-23",
        (Turn("user", "Still vegetarian, and I go to the gym on Mondays."),),
    ),
    # Related but not the target: should be kept, but may let a model infer the fact.
    Session("session_5", "2026-10-24", (Turn("user", "My favourite dish is tofu curry."),)),
]


def load_env(path: Path = Path(".env")) -> None:
    """Minimal .env reader, so python-dotenv is not needed as a dependency."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def make_memory(provider: str, path: Path) -> EmbeddingMemory:
    cfg = PROVIDERS[provider]
    key = os.environ.get(cfg["key"])
    if not key:
        raise SystemExit(f"{cfg['key']} is not set. Add it to .env or the environment.")
    client = OpenAI(api_key=key, base_url=cfg["base_url"])
    return EmbeddingMemory(path, model=cfg["model"], embed_model=cfg["embed_model"], client=client)


def show(title: str, text: str) -> None:
    print(f"\n=== {title} ===")
    print(text or "(empty)")


def run(memory: EmbeddingMemory, extended: bool) -> None:
    sessions = BASIC + (EXTENDED if extended else [])

    for i, session in enumerate(sessions, start=1):
        memory.write(session)
        show(f"After write {i} ({session.session_id})", memory.dump())

    print("\n=== Importance scores ===")
    for r in memory.records:
        print(f"  [{r['id']}] {r['importance']:>2}  {r['text']}")

    show("Retrieved before forget", memory.retrieve(QUERY))

    memory.forget(TARGET)
    show("After forget (dump)", memory.dump())

    log = json.loads(memory.forget_log_path.read_text(encoding="utf-8").splitlines()[-1])
    print("\n=== Forget log ===")
    print(f"  target: {log['target']}  mode: {log['mode']}")
    for r in log["removed"]:
        print(f"  removed [{r['id']}] {r['text']}")
    for d in log["near_duplicates"]:
        for s in d["survivors"]:
            flag = "  <-- likely duplicate" if s["likely_duplicate"] else ""
            print(f"  similar to [{d['removed_id']}]: [{s['id']}] {s['cosine']}{flag}")
    print(f"  likely_duplicates_left: {log['likely_duplicates_left']}")

    show("Retrieved after forget", memory.retrieve(QUERY))
    print("\n=== Trace (records behind that answer) ===")
    for t in memory.last_trace:
        print(
            f"  #{t['rank']} [{t['id']}] score={t['score']:.2f} "
            f"(rel={t['relevance']:.2f} rec={t['recency']:.2f} imp={t['importance']:.2f})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default="openai")
    parser.add_argument("--extended", action="store_true", help="add the harder sessions")
    parser.add_argument("--out", type=Path, help="keep the store here instead of a temp dir")
    args = parser.parse_args()

    load_env()
    if args.out:
        if args.out.exists() and any(args.out.iterdir()):
            raise SystemExit(f"{args.out} is not empty. write() appends, so use a new folder.")
        run(make_memory(args.provider, args.out), args.extended)
        print(f"\nStore kept in {args.out}")
    else:
        with TemporaryDirectory() as tmp:
            run(make_memory(args.provider, Path(tmp)), args.extended)


if __name__ == "__main__":
    main()
