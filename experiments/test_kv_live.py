"""Live smoke test of KVMemory against a real API. Costs a few cents.

Mirrors experiments/test_embedding_live.py (same facts, forget target and query), so the
approaches can be compared side by side.

    uv run python experiments/test_kv_live.py                      # OpenAI
    uv run python experiments/test_kv_live.py --provider openrouter
    uv run python experiments/test_kv_live.py --extended           # harder cases
    uv run python experiments/test_kv_live.py --out data/live_kv

Keys are read from the environment or .env: OPENAI_API_KEY or OPENROUTER_API_KEY.
Without --out the store goes in a temporary folder and is deleted at the end.
"""

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from openai import OpenAI

from dsa4213.memory import KVMemory, Session, Turn

PROVIDERS = {
    "openai": {"key": "OPENAI_API_KEY", "base_url": None, "model": "gpt-4o-mini"},
    "openrouter": {
        "key": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "openai/gpt-4o-mini",
    },
}

TARGET = "The user is vegetarian."
QUERY = "What do you know about me?"

# Same two sessions as the summary and embedding live tests.
BASIC = [
    Session("session_1", "2026-10-20", (Turn("user", "I am vegetarian."),)),
    Session(
        "session_2",
        "2026-10-21",
        (Turn("user", "I like Japanese food and I study at NUS."),),
    ),
]

# Same harder sessions as the embedding live test, each testing one expected behaviour
# of key-value memory.
EXTENDED = [
    # Paraphrase: should be its own entry, and identify should remove it with the target.
    Session("session_3", "2026-10-22", (Turn("user", "I don't eat meat these days."),)),
    # Fact sharing a turn with the target: split into two entries, so the gym fact survives.
    Session(
        "session_4",
        "2026-10-23",
        (Turn("user", "Still vegetarian, and I go to the gym on Mondays."),),
    ),
    # Related but not the target: should be kept, but may let a model infer the fact.
    Session("session_5", "2026-10-24", (Turn("user", "My favourite dish is tofu curry."),)),
]

# Written after the forget request: the updater is still on, so this re-adds the target.
RESTATED = Session("session_6", "2026-10-27", (Turn("user", "I'm vegetarian, any ramen nearby?"),))


def load_env(path: Path = Path(".env")) -> None:
    """Minimal .env reader, so python-dotenv is not needed as a dependency."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def make_memory(provider: str, path: Path) -> KVMemory:
    cfg = PROVIDERS[provider]
    key = os.environ.get(cfg["key"])
    if not key:
        raise SystemExit(f"{cfg['key']} is not set. Add it to .env or the environment.")
    client = OpenAI(api_key=key, base_url=cfg["base_url"])
    return KVMemory(path, model=cfg["model"], client=client)


def show(title: str, text: str) -> None:
    print(f"\n=== {title} ===")
    print(text or "(empty)")


def show_writes(memory: KVMemory, since: int) -> int:
    """Print the operations applied since line `since` of write_log.jsonl."""
    lines = memory.write_log_path.read_text(encoding="utf-8").splitlines()
    for line in lines[since:]:
        entry = json.loads(line)
        for step in entry["facts"]:
            print(f"  fact: {step['fact']}")
            for a in step["applied"] or [{"op": "NOOP", "key": "", "value": ""}]:
                print(f"    {a['op']:<6} {a['key']}  {a['value']}")
    return len(lines)


def run(memory: KVMemory, extended: bool) -> None:
    sessions = BASIC + (EXTENDED if extended else [])

    logged = 0
    for i, session in enumerate(sessions, start=1):
        memory.write(session)
        print(f"\n=== Write {i} ({session.session_id}) ===")
        logged = show_writes(memory, logged)

    show("Profile before forget (dump)", memory.dump())
    show("Retrieved before forget", memory.retrieve(QUERY))

    memory.forget(TARGET)
    log = json.loads(memory.forget_log_path.read_text(encoding="utf-8").splitlines()[-1])
    print("\n=== Forget log ===")
    print(f"  target: {log['target']}")
    for r in log["removed"]:
        print(f"  removed {r['key']}: {r['value']}")
    print(f"  remaining entries: {log['remaining']}")

    show("Profile after forget (dump)", memory.dump())
    show("Retrieved after forget", memory.retrieve(QUERY))
    print(f"  keys: {memory.last_keys}")

    if extended:
        memory.write(RESTATED)
        print(f"\n=== Write after forget ({RESTATED.session_id}) ===")
        show_writes(memory, logged)
        show("Profile after restating (dump)", memory.dump())


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
