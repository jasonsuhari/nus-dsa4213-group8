"""Chat with the agent in the terminal, on any memory approach. Costs well under a cent
per message on gpt-4o-mini.

    uv run python experiments/chat_live.py --provider openrouter
    uv run python experiments/chat_live.py --memory summary --store data/chat/summary

Commands inside the chat:
    /tell <text>     ingest <text> as a user message, the way the experiment feeds sessions
    /forget <fact>   run forget(), for example: /forget the user is vegetarian
    /dump            print everything stored
    /quit

Plain messages go to Agent.chat(), which reads memory but only writes to it when the
model calls the remember tool. The store folder is kept, so running again with the same
--store carries on where you stopped. Keys are read from the environment or .env.
"""

import argparse
import os
from datetime import date
from pathlib import Path

from openai import OpenAI

from dsa4213.agent import Agent
from dsa4213.memory import EmbeddingMemory, KVMemory, Session, SummaryMemory, Turn

PROVIDERS = {
    "openai": {"key": "OPENAI_API_KEY", "base_url": None, "prefix": ""},
    "openrouter": {
        "key": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "prefix": "openai/",
    },
}


def load_env(path: Path = Path(".env")) -> None:
    """Minimal .env reader, so python-dotenv is not needed as a dependency."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def make_agent(memory_name: str, provider: str, store: Path) -> Agent:
    cfg = PROVIDERS[provider]
    key = os.environ.get(cfg["key"])
    if not key:
        raise SystemExit(f"{cfg['key']} is not set. Add it to .env or the environment.")
    client = OpenAI(api_key=key, base_url=cfg["base_url"])
    model = cfg["prefix"] + "gpt-4o-mini"
    if memory_name == "kv":
        memory = KVMemory(store, model=model, client=client)
    elif memory_name == "summary":
        memory = SummaryMemory(store, model=model, client=client)
    else:
        embed_model = cfg["prefix"] + "text-embedding-3-small"
        memory = EmbeddingMemory(store, model=model, embed_model=embed_model, client=client)
    return Agent(memory, model=model, client=client)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory", choices=["kv", "summary", "embedding"], default="kv")
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default="openrouter")
    parser.add_argument("--store", type=Path, help="default: data/chat/<memory>")
    args = parser.parse_args()

    load_env()
    store = args.store or Path("data/chat") / args.memory
    agent = make_agent(args.memory, args.provider, store)
    today = date.today().isoformat()
    told = 0
    print(f"{args.memory} memory in {store}. /tell, /forget, /dump, /quit\n")

    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        command, _, rest = line.partition(" ")
        if command == "/quit":
            break
        elif command == "/dump":
            print(agent.memory.dump() or "(empty)")
        elif command == "/tell":
            told += 1
            agent.ingest(Session(f"chat-{today}-{told:03d}", today, (Turn("user", rest),)))
            print("(ingested)")
        elif command == "/forget":
            agent.forget(rest)
            print("(forgot)")
        else:
            saved = len(agent.remembered)
            print(f"agent> {agent.chat(line, date=today)}")
            for fact in agent.remembered[saved:]:
                print(f"  (remember tool saved: {fact})")
            if isinstance(agent.memory, KVMemory):
                print(f"  (keys shown to the model: {agent.memory.last_keys})")
        print()


if __name__ == "__main__":
    main()
