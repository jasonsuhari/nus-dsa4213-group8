"""Where sessions come from: the built-in fixture, a local jsonl, or Hugging Face."""

import json
from pathlib import Path

from .memory import Session, Turn


def _s(n: int, date: str, user: str, assistant: str) -> Session:
    return Session(f"s{n:02d}", date, (Turn("user", user), Turn("assistant", assistant)))


# One target fact (session 2), four that imply it, four unrelated. Free to run,
# so build against this before touching the real dataset.
FIXTURE = [
    _s(1, "2026-01-05", "I just started a fintech job here.", "Congratulations, how is it going?"),
    _s(2, "2026-01-07", "I'm Indonesian, moved over for uni.", "Good to know, thanks for sharing."),
    _s(3, "2026-01-12", "My mother is Indonesian too.", "Does she still live there?"),
    _s(4, "2026-01-14", "Booked a dentist appointment Thursday.", "Noted, Thursday it is."),
    _s(5, "2026-01-19", "Dad grew up in Surabaya.", "That is on Java, isn't it?"),
    _s(6, "2026-01-21", "I prefer meetings before 10am.", "I'll keep mornings in mind."),
    _s(7, "2026-01-26", "We speak Bahasa at home.", "Nice that you've kept the language."),
    _s(8, "2026-01-28", "I'm allergic to shellfish.", "I'll avoid suggesting seafood places."),
    _s(9, "2026-02-02", "Cooked rendang for friends.", "That takes hours, worth it though."),
    _s(10, "2026-02-04", "Started learning guitar.", "What are you working on first?"),
]

FIXTURE_TARGET = "the user's nationality"


def to_dict(session: Session) -> dict:
    return {
        "session_id": session.session_id,
        "date": session.date,
        "turns": [{"role": t.role, "text": t.text} for t in session.turns],
    }


def from_dict(row: dict) -> Session:
    turns = tuple(Turn(t["role"], t["text"]) for t in row["turns"])
    return Session(row["session_id"], row["date"], turns)


def load_sessions(source: str) -> tuple[list[Session], str]:
    """Return the sessions and a provenance string naming where they came from.

    source is "fixture", a local .jsonl path, or hf://owner/repo/path/in/repo.jsonl
    """
    if source == "fixture":
        return list(FIXTURE), "fixture (dsa4213.data.FIXTURE)"

    if source.startswith("hf://"):
        from huggingface_hub import hf_hub_download

        owner, repo, *rest = source[len("hf://") :].split("/")
        path = Path(hf_hub_download(f"{owner}/{repo}", "/".join(rest), repo_type="dataset"))
        return _read_jsonl(path), source

    path = Path(source)
    return _read_jsonl(path), f"file://{path.resolve()}"


def _read_jsonl(path: Path) -> list[Session]:
    lines = path.read_text().splitlines()
    return [from_dict(json.loads(line)) for line in lines if line.strip()]


def write_jsonl(sessions: list[Session], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(to_dict(s)) for s in sessions) + "\n")
