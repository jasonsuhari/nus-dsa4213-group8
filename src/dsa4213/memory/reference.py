"""Two working approaches that double as the experiment's controls."""

import json
from pathlib import Path

from .base import Memory, Session


class NullMemory(Memory):
    """Never-told floor. Stores nothing, so leakage here is the model's base rate."""

    def write(self, session: Session) -> None:
        pass

    def retrieve(self, query: str) -> str:
        return ""

    def forget(self, target: str) -> None:
        pass

    def dump(self) -> str:
        return ""


class FullTranscriptMemory(Memory):
    """No-delete ceiling. Keeps everything and ignores forget requests."""

    @property
    def _log(self) -> Path:
        return self.path / "sessions.jsonl"

    def write(self, session: Session) -> None:
        record = {"id": session.session_id, "date": session.date, "text": session.transcript}
        with self._log.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def retrieve(self, query: str) -> str:
        # Tail, so the most recent sessions are what survives the cap.
        return self.dump()[-self.max_chars :]

    def forget(self, target: str) -> None:
        pass

    def dump(self) -> str:
        if not self._log.exists():
            return ""
        rows = [json.loads(line) for line in self._log.read_text().splitlines() if line.strip()]
        return "\n\n".join(f"[{r['date']}] {r['text']}" for r in rows)
