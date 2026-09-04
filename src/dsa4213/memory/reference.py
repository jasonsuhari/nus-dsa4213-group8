"""Two working arms. Owner: Jason.

These exist for two reasons. They are worked examples of the interface, so
nobody has to guess what a real implementation looks like. And they are both
actual experimental controls, so they are not throwaway code.

Without both of these the leakage numbers have no scale. If the agent
recommends Indonesian restaurants to 15% of users who were never told, then a
measured 35% leakage is really 20%.
"""

import json
from pathlib import Path

from .base import Memory, Session


class NullMemory(Memory):
    """The never-told floor. Stores nothing, so the model can only guess.

    Whatever leakage this scores is the model's base rate. Subtract it from
    every other arm.
    """

    def write(self, session: Session) -> None:
        pass

    def retrieve(self, query: str) -> str:
        return ""

    def forget(self, target: str, level: int) -> None:
        pass

    def dump(self) -> str:
        return ""


class FullTranscriptMemory(Memory):
    """The no-delete ceiling. Keeps everything and ignores forget requests.

    Shows what "still fully remembers" looks like on the same probes, which is
    the upper end of the leakage scale.
    """

    @property
    def _log(self) -> Path:
        return self.path / "sessions.jsonl"

    def write(self, session: Session) -> None:
        record = {"id": session.session_id, "date": session.date, "text": session.transcript}
        with self._log.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def retrieve(self, query: str) -> str:
        # Most recent first, since that is what fits when we hit the cap.
        text = self.dump()
        return text[-self.max_chars :]

    def forget(self, target: str, level: int) -> None:
        # Deliberately a no-op. This arm is the ceiling control.
        pass

    def dump(self) -> str:
        if not self._log.exists():
            return ""
        rows = [json.loads(line) for line in self._log.read_text().splitlines() if line.strip()]
        return "\n\n".join(f"[{r['date']}] {r['text']}" for r in rows)
