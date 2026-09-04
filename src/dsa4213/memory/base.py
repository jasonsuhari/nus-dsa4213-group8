"""The contract every memory architecture implements.

Four methods, one dataclass. Jason owns this file. The three architecture
modules subclass `Memory` and fill in the bodies.

Do not add a method without telling the group. Every arm has to change, and a
method one arm needs but the others fake is a confound, not an interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

# The aggressiveness dial. Same meaning in every arm, or the sweep isn't comparable.
FORGET_LEVELS = {
    1: "exact statements of the fact only",
    2: "level 1, plus paraphrases",
    3: "level 2, plus anything above the arm's own similarity threshold",
    4: "level 3, plus one hop of entailing evidence (family, hometown, language)",
}

# Cap on what retrieve() may return, in characters. Equal across arms on purpose:
# if the summary arm returns 500 chars and the chunk arm returns 4000, every
# downstream number is measuring context length rather than architecture.
DEFAULT_MAX_CHARS = 2000


@dataclass(frozen=True)
class Turn:
    role: str  # "user" or "assistant"
    text: str


@dataclass(frozen=True)
class Session:
    session_id: str
    date: str  # ISO date. Arms that reason about time must put this in the stored text.
    turns: tuple[Turn, ...]

    @property
    def transcript(self) -> str:
        return "\n".join(f"{t.role}: {t.text}" for t in self.turns)


class Memory(ABC):
    """One memory architecture.

    Everything the arm persists lives under `path`. The harness snapshots a
    store by copying that directory, so no arm implements snapshot/restore.
    """

    def __init__(self, path: Path, config: dict | None = None) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        self.max_chars = self.config.get("max_chars", DEFAULT_MAX_CHARS)

    @abstractmethod
    def write(self, session: Session) -> None:
        """Ingest one session. Called once per session, in chronological order.

        This is the write path, and it stays active after a forget request.
        Reconstruction happens here or it doesn't happen at all.
        """

    @abstractmethod
    def retrieve(self, query: str) -> str:
        """Return context to paste into the system prompt.

        A string, not objects. The harness owns prompt construction so the
        prompt is identical across arms. Must not exceed self.max_chars.
        """

    @abstractmethod
    def forget(self, target: str, level: int) -> None:
        """Remove `target` from the store at aggressiveness `level`.

        `target` is natural language, e.g. "the user's nationality". Each arm
        interprets it its own way. That interpretation is what we're studying,
        so do not add a structured key argument to make this easier.
        """

    @abstractmethod
    def dump(self) -> str:
        """Everything currently stored, as text.

        Used to grep for reconstruction. Reconstruction is a store-level event,
        not an output-level one, so this must reflect the store and not a query.
        """
