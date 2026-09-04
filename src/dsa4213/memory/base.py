"""The contract every memory architecture implements."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

# Aggressiveness dial for forget(). Same meaning in every arm or the sweep is
# not comparable.
FORGET_LEVELS = {
    1: "exact statements only",
    2: "plus paraphrases",
    3: "plus anything above the arm's similarity threshold",
    4: "plus one hop of entailing evidence (family, hometown, language)",
}

# Equal across arms on purpose. Unequal context budgets would measure prompt
# length instead of architecture.
DEFAULT_MAX_CHARS = 2000


@dataclass(frozen=True)
class Turn:
    role: str
    text: str


@dataclass(frozen=True)
class Session:
    session_id: str
    date: str
    turns: tuple[Turn, ...]

    @property
    def transcript(self) -> str:
        return "\n".join(f"{t.role}: {t.text}" for t in self.turns)


class Memory(ABC):
    """One memory architecture. All state lives under `path`."""

    def __init__(self, path: Path, config: dict | None = None) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        self.max_chars = self.config.get("max_chars", DEFAULT_MAX_CHARS)

    @abstractmethod
    def write(self, session: Session) -> None:
        """Ingest one session, in chronological order.

        Stays active after a forget request, since reconstruction happens here.
        """

    @abstractmethod
    def retrieve(self, query: str) -> str:
        """Context for the system prompt, at most self.max_chars."""

    @abstractmethod
    def forget(self, target: str, level: int) -> None:
        """Remove `target` at the given level.

        `target` is natural language, e.g. "the user's nationality". How each
        arm interprets that is what the study measures, so it stays unstructured.
        """

    @abstractmethod
    def dump(self) -> str:
        """Everything stored, as text. Reconstruction is checked against this."""
