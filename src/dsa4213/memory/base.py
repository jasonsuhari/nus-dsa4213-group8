"""The contract every memory architecture implements."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

# Equal across approaches on purpose. Unequal context budgets would measure prompt
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

    def __init__(self, path: Path, max_chars: int = DEFAULT_MAX_CHARS) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.max_chars = max_chars

    @abstractmethod
    def write(self, session: Session) -> None:
        """Ingest one session, in chronological order.

        Stays active after a forget request, since reconstruction happens here.
        Appends. Delete the directory to start clean.
        """

    @abstractmethod
    def retrieve(self, query: str) -> str:
        """Context for the system prompt, at most self.max_chars."""

    @abstractmethod
    def forget(self, target: str) -> None:
        """Remove `target`, natural language such as "the user's nationality"."""

    @abstractmethod
    def dump(self) -> str:
        """Everything stored, as text. Reconstruction is checked against this."""
