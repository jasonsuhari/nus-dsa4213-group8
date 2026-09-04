"""Memory architectures under comparison.

Every arm implements the same four-method contract in base.py. Add a new arm
by subclassing Memory and registering it in ARMS below, so the conformance
test picks it up automatically.
"""

from .base import DEFAULT_MAX_CHARS, FORGET_LEVELS, Memory, Session, Turn
from .embedding import EmbeddingMemory
from .kv import KVMemory
from .reference import FullTranscriptMemory, NullMemory
from .summary import SummaryMemory

# The arms that must satisfy the full contract. The reference arms are controls
# with deliberately different semantics, so they are tested separately.
ARMS = [EmbeddingMemory, SummaryMemory, KVMemory]

__all__ = [
    "ARMS",
    "DEFAULT_MAX_CHARS",
    "FORGET_LEVELS",
    "EmbeddingMemory",
    "FullTranscriptMemory",
    "KVMemory",
    "Memory",
    "NullMemory",
    "Session",
    "SummaryMemory",
    "Turn",
]
