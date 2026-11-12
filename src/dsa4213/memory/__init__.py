"""Memory architectures under comparison."""

from .base import DEFAULT_MAX_CHARS, Memory, Session, Turn
from .embedding import EmbeddingMemory
from .fact_embedding import FactEmbeddingMemory
from .kv import KVMemory
from .reference import FullTranscriptMemory, NullMemory
from .summary import SummaryMemory

# Approaches that must satisfy the full contract. The reference implementations are
# deliberately different semantics, so they are tested separately.
APPROACHES = [EmbeddingMemory, FactEmbeddingMemory, SummaryMemory, KVMemory]

__all__ = [
    "APPROACHES",
    "DEFAULT_MAX_CHARS",
    "EmbeddingMemory",
    "FactEmbeddingMemory",
    "FullTranscriptMemory",
    "KVMemory",
    "Memory",
    "NullMemory",
    "Session",
    "SummaryMemory",
    "Turn",
]
