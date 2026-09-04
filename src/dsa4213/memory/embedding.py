"""Embedding retrieval arm. Owner: Jordan.

Verbatim session chunks in a vector store. Do not extract facts first: verbatim
chunks beat extracted artifacts for recall (arXiv 2601.00821), and extraction
removes the chunk granularity this arm exists to demonstrate.

    write()     chunk the session, prepend the date, embed, store
    retrieve()  hybrid BM25 and dense, rerank, top-k ~10, truncate
    dump()      concatenate every chunk

forget(target, level)
    1  retrieve candidates, LLM-verify each states the fact, delete hits
    2  accept paraphrases as hits
    3  also delete above cosine threshold, no LLM check
    4  also delete entailing evidence

Worth trying redaction as well as deletion: rewrite the chunk with the fact
stripped. Deletion versus redaction is a result.

Expect collateral damage from chunk granularity. Deleting the chunk that says
"I'm Indonesian" also drops the dentist appointment from the same session. Log
it rather than engineering around it.
"""

from .base import Memory, Session


class EmbeddingMemory(Memory):
    def write(self, session: Session) -> None:
        raise NotImplementedError("chunk, embed, store")

    def retrieve(self, query: str) -> str:
        raise NotImplementedError("hybrid search, rerank, truncate to self.max_chars")

    def forget(self, target: str, level: int) -> None:
        raise NotImplementedError("verify then delete, see module docstring")

    def dump(self) -> str:
        raise NotImplementedError("return every chunk as text")
