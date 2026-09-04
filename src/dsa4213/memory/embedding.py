"""Embedding retrieval arm. Owner: Jordan.

Verbatim session chunks in a vector store. Do not extract facts first: verbatim
chunks beat extracted artifacts for recall (arXiv 2601.00821), and extraction
removes the chunk granularity this arm exists to demonstrate.

    write()     chunk the session, prepend the date, embed, store
    retrieve()  hybrid BM25 and dense, rerank, top-k ~10, truncate
    dump()      concatenate every chunk

forget() retrieves candidates, has an LLM verify each one actually states the
fact, then deletes the hits. Worth trying redaction too: rewrite the chunk with
the fact stripped rather than dropping the whole thing.

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

    def forget(self, target: str) -> None:
        raise NotImplementedError("verify then delete, see module docstring")

    def dump(self) -> str:
        raise NotImplementedError("return every chunk as text")
