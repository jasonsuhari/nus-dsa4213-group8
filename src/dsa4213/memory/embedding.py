"""Embedding retrieval arm. Owner: Jordan.

What to build
-------------
Verbatim session chunks in a vector store. Do NOT extract facts first: the
published result is that verbatim chunks beat extracted artifacts for recall
(arXiv 2601.00821), and extraction would also delete the thing this arm is
here to demonstrate.

  write()    chunk the session (session-level is fine to start), prepend the
             date into the chunk text, embed, store.
  retrieve() hybrid BM25 + dense, rerank, top-k ~10, truncate to max_chars.
  forget()   see below.
  dump()     concatenate every stored chunk.

forget(target, level)
---------------------
  1  embed the target, retrieve candidates, LLM-verify each one actually
     states the fact, delete only confirmed hits
  2  same, but accept paraphrases as hits
  3  also delete anything above cosine threshold tau, no LLM check
  4  also delete chunks holding entailing evidence (family, hometown, language)

Consider offering redaction as well as deletion: rewrite the chunk with the
fact stripped and the rest intact. Deletion vs redaction is a real result.

Failure mode to expect
----------------------
Chunks are session-level and multi-topic. Deleting the chunk that says
"I'm Indonesian" also deletes the dentist appointment from the same session.
That collateral damage comes purely from chunk granularity, and it is a
finding, not a bug. Log it rather than engineering around it.
"""

from .base import Memory, Session


class EmbeddingMemory(Memory):
    def write(self, session: Session) -> None:
        raise NotImplementedError("Jordan: chunk, embed, store")

    def retrieve(self, query: str) -> str:
        raise NotImplementedError("Jordan: hybrid search, rerank, truncate to self.max_chars")

    def forget(self, target: str, level: int) -> None:
        raise NotImplementedError("Jordan: verify-then-delete, see module docstring for levels")

    def dump(self) -> str:
        raise NotImplementedError("Jordan: return every stored chunk as text")
