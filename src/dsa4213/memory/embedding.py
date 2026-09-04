"""Embedding retrieval approach. Owner: Jordan.

Verbatim session chunks in a vector store. Do not extract facts first, since
that removes the chunk granularity this approach exists to demonstrate.

    write()     chunk the session, prepend the date, embed, store
    retrieve()  search, take the top few, truncate
    dump()      concatenate every chunk

forget() retrieves candidates, has an LLM verify each one actually states the
fact, then deletes the hits.

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
