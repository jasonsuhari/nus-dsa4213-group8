"""Structured key-value profile arm. Owner: Wei Lun.

Mem0's four-operation updater. After each session, extract candidate facts,
look up similar existing keys, then ADD, UPDATE, DELETE or NOOP.

    write()     run the updater over the session
    retrieve()  relevant profile keys, under the budget
    dump()      serialise the whole profile

forget(target, level)
    1  delete the exact key
    2  also delete related keys the LLM identifies
    3  also delete keys whose values contain an alias
    4  also delete entailing keys (mother_nationality, languages_spoken)

Cleanest deletion of the arms and the lowest collateral damage. Shadow keys
survive though, so inferential leakage stays high. Leave the updater running
after a forget request: re-adding the key from a later session is the clearest
reconstruction path in the study.

The retraction arm is this class plus a rule list, if there is time.
"""

from .base import Memory, Session


class KVMemory(Memory):
    def write(self, session: Session) -> None:
        raise NotImplementedError("extract facts, then ADD/UPDATE/DELETE/NOOP")

    def retrieve(self, query: str) -> str:
        raise NotImplementedError("serialise relevant keys under self.max_chars")

    def forget(self, target: str, level: int) -> None:
        raise NotImplementedError("drop keys, see module docstring")

    def dump(self) -> str:
        raise NotImplementedError("serialise the whole profile")
