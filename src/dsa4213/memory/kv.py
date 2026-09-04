"""Structured key-value profile arm. Owner: Wei Lun.

What to build
-------------
Mem0's four-operation updater. After each session, extract candidate facts,
look up semantically similar existing keys, and issue one of:

  ADD     genuinely new information
  UPDATE  more recent or more detailed than what is stored
  DELETE  contradicted by the new information
  NOOP    already known, or irrelevant

  write()    run the updater over the session.
  retrieve() return the profile, or the relevant subset, under max_chars.
  forget()   see below.
  dump()     serialise the whole profile.

forget(target, level)
---------------------
  1  delete the exact key ("nationality")
  2  also delete semantically related keys the LLM identifies
  3  also delete keys whose VALUES contain an alias of the target
  4  also delete entailing keys ("mother_nationality", "languages_spoken")

Cleanest deletion of the four arms, and the lowest collateral damage.

Failure modes to expect
-----------------------
Shadow keys survive untouched, so inferential leakage stays high even though
the deletion looked surgical. And the updater keeps running: it can re-ADD
"nationality" from a later session, which is the clearest reconstruction path
in the whole study. Do not disable the updater after a forget request.

If you finish early, the retraction arm is this class plus a rule list. See
reference.py for where it would slot in.
"""

from .base import Memory, Session


class KVMemory(Memory):
    def write(self, session: Session) -> None:
        raise NotImplementedError("Wei Lun: extract facts, then ADD/UPDATE/DELETE/NOOP")

    def retrieve(self, query: str) -> str:
        raise NotImplementedError("Wei Lun: serialise relevant profile keys under max_chars")

    def forget(self, target: str, level: int) -> None:
        raise NotImplementedError("Wei Lun: drop keys, see docstring for levels")

    def dump(self) -> str:
        raise NotImplementedError("Wei Lun: serialise the whole profile")
