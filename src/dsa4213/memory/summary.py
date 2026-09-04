"""Periodic summarisation arm. Owner: Ang Xuan.

What to build
-------------
Hierarchical, not flat. A per-session summary, plus a rolling global summary
regenerated every N sessions. Keep dates inside the summary text or every
temporal question collapses.

  write()    summarise the session; every N sessions, regenerate the global one.
  retrieve() return the global summary plus the most relevant session summaries,
             truncated to max_chars.
  forget()   see below.
  dump()     every summary, session-level and global.

Decision you must make and write down
-------------------------------------
Does this arm keep the raw sessions? It needs them to regenerate a summary
with a fact removed. But if raw text is kept and never scrubbed, the fact
survives at source and the comparison against the other arms is unfair.

Recommended: keep raw sessions for regeneration only, never retrieve from
them, and have forget() scrub the raw store too. State the choice in the report.

forget(target, level)
---------------------
You cannot delete. The fact is a clause inside a sentence you want to keep.

  1  find summaries stating the fact, regenerate each from source with a
     negative constraint ("do not mention the user's nationality")
  2  also catch paraphrases when deciding which summaries are affected
  3  also regenerate any summary above the similarity threshold
  4  also strip entailing evidence during regeneration

Always regenerate the global summary afterwards.

Failure modes to expect
-----------------------
Paraphrase leak: the rewrite says "grew up overseas" instead of the fact.
Collateral rewrite: regeneration is nondeterministic, so unrelated details in
the same summary drift or vanish. Expect the highest collateral damage of the
four arms. Both are results.
"""

from .base import Memory, Session


class SummaryMemory(Memory):
    def write(self, session: Session) -> None:
        raise NotImplementedError("Ang Xuan: summarise session, refresh global summary every N")

    def retrieve(self, query: str) -> str:
        raise NotImplementedError("Ang Xuan: global + relevant session summaries, cap at max_chars")

    def forget(self, target: str, level: int) -> None:
        raise NotImplementedError("Ang Xuan: regenerate affected summaries, see docstring")

    def dump(self) -> str:
        raise NotImplementedError("Ang Xuan: return all summaries")
