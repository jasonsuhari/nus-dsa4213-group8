"""Periodic summarisation arm. Owner: Ang Xuan.

Hierarchical. Per-session summaries plus a rolling global summary regenerated
every N sessions. Keep dates in the summary text or temporal questions fail.

    write()     summarise the session, refresh the global summary every N
    retrieve()  global summary plus relevant session summaries, truncated
    dump()      every summary

forget(target, level) regenerates rather than deletes, since the fact sits
inside sentences worth keeping.

    1  regenerate affected summaries with a negative constraint
    2  catch paraphrases when choosing which summaries are affected
    3  also regenerate anything above the similarity threshold
    4  also strip entailing evidence

Open decision to record in the report: whether this arm keeps raw sessions.
It needs them to regenerate, but unscrubbed raw text means the fact survives at
source and the comparison is unfair. Suggested answer is to keep them for
regeneration only, never retrieve from them, and scrub them in forget().

Expect paraphrase leak ("grew up overseas") and drift in unrelated details,
since regeneration is nondeterministic.
"""

from .base import Memory, Session


class SummaryMemory(Memory):
    def write(self, session: Session) -> None:
        raise NotImplementedError("summarise session, refresh global summary every N")

    def retrieve(self, query: str) -> str:
        raise NotImplementedError("global plus relevant summaries, cap at self.max_chars")

    def forget(self, target: str, level: int) -> None:
        raise NotImplementedError("regenerate affected summaries, see module docstring")

    def dump(self) -> str:
        raise NotImplementedError("return all summaries")
