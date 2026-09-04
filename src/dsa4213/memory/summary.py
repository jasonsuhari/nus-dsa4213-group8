"""Periodic summarisation approach. Owner: Ang Xuan.

Hierarchical. Per-session summaries plus a rolling global summary regenerated
every N sessions. Keep dates in the summary text or temporal questions fail.

    write()     summarise the session, refresh the global summary every N
    retrieve()  global summary plus relevant session summaries, truncated
    dump()      every summary

forget() regenerates rather than deletes, since the fact sits inside sentences
worth keeping. Find the affected summaries, rebuild each from source with a
negative constraint, then rebuild the global summary.

Keep raw sessions for regeneration only. Never retrieve from them, and scrub
them in forget(), or the fact survives at source and the comparison is unfair.

Expect paraphrase leak ("grew up overseas") and drift in unrelated details,
since regeneration is nondeterministic.
"""

from .base import Memory, Session


class SummaryMemory(Memory):
    def write(self, session: Session) -> None:
        raise NotImplementedError("summarise session, refresh global summary every N")

    def retrieve(self, query: str) -> str:
        raise NotImplementedError("global plus relevant summaries, cap at self.max_chars")

    def forget(self, target: str) -> None:
        raise NotImplementedError("regenerate affected summaries, see module docstring")

    def dump(self) -> str:
        raise NotImplementedError("return all summaries")
