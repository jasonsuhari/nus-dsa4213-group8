"""Recursive summarisation approach. Owner: Ang Xuan.

A single persistent summary updated after every new turn. 
Keep dates in the summary text or temporal questions fail.

    write()     update the existing summary with new information
    retrieve()  return the current summary, truncated
    forget()    regenerate the summary without the targeted fact
    dump()      return the complete summary

forget() regenerates rather than deletes, since the fact sits inside sentences
worth keeping. Rebuild the existing summary with a negative constraint to remove 
the targeted fact while preserving non-target information. 

No raw sessions are retained. The persistent summary is used for
regeneration and retrieval.

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
