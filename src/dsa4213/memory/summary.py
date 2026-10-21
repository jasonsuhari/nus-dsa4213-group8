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

from pathlib import Path

from openai import OpenAI

from .base import DEFAULT_MAX_CHARS, Memory, Session

# tells llm how to update the memory summary with new information from the conversation 
WRITE_PROMPT = """You maintain a persistent natural-language summary of a user's
information across conversations.

Update the existing summary using the new conversation information.

Instructions:
- Incorporate new relevant information about the user.
- Preserve existing information that is still relevant and has not been contradicted.
- Update outdated or contradicted information when the user provides corrections.
- Keep the summary concise, accurate, and faithful to the provided information.
- Preserve relevant dates and temporal relationships.
- Distinguish conversation dates from dates when events actually occurred.
- Do not invent facts or infer information that was not provided.
- Avoid unnecessary repetition without omitting important details.
- Return only the updated summary, without explanations or additional formatting.
"""

FORGET_PROMPT = """You maintain a persistent natural-language summary of a user's
information across conversations.

Revise the existing summary to remove the information targeted by the
user's forgetting request.

Instructions:
- Remove the targeted fact and direct paraphrases of the targeted fact.
- Preserve all non-target information, including related facts that are not
  explicitly targeted by the forgetting request.
- Do not remove information solely because the targeted fact could be
  inferred from it.
- Preserve relevant dates and temporal relationships for retained information.
- Do not invent new facts or infer information that was not provided.
- Keep the revised summary concise, accurate, and faithful to retained information.
- Avoid unnecessary changes to unrelated information.
- Return only the revised summary, without explanations or additional formatting.
"""


class SummaryMemory(Memory):
    def __init__(self, 
                 path: Path, 
                 max_chars: int = DEFAULT_MAX_CHARS, 
                 model: str = "gpt-4o-mini", 
                 client: OpenAI | None = None) -> None:
        super().__init__(path, max_chars) # call base constructor to set path and max_chars

        self.model = model
        self.client = client or OpenAI() # use provided client or create a new one

        self.summary_path = self.path / "summary.txt"

        if self.summary_path.exists(): # check if the summary is saved 
            self.summary = self.summary_path.read_text(encoding="utf-8")
        else:
            self.summary = ""

    # helper function for to save summary
    def save_summary(self) -> None:
        self.summary_path.write_text(self.summary, encoding="utf-8")

    def write(self, session: Session) -> None:
        if not session.turns:
            return  # No new information to process

        # send the information to llm to update the summary
        response = self.client.chat.completions.create(
            model = self.model, 
            messages = [
                {
                    "role": "system",
                    "content": WRITE_PROMPT
                }, 
                {
                    "role": "user",
                    "content": (
                        f"Existing summary:\n{self.summary or '(empty)'}\n\n"
                        f"Conversation date: {session.date}\n"
                        f"New conversation information:\n{session.transcript}"
                    )
                }
            ], 
            temperature = 0
        )

        # select the first generated response, then get the generated summary
        updated_summary = (response.choices[0].message.content or "").strip()

        # this is to prevent the case where the llm returns an empty string, 
        # which would overwrite the existing summary with nothing
        if not updated_summary:
            raise ValueError("LLM returned an empty updated summary.")

        if updated_summary != self.summary:
            self.summary = updated_summary
            self.save_summary()
        

    def retrieve(self, query: str) -> str:
        return self.summary[:self.max_chars]

    def forget(self, target: str) -> None:
        if not target.strip() or not self.summary:
            return # nothing to forget

        # call the llm
        response = self.client.chat.completions.create(
            model = self.model, 
            messages = [
                {
                    "role": "system",
                    "content": FORGET_PROMPT
                }, 
                {
                    "role": "user",
                    "content": (
                        f"Existing summary:\n{self.summary}\n\n"
                        f"Targeted fact to forget:\n{target}"
                    )
                }
            ], 
            temperature = 0
        )

        # select the first generated response, then get the generated summary
        updated_summary = response.choices[0].message.content

        # check for invalid response
        if updated_summary is None:
            raise ValueError("LLM returned no revised summary.")

        updated_summary = updated_summary.strip()

        if updated_summary != self.summary:
            self.summary = updated_summary
            self.save_summary()


    def dump(self) -> str:
        return self.summary
