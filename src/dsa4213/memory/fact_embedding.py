"""Fact-level embedding retrieval: the turn-level memory with finer cards. Owner: Jordan.

Answers the presentation feedback that turn-level records delete every fact said in the
same turn as the target. Three changes to EmbeddingMemory, everything else inherited:

    write()   one LLM call per turn splits it into atomic, standalone facts, each with an
              importance rating. Earlier turns of the session are passed as context so
              pronouns resolve ("she" -> "the user's daughter Ava"). Facts the assistant
              repeats back are extracted too. One card per fact.
    forget()  facts that reveal the target are either deleted (nothing else worth keeping)
              or redacted: rewritten without the target and re-embedded, so a fact such as
              "Ava has ballet on Saturdays" survives as "The user's daughter has ballet on
              Saturdays" when the name is forgotten.
              A second pass then checks what is left and removes anything still revealing
              the target, aimed at copies the first pass missed.
    top_k     20 instead of 10, since a fact is a fraction of a turn's length. The
              character budget, and so the comparison with the other memories, is unchanged.

Retrieval scoring (relevance + recency + importance) is the same as EmbeddingMemory.
forget_log.jsonl records every deletion and redaction, before and after, per pass.
"""

import json
from pathlib import Path

from .base import DEFAULT_MAX_CHARS, Session
from .embedding import EmbeddingMemory

EXTRACT_PROMPT = """You turn one turn of a conversation between a user and a personal
assistant into the facts the assistant should remember about the user and the people in
the user's life.

Rules:
- One claim per fact. Split compound sentences: "I have asthma, so I quit the swim team"
  becomes "The user has asthma." and "The user quit the swim team."
- Each fact must stand alone. Replace pronouns and vague references with who or what they
  refer to, using names when known, so "She is allergic to peanuts" becomes "The user's
  daughter Ava is allergic to peanuts." Use the earlier conversation only to resolve
  references; do not extract facts from it.
- Keep names, numbers, dates, places and other specifics exactly as written.
- Include facts the assistant states or repeats about the user, not only the user's words.
- Skip advice, suggestions, opinions about the world, general knowledge and small talk.
- If the turn holds no such facts, return an empty list.

Rate each fact's importance for long-term memory from 1 (trivial) to 10 (core, lasting
information about the user, such as identity, health, family or strong preferences).

Return JSON of the form {"facts": [{"fact": "<sentence>", "importance": <1-10>}]} and
nothing else."""

FORGET_PROMPT = """You manage the long-term memory of a personal assistant: a list of
numbered facts about the user. The user has asked the assistant to forget one thing.

Find every fact that states the thing to forget, paraphrases it, or contains it (for
example a fact that uses a name the user asked to forget). For each, choose:
- "delete" if the fact is only about the thing to forget, or nothing else in it is worth
  keeping.
- "redact" if the fact also carries other information worth keeping. Rewrite it so the
  thing to forget is removed, keeping everything else exactly as written.

Leave out facts that do not contain the thing to forget, even if they are on a related
topic or the thing could be guessed from them. If no fact contains it, return empty lists.

Return JSON of the form
{"delete": [<id>, ...], "redact": [{"id": <id>, "text": "<rewritten fact>"}, ...]}
and nothing else."""

# Earlier turns passed to the extractor for resolving references, newest last.
CONTEXT_TURNS = 3


class FactEmbeddingMemory(EmbeddingMemory):
    def __init__(
        self,
        path: Path,
        max_chars: int = DEFAULT_MAX_CHARS,
        top_k: int = 20,
        forget_passes: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(path, max_chars, top_k=top_k, **kwargs)
        self.forget_passes = forget_passes

    # ---- write ---------------------------------------------------------------

    def _extract(self, unit: str, context: list[str]) -> list[tuple[str, int]]:
        """Atomic facts in one turn, with importance. Falls back to the whole turn.

        One retry on a malformed reply; if that fails too, the turn is stored as one
        card, as the turn-level memory would, so a bad reply never loses a turn.
        """
        earlier = "\n\n".join(context[-CONTEXT_TURNS:]) or "(none)"
        user = f"Earlier in this conversation:\n{earlier}\n\nTurn to extract from:\n{unit}"
        for _ in range(2):
            try:
                out = self._json_call(EXTRACT_PROMPT, user)
            except ValueError:
                continue
            raw = out.get("facts")
            if not isinstance(raw, list):
                continue
            facts = []
            for item in raw:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("fact", "")).strip()
                if not text:
                    continue
                try:
                    importance = min(10, max(1, int(item.get("importance", 1))))
                except (TypeError, ValueError):
                    importance = 1
                facts.append((text, importance))
            return facts
        return [(unit, self._importance(unit))]

    def write(self, session: Session) -> None:
        units = self._units(session)
        cards: list[tuple[int, str, int]] = []
        for i, unit in enumerate(units):
            cards += [(i, text, imp) for text, imp in self._extract(unit, units[:i])]
        if not cards:
            return
        embeddings = self._embed([text for _, text, _ in cards])
        start = self._next_id()
        for n, ((turn, text, imp), emb) in enumerate(zip(cards, embeddings, strict=True)):
            self.records.append(
                {
                    "id": start + n,
                    "session_id": session.session_id,
                    "date": session.date,
                    "turn_index": turn,
                    "text": text,
                    "importance": imp,
                    "embedding": emb,
                    "deleted": False,
                }
            )
        self._set_next_id(start + len(cards))
        self._save()

    # ---- forget --------------------------------------------------------------

    def _decide(self, target: str) -> tuple[set[int], dict[int, str]]:
        """Ids to delete and {id: rewritten text} to redact, over every live fact."""
        delete: set[int] = set()
        redact: dict[int, str] = {}
        live = self._live
        for start in range(0, len(live), self.identify_batch_size):
            batch = live[start : start + self.identify_batch_size]
            valid = {r["id"] for r in batch}
            listing = "\n".join(f"[{r['id']}] {r['text']}" for r in batch)
            out = self._json_call(FORGET_PROMPT, f"Thing to forget:\n{target}\n\nFacts:\n{listing}")
            for raw in out.get("delete", []) or []:
                try:
                    i = int(raw)
                except (TypeError, ValueError):
                    continue
                if i in valid:  # ignore ids the model made up
                    delete.add(i)
            for item in out.get("redact", []) or []:
                if not isinstance(item, dict):
                    continue
                try:
                    i = int(item.get("id"))
                except (TypeError, ValueError):
                    continue
                text = str(item.get("text", "")).strip()
                if i in valid and i not in delete:
                    if text:
                        redact[i] = text
                    else:
                        delete.add(i)  # an empty rewrite means nothing worth keeping
        return delete, redact

    def forget(self, target: str) -> None:
        if not target.strip() or not self._live:
            return
        passes = []
        for n in range(1, self.forget_passes + 1):
            delete, redact = self._decide(target)
            if not delete and not redact:
                break
            by_id = {r["id"]: r for r in self.records}
            removed = [by_id[i] for i in sorted(delete)]
            rewritten = []
            if redact:
                ids = sorted(redact)
                embeddings = self._embed([redact[i] for i in ids])
                for i, emb in zip(ids, embeddings, strict=True):
                    r = by_id[i]
                    rewritten.append({"id": i, "before": r["text"], "after": redact[i]})
                    r["text"], r["embedding"] = redact[i], emb
            if self.soft_delete:
                for r in removed:
                    r["deleted"] = True
            else:
                self.records = [r for r in self.records if r["id"] not in delete]
            passes.append(
                {
                    "pass": n,
                    "deleted": [{"id": r["id"], "text": r["text"]} for r in removed],
                    "redacted": rewritten,
                }
            )
        self._save()
        log = {"target": target, "mode": "soft" if self.soft_delete else "hard", "passes": passes}
        with self.forget_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log) + "\n")
