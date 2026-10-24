"""Embedding retrieval approach. Owner: Jordan.

Follows proposal Section 4.2.1, adapting the weighted retrieval of Park et al. (2023).

    write()     one record per conversational turn (a user message plus the assistant
                reply that follows it). Each record is embedded, timestamped with the
                session date, and given an LLM importance score in [1, 10]. Records are
                only appended, never merged, so near-duplicates can accumulate.
    retrieve()  score every live record by relevance + recency + importance, each
                in [0, 1] (relevance min-max scaled, recency as exponential decay,
                importance rescaled from 1-10), take the top k, and serialise them in rank order
                until self.max_chars is reached.
    forget()    an identification call reads every live record (in batches) and returns
                the ones that state the target. Hard delete removes them and rebuilds
                the store; the soft-delete variant only flags them and skips them at
                retrieval time, the way many production vector databases do.
    dump()      every record still physically stored, ordered by timestamp, with no
                scoring. This is the ground-truth check on forget().

Evidence for explaining *why* forgetting succeeds or fails:

    retrieve_with_trace()  the context plus which records produced it, with each
                           record's relevance, recency and importance. When a probe
                           leaks, this shows whether a surviving near-duplicate or a
                           correctly retained related record was in the context.
                           retrieve() records the same trace in self.last_trace.
    forget_log.jsonl       one line per forget() call: the removed records (collateral
                           damage when a removed turn also carried a retained fact) and,
                           for each removed record, the most similar records that
                           survived (redundancy the identification call missed).

State under self.path:
    records.jsonl     one JSON object per record, embedding included
    forget_log.jsonl  one line per forget() call
    meta.json         the next record id, so ids are never reused after a delete
"""

import json
import math
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from .base import DEFAULT_MAX_CHARS, Memory, Session

DEFAULT_EMBED_MODEL = "text-embedding-3-small"

# Raw cosine similarity at or above which a surviving record is flagged as a likely
# near-duplicate of a removed one. Tune it against the logged values, which are kept
# whatever the threshold.
DEFAULT_DUPLICATE_THRESHOLD = 0.8

# Park et al. (2023) decay recency by 0.995 per game hour.
DEFAULT_DECAY_PER_HOUR = 0.995

# Adapted from the importance ("poignancy") prompt in Park et al. (2023).
IMPORTANCE_PROMPT = """You rate how important a conversation turn is for an assistant's
long-term memory of the user, on a scale from 1 to 10.

1 is purely mundane small talk with nothing worth remembering (for example, greetings
or thanks). 10 is extremely important lasting information about the user (for example,
core personal facts, strong preferences, or major life events).

Return JSON of the form {"importance": <integer from 1 to 10>} and nothing else."""

IDENTIFY_PROMPT = """You manage the long-term memory of a personal assistant. The memory
is a list of numbered records, each a past conversation turn.

The user has asked the assistant to forget a specific fact. Return the ids of every
record that states the targeted fact or a direct paraphrase of it, including records
where it is mentioned alongside other information.

Do not return a record only because the targeted fact could be inferred from it, or
because it is on a related topic. If no record states the fact, return an empty list.

Return JSON of the form {"ids": [<integer>, ...]} and nothing else."""


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _minmax(values: list[float]) -> list[float]:
    """Scale to [0, 1]. Ties (including a single candidate) all map to 1."""
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def _parse_date(date: str) -> datetime | None:
    try:
        return datetime.fromisoformat(date)
    except ValueError:
        return None


class EmbeddingMemory(Memory):
    def __init__(
        self,
        path: Path,
        max_chars: int = DEFAULT_MAX_CHARS,
        model: str = "gpt-4o-mini",
        embed_model: str = DEFAULT_EMBED_MODEL,
        client: OpenAI | None = None,
        top_k: int = 10,
        decay_per_hour: float = DEFAULT_DECAY_PER_HOUR,
        weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
        soft_delete: bool = False,
        identify_batch_size: int = 50,
        duplicate_threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
        duplicates_per_record: int = 3,
    ) -> None:
        super().__init__(path, max_chars)
        self.model = model
        self.embed_model = embed_model
        self.client = client or OpenAI()
        self.top_k = top_k
        self.decay_per_hour = decay_per_hour
        # (relevance, recency, importance), all 1 in the proposal for now.
        self.weights = weights
        self.soft_delete = soft_delete
        self.identify_batch_size = identify_batch_size
        self.duplicate_threshold = duplicate_threshold
        self.duplicates_per_record = duplicates_per_record

        self.records_path = self.path / "records.jsonl"
        self.forget_log_path = self.path / "forget_log.jsonl"
        self.meta_path = self.path / "meta.json"
        self.records: list[dict] = self._load()
        # Records behind the most recent retrieve() call, for saving with each answer.
        self.last_trace: list[dict] = []

    # ---- storage -------------------------------------------------------------

    def _load(self) -> list[dict]:
        if not self.records_path.exists():
            return []
        lines = self.records_path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def _save(self) -> None:
        # Rewrites the whole file, so a hard delete leaves nothing behind on disk.
        with self.records_path.open("w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(r) + "\n")

    def _next_id(self) -> int:
        """A counter kept on disk, so a deleted record's id is never given to a new one.

        Without it, deleting the newest record and writing again would reuse its id, and
        ids in forget_log.jsonl or a saved trace could point at the wrong record.
        """
        stored = -1
        if self.meta_path.exists():
            stored = json.loads(self.meta_path.read_text(encoding="utf-8"))["next_id"] - 1
        return max(stored, max((r["id"] for r in self.records), default=-1)) + 1

    def _set_next_id(self, next_id: int) -> None:
        self.meta_path.write_text(json.dumps({"next_id": next_id}), encoding="utf-8")

    @property
    def _live(self) -> list[dict]:
        return [r for r in self.records if not r["deleted"]]

    # ---- LLM calls -----------------------------------------------------------

    def _embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.embed_model, input=texts)
        return [d.embedding for d in sorted(response.data, key=lambda d: d.index)]

    def _json_call(self, system: str, user: str) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = response.choices[0].message.content
        try:
            parsed = json.loads(content or "")
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned no valid JSON: {content!r:.200}") from e
        if not isinstance(parsed, dict):
            raise ValueError(f"LLM returned JSON that is not an object: {content!r:.200}")
        return parsed

    def _importance(self, text: str) -> int:
        raw = self._json_call(IMPORTANCE_PROMPT, text).get("importance", 1)
        try:
            score = int(raw)
        except (TypeError, ValueError):
            score = 1
        return min(10, max(1, score))

    # ---- turn grouping -------------------------------------------------------

    @staticmethod
    def _units(session: Session) -> list[str]:
        """Group messages into turns: a user message plus the replies that follow it.

        Messages before the first user message (for example a fact saved through the
        agent's remember tool, which arrives as a lone assistant message) form their
        own unit.
        """
        units: list[list[str]] = []
        for t in session.turns:
            line = f"{t.role}: {t.text}"
            if t.role == "user" or not units:
                units.append([line])
            else:
                units[-1].append(line)
        return ["\n".join(u) for u in units]

    # ---- contract ------------------------------------------------------------

    def write(self, session: Session) -> None:
        units = self._units(session)
        if not units:
            return
        embeddings = self._embed(units)
        start = self._next_id()
        for i, (text, emb) in enumerate(zip(units, embeddings, strict=True)):
            self.records.append(
                {
                    "id": start + i,
                    "session_id": session.session_id,
                    "date": session.date,
                    "turn_index": i,
                    "text": text,
                    "importance": self._importance(text),
                    "embedding": emb,
                    "deleted": False,
                }
            )
        self._set_next_id(start + len(units))
        self._save()

    def _recency(self, live: list[dict]) -> list[float]:
        """Recency(tau_i) = decay ** hours since the newest record in the store.

        As in the proposal's formula, this is not min-max scaled: exponential decay
        already lies in [0, 1], with the newest record at 1.

        Measured against the newest stored date rather than the wall clock, so scores
        depend only on the replayed history and are the same on every run.
        """
        dates = [_parse_date(r["date"]) for r in live]
        if all(d is not None for d in dates):
            newest = max(dates)
            hours = [(newest - d).total_seconds() / 3600 for d in dates]
        else:
            # Unparseable dates: fall back to insertion order, one hour per record.
            newest_id = max(r["id"] for r in live)
            hours = [float(newest_id - r["id"]) for r in live]
        return [self.decay_per_hour**h for h in hours]

    def _ranked(self, query: str) -> list[dict]:
        """Every live record with its score and the three components, highest first."""
        live = self._live
        if not live:
            return []
        (q,) = self._embed([query])
        rel = _minmax([_cosine(q, r["embedding"]) for r in live])
        rec = self._recency(live)
        # Fixed rescaling of the 1-10 rating, as in the proposal's formula.
        imp = [(r["importance"] - 1) / 9 for r in live]
        w_rel, w_rec, w_imp = self.weights
        ranked = [
            {
                "score": w_rel * a + w_rec * b + w_imp * c,
                "relevance": a,
                "recency": b,
                "importance": c,
                "record": r,
            }
            for a, b, c, r in zip(rel, rec, imp, live, strict=True)
        ]
        # Ties broken by recency of insertion, so the order is deterministic.
        ranked.sort(key=lambda x: (x["score"], x["record"]["id"]), reverse=True)
        return ranked

    def scores(self, query: str) -> list[tuple[float, dict]]:
        """Every live record with its retrieval score, highest first."""
        return [(x["score"], x["record"]) for x in self._ranked(query)]

    @staticmethod
    def _format(r: dict) -> str:
        return f"[{r['date']}] {r['text']}"

    def retrieve_with_trace(self, query: str) -> tuple[str, list[dict]]:
        """The context retrieve() returns, plus the records that made it in, in rank order.

        Each trace entry has the record's id, session, rank, score and the three
        components, its text, and whether it was cut short by the character budget.
        Save the trace with each probe's answer so leakage can be traced to a record.
        """
        parts: list[str] = []
        trace: list[dict] = []
        used = 0
        for rank, x in enumerate(self._ranked(query)[: self.top_k], start=1):
            r = x["record"]
            block = self._format(r)
            extra = len(block) + (2 if parts else 0)
            truncated = False
            if used + extra > self.max_chars:
                if parts:
                    break
                # Never return nothing just because the best record is long.
                block, truncated = block[: self.max_chars], True
            parts.append(block)
            used += extra
            trace.append(
                {
                    "rank": rank,
                    "id": r["id"],
                    "session_id": r["session_id"],
                    "score": x["score"],
                    "relevance": x["relevance"],
                    "recency": x["recency"],
                    "importance": x["importance"],
                    "truncated": truncated,
                    "text": r["text"],
                }
            )
            if truncated:
                break
        return "\n\n".join(parts), trace

    def retrieve(self, query: str) -> str:
        context, self.last_trace = self.retrieve_with_trace(query)
        return context

    def identify(self, target: str) -> list[int]:
        """Ids of live records the identification call says state the target."""
        live = self._live
        found: list[int] = []
        for start in range(0, len(live), self.identify_batch_size):
            batch = live[start : start + self.identify_batch_size]
            valid = {r["id"] for r in batch}
            listing = "\n\n".join(f"[{r['id']}] {self._format(r)}" for r in batch)
            out = self._json_call(
                IDENTIFY_PROMPT, f"Fact to forget:\n{target}\n\nRecords:\n{listing}"
            )
            for raw in out.get("ids", []) or []:
                try:
                    i = int(raw)
                except (TypeError, ValueError):
                    continue
                # Ignore hallucinated ids so a bad call cannot delete outside the batch.
                if i in valid and i not in found:
                    found.append(i)
        return found

    def near_duplicates(self, removed: list[dict], survivors: list[dict]) -> list[dict]:
        """For each removed record, the most similar surviving records by raw cosine.

        Logged whatever the threshold, so the cut-off can be chosen afterwards; records
        at or above duplicate_threshold are flagged as likely near-duplicates.
        """
        out = []
        for r in removed:
            sims = sorted(
                ((_cosine(r["embedding"], s["embedding"]), s) for s in survivors),
                key=lambda x: (x[0], x[1]["id"]),
                reverse=True,
            )[: self.duplicates_per_record]
            out.append(
                {
                    "removed_id": r["id"],
                    "survivors": [
                        {
                            "id": s["id"],
                            "session_id": s["session_id"],
                            "cosine": round(c, 4),
                            "likely_duplicate": c >= self.duplicate_threshold,
                            "text": s["text"],
                        }
                        for c, s in sims
                    ],
                }
            )
        return out

    def forget(self, target: str) -> None:
        if not target.strip() or not self._live:
            return
        ids = set(self.identify(target))
        removed = [r for r in self.records if r["id"] in ids]
        # Computed before deletion, while the removed records' embeddings still exist.
        duplicates = self.near_duplicates(removed, [r for r in self._live if r["id"] not in ids])
        if self.soft_delete:
            for r in removed:
                r["deleted"] = True
        else:
            self.records = [r for r in self.records if r["id"] not in ids]
        self._save()
        log = {
            "target": target,
            "mode": "soft" if self.soft_delete else "hard",
            "removed": [
                {"id": r["id"], "session_id": r["session_id"], "text": r["text"]} for r in removed
            ],
            "near_duplicates": duplicates,
            # Distinct survivors, since one record can resemble several removed ones.
            "likely_duplicates_left": len(
                {s["id"] for d in duplicates for s in d["survivors"] if s["likely_duplicate"]}
            ),
        }
        with self.forget_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log) + "\n")

    def dump(self) -> str:
        """Every record physically stored. Soft-deleted ones are still here, flagged."""
        ordered = sorted(
            self.records,
            key=lambda r: (_parse_date(r["date"]) or datetime.min, r["id"]),
        )
        return "\n\n".join(
            ("[soft-deleted] " if r["deleted"] else "") + self._format(r) for r in ordered
        )
