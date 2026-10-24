"""EmbeddingMemory against a fake OpenAI client, so no API key is needed."""

import hashlib
import json
import re
from types import SimpleNamespace

from dsa4213.memory import EmbeddingMemory, Session, Turn
from dsa4213.memory.embedding import IDENTIFY_PROMPT, IMPORTANCE_PROMPT

DIM = 64


def _vec(text: str) -> list[float]:
    """Hashed bag of words: texts sharing words get high cosine similarity."""
    v = [0.0] * DIM
    for w in re.findall(r"[a-z]+", text.lower()):
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
    return v


class FakeClient:
    """Importance is 8 for turns mentioning "vegetarian", else 3. Identification
    returns records containing the last word of the target."""

    def __init__(self) -> None:
        self.embeddings = SimpleNamespace(create=self._embed)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))
        self.identify_calls = 0

    def _embed(self, model, input):
        data = [SimpleNamespace(index=i, embedding=_vec(t)) for i, t in enumerate(input)]
        return SimpleNamespace(data=data)

    def _chat(self, model, messages, **kwargs):
        system, user = messages[0]["content"], messages[1]["content"]
        if system == IMPORTANCE_PROMPT:
            out = {"importance": 8 if "vegetarian" in user else 3}
        elif system == IDENTIFY_PROMPT:
            self.identify_calls += 1
            target = user.split("\n")[1]
            word = target.split()[-1].lower()
            ids = [
                int(m.group(1))
                for m in re.finditer(r"^\[(\d+)\] (.*)$", user, re.M)
                if word in m.group(2).lower()
            ]
            out = {"ids": ids + [9999]}  # 9999 is a hallucinated id, must be ignored
        else:
            raise AssertionError("unexpected prompt")
        msg = SimpleNamespace(content=json.dumps(out))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _sessions() -> list[Session]:
    return [
        Session(
            "s1",
            "2026-01-05",
            (
                Turn("user", "Hi! I am vegetarian, by the way."),
                Turn("assistant", "Noted, thanks for telling me."),
                Turn("user", "I study at NUS, Kent Ridge."),
                Turn("assistant", "Great campus."),
            ),
        ),
        Session(
            "s2",
            "2026-01-12",
            (
                Turn("user", "I really like Japanese food."),
                Turn("assistant", "Sushi or ramen?"),
            ),
        ),
    ]


def _memory(tmp_path, **kwargs) -> tuple[EmbeddingMemory, FakeClient]:
    client = FakeClient()
    mem = EmbeddingMemory(tmp_path, client=client, **kwargs)
    for s in _sessions():
        mem.write(s)
    return mem, client


def test_one_record_per_turn(tmp_path):
    mem, _ = _memory(tmp_path)
    assert [r["session_id"] for r in mem.records] == ["s1", "s1", "s2"]
    assert mem.records[0]["text"].startswith("user: Hi! I am vegetarian")
    assert "assistant: Noted" in mem.records[0]["text"]
    assert mem.records[0]["importance"] == 8


def test_lone_assistant_message_is_its_own_record(tmp_path):
    mem = EmbeddingMemory(tmp_path, client=FakeClient())
    mem.write(Session("remember-001", "2026-01-01", (Turn("assistant", "User is a cat owner"),)))
    assert len(mem.records) == 1


def test_state_persists_across_instances(tmp_path):
    mem, client = _memory(tmp_path)
    again = EmbeddingMemory(tmp_path, client=client)
    assert again.dump() == mem.dump()


def test_retrieve_ranks_relevant_first_and_respects_budget(tmp_path):
    # Relevance only. With all three weights at 1 on a tiny store, min-max scaling
    # lets recency outrank relevance, which is expected rather than a bug.
    mem, _ = _memory(tmp_path, weights=(1.0, 0.0, 0.0))
    out = mem.retrieve("where do I study at NUS Kent Ridge")
    assert out.splitlines()[0].startswith("[2026-01-05] user: I study at NUS")
    assert len(out) <= mem.max_chars

    small = EmbeddingMemory(tmp_path, max_chars=40, client=FakeClient())
    assert 0 < len(small.retrieve("anything")) <= 40


def test_retrieve_on_empty_store(tmp_path):
    assert EmbeddingMemory(tmp_path, client=FakeClient()).retrieve("hi") == ""


def test_hard_forget_removes_target_only(tmp_path):
    mem, _ = _memory(tmp_path)
    mem.forget("that I am vegetarian")

    assert "vegetarian" not in mem.dump()
    assert "vegetarian" not in mem.records_path.read_text()
    assert "Kent Ridge" in mem.dump() and "Japanese" in mem.dump()

    log = json.loads(mem.forget_log_path.read_text().splitlines()[0])
    assert log["mode"] == "hard"
    assert [r["session_id"] for r in log["removed"]] == ["s1"]


def test_soft_forget_hides_from_retrieve_but_not_dump(tmp_path):
    mem, _ = _memory(tmp_path, soft_delete=True)
    mem.forget("that I am vegetarian")

    assert "vegetarian" not in mem.retrieve("am I vegetarian")
    assert "[soft-deleted]" in mem.dump() and "vegetarian" in mem.dump()


def test_identification_runs_in_batches(tmp_path):
    mem, client = _memory(tmp_path, identify_batch_size=2)
    mem.forget("vegetarian")
    assert client.identify_calls == 2
    assert "vegetarian" not in mem.dump()


def test_write_still_works_after_forget(tmp_path):
    mem, _ = _memory(tmp_path)
    mem.forget("vegetarian")
    mem.write(Session("s3", "2026-01-19", (Turn("user", "Still vegetarian here."),)))
    assert "Still vegetarian" in mem.dump()
    assert len({r["id"] for r in mem.records}) == len(mem.records)


def test_recency_uses_session_dates(tmp_path):
    mem, _ = _memory(tmp_path)
    rec = mem._recency(mem._live)
    # s2 is newest, so it gets the top recency score.
    assert rec[-1] == 1.0 and rec[0] < 1.0


def test_retrieve_records_a_trace(tmp_path):
    mem, _ = _memory(tmp_path)
    context, trace = mem.retrieve_with_trace("where do I study")
    assert context == mem.retrieve("where do I study")
    assert mem.last_trace == trace
    assert [t["rank"] for t in trace] == list(range(1, len(trace) + 1))
    assert all(t["text"] in context for t in trace)
    assert {"score", "relevance", "recency", "importance", "id"} <= set(trace[0])


def test_forget_logs_surviving_near_duplicates(tmp_path):
    mem, _ = _memory(tmp_path, duplicate_threshold=0.5)
    # Same wording as the target turn but without the word the fake identifier matches.
    mem.write(Session("s3", "2026-01-19", (Turn("user", "Hi! I am, by the way."),)))
    mem.forget("that I am vegetarian")

    log = json.loads(mem.forget_log_path.read_text().splitlines()[0])
    (entry,) = log["near_duplicates"]
    top = entry["survivors"][0]
    assert top["session_id"] == "s3" and top["likely_duplicate"]
    assert top["cosine"] >= entry["survivors"][-1]["cosine"]
    assert log["likely_duplicates_left"] == 1


def test_ids_not_reused_after_deleting_newest(tmp_path):
    mem, client = _memory(tmp_path)
    newest = mem.records[-1]["id"]
    mem.forget("japanese")  # removes the newest record
    assert all(r["id"] != newest for r in mem.records)

    again = EmbeddingMemory(tmp_path, client=client)
    again.write(Session("s3", "2026-01-19", (Turn("user", "New turn."),)))
    assert again.records[-1]["id"] == newest + 1


def test_recency_is_raw_decay(tmp_path):
    mem, _ = _memory(tmp_path)
    rec = mem._recency(mem._live)
    # s1 is 7 days (168 hours) older than s2, the newest.
    assert rec[0] == mem.decay_per_hour**168
