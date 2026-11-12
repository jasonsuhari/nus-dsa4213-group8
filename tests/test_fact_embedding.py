"""FactEmbeddingMemory against a fake OpenAI client, so no API key is needed."""

import hashlib
import json
import re
from types import SimpleNamespace

from dsa4213.memory import FactEmbeddingMemory, Session, Turn
from dsa4213.memory.fact_embedding import EXTRACT_PROMPT, FORGET_PROMPT

DIM = 64


def _vec(text: str) -> list[float]:
    v = [0.0] * DIM
    for w in re.findall(r"[a-z]+", text.lower()):
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
    return v


def _reply(obj) -> SimpleNamespace:
    content = obj if isinstance(obj, str) else json.dumps(obj)
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class FakeClient:
    """Extraction splits the user's line on ", so " and ", and ". Importance is 8 for
    facts mentioning Ava, else 3. A turn containing BROKEN gets malformed JSON back.

    Forgetting matches the last word of the target. Facts that also mention ballet are
    redacted (the word replaced by "the daughter"), others deleted, and a made-up id is
    always added. With miss_first, the first forget call skips facts starting "Still".
    """

    def __init__(self, miss_first: bool = False) -> None:
        self.embeddings = SimpleNamespace(create=self._embed)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))
        self.miss_first = miss_first
        self.forget_calls = 0
        self.extract_inputs: list[str] = []

    def _embed(self, model, input):
        data = [SimpleNamespace(index=i, embedding=_vec(t)) for i, t in enumerate(input)]
        return SimpleNamespace(data=data)

    def _chat(self, model, messages, **kwargs):
        system, user = messages[0]["content"], messages[1]["content"]
        if system == EXTRACT_PROMPT:
            self.extract_inputs.append(user)
            turn = user.split("Turn to extract from:\n", 1)[1]
            if "BROKEN" in turn:
                return _reply("not json")
            said = re.search(r"^user: (.*)$", turn, re.M)
            parts = re.split(r", so |, and ", said.group(1)) if said else []
            facts = [p.strip(" .") for p in parts if p.strip(" .") and "thanks" not in p.lower()]
            return _reply(
                {"facts": [{"fact": f, "importance": 8 if "Ava" in f else 3} for f in facts]}
            )
        if system == FORGET_PROMPT:
            self.forget_calls += 1
            word = user.split("\n")[1].strip(" .").split()[-1]
            delete, redact = [999], []
            for m in re.finditer(r"^\[(\d+)\] (.*)$", user, re.M):
                i, text = int(m.group(1)), m.group(2)
                if word.lower() not in text.lower():
                    continue
                if self.miss_first and self.forget_calls == 1 and text.startswith("Still"):
                    continue
                if "ballet" in text:
                    redact.append({"id": i, "text": text.replace(word, "the daughter")})
                else:
                    delete.append(i)
            return _reply({"delete": delete, "redact": redact})
        raise AssertionError("unexpected prompt")


def _session(sid: str, date: str, *lines: str) -> Session:
    turns = []
    for line in lines:
        turns += [Turn("user", line), Turn("assistant", "Noted.")]
    return Session(sid, date, tuple(turns))


def _memory(tmp_path, client=None, **kwargs) -> FactEmbeddingMemory:
    mem = FactEmbeddingMemory(tmp_path, client=client or FakeClient(), **kwargs)
    mem.write(
        _session(
            "s1",
            "2026-01-05",
            "I have asthma, so I quit the swim team, and I joined chess club",
            "My daughter is Ava",
            "Ava has ballet on Saturdays",
        )
    )
    return mem


def _texts(mem) -> list[str]:
    return [r["text"] for r in mem.records]


def test_compound_turn_becomes_one_card_per_fact(tmp_path):
    mem = _memory(tmp_path)
    assert _texts(mem)[:3] == ["I have asthma", "I quit the swim team", "I joined chess club"]
    assert {r["turn_index"] for r in mem.records[:3]} == {0}
    assert mem.records[3]["importance"] == 8


def test_forgetting_one_fact_keeps_its_neighbours(tmp_path):
    mem = _memory(tmp_path)
    mem.forget("that I have asthma")
    assert "I have asthma" not in mem.dump()
    assert "I quit the swim team" in mem.dump() and "I joined chess club" in mem.dump()


def test_redaction_keeps_the_rest_of_the_fact(tmp_path):
    mem = _memory(tmp_path)
    before = next(r for r in mem.records if "ballet" in r["text"])
    old_embedding = before["embedding"]
    mem.forget("my daughter's name Ava")

    assert "Ava" not in mem.dump()
    ballet = next(r for r in mem.records if "ballet" in r["text"])
    assert ballet["text"] == "the daughter has ballet on Saturdays"
    assert ballet["embedding"] != old_embedding  # re-embedded after the rewrite

    log = json.loads(mem.forget_log_path.read_text().splitlines()[0])
    first = log["passes"][0]
    assert [d["text"] for d in first["deleted"]] == ["My daughter is Ava"]
    assert first["redacted"][0]["before"] == "Ava has ballet on Saturdays"


def test_made_up_ids_are_ignored(tmp_path):
    mem = _memory(tmp_path)
    n = len(mem.records)
    mem.forget("that I have asthma")
    assert len(mem.records) == n - 1


def test_second_pass_catches_a_missed_copy(tmp_path):
    client = FakeClient(miss_first=True)
    mem = _memory(tmp_path, client=client)
    mem.write(_session("s2", "2026-01-12", "Still asthma here"))
    mem.forget("that I have asthma")
    assert "asthma" not in mem.dump()
    passes = json.loads(mem.forget_log_path.read_text().splitlines()[0])["passes"]
    assert [d["text"] for d in passes[1]["deleted"]] == ["Still asthma here"]


def test_single_pass_leaves_the_missed_copy(tmp_path):
    mem = _memory(tmp_path, client=FakeClient(miss_first=True), forget_passes=1)
    mem.write(_session("s2", "2026-01-12", "Still asthma here"))
    mem.forget("that I have asthma")
    assert "Still asthma here" in mem.dump()


def test_malformed_extraction_falls_back_to_the_whole_turn(tmp_path):
    mem = FactEmbeddingMemory(tmp_path, client=FakeClient())
    with_importance = {"importance": 5}
    mem._importance = lambda text: with_importance["importance"]
    mem.write(_session("s1", "2026-01-05", "BROKEN reply here"))
    (record,) = mem.records
    assert record["text"] == "user: BROKEN reply here\nassistant: Noted."
    assert record["importance"] == 5


def test_turn_without_facts_stores_nothing(tmp_path):
    mem = FactEmbeddingMemory(tmp_path, client=FakeClient())
    mem.write(_session("s1", "2026-01-05", "thanks"))
    assert mem.records == []


def test_earlier_turns_are_context_for_extraction(tmp_path):
    client = FakeClient()
    _memory(tmp_path, client=client)
    assert "(none)" in client.extract_inputs[0]
    assert "My daughter is Ava" in client.extract_inputs[2].split("Turn to extract from:")[0]


def test_retrieve_respects_budget_and_wider_top_k(tmp_path):
    mem = _memory(tmp_path)
    assert mem.top_k == 20
    context = mem.retrieve("what do I do on Saturdays")
    assert context and len(context) <= mem.max_chars
    assert len(mem.last_trace) == len(mem.records)  # 5 facts, all under top_k


def test_soft_delete_flags_and_hides(tmp_path):
    mem = _memory(tmp_path, soft_delete=True)
    mem.forget("that I have asthma")
    assert "I have asthma" not in mem.retrieve("do I have asthma")
    assert "[soft-deleted]" in mem.dump()


def test_state_persists_and_ids_stay_unique(tmp_path):
    client = FakeClient()
    mem = _memory(tmp_path, client=client)
    mem.forget("that I have asthma")
    again = FactEmbeddingMemory(tmp_path, client=client)
    again.write(_session("s2", "2026-01-12", "I like tofu"))
    ids = [r["id"] for r in again.records]
    assert len(ids) == len(set(ids))
    assert "I like tofu" in again.dump() and "I quit the swim team" in again.dump()
