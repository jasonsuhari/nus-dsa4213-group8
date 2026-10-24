"""KVMemory against a fake OpenAI client, so no API key is needed.

These check the update rules and state transitions, not how well the prompts work.
"""

import json
from types import SimpleNamespace

import pytest

from dsa4213.memory import KVMemory, Session, Turn
from dsa4213.memory.kv import (
    EXTRACT_PROMPT,
    IDENTIFY_PROMPT,
    NEW_KEY,
    SELECT_PROMPT,
    UPDATE_PROMPT,
)


def completion(content, *, finish_reason="stop", refusal=None):
    message = SimpleNamespace(content=content, refusal=refusal)
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish_reason, message=message)])


class FakeClient:
    """Replies to each prompt from its own queue, in order.

    Queue items are dicts (sent as JSON), exceptions (raised), or ready-made responses.
    An empty extraction queue means "no facts", so writes can be scripted per turn.
    """

    def __init__(self, **queues) -> None:
        self.queues = {
            EXTRACT_PROMPT: list(queues.get("extract", [])),
            UPDATE_PROMPT: list(queues.get("update", [])),
            SELECT_PROMPT: list(queues.get("select", [])),
            IDENTIFY_PROMPT: list(queues.get("identify", [])),
        }
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))

    def _chat(self, model, messages, response_format, temperature):
        system, user = messages[0]["content"], messages[1]["content"]
        self.calls.append({"system": system, "user": user, "format": response_format})
        queue = self.queues[system]
        if not queue:
            if system == EXTRACT_PROMPT:
                return completion(json.dumps({"facts": []}))
            raise AssertionError("unexpected call")
        out = queue.pop(0)
        if isinstance(out, Exception):
            raise out
        return completion(json.dumps(out)) if isinstance(out, dict) else out

    def of(self, system: str) -> list[dict]:
        return [c for c in self.calls if c["system"] == system]


def op(op, key, value="", new_key=""):
    return {"op": op, "key": key, "new_key": new_key, "value": value}


def ops(*items):
    return {"operations": list(items)}


def session(*turns, sid="s1", date="2026-01-05"):
    return Session(sid, date, turns)


def seed(mem, profile):
    mem._set_profile(dict(profile))


def key_enum(call):
    schema = call["format"]["json_schema"]["schema"]
    if "operations" in schema["properties"]:
        return schema["properties"]["operations"]["items"]["properties"]["key"]["enum"]
    return schema["properties"]["keys"]["items"]["enum"]


PROFILE = {
    "diet": "The user is vegetarian.",
    "eating_habits": "The user does not eat meat.",
    "food_preference": "The user likes Japanese food.",
    "university": "The user studies at NUS, Kent Ridge.",
}


# ---- write -------------------------------------------------------------------


def test_write_adds_facts_and_persists(tmp_path):
    client = FakeClient(
        extract=[{"facts": ["The user is vegetarian.", "The user likes Japanese food."]}],
        update=[
            ops(op("ADD", NEW_KEY, "The user is vegetarian.", new_key="diet")),
            ops(op("ADD", NEW_KEY, "The user likes Japanese food.", new_key="Food Preference")),
        ],
    )
    mem = KVMemory(tmp_path, client=client)
    mem.write(
        session(Turn("user", "I'm vegetarian and love Japanese food."), Turn("assistant", "Noted!"))
    )

    assert (
        mem.dump()
        == "diet: The user is vegetarian.\nfood_preference: The user likes Japanese food."
    )
    assert KVMemory(tmp_path, client=client).dump() == mem.dump()
    assert all(c["format"]["json_schema"]["strict"] for c in client.calls)


def test_turns_group_replies_and_context_is_limited_to_earlier_turns(tmp_path):
    client = FakeClient()
    mem = KVMemory(tmp_path, client=client, context_turns=1)
    mem.write(
        session(
            Turn("user", "first"),
            Turn("assistant", "reply one"),
            Turn("user", "second"),
            Turn("assistant", "reply two"),
            Turn("user", "third"),
        )
    )
    calls = client.of(EXTRACT_PROMPT)
    assert len(calls) == 3
    assert "New turn:\nuser: second\nassistant: reply two" in calls[1]["user"]
    last = calls[2]["user"]
    assert "Conversation date: 2026-01-05" in last
    assert "user: second" in last and "user: first" not in last
    assert "Earlier turns:\n(none)" in calls[0]["user"]


def test_remember_tool_session_is_a_saved_note(tmp_path):
    client = FakeClient()
    KVMemory(tmp_path, client=client).write(
        Session("remember-001", "2026-01-05", (Turn("assistant", "User owns a cat."),))
    )
    assert "New turn:\nsaved note: User owns a cat." in client.of(EXTRACT_PROMPT)[0]["user"]


def test_later_facts_in_a_turn_see_earlier_operations(tmp_path):
    client = FakeClient(
        extract=[{"facts": ["The user is vegetarian.", "The user avoids meat."]}],
        update=[
            ops(op("ADD", NEW_KEY, "The user is vegetarian.", new_key="diet")),
            ops(op("ADD", NEW_KEY, "The user avoids meat.", new_key="meat_consumption")),
        ],
    )
    mem = KVMemory(tmp_path, client=client)
    mem.write(session(Turn("user", "I'm vegetarian, no meat.")))

    second = client.of(UPDATE_PROMPT)[1]
    assert "diet: The user is vegetarian." in second["user"]
    assert key_enum(second) == ["diet", NEW_KEY]
    assert list(mem.profile) == ["diet", "meat_consumption"]


def test_inconsistent_operations_are_normalised(tmp_path):
    client = FakeClient(
        extract=[{"facts": ["fact"]}],
        update=[
            ops(
                op("ADD", "diet", "The user is pescatarian."),  # existing key: UPDATE
                op("UPDATE", NEW_KEY, "The user lives in Clementi.", new_key="Home Area"),  # ADD
                op("DELETE", "pets"),  # missing key: ignored
                op("DELETE", "diet"),  # second op on diet: ignored
                op("ADD", NEW_KEY, "", new_key="empty"),  # no value: ignored
                op("NOOP", "university"),
            )
        ],
    )
    mem = KVMemory(tmp_path, client=client)
    seed(mem, {"diet": "The user is vegetarian.", "university": "The user studies at NUS."})
    mem.write(session(Turn("user", "...")))

    assert mem.profile == {
        "diet": "The user is pescatarian.",
        "university": "The user studies at NUS.",
        "home_area": "The user lives in Clementi.",
    }
    log = json.loads(mem.write_log_path.read_text(encoding="utf-8").splitlines()[0])
    assert [(a["op"], a["key"]) for a in log["facts"][0]["applied"]] == [
        ("UPDATE", "diet"),
        ("ADD", "home_area"),
    ]


def test_one_fact_can_update_one_entry_and_delete_another(tmp_path):
    client = FakeClient(
        extract=[{"facts": ["The user now eats chicken."]}],
        update=[
            ops(
                op("UPDATE", "meat_consumption", "The user now eats chicken."),
                op("DELETE", "diet"),
            )
        ],
    )
    mem = KVMemory(tmp_path, client=client)
    seed(mem, {"diet": "The user is vegetarian.", "meat_consumption": "The user avoids meat."})
    mem.write(session(Turn("user", "I eat chicken now.")))
    assert mem.profile == {"meat_consumption": "The user now eats chicken."}


def test_failed_call_keeps_the_last_complete_turn(tmp_path):
    client = FakeClient(
        extract=[
            {"facts": ["The user studies at NUS."]},
            {"facts": ["The user is vegetarian.", "The user likes tofu."]},
        ],
        update=[
            ops(op("ADD", NEW_KEY, "The user studies at NUS.", new_key="university")),
            ops(op("ADD", NEW_KEY, "The user is vegetarian.", new_key="diet")),
            ConnectionError("simulated outage"),
        ],
    )
    mem = KVMemory(tmp_path, client=client)
    with pytest.raises(ConnectionError):
        mem.write(session(Turn("user", "I study at NUS."), Turn("user", "Vegetarian, love tofu.")))

    # The first fact of the failed turn is not saved either, since M is replaced per turn.
    assert mem.profile == {"university": "The user studies at NUS."}
    assert KVMemory(tmp_path, client=client).profile == mem.profile


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(choices=[]),
        completion("{not json"),
        completion("[1, 2]"),
        completion('{"facts": []}', finish_reason="length"),
        completion('{"facts": []}', refusal="Cannot comply"),
    ],
)
def test_invalid_response_raises_and_preserves_profile(tmp_path, response):
    mem = KVMemory(tmp_path, client=FakeClient(extract=[response], identify=[response]))
    seed(mem, PROFILE)
    with pytest.raises(ValueError, match="LLM"):
        mem.write(session(Turn("user", "I like tofu.")))
    with pytest.raises(ValueError, match="LLM"):
        mem.forget("The user is vegetarian.")
    assert KVMemory(tmp_path, client=FakeClient()).profile == PROFILE


def test_empty_session_makes_no_calls(tmp_path):
    client = FakeClient()
    mem = KVMemory(tmp_path, client=client)
    mem.write(session())
    assert client.calls == [] and mem.dump() == ""


# ---- retrieve ----------------------------------------------------------------


def test_retrieve_follows_selection_order_and_ignores_unknown_keys(tmp_path):
    client = FakeClient(select=[{"keys": ["university", "pets", "diet", "university"]}])
    mem = KVMemory(tmp_path, client=client)
    seed(mem, PROFILE)

    out = mem.retrieve("Where do I study?")
    assert out == "university: The user studies at NUS, Kent Ridge.\ndiet: The user is vegetarian."
    assert mem.last_keys == ["university", "diet"]
    assert key_enum(client.calls[0]) == list(PROFILE)


@pytest.mark.parametrize("budget", [0, 30, 60, 2000])
def test_retrieve_respects_budget_line_by_line(tmp_path, budget):
    mem = KVMemory(tmp_path, max_chars=budget, client=FakeClient(select=[{"keys": list(PROFILE)}]))
    seed(mem, PROFILE)
    out = mem.retrieve("What do you know about me?")
    assert len(out) <= budget
    # Whole lines only, never a value cut mid-sentence.
    assert all(line in mem.dump().splitlines() for line in out.splitlines())


def test_retrieve_on_empty_store_makes_no_call(tmp_path):
    client = FakeClient()
    assert KVMemory(tmp_path, client=client).retrieve("hi") == ""
    assert client.calls == []


# ---- forget ------------------------------------------------------------------


def test_forget_removes_every_key_stating_the_target(tmp_path):
    client = FakeClient(identify=[{"keys": ["diet", "eating_habits", "made_up"]}])
    mem = KVMemory(tmp_path, client=client)
    seed(mem, PROFILE)
    mem.forget("The user is vegetarian.")

    assert mem.profile == {
        "food_preference": "The user likes Japanese food.",
        "university": "The user studies at NUS, Kent Ridge.",
    }
    assert "vegetarian" not in mem.profile_path.read_text(encoding="utf-8")
    assert "Fact to forget:\nThe user is vegetarian." in client.calls[0]["user"]

    log = json.loads(mem.forget_log_path.read_text(encoding="utf-8").splitlines()[0])
    assert [r["key"] for r in log["removed"]] == ["diet", "eating_habits"]
    assert log["remaining"] == 2


def test_forget_with_no_match_changes_nothing_but_is_logged(tmp_path):
    mem = KVMemory(tmp_path, client=FakeClient(identify=[{"keys": []}]))
    seed(mem, PROFILE)
    mem.forget("The user owns a dog.")
    assert mem.profile == PROFILE
    assert json.loads(mem.forget_log_path.read_text(encoding="utf-8"))["removed"] == []


@pytest.mark.parametrize("target", ["", " \n\t"])
def test_blank_target_is_noop(tmp_path, target):
    client = FakeClient()
    mem = KVMemory(tmp_path, client=client)
    seed(mem, PROFILE)
    mem.forget(target)
    assert mem.profile == PROFILE and client.calls == []


def test_forget_on_empty_store_makes_no_call(tmp_path):
    client = FakeClient()
    KVMemory(tmp_path, client=client).forget("The user is vegetarian.")
    assert client.calls == []


def test_write_after_forget_can_re_add_the_key(tmp_path):
    """The reconstruction path: the updater stays on after a forget request."""
    client = FakeClient(
        identify=[{"keys": ["diet"]}],
        extract=[{"facts": ["The user is vegetarian."]}],
        update=[ops(op("ADD", NEW_KEY, "The user is vegetarian.", new_key="diet"))],
    )
    mem = KVMemory(tmp_path, client=client)
    seed(mem, {"diet": "The user is vegetarian."})
    mem.forget("The user is vegetarian.")
    assert mem.dump() == ""

    mem.write(session(Turn("user", "Still vegetarian, by the way."), sid="s2"))
    assert mem.dump() == "diet: The user is vegetarian."
    assert key_enum(client.of(UPDATE_PROMPT)[0]) == [NEW_KEY]
