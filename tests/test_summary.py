"""Mocked contract/state tests; these do not prove semantic forgetting quality."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from openai import APIConnectionError

from dsa4213.memory.base import Session, Turn
from dsa4213.memory.summary import FORGET_PROMPT, WRITE_PROMPT, SummaryMemory


def completion(content, *, finish_reason="stop", refusal=None, tool_calls=None, function_call=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(
                    content=content,
                    refusal=refusal,
                    tool_calls=tool_calls,
                    function_call=function_call,
                ),
            )
        ]
    )


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    return MagicMock()


@pytest.fixture
def memory(tmp_path, client):
    return SummaryMemory(tmp_path, client=client)


def session(*turns):
    return Session("session-1", "2026-10-20", turns)


def seed(memory, text="The user studies at NUS."):
    memory.summary = text
    memory.save_summary()


def assert_saved(memory, expected):
    assert memory.dump() == expected
    assert memory.summary_path.read_text(encoding="utf-8") == expected
    assert sorted(p.name for p in memory.path.iterdir()) == ["summary.txt"]


def test_empty_summary(memory, client):
    assert memory.dump() == ""
    assert memory.retrieve("What do you know about me?") == ""
    client.chat.completions.create.assert_not_called()


def test_empty_session_is_noop(memory, client):
    memory.write(session())
    assert memory.dump() == ""
    assert list(memory.path.iterdir()) == []
    client.chat.completions.create.assert_not_called()


def test_write_and_reload(memory, client):
    client.chat.completions.create.return_value = completion("  The user studies at NUS.\n")
    memory.write(session(Turn("user", "I study at NUS.")))
    assert_saved(memory, "The user studies at NUS.")
    assert SummaryMemory(memory.path, client=client).dump() == memory.dump()
    request = client.chat.completions.create.call_args.kwargs
    assert request["model"] == "gpt-4o-mini"
    assert request["temperature"] == 0
    assert request["messages"][0]["content"] == WRITE_PROMPT
    assert "Conversation date: 2026-10-20" in request["messages"][1]["content"]
    assert "user: I study at NUS." in request["messages"][1]["content"]
    assert "Previous assistant turn" not in request["messages"][1]["content"]


def test_write_updates_existing_summary(memory, client):
    seed(memory)
    updated = "The user studies at NUS and likes Japanese food."
    client.chat.completions.create.return_value = completion(updated)
    memory.write(session(Turn("user", "I like Japanese food.")))
    assert_saved(memory, updated)
    assert (
        "Existing summary:\nThe user studies at NUS."
        in (client.chat.completions.create.call_args.kwargs["messages"][1]["content"])
    )


def test_turns_are_recursive_in_tuple_order_with_attribution(memory, client):
    first = "The user lived in Singapore in 2023."
    final = first + " The user moved to Tokyo in 2025."
    client.chat.completions.create.side_effect = [
        completion(first),
        completion(first),
        completion(final),
    ]
    turns = (
        Turn("user", "I lived in Singapore in 2023."),
        Turn("assistant", "You could try living in Kyoto."),
        Turn("user", "I moved to Tokyo in 2025."),
    )
    memory.write(session(*turns))
    calls = client.chat.completions.create.call_args_list
    assert len(calls) == 3
    for index, turn in enumerate(turns):
        content = calls[index].kwargs["messages"][1]["content"]
        previous = "(empty)" if index == 0 else first
        assert content.startswith(f"Existing summary:\n{previous}\n\n")
        assert content.endswith(f"{turn.role}: {turn.text}")
        assert all(later.text not in content for later in turns[index + 1 :])
    assert_saved(memory, final)


def test_explicit_correction_is_sent_to_write_not_forget(memory, client):
    seed(memory, "The user is vegetarian, avoids meat, and studies at NUS.")
    corrected = "As of 2026-10-20, the user is no longer vegetarian and eats meat. "
    corrected += "The user studies at NUS."
    client.chat.completions.create.return_value = completion(corrected)
    correction = "I am no longer vegetarian. I eat meat now."
    memory.write(session(Turn("user", correction)))
    request = client.chat.completions.create.call_args.kwargs
    assert request["messages"][0]["content"] == WRITE_PROMPT
    assert "The user is vegetarian, avoids meat" in request["messages"][1]["content"]
    assert request["messages"][1]["content"].endswith(f"user: {correction}")
    assert_saved(memory, corrected)


@pytest.mark.parametrize(
    "assistant_text,user_reply,mocked_summary",
    [
        ("Are you vegetarian?", "Yes.", "The user is vegetarian."),
        ("You might enjoy Japanese food.", "Maybe.", ""),
        ("What do you study?", "Statistics.", "The user studies statistics."),
        (
            "Do you prefer Japanese, Thai, or Italian food?",
            "The second one.",
            "The user prefers Thai food.",
        ),
        ("You live in Tokyo, right?", "No, Kyoto.", "The user lives in Kyoto."),
        (
            "Are you vegetarian, and do you study at NUS?",
            "I study at NUS.",
            "The user studies at NUS.",
        ),
        (
            "You might enjoy hiking. You could also try climbing. What do you think?",
            "Sounds good.",
            "",
        ),
        ("You should try rock climbing.", "No thanks.", ""),
        (
            "If you were vegetarian, tofu could be useful.",
            "I like tofu.",
            "The user likes tofu.",
        ),
        (
            "Where do you plan to live next year?",
            "Possibly Tokyo, but I haven't decided.",
            "As of 2026-10-20, the user may live in Tokyo next year but has not decided.",
        ),
    ],
    ids=[
        "clear-confirmation",
        "unconfirmed-suggestion",
        "open-ended-answer",
        "explicit-option-selection",
        "rejection-and-correction",
        "partial-answer-to-compound-question",
        "vague-agreement-to-multiple-suggestions",
        "declined-suggestion",
        "hypothetical-not-confirmed",
        "uncertain-future-plan",
    ],
)
def test_user_reply_receives_assistant_context_only(
    memory, client, assistant_text, user_reply, mocked_summary
):
    # Predetermined outputs check request construction, not semantic compliance.
    client.chat.completions.create.side_effect = [completion(""), completion(mocked_summary)]
    memory.write(session(Turn("assistant", assistant_text), Turn("user", user_reply)))
    calls = client.chat.completions.create.call_args_list
    assert len(calls) == 2  # Still one request per turn.
    assert "Previous assistant turn" not in calls[0].kwargs["messages"][1]["content"]
    request = calls[1].kwargs["messages"][1]["content"]
    assert (
        "Previous assistant turn (context only; not a user fact):\n"
        f"assistant: {assistant_text}\n\nNew conversation turn:\nuser: {user_reply}"
    ) in request
    if mocked_summary:
        assert_saved(memory, mocked_summary)
        assert SummaryMemory(memory.path, client=client).dump() == mocked_summary
    else:
        assert memory.dump() == ""
        assert list(memory.path.iterdir()) == []


def test_only_immediately_preceding_assistant_turn_is_supplied(memory, client):
    turns = (
        Turn("assistant", "Are you vegetarian?"),
        Turn("assistant", "Do you study at NUS?"),
        Turn("user", "Yes."),
        Turn("user", "I enjoy hiking."),
        Turn("assistant", "Do you live in Tokyo?"),
        Turn("assistant", "Do you live in Kyoto?"),
        Turn("user", "Yes."),
    )
    known = "The user studies at NUS."
    client.chat.completions.create.side_effect = [
        completion(""),
        completion(""),
        *[completion(known) for _ in turns[2:]],
    ]
    memory.write(session(*turns))
    calls = client.chat.completions.create.call_args_list
    assert len(calls) == len(turns)
    for index, call in enumerate(calls):
        content = call.kwargs["messages"][1]["content"]
        if index in (2, 6):
            previous_text = turns[index - 1].text
            assert f"assistant: {previous_text}\n\nNew conversation turn:" in content
            for other in turns[: index - 1]:
                if other.role == "assistant":
                    assert other.text not in content
        else:
            assert "Previous assistant turn" not in content
    assert_saved(memory, known)


def test_assistant_context_does_not_cross_sessions(memory, client):
    client.chat.completions.create.side_effect = [completion(None), completion("")]
    memory.write(Session("first", "2026-10-20", (Turn("assistant", "Are you vegetarian?"),)))
    memory.write(Session("second", "2026-10-21", (Turn("user", "Yes."),)))
    request = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert "Conversation date: 2026-10-21" in request
    assert "Previous assistant turn" not in request
    assert "Are you vegetarian?" not in request
    assert request.endswith("user: Yes.")
    assert client.chat.completions.create.call_count == 2
    assert memory.dump() == ""
    assert list(memory.path.iterdir()) == []


def test_initial_assistant_null_allows_following_user_confirmation(memory, client):
    client.chat.completions.create.side_effect = [
        completion(None),
        completion("The user is vegetarian."),
    ]
    memory.write(session(Turn("assistant", "Are you vegetarian?"), Turn("user", "Yes.")))
    calls = client.chat.completions.create.call_args_list
    assert len(calls) == 2
    content = calls[1].kwargs["messages"][1]["content"]
    assert content.startswith("Existing summary:\n(empty)\n\n")
    assert "assistant: Are you vegetarian?" in content
    assert content.endswith("New conversation turn:\nuser: Yes.")
    assert_saved(memory, "The user is vegetarian.")
    assert SummaryMemory(memory.path, client=client).dump() == memory.dump()


def test_initial_assistant_null_is_noop_without_persistence(memory, client):
    client.chat.completions.create.return_value = completion(None)
    memory.write(session(Turn("assistant", "Are you vegetarian?")))
    assert memory.dump() == ""
    assert list(memory.path.iterdir()) == []
    client.chat.completions.create.assert_called_once()


def test_assistant_null_cannot_erase_existing_memory_or_continue(memory, client):
    seed(memory)
    client.chat.completions.create.side_effect = [
        completion(None),
        completion("The user is vegetarian."),
    ]
    with pytest.raises(ValueError, match="valid summary text"):
        memory.write(session(Turn("assistant", "Are you vegetarian?"), Turn("user", "Yes.")))
    assert_saved(memory, "The user studies at NUS.")
    client.chat.completions.create.assert_called_once()


def test_initial_user_null_remains_invalid(memory, client):
    client.chat.completions.create.return_value = completion(None)
    with pytest.raises(ValueError, match="valid summary text"):
        memory.write(session(Turn("user", "I study at NUS.")))
    assert memory.dump() == ""
    assert list(memory.path.iterdir()) == []


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(choices=[]),
        completion(42),
        completion(None, finish_reason="length"),
        completion(None, finish_reason="content_filter"),
        completion(None, refusal="Cannot comply"),
        completion(None, finish_reason="tool_calls"),
        completion(None, tool_calls=[SimpleNamespace(id="unexpected-tool")]),
        completion(None, function_call=SimpleNamespace(name="unexpected-function")),
    ],
)
def test_invalid_empty_assistant_response_is_not_silently_accepted(memory, client, response):
    client.chat.completions.create.return_value = response
    with pytest.raises(ValueError, match="LLM"):
        memory.write(session(Turn("assistant", "Are you vegetarian?")))
    assert memory.dump() == ""
    assert list(memory.path.iterdir()) == []


@pytest.mark.parametrize("budget", [0, 1, 20, 2000])
def test_retrieve_budget_is_query_independent(tmp_path, client, budget):
    memory = SummaryMemory(tmp_path, max_chars=budget, client=client)
    text = "The user studies at NUS and likes 日本料理."
    seed(memory, text)
    assert memory.retrieve("Where do I study?") == text[:budget]
    assert memory.retrieve("What food do I like?") == text[:budget]
    assert len(memory.retrieve("anything")) <= budget
    assert_saved(memory, text)


def test_negative_budget_rejected(tmp_path, client):
    with pytest.raises(ValueError, match="max_chars"):
        SummaryMemory(tmp_path, max_chars=-1, client=client)


def test_forget_saves_retained_related_facts(memory, client):
    seed(memory, "The user is vegetarian, enjoys tofu, avoids meat, and studies at NUS.")
    retained = "The user enjoys tofu, avoids meat, and studies at NUS."
    client.chat.completions.create.return_value = completion(retained)
    memory.forget("The user is vegetarian.")
    assert_saved(memory, retained)
    request = client.chat.completions.create.call_args.kwargs
    assert request["messages"][0]["content"] == FORGET_PROMPT
    assert (
        "Targeted fact to forget:\nThe user is vegetarian." in (request["messages"][1]["content"])
    )
    assert request["temperature"] == 0
    client.chat.completions.create.assert_called_once()


def test_absent_target_uses_llm_and_can_leave_summary_unchanged(memory, client):
    seed(memory)
    client.chat.completions.create.return_value = completion(memory.dump())
    memory.forget("The user owns a dog.")
    assert_saved(memory, "The user studies at NUS.")
    client.chat.completions.create.assert_called_once()


@pytest.mark.parametrize("target", ["", " \n\t"])
def test_blank_target_is_noop(memory, client, target):
    seed(memory)
    memory.forget(target)
    assert_saved(memory, "The user studies at NUS.")
    client.chat.completions.create.assert_not_called()


def test_forget_empty_summary_is_noop(memory, client):
    memory.forget("The user is vegetarian.")
    assert memory.dump() == ""
    client.chat.completions.create.assert_not_called()


@pytest.mark.parametrize("text", ["", " \n"])
def test_forget_only_fact_can_persist_empty_summary(memory, client, text):
    seed(memory, "The user is vegetarian.")
    client.chat.completions.create.return_value = completion(text)
    memory.forget("The user is vegetarian.")
    assert_saved(memory, "")
    assert SummaryMemory(memory.path, client=client).dump() == ""


@pytest.mark.parametrize("text", ["", " \n"])
def test_empty_write_cannot_erase_existing_memory(memory, client, text):
    seed(memory)
    client.chat.completions.create.return_value = completion(text)
    with pytest.raises(ValueError, match="empty updated summary"):
        memory.write(session(Turn("user", "I like Japanese food.")))
    assert_saved(memory, "The user studies at NUS.")


def test_empty_write_when_no_user_facts_exist(memory, client):
    client.chat.completions.create.return_value = completion("")
    memory.write(session(Turn("assistant", "Would you like to try Japanese food?")))
    assert memory.dump() == ""
    assert list(memory.path.iterdir()) == []


@pytest.mark.parametrize("operation", ["write", "forget"])
@pytest.mark.parametrize(
    "response",
    [
        completion(None),
        completion(42),
        SimpleNamespace(choices=[]),
        completion("partial", finish_reason="length"),
        completion("blocked", finish_reason="content_filter"),
        completion("refused", refusal="Cannot comply"),
    ],
)
def test_invalid_response_preserves_memory(memory, client, operation, response):
    seed(memory)
    client.chat.completions.create.return_value = response
    with pytest.raises(ValueError, match="LLM"):
        if operation == "write":
            memory.write(session(Turn("user", "I like Japanese food.")))
        else:
            memory.forget("The user studies at NUS.")
    assert_saved(memory, "The user studies at NUS.")


@pytest.mark.parametrize("operation", ["write", "forget"])
def test_api_failure_preserves_memory(memory, client, operation):
    seed(memory)
    client.chat.completions.create.side_effect = APIConnectionError(request=MagicMock())
    with pytest.raises(APIConnectionError):
        if operation == "write":
            memory.write(session(Turn("user", "I like Japanese food.")))
        else:
            memory.forget("The user studies at NUS.")
    assert_saved(memory, "The user studies at NUS.")


def test_later_turn_failure_keeps_last_successful_turn(memory, client):
    client.chat.completions.create.side_effect = [
        completion("The user studies at NUS."),
        APIConnectionError(request=MagicMock()),
    ]
    with pytest.raises(APIConnectionError):
        memory.write(session(Turn("user", "I study at NUS."), Turn("user", "I like tofu.")))
    assert_saved(memory, "The user studies at NUS.")
    assert SummaryMemory(memory.path, client=client).dump() == memory.dump()


def test_failed_disk_replacement_preserves_memory(memory, client, monkeypatch):
    seed(memory)
    client.chat.completions.create.return_value = completion("The user likes tofu.")

    def fail_replace(*args):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="disk failure"):
        memory.write(session(Turn("user", "I like tofu.")))
    assert_saved(memory, "The user studies at NUS.")
