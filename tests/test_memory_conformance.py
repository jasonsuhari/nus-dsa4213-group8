"""What every memory arm has to do.

Stubs raise NotImplementedError and skip, so the suite stays green while people
work. When your four tests pass, the harness can drive your arm.
"""

import pytest
from conftest import DISTRACTOR_MARKERS, EPISODE, TARGET_ALIASES, TARGET_FACT

from dsa4213.memory import ARMS, FullTranscriptMemory, NullMemory


def build(arm_cls, path, config=None):
    arm = arm_cls(path, config or {})
    try:
        for session in EPISODE:
            arm.write(session)
    except NotImplementedError as e:
        pytest.skip(f"{arm_cls.__name__} not implemented yet: {e}")
    return arm


def has_target(text: str) -> bool:
    return any(alias in text.lower() for alias in TARGET_ALIASES)


@pytest.mark.parametrize("arm_cls", ARMS, ids=lambda c: c.__name__)
def test_remembers_before_forgetting(arm_cls, tmp_path):
    """An arm that cannot remember cannot be tested on forgetting."""
    arm = build(arm_cls, tmp_path)
    assert has_target(arm.retrieve("where is the user from?"))


@pytest.mark.parametrize("arm_cls", ARMS, ids=lambda c: c.__name__)
def test_forget_removes_target_from_store(arm_cls, tmp_path):
    arm = build(arm_cls, tmp_path)
    arm.forget(TARGET_FACT)
    # dump() rather than retrieve(), since reconstruction is a store-level event.
    assert not has_target(arm.dump())


@pytest.mark.parametrize("arm_cls", ARMS, ids=lambda c: c.__name__)
def test_forget_keeps_unrelated_facts(arm_cls, tmp_path):
    """Deleting everything is not a passing score."""
    arm = build(arm_cls, tmp_path)
    arm.forget(TARGET_FACT)
    survivors = [m for m in DISTRACTOR_MARKERS if m in arm.dump().lower()]
    assert len(survivors) >= 3, f"only {survivors} survived, the delete was too broad"


@pytest.mark.parametrize("arm_cls", ARMS, ids=lambda c: c.__name__)
def test_retrieve_respects_budget(arm_cls, tmp_path):
    arm = build(arm_cls, tmp_path, {"max_chars": 200})
    assert len(arm.retrieve("tell me about the user")) <= 200


def test_null_memory_is_the_floor(tmp_path):
    arm = build(NullMemory, tmp_path)
    assert arm.retrieve("where is the user from?") == ""
    assert arm.dump() == ""


def test_full_transcript_is_the_ceiling(tmp_path):
    arm = build(FullTranscriptMemory, tmp_path)
    assert has_target(arm.dump())
    arm.forget(TARGET_FACT)
    assert has_target(arm.dump())
