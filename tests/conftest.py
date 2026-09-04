"""Fixture episode: ten tiny sessions with one target fact and its residue.

Develop against this instead of waiting for Memora. It runs in a second and
costs nothing. The shape mirrors a real episode: one explicit statement of the
target, four facts that imply it, four that have nothing to do with it.

An arm that removes the target and keeps the distractors is doing well. An arm
that removes the distractors too is showing collateral damage.
"""

import pytest

from dsa4213.memory import Session, Turn

TARGET_FACT = "the user's nationality"
TARGET_ALIASES = ["indonesian", "indonesia", "jakarta"]
SHADOW_MARKERS = ["mother", "surabaya", "bahasa", "rendang"]
DISTRACTOR_MARKERS = ["dentist", "meetings", "shellfish", "guitar"]


def _s(n: int, date: str, user: str, assistant: str) -> Session:
    return Session(
        session_id=f"s{n:02d}",
        date=date,
        turns=(Turn("user", user), Turn("assistant", assistant)),
    )


EPISODE = [
    _s(1, "2026-01-05", "I just started a fintech job here.", "Congratulations, how is it going?"),
    # the target
    _s(2, "2026-01-07", "I'm Indonesian, moved over for uni.", "Good to know, thanks for sharing."),
    # shadows: each implies the target without stating it
    _s(3, "2026-01-12", "My mother is Indonesian too.", "Does she still live there?"),
    _s(4, "2026-01-14", "Booked a dentist appointment Thursday.", "Noted, Thursday it is."),
    _s(5, "2026-01-19", "Dad grew up in Surabaya.", "That is on Java, isn't it?"),
    _s(6, "2026-01-21", "I prefer meetings before 10am.", "I'll keep mornings in mind."),
    _s(7, "2026-01-26", "We speak Bahasa at home.", "Nice that you've kept the language."),
    _s(8, "2026-01-28", "I'm allergic to shellfish.", "I'll avoid suggesting seafood places."),
    _s(9, "2026-02-02", "Cooked rendang for friends.", "That takes hours, worth it though."),
    _s(10, "2026-02-04", "Started learning guitar.", "What are you working on first?"),
]


@pytest.fixture
def episode() -> list[Session]:
    return EPISODE
