import json
from dataclasses import dataclass, replace
from pathlib import Path

from .memory import Session, Turn

PROBE_TYPES = ("direct", "indirect", "semantic", "adversarial", "retention")


@dataclass(frozen=True)
class Probe:
    probe_type: str
    question: str
    scoring_target: tuple[str, ...]


@dataclass(frozen=True)
class Case:
    case_id: str
    target: str
    retained_facts: tuple[str, ...]
    sessions: tuple[Session, ...]
    never_learned_omit: frozenset[tuple[str, int]]
    probes: tuple[Probe, ...]

    def never_learned_sessions(self) -> tuple[Session, ...]:
        kept = (
            replace(
                s,
                turns=tuple(
                    t
                    for i, t in enumerate(s.turns)
                    if (s.session_id, i) not in self.never_learned_omit
                ),
            )
            for s in self.sessions
        )
        return tuple(s for s in kept if s.turns)


def load_cases(path: Path) -> list[Case]:
    return [_case(row) for row in json.loads(Path(path).read_text())]


def _case(row: dict) -> Case:
    sessions = tuple(
        Session(s["session_id"], s["date"], tuple(Turn(t["role"], t["text"]) for t in s["turns"]))
        for s in row["sessions"]
    )
    omit = frozenset((sid, i) for sid, i in row["never_learned_omit"])
    turn_keys = {(s.session_id, i) for s in sessions for i in range(len(s.turns))}
    if missing := omit - turn_keys:
        raise ValueError(f"{row['case_id']}: never_learned_omit names no turn: {sorted(missing)}")
    probes = tuple(
        Probe(p["type"], p["question"], tuple(p["scoring_target"])) for p in row["probes"]
    )
    if bad := {p.probe_type for p in probes} - set(PROBE_TYPES):
        raise ValueError(f"{row['case_id']}: unknown probe types {sorted(bad)}")
    return Case(row["case_id"], row["target"], tuple(row["retained_facts"]), sessions, omit, probes)
