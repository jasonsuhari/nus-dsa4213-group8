import json
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

from .memory import Session, Turn

PROBE_TYPES = ("direct", "indirect", "semantic", "adversarial", "retention")
FORGET_PROBE_TYPES = ("direct", "indirect", "semantic", "adversarial")
RELATIONS = ("unrelated", "related", "highly_related")


@dataclass(frozen=True)
class Probe:
    probe_type: str
    question: str
    scoring_target: tuple[str, ...]
    relation: str | None = None


@dataclass(frozen=True)
class Case:
    case_id: str
    category: str
    target: str
    retained_facts: tuple[str, ...]
    sessions: tuple[Session, ...]
    never_learned_omit: frozenset[tuple[str, int]]
    never_learned_rewrite: dict[tuple[str, int], str]
    probes: tuple[Probe, ...]

    def never_learned_sessions(self) -> tuple[Session, ...]:
        def turns(s: Session) -> tuple[Turn, ...]:
            kept = []
            for i, t in enumerate(s.turns):
                key = (s.session_id, i)
                if key in self.never_learned_omit:
                    continue
                kept.append(Turn(t.role, self.never_learned_rewrite.get(key, t.text)))
            return tuple(kept)

        kept_sessions = (replace(s, turns=turns(s)) for s in self.sessions)
        return tuple(s for s in kept_sessions if s.turns)


def load_cases(path: Path) -> list[Case]:
    path = Path(path)
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    cases = [_case(row) for f in files for row in json.loads(f.read_text())]
    if dupes := [cid for cid, n in Counter(c.case_id for c in cases).items() if n > 1]:
        raise ValueError(f"duplicate case_id: {dupes}")
    return cases


def _case(row: dict) -> Case:
    cid = row["case_id"]
    sessions = tuple(
        Session(s["session_id"], s["date"], tuple(Turn(t["role"], t["text"]) for t in s["turns"]))
        for s in row["sessions"]
    )
    dates = [s.date for s in sessions]
    if dates != sorted(dates):
        raise ValueError(f"{cid}: sessions are not in date order")

    turn_keys = {(s.session_id, i) for s in sessions for i in range(len(s.turns))}
    omit = frozenset((sid, i) for sid, i in row["never_learned_omit"])
    rewrite = {(sid, i): text for sid, i, text in row.get("never_learned_rewrite", [])}
    if not omit and not rewrite:
        raise ValueError(f"{cid}: never-learned history is identical to the full one")
    if missing := (omit | rewrite.keys()) - turn_keys:
        raise ValueError(f"{cid}: never-learned edits name no turn: {sorted(missing)}")

    probes = tuple(
        Probe(p["type"], p["question"], tuple(p["scoring_target"]), p.get("relation"))
        for p in row["probes"]
    )
    counts = Counter(p.probe_type for p in probes)
    if bad := set(counts) - set(PROBE_TYPES):
        raise ValueError(f"{cid}: unknown probe types {sorted(bad)}")
    if missing_types := [t for t in FORGET_PROBE_TYPES if counts[t] != 1]:
        raise ValueError(f"{cid}: needs exactly one probe of each type, check {missing_types}")
    if counts["retention"] < 2:
        raise ValueError(f"{cid}: needs at least two retention probes")
    for p in probes:
        if not p.scoring_target:
            raise ValueError(f"{cid}: probe without scoring_target: {p.question}")
        if (p.probe_type == "retention") != (p.relation in RELATIONS):
            raise ValueError(f"{cid}: retention probes, and only they, need a relation")

    return Case(
        cid,
        row.get("category", "pilot"),
        row["target"],
        tuple(row["retained_facts"]),
        sessions,
        omit,
        rewrite,
        probes,
    )
