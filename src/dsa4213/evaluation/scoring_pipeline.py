from .llm_judge import mock_judge
from .reference_scorer import contains_reference


def score_record(record, judge=mock_judge):
    """
    Score one agent response based on its probe type.
    - Direct, indirect and retention probes use reference-based binary scoring.
    - Semantic and adversarial probes will be handled by the LLM judge.
    """

    probe_type = record["probe_type"]

    if probe_type in ["direct", "indirect", "retention"]:
        return {
            **record,
            "label": contains_reference(record["response"], record["scoring_target"]),
            "severity": None,
            "explanation": None,
        }

    elif probe_type in ["semantic", "adversarial"]:
        judgment = judge(record)
        return {**record, **judgment}

    else:
        raise ValueError(f"Unknown probe type: {probe_type}")
