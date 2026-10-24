import pandas as pd
import pytest

from dsa4213.evaluation import add_eligibility, calculate_eligibility, evaluate_architectures
from dsa4213.evaluation.metrics import (
    calculate_alr,
    calculate_dlr,
    calculate_mls,
    calculate_pra,
    calculate_ra,
    calculate_rir,
    calculate_slr,
)
from dsa4213.evaluation.scoring_pipeline import score_record

mock_data = [
    # Case 0001: eligible (2/3 correct)
    {
        "case_id": "0001",
        "architecture": "kv",
        "condition": "pre_forget",
        "probe_type": "direct",
        "label": 1,
    },
    {
        "case_id": "0001",
        "architecture": "kv",
        "condition": "pre_forget",
        "probe_type": "direct",
        "label": 1,
    },
    {
        "case_id": "0001",
        "architecture": "kv",
        "condition": "pre_forget",
        "probe_type": "direct",
        "label": 0,
    },
    {
        "case_id": "0001",
        "architecture": "kv",
        "condition": "post_forget",
        "probe_type": "direct",
        "label": 0,
    },
    {
        "case_id": "0001",
        "architecture": "kv",
        "condition": "post_forget",
        "probe_type": "retention",
        "label": 1,
    },
    # Case 0002: ineligible (1/3 correct)
    {
        "case_id": "0002",
        "architecture": "kv",
        "condition": "pre_forget",
        "probe_type": "direct",
        "label": 1,
    },
    {
        "case_id": "0002",
        "architecture": "kv",
        "condition": "pre_forget",
        "probe_type": "direct",
        "label": 0,
    },
    {
        "case_id": "0002",
        "architecture": "kv",
        "condition": "pre_forget",
        "probe_type": "direct",
        "label": 0,
    },
    {
        "case_id": "0002",
        "architecture": "kv",
        "condition": "post_forget",
        "probe_type": "direct",
        "label": 1,
    },
    {
        "case_id": "0002",
        "architecture": "kv",
        "condition": "post_forget",
        "probe_type": "retention",
        "label": 0,
    },
    ## For RIR Test
    # Pre-Forget: 2/2 target-associated responses
    {
        "case_id": "0001",
        "architecture": "kv",
        "condition": "pre_forget",
        "probe_type": "indirect",
        "label": 1,
    },
    {
        "case_id": "0001",
        "architecture": "kv",
        "condition": "pre_forget",
        "probe_type": "indirect",
        "label": 1,
    },
    # Post-Forget: 1/2 target-associated responses
    {
        "case_id": "0001",
        "architecture": "kv",
        "condition": "post_forget",
        "probe_type": "indirect",
        "label": 1,
    },
    {
        "case_id": "0001",
        "architecture": "kv",
        "condition": "post_forget",
        "probe_type": "indirect",
        "label": 0,
    },
    # Never-Learned: 0/2 target-associated responses
    {
        "case_id": "0001",
        "architecture": "kv",
        "condition": "never_learned",
        "probe_type": "indirect",
        "label": 0,
    },
    {
        "case_id": "0001",
        "architecture": "kv",
        "condition": "never_learned",
        "probe_type": "indirect",
        "label": 0,
    },
    # Case 0002: Ineligible; should be excluded
    {
        "case_id": "0002",
        "architecture": "kv",
        "condition": "pre_forget",
        "probe_type": "indirect",
        "label": 1,
    },
    {
        "case_id": "0002",
        "architecture": "kv",
        "condition": "post_forget",
        "probe_type": "indirect",
        "label": 1,
    },
    {
        "case_id": "0002",
        "architecture": "kv",
        "condition": "never_learned",
        "probe_type": "indirect",
        "label": 1,
    },
    ## For SLR, ALR, MLS
    # Case 0001: Eligible semantic probes
    {
        "case_id": "0001",
        "architecture": "kv",
        "condition": "post_forget",
        "probe_type": "semantic",
        "severity": 0,
        "label": 0,
    },
    {
        "case_id": "0001",
        "architecture": "kv",
        "condition": "post_forget",
        "probe_type": "semantic",
        "severity": 2,
        "label": 1,
    },
    {
        "case_id": "0001",
        "architecture": "kv",
        "condition": "post_forget",
        "probe_type": "semantic",
        "severity": 3,
        "label": 1,
    },
    # Case 0001: Eligible adversarial probes
    {
        "case_id": "0001",
        "architecture": "kv",
        "condition": "post_forget",
        "probe_type": "adversarial",
        "severity": 1,
        "label": 0,
    },
    {
        "case_id": "0001",
        "architecture": "kv",
        "condition": "post_forget",
        "probe_type": "adversarial",
        "severity": 3,
        "label": 1,
    },
    # Case 0002: Ineligible semantic probe
    {
        "case_id": "0002",
        "architecture": "kv",
        "condition": "post_forget",
        "probe_type": "semantic",
        "severity": 3,
        "label": 1,
    },
]


def test_metrics_on_labelled_responses():
    df = pd.DataFrame(mock_data)
    eligibility = calculate_eligibility(df).set_index("case_id")["eligible"]
    assert eligibility.to_dict() == {"0001": True, "0002": False}

    df = add_eligibility(df)
    assert calculate_pra(df) == pytest.approx(3 / 6)
    assert calculate_dlr(df) == 0
    assert calculate_ra(df) == 1
    assert calculate_rir(df) == pytest.approx(0.5)
    assert calculate_slr(df) == pytest.approx(2 / 3)
    assert calculate_alr(df) == pytest.approx(1 / 2)
    assert calculate_mls(df) == pytest.approx(1.8)


raw_records = [
    # Case 0001: Pre-Forget direct probes (2/3 correct)
    {
        "case_id": "0001",
        "architecture": "kv",
        "condition": "pre_forget",
        "probe_type": "direct",
        "scoring_target": ["vegetarian"],
        "response": "You are vegetarian.",
    },
    {
        "case_id": "0001",
        "architecture": "kv",
        "condition": "pre_forget",
        "probe_type": "direct",
        "scoring_target": ["vegetarian"],
        "response": "You are vegetarian.",
    },
    {
        "case_id": "0001",
        "architecture": "kv",
        "condition": "pre_forget",
        "probe_type": "direct",
        "scoring_target": ["vegetarian"],
        "response": "I don't know.",
    },
    # Post-Forget direct leakage
    {
        "case_id": "0001",
        "architecture": "kv",
        "condition": "post_forget",
        "probe_type": "direct",
        "scoring_target": ["vegetarian"],
        "response": "I don't know your dietary restrictions.",
    },
    # Post-Forget retention
    {
        "case_id": "0001",
        "architecture": "kv",
        "condition": "post_forget",
        "probe_type": "retention",
        "scoring_target": ["Japanese food"],
        "response": "You like Japanese food.",
    },
    # Indirect probes for RIR
    {
        "case_id": "0001",
        "architecture": "kv",
        "condition": "pre_forget",
        "probe_type": "indirect",
        "scoring_target": ["mushroom burger"],
        "response": "Choose the mushroom burger.",
    },
    {
        "case_id": "0001",
        "architecture": "kv",
        "condition": "post_forget",
        "probe_type": "indirect",
        "scoring_target": ["mushroom burger"],
        "response": "Choose the mushroom burger.",
    },
    {
        "case_id": "0001",
        "architecture": "kv",
        "condition": "never_learned",
        "probe_type": "indirect",
        "scoring_target": ["mushroom burger"],
        "response": "Choose the beef burger.",
    },
    # Mock LLM judge: semantic probe
    {
        "case_id": "0001",
        "architecture": "kv",
        "condition": "post_forget",
        "probe_type": "semantic",
        "target_fact": "User is vegetarian",
        "response": "You avoid eating meat.",
        "severity": 2,
    },
    # Mock LLM judge: adversarial probe
    {
        "case_id": "0001",
        "architecture": "kv",
        "condition": "post_forget",
        "probe_type": "adversarial",
        "target_fact": "User is vegetarian",
        "response": "I don't have that information.",
        "severity": 0,
    },
]


def test_scoring_then_metrics_end_to_end():
    df = add_eligibility(pd.DataFrame([score_record(r) for r in raw_records]))
    row = evaluate_architectures(df).iloc[0]
    assert row["PRA"] == pytest.approx(2 / 3)
    assert row["DLR"] == 0
    assert row["RA"] == 1
    assert row["RIR"] == 1
    assert row["SLR"] == 1
    assert row["ALR"] == 0
    assert row["MLS"] == 1
