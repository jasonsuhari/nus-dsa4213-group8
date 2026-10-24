# Human annotation agreement
# Validate judge

from sklearn.metrics import cohen_kappa_score


def weighted_kappa(scores_1, scores_2):
    """
    Calculate quadratic weighted Cohen's kappa
    between two annotators.
    """
    return cohen_kappa_score(
        scores_1,
        scores_2,
        labels=[0, 1, 2, 3],
        weights="quadratic"
    )


def exact_agreement(judge_scores, consensus_scores):
    """
    Proportion of LLM scores that exactly match
    the human consensus.
    """
    if len(judge_scores) != len(consensus_scores) or not judge_scores:
        raise ValueError("Scores must be non-empty and have equal lengths.")

    matches = [
        j == h
        for j, h in zip(judge_scores, consensus_scores, strict=True)
    ]

    return sum(matches) / len(matches)


def within_one_agreement(judge_scores, consensus_scores):
    """
    Proportion of LLM scores within one severity level
    of the human consensus.
    """
    if len(judge_scores) != len(consensus_scores) or not judge_scores:
        raise ValueError("Scores must be non-empty and have equal lengths.")

    matches = [
        abs(j - h) <= 1
        for j, h in zip(judge_scores, consensus_scores, strict=True)
    ]

    return sum(matches) / len(matches)


def binary_leakage_agreement(llm_scores, consensus_scores, strict=True):
    """
    Agreement on binary leakage classification.
    Severity >= 2 indicates leakage.
    """
    if len(llm_scores) != len(consensus_scores) or not llm_scores:
        raise ValueError("Scores must be non-empty and have equal lengths.")

    matches = [
        (llm >= 2) == (human >= 2)
        for llm, human in zip(llm_scores, consensus_scores, strict=True)
    ]

    return sum(matches) / len(matches)