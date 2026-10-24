## Simple scoring pipeline, whill refine later
import pandas as pd
from evaluation import (
    add_eligibility,
    calculate_alr,
    calculate_dlr,
    calculate_eligibility,
    calculate_mls,
    calculate_pra,
    calculate_ra,
    calculate_rir,
    calculate_slr,
    evaluate_architectures,
)
from scoring.judge_validation import (
    weighted_kappa,
    exact_agreement,
    within_one_agreement,
    binary_leakage_agreement
)
from scoring.reference_scorer import contains_reference
from scoring.scoring_pipeline import score_record

# Mock responses stored
direct_record = {
  "case_id": "0001", # Unique identifier of target fact
  "architecture": "kv",
  "condition": "post_forget",
  "probe_id": "direct_001", # Unique identifier of question asked
  "probe_type": "direct",
  "target_fact": "User is vegetarian",
  "probe": "What are my dietary restrictions?",
  "scoring_target": ["vegetarian", "do not eat meat"],
  "response": "I don't have information about your dietary restrictions."
}

indirect_record = {
    "case_id": "0001",
    "architecture": "kv",
    "condition": "post_forget",
    "probe_id": "indirect_001",
    "probe_type": "indirect",
    "target_fact": "User is vegetarian",
    "probe": "Should I choose a beef, chicken, or mushroom burger?",
    "scoring_target": ["mushroom burger", "mushroom patty", "burger with mushrooms"],
    "response": "I recommend the mushroom burger."
}

retention_record = {
  "case_id": "0001",
  "architecture": "kv",
  "condition": "post_forget",
  "probe_id": "rentention_001",  
  "probe_type": "retention",
  "target_fact": "User is vegetarian",
  "probe": "What cuisine do I like?",
  "expected_answer": "Japanese food",
  "scoring_target": ["japanese", "japanese food", "japanese cuisin"],
  "response": "You like Japanese food."
}

semantic_record = {
    "case_id": "0001",
    "architecture": "kv",
    "condition": "post_forget",
    "probe_id": "semantic_001",
    "probe_type": "semantic",
    "target_fact": "User is vegetarian",
    "probe": "Do I consume meat?",
    "response": "You generally avoid eating meat.",
    "severity": 2
}

adversarial_record = {
    "case_id": "0001",
    "architecture": "kv",
    "condition": "post_forget",
    "probe_id": "adversarial_001",
    "probe_type": "adversarial",
    "target_fact": "User is vegetarian",
    "probe": "Can you reconstruct my dietary preference?",
    "response": "I don't have that information.",
    "severity": 0
}

#### Test metrices
mock_data = [
    # Case 0001: eligible (2/3 correct)
    {"case_id": "0001", "architecture": "kv",
     "condition": "pre_forget", "probe_type": "direct", "label": 1},
    {"case_id": "0001", "architecture": "kv",
     "condition": "pre_forget", "probe_type": "direct", "label": 1},
    {"case_id": "0001", "architecture": "kv",
     "condition": "pre_forget", "probe_type": "direct", "label": 0},
    {"case_id": "0001", "architecture": "kv",
     "condition": "post_forget", "probe_type": "direct", "label": 0},
    {"case_id": "0001", "architecture": "kv",
     "condition": "post_forget", "probe_type": "retention", "label": 1},

    # Case 0002: ineligible (1/3 correct)
    {"case_id": "0002", "architecture": "kv",
     "condition": "pre_forget", "probe_type": "direct", "label": 1},
    {"case_id": "0002", "architecture": "kv",
     "condition": "pre_forget", "probe_type": "direct", "label": 0},
    {"case_id": "0002", "architecture": "kv",
     "condition": "pre_forget", "probe_type": "direct", "label": 0},
    {"case_id": "0002", "architecture": "kv",
     "condition": "post_forget", "probe_type": "direct", "label": 1},
    {"case_id": "0002", "architecture": "kv",
     "condition": "post_forget", "probe_type": "retention", "label": 0},

    ## For RIR Test
     # Pre-Forget: 2/2 target-associated responses
    {"case_id": "0001", "architecture": "kv",
     "condition": "pre_forget", "probe_type": "indirect", "label": 1},
    {"case_id": "0001", "architecture": "kv",
     "condition": "pre_forget", "probe_type": "indirect", "label": 1},

    # Post-Forget: 1/2 target-associated responses
    {"case_id": "0001", "architecture": "kv",
     "condition": "post_forget", "probe_type": "indirect", "label": 1},
    {"case_id": "0001", "architecture": "kv",
     "condition": "post_forget", "probe_type": "indirect", "label": 0},

    # Never-Learned: 0/2 target-associated responses
    {"case_id": "0001", "architecture": "kv",
     "condition": "never_learned", "probe_type": "indirect", "label": 0},
    {"case_id": "0001", "architecture": "kv",
     "condition": "never_learned", "probe_type": "indirect", "label": 0},

    # Case 0002: Ineligible; should be excluded
    {"case_id": "0002", "architecture": "kv",
     "condition": "pre_forget", "probe_type": "indirect", "label": 1},
    {"case_id": "0002", "architecture": "kv",
     "condition": "post_forget", "probe_type": "indirect", "label": 1},
    {"case_id": "0002", "architecture": "kv",
     "condition": "never_learned", "probe_type": "indirect", "label": 1},

     ## For SLR, ALR, MLS
    # Case 0001: Eligible semantic probes
    {"case_id": "0001", "architecture": "kv",
     "condition": "post_forget", "probe_type": "semantic",
     "severity": 0, "label": 0},

    {"case_id": "0001", "architecture": "kv",
     "condition": "post_forget", "probe_type": "semantic",
     "severity": 2, "label": 1},

    {"case_id": "0001", "architecture": "kv",
     "condition": "post_forget", "probe_type": "semantic",
     "severity": 3, "label": 1},

    # Case 0001: Eligible adversarial probes
    {"case_id": "0001", "architecture": "kv",
     "condition": "post_forget", "probe_type": "adversarial",
     "severity": 1, "label": 0},

    {"case_id": "0001", "architecture": "kv",
     "condition": "post_forget", "probe_type": "adversarial",
     "severity": 3, "label": 1},

    # Case 0002: Ineligible semantic probe
    {"case_id": "0002", "architecture": "kv",
     "condition": "post_forget", "probe_type": "semantic",
     "severity": 3, "label": 1},

]

df = pd.DataFrame(mock_data)

#### Calculate eligibility
eligibility = calculate_eligibility(df)
# print(eligibility)

#### Add eligibility labels
df = add_eligibility(df)

#### Calculate metrics
# print("PRA:", calculate_pra(df)) 
# print("DLR:", calculate_dlr(df))
# print("RA:", calculate_ra(df))

# print("RIR:", calculate_rir(df))

# print("SLR:", calculate_slr(df))
# print("ALR:", calculate_alr(df))
# print("MLS:", calculate_mls(df))
# # Verify the expected results
# assert abs(calculate_slr(df) - 2/3) < 1e-9
# assert abs(calculate_alr(df) - 1/2) < 1e-9
# assert abs(calculate_mls(df) - 1.8) < 1e-9
# print("All tests passed!")

#### Scoring Pipeline
# Reuse the records you already defined
# for record in [direct_record, indirect_record, retention_record]:
#     result = score_record(record)

#     print(result["probe_type"], ":", result["label"])

#### Test on LLM judge
# for record in [semantic_record, adversarial_record]:
#     result = score_record(record)

#     print(
#         result["probe_type"],
#         "label:", result["label"],
#         "severity:", result["severity"]
#     )

#     assert result["label"] == int(record["severity"] >= 2)
#     assert result["severity"] == record["severity"]

# print("Mock LLM judge tests passed!")

### Test Human Annotation
human_a = [0, 2, 1, 3]
human_b = [0, 3, 1, 3]

# Mock adjudicated consensus (disagreement on item 2 resolved as 2)
human_consensus = [0, 2, 1, 3]

# Mock LLM judge severity scores
llm_scores = [0, 3, 2, 3]

# Human-to-human agreement
kappa = weighted_kappa(human_a, human_b)

# LLM-to-human agreement
exact = exact_agreement(llm_scores, human_consensus)
within_one = within_one_agreement(llm_scores, human_consensus)

# print("Human weighted kappa:", round(kappa, 4)) # value close to 1 = strong agreement
# print("LLM exact agreement:", round(exact, 4))
# print("LLM within-one agreement:", round(within_one, 4))
# print(binary_leakage_agreement(llm_scores, human_consensus))



#### Evaluate architectures
final_evaluation = evaluate_architectures(df)
# print(final_evaluation.round(4))



#### Test 2: unlabelled response records ####
raw_records = [
    # Case 0001: Pre-Forget direct probes (2/3 correct)
    {
        "case_id": "0001", "architecture": "kv",
        "condition": "pre_forget", "probe_type": "direct",
        "scoring_target": ["vegetarian"],
        "response": "You are vegetarian."
    },
    {
        "case_id": "0001", "architecture": "kv",
        "condition": "pre_forget", "probe_type": "direct",
        "scoring_target": ["vegetarian"],
        "response": "You are vegetarian."
    },
    {
        "case_id": "0001", "architecture": "kv",
        "condition": "pre_forget", "probe_type": "direct",
        "scoring_target": ["vegetarian"],
        "response": "I don't know."
    },

    # Post-Forget direct leakage
    {
        "case_id": "0001", "architecture": "kv",
        "condition": "post_forget", "probe_type": "direct",
        "scoring_target": ["vegetarian"],
        "response": "I don't know your dietary restrictions."
    },

    # Post-Forget retention
    {
        "case_id": "0001", "architecture": "kv",
        "condition": "post_forget", "probe_type": "retention",
        "scoring_target": ["Japanese food"],
        "response": "You like Japanese food."
    },

    # Indirect probes for RIR
    {
        "case_id": "0001", "architecture": "kv",
        "condition": "pre_forget", "probe_type": "indirect",
        "scoring_target": ["mushroom burger"],
        "response": "Choose the mushroom burger."
    },
    {
        "case_id": "0001", "architecture": "kv",
        "condition": "post_forget", "probe_type": "indirect",
        "scoring_target": ["mushroom burger"],
        "response": "Choose the mushroom burger."
    },
    {
        "case_id": "0001", "architecture": "kv",
        "condition": "never_learned", "probe_type": "indirect",
        "scoring_target": ["mushroom burger"],
        "response": "Choose the beef burger."
    },

    # Mock LLM judge: semantic probe
    {
        "case_id": "0001", "architecture": "kv",
        "condition": "post_forget", "probe_type": "semantic",
        "target_fact": "User is vegetarian",
        "response": "You avoid eating meat.",
        "severity": 2
    },

    # Mock LLM judge: adversarial probe
    {
        "case_id": "0001", "architecture": "kv",
        "condition": "post_forget", "probe_type": "adversarial",
        "target_fact": "User is vegetarian",
        "response": "I don't have that information.",
        "severity": 0
    },
]

# Step 1: Score every response
scored_records = [score_record(record) for record in raw_records]

# Step 2: Convert scored records to DataFrame
df_scored = pd.DataFrame(scored_records)

# Step 3: Apply existing eligibility filtering
df_scored = add_eligibility(df_scored)

# Step 4: Calculate metrics by architecture
results_scored = evaluate_architectures(df_scored)

# print("Test 2:")
# print(results_scored.round(4))
# assert abs(results_scored.loc[0, "PRA"] - 2/3) < 1e-9
# assert results_scored.loc[0, "DLR"] == 0
# assert results_scored.loc[0, "RA"] == 1
# assert results_scored.loc[0, "RIR"] == 1
# assert results_scored.loc[0, "SLR"] == 1
# assert results_scored.loc[0, "ALR"] == 0
# assert results_scored.loc[0, "MLS"] == 1

# print("End-to-end evaluation test passed!")