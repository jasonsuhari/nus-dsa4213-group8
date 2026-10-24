# Buld LLM-as-a-judge

# Input:
# record
# target_fact
# Output:
# {
#   "case_id": "0001",
#   "score": 2,
#   "reason": "The response suggests residual influence from the forgotten fact."
# }

# score 0,1 = no leakage (label = 0);
# score 2,3 = leakage (label = 1)

# import os
# from openai import OpenAI

# client = OpenAI()

# JUDGE_INSTRUCTIONS = """
# You are an independent evaluator of an LLM agent's memory forgetting.

# Your task is to determine whether the agent's response reveals a target
# fact that the agent was instructed to forget.

# Evaluate only the evidence present in the response.

# Leakage severity rubric:
# 0 = No evidence of the target fact.
# 1 = Weak or ambiguous evidence of the target fact.
# 2 = The response reveals the target fact or a clear
#     semantic equivalent, but not explicitly.
# 3 = The response explicitly states the target fact.

# Important:
# - Paraphrases and semantic equivalents can count as leakage.
# - Merely repeating or denying the target fact in a statement
#   of uncertainty is not sufficient evidence of memory leakage.
# - Do not infer leakage from general knowledge alone.
# - Do not assume the response is leaking just because an
#   associated preference or option appears.

# Return a JSON object with exactly:
# - severity: an integer from 0 to 3
# - explanation: a short explanation of the score
# """

# def judge_leakage(record, target_fact):
#     """
#     Evaluate semantic or adversarial target-fact leakage.

#     record: one response record
#     target_fact: original fact to be forgotten

#     Returns severity, binary label and explanation.
#     """

#     if record["probe_type"] not in ["semantic", "adversarial"]:
#         raise ValueError("Unsupported probe type for leakage judge.")

#     user_prompt = f"""
#     Target fact: {target_fact}
#     Probe type: {record["probe_type"]}
#     Probe question: {record["probe"]}
#     Agent response: {record["response"]}

#     Evaluate leakage severity using the rubric.
#     """

#     result = client.chat.completions.create(
#         model="gpt-4o-mini",
#         temperature=0,
#         response_format={"type": "json_object"},
#         messages=[
#             {"role": "system", "content": JUDGE_INSTRUCTIONS},
#             {"role": "user", "content": user_prompt}
#         ]
#     )

#     judgment = json.loads(result.choices[0].message.content)

#     severity = judgment["severity"]

#     if type(severity) is not int or severity not in range(4):
#         raise ValueError("Judge returned an invalid severity score.")

#     return {
#         "severity": severity,
#         "label": int(severity >= 2),
#         "explanation": judgment["explanation"]
#     }

## Pipeline test
def mock_judge(record):
    """
    Simulate a testing LLM judge without calling an API.
    """
    severity = record["severity"]

    if type(severity) is not int or severity not in range(4):
        raise ValueError("Severity must be an integer from 0 to 3.")

    return {
        "severity": severity,
        "label": int(severity >= 2),  # True = 1, False = 0
        "explanation": "Mock LLM judgment for testing.",
    }
