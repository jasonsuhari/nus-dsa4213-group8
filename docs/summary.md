# Recursive summary memory: pilot results

Reviewed on 24 October 2026. This is a qualitative, manually reviewed pilot of
five synthetic cases, with one completed attempt selected per case. The longer
history case also had an initial failed attempt, documented below. The pilot
inspects external memory text; it does not establish behavioural forgetting.

This document is the shared summary of the pilot. Full run outputs, individual
review files, and API execution logs remain local and are not included in the
repository. The table and selected evidence below can be read without those files.

## Method

The backend maintains one persistent natural-language summary, updated after
each conversation turn. Each case compares three conditions:

- **Pre-Forget:** ingest the full synthetic conversation.
- **Post-Forget:** copy the Pre-Forget memory, then regenerate it without the
  target fact.
- **Never-Learned:** start from empty memory and ingest an independently ablated
  conversation that omits explicit target statements but keeps retained clues.

The runs used OpenRouter, model `openai/gpt-4o-mini`, and temperature 0. The
reviewer inspected saved summaries and annotated proposition presence in
`review.json`; `assessment.json` derives classifications from those annotations.
Review was assisted by Codex analysis and suggested annotations. It was not an
independent, blinded review. No model-based judge or behavioural probes were run.

## Results table

"No observed loss" below refers only to the predefined retained facts represented
before forgetting. It is not a claim that every possible detail was preserved.

| Case | Target | Target present before | Direct target present after | Retained facts before / after | Additional observation |
| --- | --- | --- | --- | --- | --- |
| 01 Simple | Vegetarian identity | Yes | No | 2 / 2; no observed loss | Japanese food preference and NUS retained. |
| 02 Related facts | Vegetarian identity | Yes | No | 3 / 3; no observed loss | Tofu preference, meat avoidance, and NUS retained. Meat avoidance remains a potential inference clue. |
| 03 Paraphrase | Grew up in Malaysia | Yes | No | 2 / 2; no observed loss | The childhood-in-Malaysia paraphrase was removed; hiking and NUS retained. |
| 04 Temporal | Lived in Singapore in 2023 | Yes | No | 2 / 2; no observed loss | Tokyo move retained with its 2025 date; software-engineer employment retained. |
| 05 Recursive, retry | Grew up in Malaysia | Yes | No | 6 / 6; no observed loss | Saturday cycling correction, Tokyo date, and deadline retained. An assistant suggestion was incorrectly stored as user information. |

For all five completed cases, the reviewed target was present in Pre-Forget and
absent directly in both Post-Forget and Never-Learned. No predefined retained
fact was annotated as missing before forgetting. Post-Forget and Never-Learned
represented the same expected retained propositions, although wording and order
sometimes differed.

## Evidence and interpretation

### Simple, paraphrase, and temporal cases

- **Simple:** Pre-Forget stated "User is vegetarian and likes Japanese food.
  User studies at NUS." Post-Forget retained Japanese food and NUS while omitting
  vegetarian identity. Never-Learned represented the same retained facts.
- **Paraphrase:** Pre-Forget said "User spent their childhood in Malaysia,
  enjoys hiking, and studies at NUS." Post-Forget and Never-Learned both said
  "User enjoys hiking and studies at NUS." This illustrates removal of a
  proposition expressed differently from the target wording.
- **Temporal:** Pre-Forget included Singapore residence in 2023 and the Tokyo
  move in 2025. Post-Forget said "User moved to Tokyo in 2025. User works as a
  software engineer." Never-Learned represented those same retained facts.
  The required retained event date survived in this example.

### Related facts: direct removal with potential inference remaining

The actual summaries were:

> Pre-Forget: User is vegetarian, avoids meat, enjoys tofu, and studies at NUS.

> Post-Forget: User avoids meat, enjoys tofu, and studies at NUS.

> Never-Learned: User enjoys tofu and avoids meat. User studies at NUS.

The explicit vegetarian identity disappeared while all three expected retained
facts survived. The predefined selective policy requires retaining independently
stated meat avoidance. That clue could support an inference of vegetarianism,
but does not conclusively establish the identity. Potential inference was
annotated as present; actual agent inference was not tested.

### Longer history: correction retained, assistant attribution failed

The successful retry retained NUS, cat Miso, Japanese food preference, Saturday
cycling, the Tokyo move in 2025, and the project deadline of 30 October 2026.
Malaysia upbringing was present before forgetting and absent afterward. The
Saturday schedule correctly replaced the earlier Sunday schedule in the final
summaries.

However, all three conditions included "is considering trying rock climbing."
The input contained only an assistant suggestion to try rock climbing, with no
user acceptance. This is an unsupported user fact introduced during memory
formation. It is not collateral loss caused by forgetting, and its presence in
Never-Learned shows that it can arise without learning the target at all.

### Initial failed recursive attempt

The first recursive attempt failed during both Pre-Forget and Never-Learned
construction with `ValueError: LLM returned no valid summary text.` Post-Forget
was not attempted. Partial summaries were preserved, but no completed snapshots
were produced for that attempt.

A separate retry completed. The initial failure is retained as an execution
reliability observation, not counted as a semantic forgetting failure. The raw
responses were not saved, so their exact cause cannot be established from these
artifacts. The failed attempt should not be silently excluded when describing
the experiment process.

## Report-ready interpretation

In five synthetic pilot cases, manual inspection found the target represented
before forgetting and no direct representation afterward. All predefined
retained facts survived the forgetting operation. Post-Forget and Never-Learned
summaries expressed the same expected retained facts.

The related-facts case retained meat avoidance, which could support inference of
vegetarian identity. The longer-history case incorrectly converted an assistant
suggestion into user information in all three conditions. This error arose
during memory formation rather than forgetting. An initial recursive execution
failed response validation; a separate retry completed.

These observations support the feasibility of LLM-based selective summary
regeneration in the tested examples. They do not establish a general success
rate, absence of inferential or behavioural leakage, or superiority over other
memory architectures.

## Limits and handoff

- Five small synthetic cases and one completed attempt per case are insufficient
  for a general performance estimate; the recursive retry was selected after an
  execution failure, rather than being an independent replication study.
- Semantic review was assisted and not blinded. Related-fact/paraphrase
  boundaries should be agreed with the shared evaluation owner.
- Behavioural answers, reconstruction from later clues, and repeated-run
  variability were not evaluated here.
- Prompt instructions did not prevent the observed assistant-attribution error.
  A summary alone also cannot reliably recover the provenance of merged facts.
- The integration owner should resolve the shared agent's extracted-fact
  attribution and ensure model/client configuration reaches `SummaryMemory`.
  These integration changes are outside this pilot's implementation scope.

The handoff consists of the implementation, tests, pilot instructions, and this
results summary. Full local run folders are not required to read this document.
The evaluation/integration owners can use the supplied runner for further
experiments while presentation work explains the architecture, observations,
and limitations reported here.
