# Recursive summary memory

Current implementation and qualitative live evaluation, 25 October 2026.
The implementation is in
[`src/dsa4213/memory/summary.py`](../src/dsa4213/memory/summary.py), and the five-case
pilot is in
[`experiments/test_summary_live.py`](../experiments/test_summary_live.py).

## Architecture

Each user and experimental condition has its own directory containing one
evolving `summary.txt`. Raw sessions are not archived by the memory backend.

| Operation | Behaviour |
| --- | --- |
| `write(session)` | Recursively updates the summary once per turn, in tuple order, with one LLM call per turn. Callers supply sessions chronologically. |
| `retrieve(query)` | Returns the summary prefix under the common character budget, 2,000 by default. No query-specific selection; storage itself is not capped. |
| `forget(target)` | Asks the LLM to regenerate the summary without the targeted proposition and direct paraphrases, preserving non-target facts. |
| `dump()` | Returns the entire current summary, including text beyond the retrieval budget. |

When a user reply immediately follows an assistant turn in the same session,
the previous assistant message is supplied as temporary context. For example,
"Are you vegetarian?" gives meaning to "Yes." Open-ended answers, selections,
and corrections can also use that context. The prompt requires unambiguous user
support for facts; assistant suggestions and ambiguous assent must not establish
definite personal information. Live results below show incomplete compliance.

Only the adjacent assistant turn is supplied, with no additional LLM call or
context carried between sessions. Updates are saved through atomic replacement
and reloaded by new instances. API errors and structurally invalid responses
propagate while preserving the last good state. A failed session can be partly
ingested because earlier successful turns remain saved.

Corrections belong in `write()`: "I eat meat now" updates current dietary facts.
`forget()` removes information and does not establish its opposite. Independently
stated related facts, such as meat avoidance, remain unless also targeted.
An unannotated summary may lose the provenance needed to distinguish those
facts from model-generated paraphrases.

## Run and review

Run from the repository root. Default execution writes an offline plan;
`--live` explicitly enables API requests. The runner is safe to import during
unit testing. No Python dotenv dependency is needed: uv can load `.env`.

```powershell
uv run pytest -q
uv run ruff check .

# Offline plan; a fresh timestamped output directory is created.
uv run python experiments/test_summary_live.py

# Live five-case pilot using OPENROUTER_API_KEY from .env.
uv run --env-file .env python experiments/test_summary_live.py --live --provider openrouter

# Recompute manual assessments offline after editing review.json files.
uv run python experiments/test_summary_live.py --review-run runs/YOUR_RUN_FOLDER
```

OpenAI is the default provider and uses `OPENAI_API_KEY`; OpenRouter defaults to
`openai/gpt-4o-mini`. Both use temperature 0. `--model` and `--base-url` override
defaults. Use `--case 02_related` to select one case, or repeat `--case` to select
several. `--output` must name a new directory. All five cases require at most
47 summary requests, excluding up to two SDK retries per request. Credentials
are never written to the manifest. Temperature 0 does not ensure identical reruns.

Each case builds three conditions:

- **Pre-Forget:** learn the full conversation.
- **Post-Forget:** copy Pre-Forget memory, then apply the forgetting request.
- **Never-Learned:** start empty and ingest input with explicit target turns
  omitted while keeping retained clues.

Inspect `case.json`, `pre_forget.txt`, `post_forget.txt`, and `never_learned.txt`.
Text diffs help inspection but are not semantic scores. Check `result.json` for
errors and completion; partial memory is not a completed snapshot. In
`review.json`, annotate target and retained-fact presence with `true`/`false`,
leave `null` for uncertainty, and quote evidence in notes. Recompute
`assessment.json` using the offline command above. Incomplete cases are skipped.

| Finding | Evidence required |
| --- | --- |
| Target formation failure | Target absent before forgetting; forgetting is not evaluable for that target. |
| Incomplete forgetting/paraphrase leakage | Target or a direct semantic equivalent remains after forgetting. |
| Collateral loss | A non-target fact present before forgetting disappears afterward. |
| Retention failure during formation | An expected retained fact is already absent before forgetting. |
| Potential inferential leakage | Direct target absent, but retained information may support inference. |

`other_hallucinations_or_drift` records additional observations. It is not scored
by the assessment generator, so an empty collateral-loss list does not imply
that every summary is error-free. Never-Learned may itself infer unsupported
information and is not a guaranteed target-free oracle.

## Forgetting pilot results

The evaluated working-tree version used OpenRouter, `openai/gpt-4o-mini`, and
temperature 0. All five cases completed and the user reviewed their snapshots.
Review was assisted by Codex and was not independent or blinded. No model judge
or behavioural response probe was run.

| Case and target | Target present before | Direct target present after | Predefined retained facts before / after | Additional observation |
| --- | --- | --- | --- | --- |
| Simple: vegetarian identity | Yes | No | 2 / 2 | Japanese food preference and NUS retained. |
| Related: vegetarian identity | Yes | No | 3 / 3 | Tofu, meat avoidance, and NUS retained; inference annotated false by reviewer. |
| Paraphrase: grew up in Malaysia | Yes | No | 2 / 2 | Childhood paraphrase removed; hiking and NUS retained. |
| Temporal: Singapore residence in 2023 | Yes | No | 2 / 2 | Tokyo move retained with its 2025 date; employment retained. |
| Recursive: grew up in Malaysia | Yes | No | 6 / 6 | Saturday cycling, Tokyo date, and deadline retained; unsupported climbing interest persists. |

For the related case, Post-Forget said "User avoids meat, enjoys tofu, and studies
at NUS." The reviewer judged these clues insufficient to establish vegetarian
identity and annotated inference false. That judgement does not prove a model
could never infer or guess it; use an agreed inferability threshold for group
comparisons. Actual inference was not tested.

In the recursive case, Pre-Forget and Post-Forget said the user was "open to
trying rock climbing"; Never-Learned said "considering trying rock climbing."
The user had not accepted the assistant's suggestion. This unsupported fact
arose during memory formation and persisted after forgetting; it is not
collateral loss caused by deletion.

## Focused context checks

These separate checks evaluate `write()`, not forgetting. One exchange started
empty; eight were seeded with "The user has a cat named Miso." Each used an
assistant turn followed by a user reply. They were run using a local diagnostic
script, not a built-in option of `test_summary_live.py`.

| Exchange | Execution | Observed result |
| --- | --- | --- |
| Empty memory: vegetarian question / "Yes." | Incomplete | First response had normal stop, no refusal, and null content; validation raised before the user reply. |
| Seeded vegetarian confirmation | Complete | Identity added and Miso retained. |
| "What do you study?" / "Statistics." | Complete | Answer resolved correctly; Miso dropped. |
| Food choices / "The second one." | Complete | Thai preference resolved correctly; Miso dropped. |
| Tokyo assumption / "No, Kyoto." | Complete | Final Kyoto location correct; assistant-only update had already stored unsupported Tokyo residence. |
| Vegetarian/NUS question / "I study at NUS." | Complete | NUS recorded without vegetarian identity; Miso dropped. |
| Hiking/climbing suggestions / "Sounds good." | Complete | Unsupported hiking enjoyment and climbing interest introduced. |
| Climbing suggestion / "No thanks." | Complete | Only Miso retained; no lasting dislike inferred. |
| Future city / "Possibly Tokyo, but I haven't decided." | Complete | Uncertainty and Miso retained. |

Eight of nine exchanges completed, using 17 recorded completion requests. This
is an execution count, not an accuracy score. The three Miso losses occurred
at the user update, without a forgetting request. The null response identifies
the failure path but does not establish why the provider produced missing text.

Review `result.json` and the per-request `request-N.json`/`response-N.json` files;
intermediate responses can reveal errors hidden by a final correction. Record
interpretations separately, preserving raw outputs. These files do not use the
forgetting-review schema and must not be passed to `--review-run`. Their raw
`semantic_review` fields remain `pending`; the table reflects Codex-assisted
inspection, not completed independent human annotations.

## Limits and integration handoff

The latest offline suite passed 105 tests after runner consolidation, including
CLI/import-safety checks; Ruff passed. Mocked tests verify code paths, not LLM
semantic compliance. These small live checks show direct target removal in the
five examples but also formation losses, unsupported facts, and an execution
failure. They do not demonstrate general accuracy improvement, behavioural
forgetting, reconstruction resistance, or superiority over another backend.

- Earlier conversational context is unavailable unless already in the summary;
  only the adjacent assistant turn is supplied temporarily.
- Inspect `dump()`: retrieval truncation can hide a target still stored in memory.
- Later conversations can reintroduce a fact; there is no deletion ledger.

The handoff includes the implementation, tests, pilot runner, and findings
summarised here. Full raw run outputs are not included. Use fresh output
directories for further experiments and keep existing results intact.
