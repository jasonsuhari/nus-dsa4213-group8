# Summary memory pilot

The baseline remains one persistent `summary.txt` per synthetic user. `write()`
updates it once per turn in `Session.turns` order; sessions must be supplied in
date order by the caller. Each request contains the current summary, one turn's
role/text, and the session date. Raw sessions are not saved by `SummaryMemory`.
`retrieve()` returns the summary prefix under the common character budget;
`dump()` returns its complete text. Storage itself is not capped by that budget.

## Review findings and corrections

- The old implementation made one write request per session despite describing
  turn-level recursion. The runner's `Agent.ingest()` passes a whole session, so
  iteration belongs inside `SummaryMemory.write()`.
- The write prompt now distinguishes user statements from assistant suggestions,
  preserves dated history and corrections, and leaves memory unchanged when a
  turn adds nothing. The forget prompt requests unchanged text for absent targets,
  no record of deletion, and empty text when nothing remains.
- `agent.py` is unchanged. `Agent._remember()` labels extracted user facts as
  assistant turns, which the stricter write prompt may ignore. Direct ingestion
  of user-attributed synthetic sessions is the supported path for this pilot.
  Resolving that attribution requires coordination with the shared agent owner;
  the summariser cannot verify the origin of an assistant-only fact. No shared
  signatures or other memory backends changed; embedding and KV are currently
  stubs. No response-evaluation runner exists in this checkout.
- Missing choices/text, refusals, and incomplete completions raise before saving.
  An empty write cannot erase existing information. Empty text with a normal
  completion is allowed for initially empty memory or after forgetting the only
  fact. Semantic validity still needs review: the implementation cannot tell a
  mistaken empty forget response from a legitimate complete deletion.
- API failures propagate. Each successful turn is saved atomically before the
  next request; failures preserve the last successful turn. A failed session can
  therefore be partially ingested. Replaying it is not guaranteed idempotent.
- Old tests used an invalid package import and some created real clients. The
  tests now inject mocked clients, require no credentials, and check contracts
  and state transitions. They do not test model understanding of the prompts.
- The former live script made requests on import and depended on undeclared
  `dotenv`. It is now an import-safe entry point for this pilot.

## Execute from the repository root

Use the existing environment, or run `uv sync` first. The commands below use uv
to load `.env`; no Python dotenv dependency is needed. `.env` is gitignored.

```powershell
uv run pytest -q
uv run ruff check .

# Offline plan: saves the fixtures/settings and prints the request count.
uv run python -m experiments.summary_pilot --output runs/summary-pilot-plan

# OpenAI: .env contains OPENAI_API_KEY. Explicitly enables live requests.
uv run --env-file .env python -m experiments.summary_pilot --live --output runs/summary-pilot-live

# Lower-cost first check: run only the related-facts case in a new directory.
uv run --env-file .env python -m experiments.summary_pilot --live --case 02_related --output runs/summary-pilot-related
```

The default model is the shared `gpt-4o-mini` with temperature 0. If `.env`
contains `OPENROUTER_API_KEY`, select OpenRouter explicitly. The pilot uses that
key with `https://openrouter.ai/api/v1` and `openai/gpt-4o-mini`:

```powershell
uv run --env-file .env python -m experiments.summary_pilot --live --provider openrouter --case 02_related --output runs/summary-related-openrouter
```

Use `--provider openai` (the default) with `OPENAI_API_KEY` for OpenAI. For either
provider, `--model` and `--base-url` override its defaults. The manifest records
the provider, model, URL, and credential variable name, never the credential value.
The script reads credentials only for live execution. A provider must support Chat Completions
call and temperature 0. No live pilot is run by unit tests or module imports.

Omit `--output` to create a timestamped directory. Existing output directories
are refused to avoid mixing memories from runs. A dry-run directory cannot be
reused for a live run. Five cases require at most 47 summary requests, excluding
the SDK's up to two retries per request. There are no extra model-judge calls.
Use `--case` repeatedly to select cases. Run separate output directories for
replicates; temperature 0 does not guarantee identical results.

## Outputs and review

The run saves model/settings, SDK/Python versions, prompts, timestamp, git commit
(including the dirty marker), and exact synthetic inputs. Each case saves:

- `case.json` and `never_learned_input.json`: target, retained facts, dated turns,
  omitted target-turn indices, and review notes.
- `pre_forget.txt`, `post_forget.txt`, `never_learned.txt`: complete summaries.
  Post-Forget starts from a copy of Pre-Forget; Never-Learned starts empty and
  receives the conversations with only annotated explicit target turns omitted.
- `pre_vs_post_forget.diff` and `pre_vs_never_learned.diff`: text differences for
  inspection, without interpreting them as semantic scores.
- `result.json`: available snapshots, completion status, and stage-specific errors.
  Partial memory remains on disk for diagnosis, but is not a completed snapshot.
- `review.json`: initially unreviewed semantic annotations. Set presence values
  to JSON `true`/`false`; leave `null` for uncertainty. Add supporting quotations
  and comments in `notes`. Check all three summaries against each predefined fact.
- `assessment.json`: initially pending; classifications after manual annotation.

After editing reviews, recompute assessments offline:

```powershell
uv run python -m experiments.summary_pilot --review-run runs/summary-pilot-live
```

Use the following distinctions when reviewing:

| Finding | Evidence required |
| --- | --- |
| Memory formation failure | Target absent from Pre-Forget; forgetting is not evaluable for that case. |
| Forgetting failure | Direct target or a semantic paraphrase present both before and after forgetting. |
| Collateral information loss | Expected retained fact present before forgetting and absent afterward. |
| Retention failure during formation | Expected retained fact already absent before forgetting. |
| Potential inferential leakage | Target absent directly after forgetting, but plausibly inferable from retained facts. |

In case 2, independently stated tofu preference and meat avoidance must survive.
These can allow inference of vegetarianism; that is distinct from explicit diet
identity remaining in memory. Document ambiguous paraphrases rather than forcing
a binary judgement. In case 3, childhood in Malaysia is a direct equivalent of
growing up there. In case 4, a retained location without its date is incomplete.
In case 5, check the corrected cycling day, repeated facts, deadline, and whether
assistant rock-climbing suggestions were incorrectly stored.

Never-Learned can itself infer and explicitly state the target from clues. Mark
that in `never_learned_direct` and record the evidence. It is a useful control,
not a guaranteed target-free oracle. Inspect hallucinations, date drift, and
unrequested changes in all three branches. Failed/incomplete cases are skipped
by the offline assessment command; fix the cause and rerun in a fresh directory.

## Limits

### Corrections and forgetting have different scopes

`write()` processes new information; `forget()` removes a specified proposition.
Do not pass a new status statement to `forget()` and expect it to perform a
correction. The caller selects the operation; `SummaryMemory` does not route
natural-language messages between them.

| User intent | Operation and expected treatment |
| --- | --- |
| "I am no longer vegetarian." | `write()`: update the current identity; do not invent a specific new diet. Independently stated related facts need their own contradiction to be updated. |
| "I am no longer vegetarian; I eat meat now." | `write()`: update both current vegetarian identity and current meat avoidance. Historical facts may remain explicitly dated if relevant. |
| "Forget that I am vegetarian." | `forget()`: remove that identity and its direct paraphrases within scope. Under this pilot's original policy, independently stated meat avoidance and tofu preference remain. |
| "Forget my vegetarian identity and that I avoid meat." | `forget()`: explicitly target both facts; keep unrelated information. Do not write that the user now eats meat. |

If "does not eat meat" was generated only as a restatement of vegetarianism, it
should be removed as a target paraphrase. If it was independently supplied by the
user, the original related-facts case requires keeping it. Once both are merged
into an unannotated summary, their origin can be indistinguishable. Prompt rules
cannot guarantee this distinction without provenance; this baseline deliberately
does not add a provenance store.

### Failure and leakage checks

| Issue | What this baseline can do and what still needs evaluation |
| --- | --- |
| Direct/paraphrase leakage | Ask regeneration to remove the proposition and direct equivalents, including in-scope history and deletion explanations; inspect full snapshots semantically. |
| Inferential leakage | Preserve independently stated clues under the agreed selective policy and report inference separately. Eliminating all inference can require deleting facts the task says to retain. |
| Collateral loss | Ask for minimal changes; compare each non-target fact before and after. This is unintended deletion, rather than leakage. |
| Reconstruction/relearning | Report when retained clues or later conversations restore the target. No deletion ledger blocks relearning. |
| Behavioural leakage | Test answers separately on disposable memory copies. An absent stored sentence does not guarantee the agent will not infer or guess it. |
| Apparent deletion through truncation | Inspect `dump()`, since a target outside the retrieval prefix still exists in memory. |
| Artifact leakage | Pre-Forget snapshots and synthetic inputs intentionally retain targets for evaluation; active-memory deletion does not erase those files. |

This is a small qualitative pilot, not a benchmark success estimate. A summariser
can lose facts before deletion, rewrite unrelated details, or leave paraphrases.
Per-turn calls increase cost and may lose conversational references such as an
isolated "yes" because prior raw turns are not retained. Prefix retrieval can
hide information still present in `dump()`; inspect the full snapshots first.
Later conversations can reintroduce a fact because there is no deletion ledger.

Text-level absence does not demonstrate behavioural forgetting. This pilot does
not run chat-response probes. The shared `Agent.chat()` can remember new facts,
so any later behavioural evaluation should use disposable copies and inspect
`agent.remembered` separately. SummaryMemory stores no raw-session archive; the
pilot intentionally keeps synthetic inputs and pre-deletion snapshots outside
the active memory as evaluation artifacts. Those artifacts still contain targets.
