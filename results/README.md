# Final results

The experiment the presentation reports. All 56 cases in `experiments/cases/` were run
through five memories: null, embedding, summary, key-value, and full transcript. Each
memory saw each case three ways:

- `pre_forget`: the chat was learned.
- `post_forget`: the chat was learned, then `forget()` got the user's request.
- `never_learned`: the history had the target fact edited out.

That gives 634 probe questions × 5 memories × 3 versions = 9,510 answers. The agent was
gpt-4o-mini at temperature 0 over OpenRouter, and every answer was scored by a gpt-4o judge.

## Files

| file | what |
|---|---|
| `responses.jsonl.gz` | all 9,510 answers, with the probe, the memory context the agent saw, and the answer key. Unscored. `gunzip` it into a run folder to rescore |
| `metrics.csv` | PRA, DLR, RA, LE, RE, RIR, SLR, ALR, MLS per memory: overall with 95% bootstrap intervals over cases, then per case category. `*_never` columns are the same leak rates on the never-learned history |
| `retention.csv` | retention accuracy before and after forgetting, split by how related the question is to the forgotten fact |
| `paired.csv` | each pair of memories compared on the 39 cases all three learned. `separated` is true when the 95% interval of the difference excludes 0 |
| `annotation.csv` | 336 leakage answers for the two human raters, blind to memory, condition and judge score |
| `summary.json` | which run each memory's answers came from, the model, and the per-case status |

`annotation_key.csv` and `scored.jsonl` hold the judge's label for every answer, so they
stay out until the human rating is done. Otherwise a rater could look up the judge's score.

## Headline numbers

Eligible means the memory recalled the fact before forgetting (PRA above 0.6 on that case).
Leak rates are over eligible cases only.

| memory | PRA | eligible | DLR | SLR | ALR | RA | RIR |
|---|---|---|---|---|---|---|---|
| embedding | 0.97 | 56 | 0.17 | 0.14 | 0.23 | 0.72 | 0.22 |
| summary | 0.85 | 47 | 0.18 | 0.15 | 0.26 | 0.92 | 0.34 |
| key-value | 0.77 | 45 | 0.04 | 0.07 | 0.11 | 0.70 | 0.17 |
| full transcript | 0.99 | 56 | 0.99 | 0.89 | 1.00 | 0.97 | 0.96 |

Null recalled nothing, so it has no eligible cases. The leak rate on the never-learned
history is about 0.03 direct and 0.09 adversarial for every memory. That is the floor: what
the agent guesses from the facts it was allowed to keep.

Retention on questions closely tied to the forgotten fact, before → after forgetting:
embedding 0.96 → 0.44, summary 0.88 → 0.79, key-value 0.69 → 0.64.

## What the paired comparison supports

On the 39 cases that embedding, summary and key-value all learned:

- Key-value leaks less than both others on direct (DLR), adversarial (ALR) and severity (MLS).
  On semantic leaks (SLR) it is clearly below embedding but not separable from summary.
- Embedding and summary cannot be told apart on any leak metric.
- Summary keeps more of the user's other facts than both others after forgetting (RA),
  and more of the facts closely tied to the forgotten one (`RA_highly_related`).
- No pair separates on residual influence (RE) or learning effect (LE).

So there is no single best memory. Key-value forgets most cleanly but learns the fewest
facts (PRA 0.77) and keeps fewer of the others. Summary keeps the most but leaks like
embedding. Embedding learns almost everything but forgetting takes nearby facts with it.

## By case category

| category | DLR emb / sum / kv | ALR emb / sum / kv | PRA emb / sum / kv |
|---|---|---|---|
| echo | 0.53 / 0.30 / 0.15 | 0.70 / 0.40 / 0.22 | 0.97 / 1.00 / 0.87 |
| entangled | 0.20 / 0.20 / 0.00 | 0.10 / 0.20 / 0.11 | 0.93 / 1.00 / 0.87 |
| long_history | 0.00 / 0.13 / 0.00 | 0.10 / 0.20 / 0.00 | 0.97 / 0.47 / 0.70 |
| near_miss | 0.10 / 0.12 / 0.00 | 0.10 / 0.12 / 0.14 | 1.00 / 0.83 / 0.70 |
| paraphrase | 0.10 / 0.25 / 0.07 | 0.20 / 0.38 / 0.11 | 0.97 / 0.87 / 0.87 |
| pilot | 0.00 / 0.00 / 0.00 | 0.17 / 0.17 / 0.00 | 1.00 / 0.94 / 0.50 |

Categories have 6 to 10 cases each, fewer once ineligible ones drop out, so treat these
as where to look, not as results. Two stand out: embedding leaks most on echo cases, where
the fact is repeated across several turns, and on entangled cases embedding's closely
related retention drops from 0.97 to 0.14. Summary learns only
0.47 of long_history facts.

## How these were made

```bash
uv run --env-file .env python -m experiments.run_forgetting   # writes runs/<run_id>/responses.jsonl
uv run --env-file .env python -m experiments.score_forgetting runs/<run_id>
```

Embedding, key-value, null and full come from run `9f50e79e`. Summary comes from run
`887556c0`, rerun after the summary write fixes in `src/dsa4213/memory/summary.py`. Between
the two runs only `summary.py` and the judge's retry on malformed JSON changed, so mixing
them is fair. One summary case (`echo-low-sodium-kidney`) failed once with "LLM did not
complete the summary normally" and passed on a rerun; `summary.json` records it. Both runs
predate the squash merge, so the commit ids in `summary.json` are not on GitHub. The code is
what main has now.

## Limits

- The judge is gpt-4o. On 60 hand-labelled leakage answers (`experiments/judge_check.jsonl`)
  it agrees with the labels at kappa 0.75 to 0.79. The two-rater check on `annotation.csv`
  has not been done.
- The summary memory still wipes its own summary now and then: 7 times in about 1,034 turns.
  This lowers its PRA and is a finding about summary memory, not a bug in the runner.
- RIR is RE / LE, which swings when LE is small. Read RE and LE beside it.
- One run per memory, one agent model. No variance across seeds or models.
