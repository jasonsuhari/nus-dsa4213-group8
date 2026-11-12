# Fact-level embedding ablation (report Section 5.5)

Run `d906d91f` put fact-level embedding (`embedding_fact`, `src/dsa4213/memory/fact_embedding.py`)
and turn-level embedding (`embedding`) through all 56 cases in the same run, so the two can be
compared case by case. That is 634 probes × 2 memories × 3 versions = 3,804 answers. The agent
was gpt-4o-mini at temperature 0 through the OpenAI API, and every answer was scored by the
same gpt-4o judge as the main run.

The run used commit `720f22b`, which was squash merged into main as PR #17. The only later
change to the code it ran is PR #18, which edits two comments.

## Files

| file | what |
|---|---|
| `responses.jsonl.gz` | all 3,804 answers with the probe, the memory context the agent saw, and the answer key |
| `scored.jsonl.gz` | the same answers with the judge's label on each |
| `metrics.csv` | PRA, DLR, RA, LE, RE, RIR, SLR, ALR, MLS for both memories, overall and per case category |
| `retention.csv` | retention before and after forgetting, by how related the question is to the target |
| `paired.csv` | turn-level minus fact-level on all 56 cases, with 95% paired bootstrap intervals |
| `annotation.csv`, `annotation_key.csv` | the blind rating sample `score_forgetting` always writes. Not rated |
| `summary.json` | model, commit and per-case status |
| `stores/` | every store, laid out as in `results/stores/`. Fact-level `forget_log.jsonl` lists each pass's deleted and redacted facts, with the text before and after each redaction |

## Numbers in the report's Table 5.5-1

| metric | turn-level | fact-level | turn minus fact, 95% interval |
|---|---|---|---|
| PRA | 0.970 | 0.976 | |
| DLR | 0.143 | 0.083 | +0.060 [-0.024, 0.161] |
| SLR | 0.161 | 0.089 | +0.071 [-0.036, 0.179] |
| ALR | 0.196 | 0.214 | -0.018 [-0.143, 0.107] |
| RA | 0.737 | 0.898 | -0.161 [-0.234, -0.084] |
| highly related kept, before → after | 0.963 → 0.488 | 0.950 → 0.825 | -0.337 [-0.476, -0.183] |
| RIR | 0.244 | 0.171 | |

Both memories learned all 56 cases. Only RA and highly related retention separate. The
turn-level arm here leaks less than turn-level embedding in the main run (DLR 0.143 against
0.167). The embedding and agent code and the cases are identical between the two runs; the
main run went through OpenRouter and this one through the OpenAI API. Read differences of that
size as run-to-run noise.

## Rescoring

```bash
mkdir -p runs/fact-level
cp results/fact-level/summary.json runs/fact-level/
gunzip -c results/fact-level/scored.jsonl.gz > runs/fact-level/scored.jsonl
uv run python -m experiments.score_forgetting runs/fact-level
```
