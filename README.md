# Fact-level forgetting in LLM agent memory

NUS DSA4213 Group 8: Jason Suhari, Jordan Koh Shi Rong, Lee Ang Xuan, Mandy Yap Zhi Wei,
Ong Wei Lun.

A chat agent remembers a user across sessions through a memory. The user then asks it to
forget one fact. We test whether it does: whether the agent still reveals the fact when asked
directly, indirectly or adversarially, and whether forgetting also destroys facts it should
have kept. We compare three memories behind one interface, embedding retrieval, a rolling
summary and a key-value profile, plus a fact-level variant of embedding, a no-memory control
and a full-transcript control. The test set is 56 synthetic multi-session cases. No real
user's data is used.

## Where each number in the report comes from

| report | what | file |
|---|---|---|
| 5.1 | the 56 cases and 634 probes | `experiments/cases/*.json` |
| 5.2 | judge check: gpt-4o against 60 labelled leakage answers, kappa 0.75 to 0.79 | `experiments/judge_check.jsonl`, rerun with `experiments/check_judge.py` |
| 5.3, Table "Overall evaluation results" | PRA, eligible cases, DLR, SLR, ALR, RA, RIR per memory | `results/metrics.csv`, rows with `category` = `all` |
| 5.3, controls | null and full transcript; leakage on the never-learned history | `results/metrics.csv`, rows `null` and `full`, columns `*_never` |
| 5.4, Fig. 5.4-1 | paired differences on the 39 cases all three memories learned | `results/paired.csv` |
| 5.4, Fig. 5.4-2 | results by case family | `results/metrics.csv`, rows per `category`; the highly related panel from `results/scored.jsonl.gz` |
| 5.5, Table 5.5-1 | fact-level vs turn-level embedding in one run | `results/fact-level/metrics.csv`, `paired.csv`, `retention.csv` |
| 6.1 | the case examples (`entangled-tesla`, `echo-daughter-ava`, `entangled-engaged`, `diet`, `home`, `paraphrase-muslim`) | each store and forget log in `results/stores/<case>/`; the answers in `results/responses.jsonl.gz` |
| 6.2, Table 6.2-1 | highly related retention 0.96 → 0.44; echo and long-history rates | `results/retention.csv`; `results/metrics.csv` per category |
| 6.5, Table 6.5-1, Fig. 6.5-1 | records stored, removed by one forget, share of text removed, key-value facts per turn | `results/forget_reach.csv`, from `experiments/forget_reach.py`. The calls per operation are read from the code in `src/dsa4213/memory/` |

`results/README.md` and `results/fact-level/README.md` describe every file and give the
headline tables.

## Layout

| path | what |
|---|---|
| `src/dsa4213/agent.py` | the chat agent. Every memory plugs in through `write`, `retrieve`, `forget` and `dump` (`src/dsa4213/memory/base.py`) |
| `src/dsa4213/memory/` | the memories: `embedding.py` (turn-level), `fact_embedding.py` (fact-level), `summary.py`, `kv.py`, and the null and full-transcript controls in `reference.py` |
| `src/dsa4213/evaluation/` | the gpt-4o judges (`llm_judge.py`) and the metrics with eligibility (`metrics.py`) |
| `src/dsa4213/cases.py` | loads and validates the cases |
| `experiments/run_forgetting.py` | runs the experiment |
| `experiments/score_forgetting.py` | judges the answers and writes the metric tables |
| `experiments/forget_reach.py` | measures what one forget request removed from each store |
| `experiments/check_judge.py` | checks the judge against `judge_check.jsonl` |
| `experiments/chat_live.py`, `test_*_live.py` | interactive chat and the live smoke tests each memory was built with |
| `results/` | the main run (9,510 answers) |
| `results/fact-level/` | the fact-level run behind Section 5.5 (3,804 answers) |
| `results/other-runs/` | every other experiment run, with a README saying what each was. None is used in the report |
| `tests/` | offline tests. They make no API calls |

## Setup

Needs [uv](https://docs.astral.sh/uv/) and Python 3.11 or later.

```bash
uv sync
cp .env.example .env    # then put your key in OPENAI_API_KEY
uv run pytest           # offline, no key needed
```

The main run went through OpenRouter: it set `OPENAI_BASE_URL=https://openrouter.ai/api/v1`
and an OpenRouter key as `OPENAI_API_KEY` in `.env`. The fact-level run used the OpenAI API
directly, with only `OPENAI_API_KEY` set.

## Rerun the experiment

```bash
uv run --env-file .env python -m experiments.run_forgetting \
    --only embedding --only summary --only kv --only null --only full
uv run --env-file .env python -m experiments.score_forgetting runs/<run_id>
```

The first command is the main run's configuration: 56 cases × 5 memories, with gpt-4o-mini
as the agent and inside the memories, and text-embedding-3-small for embeddings. It writes
`runs/<run_id>/responses.jsonl`, every store under `runs/<run_id>/stores/`, and
`summary.json`; the main run took about 15 minutes. Without
`--only` it also runs `embedding_fact`. Add `--case diet` to try one case. The second command
judges every answer with gpt-4o and writes the tables. Answers will differ a little from ours:
the models are sampled at temperature 0 but are not deterministic.

## Rescore the saved answers without an API key

`score_forgetting` reuses the judge labels in `scored.jsonl` when it is present, so it makes
no API calls:

```bash
mkdir -p runs/main
cp results/summary.json runs/main/
gunzip -c results/scored.jsonl.gz > runs/main/scored.jsonl
uv run python -m experiments.score_forgetting runs/main
diff runs/main/metrics.csv results/metrics.csv    # no output: identical
```

The same works for `results/fact-level/`. To judge the answers again from scratch, unpack
`responses.jsonl.gz` instead of `scored.jsonl.gz` and set an API key.

To recompute the Section 6.5 store measurements:

```bash
uv run python -m experiments.forget_reach results/stores results/fact-level/stores
```

## Who wrote what

The memories and evaluation came in as pull requests, so `git log` and the PR list show each
member's work: embedding memory by Jordan (#9) and its fact-level variant (#17), summary
memory by Ang Xuan (#7, #8, #13), key-value memory by Wei Lun (#12), the evaluation framework
by Mandy (#10), and the agent, cases, runner and scorer by Jason (#4, #14, #15).
`CONTRIBUTING.md` has the team's working rules.

## Limits

- One run per memory with one agent model, and each probe answered once.
- The cases are synthetic. They fix where each fact is stated, but are not real chats.
- The judge's labels are checked against 60 labelled answers, not against two independent
  human raters; the two-rater sheet `results/annotation.csv` was never filled in.
- Forgetting here removes text from a memory store. It does not change model weights or
  anything already sent to the model provider.
