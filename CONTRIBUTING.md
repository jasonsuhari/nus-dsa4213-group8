# Working on this repo

Group 8: Jason Suhari, Jordan Koh Shi Rong, Lee Ang Xuan, Mandy Yap Zhi Wei, Ong Wei Lun

## Commit under your own account

The syllabus lists version histories as evidence the instructor may use when
adjusting individual marks away from the group mark. If we all commit from one
laptop, that evidence is gone. Check your identity before your first commit:

```bash
git config user.name "Your Name"
git config user.email "your@email.com"   # the one on your GitHub account
```

When pairing, add a `Co-authored-by:` trailer.

## Setup

```bash
uv sync
uv run pre-commit install
cp .env.example .env
```

## Workflow

Branch, PR, squash merge. No direct pushes to `main`.

```bash
git switch -c wei-lun/eval-harness   # <name>/<what>
uv run pytest && uv run ruff check .
git push -u origin HEAD
gh pr create
```

## Layout

| path | what |
|---|---|
| `src/dsa4213/` | agent, memories, evaluation. All real logic lives here |
| `experiments/` | runnable scripts and the cases |
| `runs/` | gitignored. Each run writes `runs/<run_id>/`, and `run_forgetting` appends a line to `runs/runs.jsonl` |
| `results/` | committed copies of the runs we report, and every other run under `results/other-runs/` |
| `data/` | gitignored scratch output of the live scripts |

## Logging runs

`experiments/run_forgetting.py` calls `log_run` (`src/dsa4213/runlog.py`) at the end of every
run, which records the run id, commit, model and answer counts in `runs/runs.jsonl`. To keep a
run, copy its folder into `results/` and say what it was in the README there.

Agent runs are not bit-reproducible. LLM sampling and rolling model snapshots
see to that. Log the model and its snapshot date, fix seeds where you can,
repeat runs and report the variance, and state the limitation in the report
rather than glossing over it.
