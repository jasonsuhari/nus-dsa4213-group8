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
| `src/dsa4213/` | agent, tools, eval harness. All real logic lives here |
| `notebooks/` | thin and exploratory. Import from `src/`, don't define logic there |
| `experiments/` | runnable scripts, one per experiment |
| `runs/runs.jsonl` | append-only log of every run (committed) |
| `data/` | gitignored. `data/README.md` records how to rebuild it |
| `docs/` | proposal and report drafts |

Keep logic out of notebooks. Five people editing one notebook is a merge
conflict factory, and notebook diffs are unreadable even when they do merge.

## Logging runs

One line per run. This is what the evaluation, cost, and reproducibility
sections of the report get written from.

```python
from dsa4213.runlog import log_run

log_run(task="tool_use_bench", model="claude-opus-5", temperature=0.0,
        seed=42, n_tokens=12_400, cost_usd=0.18, latency_s=6.2, score=0.71)
```

`run_id`, `timestamp` and the git commit are filled in automatically.

Agent runs are not bit-reproducible. LLM sampling and rolling model snapshots
see to that. Log the model and its snapshot date, fix seeds where you can,
repeat runs and report the variance, and state the limitation in the report
rather than glossing over it.
