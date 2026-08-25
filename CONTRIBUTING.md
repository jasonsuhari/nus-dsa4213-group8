# Working on this repo

**Group 8** — Jason Suhari, Jordan Koh Shi Rong, Lee Ang Xuan, Mandy Yap Zhi Wei, Ong Wei Lun

## The one rule that affects your grade

Everyone commits under their own GitHub account. The syllabus lists *version
histories* as evidence the instructor may use to adjust individual marks away
from the group mark. Pair-programming off one laptop all semester destroys that
evidence. If you do pair, use `Co-authored-by:` in the commit message.

```bash
git config user.name "Your Name"
git config user.email "your@email.com"   # the one on your GitHub account
```

## Setup

```bash
uv sync                    # installs deps + dev tools into .venv
uv run pre-commit install  # hooks: ruff, nbstripout, secret detection
cp .env.example .env       # then fill in your API keys
```

## Workflow

`main` is protected — no direct pushes. Branch, PR, one review, squash merge.

```bash
git switch -c wei-lun/eval-harness   # <name>/<what>
# work
uv run pytest && uv run ruff check .
git push -u origin HEAD
gh pr create
```

## Layout

| path | what |
|---|---|
| `src/dsa4213/` | agent, tools, eval harness — all real logic lives here |
| `notebooks/` | thin, exploratory; import from `src/`, don't define logic |
| `experiments/` | runnable scripts, one per experiment |
| `runs/runs.jsonl` | append-only log of every run (committed) |
| `data/` | gitignored; `data/README.md` says how to rebuild it |
| `docs/` | meeting notes, proposal/report drafts |

**Keep logic out of notebooks.** They diff badly and five people editing one
notebook is a merge conflict factory. Notebooks call `src/`, nothing more.

## Logging runs

Every experiment run gets one JSONL line. This is not bookkeeping for its own
sake — it feeds evaluation evidence, cost analysis, and reproducibility, which
are 14% of the course between them.

```python
from dsa4213.runlog import log_run

log_run(
    task="tool_use_bench",
    model="claude-opus-5",
    temperature=0.0,
    seed=42,
    n_tokens=12_400,
    cost_usd=0.18,
    latency_s=6.2,
    score=0.71,
)
```

`run_id`, `timestamp`, and the git commit are recorded for you.

Agent runs are not bit-reproducible — LLM sampling and rolling model snapshots
see to that. Log the model *and* its snapshot date, fix seeds where you can,
report variance across repeats, and say so plainly in the report's
Reproducibility section. Stating the limit reads as rigour.

## Meeting notes

One file per meeting in `docs/meetings/YYYY-MM-DD.md`. Same reason as the
commit rule: meeting records are named in the syllabus as grading evidence.
