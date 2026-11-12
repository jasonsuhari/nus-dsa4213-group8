# Other runs

The report's numbers come from `results/` (main run) and `results/fact-level/` (Section 5.5).
Everything here is the other experiment runs we made on the way, kept so every run is on
record. None of these numbers is in the report.

Each folder has the run's `summary.json` (model, commit, per-case status) and, when the run
was scored, the same tables as `results/` plus `scored.jsonl.gz`, which holds every answer
with its judge label. Unscored runs have `responses.jsonl.gz` instead. `stores.tar.gz` holds
the run's stores with each record's embedding vector dropped. Most commit ids in
`summary.json` were made on local branches before a squash merge, so they are not on GitHub.
A `-dirty` suffix means the run had uncommitted changes on top of that commit.

## Runs of the code on main

| run | memories | answers | what it was |
|---|---|---|---|
| `9f50e79e` | all five | 9,510 | The first full run. Its embedding, key-value, null and full answers are the main results. Its summary arm ran an older `summary.py` (PRA 0.98, 55 of 56 cases learned) and was replaced by `887556c0` |
| `887556c0` | summary | 1,902 | Summary rerun with the `summary.py` now on main. These are the main results' summary answers. Its stores are in `results/stores/` |
| `43a7aacf` | summary | 867 | Summary as merged at the time: 31 of 56 cases crashed with "LLM returned no valid summary text". Not scorable, since the scorer refuses runs with failed cases |
| `3bf4f7ec` | summary | 1,902 | Summary with only that crash fixed: it learned 18 of 56 cases (PRA 0.32). The prompt fix in `887556c0` brought this to 47 |

## Runs of memories that are not on main

These ran code that is not on GitHub. The earlier fact-level memory (`embedding_facts.py`) and
the forget replays are on Jason's unpushed local branches `jason/file-memory` and
`jason/checked-forget`. The file memories (`files.py`) were never committed. So these runs
cannot be rerun from this repo; their answers, labels and stores are here as a record.

| run | memories | answers | what it was |
|---|---|---|---|
| `cf14839e` | embedding_fact | 1,902 | An earlier fact-level embedding (`embedding_facts.py`, delete only, no redaction). Superseded by Jordan's `fact_embedding.py` in `results/fact-level/`. Unscored on its own; see the next row |
| `compare-fact-cf14839e` | embedding, embedding_fact | 3,804 | `cf14839e` scored beside the main run's turn-level embedding (fact-level PRA 0.96, DLR 0.10, RA 0.89) |
| `combined-9f50e79e-887556c0-cf14839e` | six | 11,412 | The main results plus `cf14839e`, scored together for six-way paired comparisons |
| `7d41e4de` | files, files_topic | 3,804 | Two file-based memories that keep facts in named markdown files, one file per fact or per topic. One case, `echo-red-honda-civic/files_topic`, stopped early and was retried as run `f1aa4a3d` (recorded in `summary.json`) |
| `combined-files-7d41e4de` | eight | 15,216 | The six above plus the two file memories, scored together |
| `replay-fixA-7d41e4de` | files, files_topic | 2,536 | Replays only `forget()` on `7d41e4de`'s stores with two changed forget steps: A asks the model to also rename the file, B regenerates the file's name and description in a separate call. The replay script imported `files.py`, so it is not included. `replay_scored.json.gz` has the post-forget answers; `retry_runs.tar.gz` holds the three one-case retries `f1aa4a3d`, `4204e289` and `884568d0` |
| `checked-forget-replays/` | embedding_fact | 316 or 948 per arm | Replays only `forget()` on `cf14839e`'s fact-level stores, on 28 tuning cases, once per forget-step variant (one folder per arm, `scored.json.gz` has the 316 post-forget answers, plus the pre-forget and never-learned answers for the two `k30` arms, which change retrieval, `forget_stats.json` the calls and records removed per case). The `table-tune-*.csv` files compare the arms |

## Left out

- `smoke/6f872b6d`: a one-case smoke test of the fact-level memory (30 answers).
- Runs logged in `results/runs.jsonl` whose folders no longer exist: pilot and
  full-suite runs made before `9f50e79e` (`ebf9d16d`, `8cab130d`, `f50cc6a0`, `678dce7f`,
  `8fe74557`, `25957b8d`, `575e65cb`), one-case and smoke runs (`5849f12f`, `b96e2691`,
  `7285fbc9`, `3811cb58`), and runs that wrote no answers (`4174236d`, `0ae9c73b`,
  `93912901`, `eb3ac0a5`, `ad1d3c8a`, `ad38a185`). `5849f12f` is the one-case summary refill
  that `887556c0` and the main results include, recorded in their `summary.json`.

`results/runs.jsonl` is the log `run_forgetting` appends to, merged from every machine
that ran it.
