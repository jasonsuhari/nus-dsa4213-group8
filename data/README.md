Datasets are gitignored. Record what belongs here and where it came from, so a
fresh clone can rebuild it.

| file | source | notes |
|---|---|---|
| `memora/<split>/<persona>.jsonl` | `git clone https://github.com/geniesinc/Memora.git`, then `scripts/convert_memora.py` | 10 personas, ~155 sessions each on weekly. Keeps Memora's `operation` and `share_memory` fields |
| `stores/<approach>/` | `uv run python scripts/build_memory.py` | each has a `manifest.json` with the source, sha and mint time |
