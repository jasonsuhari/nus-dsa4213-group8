from dsa4213.runlog import load_runs, log_run


def test_roundtrip_and_append(tmp_path):
    p = tmp_path / "runs.jsonl"
    a = log_run(p, model="claude-opus-5", score=0.8)
    b = log_run(p, model="claude-sonnet-5", score=0.6)

    runs = load_runs(p)
    assert [r["run_id"] for r in runs] == [a["run_id"], b["run_id"]]
    assert runs[0]["model"] == "claude-opus-5"
    assert all(r["commit"] and r["timestamp"] for r in runs)


def test_missing_file_is_empty(tmp_path):
    assert load_runs(tmp_path / "nope.jsonl") == []
