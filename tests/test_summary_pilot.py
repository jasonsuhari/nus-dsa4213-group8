"""Offline checks for pilot isolation, outputs, and honest review classifications."""

import json
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def pilot(monkeypatch):
    class NoRealClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError("Offline tests must not create an OpenAI client")

    monkeypatch.setattr("openai.OpenAI", NoRealClient)
    path = Path(__file__).resolve().parents[1] / "experiments" / "summary_pilot.py"
    return SimpleNamespace(**runpy.run_path(str(path)))


def completion(text):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop", message=SimpleNamespace(content=text, refusal=None)
            )
        ]
    )


def test_cases_omit_only_annotated_target_turns(pilot):
    assert len(pilot.CASES) == 5
    for case in pilot.CASES:
        never = case.never_learned_sessions()
        assert [(s.session_id, s.date) for s in never] == [
            (s.session_id, s.date) for s in case.sessions
        ]
        removed = 0
        for original, ablated in zip(case.sessions, never, strict=True):
            expected = tuple(
                t
                for i, t in enumerate(original.turns)
                if (original.session_id, i) not in case.omitted_turns
            )
            assert ablated.turns == expected
            removed += len(original.turns) - len(ablated.turns)
        assert removed == len(case.omitted_turns)
    related = pilot.CASES[1].never_learned_sessions()[0].transcript
    assert "I avoid meat." in related
    assert "I enjoy tofu." in related
    assert "I am vegetarian." not in related


def test_default_execution_only_writes_plan(pilot, tmp_path):
    output = tmp_path / "plan"
    assert pilot.main(["--output", str(output)]) == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["live"] is False
    assert manifest["planned_requests"] == sum(case.planned_requests() for case in pilot.CASES)
    assert not list(output.glob("*/summary.txt"))
    with pytest.raises(FileExistsError):
        pilot.main(["--output", str(output)])


def test_three_snapshots_are_built_independently(pilot, tmp_path):
    client = MagicMock()
    client.chat.completions.create.side_effect = [completion(f"summary {i}") for i in range(6)]
    case = pilot.CASES[0]
    path = tmp_path / case.case_id
    report = pilot.run_case(case, path, client, "test-model")
    assert report["complete"] is True
    assert report["errors"] == []
    assert (path / "pre_forget.txt").read_text() == "summary 2"
    assert (path / "post_forget.txt").read_text() == "summary 3"
    assert (path / "never_learned.txt").read_text() == "summary 5"
    assert (path / "pre_memory" / "summary.txt").read_text() == "summary 2"
    assert (path / "pre_vs_post_forget.diff").exists()
    calls = client.chat.completions.create.call_args_list
    assert "Existing summary:\nsummary 2" in calls[3].kwargs["messages"][1]["content"]
    assert "Existing summary:\n(empty)" in calls[4].kwargs["messages"][1]["content"]
    assessment = json.loads((path / "assessment.json").read_text())
    assert assessment["forgetting"] == "pending_review"
    assert assessment["memory_formation_failure"] is None


@pytest.mark.parametrize("failure_at", [1, 3])
def test_api_error_is_recorded_and_never_learned_still_runs(pilot, tmp_path, failure_at):
    client = MagicMock()
    results = [completion(f"summary {i}") for i in range(6)]
    results[failure_at] = RuntimeError("simulated API error")
    client.chat.completions.create.side_effect = results
    path = tmp_path / "case"
    report = pilot.run_case(pilot.CASES[0], path, client, "test-model")
    assert not report["complete"]
    assert report["errors"][0]["stage"] == ("pre_forget" if failure_at == 1 else "post_forget")
    assert "never_learned" in report["snapshots"]
    assert "post_forget" not in report["snapshots"]
    assert (path / "result.json").exists()


def test_manual_review_distinguishes_failure_types(pilot):
    review = pilot.review_template(pilot.CASES[1])
    review["target"].update(pre_direct=False, post_direct=False)
    assert pilot.assess_review(review)["forgetting"].startswith("not_evaluable")
    review["target"].update(pre_direct=True, post_direct=True)
    assert pilot.assess_review(review)["forgetting"] == "failure"
    review["target"].update(post_direct=False, post_inferable_from_retained_facts=True)
    review["retained"][0].update(pre=True, post=False)
    review["retained"][1].update(pre=False, post=False)
    assessment = pilot.assess_review(review)
    assert assessment["forgetting"] == "no_direct_target_observed"
    assert assessment["potential_inferential_leakage"] is True
    assert assessment["collateral_information_loss"] == [review["retained"][0]["fact"]]
    assert assessment["retained_facts_missing_before_forget"] == [review["retained"][1]["fact"]]


def test_old_live_script_is_import_safe(monkeypatch):
    def no_real_client(*args, **kwargs):
        raise AssertionError("Import must never create a client")

    monkeypatch.setattr("openai.OpenAI", no_real_client)
    path = Path(__file__).resolve().parents[1] / "experiments" / "test_summary_live.py"
    runpy.run_path(str(path))


def test_offline_review_command_updates_assessments(pilot, tmp_path):
    directory = tmp_path / "01_simple"
    directory.mkdir()
    review = pilot.review_template(pilot.CASES[0])
    review["target"].update(pre_direct=True, post_direct=False)
    for fact in review["retained"]:
        fact.update(pre=True, post=True, never_learned=True)
    pilot.write_json(directory / "review.json", review)
    pilot.write_json(directory / "result.json", {"complete": True})
    assert pilot.main(["--review-run", str(tmp_path)]) == 0
    assessment = json.loads((directory / "assessment.json").read_text())
    assert assessment["forgetting"] == "no_direct_target_observed"
    assert assessment["collateral_information_loss"] == []
    assert assessment["retained_facts_awaiting_review"] == []


def test_offline_review_rejects_non_boolean_annotations(pilot, tmp_path):
    directory = tmp_path / "01_simple"
    directory.mkdir()
    review = pilot.review_template(pilot.CASES[0])
    review["target"]["pre_direct"] = "false"
    pilot.write_json(directory / "review.json", review)
    pilot.write_json(directory / "result.json", {"complete": True})
    with pytest.raises(SystemExit) as error:
        pilot.main(["--review-run", str(tmp_path)])
    assert error.value.code == 2


@pytest.mark.parametrize(
    "provider,key_env,expected_model,expected_url",
    [
        ("openai", "OPENAI_API_KEY", "gpt-4o-mini", "https://test.example/v1"),
        ("openrouter", "OPENROUTER_API_KEY", "openai/gpt-4o-mini", "https://openrouter.ai/api/v1"),
    ],
)
def test_provider_selects_credentials_and_defaults_without_logging_key(
    pilot, tmp_path, monkeypatch, provider, key_env, expected_model, expected_url
):
    monkeypatch.setenv(key_env, "pilot-test-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://test.example/v1")
    received = {}

    class FakeClient:
        def __init__(self, **kwargs):
            received.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def fake_run_case(case, directory, client, model):
        assert model == expected_model
        return {"case_id": case.case_id, "complete": True}

    monkeypatch.setitem(pilot.main.__globals__, "OpenAI", FakeClient)
    monkeypatch.setitem(pilot.main.__globals__, "run_case", fake_run_case)
    output = tmp_path / provider
    assert (
        pilot.main(
            ["--live", "--provider", provider, "--case", "02_related", "--output", str(output)]
        )
        == 0
    )
    assert received["api_key"] == "pilot-test-secret"
    assert received["base_url"] == expected_url
    manifest_text = (output / "manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["model"] == expected_model
    assert manifest["api_key_env"] == key_env
    assert manifest["provider"] == provider
    assert "pilot-test-secret" not in manifest_text


@pytest.mark.parametrize(
    "provider,key_env", [("openai", "OPENAI_API_KEY"), ("openrouter", "OPENROUTER_API_KEY")]
)
def test_missing_credentials_have_actionable_error(
    pilot, tmp_path, monkeypatch, capsys, provider, key_env
):
    monkeypatch.delenv(key_env, raising=False)
    output = tmp_path / provider
    assert (
        pilot.main(
            ["--live", "--provider", provider, "--case", "02_related", "--output", str(output)]
        )
        == 1
    )
    error = json.loads((output / "error.json").read_text(encoding="utf-8"))
    assert error["type"] == "ValueError"
    assert f"Missing {key_env}" in error["message"]
    assert "--env-file .env" in capsys.readouterr().out
