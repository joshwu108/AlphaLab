"""End-to-end CPU runs of the experiment harness on tiny synthetic data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from alphalab.errors import CheckpointError, ConfigError
from alphalab.experiments import config_from_dict, run_experiment
from alphalab.reporting import (
    comparability_issues,
    load_run,
    render_comparison,
    render_run_report,
    run_summary,
)


def _config(method: str, **overrides: Any) -> Any:
    raw: dict[str, Any] = {
        "schema_version": 1,
        "name": f"tiny-{method}",
        "method": method,
        "seed": 1,
        "data": {"synthetic": {"n_periods": 220, "n_instruments": 20, "signal_strength": 0.5}},
        "split": {"horizon": 1, "embargo": 1, "min_rows": 20},
        "grammar": {"max_depth": 3, "max_size": 7, "max_lookback": 20},
        "budget": {"max_evaluations": 40},
        "selection": {"max_factors": 3, "candidates": 10},
        "gp": {"population_size": 12},
        "rl": {"max_tokens": 15, "episodes_per_update": 8, "hidden_dim": 16, "embed_dim": 8},
    }
    raw.update(overrides)
    return config_from_dict(raw)


def _read(run_dir: Path, name: str) -> Any:
    return json.loads((run_dir / name).read_text())


@pytest.mark.parametrize("method", ["random", "gp"])
def test_completed_run_writes_audited_outputs(method: str, tmp_path: Path) -> None:
    outcome = run_experiment(_config(method), tmp_path / method)
    manifest = _read(outcome.run_dir, "manifest.json")
    assert outcome.status == manifest["status"] == "completed"
    assert manifest["budget"]["used"] == 40
    assert manifest["partition_access"]["scores_by_partition"]["test"] == 0
    assert manifest["data"]["fingerprint"] and manifest["environment"]["python"]
    assert {"config.json", "history.csv", "result.json"} <= set(manifest["outputs"])
    result = _read(outcome.run_dir, "result.json")
    assert 1 <= len(result["selected"]) <= 3 and result["test"] is None
    assert len(result["anytime_best"]) == 40
    assert result["combination"]["weights"]


def test_rl_run_completes_on_cpu(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    outcome = run_experiment(_config("rl"), tmp_path / "rl")
    assert outcome.status == "completed"
    assert (outcome.run_dir / "checkpoints" / "rl_checkpoint.pt").is_file()
    assert (outcome.run_dir / "rl_metrics.jsonl").read_text().strip()


def test_test_partition_is_read_only_when_configured(tmp_path: Path) -> None:
    outcome = run_experiment(
        _config("random", report={"evaluate_test": True}), tmp_path / "with-test"
    )
    manifest = _read(outcome.run_dir, "manifest.json")
    assert manifest["partition_access"]["test_unlocked"] is True
    reads = manifest["partition_access"]["scores_by_partition"]["test"]
    assert reads == len(outcome.result["selected"]) == len(outcome.result["test"])


@pytest.mark.parametrize("method", ["random", "gp", "rl"])
def test_dry_run_validates_without_searching(method: str, tmp_path: Path) -> None:
    if method == "rl":
        pytest.importorskip("torch")
    outcome = run_experiment(_config(method), tmp_path / "dry", dry_run=True)
    manifest = _read(outcome.run_dir, "manifest.json")
    # Three sampled expressions, plus one policy rollout for RL.
    assert manifest["status"] == "dry_run" and manifest["budget"]["used"] <= 4
    assert len(outcome.result["dry_run"]["sampled_scores"]) == 3


def test_failed_run_records_the_traceback(tmp_path: Path) -> None:
    config = _config("random", data={"source": "file", "path": str(tmp_path / "missing.csv")})
    with pytest.raises(FileNotFoundError):
        run_experiment(config, tmp_path / "failed")
    manifest = _read(tmp_path / "failed", "manifest.json")
    assert manifest["status"] == "failed"
    assert manifest["failure"]["type"] == "FileNotFoundError"
    assert "Traceback" in manifest["failure"]["traceback"]


def test_existing_runs_are_protected_and_resume_is_validated(tmp_path: Path) -> None:
    config = _config("gp")
    run_dir = tmp_path / "gp"
    first = run_experiment(config, run_dir)
    with pytest.raises(ConfigError, match="already contains"):
        run_experiment(config, run_dir)
    resumed = run_experiment(config, run_dir, resume=True)
    assert resumed.result["search"]["budget"] == first.result["search"]["budget"]
    with pytest.raises(CheckpointError, match="differs"):
        run_experiment(_config("gp", seed=2), run_dir, resume=True)
    with pytest.raises(CheckpointError, match="no manifest"):
        run_experiment(config, tmp_path / "never", resume=True)


def test_reports_and_comparisons(tmp_path: Path) -> None:
    runs = [run_experiment(_config(m), tmp_path / m).run_dir for m in ("random", "gp")]
    records = [load_run(r) for r in runs]
    assert comparability_issues(records) == []
    summary = run_summary(records[0])
    assert summary["budget_used"] == 40 and summary["test_reads"] == 0
    text = render_run_report(records[1])
    assert "not a measure of realizable trading profit" in text
    assert "selection-biased" in text
    assert "RST" not in render_run_report(records[1], rst=True)
    assert "Comparability: OK" in render_comparison(records)
    assert render_comparison(records, fmt="csv").startswith("run,")
    assert json.loads(render_comparison(records, fmt="json"))["issues"] == []
    other = run_experiment(_config("random", budget={"max_evaluations": 30}), tmp_path / "small")
    issues = comparability_issues([*records, load_run(other.run_dir)])
    assert issues == ["runs differ in config.budget.max_evaluations"]


def test_incomplete_runs_cannot_be_reported(tmp_path: Path) -> None:
    from alphalab.errors import AlphaLabError

    run_experiment(_config("random"), tmp_path / "dry", dry_run=True)
    with pytest.raises(AlphaLabError, match="not a completed run"):
        load_run(tmp_path / "dry")
    with pytest.raises(AlphaLabError, match="no manifest"):
        load_run(tmp_path / "nothing")
