"""Tests for the command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alphalab.cli import main

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _write_config(tmp_path: Path, data_path: Path | None = None) -> Path:
    data = (
        f'[data]\nsource = "file"\npath = "{data_path}"\n'
        if data_path
        else "[data.synthetic]\nn_periods = 200\nn_instruments = 15\nsignal_strength = 0.5\n"
    )
    path = tmp_path / "c.toml"
    path.write_text(
        'schema_version = 1\nname = "cli"\nmethod = "random"\n'
        + data
        + "[grammar]\nmax_depth = 3\nmax_size = 7\nmax_lookback = 20\n"
        + "[budget]\nmax_evaluations = 15\n[selection]\nmax_factors = 2\ncandidates = 5\n"
    )
    return path


def test_operators_lists_every_operator(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["operators"]) == 0
    assert "ts_corr" in capsys.readouterr().out


def test_synth_validate_and_eval(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = tmp_path / "bars.csv"
    assert main(["synth", "--out", str(data), "--periods", "200", "--instruments", "15"]) == 0
    assert "planted factor" in capsys.readouterr().out
    assert main(["validate-data", str(data)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["report"]["n_instruments"] == 15 and len(report["fingerprint"]) == 64
    config = _write_config(tmp_path, data)
    assert main(["eval-expr", "div(delay($close, 5), $close)", "--config", str(config)]) == 0
    scores = json.loads(capsys.readouterr().out)
    assert set(scores) == {"train", "valid"}


def test_run_report_and_compare(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = _write_config(tmp_path)
    run_dir = tmp_path / "run"
    assert main(["check-config", str(config)]) == 0
    assert "config_hash" in capsys.readouterr().out
    assert main(["run", str(config), "--output", str(run_dir)]) == 0
    assert "completed" in capsys.readouterr().out
    assert main(["report", str(run_dir), "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["method"] == "random"
    assert main(["compare", str(run_dir), str(run_dir), "--format", "rst"]) == 0
    assert "Comparability: OK" in capsys.readouterr().out


def test_invalid_input_exits_with_usage_status(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["check-config", str(tmp_path / "missing.toml")]) == 2
    assert "error:" in capsys.readouterr().err
    assert main(["eval-expr", "lead($close, 1)", "--config", str(_write_config(tmp_path))]) == 2
    assert main(["validate-data", str(tmp_path / "x.csv"), "--column-map", "bad"]) == 2
    with pytest.raises(SystemExit):
        main(["frobnicate"])
