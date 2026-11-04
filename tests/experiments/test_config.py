"""Tests for experiment configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from alphalab.errors import ConfigError
from alphalab.experiments import config_from_dict, derive_seed, load_config

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def _base(**overrides: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {"schema_version": 1, "name": "t", "method": "random", "seed": 3}
    raw.update(overrides)
    return raw


@pytest.mark.parametrize("name", sorted(p.name for p in CONFIGS.glob("*.toml")))
def test_shipped_configs_are_valid(name: str) -> None:
    config = load_config(CONFIGS / name)
    assert len(config.config_hash()) == 64


def test_synthetic_configs_are_mutually_comparable() -> None:
    configs = [load_config(CONFIGS / f"synthetic_{m}.toml") for m in ("random", "gp", "rl")]
    shared = ("data", "split", "grammar", "evaluation", "budget", "selection", "report")
    for key in shared:
        assert len({repr(c.to_dict()[key]) for c in configs}) == 1, key


def test_defaults_and_seed_propagation() -> None:
    config = config_from_dict(_base(method="gp"))
    assert config.gp.seed == 3 and config.rl.seed == 3
    assert config.split.horizon == 1 and config.max_evaluations == 500


def test_hash_changes_with_any_setting() -> None:
    a = config_from_dict(_base())
    b = config_from_dict(_base(budget={"max_evaluations": 501}))
    assert a.config_hash() != b.config_hash()
    assert a.config_hash() == config_from_dict(_base()).config_hash()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"schema_version": 2}, "schema_version"),
        ({"method": "sac"}, "method"),
        ({"name": " "}, "name"),
        ({"seed": -1}, "seed"),
        ({"surprise": 1}, "unknown top-level"),
        ({"grammar": {"max_depht": 3}}, r"\[grammar\] unknown keys"),
        ({"grammar": {"operators": ["lead"]}}, "grammar"),
        ({"data": {"source": "file"}}, "path is required"),
        ({"data": {"source": "s3"}}, "source"),
        ({"split": {"valid_start": "2020-01-01"}}, "together"),
        ({"budget": {"max_evaluations": 0}}, "max_evaluations"),
        ({"budget": {"max": 5}}, r"\[budget\] unknown"),
        ({"selection": {"max_factors": 10, "candidates": 5}}, "max_factors"),
        ({"gp": {"population_size": 1}}, r"\[gp\] invalid"),
        ({"rl": {"learning_rat": 0.1}}, r"\[rl\] unknown"),
        ({"evaluation": {"metric": "sharpe"}}, r"\[evaluation\] invalid"),
        ({"rl": {"reward_mode": "sharpe"}}, r"\[rl\] invalid"),
        ({"method": "rl", "rl": {"max_tokens": 12}}, "cannot express"),
        ({"rl": {"learning_rate": -1.0}}, r"\[rl\] invalid"),
        ({"budget": {"max_evaluations": True}}, "max_evaluations"),
        ({"selection": {"max_abs_correlation": 0.0}}, "max_abs_correlation"),
        (
            {"split": {"valid_start": "2020", "test_start": "2021", "train_fraction": 0.5}},
            "not both",
        ),
    ],
)
def test_invalid_configs_are_rejected(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        config_from_dict(_base(**overrides))


def test_load_config_errors_and_relative_paths(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.toml")
    bad = tmp_path / "bad.toml"
    bad.write_text("name = [")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(bad)
    good = tmp_path / "sub" / "c.toml"
    good.parent.mkdir()
    good.write_text(
        'schema_version = 1\nname = "x"\nmethod = "random"\n[data]\nsource = "file"\n'
        'path = "../bars.csv"\n'
    )
    assert load_config(good).data.path == str((tmp_path / "bars.csv").resolve())


def test_rl_settings_are_validated_without_torch() -> None:
    config = config_from_dict(_base(method="rl", rl={"hidden_dim": 16}))
    assert config.rl_settings().hidden_dim == 16 and config.rl_settings().seed == 3


def test_derived_seeds_are_stable_and_distinct() -> None:
    assert derive_seed(0, "gp") == derive_seed(0, "gp")
    assert derive_seed(0, "gp") != derive_seed(0, "rl")
    assert 0 <= derive_seed(123, "x") < 2**63
