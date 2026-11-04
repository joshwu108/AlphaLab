"""Versioned experiment configuration loaded from TOML.

A configuration fully determines a run: data source, partitions, grammar,
scoring, budget, method hyperparameters (including model and optimizer
settings for RL) and selection rules. Unknown keys are errors, so a typo can
never silently fall back to a default. The resolved configuration is hashed;
the hash identifies the run in manifests and guards checkpoint resumption.

Schema version 1 layout::

    schema_version = 1
    name = "synthetic-gp"
    method = "gp"            # "random" | "gp" | "rl"
    seed = 0

    [data]                   # source = "synthetic" | "file"
    [data.synthetic]         # SyntheticSpec fields
    [data.contract]          # DataContract fields
    [split]                  # horizon, embargo, fractions or dates
    [grammar]                # Grammar fields
    [evaluation]             # EvaluationSettings fields
    [budget]                 # max_evaluations
    [selection]              # max_factors, max_abs_correlation, candidates
    [gp] / [rl]              # GPSettings / RLSettings fields (seed defaults to top-level)
    [report]                 # evaluate_test
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from alphalab.data.contract import DataContract
from alphalab.data.synthetic import SyntheticSpec
from alphalab.errors import ConfigError
from alphalab.evaluation.evaluator import EvaluationSettings
from alphalab.expressions.grammar import Grammar
from alphalab.search.genetic import GPSettings

SCHEMA_VERSION = 1
METHODS = ("random", "gp", "rl")
Method = Literal["random", "gp", "rl"]


@dataclass(frozen=True)
class DataConfig:
    """Where the panel comes from."""

    source: Literal["synthetic", "file"] = "synthetic"
    path: str | None = None
    column_map: Mapping[str, str] = field(default_factory=dict)
    synthetic: SyntheticSpec = field(default_factory=SyntheticSpec)
    contract: DataContract = field(default_factory=DataContract)


@dataclass(frozen=True)
class SplitConfig:
    """Chronological partitioning; either fractions or dates must be given."""

    horizon: int = 1
    embargo: int = 0
    train_fraction: float | None = 0.6
    valid_fraction: float | None = 0.2
    valid_start: str | None = None
    test_start: str | None = None
    min_rows: int = 20


@dataclass(frozen=True)
class SelectionConfig:
    """Final validation-based selection, identical for every method."""

    max_factors: int = 10
    max_abs_correlation: float = 0.7
    candidates: int = 50


@dataclass(frozen=True)
class ReportConfig:
    """Post-selection reporting."""

    evaluate_test: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete, validated experiment definition."""

    name: str
    method: Method
    seed: int = 0
    schema_version: int = SCHEMA_VERSION
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    grammar: Grammar = field(default_factory=Grammar)
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)
    max_evaluations: int = 500
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    gp: GPSettings = field(default_factory=GPSettings)
    rl: Mapping[str, Any] = field(default_factory=dict)
    report: ReportConfig = field(default_factory=ReportConfig)

    def to_dict(self) -> dict[str, Any]:
        """Resolved configuration as plain data (the input to :meth:`config_hash`)."""
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "method": self.method,
            "seed": self.seed,
            "data": {
                "source": self.data.source,
                "path": self.data.path,
                "column_map": dict(self.data.column_map),
                "synthetic": dataclasses.asdict(self.data.synthetic),
                "contract": dataclasses.asdict(self.data.contract),
            },
            "split": dataclasses.asdict(self.split),
            "grammar": self.grammar.to_dict(),
            "evaluation": dataclasses.asdict(self.evaluation),
            "budget": {"max_evaluations": self.max_evaluations},
            "selection": dataclasses.asdict(self.selection),
            "gp": dataclasses.asdict(self.gp),
            "rl": dict(self.rl),
            "report": dataclasses.asdict(self.report),
        }

    def config_hash(self) -> str:
        """SHA-256 of the canonical JSON form of :meth:`to_dict`."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def rl_settings(self) -> Any:
        """``RLSettings`` built from the ``[rl]`` section (imports torch lazily)."""
        from alphalab.rl.trainer import RLSettings

        values = {"seed": self.seed, **dict(self.rl)}
        return _build(RLSettings, values, "rl")


def _build(cls: type, values: Mapping[str, Any], section: str) -> Any:
    """Instantiate dataclass ``cls`` from ``values`` with strict key checking."""
    names = {f.name for f in dataclasses.fields(cls) if f.init}
    unknown = sorted(set(values) - names)
    if unknown:
        raise ConfigError(f"[{section}] unknown keys {unknown}; allowed: {sorted(names)}")
    converted = {k: tuple(v) if isinstance(v, list) else v for k, v in values.items()}
    try:
        return cls(**converted)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"[{section}] invalid value: {exc}") from exc


def _section(raw: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, Mapping):
        raise ConfigError(f"[{key}] must be a table")
    return dict(value)


def _validate_rl_keys(rl: Mapping[str, Any]) -> None:
    # Validated without importing torch: the field names are mirrored here and
    # checked against RLSettings in tests.
    allowed = {
        "max_tokens", "embed_dim", "hidden_dim", "num_layers", "learning_rate",
        "episodes_per_update", "entropy_coef", "value_coef", "grad_clip", "reward_mode",
        "diversity_weight", "pool_capacity", "max_updates", "stall_updates",
        "checkpoint_every", "seed", "device", "top_k",
    }  # fmt: skip
    unknown = sorted(set(rl) - allowed)
    if unknown:
        raise ConfigError(f"[rl] unknown keys {unknown}; allowed: {sorted(allowed)}")


def config_from_dict(raw: Mapping[str, Any]) -> ExperimentConfig:
    """Validate and build an :class:`ExperimentConfig` from parsed TOML data.

    Raises:
        ConfigError: Unknown or missing keys, wrong types, or inconsistent values.
    """
    known = {
        "schema_version", "name", "method", "seed", "data", "split", "grammar",
        "evaluation", "budget", "selection", "gp", "rl", "report",
    }  # fmt: skip
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError(f"unknown top-level keys {unknown}")
    version = raw.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {SCHEMA_VERSION}, got {version!r}")
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ConfigError("name must be a non-empty string")
    method = raw.get("method")
    if method not in METHODS:
        raise ConfigError(f"method must be one of {METHODS}, got {method!r}")
    seed = raw.get("seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ConfigError("seed must be a non-negative integer")

    data_raw = _section(raw, "data")
    synthetic = _build(SyntheticSpec, _section(data_raw, "synthetic"), "data.synthetic")
    contract = _build(DataContract, _section(data_raw, "contract"), "data.contract")
    data_fields = {k: v for k, v in data_raw.items() if k not in ("synthetic", "contract")}
    data = _build(DataConfig, {**data_fields, "synthetic": synthetic, "contract": contract}, "data")
    if isinstance(data.column_map, tuple):
        raise ConfigError("[data] column_map must be a table")
    if data.source not in ("synthetic", "file"):
        raise ConfigError(f"[data] source must be 'synthetic' or 'file', got {data.source!r}")
    if data.source == "file" and not data.path:
        raise ConfigError("[data] path is required when source = 'file'")

    split = _build(SplitConfig, _section(raw, "split"), "split")
    by_dates = split.valid_start is not None or split.test_start is not None
    if by_dates and (split.valid_start is None or split.test_start is None):
        raise ConfigError("[split] valid_start and test_start must be given together")
    if not by_dates and (split.train_fraction is None or split.valid_fraction is None):
        raise ConfigError("[split] give train_fraction and valid_fraction, or both dates")

    grammar_raw = _section(raw, "grammar")
    try:
        grammar = _build(Grammar, grammar_raw, "grammar")
    except Exception as exc:
        if isinstance(exc, ConfigError):
            raise
        raise ConfigError(f"[grammar] {exc}") from exc
    evaluation = _build(EvaluationSettings, _section(raw, "evaluation"), "evaluation")
    budget = _section(raw, "budget")
    if set(budget) - {"max_evaluations"}:
        raise ConfigError(f"[budget] unknown keys {sorted(set(budget) - {'max_evaluations'})}")
    max_evaluations = budget.get("max_evaluations", 500)
    if not isinstance(max_evaluations, int) or max_evaluations < 1:
        raise ConfigError("[budget] max_evaluations must be a positive integer")
    selection = _build(SelectionConfig, _section(raw, "selection"), "selection")
    if selection.max_factors < 1 or selection.candidates < selection.max_factors:
        raise ConfigError("[selection] need 1 <= max_factors <= candidates")
    gp = _build(GPSettings, {"seed": seed, **_section(raw, "gp")}, "gp")
    rl = _section(raw, "rl")
    _validate_rl_keys(rl)
    report = _build(ReportConfig, _section(raw, "report"), "report")
    return ExperimentConfig(
        name=name.strip(),
        method=method,
        seed=seed,
        data=data,
        split=split,
        grammar=grammar,
        evaluation=evaluation,
        max_evaluations=max_evaluations,
        selection=selection,
        gp=gp,
        rl={"seed": seed, **rl},
        report=report,
    )


def load_config(path: str | Path) -> ExperimentConfig:
    """Read and validate a TOML configuration file.

    Relative ``data.path`` values are resolved against the configuration
    file's directory.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"configuration file not found: {path}")
    try:
        raw = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: invalid TOML: {exc}") from exc
    data = raw.get("data")
    if isinstance(data, dict) and isinstance(data.get("path"), str):
        candidate = Path(data["path"]).expanduser()
        if not candidate.is_absolute():
            data["path"] = str((path.parent / candidate).resolve())
    return config_from_dict(raw)
