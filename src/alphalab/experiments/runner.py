"""End-to-end experiment execution.

``run_experiment`` performs, in order:

1. Load and validate data, build the panel, labels and purged split.
2. Build a :class:`~alphalab.evaluation.FactorEvaluator` with the configured
   budget and grammar (test partition locked).
3. Run the configured search method until its budget is spent.
4. Select a diverse factor set on the **validation** partition, identically for
   every method, and fit a ridge combination of the selected factors on train.
5. Only if ``report.evaluate_test`` is set: unlock the test partition and score
   the already-selected factors once.
6. Write outputs and a manifest; on failure, write a ``failed`` manifest with
   the traceback and re-raise.

A **dry run** executes steps 1-2, scores a handful of sampled expressions,
constructs the search method (and, for RL, one policy rollout without a
gradient step), and writes a ``dry_run`` manifest. It validates a
configuration end-to-end at negligible cost.

Run directory layout::

    <run_dir>/
        manifest.json       provenance, status, budget, partition access
        config.json         resolved configuration
        history.csv         every charged evaluation, in order
        result.json         search summary, selection, combination, test scores
        checkpoints/        method checkpoints (GP JSON, RL torch)
        rl_metrics.jsonl    per-update RL diagnostics (RL only)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alphalab.data.io import load_market_data
from alphalab.data.labels import panel_forward_returns
from alphalab.data.panel import MarketPanel, panel_from_frame
from alphalab.data.splits import ChronologicalSplit, split_by_dates, split_by_fractions
from alphalab.data.synthetic import make_synthetic_panel
from alphalab.errors import CheckpointError, ConfigError
from alphalab.evaluation import metrics
from alphalab.evaluation.budget import EvaluationBudget
from alphalab.evaluation.evaluator import FactorEvaluator
from alphalab.evaluation.export import write_json
from alphalab.evaluation.selection import combine, fit_linear_combination
from alphalab.experiments.config import ExperimentConfig
from alphalab.experiments.provenance import MANIFEST_NAME, RunManifest, seed_everything
from alphalab.expressions.grammar import sample_expression
from alphalab.search.base import SearchResult, select_on_validation
from alphalab.search.genetic import GeneticProgramming
from alphalab.search.random_search import RandomSearch

DRY_RUN_SAMPLES = 3


@dataclass(frozen=True)
class RunOutcome:
    """Paths and status of a finished run."""

    run_dir: Path
    status: str
    result: dict[str, Any]


def load_panel(config: ExperimentConfig) -> tuple[MarketPanel, dict[str, Any]]:
    """Build the panel described by ``config.data`` and a provenance record."""
    data = config.data
    if data.source == "synthetic":
        panel = make_synthetic_panel(data.synthetic)
        return panel, {"source": "synthetic", "planted": data.synthetic.planted_expression}
    assert data.path is not None
    frame, report = load_market_data(data.path, data.contract, data.column_map)
    panel = panel_from_frame(frame, metadata={"source": "file", "path": str(data.path)})
    return panel, {"source": "file", "path": str(data.path), "validation": report.__dict__}


def build_split(config: ExperimentConfig, panel: MarketPanel) -> ChronologicalSplit:
    """Purged chronological split; the first training row skips the grammar's lookback."""
    s = config.split
    first_row = min(config.grammar.max_lookback, panel.n_periods // 4)
    if s.valid_start is not None and s.test_start is not None:
        return split_by_dates(
            panel.timestamps,
            s.valid_start,
            s.test_start,
            horizon=s.horizon,
            embargo=s.embargo,
            first_row=first_row,
            min_rows=s.min_rows,
        )
    assert s.train_fraction is not None and s.valid_fraction is not None
    return split_by_fractions(
        panel.n_periods,
        s.train_fraction,
        s.valid_fraction,
        horizon=s.horizon,
        embargo=s.embargo,
        first_row=first_row,
        min_rows=s.min_rows,
    )


def build_evaluator(config: ExperimentConfig) -> tuple[FactorEvaluator, dict[str, Any]]:
    """Panel, target, split and evaluator for ``config``; returns data provenance too."""
    panel, provenance = load_panel(config)
    split = build_split(config, panel)
    target = panel_forward_returns(panel, config.split.horizon)
    evaluator = FactorEvaluator(
        panel,
        target,
        split,
        settings=config.evaluation,
        budget=EvaluationBudget(config.max_evaluations),
        grammar=config.grammar,
    )
    provenance.update(
        fingerprint=panel.fingerprint,
        shape=list(panel.shape),
        start=str(pd.Timestamp(panel.timestamps[0])),
        end=str(pd.Timestamp(panel.timestamps[-1])),
        fields=sorted(panel.fields),
    )
    return evaluator, provenance


def _search(
    config: ExperimentConfig, evaluator: FactorEvaluator, run_dir: Path, *, resume: bool
) -> SearchResult:
    checkpoints = run_dir / "checkpoints"
    if config.method == "random":
        # Random search is cheap and deterministic: a resumed run simply replays it.
        return RandomSearch(config.grammar, seed=config.seed).run(evaluator)
    if config.method == "gp":
        gp = GeneticProgramming(
            config.grammar, config.gp, checkpoint_path=checkpoints / "gp_checkpoint.json"
        )
        return gp.run(evaluator, resume=resume)
    from alphalab.rl.trainer import PolicyGradientTrainer

    trainer = PolicyGradientTrainer(
        config.grammar,
        config.rl_settings(),
        checkpoint_dir=checkpoints,
        metrics_path=run_dir / "rl_metrics.jsonl",
    )
    return trainer.run(evaluator, resume=resume)


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _selection(
    config: ExperimentConfig, evaluator: FactorEvaluator, result: SearchResult
) -> dict[str, Any]:
    candidates = result.history.top(config.selection.candidates)
    chosen = select_on_validation(
        evaluator,
        candidates,
        max_factors=config.selection.max_factors,
        max_abs_correlation=config.selection.max_abs_correlation,
    )
    selected = [{"train": t.to_dict(), "valid": v.to_dict()} for t, v in chosen]
    combination: dict[str, Any] = {"weights": [], "train_rank_ic": None, "valid_rank_ic": None}
    if chosen:
        expressions = [t.expression for t, _ in chosen]
        signed = [
            [
                t.direction * evaluator.factor_values(e, p)
                for e, (t, _) in zip(expressions, chosen, strict=True)
            ]
            for p in ("train", "valid")
        ]
        weights = fit_linear_combination(signed[0], evaluator.target_values("train"))
        for name, values in zip(("train", "valid"), signed, strict=True):
            ric = metrics.rank_information_coefficient(
                combine(values, weights), evaluator.target_values(name)
            )
            combination[f"{name}_rank_ic"] = _finite(metrics.summarize_series(ric).mean)
        combination["weights"] = [
            {"expression": e, "weight": float(w)} for e, w in zip(expressions, weights, strict=True)
        ]
        train_values = [evaluator.factor_values(e, "train") for e in expressions]
        combination["train_redundancy"] = _finite(metrics.redundancy(train_values))
    return {"selected": selected, "combination": combination}


def _test_scores(evaluator: FactorEvaluator, selection: dict[str, Any]) -> list[dict[str, Any]]:
    evaluator.unlock_test("final evaluation of factors already selected on validation")
    return [
        evaluator.score(item["train"]["expression"], "test").to_dict()
        for item in selection["selected"]
    ]


def _dry_run(config: ExperimentConfig, evaluator: FactorEvaluator) -> dict[str, Any]:
    rng = np.random.default_rng(config.seed)
    samples = [sample_expression(config.grammar, rng) for _ in range(DRY_RUN_SAMPLES)]
    scored = [evaluator.score(node).to_dict() for node in samples]
    info: dict[str, Any] = {"sampled_scores": scored}
    if config.method == "gp":
        GeneticProgramming(config.grammar, config.gp)
    elif config.method == "rl":
        from alphalab.rl.trainer import PolicyGradientTrainer

        trainer = PolicyGradientTrainer(config.grammar, config.rl_settings())
        env, policy, _, generator = trainer.build(evaluator)
        episode = trainer.rollout(env, policy, generator)
        info["rl_rollout"] = {"tokens": len(episode.actions), "expression": episode.expression}
        info["rl_parameters"] = int(sum(p.numel() for p in policy.parameters()))
    return info


def _prepare_run_dir(config: ExperimentConfig, run_dir: Path, resume: bool) -> None:
    manifest_path = run_dir / MANIFEST_NAME
    if resume:
        if not manifest_path.is_file():
            raise CheckpointError(f"cannot resume: no manifest in {run_dir}")
        previous = json.loads(manifest_path.read_text())
        if previous.get("config_hash") != config.config_hash():
            raise CheckpointError("cannot resume: configuration differs from the original run")
        return
    if manifest_path.exists():
        raise ConfigError(f"{run_dir} already contains a run; choose a new directory or resume")
    run_dir.mkdir(parents=True, exist_ok=True)


def run_experiment(
    config: ExperimentConfig, run_dir: str | Path, *, dry_run: bool = False, resume: bool = False
) -> RunOutcome:
    """Execute ``config`` and write all outputs to ``run_dir``.

    Raises:
        ConfigError: ``run_dir`` already holds a run and ``resume`` is false.
        CheckpointError: ``resume`` is requested for a missing or different run.
        Exception: Any failure inside the run is recorded in the manifest and re-raised.
    """
    run_dir = Path(run_dir)
    _prepare_run_dir(config, run_dir, resume)
    seed_everything(config.seed)
    write_json(config.to_dict(), run_dir / "config.json")
    manifest = RunManifest.start(run_dir, config.to_dict(), config.config_hash())
    manifest.update(dry_run=dry_run, resumed=resume)
    try:
        evaluator, provenance = build_evaluator(config)
        manifest.update(data=provenance, split=evaluator.split.to_dict())
        manifest.write()
        if dry_run:
            result: dict[str, Any] = {"dry_run": _dry_run(config, evaluator)}
            status = "dry_run"
        else:
            search = _search(config, evaluator, run_dir, resume=resume)
            pd.DataFrame(search.history.to_list()).to_csv(run_dir / "history.csv", index=False)
            result = {
                "search": search.to_dict(),
                "anytime_best": [float(x) for x in search.history.best_so_far()],
                **_selection(config, evaluator, search),
            }
            result["test"] = (
                _test_scores(evaluator, result) if config.report.evaluate_test else None
            )
            status = "completed"
        assert evaluator.budget is not None
        manifest.update(
            budget=evaluator.budget.to_dict(), partition_access=evaluator.access_summary()
        )
        write_json(result, run_dir / "result.json")
        manifest.update(
            outputs=sorted(p.name for p in run_dir.iterdir() if p.name != MANIFEST_NAME)
        )
        manifest.finish(status)
        return RunOutcome(run_dir, status, result)
    except BaseException as exc:
        manifest.fail(exc)
        raise
