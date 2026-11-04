"""Tests for random search, genetic programming and validation-based selection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from alphalab.data import SyntheticSpec
from alphalab.errors import CheckpointError
from alphalab.expressions import Call, Grammar, sample_expression
from alphalab.search import (
    GeneticProgramming,
    GPSettings,
    RandomSearch,
    Variation,
    select_on_validation,
)
from tests.conftest import build_evaluator

GRAMMAR = Grammar(max_depth=4, max_size=10, max_lookback=30)


def _signature(result: object) -> list[tuple[str, float]]:
    return [(r.expression, r.fitness) for r in result.history.records]  # type: ignore[attr-defined]


def test_random_search_is_deterministic_and_budget_exact(strong_spec: SyntheticSpec) -> None:
    first = RandomSearch(GRAMMAR, seed=4).run(build_evaluator(strong_spec, budget=60))
    second = RandomSearch(GRAMMAR, seed=4).run(build_evaluator(strong_spec, budget=60))
    assert _signature(first) == _signature(second)
    assert len(first.history.records) == first.budget["used"] == 60
    keys = [r.key for r in first.history.records]
    assert len(set(keys)) == len(keys)
    assert first.best[0].fitness == max(r.fitness for r in first.history.records if r.valid)


def test_random_search_requires_a_budget(strong_spec: SyntheticSpec) -> None:
    with pytest.raises(ValueError, match="budget"):
        RandomSearch(GRAMMAR).run(build_evaluator(strong_spec))


def test_random_search_stops_when_the_space_is_exhausted(strong_spec: SyntheticSpec) -> None:
    tiny = Grammar(features=("close",), operators=("neg",), windows=(5,), max_depth=2)
    result = RandomSearch(tiny, seed=0).run(build_evaluator(strong_spec, budget=10))
    assert result.stopped_reason.startswith("search space exhausted")
    assert len(result.history.records) == 1


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 2**32 - 1))
def test_variation_operators_preserve_grammar_validity(seed: int) -> None:
    rng = np.random.default_rng(seed)
    variation = Variation(GRAMMAR, rng)
    a, b = sample_expression(GRAMMAR, rng), sample_expression(GRAMMAR, rng)
    for child in (
        variation.crossover(a, b),
        variation.subtree_mutation(a),
        variation.point_mutation(a),
        variation.hoist_mutation(a),
        variation.mutate(b),
    ):
        assert isinstance(child, Call)
        GRAMMAR.check(child)


def test_gp_is_deterministic_and_respects_budget(strong_spec: SyntheticSpec) -> None:
    gp_settings = GPSettings(population_size=16, seed=2, diversity_weight=0.1)
    first = GeneticProgramming(GRAMMAR, gp_settings).run(build_evaluator(strong_spec, budget=80))
    second = GeneticProgramming(GRAMMAR, gp_settings).run(build_evaluator(strong_spec, budget=80))
    assert _signature(first) == _signature(second)
    assert first.budget["used"] == len(first.history.records) <= 80
    assert first.extra["generations"] >= 2
    assert {r.source for r in first.history.records} >= {"init", "offspring"}


def test_gp_finds_a_predictive_factor_on_planted_data(strong_spec: SyntheticSpec) -> None:
    result = GeneticProgramming(GRAMMAR, GPSettings(population_size=24, seed=0)).run(
        build_evaluator(strong_spec, budget=150)
    )
    assert result.best[0].fitness > 0.1


def test_gp_resume_reproduces_an_uninterrupted_run(
    strong_spec: SyntheticSpec, tmp_path: Path
) -> None:
    base = GPSettings(population_size=12, seed=5)
    full = GeneticProgramming(GRAMMAR, base).run(build_evaluator(strong_spec, budget=70))

    path = tmp_path / "gp.json"
    limited = GPSettings(population_size=12, seed=5, max_generations=2)
    partial = GeneticProgramming(GRAMMAR, limited, checkpoint_path=path).run(
        build_evaluator(strong_spec, budget=70)
    )
    assert partial.stopped_reason == "max_generations reached"
    resumed = GeneticProgramming(GRAMMAR, base, checkpoint_path=path).run(
        build_evaluator(strong_spec, budget=70), resume=True
    )
    assert _signature(resumed) == _signature(full)
    assert resumed.budget == full.budget


def test_gp_checkpoint_mismatch_and_corruption_are_rejected(
    strong_spec: SyntheticSpec, tmp_path: Path
) -> None:
    path = tmp_path / "gp.json"
    gp = GeneticProgramming(
        GRAMMAR, GPSettings(population_size=8, max_generations=1), checkpoint_path=path
    )
    gp.run(build_evaluator(strong_spec, budget=40))
    other = GeneticProgramming(GRAMMAR, GPSettings(population_size=8, seed=9), checkpoint_path=path)
    with pytest.raises(CheckpointError, match="different"):
        other.run(build_evaluator(strong_spec, budget=40), resume=True)
    data = json.loads(path.read_text())
    data["method"] = "random"
    path.write_text(json.dumps(data))
    with pytest.raises(CheckpointError, match="not 'gp'"):
        gp.run(build_evaluator(strong_spec, budget=40), resume=True)
    path.write_text("{broken")
    with pytest.raises(CheckpointError, match="not valid JSON"):
        gp.run(build_evaluator(strong_spec, budget=40), resume=True)
    with pytest.raises(CheckpointError, match="no checkpoint_path"):
        GeneticProgramming(GRAMMAR).run(build_evaluator(strong_spec, budget=5), resume=True)


def test_gp_settings_validation() -> None:
    with pytest.raises(ValueError):
        GPSettings(population_size=1)
    with pytest.raises(ValueError):
        GPSettings(population_size=4, tournament_size=5)
    with pytest.raises(ValueError):
        GPSettings(elitism=64)


def test_validation_selection_is_diverse_and_never_reads_test(strong_spec: SyntheticSpec) -> None:
    evaluator = build_evaluator(strong_spec, budget=80)
    result = RandomSearch(GRAMMAR, seed=1).run(evaluator)
    used = evaluator.budget.used if evaluator.budget else 0
    chosen = select_on_validation(evaluator, result.best, max_factors=5, max_abs_correlation=0.6)
    assert 1 <= len(chosen) <= 5
    assert all(valid.partition == "valid" for _, valid in chosen)
    fitness = [valid.fitness for _, valid in chosen]
    assert fitness == sorted(fitness, reverse=True)
    assert evaluator.access_summary()["scores_by_partition"]["test"] == 0  # type: ignore[index]
    assert evaluator.budget is not None and evaluator.budget.used == used
