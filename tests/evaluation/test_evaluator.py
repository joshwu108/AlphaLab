"""Tests for partition-guarded scoring and budget accounting."""

from __future__ import annotations

import pytest

from alphalab.data import SyntheticSpec
from alphalab.errors import BudgetExhaustedError, ExpressionLimitError, PartitionAccessError
from alphalab.evaluation import INVALID_FITNESS, EvaluationBudget
from alphalab.expressions import Grammar
from tests.conftest import build_evaluator


def test_planted_signal_scores_positively_on_train_and_valid(strong_spec: SyntheticSpec) -> None:
    evaluator = build_evaluator(strong_spec)
    planted = strong_spec.planted_expression
    train = evaluator.score(planted, "train")
    valid = evaluator.score(planted, "valid")
    assert train.valid and train.direction == 1
    assert train.rank_ic_mean > 0.1 and valid.rank_ic_mean > 0.1
    assert valid.fitness == pytest.approx(valid.rank_ic_mean)


def test_negated_factor_gets_negative_direction(strong_spec: SyntheticSpec) -> None:
    evaluator = build_evaluator(strong_spec)
    negated = f"neg({strong_spec.planted_expression})"
    train = evaluator.score(negated)
    valid = evaluator.score(negated, "valid")
    assert train.direction == -1 and train.fitness > 0.1
    assert valid.fitness > 0.1 and valid.rank_ic_mean < 0


def test_test_partition_is_locked_until_explicitly_unlocked(strong_spec: SyntheticSpec) -> None:
    evaluator = build_evaluator(strong_spec)
    with pytest.raises(PartitionAccessError):
        evaluator.score("ts_mean($close, 5)", "test")
    with pytest.raises(PartitionAccessError):
        evaluator.factor_values("ts_mean($close, 5)", "test")
    with pytest.raises(ValueError):
        evaluator.unlock_test(" ")
    evaluator.unlock_test("final report")
    assert evaluator.score("ts_mean($close, 5)", "test").partition == "test"
    summary = evaluator.access_summary()
    assert summary["test_unlocked"] is True
    assert summary["scores_by_partition"]["test"] == 1  # type: ignore[index]


def test_budget_charges_unique_train_candidates_only(strong_spec: SyntheticSpec) -> None:
    evaluator = build_evaluator(strong_spec, budget=2)
    evaluator.score("ts_mean($close, 5)")
    evaluator.score("ts_mean($close, 5)")
    evaluator.score("add($open, $close)")
    evaluator.score("add($close, $open)")  # same canonical key
    evaluator.score("ts_mean($close, 5)", "valid")  # never charged
    assert evaluator.budget is not None
    assert evaluator.budget.used == 2 and evaluator.budget.duplicate_requests == 2
    with pytest.raises(BudgetExhaustedError):
        evaluator.score("ts_max($high, 10)")


def test_validation_score_computes_train_direction_first(strong_spec: SyntheticSpec) -> None:
    evaluator = build_evaluator(strong_spec, budget=5)
    evaluator.score("ts_std($close, 10)", "valid")
    assert evaluator.budget is not None and evaluator.budget.used == 1


def test_low_coverage_candidates_are_invalid() -> None:
    evaluator = build_evaluator(SyntheticSpec(n_periods=120, n_instruments=10, seed=1), budget=3)
    score = evaluator.score("ts_mean($close, 100)")
    assert not score.valid and score.fitness == INVALID_FITNESS
    assert "coverage" in (score.reason or "")
    assert evaluator.budget is not None and evaluator.budget.invalid_candidates == 1


def test_complexity_penalty_reduces_fitness(strong_spec: SyntheticSpec) -> None:
    plain = build_evaluator(strong_spec).score(strong_spec.planted_expression)
    penalized = build_evaluator(strong_spec, complexity_penalty=0.01).score(
        strong_spec.planted_expression
    )
    assert penalized.fitness == pytest.approx(plain.fitness - 0.01 * plain.size)


def test_grammar_is_enforced_before_charging(strong_spec: SyntheticSpec) -> None:
    evaluator = build_evaluator(strong_spec)
    evaluator.grammar = Grammar(max_depth=2)
    evaluator.budget = EvaluationBudget(3)
    with pytest.raises(ExpressionLimitError):
        evaluator.score("ts_mean(ts_mean($close, 5), 5)")
    assert evaluator.budget.used == 0


def test_budget_state_round_trip() -> None:
    budget = EvaluationBudget(5)
    budget.charge("a")
    budget.charge("b")
    budget.charge("a")
    restored = EvaluationBudget.from_state(budget.state())
    assert restored.to_dict() == budget.to_dict()
    assert restored.is_charged("a") and not restored.charge("b")
