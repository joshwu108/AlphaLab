"""Tests for diversity-aware selection, factor pools and exports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphalab.data import SyntheticSpec
from alphalab.evaluation import (
    Candidate,
    DiversityPenalty,
    FactorPool,
    combine,
    export_scores,
    fit_linear_combination,
    metrics,
    select_diverse,
)
from tests.conftest import build_evaluator

RNG = np.random.default_rng(4)
Z1, Z2, Z3 = (RNG.normal(size=(60, 20)) for _ in range(3))


def test_select_diverse_skips_redundant_candidates() -> None:
    candidates = [
        Candidate("a", 0.30, Z1),
        Candidate("a_scaled", 0.29, 3.0 * Z1),
        Candidate("b", 0.20, Z2),
        Candidate("bad", float("nan"), Z3),
    ]
    kept = select_diverse(candidates, max_factors=3, max_abs_correlation=0.7)
    assert [c.expression for c in kept] == ["a", "b"]
    assert len(select_diverse(candidates, max_factors=1)) == 1
    with pytest.raises(ValueError):
        select_diverse(candidates, max_factors=0)


def test_linear_combination_recovers_known_weights() -> None:
    target = 2.0 * Z1 - 1.0 * Z2 + 0.01 * Z3
    weights = fit_linear_combination([Z1, Z2], target, ridge=1e-8)
    # Per-session z-scoring rescales each factor by its own sample dispersion,
    # so the recovered ratio is close to, not exactly, the generating ratio.
    assert weights[0] / -weights[1] == pytest.approx(2.0, rel=0.15)
    combined = combine([Z1, Z2], weights)
    assert combined.shape == Z1.shape
    ic = metrics.information_coefficient(combined, target)
    assert np.nanmean(ic) > 0.95


def test_factor_pool_keeps_only_improving_members() -> None:
    target = Z1 + 0.5 * Z2 + RNG.normal(size=Z1.shape)
    pool = FactorPool(capacity=2)
    assert pool.try_add("z1", Z1, target) > 0
    assert pool.try_add("z1", Z1, target) == 0.0
    assert pool.try_add("z2", Z2, target) > 0
    before = pool.score
    pool.try_add("z3", Z3, target)
    assert len(pool.expressions) <= 2 and pool.score >= before
    exported = pool.to_dict()
    assert len(exported["members"]) == len(pool.expressions)  # type: ignore[arg-type]


def test_diversity_penalty_uses_reference_set() -> None:
    penalty = DiversityPenalty(0.5, lambda: [Z1])
    assert penalty(Z1) == pytest.approx(0.5)
    assert penalty(Z2) < 0.1
    assert DiversityPenalty(0.0)(Z1) == 0.0


@pytest.mark.parametrize("suffix", [".json", ".csv", ".parquet"])
def test_scores_export(tmp_path: Path, suffix: str) -> None:
    evaluator = build_evaluator(SyntheticSpec(n_periods=120, n_instruments=10, seed=2))
    scores = [evaluator.score("ts_mean($close, 100)"), evaluator.score("delta($close, 1)")]
    path = export_scores(scores, tmp_path / f"scores{suffix}")
    if suffix == ".json":
        data = json.loads(path.read_text())
        assert data[0]["rank_ic_mean"] is None  # NaN written as null
        assert data[1]["expression"] == "delta($close, 1)"
    else:
        frame = pd.read_csv(path) if suffix == ".csv" else pd.read_parquet(path)
        assert len(frame) == 2
    with pytest.raises(ValueError):
        export_scores(scores, tmp_path / "scores.xlsx")
