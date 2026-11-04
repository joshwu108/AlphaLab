"""Tests for factor-quality metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphalab.evaluation import metrics

RNG = np.random.default_rng(0)
A = RNG.normal(size=(40, 12))
B = 0.5 * A + RNG.normal(size=(40, 12))


def test_rowwise_correlations_match_pandas() -> None:
    fa, fb = pd.DataFrame(A), pd.DataFrame(B)
    pearson = fa.T.corrwith(fb.T).to_numpy()
    spearman = fa.rank(axis=1).T.corrwith(fb.rank(axis=1).T).to_numpy()
    np.testing.assert_allclose(metrics.information_coefficient(A, B), pearson)
    np.testing.assert_allclose(metrics.rank_information_coefficient(A, B), spearman)


def test_correlation_ignores_nan_pairs_and_needs_min_obs() -> None:
    a = A.copy()
    a[0, :8] = np.nan
    a[1, 0] = np.nan
    out = metrics.information_coefficient(a, B, min_obs=5)
    assert np.isnan(out[0])
    expected = np.corrcoef(a[1, 1:], B[1, 1:])[0, 1]
    assert out[1] == pytest.approx(expected)


def test_zero_variance_rows_are_nan() -> None:
    constant = np.ones_like(A)
    assert np.isnan(metrics.information_coefficient(constant, B)).all()


def test_summary_statistics() -> None:
    summary = metrics.summarize_series(np.array([0.1, 0.3, np.nan, -0.1]))
    assert summary.n == 3
    assert summary.mean == pytest.approx(0.1)
    assert summary.std == pytest.approx(0.2)
    assert summary.ir == pytest.approx(0.5)
    assert summary.t_stat == pytest.approx(0.5 * np.sqrt(3))
    assert summary.hit_rate == pytest.approx(2 / 3)
    assert np.isnan(metrics.summarize_series(np.array([np.nan])).mean)


def test_turnover_extremes() -> None:
    stable = np.tile(np.arange(10.0), (20, 1))
    assert metrics.turnover(stable) == pytest.approx(0.0)
    flipping = stable * np.where(np.arange(20) % 2 == 0, 1.0, -1.0)[:, None]
    assert metrics.turnover(flipping) == pytest.approx(1.0)
    assert 0 < metrics.turnover(A) < 1


def test_rank_autocorrelation_and_sign_consistency() -> None:
    stable = np.tile(np.arange(10.0), (20, 1))
    assert metrics.rank_autocorrelation(stable) == pytest.approx(1.0)
    assert metrics.block_sign_consistency(np.array([0.1, 0.2, 0.1, 0.3])) == 1.0
    assert metrics.block_sign_consistency(np.array([0.5, 0.4, -0.1, -0.1]), 2) == 0.5


def test_correlation_matrix_and_redundancy() -> None:
    matrix = metrics.correlation_matrix([A, 2 * A, B])
    assert matrix[0, 1] == pytest.approx(1.0)
    assert np.allclose(matrix, matrix.T)
    assert metrics.redundancy([A]) == 0.0
    assert metrics.redundancy([A, -A]) == pytest.approx(1.0)


def test_per_instrument_correlation_shape() -> None:
    out = metrics.per_instrument_correlation(A, B, min_obs=10)
    assert out.shape == (12,) and np.isfinite(out).all()


def test_coverage() -> None:
    a = A.copy()
    a[:10] = np.nan
    assert metrics.coverage(a) == pytest.approx(0.75)


def test_constant_rows_with_rounding_noise_have_undefined_correlation() -> None:
    constant = np.full_like(A, 123456.789)
    assert np.isnan(metrics.information_coefficient(constant, B)).all()
    assert np.isnan(metrics.rank_information_coefficient(constant, B)).all()
