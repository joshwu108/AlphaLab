"""Numerical and causal correctness of the CPU evaluator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from alphalab.data import MarketPanel, SyntheticSpec, make_synthetic_panel
from alphalab.errors import ExpressionError, ExpressionTypeError
from alphalab.expressions import (
    Constant,
    ExpressionCache,
    Grammar,
    evaluate,
    parse,
    sample_expression,
)

PANEL = make_synthetic_panel(SyntheticSpec(n_periods=90, n_instruments=8, seed=5))
CLOSE = pd.DataFrame(PANEL.field("close"))
VOLUME = pd.DataFrame(PANEL.field("volume"))


@pytest.mark.parametrize(
    ("text", "reference"),
    [
        ("ts_mean($close, 7)", CLOSE.rolling(7).mean()),
        ("ts_sum($close, 7)", CLOSE.rolling(7).sum()),
        ("ts_std($close, 7)", CLOSE.rolling(7).std()),
        ("ts_min($close, 7)", CLOSE.rolling(7).min()),
        ("ts_max($close, 7)", CLOSE.rolling(7).max()),
        ("delay($close, 3)", CLOSE.shift(3)),
        ("delta($close, 3)", CLOSE - CLOSE.shift(3)),
        ("ts_corr($close, $volume, 10)", CLOSE.rolling(10).corr(VOLUME)),
        ("cs_rank($close)", (CLOSE.rank(axis=1) - 1).div(CLOSE.count(axis=1) - 1, axis=0)),
        (
            "cs_zscore($close)",
            CLOSE.sub(CLOSE.mean(axis=1), axis=0).div(CLOSE.std(axis=1, ddof=0), axis=0),
        ),
        ("div($close, $volume)", CLOSE / VOLUME),
        ("neg(abs($close))", -CLOSE.abs()),
    ],
)
def test_kernels_match_pandas_reference(text: str, reference: pd.DataFrame) -> None:
    np.testing.assert_allclose(evaluate(text, PANEL), reference.to_numpy(), rtol=1e-9, atol=1e-12)


def test_truncated_ema_matches_explicit_weights() -> None:
    w = 6
    alpha = 2 / (w + 1)
    weights = (1 - alpha) ** np.arange(w)[::-1]
    expected = CLOSE.rolling(w).apply(lambda v: float(v @ weights / weights.sum()), raw=True)
    np.testing.assert_allclose(evaluate(f"ts_ema($close, {w})", PANEL), expected.to_numpy())


def test_protected_division_and_non_finite_values_become_nan() -> None:
    out = evaluate("div($close, sub($close, $close))", PANEL)
    assert np.isnan(out).all()
    assert np.isfinite(
        evaluate("mul($volume, $volume)", PANEL)[np.isfinite(PANEL.field("volume"))]
    ).all()


def test_rolling_windows_are_strict_about_missing_values() -> None:
    panel = make_synthetic_panel(SyntheticSpec(n_periods=40, n_instruments=3, missing_rate=0.0))
    close = np.array(panel.field("close"))
    close[20, 0] = np.nan
    gappy = MarketPanel(panel.timestamps, panel.instruments, {"close": close})
    out = evaluate("ts_mean($close, 5)", gappy)
    assert np.isnan(out[20:25, 0]).all() and np.isfinite(out[25, 0])


def test_cross_sectional_rows_with_one_value_are_nan() -> None:
    close = np.full((3, 4), np.nan)
    close[0] = [1.0, 2.0, 2.0, 4.0]
    close[1, 0] = 5.0
    panel = MarketPanel(PANEL.timestamps[:3], ("A", "B", "C", "D"), {"close": close})
    ranks = evaluate("cs_rank($close)", panel)
    np.testing.assert_allclose(ranks[0], [0.0, 0.5, 0.5, 1.0])
    assert np.isnan(ranks[1:]).all()
    assert np.isnan(evaluate("cs_zscore($close)", panel)[1:]).all()


def test_bare_constants_and_missing_features_are_rejected() -> None:
    with pytest.raises(ExpressionTypeError):
        evaluate(Constant(1.0), PANEL)
    with pytest.raises(ExpressionError, match="not in the panel"):
        evaluate("ts_mean($vwap_adj, 5)", PANEL)


GRAMMAR = Grammar(max_depth=4, max_size=10, max_lookback=40)


@settings(max_examples=60, deadline=None)
@given(seed=st.integers(0, 2**32 - 1), cut=st.integers(45, 85))
def test_evaluation_is_causal(seed: int, cut: int) -> None:
    """Changing data after row `cut` never changes outputs at rows <= `cut`."""
    node = sample_expression(GRAMMAR, np.random.default_rng(seed))
    rng = np.random.default_rng(seed + 1)
    perturbed = {
        name: np.vstack([v[: cut + 1], v[cut + 1 :] * rng.uniform(0.5, 1.5, v[cut + 1 :].shape)])
        for name, v in PANEL.fields.items()
    }
    future_changed = MarketPanel(PANEL.timestamps, PANEL.instruments, perturbed)
    base = evaluate(node, PANEL)[: cut + 1]
    other = evaluate(node, future_changed)[: cut + 1]
    np.testing.assert_array_equal(base, other)


@settings(max_examples=60, deadline=None)
@given(seed=st.integers(0, 2**32 - 1))
def test_lookback_bounds_the_first_valid_row(seed: int) -> None:
    node = sample_expression(GRAMMAR, np.random.default_rng(seed))
    out = evaluate(node, PANEL)
    assert np.isnan(out[: node.lookback]).all()


@pytest.mark.parametrize(
    "text", ["delay($close, 5)", "ts_mean(delta($close, 2), 4)", "ts_corr($close, $open, 10)"]
)
def test_lookback_is_exact_for_simple_chains(text: str) -> None:
    node = parse(text)
    out = evaluate(node, PANEL)
    assert np.isnan(out[: node.lookback]).all() and np.isfinite(out[node.lookback]).all()


def test_cache_reuses_shared_subexpressions_and_respects_fingerprints() -> None:
    cache = ExpressionCache()
    evaluate("ts_mean($close, 5)", PANEL, cache)
    misses = cache.stats.misses
    out = evaluate("sub(ts_mean($close, 5), $open)", PANEL, cache)
    assert cache.stats.hits >= 1 and cache.stats.misses == misses + 3 - 1
    with pytest.raises(ValueError):
        out[0, 0] = 1.0
    other = make_synthetic_panel(SyntheticSpec(n_periods=90, n_instruments=8, seed=6))
    hits = cache.stats.hits
    evaluate("ts_mean($close, 5)", other, cache)
    assert cache.stats.hits == hits
    assert cache.invalidate(other.fingerprint) == 2
    assert cache.invalidate() >= 1 and len(cache) == 0


def test_cache_evicts_least_recently_used_entries() -> None:
    one_array = PANEL.field("close").nbytes
    cache = ExpressionCache(max_bytes=2 * one_array)
    for text in ("ts_mean($close, 5)", "ts_mean($close, 6)"):
        evaluate(text, PANEL, cache)
    assert cache.stats.evictions >= 1 and cache.stats.bytes_used <= 2 * one_array
    cache.clear()
    assert cache.stats == type(cache.stats)(0, 0, 0, 0, 0)


def test_cached_and_uncached_results_are_identical() -> None:
    node = sample_expression(GRAMMAR, np.random.default_rng(99))
    np.testing.assert_array_equal(evaluate(node, PANEL), evaluate(node, PANEL, ExpressionCache()))
