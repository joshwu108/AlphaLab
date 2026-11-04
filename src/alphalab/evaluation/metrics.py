"""Factor-quality metrics on ``(T, N)`` arrays.

All functions take arrays already restricted to the rows of one partition and
ignore NaN pairs. They measure *predictive association* between a factor and
forward returns, not tradable profit: an information coefficient (IC) says
nothing about transaction costs, capacity, borrow availability or market
impact, and a positive IC can coexist with a strategy that loses money after
costs.

Definitions
-----------
* **IC** (Pearson) at session ``t``: correlation across instruments between
  ``factor[t]`` and ``target[t]``.
* **Rank IC** (Spearman): the same on cross-sectional ranks.
* **ICIR**: mean IC divided by its standard deviation across sessions.
* **t-statistic**: ``mean / (std / sqrt(n))``. Overlapping multi-session labels
  make per-session ICs serially correlated, which inflates this statistic; it
  is reported for diagnostics, not as a significance test.
* **Turnover**: average one-way change of unit-gross long/short weights
  (cross-sectional z-scores scaled to ``sum |w| = 1``) between consecutive
  sessions, in ``[0, 1]``.
* **Rank autocorrelation**: mean rank correlation between ``factor[t]`` and
  ``factor[t - lag]``; high values mean a stable ordering.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from alphalab.expressions.operators import EPSILON, row_varies

DEFAULT_MIN_CROSS_SECTION = 5  # Minimum valid instruments for a per-session correlation.


def _rank_rows(values: np.ndarray) -> np.ndarray:
    return pd.DataFrame(values).rank(axis=1, method="average").to_numpy(dtype=np.float64)


def rowwise_correlation(
    a: np.ndarray,
    b: np.ndarray,
    *,
    method: str = "pearson",
    min_obs: int = DEFAULT_MIN_CROSS_SECTION,
) -> np.ndarray:
    """Per-row correlation between ``a`` and ``b`` over jointly finite entries.

    Args:
        a: ``(T, N)`` array.
        b: ``(T, N)`` array.
        method: ``"pearson"`` or ``"spearman"``.
        min_obs: Rows with fewer jointly finite pairs, or with zero variance,
            produce NaN.

    Returns:
        ``(T,)`` array of correlations.
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    if method not in ("pearson", "spearman"):
        raise ValueError(f"unknown correlation method {method!r}")
    mask = np.isfinite(a) & np.isfinite(b)
    x = np.where(mask, a, np.nan)
    y = np.where(mask, b, np.nan)
    if method == "spearman":
        x, y = _rank_rows(x), _rank_rows(y)
    count = mask.sum(axis=1).astype(np.float64)
    safe_count = np.where(count > 0, count, 1.0)
    x0 = np.where(mask, x, 0.0)
    y0 = np.where(mask, y, 0.0)
    xc = np.where(mask, x - (x0.sum(axis=1) / safe_count)[:, None], 0.0)
    yc = np.where(mask, y - (y0.sum(axis=1) / safe_count)[:, None], 0.0)
    cov = (xc * yc).sum(axis=1)
    denom = np.sqrt((xc * xc).sum(axis=1) * (yc * yc).sum(axis=1))
    varies = (
        row_varies(np.where(mask, x, np.nan))[:, 0] & row_varies(np.where(mask, y, np.nan))[:, 0]
    )
    ok = (count >= min_obs) & varies & (denom > EPSILON)
    return np.where(ok, cov / np.where(ok, denom, 1.0), np.nan)


@dataclass(frozen=True)
class SeriesSummary:
    """Summary statistics of a per-session metric series."""

    mean: float
    std: float
    ir: float
    t_stat: float
    hit_rate: float
    n: int

    def to_dict(self) -> dict[str, float | int]:
        """Plain dictionary."""
        return asdict(self)


def summarize_series(values: np.ndarray) -> SeriesSummary:
    """Mean, sample std, ratio, t-statistic and positive fraction of finite values.

    Undefined statistics (fewer than two observations, zero dispersion) are NaN.
    """
    finite = values[np.isfinite(values)]
    n = int(finite.size)
    if n == 0:
        return SeriesSummary(math.nan, math.nan, math.nan, math.nan, math.nan, 0)
    mean = float(finite.mean())
    std = float(finite.std(ddof=1)) if n > 1 else math.nan
    ir = mean / std if n > 1 and std > EPSILON else math.nan
    t_stat = ir * math.sqrt(n) if not math.isnan(ir) else math.nan
    hit = float((finite > 0).mean())
    return SeriesSummary(mean, std, ir, t_stat, hit, n)


def information_coefficient(
    factor: np.ndarray, target: np.ndarray, *, min_obs: int = DEFAULT_MIN_CROSS_SECTION
) -> np.ndarray:
    """Per-session Pearson IC."""
    return rowwise_correlation(factor, target, method="pearson", min_obs=min_obs)


def rank_information_coefficient(
    factor: np.ndarray, target: np.ndarray, *, min_obs: int = DEFAULT_MIN_CROSS_SECTION
) -> np.ndarray:
    """Per-session Spearman (rank) IC."""
    return rowwise_correlation(factor, target, method="spearman", min_obs=min_obs)


def per_instrument_correlation(
    factor: np.ndarray, target: np.ndarray, *, min_obs: int = 20
) -> np.ndarray:
    """Time-series Pearson correlation per instrument, shape ``(N,)``."""
    return rowwise_correlation(factor.T, target.T, method="pearson", min_obs=min_obs)


def factor_correlation(
    a: np.ndarray,
    b: np.ndarray,
    *,
    method: str = "pearson",
    min_obs: int = DEFAULT_MIN_CROSS_SECTION,
) -> float:
    """Mean per-session cross-sectional correlation between two factors (NaN if undefined)."""
    series = rowwise_correlation(a, b, method=method, min_obs=min_obs)
    finite = series[np.isfinite(series)]
    return float(finite.mean()) if finite.size else math.nan


def correlation_matrix(factors: list[np.ndarray], *, method: str = "pearson") -> np.ndarray:
    """Symmetric matrix of :func:`factor_correlation` values with unit diagonal."""
    k = len(factors)
    out = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            out[i, j] = out[j, i] = factor_correlation(factors[i], factors[j], method=method)
    return out


def redundancy(factors: list[np.ndarray], *, method: str = "pearson") -> float:
    """Mean absolute off-diagonal correlation of a factor set (0 for fewer than two)."""
    if len(factors) < 2:
        return 0.0
    matrix = np.abs(correlation_matrix(factors, method=method))
    upper = matrix[np.triu_indices(len(factors), k=1)]
    finite = upper[np.isfinite(upper)]
    return float(finite.mean()) if finite.size else math.nan


def unit_gross_weights(factor: np.ndarray) -> np.ndarray:
    """Demeaned per-session weights with ``sum |w| = 1``; missing entries get 0."""
    finite = np.isfinite(factor)
    count = finite.sum(axis=1, keepdims=True)
    mean = np.where(finite, factor, 0.0).sum(axis=1, keepdims=True) / np.where(count > 0, count, 1)
    centered = np.where(finite, factor - mean, 0.0)
    gross = np.abs(centered).sum(axis=1, keepdims=True)
    return np.where(gross > EPSILON, centered / np.where(gross > EPSILON, gross, 1.0), 0.0)


def turnover(factor: np.ndarray) -> float:
    """Mean one-way turnover of unit-gross weights between consecutive sessions."""
    if factor.shape[0] < 2:
        return math.nan
    weights = unit_gross_weights(factor)
    active = np.abs(weights).sum(axis=1) > 0
    both = active[1:] & active[:-1]
    if not both.any():
        return math.nan
    change = 0.5 * np.abs(np.diff(weights, axis=0)).sum(axis=1)
    return float(change[both].mean())


def rank_autocorrelation(factor: np.ndarray, lag: int = 1) -> float:
    """Mean rank correlation between ``factor[t]`` and ``factor[t - lag]``."""
    if lag < 1 or factor.shape[0] <= lag:
        return math.nan
    series = rowwise_correlation(factor[lag:], factor[:-lag], method="spearman")
    finite = series[np.isfinite(series)]
    return float(finite.mean()) if finite.size else math.nan


def block_sign_consistency(ic: np.ndarray, n_blocks: int = 4) -> float:
    """Fraction of contiguous blocks whose mean IC has the sign of the overall mean.

    A stability diagnostic: a factor whose IC flips sign between sub-periods
    scores low even if the full-sample mean looks good.
    """
    finite_mean = np.nanmean(ic) if np.isfinite(ic).any() else math.nan
    if math.isnan(finite_mean) or n_blocks < 1:
        return math.nan
    blocks = [b for b in np.array_split(ic, n_blocks) if np.isfinite(b).any()]
    if not blocks:
        return math.nan
    signs = [np.sign(np.nanmean(b)) == np.sign(finite_mean) for b in blocks]
    return float(np.mean(signs))


def coverage(factor: np.ndarray) -> float:
    """Fraction of finite entries."""
    return float(np.isfinite(factor).mean()) if factor.size else 0.0
