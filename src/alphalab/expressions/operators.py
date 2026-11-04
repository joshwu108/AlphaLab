"""Operator registry and deterministic CPU kernels.

Every operator is declared once as an :class:`OperatorSpec`. The spec drives
type checking, serialization, lookback analysis, grammar sampling and the RL
action space, so adding an operator here makes it available everywhere.

Semantics shared by all kernels
-------------------------------
* Inputs and outputs are ``float64`` arrays of shape ``(T, N)``: rows are
  sessions in chronological order, columns are instruments.
* **Causality.** Time-series kernels read only rows ``<= t`` to produce row
  ``t``. Windows are positive integers, so no operator can reach forward.
* **Missing data.** Rolling kernels are strict: any NaN inside a window makes
  that output NaN. The first ``lookback`` rows are NaN by construction.
* **Non-finite results** (division by ~0, overflow) become NaN; infinities are
  never propagated.
* Cross-sectional kernels operate within one row and need at least two finite
  values; otherwise the row is NaN.
* **Zero dispersion is detected exactly.** A window or cross-section whose
  values are all equal (for example a halted, stale price) has standard
  deviation exactly 0 and undefined correlation/z-score (NaN). The test is
  ``max == min`` rather than ``std < epsilon``: rounding in the mean of a large
  constant such as 123456.789 leaves a spurious std near 1e-11, which an
  absolute epsilon would accept and a later ``div`` would blow up.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

Category = Literal["elementwise", "time_series", "cross_sectional"]

# Denominators with absolute value below this are treated as zero.
EPSILON = 1e-12


def sanitize(values: np.ndarray) -> np.ndarray:
    """Return ``values`` with every non-finite entry replaced by NaN."""
    out = np.asarray(values, dtype=np.float64)
    if not np.isfinite(out).all():
        out = np.where(np.isfinite(out), out, np.nan)
    return out


def row_varies(x: np.ndarray) -> np.ndarray:
    """``(T, 1)`` mask: ``True`` where a row's finite values are not all equal."""
    finite = np.isfinite(x)
    high = np.where(finite, x, -np.inf).max(axis=1, keepdims=True)
    low = np.where(finite, x, np.inf).min(axis=1, keepdims=True)
    return high > low


def _rolling(x: np.ndarray, window: int, reduce: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Apply ``reduce`` over trailing windows; rows before ``window - 1`` are NaN."""
    out = np.full(x.shape, np.nan)
    if window <= x.shape[0]:
        view = sliding_window_view(x, window, axis=0)  # (T - w + 1, N, w); last = newest
        out[window - 1 :] = reduce(view)
    return out


def _delay(x: np.ndarray, d: int) -> np.ndarray:
    out = np.full(x.shape, np.nan)
    if d < x.shape[0]:
        out[d:] = x[:-d]
    return out


def _delta(x: np.ndarray, d: int) -> np.ndarray:
    return x - _delay(x, d)


def _ts_mean(x: np.ndarray, w: int) -> np.ndarray:
    return _rolling(x, w, lambda v: v.mean(axis=-1))


def _ts_sum(x: np.ndarray, w: int) -> np.ndarray:
    return _rolling(x, w, lambda v: v.sum(axis=-1))


def _ts_std(x: np.ndarray, w: int) -> np.ndarray:
    # Sample standard deviation (ddof=1), the convention of pandas and most
    # factor libraries; requires w >= 2. Constant windows are exactly 0.
    def reduce(v: np.ndarray) -> np.ndarray:
        std = v.std(axis=-1, ddof=1)
        return np.where(v.max(axis=-1) == v.min(axis=-1), 0.0, std)

    return _rolling(x, w, reduce)


def _ts_min(x: np.ndarray, w: int) -> np.ndarray:
    return _rolling(x, w, lambda v: v.min(axis=-1))


def _ts_max(x: np.ndarray, w: int) -> np.ndarray:
    return _rolling(x, w, lambda v: v.max(axis=-1))


def _ts_ema(x: np.ndarray, w: int) -> np.ndarray:
    # Truncated exponential moving average: smoothing alpha = 2 / (w + 1)
    # applied over exactly the last w rows with renormalized weights. Truncation
    # gives the EMA the same finite lookback and missing-data rule as the other
    # rolling operators, so its output never depends on how much history
    # precedes the panel.
    alpha = 2.0 / (w + 1.0)
    weights = (1.0 - alpha) ** np.arange(w - 1, -1, -1, dtype=np.float64)
    weights /= weights.sum()
    return _rolling(x, w, lambda v: v @ weights)


def _ts_corr(x: np.ndarray, y: np.ndarray, w: int) -> np.ndarray:
    def reduce_pair(stacked: np.ndarray) -> np.ndarray:
        a, b = stacked[0], stacked[1]
        varies = (a.max(axis=-1) > a.min(axis=-1)) & (b.max(axis=-1) > b.min(axis=-1))
        a = a - a.mean(axis=-1, keepdims=True)
        b = b - b.mean(axis=-1, keepdims=True)
        denom = np.sqrt((a * a).sum(axis=-1) * (b * b).sum(axis=-1))
        cov = (a * b).sum(axis=-1)
        ok = varies & (denom > EPSILON)
        return np.where(ok, cov / np.where(ok, denom, 1.0), np.nan)

    out = np.full(x.shape, np.nan)
    if w <= x.shape[0]:
        vx = sliding_window_view(x, w, axis=0)
        vy = sliding_window_view(y, w, axis=0)
        out[w - 1 :] = reduce_pair(np.stack([vx, vy]))
    return out


def _cs_rank(x: np.ndarray) -> np.ndarray:
    # Average-tie ranks scaled to [0, 1] by (rank - 1) / (count - 1).
    frame = pd.DataFrame(x)
    ranks = frame.rank(axis=1, method="average").to_numpy(dtype=np.float64)
    count = np.isfinite(x).sum(axis=1, keepdims=True).astype(np.float64)
    scaled = (ranks - 1.0) / np.where(count > 1, count - 1.0, 1.0)
    return np.where(count > 1, scaled, np.nan)


def _cs_zscore(x: np.ndarray) -> np.ndarray:
    finite = np.isfinite(x)
    count = finite.sum(axis=1, keepdims=True).astype(np.float64)
    filled = np.where(finite, x, 0.0)
    mean = filled.sum(axis=1, keepdims=True) / np.where(count > 0, count, 1.0)
    centered = np.where(finite, x - mean, 0.0)
    std = np.sqrt((centered**2).sum(axis=1, keepdims=True) / np.where(count > 0, count, 1.0))
    ok = (count > 1) & row_varies(x) & (std > EPSILON)
    return np.where(ok & finite, (x - mean) / np.where(ok, std, 1.0), np.nan)


def _div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Protected division: |b| < EPSILON yields NaN rather than a huge value,
    # so near-zero denominators cannot dominate correlation-based rewards.
    b = np.asarray(b, dtype=np.float64)
    safe = np.abs(b) >= EPSILON
    return np.where(safe, np.asarray(a) / np.where(safe, b, 1.0), np.nan)


@dataclass(frozen=True)
class OperatorSpec:
    """Declaration of one operator.

    Attributes:
        name: Serialized name, for example ``"ts_mean"``.
        arity: Number of expression operands (the window is not counted).
        category: ``elementwise``, ``time_series`` or ``cross_sectional``.
        takes_window: Whether the operator has an integer window parameter.
        min_window: Smallest legal window.
        lookback_offset: History rows consumed are ``window + lookback_offset``;
            ``delay(x, d)`` consumes ``d`` rows (offset 0) while a rolling
            reduction over ``w`` rows consumes ``w - 1`` (offset -1).
        commutative: Operand order is irrelevant; used for canonical ordering.
        allows_constant: Operands may be constants (binary elementwise only).
            At least one operand must still be a series.
        kernel: Implementation; receives operand arrays/scalars and, if
            ``takes_window``, the window as the final argument.
        description: One-line meaning for documentation and CLI listings.
    """

    name: str
    arity: int
    category: Category
    takes_window: bool
    kernel: Callable[..., np.ndarray]
    min_window: int = 1
    lookback_offset: int = 0
    commutative: bool = False
    allows_constant: bool = False
    description: str = ""

    def lookback(self, window: int | None) -> int:
        """Rows of history this operator adds on top of its operands' lookback."""
        if not self.takes_window or window is None:
            return 0
        return window + self.lookback_offset


def _spec(
    name: str, arity: int, category: Category, kernel: Callable[..., np.ndarray], **kw: object
) -> OperatorSpec:
    return OperatorSpec(name=name, arity=arity, category=category, kernel=kernel, **kw)  # type: ignore[arg-type]


OPERATORS: dict[str, OperatorSpec] = {
    spec.name: spec
    for spec in (
        _spec(
            "add",
            2,
            "elementwise",
            np.add,
            takes_window=False,
            commutative=True,
            allows_constant=True,
            description="a + b",
        ),
        _spec(
            "sub",
            2,
            "elementwise",
            np.subtract,
            takes_window=False,
            allows_constant=True,
            description="a - b",
        ),
        _spec(
            "mul",
            2,
            "elementwise",
            np.multiply,
            takes_window=False,
            commutative=True,
            allows_constant=True,
            description="a * b",
        ),
        _spec(
            "div",
            2,
            "elementwise",
            _div,
            takes_window=False,
            allows_constant=True,
            description="a / b, NaN where |b| < 1e-12",
        ),
        _spec("neg", 1, "elementwise", np.negative, takes_window=False, description="-a"),
        _spec("abs", 1, "elementwise", np.abs, takes_window=False, description="|a|"),
        _spec(
            "delay",
            1,
            "time_series",
            _delay,
            takes_window=True,
            min_window=1,
            description="a[t - d]",
        ),
        _spec(
            "delta",
            1,
            "time_series",
            _delta,
            takes_window=True,
            min_window=1,
            description="a[t] - a[t - d]",
        ),
        _spec(
            "ts_mean",
            1,
            "time_series",
            _ts_mean,
            takes_window=True,
            min_window=2,
            lookback_offset=-1,
            description="mean of the last w rows",
        ),
        _spec(
            "ts_sum",
            1,
            "time_series",
            _ts_sum,
            takes_window=True,
            min_window=2,
            lookback_offset=-1,
            description="sum of the last w rows",
        ),
        _spec(
            "ts_std",
            1,
            "time_series",
            _ts_std,
            takes_window=True,
            min_window=2,
            lookback_offset=-1,
            description="sample std of the last w rows",
        ),
        _spec(
            "ts_min",
            1,
            "time_series",
            _ts_min,
            takes_window=True,
            min_window=2,
            lookback_offset=-1,
            description="minimum of the last w rows",
        ),
        _spec(
            "ts_max",
            1,
            "time_series",
            _ts_max,
            takes_window=True,
            min_window=2,
            lookback_offset=-1,
            description="maximum of the last w rows",
        ),
        _spec(
            "ts_ema",
            1,
            "time_series",
            _ts_ema,
            takes_window=True,
            min_window=2,
            lookback_offset=-1,
            description="EMA (alpha = 2/(w+1)) truncated to w rows",
        ),
        _spec(
            "ts_corr",
            2,
            "time_series",
            _ts_corr,
            takes_window=True,
            min_window=3,
            lookback_offset=-1,
            commutative=True,
            description="Pearson correlation of a and b over the last w rows",
        ),
        _spec(
            "cs_rank",
            1,
            "cross_sectional",
            _cs_rank,
            takes_window=False,
            description="per-session rank scaled to [0, 1]",
        ),
        _spec(
            "cs_zscore",
            1,
            "cross_sectional",
            _cs_zscore,
            takes_window=False,
            description="per-session (a - mean) / std",
        ),
    )
}


def get_operator(name: str) -> OperatorSpec:
    """Return the spec for ``name``.

    Raises:
        KeyError: Unknown operator; the message lists the registered names.
    """
    try:
        return OPERATORS[name]
    except KeyError:
        raise KeyError(f"unknown operator {name!r}; known: {sorted(OPERATORS)}") from None


# Public aliases of kernels reused outside the expression engine.
cross_sectional_zscore = _cs_zscore
cross_sectional_rank = _cs_rank
