"""Deterministic synthetic market data with a known planted signal.

Synthetic panels exist for testing and dry runs only. They let tests assert
that the pipeline *can* recover a signal that is known to exist, which real
data cannot provide. They make no claim about real markets.

Generative model
----------------
For each session ``t >= 1`` and instrument ``i`` the log return is::

    r[t, i] = m[t] + strength * sigma[i] * s[t-1, i] + sigma[i] * e[t, i]

where ``m`` is a common market shock, ``sigma[i]`` an instrument volatility,
``e`` i.i.d. standard normal noise and ``s[t-1, i]`` the cross-sectional
z-score of the *negative* trailing ``reversal_window``-session log return
known at ``t - 1``. The planted factor is therefore short-term reversal, which
the expression language writes as ``div(delay($close, W), $close)`` and which
predicts the one-session-ahead forward return at every row.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alphalab.data.panel import MarketPanel

DEFAULT_START = "2015-01-02"


@dataclass(frozen=True)
class SyntheticSpec:
    """Parameters of the synthetic generator.

    Attributes:
        n_periods: Sessions ``T``.
        n_instruments: Instruments ``N``.
        seed: Seed for :func:`numpy.random.default_rng`; equal specs give
            bit-identical panels.
        signal_strength: Planted predictability in units of each instrument's
            volatility. Zero produces pure noise.
        reversal_window: Lookback ``W`` of the planted reversal factor.
        missing_rate: Fraction of ``(t, i)`` cells blanked to NaN in every
            field, emulating halts and data gaps.
        start: First session date.
    """

    n_periods: int = 256
    n_instruments: int = 24
    seed: int = 0
    signal_strength: float = 0.15
    reversal_window: int = 5
    missing_rate: float = 0.0
    start: str = DEFAULT_START

    def __post_init__(self) -> None:
        if self.n_periods <= self.reversal_window + 1:
            raise ValueError("n_periods must exceed reversal_window + 1")
        if self.n_instruments < 2:
            raise ValueError("n_instruments must be at least 2")
        if not 0.0 <= self.missing_rate < 1.0:
            raise ValueError("missing_rate must lie in [0, 1)")

    @property
    def planted_expression(self) -> str:
        """Canonical expression of the planted factor."""
        return f"div(delay($close, {self.reversal_window}), $close)"


def _zscore_rows(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean()
    scale = values.std()
    return centered / scale if scale > 0 else np.zeros_like(values)


def _simulate_log_close(spec: SyntheticSpec, rng: np.random.Generator) -> np.ndarray:
    t_count, n_count, window = spec.n_periods, spec.n_instruments, spec.reversal_window
    sigma = rng.uniform(0.01, 0.03, size=n_count)
    market = rng.normal(0.0, 0.008, size=t_count)
    noise = rng.normal(0.0, 1.0, size=(t_count, n_count))
    log_close = np.empty((t_count, n_count))
    log_close[0] = np.log(rng.uniform(20.0, 200.0, size=n_count))
    for t in range(1, t_count):
        if t - 1 >= window:
            trailing = log_close[t - 1] - log_close[t - 1 - window]
            signal = _zscore_rows(-trailing)
        else:
            signal = np.zeros(n_count)
        step = market[t] + spec.signal_strength * sigma * signal + sigma * noise[t]
        log_close[t] = log_close[t - 1] + step
    return log_close


def make_synthetic_panel(spec: SyntheticSpec | None = None) -> MarketPanel:
    """Generate a deterministic OHLCV(+VWAP) panel from ``spec``.

    The OHLC fields satisfy ``low <= open, close <= high`` by construction so
    the panel round-trips through :func:`~alphalab.data.validation.validate_market_frame`.
    """
    spec = spec or SyntheticSpec()
    rng = np.random.default_rng(spec.seed)
    log_close = _simulate_log_close(spec, rng)
    close = np.exp(log_close)
    t_count, n_count = close.shape
    prev_close = np.vstack([close[:1], close[:-1]])
    open_ = prev_close * np.exp(rng.normal(0.0, 0.002, size=(t_count, n_count)))
    wick = np.abs(rng.normal(0.0, 0.004, size=(2, t_count, n_count)))
    high = np.maximum(open_, close) * np.exp(wick[0])
    low = np.minimum(open_, close) * np.exp(-wick[1])
    base_volume = rng.uniform(1e5, 5e6, size=n_count)
    volume = np.round(base_volume * rng.lognormal(0.0, 0.3, size=(t_count, n_count)))
    vwap = (high + low + close) / 3.0
    fields = {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "vwap": vwap,
    }
    if spec.missing_rate > 0:
        blank = rng.random((t_count, n_count)) < spec.missing_rate
        fields = {k: np.where(blank, np.nan, v) for k, v in fields.items()}
    timestamps = pd.bdate_range(spec.start, periods=t_count).to_numpy(dtype="datetime64[ns]")
    instruments = tuple(f"SYN{i:03d}" for i in range(n_count))
    metadata = {
        "source": "synthetic",
        "seed": str(spec.seed),
        "signal_strength": repr(spec.signal_strength),
        "reversal_window": str(spec.reversal_window),
        "missing_rate": repr(spec.missing_rate),
    }
    return MarketPanel(
        timestamps=timestamps, instruments=instruments, fields=fields, metadata=metadata
    )


def make_synthetic_frame(spec: SyntheticSpec | None = None) -> pd.DataFrame:
    """Long-format version of :func:`make_synthetic_panel`, suitable for file round trips."""
    return make_synthetic_panel(spec).to_frame()
