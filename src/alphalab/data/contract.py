"""Normalized market-data contract.

AlphaLab ingests *long-format* daily bars: one row per ``(timestamp,
instrument)`` with OHLCV fields. This module only declares the contract; the
checks live in :mod:`alphalab.data.validation` and research-specific
preprocessing (panels, labels, splits) lives elsewhere, so that validation never
silently changes the meaning of the data.

Adjusted-price semantics
------------------------
Prices are taken as given unless the optional ``adj_factor`` column is present.
``adj_factor`` must be the *cumulative forward adjustment factor known at that
timestamp*: it starts at an arbitrary positive value and is multiplied at each
split or dividend **on or after** the event date. Adjusted prices are
``price * adj_factor`` and adjusted volume is ``volume / adj_factor``.

Back-adjusted series (scaled so that the *latest* price is unadjusted) are not
causal: their level at time ``t`` depends on corporate actions after ``t``.
Scale-free expressions (ratios, ranks, correlations) are unaffected, but
level-dependent expressions would leak. Vendors' back-adjusted data must be
converted to forward factors before ingestion.

Sessions
--------
Each timestamp identifies one trading session (one daily bar). Intraday data is
out of scope for this version; timestamps with a time-of-day component are
accepted but each distinct timestamp is still treated as one session.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TIMESTAMP_COLUMN = "timestamp"
INSTRUMENT_COLUMN = "instrument"
PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")
VOLUME_COLUMN = "volume"
REQUIRED_COLUMNS: tuple[str, ...] = (
    TIMESTAMP_COLUMN,
    INSTRUMENT_COLUMN,
    *PRICE_COLUMNS,
    VOLUME_COLUMN,
)
# Optional columns with defined semantics. Any other column is preserved by
# validation but ignored when building panels unless requested explicitly.
VWAP_COLUMN = "vwap"
ADJ_FACTOR_COLUMN = "adj_factor"
OPTIONAL_COLUMNS: tuple[str, ...] = (VWAP_COLUMN, ADJ_FACTOR_COLUMN)

# Numeric fields that become (time x instrument) panel features.
FEATURE_FIELDS: tuple[str, ...] = (*PRICE_COLUMNS, VOLUME_COLUMN, VWAP_COLUMN)

DuplicatePolicy = Literal["error", "keep_last", "keep_first"]


@dataclass(frozen=True)
class DataContract:
    """Tunable rules applied by :func:`alphalab.data.validation.validate_market_frame`.

    Attributes:
        duplicate_policy: What to do with repeated ``(timestamp, instrument)``
            keys. ``"error"`` is the default because duplicates usually signal
            a vendor or join bug rather than legitimate data.
        allow_missing_values: Whether NaN values are permitted in numeric
            fields. Missing values are kept as NaN; filling them is a
            preprocessing decision, not a validation decision.
        check_ohlc_consistency: Require ``low <= min(open, close)`` and
            ``high >= max(open, close)`` on rows where all four are present.
        relative_tolerance: Relative slack for the OHLC consistency check,
            absorbing rounding in vendor data.
        min_instruments: Minimum number of distinct instruments; cross-sectional
            statistics are meaningless with fewer than two.
        min_periods: Minimum number of distinct timestamps.
    """

    duplicate_policy: DuplicatePolicy = "error"
    allow_missing_values: bool = True
    check_ohlc_consistency: bool = True
    relative_tolerance: float = 1e-9
    min_instruments: int = 2
    min_periods: int = 2

    def __post_init__(self) -> None:
        if self.duplicate_policy not in ("error", "keep_last", "keep_first"):
            raise ValueError(f"unknown duplicate_policy {self.duplicate_policy!r}")
        if self.relative_tolerance < 0:
            raise ValueError("relative_tolerance must be non-negative")
        if self.min_instruments < 1 or self.min_periods < 1:
            raise ValueError("min_instruments and min_periods must be at least 1")
