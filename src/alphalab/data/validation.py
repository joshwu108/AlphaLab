"""Validation of long-format market data against :class:`DataContract`.

Validation checks and normalizes representation (dtypes, ordering, duplicate
keys) but never imputes, filters outliers, or otherwise changes the economic
content of a row. Every rule that rejects data raises
:class:`~alphalab.errors.DataValidationError` with the offending count and an
example key so that vendor problems can be traced.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from alphalab.data.contract import (
    ADJ_FACTOR_COLUMN,
    INSTRUMENT_COLUMN,
    OPTIONAL_COLUMNS,
    PRICE_COLUMNS,
    REQUIRED_COLUMNS,
    TIMESTAMP_COLUMN,
    VOLUME_COLUMN,
    VWAP_COLUMN,
    DataContract,
)
from alphalab.errors import DataValidationError


@dataclass(frozen=True)
class ValidationReport:
    """Summary of a successful validation pass.

    Attributes:
        n_rows_in: Rows received.
        n_rows_out: Rows returned after duplicate handling.
        n_duplicates_dropped: Rows removed by a ``keep_*`` duplicate policy.
        n_instruments: Distinct instruments.
        n_periods: Distinct timestamps.
        start: First timestamp (ISO 8601).
        end: Last timestamp (ISO 8601).
        missing_values: NaN count per numeric column.
        warnings: Non-fatal observations (for example, timezone conversion).
    """

    n_rows_in: int
    n_rows_out: int
    n_duplicates_dropped: int
    n_instruments: int
    n_periods: int
    start: str
    end: str
    missing_values: Mapping[str, int] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


def _example(frame: pd.DataFrame, mask: pd.Series) -> str:
    row = frame.loc[mask].iloc[0]
    return f"(timestamp={row[TIMESTAMP_COLUMN]}, instrument={row[INSTRUMENT_COLUMN]})"


def _normalize_timestamps(frame: pd.DataFrame, warnings: list[str]) -> pd.Series:
    try:
        ts = pd.to_datetime(frame[TIMESTAMP_COLUMN], utc=False)
    except (ValueError, TypeError) as exc:
        raise DataValidationError(f"timestamp column is not parseable: {exc}") from exc
    if isinstance(ts.dtype, pd.DatetimeTZDtype):
        # Panels use timezone-naive UTC so that fingerprints do not depend on
        # the vendor's display timezone.
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
        warnings.append("timezone-aware timestamps were converted to naive UTC")
    if ts.isna().any():
        raise DataValidationError(f"{int(ts.isna().sum())} rows have a missing timestamp")
    return ts.astype("datetime64[ns]")


def _coerce_numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for column in columns:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype(np.float64)
        except (ValueError, TypeError) as exc:
            raise DataValidationError(f"column {column!r} is not numeric: {exc}") from exc
        values = frame[column].to_numpy()
        if np.isinf(values).any():
            raise DataValidationError(f"column {column!r} contains infinite values")


def _check_signs(frame: pd.DataFrame, price_columns: tuple[str, ...]) -> None:
    for column in price_columns:
        bad = frame[column] <= 0
        if bad.any():
            raise DataValidationError(
                f"{int(bad.sum())} non-positive values in {column!r}, e.g. {_example(frame, bad)}"
            )
    bad = frame[VOLUME_COLUMN] < 0
    if bad.any():
        raise DataValidationError(f"{int(bad.sum())} negative volumes, e.g. {_example(frame, bad)}")
    if ADJ_FACTOR_COLUMN in frame:
        bad = frame[ADJ_FACTOR_COLUMN] <= 0
        if bad.any():
            raise DataValidationError(
                f"{int(bad.sum())} non-positive adjustment factors, e.g. {_example(frame, bad)}"
            )


def _check_ohlc(frame: pd.DataFrame, tolerance: float) -> None:
    o, h, lo, c = (frame[col] for col in PRICE_COLUMNS)
    complete = o.notna() & h.notna() & lo.notna() & c.notna()
    slack_high = h * (1 + tolerance)
    slack_low = lo * (1 - tolerance)
    bad = complete & ((slack_high < np.maximum(o, c)) | (slack_low > np.minimum(o, c)) | (lo > h))
    if bad.any():
        raise DataValidationError(
            f"{int(bad.sum())} rows violate low <= open, close <= high, e.g. {_example(frame, bad)}"
        )


def _handle_duplicates(frame: pd.DataFrame, contract: DataContract) -> tuple[pd.DataFrame, int]:
    keys = [TIMESTAMP_COLUMN, INSTRUMENT_COLUMN]
    duplicated = frame.duplicated(keys, keep=False)
    if not duplicated.any():
        return frame, 0
    if contract.duplicate_policy == "error":
        raise DataValidationError(
            f"{int(duplicated.sum())} rows share a (timestamp, instrument) key, "
            f"e.g. {_example(frame, duplicated)}; set duplicate_policy to resolve"
        )
    keep: Literal["first", "last"] = "last" if contract.duplicate_policy == "keep_last" else "first"
    # A stable sort on the input order keeps "last" meaning "last in the file".
    deduplicated = frame.drop_duplicates(keys, keep=keep)
    return deduplicated, len(frame) - len(deduplicated)


def validate_market_frame(
    frame: pd.DataFrame, contract: DataContract | None = None
) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate a long-format market-data frame and return a normalized copy.

    The returned frame has ``timestamp`` as ``datetime64[ns]`` (naive UTC),
    ``instrument`` as ``str``, numeric fields as ``float64``, a fresh
    ``RangeIndex``, and is sorted by ``(timestamp, instrument)``. The input
    frame is never modified.

    Args:
        frame: Long-format bars containing at least
            :data:`~alphalab.data.contract.REQUIRED_COLUMNS`.
        contract: Validation rules; defaults to :class:`DataContract()`.

    Returns:
        The normalized frame and a :class:`ValidationReport`.

    Raises:
        DataValidationError: The frame violates the contract.
    """
    contract = contract or DataContract()
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise DataValidationError(f"missing required columns: {missing}")
    if frame.empty:
        raise DataValidationError("market data frame is empty")

    warnings: list[str] = []
    out = frame.copy()
    out[TIMESTAMP_COLUMN] = _normalize_timestamps(out, warnings)
    if out[INSTRUMENT_COLUMN].isna().any():
        raise DataValidationError("instrument identifiers must not be missing")
    out[INSTRUMENT_COLUMN] = out[INSTRUMENT_COLUMN].astype(str)
    if (out[INSTRUMENT_COLUMN].str.strip() == "").any():
        raise DataValidationError("instrument identifiers must not be blank")

    present_optional = tuple(c for c in OPTIONAL_COLUMNS if c in out.columns)
    numeric = (*PRICE_COLUMNS, VOLUME_COLUMN, *present_optional)
    _coerce_numeric(out, numeric)
    if not contract.allow_missing_values:
        counts = out[list(numeric)].isna().sum()
        if counts.any():
            raise DataValidationError(
                f"missing values are not allowed: {counts[counts > 0].to_dict()}"
            )
    price_like = PRICE_COLUMNS + ((VWAP_COLUMN,) if VWAP_COLUMN in out else ())
    _check_signs(out, price_like)
    if contract.check_ohlc_consistency:
        _check_ohlc(out, contract.relative_tolerance)

    out, dropped = _handle_duplicates(out, contract)
    out = out.sort_values([TIMESTAMP_COLUMN, INSTRUMENT_COLUMN], kind="stable").reset_index(
        drop=True
    )

    n_instruments = int(out[INSTRUMENT_COLUMN].nunique())
    n_periods = int(out[TIMESTAMP_COLUMN].nunique())
    if n_instruments < contract.min_instruments:
        raise DataValidationError(
            f"{n_instruments} instruments found; contract requires {contract.min_instruments}"
        )
    if n_periods < contract.min_periods:
        raise DataValidationError(
            f"{n_periods} timestamps found; contract requires {contract.min_periods}"
        )
    report = ValidationReport(
        n_rows_in=len(frame),
        n_rows_out=len(out),
        n_duplicates_dropped=dropped,
        n_instruments=n_instruments,
        n_periods=n_periods,
        start=out[TIMESTAMP_COLUMN].iloc[0].isoformat(),
        end=out[TIMESTAMP_COLUMN].iloc[-1].isoformat(),
        missing_values={c: int(out[c].isna().sum()) for c in numeric},
        warnings=tuple(warnings),
    )
    return out, report
