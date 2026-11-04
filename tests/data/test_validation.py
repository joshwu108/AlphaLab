"""Tests for the market-data contract and validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphalab.data import DataContract, make_synthetic_frame, validate_market_frame
from alphalab.data.synthetic import SyntheticSpec
from alphalab.errors import DataValidationError


def _frame() -> pd.DataFrame:
    return make_synthetic_frame(SyntheticSpec(n_periods=20, n_instruments=3, seed=1))


def test_valid_frame_is_normalized_and_sorted() -> None:
    raw = _frame().sample(frac=1.0, random_state=0)
    out, report = validate_market_frame(raw)
    assert report.n_rows_in == report.n_rows_out == len(raw)
    assert report.n_instruments == 3 and report.n_periods == 20
    assert out["timestamp"].is_monotonic_increasing
    assert out["timestamp"].dtype == np.dtype("datetime64[ns]")
    assert list(out.index) == list(range(len(out)))


def test_validation_does_not_modify_the_input() -> None:
    raw = _frame()
    snapshot = raw.copy()
    validate_market_frame(raw.sample(frac=1.0, random_state=1))
    pd.testing.assert_frame_equal(raw, snapshot)


def test_missing_required_column_is_reported() -> None:
    with pytest.raises(DataValidationError, match="missing required columns"):
        validate_market_frame(_frame().drop(columns=["volume"]))


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("close", -1.0, "non-positive"),
        ("volume", -5.0, "negative volumes"),
        ("high", 1e-6, "violate"),
        ("open", np.inf, "infinite"),
    ],
)
def test_bad_values_are_rejected(column: str, value: float, message: str) -> None:
    frame = _frame()
    frame.loc[3, column] = value
    with pytest.raises(DataValidationError, match=message):
        validate_market_frame(frame)


def test_duplicates_raise_by_default_and_can_keep_last() -> None:
    frame = _frame()
    dup = frame.iloc[[5]].copy()
    dup["close"] = dup["close"] * 1.0001
    dup["high"] = np.maximum(dup["high"], dup["close"])
    doubled = pd.concat([frame, dup], ignore_index=True)
    with pytest.raises(DataValidationError, match="share a"):
        validate_market_frame(doubled)
    out, report = validate_market_frame(doubled, DataContract(duplicate_policy="keep_last"))
    assert report.n_duplicates_dropped == 1
    key = (out["timestamp"] == dup["timestamp"].iloc[0]) & (
        out["instrument"] == dup["instrument"].iloc[0]
    )
    assert out.loc[key, "close"].item() == pytest.approx(dup["close"].item())


def test_missing_values_are_counted_or_rejected() -> None:
    frame = _frame()
    frame.loc[2, "volume"] = np.nan
    _, report = validate_market_frame(frame)
    assert report.missing_values["volume"] == 1
    with pytest.raises(DataValidationError, match="not allowed"):
        validate_market_frame(frame, DataContract(allow_missing_values=False))


def test_timezone_aware_timestamps_become_naive_utc() -> None:
    frame = _frame()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"]).dt.tz_localize("America/New_York")
    out, report = validate_market_frame(frame)
    assert out["timestamp"].dt.tz is None
    assert report.warnings


def test_too_few_instruments_is_rejected() -> None:
    frame = _frame()
    frame = frame[frame["instrument"] == frame["instrument"].iloc[0]]
    with pytest.raises(DataValidationError, match="instruments found"):
        validate_market_frame(frame)


def test_contract_rejects_unknown_duplicate_policy() -> None:
    with pytest.raises(ValueError, match="duplicate_policy"):
        DataContract(duplicate_policy="ignore")  # type: ignore[arg-type]
