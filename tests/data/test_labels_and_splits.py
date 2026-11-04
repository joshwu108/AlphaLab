"""Tests for forward-return labels and purged chronological splits."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from alphalab.data import (
    forward_returns,
    split_by_boundaries,
    split_by_dates,
    split_by_fractions,
)
from alphalab.errors import DataValidationError


def test_forward_returns_use_exactly_horizon_rows_ahead() -> None:
    close = np.array([[100.0], [110.0], [121.0], [133.1]])
    out = forward_returns(close, 2)
    np.testing.assert_allclose(out[:2, 0], [0.21, 0.21])
    assert np.isnan(out[2:]).all()


def test_forward_returns_reject_non_positive_horizon() -> None:
    with pytest.raises(ValueError):
        forward_returns(np.ones((3, 2)), 0)


@given(
    n=st.integers(60, 400),
    horizon=st.integers(1, 5),
    embargo=st.integers(0, 3),
    c1_frac=st.floats(0.3, 0.4),
    c2_frac=st.floats(0.6, 0.8),
)
def test_purged_partitions_never_share_label_information(
    n: int, horizon: int, embargo: int, c1_frac: float, c2_frac: float
) -> None:
    c1, c2 = int(n * c1_frac), int(n * c2_frac)
    split = split_by_boundaries(n, c1, c2, horizon=horizon, embargo=embargo)
    parts = (split.train, split.valid, split.test)
    for left, right in itertools.pairwise(parts):
        last_label_row = left.stop - 1 + horizon
        assert last_label_row + embargo < right.start
    assert split.test.stop - 1 + horizon <= n - 1


def test_fraction_and_date_splits_agree() -> None:
    import pandas as pd

    times = pd.bdate_range("2020-01-01", periods=100).to_numpy()
    by_fraction = split_by_fractions(100, 0.6, 0.2, horizon=1)
    by_date = split_by_dates(
        times, str(pd.Timestamp(times[60]).date()), str(pd.Timestamp(times[80]).date()), horizon=1
    )
    assert by_fraction == by_date
    assert by_fraction.to_dict()["valid"] == [60, 79]


def test_split_rejects_partitions_emptied_by_purging() -> None:
    with pytest.raises(DataValidationError, match="rows after purging"):
        split_by_boundaries(30, 10, 12, horizon=3)


def test_split_rejects_bad_order() -> None:
    with pytest.raises(DataValidationError):
        split_by_boundaries(30, 20, 10, horizon=1)


def test_partition_lookup() -> None:
    split = split_by_fractions(100, 0.5, 0.25, horizon=1)
    assert split.partition("valid") is split.valid
    with pytest.raises(KeyError):
        split.partition("holdout")
