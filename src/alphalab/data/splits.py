"""Chronological train / validation / test partitions with purging and embargo.

Partitions are half-open ranges of *prediction rows*. A row ``t`` belongs to a
partition when its label interval ``[t, t + horizon]`` (see
:mod:`alphalab.data.labels`) lies entirely before the next partition's first
row, minus an optional embargo::

    rows:   0 ............ c1 ............ c2 ............ T
            |--- train ---|  |--- valid ---|  |--- test ----|
                        ^purge+embargo   ^purge+embargo   ^last `horizon` rows unlabeled

* **Purge.** Train rows ``t >= c1 - horizon`` are removed because their label
  interval reaches the first validation row; likewise for validation before
  ``c2``. This prevents overlapping-return leakage when ``horizon > 1``.
* **Embargo.** An additional ``embargo`` rows are removed before each boundary
  to absorb serial correlation that outlives the label interval.
* **Features** at a row may use any earlier row, including rows from an earlier
  partition. That is causal: earlier data is known at prediction time.

The test partition is meant to be read exactly once, after model selection;
:class:`alphalab.evaluation.evaluator.FactorEvaluator` enforces that.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from alphalab.errors import DataValidationError

PARTITION_NAMES = ("train", "valid", "test")


@dataclass(frozen=True)
class Partition:
    """Half-open range of prediction rows ``[start, stop)``."""

    name: str
    start: int
    stop: int

    @property
    def size(self) -> int:
        """Number of rows."""
        return self.stop - self.start

    def as_slice(self) -> slice:
        """Row slice for indexing ``(T, N)`` arrays."""
        return slice(self.start, self.stop)


@dataclass(frozen=True)
class ChronologicalSplit:
    """Three disjoint, ordered partitions plus the parameters that produced them."""

    train: Partition
    valid: Partition
    test: Partition
    horizon: int
    embargo: int
    n_periods: int

    def __post_init__(self) -> None:
        ordered = (self.train, self.valid, self.test)
        for left, right in itertools.pairwise(ordered):
            # Label interval of the last row of `left` must end before `right` starts.
            if left.stop - 1 + self.horizon + self.embargo >= right.start:
                raise DataValidationError(
                    f"{left.name} labels overlap {right.name}: purge/embargo violated"
                )
        if self.test.stop + self.horizon > self.n_periods:
            raise DataValidationError("test partition includes rows without a full label")

    def partition(self, name: str) -> Partition:
        """Return the partition called ``name`` (``train``, ``valid`` or ``test``)."""
        if name not in PARTITION_NAMES:
            raise KeyError(f"unknown partition {name!r}; expected one of {PARTITION_NAMES}")
        part: Partition = getattr(self, name)
        return part

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable description for manifests."""
        return {
            "horizon": self.horizon,
            "embargo": self.embargo,
            "n_periods": self.n_periods,
            **{p.name: [p.start, p.stop] for p in (self.train, self.valid, self.test)},
        }


def split_by_boundaries(
    n_periods: int,
    valid_start: int,
    test_start: int,
    *,
    horizon: int,
    embargo: int = 0,
    first_row: int = 0,
    min_rows: int = 1,
) -> ChronologicalSplit:
    """Build a purged split from explicit row boundaries.

    Args:
        n_periods: Total rows ``T``.
        valid_start: First validation row ``c1``.
        test_start: First test row ``c2``.
        horizon: Label horizon in rows.
        embargo: Extra rows removed before each boundary.
        first_row: First usable training row, for example the maximum
            expression lookback when it is known in advance.
        min_rows: Minimum rows each partition must keep after purging.

    Raises:
        DataValidationError: Boundaries are out of order or a partition is
            too small after purging.
    """
    if horizon < 1 or embargo < 0:
        raise DataValidationError("horizon must be >= 1 and embargo >= 0")
    if not 0 <= first_row < valid_start < test_start < n_periods:
        raise DataValidationError(
            f"need 0 <= first_row < valid_start < test_start < n_periods, got "
            f"{first_row}, {valid_start}, {test_start}, {n_periods}"
        )
    gap = horizon + embargo
    train = Partition("train", first_row, valid_start - gap)
    valid = Partition("valid", valid_start, test_start - gap)
    test = Partition("test", test_start, n_periods - horizon)
    for part in (train, valid, test):
        if part.size < min_rows:
            raise DataValidationError(
                f"{part.name} partition has {max(part.size, 0)} rows after purging; "
                f"need at least {min_rows}"
            )
    return ChronologicalSplit(train, valid, test, horizon, embargo, n_periods)


def split_by_fractions(
    n_periods: int,
    train_fraction: float,
    valid_fraction: float,
    *,
    horizon: int,
    embargo: int = 0,
    first_row: int = 0,
    min_rows: int = 1,
) -> ChronologicalSplit:
    """Split by fractions of the timeline; the test fraction is the remainder."""
    if not (0 < train_fraction < 1 and 0 < valid_fraction < 1):
        raise DataValidationError("fractions must lie strictly between 0 and 1")
    if train_fraction + valid_fraction >= 1:
        raise DataValidationError("train_fraction + valid_fraction must be < 1")
    valid_start = int(np.floor(n_periods * train_fraction))
    test_start = int(np.floor(n_periods * (train_fraction + valid_fraction)))
    return split_by_boundaries(
        n_periods,
        valid_start,
        test_start,
        horizon=horizon,
        embargo=embargo,
        first_row=first_row,
        min_rows=min_rows,
    )


def split_by_dates(
    timestamps: np.ndarray,
    valid_start: str,
    test_start: str,
    *,
    horizon: int,
    embargo: int = 0,
    first_row: int = 0,
    min_rows: int = 1,
) -> ChronologicalSplit:
    """Split at the first sessions on or after ``valid_start`` and ``test_start``."""
    times = np.asarray(timestamps, dtype="datetime64[ns]")
    c1 = int(np.searchsorted(times, np.datetime64(pd.Timestamp(valid_start)), side="left"))
    c2 = int(np.searchsorted(times, np.datetime64(pd.Timestamp(test_start)), side="left"))
    return split_by_boundaries(
        times.size, c1, c2, horizon=horizon, embargo=embargo, first_row=first_row, min_rows=min_rows
    )
