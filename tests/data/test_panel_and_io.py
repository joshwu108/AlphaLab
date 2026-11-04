"""Tests for panels, ingestion, universes and synthetic data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphalab.data import (
    MarketPanel,
    SyntheticSpec,
    Universe,
    load_market_data,
    make_synthetic_frame,
    make_synthetic_panel,
    panel_from_frame,
    validate_market_frame,
    write_market_frame,
)
from alphalab.errors import DataValidationError


def test_synthetic_panel_is_deterministic() -> None:
    a = make_synthetic_panel(SyntheticSpec(n_periods=50, n_instruments=5, seed=11))
    b = make_synthetic_panel(SyntheticSpec(n_periods=50, n_instruments=5, seed=11))
    c = make_synthetic_panel(SyntheticSpec(n_periods=50, n_instruments=5, seed=12))
    assert a.fingerprint == b.fingerprint
    assert a.fingerprint != c.fingerprint
    np.testing.assert_array_equal(a.field("close"), b.field("close"))


def test_panel_arrays_are_read_only(small_panel: MarketPanel) -> None:
    with pytest.raises(ValueError):
        small_panel.field("close")[0, 0] = 1.0


def test_unknown_field_lists_alternatives(small_panel: MarketPanel) -> None:
    with pytest.raises(KeyError, match="available"):
        small_panel.field("adj_close")


@pytest.mark.parametrize("suffix", [".csv", ".parquet"])
def test_file_round_trip_preserves_fingerprint(tmp_path: Path, suffix: str) -> None:
    spec = SyntheticSpec(n_periods=30, n_instruments=4, seed=2, missing_rate=0.05)
    panel = make_synthetic_panel(spec)
    path = write_market_frame(panel.to_frame(), tmp_path / f"bars{suffix}")
    frame, report = load_market_data(path)
    rebuilt = panel_from_frame(frame, metadata=dict(panel.metadata))
    assert report.n_instruments == 4
    for name in panel.fields:
        np.testing.assert_allclose(rebuilt.field(name), panel.field(name), rtol=1e-12)
    if suffix == ".parquet":
        assert rebuilt.fingerprint == panel.fingerprint


def test_column_map_renames_vendor_columns(tmp_path: Path) -> None:
    frame = make_synthetic_frame(SyntheticSpec(n_periods=12, n_instruments=2))
    vendor = frame.rename(columns={"timestamp": "date", "instrument": "ticker"})
    path = write_market_frame(vendor, tmp_path / "vendor.csv")
    out, _ = load_market_data(path, column_map={"date": "timestamp", "ticker": "instrument"})
    assert {"timestamp", "instrument"} <= set(out.columns)


def test_unsupported_suffix_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bars.xlsx"
    path.write_text("x")
    with pytest.raises(DataValidationError, match="unsupported"):
        load_market_data(path)


def test_adjustment_factor_scales_prices_and_volume() -> None:
    frame = make_synthetic_frame(SyntheticSpec(n_periods=10, n_instruments=2))
    frame["adj_factor"] = 2.0
    validated, _ = validate_market_frame(frame)
    raw = panel_from_frame(validated, adjust=False)
    adjusted = panel_from_frame(validated, adjust=True)
    np.testing.assert_allclose(adjusted.field("close"), 2.0 * raw.field("close"))
    np.testing.assert_allclose(adjusted.field("volume"), raw.field("volume") / 2.0)


def test_missing_cells_become_nan_in_the_panel() -> None:
    frame = make_synthetic_frame(SyntheticSpec(n_periods=10, n_instruments=3))
    frame = frame.drop(index=4)
    validated, _ = validate_market_frame(frame)
    panel = panel_from_frame(validated)
    assert np.isnan(panel.field("close")).sum() == 1


def test_panel_rejects_unsorted_timestamps() -> None:
    times = pd.to_datetime(["2020-01-02", "2020-01-01"]).to_numpy()
    with pytest.raises(DataValidationError, match="strictly increasing"):
        MarketPanel(times, ("A", "B"), {"close": np.ones((2, 2))})


def test_slice_and_mask_produce_new_panels(small_panel: MarketPanel) -> None:
    part = small_panel.slice_periods(10, 20)
    assert part.shape == (10, small_panel.n_instruments)
    assert part.fingerprint != small_panel.fingerprint
    mask = np.ones(small_panel.shape, dtype=bool)
    mask[:, 0] = False
    masked = small_panel.with_mask(mask)
    assert np.isnan(masked.field("close")[:, 0]).all()
    assert not np.isnan(small_panel.field("close")[:, 0]).any()


def test_universe_masks_non_members(small_panel: MarketPanel) -> None:
    first, second = small_panel.instruments[:2]
    cutoff = str(pd.Timestamp(small_panel.timestamps[50]).date())
    universe = Universe.from_records(
        [
            {"instrument": first, "last_date": cutoff, "sector": "tech"},
            {"instrument": second},
        ]
    )
    masked = universe.apply(small_panel)
    close = masked.field("close")
    assert not np.isnan(close[:51, 0]).any() and np.isnan(close[51:, 0]).all()
    assert not np.isnan(close[:, 1]).any()
    assert np.isnan(close[:, 2:]).all()  # not in the universe
    assert universe.members[0].attributes == {"sector": "tech"}


def test_universe_rejects_inverted_interval() -> None:
    with pytest.raises(DataValidationError, match="after"):
        Universe.from_records([{"instrument": "A", "first_date": "2021", "last_date": "2020"}])
