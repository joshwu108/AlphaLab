"""Market-data contract, validation, panels, labels and chronological splits."""

from __future__ import annotations

from alphalab.data.contract import DataContract
from alphalab.data.io import load_market_data, read_market_frame, write_market_frame
from alphalab.data.labels import forward_returns, panel_forward_returns
from alphalab.data.panel import MarketPanel, panel_from_frame
from alphalab.data.splits import (
    ChronologicalSplit,
    Partition,
    split_by_boundaries,
    split_by_dates,
    split_by_fractions,
)
from alphalab.data.synthetic import SyntheticSpec, make_synthetic_frame, make_synthetic_panel
from alphalab.data.universe import InstrumentInfo, Universe
from alphalab.data.validation import ValidationReport, validate_market_frame

__all__ = [
    "ChronologicalSplit",
    "DataContract",
    "InstrumentInfo",
    "MarketPanel",
    "Partition",
    "SyntheticSpec",
    "Universe",
    "ValidationReport",
    "forward_returns",
    "load_market_data",
    "make_synthetic_frame",
    "make_synthetic_panel",
    "panel_forward_returns",
    "panel_from_frame",
    "read_market_frame",
    "split_by_boundaries",
    "split_by_dates",
    "split_by_fractions",
    "validate_market_frame",
    "write_market_frame",
]
