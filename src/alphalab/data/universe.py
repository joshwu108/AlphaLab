"""Point-in-time instrument-universe metadata.

A universe records, for each instrument, the interval during which it is
eligible for research. Applying a universe masks observations outside that
interval to NaN, so an instrument contributes to cross-sectional statistics
only while it was actually a member.

Survivorship bias is a property of the *input*: a universe built only from
instruments alive today excludes delisted names regardless of masking. The
membership intervals must therefore come from a point-in-time source.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alphalab.data.panel import MarketPanel
from alphalab.errors import DataValidationError


@dataclass(frozen=True)
class InstrumentInfo:
    """Static and membership metadata for one instrument.

    Attributes:
        instrument: Identifier matching the panel's instrument axis.
        first_date: First eligible session (inclusive); ``None`` means unbounded.
        last_date: Last eligible session (inclusive); ``None`` means unbounded.
        attributes: Static descriptive fields such as sector or exchange.
    """

    instrument: str
    first_date: str | None = None
    last_date: str | None = None
    attributes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            self.first_date
            and self.last_date
            and (pd.Timestamp(self.first_date) > pd.Timestamp(self.last_date))
        ):
            raise DataValidationError(f"{self.instrument}: first_date is after last_date")


@dataclass(frozen=True)
class Universe:
    """A set of instruments with membership intervals."""

    members: tuple[InstrumentInfo, ...]

    def __post_init__(self) -> None:
        names = [m.instrument for m in self.members]
        if len(set(names)) != len(names):
            raise DataValidationError("universe contains duplicate instruments")

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, str | None]]) -> Universe:
        """Build a universe from dictionaries with ``instrument``, ``first_date``,
        ``last_date`` and any additional string attributes."""
        members = []
        for record in records:
            data = dict(record)
            instrument = data.pop("instrument", None)
            if not instrument:
                raise DataValidationError("every universe record needs an 'instrument'")
            first = data.pop("first_date", None)
            last = data.pop("last_date", None)
            attributes = {k: str(v) for k, v in data.items() if v is not None}
            members.append(InstrumentInfo(str(instrument), first, last, attributes))
        return cls(tuple(members))

    @property
    def instruments(self) -> tuple[str, ...]:
        """Member identifiers in declaration order."""
        return tuple(m.instrument for m in self.members)

    def membership_mask(self, timestamps: np.ndarray, instruments: tuple[str, ...]) -> np.ndarray:
        """Return a boolean ``(T, N)`` mask that is ``True`` where an instrument is a member.

        Instruments on the panel axis that are absent from the universe are
        never members.
        """
        times = pd.DatetimeIndex(np.asarray(timestamps, dtype="datetime64[ns]"))
        by_name = {m.instrument: m for m in self.members}
        mask = np.zeros((len(times), len(instruments)), dtype=bool)
        for column, name in enumerate(instruments):
            info = by_name.get(name)
            if info is None:
                continue
            eligible = np.ones(len(times), dtype=bool)
            if info.first_date:
                eligible &= times >= pd.Timestamp(info.first_date)
            if info.last_date:
                eligible &= times <= pd.Timestamp(info.last_date)
            mask[:, column] = eligible
        return mask

    def apply(self, panel: MarketPanel) -> MarketPanel:
        """Return ``panel`` with non-member observations set to NaN."""
        return panel.with_mask(self.membership_mask(panel.timestamps, panel.instruments))
