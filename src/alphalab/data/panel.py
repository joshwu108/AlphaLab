"""Dense (time x instrument) panels built from validated long-format data.

A :class:`MarketPanel` is the only data structure the expression engine reads.
Every field is a read-only ``float64`` array of shape ``(T, N)`` aligned to a
strictly increasing timestamp axis and a fixed instrument axis; absent
observations are NaN. Panels are immutable and carry a content fingerprint that
identifies them in caches, manifests and checkpoints.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
import pandas as pd

from alphalab.data.contract import (
    ADJ_FACTOR_COLUMN,
    FEATURE_FIELDS,
    INSTRUMENT_COLUMN,
    PRICE_COLUMNS,
    TIMESTAMP_COLUMN,
    VOLUME_COLUMN,
    VWAP_COLUMN,
)
from alphalab.errors import DataValidationError

FINGERPRINT_VERSION = "alphalab-panel-v1"


def _readonly(array: np.ndarray) -> np.ndarray:
    out = np.array(array, dtype=np.float64, copy=True, order="C")
    out.flags.writeable = False
    return out


@dataclass(frozen=True, eq=False)
class MarketPanel:
    """Immutable aligned market data.

    Attributes:
        timestamps: ``datetime64[ns]`` array of shape ``(T,)``, strictly increasing.
        instruments: Instrument identifiers of length ``N``, in column order.
        fields: Mapping from field name to a read-only ``(T, N)`` float array.
        metadata: Free-form provenance (source path, vendor, notes). Included
            in the fingerprint so that identically valued panels from
            different sources are not confused.
        fingerprint: SHA-256 hex digest of the panel contents; computed on
            construction.
    """

    timestamps: np.ndarray
    instruments: tuple[str, ...]
    fields: Mapping[str, np.ndarray]
    metadata: Mapping[str, str] = field(default_factory=dict)
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps, dtype="datetime64[ns]").copy()
        timestamps.flags.writeable = False
        if timestamps.ndim != 1 or timestamps.size == 0:
            raise DataValidationError("timestamps must be a non-empty 1-D array")
        if np.any(np.diff(timestamps.view(np.int64)) <= 0):
            raise DataValidationError("timestamps must be strictly increasing")
        instruments = tuple(str(i) for i in self.instruments)
        if len(set(instruments)) != len(instruments):
            raise DataValidationError("instrument identifiers must be unique")
        if not self.fields:
            raise DataValidationError("a panel needs at least one field")
        shape = (timestamps.size, len(instruments))
        frozen: dict[str, np.ndarray] = {}
        for name in sorted(self.fields):
            array = _readonly(self.fields[name])
            if array.shape != shape:
                raise DataValidationError(
                    f"field {name!r} has shape {array.shape}, expected {shape}"
                )
            if np.isinf(array).any():
                raise DataValidationError(f"field {name!r} contains infinite values")
            frozen[name] = array
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "instruments", instruments)
        object.__setattr__(self, "fields", MappingProxyType(frozen))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(self, "fingerprint", self._compute_fingerprint())

    def _compute_fingerprint(self) -> str:
        digest = hashlib.sha256()
        header = {
            "version": FINGERPRINT_VERSION,
            "instruments": list(self.instruments),
            "fields": sorted(self.fields),
            "metadata": dict(sorted(self.metadata.items())),
        }
        digest.update(json.dumps(header, sort_keys=True).encode())
        digest.update(self.timestamps.view(np.int64).tobytes())
        for name in sorted(self.fields):
            # NaN payload bits may differ between producers; canonicalize them.
            values = self.fields[name]
            canonical = np.where(np.isnan(values), np.nan, values)
            digest.update(name.encode())
            digest.update(canonical.tobytes())
        return digest.hexdigest()

    @property
    def n_periods(self) -> int:
        """Number of timestamps ``T``."""
        return int(self.timestamps.size)

    @property
    def n_instruments(self) -> int:
        """Number of instruments ``N``."""
        return len(self.instruments)

    @property
    def shape(self) -> tuple[int, int]:
        """``(T, N)``."""
        return self.n_periods, self.n_instruments

    def field(self, name: str) -> np.ndarray:
        """Return the read-only ``(T, N)`` array for ``name``.

        Raises:
            KeyError: The field does not exist; the message lists available fields.
        """
        try:
            return self.fields[name]
        except KeyError:
            raise KeyError(f"unknown field {name!r}; available: {sorted(self.fields)}") from None

    def slice_periods(self, start: int, stop: int) -> MarketPanel:
        """Return a new panel restricted to rows ``[start, stop)``."""
        if not 0 <= start < stop <= self.n_periods:
            raise IndexError(f"invalid period slice [{start}, {stop}) for T={self.n_periods}")
        return MarketPanel(
            timestamps=self.timestamps[start:stop],
            instruments=self.instruments,
            fields={k: v[start:stop] for k, v in self.fields.items()},
            metadata=self.metadata,
        )

    def with_mask(self, mask: np.ndarray) -> MarketPanel:
        """Return a new panel with entries where ``mask`` is ``False`` set to NaN.

        Used to apply point-in-time universe membership.
        """
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self.shape:
            raise DataValidationError(f"mask shape {mask.shape} does not match panel {self.shape}")
        return MarketPanel(
            timestamps=self.timestamps,
            instruments=self.instruments,
            fields={k: np.where(mask, v, np.nan) for k, v in self.fields.items()},
            metadata=self.metadata,
        )

    def to_frame(self) -> pd.DataFrame:
        """Return the panel in long format, one row per non-empty ``(timestamp, instrument)``."""
        index = pd.MultiIndex.from_product(
            [pd.DatetimeIndex(self.timestamps), list(self.instruments)],
            names=[TIMESTAMP_COLUMN, INSTRUMENT_COLUMN],
        )
        frame = pd.DataFrame({k: v.reshape(-1) for k, v in self.fields.items()}, index=index)
        return frame.dropna(how="all").reset_index()


def panel_from_frame(
    frame: pd.DataFrame,
    *,
    fields: tuple[str, ...] | None = None,
    adjust: bool = True,
    metadata: Mapping[str, str] | None = None,
) -> MarketPanel:
    """Pivot a *validated* long-format frame into a :class:`MarketPanel`.

    Args:
        frame: Output of :func:`~alphalab.data.validation.validate_market_frame`.
        fields: Fields to include; defaults to every feature field present.
        adjust: Apply ``adj_factor`` (when present) to prices and volume, as
            described in :mod:`alphalab.data.contract`.
        metadata: Provenance recorded on the panel.

    Returns:
        The panel; missing ``(timestamp, instrument)`` cells are NaN.
    """
    available = [f for f in FEATURE_FIELDS if f in frame.columns]
    selected = list(fields) if fields is not None else available
    unknown = [f for f in selected if f not in frame.columns]
    if unknown:
        raise DataValidationError(f"requested fields not in frame: {unknown}")
    work = frame.copy()
    if adjust and ADJ_FACTOR_COLUMN in work.columns:
        factor = work[ADJ_FACTOR_COLUMN]
        for column in (*PRICE_COLUMNS, VWAP_COLUMN):
            if column in work.columns:
                work[column] = work[column] * factor
        work[VOLUME_COLUMN] = work[VOLUME_COLUMN] / factor
    timestamps = np.sort(work[TIMESTAMP_COLUMN].unique()).astype("datetime64[ns]")
    instruments = tuple(sorted(work[INSTRUMENT_COLUMN].unique()))
    arrays = {}
    for name in selected:
        pivot = work.pivot(index=TIMESTAMP_COLUMN, columns=INSTRUMENT_COLUMN, values=name)
        pivot = pivot.reindex(index=pd.DatetimeIndex(timestamps), columns=list(instruments))
        arrays[name] = pivot.to_numpy(dtype=np.float64)
    return MarketPanel(
        timestamps=timestamps,
        instruments=instruments,
        fields=arrays,
        metadata=dict(metadata or {}),
    )
