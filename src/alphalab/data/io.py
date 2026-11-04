"""CSV and Parquet ingestion of long-format market data.

Readers only parse files and delegate every rule to
:func:`alphalab.data.validation.validate_market_frame`. Parquet support needs
the optional ``pyarrow`` dependency (``pip install alphalab[parquet]``).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from alphalab.data.contract import DataContract
from alphalab.data.validation import ValidationReport, validate_market_frame
from alphalab.errors import DataValidationError

CSV_SUFFIXES = (".csv", ".csv.gz")
PARQUET_SUFFIXES = (".parquet", ".pq")


def _require_pyarrow() -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without pyarrow
        raise ImportError(
            "Parquet support requires pyarrow; install it with `pip install alphalab[parquet]`"
        ) from exc


def read_market_frame(path: str | Path) -> pd.DataFrame:
    """Read a CSV or Parquet file into a DataFrame without validating it.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        DataValidationError: The file suffix is not supported.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"market data file not found: {path}")
    name = path.name.lower()
    if name.endswith(CSV_SUFFIXES):
        return pd.read_csv(path)
    if name.endswith(PARQUET_SUFFIXES):
        _require_pyarrow()
        return pd.read_parquet(path)
    raise DataValidationError(
        f"unsupported file type {path.suffix!r}; expected one of {CSV_SUFFIXES + PARQUET_SUFFIXES}"
    )


def load_market_data(
    path: str | Path,
    contract: DataContract | None = None,
    column_map: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Read, rename and validate a market-data file.

    Args:
        path: CSV (optionally gzip-compressed) or Parquet file.
        contract: Validation rules.
        column_map: Optional ``{vendor_name: contract_name}`` renaming applied
            before validation, for example ``{"date": "timestamp", "ticker":
            "instrument"}``.

    Returns:
        The validated frame and its :class:`ValidationReport`.
    """
    frame = read_market_frame(path)
    if column_map:
        frame = frame.rename(columns=dict(column_map))
    return validate_market_frame(frame, contract)


def write_market_frame(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write a long-format frame to CSV or Parquet, chosen by suffix."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = path.name.lower()
    if name.endswith(CSV_SUFFIXES):
        frame.to_csv(path, index=False)
    elif name.endswith(PARQUET_SUFFIXES):
        _require_pyarrow()
        frame.to_parquet(path, index=False)
    else:
        raise DataValidationError(f"unsupported output file type {path.suffix!r}")
    return path
