"""Export of factor scores to JSON, CSV or Parquet.

JSON is the canonical machine-readable format. NaN, which JSON cannot
represent, is written as ``null``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from alphalab.evaluation.evaluator import FactorScore


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item") and callable(value.item):  # numpy scalar
        return _json_safe(value.item())
    return value


def to_json_text(payload: Any) -> str:
    """Serialize ``payload`` deterministically (sorted keys, NaN as null)."""
    return json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False)


def write_json(payload: Any, path: str | Path) -> Path:
    """Write ``payload`` as JSON to ``path``, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json_text(payload) + "\n")
    return path


def scores_frame(scores: Iterable[FactorScore]) -> pd.DataFrame:
    """One row per score."""
    return pd.DataFrame([s.to_dict() for s in scores])


def export_scores(scores: Iterable[FactorScore], path: str | Path) -> Path:
    """Write scores to ``.json``, ``.csv`` or ``.parquet`` according to the suffix."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(scores)
    suffix = path.suffix.lower()
    if suffix == ".json":
        return write_json([s.to_dict() for s in rows], path)
    frame = scores_frame(rows)
    if suffix == ".csv":
        frame.to_csv(path, index=False)
    elif suffix in (".parquet", ".pq"):
        frame.to_parquet(path, index=False)
    else:
        raise ValueError(f"unsupported export format {suffix!r}; use .json, .csv or .parquet")
    return path
