"""Prediction targets.

Labels are the *only* place where AlphaLab intentionally reads the future. They
are produced from the panel separately from factor features and are never
exposed to the expression engine, so no expression can reference them.

Horizon semantics
-----------------
For prediction time ``t`` and horizon ``h`` the label is the simple forward
return ``close[t + h] / close[t] - 1``. It depends on prices at rows
``t .. t + h`` inclusive; its *label interval* is therefore ``[t, t + h]``.
The last ``h`` rows have no label (NaN). Partitioning code relies on this
interval to purge overlapping samples.
"""

from __future__ import annotations

import numpy as np

from alphalab.data.panel import MarketPanel


def forward_returns(close: np.ndarray, horizon: int) -> np.ndarray:
    """Return the ``horizon``-step forward simple return for every row.

    Args:
        close: ``(T, N)`` adjusted close prices.
        horizon: Number of sessions ahead; must be at least 1.

    Returns:
        ``(T, N)`` float array; rows ``T - horizon`` onward and any row whose
        start or end price is missing are NaN.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    close = np.asarray(close, dtype=np.float64)
    out = np.full_like(close, np.nan)
    if horizon < close.shape[0]:
        out[:-horizon] = close[horizon:] / close[:-horizon] - 1.0
    return out


def panel_forward_returns(panel: MarketPanel, horizon: int, field: str = "close") -> np.ndarray:
    """Forward returns computed from ``panel.field(field)``."""
    return forward_returns(panel.field(field), horizon)
