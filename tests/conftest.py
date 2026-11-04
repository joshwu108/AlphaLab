"""Shared deterministic fixtures."""

from __future__ import annotations

import pytest

from alphalab.data import MarketPanel, SyntheticSpec, make_synthetic_panel


@pytest.fixture(scope="session")
def small_panel() -> MarketPanel:
    """A small deterministic panel with a planted reversal signal."""
    return make_synthetic_panel(SyntheticSpec(n_periods=160, n_instruments=16, seed=7))


@pytest.fixture(scope="session")
def strong_spec() -> SyntheticSpec:
    """Spec whose planted signal is strong enough for search tests to find."""
    return SyntheticSpec(n_periods=320, n_instruments=30, seed=3, signal_strength=0.6)
