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


def build_evaluator(
    spec: SyntheticSpec,
    *,
    horizon: int = 1,
    budget: int | None = None,
    complexity_penalty: float = 0.0,
):  # type: ignore[no-untyped-def]
    """Evaluator over a synthetic panel with a 60/20/20 purged split."""
    from alphalab.data import panel_forward_returns, split_by_fractions
    from alphalab.evaluation import EvaluationBudget, EvaluationSettings, FactorEvaluator

    panel = make_synthetic_panel(spec)
    split = split_by_fractions(panel.n_periods, 0.6, 0.2, horizon=horizon)
    return FactorEvaluator(
        panel,
        panel_forward_returns(panel, horizon),
        split,
        settings=EvaluationSettings(complexity_penalty=complexity_penalty),
        budget=EvaluationBudget(budget) if budget else None,
    )
