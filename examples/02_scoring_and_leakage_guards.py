"""Score factors on purged partitions and show the locked test partition.

Run: python examples/02_scoring_and_leakage_guards.py
"""

from __future__ import annotations

from alphalab.data import (
    SyntheticSpec,
    make_synthetic_panel,
    panel_forward_returns,
    split_by_fractions,
)
from alphalab.errors import PartitionAccessError
from alphalab.evaluation import EvaluationBudget, FactorEvaluator


def main() -> None:
    spec = SyntheticSpec(n_periods=300, n_instruments=30, seed=2, signal_strength=0.4)
    panel = make_synthetic_panel(spec)
    horizon = 5
    split = split_by_fractions(panel.n_periods, 0.6, 0.2, horizon=horizon, embargo=2)
    print(f"split: {split.to_dict()}")
    evaluator = FactorEvaluator(
        panel, panel_forward_returns(panel, horizon), split, budget=EvaluationBudget(10)
    )
    for text in (spec.planted_expression, "ts_std($close, 20)"):
        train, valid = evaluator.score(text, "train"), evaluator.score(text, "valid")
        print(
            f"{text:<32} train rank IC {train.rank_ic_mean:+.3f}  "
            f"valid rank IC {valid.rank_ic_mean:+.3f}  turnover {valid.turnover:.2f}"
        )
    try:
        evaluator.score(spec.planted_expression, "test")
    except PartitionAccessError as exc:
        print(f"test partition locked: {exc}")
    print(f"budget: {evaluator.budget.to_dict() if evaluator.budget else None}")
    print("Note: these are synthetic smoke numbers, not research results.")


if __name__ == "__main__":
    main()
