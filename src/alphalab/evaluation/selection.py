"""Diversity-aware factor selection and linear factor combination.

Searching for single factors with the highest IC tends to return many
near-duplicates of the same signal. Two complementary tools counter that:

* :func:`select_diverse` greedily keeps the best candidates whose absolute
  cross-sectional correlation with every already-kept factor stays below a
  threshold.
* :class:`FactorPool` maintains a bounded set of factors combined by ridge
  regression on z-scored values, in the spirit of the AlphaGen formulation
  (Yu et al., 2023): a candidate is valuable if it improves the *combination*,
  not merely if it is strong on its own. Weights are fitted on the train
  partition only.

:class:`DiversityPenalty` turns redundancy into a fitness penalty for search
methods (GP and RL) that should be steered away from known factors.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from alphalab.evaluation import metrics
from alphalab.expressions.operators import cross_sectional_zscore

DEFAULT_RIDGE = 1e-3


@dataclass(frozen=True)
class Candidate:
    """A scored factor and its train-partition values."""

    expression: str
    fitness: float
    values: np.ndarray


def select_diverse(
    candidates: Sequence[Candidate],
    *,
    max_factors: int,
    max_abs_correlation: float = 0.7,
) -> list[Candidate]:
    """Greedy selection by fitness subject to a pairwise-correlation cap.

    Args:
        candidates: Scored candidates; order does not matter.
        max_factors: Maximum number of factors returned.
        max_abs_correlation: A candidate is skipped if its mean absolute
            cross-sectional correlation with any kept factor reaches this value.

    Returns:
        Kept candidates in descending fitness order. Ties are broken by
        expression string so the result is deterministic.
    """
    if max_factors < 1:
        raise ValueError("max_factors must be at least 1")
    ordered = sorted(candidates, key=lambda c: (-c.fitness, c.expression))
    kept: list[Candidate] = []
    for candidate in ordered:
        if not math.isfinite(candidate.fitness):
            continue
        redundant = False
        for other in kept:
            corr = metrics.factor_correlation(candidate.values, other.values)
            if math.isnan(corr) or abs(corr) >= max_abs_correlation:
                redundant = True
                break
        if not redundant:
            kept.append(candidate)
        if len(kept) == max_factors:
            break
    return kept


@dataclass
class DiversityPenalty:
    """Fitness penalty proportional to the strongest correlation with a reference set.

    ``penalty = weight * max_j |corr(candidate, reference_j)|``; zero for an
    empty reference set. ``reference_provider`` returns the current reference
    factors (for example GP elites or the RL factor pool).
    """

    weight: float
    reference_provider: Callable[[], Sequence[np.ndarray]] = field(default=lambda: ())

    def __call__(self, values: np.ndarray) -> float:
        """Penalty for a candidate with the given train values."""
        return self.against(values, self.reference_provider())

    def against(self, values: np.ndarray, references: Sequence[np.ndarray]) -> float:
        """Penalty for ``values`` relative to an explicit reference set."""
        if self.weight == 0:
            return 0.0
        worst = 0.0
        for reference in references:
            corr = metrics.factor_correlation(values, reference)
            if math.isfinite(corr):
                worst = max(worst, abs(corr))
        return self.weight * worst


def _design(values: Sequence[np.ndarray]) -> np.ndarray:
    """Stack z-scored factors into ``(T, N, K)`` with NaN replaced by 0 (neutral)."""
    stacked = np.stack(
        [cross_sectional_zscore(np.asarray(v, dtype=np.float64)) for v in values], axis=-1
    )
    return np.where(np.isfinite(stacked), stacked, 0.0)


def fit_linear_combination(
    values: Sequence[np.ndarray], target: np.ndarray, *, ridge: float = DEFAULT_RIDGE
) -> np.ndarray:
    """Ridge weights mapping z-scored factors to the z-scored target.

    The regression pools every ``(session, instrument)`` observation where the
    target is finite. Z-scoring per session removes market-wide moves, so the
    weights explain cross-sectional differences, which is what IC measures.
    """
    if not values:
        raise ValueError("at least one factor is required")
    design = _design(values)
    y = cross_sectional_zscore(np.asarray(target, dtype=np.float64))
    mask = np.isfinite(y)
    x = design[mask]
    yv = y[mask]
    k = x.shape[1]
    gram = x.T @ x + ridge * max(len(yv), 1) * np.eye(k)
    weights: np.ndarray = np.linalg.solve(gram, x.T @ yv)
    return weights


def combine(values: Sequence[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Weighted sum of z-scored factors; sessions with no finite inputs are NaN."""
    design = _design(values)
    combined = design @ np.asarray(weights, dtype=np.float64)
    any_finite = np.any(np.stack([np.isfinite(v) for v in values], axis=-1), axis=-1)
    return np.where(any_finite, combined, np.nan)


@dataclass
class FactorPool:
    """Bounded set of factors combined linearly; fitted on train data only.

    Attributes:
        capacity: Maximum number of factors.
        ridge: Ridge strength for :func:`fit_linear_combination`.
        expressions: Members in insertion order.
        weights: Current combination weights (aligned with ``expressions``).
        score: Mean rank IC of the combination on the train target.
    """

    capacity: int
    ridge: float = DEFAULT_RIDGE
    expressions: list[str] = field(default_factory=list)
    weights: np.ndarray = field(default_factory=lambda: np.zeros(0))
    score: float = 0.0
    _values: list[np.ndarray] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError("capacity must be at least 1")

    @property
    def values(self) -> list[np.ndarray]:
        """Train values of the members."""
        return list(self._values)

    def _evaluate(self, values: list[np.ndarray], target: np.ndarray) -> tuple[np.ndarray, float]:
        weights = fit_linear_combination(values, target, ridge=self.ridge)
        ric = metrics.rank_information_coefficient(combine(values, weights), target)
        summary = metrics.summarize_series(ric)
        return weights, summary.mean if math.isfinite(summary.mean) else 0.0

    def try_add(self, expression: str, values: np.ndarray, target: np.ndarray) -> float:
        """Add a factor if it raises the combination score; return the score change.

        When the pool is full, the member with the smallest absolute weight in
        the enlarged fit is dropped before comparing. Members are unchanged and
        the return value is ``<= 0`` when the candidate does not help.
        """
        if expression in self.expressions:
            return 0.0
        trial_expr = [*self.expressions, expression]
        trial_values = [*self._values, values]
        weights, score = self._evaluate(trial_values, target)
        if len(trial_expr) > self.capacity:
            drop = int(np.argmin(np.abs(weights)))
            del trial_expr[drop], trial_values[drop]
            weights, score = self._evaluate(trial_values, target)
        improvement = score - self.score
        if improvement > 0:
            self.expressions, self._values = trial_expr, trial_values
            self.weights, self.score = weights, score
        return improvement

    def to_dict(self) -> dict[str, object]:
        """Members, weights and score for export."""
        return {
            "capacity": self.capacity,
            "ridge": self.ridge,
            "score": self.score,
            "members": [
                {"expression": e, "weight": float(w)}
                for e, w in zip(self.expressions, self.weights, strict=True)
            ],
        }
