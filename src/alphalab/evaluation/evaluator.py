"""Partition-aware factor scoring with a locked test partition.

:class:`FactorEvaluator` is the single gateway through which search algorithms
observe data quality. It

* evaluates expressions on the full panel (causally, so validation rows may
  use training-period history) and slices the rows of one partition;
* computes IC, rank IC, turnover, stability and coverage on those rows;
* charges the shared :class:`~alphalab.evaluation.budget.EvaluationBudget`
  for train-partition scores, which are the search reward;
* **refuses test-partition access** until :meth:`unlock_test` is called, and
  records every partition access so manifests can prove the test set was
  untouched during search and selection.

Fitness
-------
``fitness = |primary metric| - complexity_penalty * size`` on the train
partition. The absolute value lets search find factors that predict with
either sign; the sign learned on train is stored as ``direction`` and applied,
unchanged, when the factor is scored on validation or test data.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from alphalab.data.panel import MarketPanel
from alphalab.data.splits import PARTITION_NAMES, ChronologicalSplit
from alphalab.errors import PartitionAccessError
from alphalab.evaluation import metrics
from alphalab.evaluation.budget import EvaluationBudget
from alphalab.expressions.cache import ExpressionCache
from alphalab.expressions.evaluator import evaluate
from alphalab.expressions.grammar import Grammar
from alphalab.expressions.nodes import Node
from alphalab.expressions.parser import parse

PrimaryMetric = Literal["rank_ic", "ic"]
REWARD_PARTITION = "train"
# Fitness assigned to candidates that cannot be scored (too little coverage,
# zero cross-sectional variance). Every scoreable candidate has fitness above
# -complexity_penalty * max_size, which is far above this value for sane
# penalties.
INVALID_FITNESS = -1.0


@dataclass(frozen=True)
class EvaluationSettings:
    """How candidates are scored.

    Attributes:
        metric: Primary metric for fitness: ``"rank_ic"`` (default, robust to
            outliers) or ``"ic"``.
        complexity_penalty: Fitness cost per expression node.
        min_coverage: Minimum fraction of finite factor values on the
            partition for the candidate to count as valid.
        min_cross_section: Minimum jointly valid instruments per session.
        stability_blocks: Blocks used by the sign-consistency diagnostic.
    """

    metric: PrimaryMetric = "rank_ic"
    complexity_penalty: float = 0.0
    min_coverage: float = 0.5
    min_cross_section: int = metrics.DEFAULT_MIN_CROSS_SECTION
    stability_blocks: int = 4

    def __post_init__(self) -> None:
        if self.metric not in ("rank_ic", "ic"):
            raise ValueError(f"unknown metric {self.metric!r}")
        if self.complexity_penalty < 0 or not 0 <= self.min_coverage <= 1:
            raise ValueError("complexity_penalty must be >= 0 and min_coverage in [0, 1]")


@dataclass(frozen=True)
class FactorScore:
    """All metrics for one expression on one partition."""

    expression: str
    key: str
    partition: str
    valid: bool
    reason: str | None
    fitness: float
    direction: int
    ic_mean: float
    ic_std: float
    icir: float
    rank_ic_mean: float
    rank_ic_std: float
    rank_icir: float
    rank_ic_t_stat: float
    rank_ic_hit_rate: float
    sign_consistency: float
    turnover: float
    rank_autocorrelation: float
    coverage: float
    n_periods: int
    size: int
    depth: int
    lookback: int
    metric: str

    def to_dict(self) -> dict[str, object]:
        """Plain dictionary for JSON/CSV export."""
        return asdict(self)

    @property
    def primary(self) -> float:
        """Signed value of the primary metric on this partition (rank IC or IC)."""
        return self.rank_ic_mean if self.metric == "rank_ic" else self.ic_mean


@dataclass(frozen=True)
class AccessRecord:
    """One partition read, kept for the run manifest."""

    partition: str
    key: str
    charged: bool


class FactorEvaluator:
    """Scores expressions on chronological partitions of one panel."""

    def __init__(
        self,
        panel: MarketPanel,
        target: np.ndarray,
        split: ChronologicalSplit,
        *,
        settings: EvaluationSettings | None = None,
        budget: EvaluationBudget | None = None,
        grammar: Grammar | None = None,
        cache: ExpressionCache | None = None,
    ) -> None:
        """Create an evaluator.

        Args:
            panel: Feature data.
            target: ``(T, N)`` forward returns aligned to ``panel``.
            split: Chronological partitions whose ``horizon`` matches ``target``.
            settings: Scoring rules.
            budget: Shared budget charged for train scores; ``None`` means unlimited.
            grammar: If given, every candidate is checked against it before scoring.
            cache: Shared sub-expression cache; one is created when omitted.
        """
        if target.shape != panel.shape:
            raise ValueError(f"target shape {target.shape} does not match panel {panel.shape}")
        if split.n_periods != panel.n_periods:
            raise ValueError("split was built for a different number of periods")
        self.panel = panel
        self.target = np.array(target, dtype=np.float64)
        self.target.flags.writeable = False
        self.split = split
        self.settings = settings or EvaluationSettings()
        self.budget = budget
        self.grammar = grammar
        self.cache = cache if cache is not None else ExpressionCache()
        self._test_unlocked = False
        self._test_unlock_reason: str | None = None
        self._scores: dict[tuple[str, str], FactorScore] = {}
        self._directions: dict[str, int] = {}
        self.access_log: list[AccessRecord] = []

    @property
    def test_unlocked(self) -> bool:
        """Whether the test partition may be read."""
        return self._test_unlocked

    def unlock_test(self, reason: str) -> None:
        """Permit test-partition scoring. Call only after search and selection end."""
        if not reason.strip():
            raise ValueError("a reason is required to unlock the test partition")
        self._test_unlocked = True
        self._test_unlock_reason = reason

    def _check_access(self, partition: str) -> None:
        if partition not in PARTITION_NAMES:
            raise KeyError(f"unknown partition {partition!r}")
        if partition == "test" and not self._test_unlocked:
            raise PartitionAccessError(
                "the test partition is locked; it may only be scored after search and "
                "selection have finished (FactorEvaluator.unlock_test)"
            )

    def factor_values(self, expression: Node | str, partition: str) -> np.ndarray:
        """Factor values restricted to the rows of ``partition`` (read-only view)."""
        self._check_access(partition)
        node = parse(expression) if isinstance(expression, str) else expression
        rows = self.split.partition(partition).as_slice()
        return evaluate(node, self.panel, self.cache)[rows]

    def target_values(self, partition: str) -> np.ndarray:
        """Forward returns for the rows of ``partition``."""
        self._check_access(partition)
        return self.target[self.split.partition(partition).as_slice()]

    def score(self, expression: Node | str, partition: str = REWARD_PARTITION) -> FactorScore:
        """Score ``expression`` on ``partition``, memoized by canonical key.

        Train scores are charged to the budget (unique keys only). Validation
        and test scores reuse the sign learned on train; if no train score
        exists yet it is computed without charging, so validation and test
        scoring never consume search budget.

        Raises:
            PartitionAccessError: ``partition`` is ``"test"`` and still locked.
            BudgetExhaustedError: A new train score is needed and the budget is spent.
            ExpressionTypeError, ExpressionLimitError: The grammar rejects the candidate.
        """
        self._check_access(partition)
        node = parse(expression) if isinstance(expression, str) else expression
        memo_key = (node.key, partition)
        cached = self._scores.get(memo_key)
        if cached is not None:
            if (
                partition == REWARD_PARTITION
                and self.budget is not None
                and self.budget.charge(node.key)
                and not cached.valid
            ):
                self.budget.invalid_candidates += 1
            return cached
        if self.grammar is not None:
            self.grammar.check(node)
        charged = False
        if partition == REWARD_PARTITION and self.budget is not None:
            charged = self.budget.charge(node.key)
        elif partition != REWARD_PARTITION and node.key not in self._directions:
            # Orienting a validation/test score needs the train sign. Computing
            # it here is deliberately not charged: only reward-partition
            # requests from a search loop consume budget. A later train
            # request for this key is charged through the memo path above.
            train = self._compute(node, REWARD_PARTITION)
            self._directions[node.key] = train.direction
            self._scores[(node.key, REWARD_PARTITION)] = train
        result = self._compute(node, partition)
        if partition == REWARD_PARTITION:
            self._directions[node.key] = result.direction
            if not result.valid and charged and self.budget is not None:
                self.budget.invalid_candidates += 1
        self._scores[memo_key] = result
        self.access_log.append(AccessRecord(partition, node.key, charged))
        return result

    def _compute(self, node: Node, partition: str) -> FactorScore:
        settings = self.settings
        factor = self.factor_values(node, partition)
        target = self.target_values(partition)
        cover = metrics.coverage(factor)
        ic = metrics.information_coefficient(factor, target, min_obs=settings.min_cross_section)
        ric = metrics.rank_information_coefficient(
            factor, target, min_obs=settings.min_cross_section
        )
        ic_sum, ric_sum = metrics.summarize_series(ic), metrics.summarize_series(ric)
        primary = ric_sum.mean if settings.metric == "rank_ic" else ic_sum.mean
        reason: str | None = None
        if cover < settings.min_coverage:
            reason = f"coverage {cover:.3f} below {settings.min_coverage}"
        elif math.isnan(primary):
            reason = "primary metric undefined (no valid cross-sections)"
        valid = reason is None
        if partition == REWARD_PARTITION:
            direction = 1 if (valid and primary >= 0) or not valid else -1
            fitness = (
                abs(primary) - settings.complexity_penalty * node.size if valid else INVALID_FITNESS
            )
        else:
            direction = self._directions.get(node.key, 1)
            fitness = direction * primary - settings.complexity_penalty * node.size
            fitness = fitness if valid else INVALID_FITNESS
        return FactorScore(
            expression=node.canonical,
            key=node.key,
            partition=partition,
            valid=valid,
            reason=reason,
            fitness=float(fitness),
            direction=direction,
            ic_mean=ic_sum.mean,
            ic_std=ic_sum.std,
            icir=ic_sum.ir,
            rank_ic_mean=ric_sum.mean,
            rank_ic_std=ric_sum.std,
            rank_icir=ric_sum.ir,
            rank_ic_t_stat=ric_sum.t_stat,
            rank_ic_hit_rate=ric_sum.hit_rate,
            sign_consistency=metrics.block_sign_consistency(ric, settings.stability_blocks),
            turnover=metrics.turnover(factor),
            rank_autocorrelation=metrics.rank_autocorrelation(factor),
            coverage=cover,
            n_periods=int(factor.shape[0]),
            size=node.size,
            depth=node.depth,
            lookback=node.lookback,
            metric=settings.metric,
        )

    def access_summary(self) -> dict[str, object]:
        """Counts of partition reads and the test-unlock status, for manifests."""
        counts = {name: 0 for name in PARTITION_NAMES}
        for record in self.access_log:
            counts[record.partition] += 1
        return {
            "scores_by_partition": counts,
            "test_unlocked": self._test_unlocked,
            "test_unlock_reason": self._test_unlock_reason,
        }
