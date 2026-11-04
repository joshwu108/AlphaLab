"""Random search baseline.

Samples valid expressions from the grammar, skips candidates already seen
(by canonical key, so ``add($a, $b)`` and ``add($b, $a)`` are one candidate),
and scores each new one on the train partition until the budget is spent.
It is the reference point every learned method must beat at equal budget.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alphalab.errors import BudgetExhaustedError, ExpressionLimitError
from alphalab.evaluation.evaluator import FactorEvaluator
from alphalab.expressions.grammar import Grammar, sample_expression
from alphalab.search.base import SearchHistory, SearchResult

# Consecutive duplicate samples tolerated before concluding the grammar's
# reachable space is exhausted.
MAX_CONSECUTIVE_DUPLICATES = 10_000


@dataclass
class RandomSearch:
    """Uniform-structure random sampling with deduplication.

    Attributes:
        grammar: Vocabulary and limits.
        seed: Seed for the sampler.
        top_k: Number of best records kept in the result.
    """

    grammar: Grammar
    seed: int = 0
    top_k: int = 20
    method: str = "random"

    def run(self, evaluator: FactorEvaluator) -> SearchResult:
        """Sample and score until the budget is exhausted."""
        if evaluator.budget is None:
            raise ValueError("random search requires an evaluator with a budget")
        budget = evaluator.budget
        rng = np.random.default_rng(self.seed)
        history = SearchHistory()
        seen: set[str] = set()
        duplicates = 0
        stopped = "budget exhausted"
        while not budget.exhausted:
            try:
                node = sample_expression(self.grammar, rng)
            except ExpressionLimitError:
                stopped = "grammar could not produce a valid sample"
                break
            if node.key in seen:
                duplicates += 1
                if duplicates >= MAX_CONSECUTIVE_DUPLICATES:
                    stopped = "search space exhausted (only duplicates sampled)"
                    break
                continue
            duplicates = 0
            seen.add(node.key)
            try:
                score = evaluator.score(node)
            except BudgetExhaustedError:
                break
            history.add(score, source="random", iteration=len(history.records))
        return SearchResult(
            method=self.method,
            history=history,
            best=history.top(self.top_k),
            budget=budget.to_dict(),
            stopped_reason=stopped,
            extra={"seed": self.seed},
        )
