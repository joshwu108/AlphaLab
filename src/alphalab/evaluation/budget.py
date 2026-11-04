"""Search-budget accounting shared by every search method.

Comparisons between random search, genetic programming and RL are only fair
when each method is charged the same way. The unit of cost is one **unique
candidate expression scored on the reward partition** (train). Re-scoring an
expression whose canonical key was already charged is free, because the
evaluator memoizes scores; this mirrors the fact that a real search would look
the result up rather than recompute it. Validation and test scoring are never
charged, and are only used outside the search loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from alphalab.errors import BudgetExhaustedError


@dataclass
class EvaluationBudget:
    """Mutable counter of charged evaluations.

    Attributes:
        max_evaluations: Total unique candidates that may be charged.
        used: Unique candidates charged so far.
        duplicate_requests: Requests for already-charged keys (free).
        invalid_candidates: Charged candidates that failed validity checks.
    """

    max_evaluations: int
    used: int = 0
    duplicate_requests: int = 0
    invalid_candidates: int = 0
    _charged: set[str] = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        if self.max_evaluations < 1:
            raise ValueError("max_evaluations must be at least 1")

    @property
    def remaining(self) -> int:
        """Evaluations still available."""
        return self.max_evaluations - self.used

    @property
    def exhausted(self) -> bool:
        """``True`` when no new candidate can be charged."""
        return self.used >= self.max_evaluations

    def is_charged(self, key: str) -> bool:
        """Whether ``key`` has already been paid for."""
        return key in self._charged

    def charge(self, key: str) -> bool:
        """Charge ``key`` once.

        Returns:
            ``True`` if this call consumed budget, ``False`` for a duplicate.

        Raises:
            BudgetExhaustedError: A new key was requested with no budget left.
        """
        if key in self._charged:
            self.duplicate_requests += 1
            return False
        if self.exhausted:
            raise BudgetExhaustedError(
                f"evaluation budget of {self.max_evaluations} unique candidates is exhausted"
            )
        self._charged.add(key)
        self.used += 1
        return True

    def to_dict(self) -> dict[str, int]:
        """Counters for manifests and checkpoints."""
        return {
            "max_evaluations": self.max_evaluations,
            "used": self.used,
            "duplicate_requests": self.duplicate_requests,
            "invalid_candidates": self.invalid_candidates,
        }

    def state(self) -> dict[str, object]:
        """Full state including charged keys, for checkpoint/resume."""
        return {**self.to_dict(), "charged": sorted(self._charged)}

    @classmethod
    def from_state(cls, state: dict[str, object]) -> EvaluationBudget:
        """Rebuild a budget saved by :meth:`state`."""
        charged = state.get("charged", [])
        assert isinstance(charged, list)
        budget = cls(
            max_evaluations=int(state["max_evaluations"]),  # type: ignore[call-overload]
            used=int(state["used"]),  # type: ignore[call-overload]
            duplicate_requests=int(state.get("duplicate_requests", 0)),  # type: ignore[call-overload]
            invalid_candidates=int(state.get("invalid_candidates", 0)),  # type: ignore[call-overload]
        )
        budget._charged = {str(k) for k in charged}
        if len(budget._charged) != budget.used:
            raise ValueError("budget state is inconsistent: charged keys != used")
        return budget
