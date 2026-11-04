"""Sequential expression-building environment.

:class:`ExpressionEnv` exposes the RPN builder of :mod:`alphalab.rl.tokens`
through a Gymnasium-style ``reset`` / ``step`` interface without depending on
Gymnasium. Rewards are sparse: every intermediate step returns 0 and the
terminal END step returns the score of the finished expression.

Observation: ``int64`` array of length ``max_tokens`` with the emitted actions
shifted by one (``action + 1``) and zero padding; index 0 doubles as the
begin-of-sequence symbol for recurrent policies. The legal-action mask is
returned in ``info["action_mask"]`` after every reset and step.

Reward functions
----------------
A scorer maps a finished expression to a reward. :class:`FitnessScorer` uses
train fitness from a :class:`~alphalab.evaluation.FactorEvaluator` (charging
its budget), optionally minus a diversity penalty against the best factors
found so far. :class:`PoolGainScorer` rewards the improvement of a linearly
combined :class:`~alphalab.evaluation.FactorPool`, the AlphaGen formulation.
Both use only the train partition.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alphalab.evaluation.evaluator import INVALID_FITNESS, FactorEvaluator, FactorScore
from alphalab.evaluation.selection import DiversityPenalty, FactorPool
from alphalab.expressions.grammar import Grammar
from alphalab.expressions.nodes import Node
from alphalab.rl.tokens import BuilderState, RPNBuilder


@dataclass(frozen=True)
class ScoreOutcome:
    """Reward and diagnostics for one finished expression."""

    reward: float
    score: FactorScore | None = None
    charged: bool = False


class Scorer:
    """Base class of reward functions; subclasses implement :meth:`__call__`."""

    def __call__(self, node: Node) -> ScoreOutcome:  # pragma: no cover - interface
        raise NotImplementedError

    def state(self) -> dict[str, Any]:
        """Serializable internal state (for checkpoints)."""
        return {}

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore :meth:`state`."""


@dataclass
class FitnessScorer(Scorer):
    """Train fitness, optionally penalized by correlation with the top factors so far.

    Attributes:
        evaluator: Scoring gateway (its budget is charged for new expressions).
        diversity_weight: Weight of :class:`DiversityPenalty`; 0 disables it.
        reference_size: Number of best factors used as the diversity reference.
    """

    evaluator: FactorEvaluator
    diversity_weight: float = 0.0
    reference_size: int = 5
    _best: list[tuple[float, str]] = field(default_factory=list)

    def __call__(self, node: Node) -> ScoreOutcome:
        budget = self.evaluator.budget
        before = budget.used if budget else 0
        score = self.evaluator.score(node)
        charged = budget is not None and budget.used > before
        reward = score.fitness
        if score.valid and self.diversity_weight > 0:
            refs = [
                self.evaluator.factor_values(expr, "train")
                for _, expr in self._best
                if expr != score.expression
            ]
            values = self.evaluator.factor_values(node, "train")
            reward -= DiversityPenalty(self.diversity_weight).against(values, refs)
        if score.valid and score.expression not in {e for _, e in self._best}:
            self._best.append((score.fitness, score.expression))
            self._best.sort(key=lambda item: (-item[0], item[1]))
            del self._best[self.reference_size :]
        return ScoreOutcome(float(reward), score, charged)

    def state(self) -> dict[str, Any]:
        return {"best": [[f, e] for f, e in self._best]}

    def load_state(self, state: dict[str, Any]) -> None:
        self._best = [(float(f), str(e)) for f, e in state.get("best", [])]


@dataclass
class PoolGainScorer(Scorer):
    """Reward = gain in train rank IC of a ridge-combined factor pool.

    Invalid candidates receive :data:`INVALID_FITNESS`; valid candidates that
    do not improve the pool receive their (non-positive) gain.
    """

    evaluator: FactorEvaluator
    capacity: int = 10
    pool: FactorPool = field(init=False)

    def __post_init__(self) -> None:
        self.pool = FactorPool(self.capacity)

    def __call__(self, node: Node) -> ScoreOutcome:
        budget = self.evaluator.budget
        before = budget.used if budget else 0
        score = self.evaluator.score(node)
        charged = budget is not None and budget.used > before
        if not score.valid:
            return ScoreOutcome(INVALID_FITNESS, score, charged)
        gain = self.pool.try_add(
            score.expression,
            self.evaluator.factor_values(node, "train"),
            self.evaluator.target_values("train"),
        )
        return ScoreOutcome(float(gain), score, charged)

    def state(self) -> dict[str, Any]:
        return {"members": list(self.pool.expressions)}

    def load_state(self, state: dict[str, Any]) -> None:
        self.pool = FactorPool(self.capacity)
        target = self.evaluator.target_values("train")
        for expression in state.get("members", []):
            self.pool.try_add(expression, self.evaluator.factor_values(expression, "train"), target)


class ExpressionEnv:
    """Gymnasium-style environment that builds one expression per episode."""

    def __init__(
        self,
        grammar: Grammar,
        scorer: Scorer | Callable[[Node], ScoreOutcome],
        *,
        max_tokens: int = 20,
        require_operator: bool = True,
        invalid_reward: float = INVALID_FITNESS,
    ) -> None:
        self.builder = RPNBuilder(grammar, max_tokens=max_tokens, require_operator=require_operator)
        self.scorer = scorer
        self.invalid_reward = invalid_reward
        self.state = BuilderState()
        self._mask = self.builder.action_mask(self.state)

    @property
    def n_actions(self) -> int:
        """Size of the discrete action space."""
        return self.builder.n_actions

    @property
    def max_tokens(self) -> int:
        """Maximum episode length."""
        return self.builder.max_tokens

    def observation(self) -> np.ndarray:
        """Current token prefix, shifted by one and zero-padded."""
        obs = np.zeros(self.max_tokens, dtype=np.int64)
        tokens = self.state.tokens
        obs[: len(tokens)] = np.asarray(tokens, dtype=np.int64) + 1
        return obs

    def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode."""
        self.state = BuilderState()
        self._mask = self.builder.action_mask(self.state)
        return self.observation(), {"action_mask": self._mask.copy()}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply ``action``; returns ``(obs, reward, terminated, truncated, info)``.

        Raises:
            InvalidActionError: ``action`` is not legal (the mask was ignored).
            BudgetExhaustedError: The scorer's evaluator has no budget left;
                the episode is not completed.
        """
        self.state = self.builder.step(self.state, int(action))
        info: dict[str, Any] = {}
        if self.state.done:
            node = self.builder.expression(self.state)
            outcome = self.scorer(node)
            self._mask = np.zeros(self.n_actions, dtype=bool)
            info.update(
                action_mask=self._mask.copy(),
                expression=node.canonical,
                outcome=outcome,
                dead_end=False,
            )
            return self.observation(), outcome.reward, True, False, info
        self._mask = self.builder.action_mask(self.state)
        if not self._mask.any():
            # Dead end: no completion satisfies the grammar limits.
            info.update(action_mask=self._mask.copy(), dead_end=True)
            return self.observation(), self.invalid_reward, False, True, info
        info.update(action_mask=self._mask.copy(), dead_end=False)
        return self.observation(), 0.0, False, False, info
