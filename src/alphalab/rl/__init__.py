"""Grammar-constrained RL for expression construction.

The environment and token grammar (``alphalab.rl.tokens``, ``alphalab.rl.env``)
depend only on NumPy. The neural policy and trainer live in
``alphalab.rl.policy`` and ``alphalab.rl.trainer`` and require the optional
``torch`` dependency (``pip install alphalab[rl]``); import them explicitly.
"""

from __future__ import annotations

from alphalab.rl.env import (
    ExpressionEnv,
    FitnessScorer,
    PoolGainScorer,
    ScoreOutcome,
    Scorer,
)
from alphalab.rl.settings import RLSettings
from alphalab.rl.tokens import (
    BuilderState,
    InvalidActionError,
    RPNBuilder,
    Token,
    TokenKind,
)

__all__ = [
    "BuilderState",
    "ExpressionEnv",
    "FitnessScorer",
    "InvalidActionError",
    "PoolGainScorer",
    "RLSettings",
    "RPNBuilder",
    "ScoreOutcome",
    "Scorer",
    "Token",
    "TokenKind",
]
