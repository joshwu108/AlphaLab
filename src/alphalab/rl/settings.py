"""Hyperparameters of the RL search, importable without torch.

Kept separate from :mod:`alphalab.rl.trainer` so that experiment
configurations can be fully validated on machines without torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RewardMode = Literal["fitness", "pool_gain"]


@dataclass(frozen=True)
class RLSettings:
    """Hyperparameters of the policy-gradient search.

    Attributes:
        max_tokens: Maximum RPN tokens per expression.
        embed_dim: Token embedding width.
        hidden_dim: GRU hidden width.
        num_layers: GRU layers.
        learning_rate: Adam learning rate.
        episodes_per_update: Episodes collected per gradient step.
        entropy_coef: Entropy bonus weight.
        value_coef: Value-loss weight.
        grad_clip: Global gradient-norm clip.
        reward_mode: ``"fitness"`` or ``"pool_gain"`` (see :mod:`alphalab.rl.env`).
        diversity_weight: Diversity penalty for ``"fitness"`` rewards.
        pool_capacity: Pool size for ``"pool_gain"`` rewards.
        max_updates: Optional limit on gradient steps; the budget normally stops training.
        stall_updates: Stop after this many consecutive updates without a new
            (budget-charged) expression, which signals policy collapse.
        checkpoint_every: Updates between checkpoints (when a directory is set).
        seed: Seed for initialization and sampling.
        device: Torch device string; tests use ``"cpu"``.
        top_k: Number of best records kept in the result.
    """

    max_tokens: int = 20
    embed_dim: int = 32
    hidden_dim: int = 64
    num_layers: int = 1
    learning_rate: float = 1e-3
    episodes_per_update: int = 32
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    grad_clip: float = 1.0
    reward_mode: RewardMode = "fitness"
    diversity_weight: float = 0.0
    pool_capacity: int = 10
    max_updates: int | None = None
    stall_updates: int = 50
    checkpoint_every: int = 10
    seed: int = 0
    device: str = "cpu"
    top_k: int = 20

    def __post_init__(self) -> None:
        if self.episodes_per_update < 1 or self.learning_rate <= 0:
            raise ValueError("episodes_per_update must be >= 1 and learning_rate > 0")
        if self.reward_mode not in ("fitness", "pool_gain"):
            raise ValueError(f"unknown reward_mode {self.reward_mode!r}")
        if self.stall_updates < 1 or self.checkpoint_every < 1:
            raise ValueError("stall_updates and checkpoint_every must be >= 1")
        if self.max_tokens < 3 or min(self.embed_dim, self.hidden_dim, self.num_layers) < 1:
            raise ValueError("max_tokens must be >= 3 and network sizes >= 1")
        if min(self.entropy_coef, self.value_coef, self.diversity_weight) < 0:
            raise ValueError("loss coefficients and diversity_weight must be non-negative")
        if self.grad_clip <= 0 or self.pool_capacity < 1 or self.top_k < 1:
            raise ValueError("grad_clip must be > 0; pool_capacity and top_k >= 1")
        if self.max_updates is not None and self.max_updates < 1:
            raise ValueError("max_updates must be >= 1 when given")
