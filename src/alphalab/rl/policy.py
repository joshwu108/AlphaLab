"""Recurrent actor-critic policy over RPN token sequences (requires torch).

Architecture::

    tokens (BOS = 0, action a -> a + 1)
        -> Embedding(n_actions + 1, embed_dim)
        -> GRU(embed_dim, hidden_dim, num_layers)
        -> actor:  Linear(hidden_dim, n_actions)   (logits, masked by the grammar)
        -> critic: Linear(hidden_dim, 1)            (state-value baseline)

The same weights serve two code paths that must agree exactly: step-by-step
sampling with a carried hidden state (:meth:`RecurrentPolicy.step`) and
teacher-forced re-evaluation of whole episodes for the gradient
(:meth:`RecurrentPolicy.forward`). Tests assert the two produce identical
log-probabilities.
"""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised only without torch
    raise ImportError(
        "alphalab.rl.policy requires torch; install it with `pip install alphalab[rl]`"
    ) from exc


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Replace logits of illegal actions by the most negative finite value.

    A finite fill (rather than ``-inf``) keeps ``softmax``, ``log_softmax``
    and the entropy ``sum p log p`` free of NaN: illegal actions get exactly
    zero probability and contribute ``0 * finite = 0`` to the entropy.
    """
    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


class RecurrentPolicy(nn.Module):
    """GRU actor-critic.

    Args:
        n_actions: Vocabulary size of the environment.
        embed_dim: Token embedding width.
        hidden_dim: GRU hidden width.
        num_layers: Stacked GRU layers.
    """

    def __init__(
        self, n_actions: int, *, embed_dim: int = 32, hidden_dim: int = 64, num_layers: int = 1
    ) -> None:
        super().__init__()
        if n_actions < 2:
            raise ValueError("n_actions must be at least 2")
        self.n_actions = n_actions
        self.embedding = nn.Embedding(n_actions + 1, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.actor = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forced pass.

        Args:
            inputs: ``(B, L)`` long tensor; position ``t`` holds the token
                emitted at step ``t - 1`` (shifted by one) and position 0 is BOS.

        Returns:
            ``logits`` of shape ``(B, L, n_actions)`` and ``values`` ``(B, L)``.
        """
        hidden, _ = self.gru(self.embedding(inputs))
        return self.actor(hidden), self.critic(hidden).squeeze(-1)

    def step(
        self, previous: torch.Tensor, hidden: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One recurrent step for sampling.

        Args:
            previous: ``(B,)`` long tensor of shifted previous tokens (0 = BOS).
            hidden: GRU state from the previous step, or ``None`` at the start.

        Returns:
            ``logits (B, n_actions)``, ``values (B,)`` and the new hidden state.
        """
        out, hidden = self.gru(self.embedding(previous.unsqueeze(1)), hidden)
        out = out.squeeze(1)
        return self.actor(out), self.critic(out).squeeze(-1), hidden
