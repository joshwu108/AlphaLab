"""Policy-gradient training loop for expression construction (requires torch).

Algorithm: episodic REINFORCE with a learned value baseline (an actor-critic
without bootstrapping). With sparse terminal rewards and discount 1, every
step of an episode has return ``R``; the loss on a batch of episodes is::

    L = - mean_t[(R_norm - V(s_t)).detach() * log pi(a_t | s_t)]
        + value_coef * mean_t[(V(s_t) - R_norm)^2]
        - entropy_coef * mean_t[H(pi(. | s_t))]

where ``R_norm`` is the reward standardized by running statistics, ``pi`` is
the grammar-masked policy and means run over real (non-padding) steps.
REINFORCE was chosen over PPO for the first version because episodes are short
and cheap relative to reward evaluation, it has no clipping or epoch
hyperparameters, and it is easy to verify; the batch structure leaves room for
a PPO update later. See ``planning/decisions.md``.

Determinism: the model is initialized after ``torch.manual_seed(seed)`` and
actions are sampled from a dedicated ``torch.Generator``. Checkpoints store the
model, optimizer, generator, reward statistics, scorer state, history and
budget, so resuming reproduces an uninterrupted CPU run exactly.

This module is complete and tested at toy scale on CPU. **No production
training has been run**; see ``planning/GPU_TRAINING_HANDOFF.md``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover - exercised only without torch
    raise ImportError(
        "alphalab.rl.trainer requires torch; install it with `pip install alphalab[rl]`"
    ) from exc

from alphalab.errors import BudgetExhaustedError, CheckpointError
from alphalab.evaluation.budget import EvaluationBudget
from alphalab.evaluation.evaluator import FactorEvaluator
from alphalab.evaluation.export import to_json_text
from alphalab.expressions.grammar import Grammar
from alphalab.rl.env import ExpressionEnv, FitnessScorer, PoolGainScorer, ScoreOutcome, Scorer
from alphalab.rl.policy import RecurrentPolicy, masked_logits
from alphalab.rl.settings import RLSettings
from alphalab.search.base import SearchHistory, SearchResult

__all__ = ["Episode", "PolicyGradientTrainer", "RLSettings", "RunningMeanStd", "UpdateMetrics"]

CHECKPOINT_FORMAT = 1
CHECKPOINT_NAME = "rl_checkpoint.pt"


@dataclass
class RunningMeanStd:
    """Running mean and variance (Chan et al. parallel update) for reward normalization."""

    count: float = 0.0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, values: np.ndarray) -> None:
        """Fold a batch of values into the statistics."""
        n = float(values.size)
        if n == 0:
            return
        batch_mean = float(values.mean())
        batch_m2 = float(((values - batch_mean) ** 2).sum())
        delta = batch_mean - self.mean
        total = self.count + n
        self.mean += delta * n / total
        self.m2 += batch_m2 + delta * delta * self.count * n / total
        self.count = total

    @property
    def std(self) -> float:
        """Population standard deviation (1 before any data)."""
        return math.sqrt(self.m2 / self.count) if self.count > 1 else 1.0

    def normalize(self, values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Standardize ``values`` with the current statistics."""
        return (values - self.mean) / (self.std + eps)


@dataclass
class Episode:
    """One rollout."""

    actions: list[int]
    masks: list[np.ndarray]
    reward: float
    expression: str | None
    dead_end: bool
    outcome: ScoreOutcome | None = None


@dataclass
class UpdateMetrics:
    """Diagnostics of one gradient step."""

    update: int
    episodes: int
    mean_reward: float
    max_reward: float
    mean_length: float
    dead_ends: int
    new_expressions: int
    policy_loss: float
    value_loss: float
    entropy: float
    budget_used: int


@dataclass
class _Progress:
    update: int = 0
    stall: int = 0
    metrics: list[UpdateMetrics] = field(default_factory=list)


class PolicyGradientTrainer:
    """REINFORCE-with-baseline search over the RPN expression grammar."""

    method = "rl"

    def __init__(
        self,
        grammar: Grammar,
        settings: RLSettings | None = None,
        *,
        checkpoint_dir: str | Path | None = None,
        metrics_path: str | Path | None = None,
        scorer_factory: Callable[[FactorEvaluator], Scorer] | None = None,
    ) -> None:
        """Create a trainer.

        Args:
            grammar: Expression vocabulary and limits.
            settings: Hyperparameters.
            checkpoint_dir: Directory for ``rl_checkpoint.pt``; ``None`` disables checkpoints.
            metrics_path: JSON-lines file receiving one record per update.
            scorer_factory: Overrides the reward function derived from
                ``settings.reward_mode`` (used by tests with synthetic rewards).
        """
        self.scorer_factory = scorer_factory
        self.grammar = grammar
        self.settings = settings or RLSettings()
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.metrics_path = Path(metrics_path) if metrics_path else None
        self.device = torch.device(self.settings.device)
        self.policy: RecurrentPolicy | None = None
        self.environment: ExpressionEnv | None = None

    # -- construction ---------------------------------------------------------

    def _scorer(self, evaluator: FactorEvaluator) -> Scorer:
        if self.scorer_factory is not None:
            return self.scorer_factory(evaluator)
        if self.settings.reward_mode == "pool_gain":
            return PoolGainScorer(evaluator, capacity=self.settings.pool_capacity)
        return FitnessScorer(evaluator, diversity_weight=self.settings.diversity_weight)

    def build(
        self, evaluator: FactorEvaluator
    ) -> tuple[ExpressionEnv, RecurrentPolicy, torch.optim.Optimizer, torch.Generator]:
        """Create the environment, seeded policy, optimizer and sampling generator."""
        env = ExpressionEnv(
            self.grammar, self._scorer(evaluator), max_tokens=self.settings.max_tokens
        )
        torch.manual_seed(self.settings.seed)
        policy = RecurrentPolicy(
            env.n_actions,
            embed_dim=self.settings.embed_dim,
            hidden_dim=self.settings.hidden_dim,
            num_layers=self.settings.num_layers,
        ).to(self.device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=self.settings.learning_rate)
        generator = torch.Generator(device="cpu").manual_seed(self.settings.seed)
        return env, policy, optimizer, generator

    # -- rollouts -----------------------------------------------------------

    @torch.no_grad()
    def rollout(
        self, env: ExpressionEnv, policy: RecurrentPolicy, generator: torch.Generator
    ) -> Episode:
        """Sample one episode with the masked policy.

        Raises:
            BudgetExhaustedError: Propagated from the scorer at END.
        """
        _, info = env.reset()
        previous = torch.zeros(1, dtype=torch.long, device=self.device)
        hidden: torch.Tensor | None = None
        actions: list[int] = []
        masks: list[np.ndarray] = []
        while True:
            mask = info["action_mask"]
            logits, _, hidden = policy.step(previous, hidden)
            mask_t = torch.as_tensor(mask, device=self.device).unsqueeze(0)
            probs = torch.softmax(masked_logits(logits, mask_t), dim=-1).cpu()
            action = int(torch.multinomial(probs, 1, generator=generator).item())
            actions.append(action)
            masks.append(mask)
            _, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                return Episode(
                    actions,
                    masks,
                    float(reward),
                    info.get("expression"),
                    bool(info["dead_end"]),
                    info.get("outcome"),
                )
            previous = torch.tensor([action + 1], dtype=torch.long, device=self.device)

    def _batch_tensors(
        self, episodes: list[Episode], n_actions: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        length = self.settings.max_tokens
        batch = len(episodes)
        inputs = np.zeros((batch, length), dtype=np.int64)
        actions = np.zeros((batch, length), dtype=np.int64)
        masks = np.ones((batch, length, n_actions), dtype=bool)  # padding: all legal (ignored)
        valid = np.zeros((batch, length), dtype=bool)
        for i, episode in enumerate(episodes):
            steps = len(episode.actions)
            actions[i, :steps] = episode.actions
            inputs[i, 1:steps] = np.asarray(episode.actions[:-1], dtype=np.int64) + 1
            masks[i, :steps] = np.stack(episode.masks)
            valid[i, :steps] = True
        to = lambda a: torch.as_tensor(a, device=self.device)  # noqa: E731
        return to(inputs), to(actions), to(masks), to(valid)

    def log_probs(
        self, policy: RecurrentPolicy, episodes: list[Episode], n_actions: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Teacher-forced log-probabilities, values, entropies and the step mask."""
        inputs, actions, masks, valid = self._batch_tensors(episodes, n_actions)
        logits, values = policy(inputs)
        log_pi = torch.log_softmax(masked_logits(logits, masks), dim=-1)
        chosen = log_pi.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        entropy = -(log_pi.exp() * log_pi).sum(-1)
        return chosen, values, entropy, valid

    def update(
        self,
        policy: RecurrentPolicy,
        optimizer: torch.optim.Optimizer,
        episodes: list[Episode],
        normalizer: RunningMeanStd,
        n_actions: int,
    ) -> tuple[float, float, float]:
        """One gradient step; returns ``(policy_loss, value_loss, entropy)``."""
        rewards = np.array([e.reward for e in episodes], dtype=np.float64)
        normalizer.update(rewards)
        returns = torch.as_tensor(
            normalizer.normalize(rewards), dtype=torch.float32, device=self.device
        )
        chosen, values, entropy, valid = self.log_probs(policy, episodes, n_actions)
        target = returns.unsqueeze(1).expand_as(values)
        advantage = (target - values).detach()
        steps = valid.float()
        denom = steps.sum().clamp_min(1.0)
        policy_loss = -(advantage * chosen * steps).sum() / denom
        value_loss = (((values - target) ** 2) * steps).sum() / denom
        mean_entropy = (entropy * steps).sum() / denom
        settings = self.settings
        loss = policy_loss + settings.value_coef * value_loss - settings.entropy_coef * mean_entropy
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), settings.grad_clip)
        optimizer.step()
        return policy_loss.item(), value_loss.item(), mean_entropy.item()

    # -- checkpointing ------------------------------------------------------

    def _identity(self, evaluator: FactorEvaluator, env: ExpressionEnv) -> dict[str, Any]:
        settings = {k: v for k, v in asdict(self.settings).items() if k != "max_updates"}
        return json.loads(
            to_json_text(
                {
                    "panel": evaluator.panel.fingerprint,
                    "split": evaluator.split.to_dict(),
                    "settings": settings,
                    "grammar": self.grammar.to_dict(),
                    "vocabulary": env.builder.vocabulary(),
                }
            )
        )

    def checkpoint_path(self) -> Path:
        """Location of the checkpoint file."""
        if self.checkpoint_dir is None:
            raise CheckpointError("no checkpoint_dir configured")
        return self.checkpoint_dir / CHECKPOINT_NAME

    def _save(
        self,
        evaluator: FactorEvaluator,
        env: ExpressionEnv,
        policy: RecurrentPolicy,
        optimizer: torch.optim.Optimizer,
        generator: torch.Generator,
        normalizer: RunningMeanStd,
        history: SearchHistory,
        progress: _Progress,
    ) -> None:
        if self.checkpoint_dir is None or evaluator.budget is None:
            return
        path = self.checkpoint_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": CHECKPOINT_FORMAT,
            "method": self.method,
            "identity_json": to_json_text(self._identity(evaluator, env)),
            "update": progress.update,
            "stall": progress.stall,
            "model": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "generator": generator.get_state(),
            "normalizer": asdict(normalizer),
            "history_json": json.dumps(history.to_list()),
            "budget_json": json.dumps(evaluator.budget.state()),
            "scorer_json": json.dumps(env.scorer.state() if isinstance(env.scorer, Scorer) else {}),
            "metrics_json": json.dumps([asdict(m) for m in progress.metrics]),
        }
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)

    def _load(
        self,
        evaluator: FactorEvaluator,
        env: ExpressionEnv,
        policy: RecurrentPolicy,
        optimizer: torch.optim.Optimizer,
        generator: torch.Generator,
    ) -> tuple[RunningMeanStd, SearchHistory, _Progress]:
        path = self.checkpoint_path()
        if not path.is_file():
            raise CheckpointError(f"checkpoint not found: {path}")
        try:
            data = torch.load(path, map_location=self.device, weights_only=True)
        except Exception as exc:  # torch raises several unrelated types for corrupt files
            raise CheckpointError(f"cannot read checkpoint {path}: {exc}") from exc
        if data.get("format") != CHECKPOINT_FORMAT or data.get("method") != self.method:
            raise CheckpointError("checkpoint format or method does not match")
        if json.loads(data["identity_json"]) != self._identity(evaluator, env):
            raise CheckpointError(
                "checkpoint was written for different data, split, settings, grammar or vocabulary"
            )
        policy.load_state_dict(data["model"])
        optimizer.load_state_dict(data["optimizer"])
        generator.set_state(data["generator"])
        evaluator.budget = EvaluationBudget.from_state(json.loads(data["budget_json"]))
        if isinstance(env.scorer, Scorer):
            env.scorer.load_state(json.loads(data["scorer_json"]))
        progress = _Progress(
            update=int(data["update"]),
            stall=int(data["stall"]),
            metrics=[UpdateMetrics(**m) for m in json.loads(data["metrics_json"])],
        )
        return (
            RunningMeanStd(**data["normalizer"]),
            SearchHistory.from_list(json.loads(data["history_json"])),
            progress,
        )

    # -- main loop -----------------------------------------------------------

    def _log(self, metrics: UpdateMetrics) -> None:
        if self.metrics_path is None:
            return
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("a") as handle:
            handle.write(to_json_text(asdict(metrics)).replace("\n", "") + "\n")

    def run(self, evaluator: FactorEvaluator, *, resume: bool = False) -> SearchResult:
        """Train until the budget, ``max_updates`` or the stall guard stops the run."""
        if evaluator.budget is None:
            raise ValueError("RL search requires an evaluator with a budget")
        env, policy, optimizer, generator = self.build(evaluator)
        if resume:
            normalizer, history, progress = self._load(evaluator, env, policy, optimizer, generator)
        else:
            normalizer, history, progress = RunningMeanStd(), SearchHistory(), _Progress()
        settings = self.settings
        stopped = "budget exhausted"
        while True:
            if settings.max_updates is not None and progress.update >= settings.max_updates:
                stopped = "max_updates reached"
                break
            episodes: list[Episode] = []
            new = 0
            exhausted = False
            for _ in range(settings.episodes_per_update):
                try:
                    episode = self.rollout(env, policy, generator)
                except BudgetExhaustedError:
                    exhausted = True
                    break
                episodes.append(episode)
                outcome = episode.outcome
                if outcome is not None and outcome.charged and outcome.score is not None:
                    history.add(outcome.score, source="policy", iteration=progress.update)
                    new += 1
            if not episodes:
                break
            policy_loss, value_loss, entropy = self.update(
                policy, optimizer, episodes, normalizer, env.n_actions
            )
            progress.update += 1
            progress.stall = 0 if new else progress.stall + 1
            rewards = [e.reward for e in episodes]
            metrics = UpdateMetrics(
                update=progress.update,
                episodes=len(episodes),
                mean_reward=float(np.mean(rewards)),
                max_reward=float(np.max(rewards)),
                mean_length=float(np.mean([len(e.actions) for e in episodes])),
                dead_ends=sum(e.dead_end for e in episodes),
                new_expressions=new,
                policy_loss=policy_loss,
                value_loss=value_loss,
                entropy=entropy,
                budget_used=evaluator.budget.used,
            )
            progress.metrics.append(metrics)
            self._log(metrics)
            if progress.update % settings.checkpoint_every == 0 or exhausted:
                self._save(
                    evaluator, env, policy, optimizer, generator, normalizer, history, progress
                )
            if exhausted or evaluator.budget.exhausted:
                break
            if progress.stall >= settings.stall_updates:
                stopped = f"stalled: no new expression in {settings.stall_updates} updates"
                break
        self._save(evaluator, env, policy, optimizer, generator, normalizer, history, progress)
        self.policy = policy
        self.environment = env
        return SearchResult(
            method=self.method,
            history=history,
            best=history.top(settings.top_k),
            budget=evaluator.budget.to_dict(),
            stopped_reason=stopped,
            extra={
                "updates": progress.update,
                "settings": asdict(settings),
                "metrics": [asdict(m) for m in progress.metrics],
                "training_status": "toy-scale CPU run; not a research result",
            },
        )
