"""Forward/backward and training-loop tests for the torch policy (CPU only)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from alphalab.data import SyntheticSpec  # noqa: E402
from alphalab.errors import CheckpointError  # noqa: E402
from alphalab.expressions import Grammar, Node  # noqa: E402
from alphalab.rl import ExpressionEnv, ScoreOutcome, Scorer  # noqa: E402
from alphalab.rl.policy import RecurrentPolicy, masked_logits  # noqa: E402
from alphalab.rl.trainer import (  # noqa: E402
    Episode,
    PolicyGradientTrainer,
    RLSettings,
    RunningMeanStd,
)
from tests.conftest import build_evaluator  # noqa: E402

pytestmark = pytest.mark.torch

GRAMMAR = Grammar(max_depth=3, max_size=6, max_lookback=20)
TOY_SPEC = SyntheticSpec(n_periods=200, n_instruments=12, seed=1)


class CloseOnly(Scorer):
    """Synthetic reward: 1 for expressions that use only $close."""

    def __call__(self, node: Node) -> ScoreOutcome:
        return ScoreOutcome(1.0 if node.features == {"close"} else 0.0)


def _settings(**overrides: object) -> RLSettings:
    base = {
        "max_tokens": 8,
        "episodes_per_update": 8,
        "embed_dim": 16,
        "hidden_dim": 32,
        "seed": 0,
    }
    base.update(overrides)
    return RLSettings(**base)  # type: ignore[arg-type]


def _signature(result: object) -> list[tuple[str, float]]:
    return [(r.expression, r.fitness) for r in result.history.records]  # type: ignore[attr-defined]


def test_masked_logits_give_zero_probability_and_finite_entropy() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[True, False, True]])
    probs = torch.softmax(masked_logits(logits, mask), dim=-1)
    assert probs[0, 1].item() == 0.0
    log_p = torch.log_softmax(masked_logits(logits, mask), dim=-1)
    assert torch.isfinite(-(log_p.exp() * log_p).sum())


def test_sampling_and_teacher_forcing_agree() -> None:
    trainer = PolicyGradientTrainer(GRAMMAR, _settings(), scorer_factory=lambda ev: CloseOnly())
    env, policy, _, generator = trainer.build(build_evaluator(TOY_SPEC, budget=5))
    episodes = [trainer.rollout(env, policy, generator) for _ in range(4)]

    stepwise = []
    with torch.no_grad():
        for episode in episodes:
            hidden, previous, total = None, torch.zeros(1, dtype=torch.long), 0.0
            for action, mask in zip(episode.actions, episode.masks, strict=True):
                logits, _, hidden = policy.step(previous, hidden)
                log_p = torch.log_softmax(masked_logits(logits, torch.as_tensor(mask)[None]), -1)
                total += log_p[0, action].item()
                previous = torch.tensor([action + 1])
            stepwise.append(total)
        chosen, _, _, valid = trainer.log_probs(policy, episodes, env.n_actions)
    batched = (chosen * valid).sum(dim=1).tolist()
    np.testing.assert_allclose(batched, stepwise, rtol=1e-5, atol=1e-5)


def test_update_produces_finite_gradients_for_every_parameter() -> None:
    trainer = PolicyGradientTrainer(GRAMMAR, _settings(), scorer_factory=lambda ev: CloseOnly())
    env, policy, optimizer, generator = trainer.build(build_evaluator(TOY_SPEC, budget=5))
    episodes = [trainer.rollout(env, policy, generator) for _ in range(6)]
    episodes[0] = Episode(episodes[0].actions, episodes[0].masks, 1.0, None, False)
    before = [p.detach().clone() for p in policy.parameters()]
    trainer.update(policy, optimizer, episodes, RunningMeanStd(), env.n_actions)
    for param, old in zip(policy.parameters(), before, strict=True):
        assert param.grad is not None and torch.isfinite(param.grad).all()
        assert not torch.equal(param, old)


def test_running_mean_std_matches_numpy() -> None:
    rng = np.random.default_rng(0)
    data = [rng.normal(3.0, 2.0, size=n) for n in (5, 1, 17)]
    stats = RunningMeanStd()
    for batch in data:
        stats.update(batch)
    joined = np.concatenate(data)
    assert stats.mean == pytest.approx(joined.mean())
    assert stats.std == pytest.approx(joined.std())


def test_policy_learns_a_synthetic_reward() -> None:
    trainer = PolicyGradientTrainer(
        GRAMMAR,
        _settings(episodes_per_update=16, learning_rate=3e-2, max_updates=30),
        scorer_factory=lambda ev: CloseOnly(),
    )
    result = trainer.run(build_evaluator(TOY_SPEC, budget=5))
    metrics = result.extra["metrics"]
    assert metrics[0]["mean_reward"] < 0.3
    assert np.mean([m["mean_reward"] for m in metrics[-5:]]) > 0.8


def test_training_on_planted_data_is_deterministic_and_budget_matched(
    strong_spec: SyntheticSpec, tmp_path: Path
) -> None:
    settings = _settings(max_tokens=10, episodes_per_update=12)
    log = tmp_path / "metrics.jsonl"
    first = PolicyGradientTrainer(GRAMMAR, settings, metrics_path=log).run(
        build_evaluator(strong_spec, budget=40)
    )
    second = PolicyGradientTrainer(GRAMMAR, settings).run(build_evaluator(strong_spec, budget=40))
    assert _signature(first) == _signature(second)
    assert len(first.history.records) == first.budget["used"] == 40
    assert first.stopped_reason == "budget exhausted"
    lines = log.read_text().splitlines()
    assert len(lines) == first.extra["updates"] and json.loads(lines[0])["update"] == 1


def test_resume_reproduces_an_uninterrupted_run(strong_spec: SyntheticSpec, tmp_path: Path) -> None:
    settings = _settings(max_tokens=10, episodes_per_update=6, checkpoint_every=1)
    full = PolicyGradientTrainer(GRAMMAR, settings).run(build_evaluator(strong_spec, budget=30))

    limited = _settings(max_tokens=10, episodes_per_update=6, checkpoint_every=1, max_updates=2)
    partial = PolicyGradientTrainer(GRAMMAR, limited, checkpoint_dir=tmp_path).run(
        build_evaluator(strong_spec, budget=30)
    )
    assert partial.stopped_reason == "max_updates reached"
    resumed_trainer = PolicyGradientTrainer(GRAMMAR, settings, checkpoint_dir=tmp_path)
    resumed = resumed_trainer.run(build_evaluator(strong_spec, budget=30), resume=True)
    assert _signature(resumed) == _signature(full)
    assert resumed.extra["updates"] == full.extra["updates"]


def test_resume_rejects_mismatched_or_missing_checkpoints(
    strong_spec: SyntheticSpec, tmp_path: Path
) -> None:
    trainer = PolicyGradientTrainer(GRAMMAR, _settings(max_updates=1), checkpoint_dir=tmp_path)
    trainer.run(build_evaluator(strong_spec, budget=20))
    other = PolicyGradientTrainer(GRAMMAR, _settings(seed=1), checkpoint_dir=tmp_path)
    with pytest.raises(CheckpointError, match="different"):
        other.run(build_evaluator(strong_spec, budget=20), resume=True)
    (tmp_path / "rl_checkpoint.pt").write_bytes(b"not a checkpoint")
    with pytest.raises(CheckpointError, match="cannot read"):
        trainer.run(build_evaluator(strong_spec, budget=20), resume=True)
    with pytest.raises(CheckpointError, match="not found"):
        PolicyGradientTrainer(GRAMMAR, _settings(), checkpoint_dir=tmp_path / "none").run(
            build_evaluator(strong_spec, budget=20), resume=True
        )


def test_pool_gain_reward_mode_runs(strong_spec: SyntheticSpec) -> None:
    trainer = PolicyGradientTrainer(GRAMMAR, _settings(reward_mode="pool_gain", max_updates=2))
    result = trainer.run(build_evaluator(strong_spec, budget=30))
    assert result.extra["updates"] == 2
    assert isinstance(trainer.environment, ExpressionEnv)


def test_settings_validation() -> None:
    with pytest.raises(ValueError):
        RLSettings(reward_mode="sharpe")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        RLSettings(episodes_per_update=0)
    with pytest.raises(ValueError):
        RecurrentPolicy(1)


def test_resume_rewrites_metrics_logged_after_the_last_checkpoint(
    strong_spec: SyntheticSpec, tmp_path: Path
) -> None:
    log = tmp_path / "metrics.jsonl"
    limited = _settings(max_tokens=10, episodes_per_update=6, max_updates=2, checkpoint_every=1)
    PolicyGradientTrainer(GRAMMAR, limited, checkpoint_dir=tmp_path, metrics_path=log).run(
        build_evaluator(strong_spec, budget=60)
    )
    with log.open("a") as handle:  # an update logged after the checkpoint, then lost
        handle.write(json.dumps({"update": 3, "orphan": True}) + "\n")
    longer = _settings(max_tokens=10, episodes_per_update=6, max_updates=4, checkpoint_every=1)
    PolicyGradientTrainer(GRAMMAR, longer, checkpoint_dir=tmp_path, metrics_path=log).run(
        build_evaluator(strong_spec, budget=60), resume=True
    )
    updates = [json.loads(line)["update"] for line in log.read_text().splitlines()]
    assert updates == [1, 2, 3, 4]
