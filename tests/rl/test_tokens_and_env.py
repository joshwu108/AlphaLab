"""Tests for the RPN token grammar, action masks and environment (no torch required)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from alphalab.data import SyntheticSpec
from alphalab.errors import BudgetExhaustedError
from alphalab.expressions import Grammar, Node, parse
from alphalab.rl import (
    BuilderState,
    ExpressionEnv,
    FitnessScorer,
    InvalidActionError,
    PoolGainScorer,
    RPNBuilder,
    ScoreOutcome,
    TokenKind,
)
from tests.conftest import build_evaluator

GRAMMAR = Grammar(max_depth=4, max_size=10, max_lookback=30)


def _null_scorer(node: Node) -> ScoreOutcome:
    return ScoreOutcome(0.0)


def _rollout(env: ExpressionEnv, rng: np.random.Generator) -> tuple[bool, list[int]]:
    _, info = env.reset()
    actions = []
    while True:
        action = int(rng.choice(np.flatnonzero(info["action_mask"])))
        actions.append(action)
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return terminated, actions


def _action(builder: RPNBuilder, kind: TokenKind, value: object) -> int:
    return next(i for i, t in enumerate(builder.tokens) if t.kind is kind and t.value == value)


def test_vocabulary_covers_the_grammar() -> None:
    builder = RPNBuilder(GRAMMAR)
    kinds = [t.kind for t in builder.tokens]
    assert kinds.count(TokenKind.FEATURE) == len(GRAMMAR.features)
    assert kinds.count(TokenKind.OPERATOR) == len(GRAMMAR.operators)
    assert builder.tokens[builder.end_action].kind is TokenKind.END
    assert builder.vocabulary()[0] == "$open"


def test_initial_mask_allows_only_features() -> None:
    builder = RPNBuilder(GRAMMAR)
    mask = builder.action_mask(BuilderState())
    allowed = {builder.tokens[i].kind for i in np.flatnonzero(mask)}
    assert allowed == {TokenKind.FEATURE}


def test_pending_constants_and_windows_restrict_the_next_token() -> None:
    builder = RPNBuilder(GRAMMAR)
    state = builder.step(BuilderState(), _action(builder, TokenKind.FEATURE, "close"))
    assert not builder.action_mask(state)[builder.end_action]  # bare feature is not a factor
    with_constant = builder.step(state, _action(builder, TokenKind.CONSTANT, 2.0))
    ops = {builder.tokens[i].value for i in np.flatnonzero(builder.action_mask(with_constant))}
    assert ops == {"add", "sub", "mul", "div"}
    with_window = builder.step(state, _action(builder, TokenKind.WINDOW, 5))
    ops = {builder.tokens[i].value for i in np.flatnonzero(builder.action_mask(with_window))}
    assert ops == {"delay", "delta", "ts_mean", "ts_sum", "ts_std", "ts_min", "ts_max", "ts_ema"}


def test_degenerate_forms_are_masked() -> None:
    builder = RPNBuilder(GRAMMAR)
    close = _action(builder, TokenKind.FEATURE, "close")
    state = builder.step(BuilderState(), close)
    state = builder.step(state, _action(builder, TokenKind.OPERATOR, "neg"))
    assert not builder.action_mask(state)[_action(builder, TokenKind.OPERATOR, "neg")]
    pair = builder.step(builder.step(BuilderState(), close), close)
    mask = builder.action_mask(pair)
    assert not mask[_action(builder, TokenKind.OPERATOR, "sub")]
    assert mask[_action(builder, TokenKind.OPERATOR, "mul")]


def test_illegal_actions_raise() -> None:
    builder = RPNBuilder(GRAMMAR)
    with pytest.raises(InvalidActionError):
        builder.step(BuilderState(), builder.end_action)
    with pytest.raises(InvalidActionError):
        builder.step(BuilderState(), builder.n_actions)


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 2**32 - 1))
def test_masked_rollouts_finish_with_valid_expressions(seed: int) -> None:
    env = ExpressionEnv(GRAMMAR, _null_scorer, max_tokens=12)
    rng = np.random.default_rng(seed)
    for _ in range(20):
        terminated, actions = _rollout(env, rng)
        assert terminated, "masked rollouts should not dead-end"
        assert len(actions) <= 12
        node = env.builder.expression(env.state)
        GRAMMAR.check(node)
        assert parse(node.canonical) == node
        replay = BuilderState()
        for action in env.builder.encode(node):
            replay = env.builder.step(replay, action)
        assert env.builder.expression(replay) == node


def test_observation_encoding() -> None:
    env = ExpressionEnv(GRAMMAR, _null_scorer, max_tokens=8)
    obs, info = env.reset()
    assert obs.tolist() == [0] * 8 and info["action_mask"].any()
    action = int(np.flatnonzero(info["action_mask"])[0])
    obs, reward, terminated, truncated, _ = env.step(action)
    assert obs[0] == action + 1 and obs[1:].sum() == 0
    assert (reward, terminated, truncated) == (0.0, False, False)


def test_terminal_step_returns_the_scorer_reward(strong_spec: SyntheticSpec) -> None:
    evaluator = build_evaluator(strong_spec, budget=3)
    env = ExpressionEnv(GRAMMAR, FitnessScorer(evaluator), max_tokens=12)
    env.reset()
    node = parse(strong_spec.planted_expression)
    for action in env.builder.encode(node):
        _, reward, terminated, _, info = env.step(action)
    assert terminated and info["expression"] == node.canonical
    assert reward == pytest.approx(evaluator.score(node).fitness)
    assert info["outcome"].charged and evaluator.budget is not None
    assert evaluator.budget.used == 1


def test_budget_exhaustion_propagates(strong_spec: SyntheticSpec) -> None:
    evaluator = build_evaluator(strong_spec, budget=1)
    env = ExpressionEnv(GRAMMAR, FitnessScorer(evaluator), max_tokens=12)
    for text in ("ts_mean($close, 5)", "ts_std($close, 5)"):
        env.reset()
        actions = env.builder.encode(parse(text))
        if text.startswith("ts_std"):
            for action in actions[:-1]:
                env.step(action)
            with pytest.raises(BudgetExhaustedError):
                env.step(actions[-1])
        else:
            for action in actions:
                env.step(action)


def test_scorers_track_state(strong_spec: SyntheticSpec) -> None:
    evaluator = build_evaluator(strong_spec, budget=10)
    fitness = FitnessScorer(evaluator, diversity_weight=0.5)
    planted = parse(strong_spec.planted_expression)
    first = fitness(planted)
    assert first.reward == pytest.approx(first.score.fitness)  # type: ignore[union-attr]
    assert fitness(planted).reward == pytest.approx(first.reward)  # never penalized vs itself
    mirrored = fitness(parse(f"neg({strong_spec.planted_expression})"))
    assert mirrored.score is not None
    # |corr| with the stored planted factor is 1, so the full weight is subtracted.
    assert mirrored.reward == pytest.approx(mirrored.score.fitness - 0.5)
    restored = FitnessScorer(evaluator)
    restored.load_state(fitness.state())
    assert restored.state() == fitness.state()

    pool = PoolGainScorer(evaluator, capacity=3)
    gain = pool(planted)
    assert gain.reward > 0 and pool(planted).reward == 0.0
    clone = PoolGainScorer(evaluator, capacity=3)
    clone.load_state(pool.state())
    assert clone.pool.expressions == pool.pool.expressions
