"""Tests for AST construction, canonical form, hashing and parsing."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from alphalab.errors import ExpressionParseError, ExpressionTypeError
from alphalab.expressions import Call, Constant, Feature, Grammar, call, parse, sample_expression

CLOSE, VOLUME = Feature("close"), Feature("volume")


def test_canonical_form_and_metrics() -> None:
    node = call("ts_corr", CLOSE, call("ts_mean", VOLUME, window=5), window=10)
    assert node.canonical == "ts_corr($close, ts_mean($volume, 5), 10)"
    assert node.depth == 3 and node.size == 4
    assert node.lookback == 9 + 4
    assert node.features == {"close", "volume"}


def test_commutative_operands_are_canonically_ordered() -> None:
    a = call("add", VOLUME, CLOSE)
    b = call("add", CLOSE, VOLUME)
    assert a == b and a.key == b.key and hash(a) == hash(b)
    assert call("sub", VOLUME, CLOSE) != call("sub", CLOSE, VOLUME)


def test_negative_zero_constants_are_normalized() -> None:
    assert Constant(-0.0).canonical == "0.0"


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: call("add", CLOSE), "takes 2"),
        (lambda: call("add", Constant(1.0), Constant(2.0)), "at least one series"),
        (lambda: call("ts_mean", Constant(1.0), window=5), "must all be series"),
        (lambda: call("ts_mean", CLOSE), "integer window"),
        (lambda: call("delay", CLOSE, window=0), "window must be >= 1"),
        (lambda: call("delay", CLOSE, window=-3), "window must be >= 1"),
        (lambda: call("ts_std", CLOSE, window=1), "window must be >= 2"),
        (lambda: call("cs_rank", CLOSE, window=3), "does not take a window"),
        (lambda: Constant(float("inf")), "finite"),
        (lambda: Feature("bad name"), "invalid feature"),
    ],
)
def test_ill_typed_trees_cannot_be_built(build: object, message: str) -> None:
    with pytest.raises(ExpressionTypeError, match=message):
        build()  # type: ignore[operator]


def test_unknown_operator_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown operator"):
        Call("lead", (CLOSE,), 1)


def test_parse_examples() -> None:
    node = parse(" div( delay($close,5) , $close ) ")
    assert node == call("div", call("delay", CLOSE, window=5), CLOSE)
    assert parse("add($close, 2)") == call("add", CLOSE, Constant(2.0))
    assert parse("mul(-1.5e-1, $volume)").canonical == "mul($volume, -0.15)"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("ts_mean($close)", "integer window"),
        ("ts_mean($close, 2.5)", "integer window"),
        ("delay($close, -1)", "window must be"),
        ("lead($close, 1)", "unknown operator"),
        ("add($close, $open", "end of expression"),
        ("add($close, $open) $high", "trailing"),
        ("add($close; $open)", "unexpected character"),
    ],
)
def test_parse_errors(text: str, message: str) -> None:
    with pytest.raises(ExpressionParseError, match=message):
        parse(text)


@settings(max_examples=150, deadline=None)
@given(seed=st.integers(0, 2**32 - 1))
def test_parse_inverts_canonical_serialization(seed: int) -> None:
    node = sample_expression(Grammar(), np.random.default_rng(seed))
    assert parse(node.canonical) == node
    assert parse(node.canonical).key == node.key


def test_walk_and_replace_produce_new_trees() -> None:
    node = parse("sub(ts_mean($close, 5), $open)")
    paths = dict(node.walk())
    assert paths[(0, 0)] == CLOSE
    replaced = node.replace((0, 0), VOLUME)
    assert replaced.canonical == "sub(ts_mean($volume, 5), $open)"
    assert node.canonical == "sub(ts_mean($close, 5), $open)"
    with pytest.raises(ExpressionTypeError):
        node.replace((1,), Constant(1.0)).replace((0,), Constant(2.0))
