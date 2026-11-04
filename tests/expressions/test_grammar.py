"""Tests for the search grammar and the random expression sampler."""

from __future__ import annotations

import numpy as np
import pytest

from alphalab.errors import ExpressionLimitError, ExpressionTypeError
from alphalab.expressions import Call, Grammar, parse, sample_expression


def test_sampler_is_deterministic_and_respects_limits() -> None:
    grammar = Grammar(max_depth=4, max_size=9, max_lookback=30)
    first = [sample_expression(grammar, np.random.default_rng(1)) for _ in range(3)]
    second = [sample_expression(grammar, np.random.default_rng(1)) for _ in range(3)]
    assert first == second
    rng = np.random.default_rng(2)
    for _ in range(200):
        node = sample_expression(grammar, rng)
        assert isinstance(node, Call)
        assert node.depth <= 4 and node.size <= 9 and node.lookback <= 30
        grammar.check(node)


def test_check_reports_each_limit() -> None:
    grammar = Grammar(features=("close",), operators=("ts_mean", "add"), windows=(5,), max_depth=3)
    with pytest.raises(ExpressionTypeError, match="feature"):
        grammar.check(parse("ts_mean($open, 5)"))
    with pytest.raises(ExpressionTypeError, match="operator"):
        grammar.check(parse("ts_std($close, 5)"))
    with pytest.raises(ExpressionTypeError, match="window"):
        grammar.check(parse("ts_mean($close, 6)"))
    with pytest.raises(ExpressionLimitError, match="depth"):
        grammar.check(parse("ts_mean(ts_mean(ts_mean($close, 5), 5), 5)"))
    assert grammar.is_valid(parse("add(ts_mean($close, 5), 1.0)"))


def test_grammar_rejects_operators_without_usable_windows() -> None:
    with pytest.raises(ExpressionTypeError, match="no configured window"):
        Grammar(operators=("ts_corr",), windows=(1, 2))


def test_grammar_serializes_to_plain_data() -> None:
    data = Grammar().to_dict()
    assert "_specs" not in data and data["max_depth"] == 5
