"""Deterministic CPU reference evaluator.

:func:`evaluate` computes an expression on a :class:`~alphalab.data.MarketPanel`
bottom-up, optionally memoizing every sub-expression in an
:class:`~alphalab.expressions.cache.ExpressionCache`. The evaluator is the
reference semantics for the language: any future accelerated backend must
match it.

Floating-point warnings from intermediate operations are suppressed inside the
kernels because every non-finite value is converted to NaN by
:func:`~alphalab.expressions.operators.sanitize`; the documented NaN semantics
are the contract, not the warnings.
"""

from __future__ import annotations

import numpy as np

from alphalab.data.panel import MarketPanel
from alphalab.errors import ExpressionError, ExpressionTypeError
from alphalab.expressions.cache import ExpressionCache
from alphalab.expressions.nodes import Call, Constant, Feature, Node, ValueKind
from alphalab.expressions.operators import sanitize
from alphalab.expressions.parser import parse


def _evaluate_node(
    node: Node, panel: MarketPanel, cache: ExpressionCache | None
) -> np.ndarray | float:
    if isinstance(node, Constant):
        return node.value
    key = (panel.fingerprint, node.canonical)
    if cache is not None:
        hit = cache.get(key)
        if hit is not None:
            return hit
    if isinstance(node, Feature):
        if node.name not in panel.fields:
            raise ExpressionError(
                f"feature ${node.name} is not in the panel; available: {sorted(panel.fields)}"
            )
        result = panel.field(node.name)
    elif isinstance(node, Call):
        operands = [_evaluate_node(a, panel, cache) for a in node.args]
        extra = (node.window,) if node.window is not None else ()
        with np.errstate(all="ignore"):
            result = sanitize(node.spec.kernel(*operands, *extra))
        if np.ndim(result) != 2:
            result = np.broadcast_to(result, panel.shape).astype(np.float64)
    else:  # pragma: no cover - Node is closed over the three subclasses
        raise ExpressionTypeError(f"unsupported node type {type(node).__name__}")
    if cache is not None:
        return cache.put(key, result)
    return result


def evaluate(
    expression: Node | str,
    panel: MarketPanel,
    cache: ExpressionCache | None = None,
) -> np.ndarray:
    """Evaluate a series-valued expression on ``panel``.

    Args:
        expression: Node or canonical string.
        panel: Input data; only fields referenced by the expression are read.
        cache: Optional shared sub-expression cache.

    Returns:
        A ``(T, N)`` float64 array. It is read-only when served from or stored
        in a cache; copy it before modifying.

    Raises:
        ExpressionTypeError: The expression is a bare constant.
        ExpressionError: A referenced feature is missing from the panel.
    """
    node = parse(expression) if isinstance(expression, str) else expression
    if node.kind is not ValueKind.SERIES:
        raise ExpressionTypeError("a factor must be series-valued, not a bare constant")
    result = _evaluate_node(node, panel, cache)
    assert isinstance(result, np.ndarray)
    return result
