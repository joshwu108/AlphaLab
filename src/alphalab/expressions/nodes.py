"""Typed abstract syntax tree for factor expressions.

Three node types exist:

* :class:`Feature` -- a panel field such as ``$close`` (kind ``SERIES``).
* :class:`Constant` -- a finite scalar (kind ``CONSTANT``).
* :class:`Call` -- an operator applied to operand nodes and, for time-series
  operators, an integer window (kind ``SERIES``).

Nodes are immutable and **type-checked on construction**: an ill-typed or
look-ahead-capable tree cannot be built. Operands of commutative operators are
sorted on construction, so structural equality coincides with equality of the
canonical serialization; ``add($a, $b)`` and ``add($b, $a)`` are the same node.

Canonical form
--------------
``name(arg, arg[, window])`` with ``$field`` features and shortest round-trip
float literals, for example ``ts_corr($close, $volume, 10)``. The SHA-256 of the
canonical string is the expression's stable identity across processes.
"""

from __future__ import annotations

import enum
import hashlib
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from functools import cached_property

from alphalab.errors import ExpressionTypeError
from alphalab.expressions.operators import OperatorSpec, get_operator


class ValueKind(enum.Enum):
    """Static type of a node's value."""

    SERIES = "series"
    CONSTANT = "constant"


Path = tuple[int, ...]


class Node:
    """Base class; use :class:`Feature`, :class:`Constant` or :class:`Call`."""

    kind: ValueKind

    @property
    def children(self) -> tuple[Node, ...]:
        """Operand nodes (empty for leaves)."""
        return ()

    @cached_property
    def canonical(self) -> str:
        """Canonical serialization."""
        return self._serialize()

    @cached_property
    def key(self) -> str:
        """Stable SHA-256 identity of :attr:`canonical` (hex)."""
        return hashlib.sha256(self.canonical.encode()).hexdigest()

    @cached_property
    def depth(self) -> int:
        """Tree depth; a leaf has depth 1."""
        return 1 + max((c.depth for c in self.children), default=0)

    @cached_property
    def size(self) -> int:
        """Number of nodes; windows are parameters, not nodes."""
        return 1 + sum(c.size for c in self.children)

    @cached_property
    def lookback(self) -> int:
        """Rows of history needed before the first non-NaN output.

        For a time-series operator with lookback ``L`` over an operand with
        lookback ``M`` the total is ``L + M``: the operand's first valid row is
        ``M`` and the operator needs ``L`` further rows.
        """
        return self._own_lookback() + max((c.lookback for c in self.children), default=0)

    @cached_property
    def features(self) -> frozenset[str]:
        """Panel fields referenced anywhere in the tree."""
        out: frozenset[str] = frozenset()
        for child in self.children:
            out |= child.features
        return out

    def _own_lookback(self) -> int:
        return 0

    def _serialize(self) -> str:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.canonical

    def walk(self, path: Path = ()) -> Iterator[tuple[Path, Node]]:
        """Yield ``(path, node)`` pairs in pre-order; ``path`` indexes ``children``."""
        yield path, self
        for index, child in enumerate(self.children):
            yield from child.walk((*path, index))

    def subtree(self, path: Path) -> Node:
        """Return the node at ``path``."""
        node: Node = self
        for index in path:
            node = node.children[index]
        return node

    def replace(self, path: Path, new: Node) -> Node:
        """Return a new tree with the node at ``path`` replaced by ``new``.

        Raises:
            ExpressionTypeError: The replacement makes the tree ill-typed.
        """
        if not path:
            return new
        if not isinstance(self, Call):
            raise IndexError(f"path {path} descends into a leaf")
        head, rest = path[0], path[1:]
        args = list(self.args)
        args[head] = args[head].replace(rest, new)
        return Call(self.op, tuple(args), self.window)


@dataclass(frozen=True, eq=True)
class Feature(Node):
    """Reference to a panel field."""

    name: str
    kind: ValueKind = field(default=ValueKind.SERIES, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ExpressionTypeError(f"invalid feature name {self.name!r}")

    def _serialize(self) -> str:
        return f"${self.name}"

    @cached_property
    def features(self) -> frozenset[str]:
        """The single referenced field."""
        return frozenset({self.name})


def format_constant(value: float) -> str:
    """Shortest round-trip representation; ``-0.0`` is normalized to ``0.0``."""
    value = float(value) + 0.0
    return repr(value)


@dataclass(frozen=True, eq=True)
class Constant(Node):
    """Finite scalar operand."""

    value: float
    kind: ValueKind = field(default=ValueKind.CONSTANT, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise ExpressionTypeError(f"constants must be finite, got {self.value!r}")
        object.__setattr__(self, "value", float(self.value) + 0.0)

    def _serialize(self) -> str:
        return format_constant(self.value)


def _check_call(spec: OperatorSpec, args: tuple[Node, ...], window: int | None) -> None:
    if len(args) != spec.arity:
        raise ExpressionTypeError(f"{spec.name} takes {spec.arity} operand(s), got {len(args)}")
    for arg in args:
        if not isinstance(arg, Node):
            raise ExpressionTypeError(f"{spec.name} operand {arg!r} is not an expression node")
    kinds = [a.kind for a in args]
    if spec.allows_constant:
        if ValueKind.SERIES not in kinds:
            raise ExpressionTypeError(f"{spec.name} needs at least one series operand")
    elif any(k is not ValueKind.SERIES for k in kinds):
        raise ExpressionTypeError(f"{spec.name} operands must all be series")
    if spec.takes_window:
        if window is None or isinstance(window, bool) or not isinstance(window, int):
            raise ExpressionTypeError(f"{spec.name} requires an integer window")
        if window < spec.min_window:
            # Windows below the minimum include zero and negative values, which
            # would read the current or future rows; they are never allowed.
            raise ExpressionTypeError(
                f"{spec.name} window must be >= {spec.min_window}, got {window}"
            )
    elif window is not None:
        raise ExpressionTypeError(f"{spec.name} does not take a window")


@dataclass(frozen=True, eq=True)
class Call(Node):
    """Operator application."""

    op: str
    args: tuple[Node, ...]
    window: int | None = None
    kind: ValueKind = field(default=ValueKind.SERIES, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        spec = get_operator(self.op)
        args = tuple(self.args)
        _check_call(spec, args, self.window)
        if spec.commutative:
            args = tuple(sorted(args, key=lambda a: a.canonical))
        object.__setattr__(self, "args", args)

    @property
    def spec(self) -> OperatorSpec:
        """Operator declaration."""
        return get_operator(self.op)

    @property
    def children(self) -> tuple[Node, ...]:
        """Operands."""
        return self.args

    def _own_lookback(self) -> int:
        return self.spec.lookback(self.window)

    def _serialize(self) -> str:
        parts = [a.canonical for a in self.args]
        if self.window is not None:
            parts.append(str(self.window))
        return f"{self.op}({', '.join(parts)})"


def call(op: str, *args: Node, window: int | None = None) -> Call:
    """Convenience constructor: ``call("ts_mean", Feature("close"), window=5)``."""
    return Call(op, tuple(args), window)


def feature(name: str) -> Feature:
    """Convenience constructor for :class:`Feature`."""
    return Feature(name)


def transform(node: Node, fn: Callable[[Node], Node]) -> Node:
    """Rebuild ``node`` bottom-up, applying ``fn`` to every rebuilt node."""
    if isinstance(node, Call):
        rebuilt = Call(node.op, tuple(transform(a, fn) for a in node.args), node.window)
        return fn(rebuilt)
    return fn(node)
