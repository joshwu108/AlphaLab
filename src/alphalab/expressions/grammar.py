"""Search grammar: the vocabulary and limits shared by every search method.

A :class:`Grammar` fixes which features, operators, windows and constants may
appear and how large an expression may be. Random search, genetic programming
and the RL environment all draw from the same grammar, which keeps their
hypothesis spaces comparable.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field

import numpy as np

from alphalab.errors import ExpressionLimitError, ExpressionTypeError
from alphalab.expressions.nodes import Call, Constant, Feature, Node, ValueKind
from alphalab.expressions.operators import OPERATORS, OperatorSpec, get_operator

DEFAULT_FEATURES = ("open", "high", "low", "close", "volume", "vwap")
DEFAULT_WINDOWS = (1, 5, 10, 20, 40)
DEFAULT_CONSTANTS = (-1.0, -0.5, 0.5, 1.0, 2.0)

# Probability that a binary operand slot that admits constants gets one when sampling.
CONSTANT_PROBABILITY = 0.2
MAX_SAMPLING_ATTEMPTS = 1_000


@dataclass(frozen=True)
class Grammar:
    """Vocabulary and size limits.

    Attributes:
        features: Panel fields that may appear as ``$name``.
        operators: Operator names drawn from the registry.
        windows: Candidate window lengths; each operator uses those at or
            above its ``min_window``.
        constants: Candidate scalar constants.
        max_depth: Maximum tree depth (a leaf has depth 1).
        max_size: Maximum number of nodes.
        max_lookback: Maximum required history in rows.
        forbid_degenerate: Reject degenerate forms (see :func:`is_degenerate`).
            Applies to every search method alike, so no method gains budget
            by being the only one that skips them.
        constants_last: A constant may only be the *second* operand of a
            non-commutative operator: ``sub(x, 1.0)`` is allowed,
            ``sub(1.0, x)`` is not. The RL builder writes Reverse Polish
            Notation and cannot place a constant first, so this keeps the
            hypothesis space identical for all methods. Little is lost for
            rank-based scoring: ``sub(c, x)`` ranks like ``neg(x)`` and
            ``div(c, x)`` ranks like ``neg(x)`` wherever ``x`` keeps one sign.
    """

    features: tuple[str, ...] = DEFAULT_FEATURES
    operators: tuple[str, ...] = tuple(OPERATORS)
    windows: tuple[int, ...] = DEFAULT_WINDOWS
    constants: tuple[float, ...] = DEFAULT_CONSTANTS
    max_depth: int = 5
    max_size: int = 15
    max_lookback: int = 60
    forbid_degenerate: bool = True
    constants_last: bool = True
    _specs: Mapping[str, OperatorSpec] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.features:
            raise ExpressionTypeError("a grammar needs at least one feature")
        specs = {name: get_operator(name) for name in self.operators}
        for spec in specs.values():
            if spec.takes_window and not self.windows_for(spec.name, specs):
                raise ExpressionTypeError(
                    f"no configured window satisfies {spec.name} (min {spec.min_window})"
                )
        if any(w < 1 for w in self.windows):
            raise ExpressionTypeError("windows must be positive")
        if self.max_depth < 1 or self.max_size < 1 or self.max_lookback < 0:
            raise ExpressionLimitError("max_depth and max_size must be >= 1, max_lookback >= 0")
        object.__setattr__(self, "features", tuple(self.features))
        object.__setattr__(self, "operators", tuple(self.operators))
        object.__setattr__(self, "windows", tuple(sorted(set(self.windows))))
        object.__setattr__(self, "constants", tuple(float(c) for c in self.constants))
        object.__setattr__(self, "_specs", specs)

    def windows_for(
        self, op: str, specs: Mapping[str, OperatorSpec] | None = None
    ) -> tuple[int, ...]:
        """Windows usable by ``op``."""
        spec = (specs or self._specs)[op]
        return tuple(w for w in sorted(set(self.windows)) if w >= spec.min_window)

    def spec(self, op: str) -> OperatorSpec:
        """Spec of an operator in this grammar."""
        return self._specs[op]

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable form (for configurations and manifests)."""
        data = asdict(self)
        data.pop("_specs", None)
        return data

    def check(self, node: Node) -> None:
        """Validate ``node`` against this grammar.

        Raises:
            ExpressionTypeError: Unknown feature, operator, window or constant.
            ExpressionLimitError: Depth, size or lookback limit exceeded.
        """
        if node.kind is not ValueKind.SERIES:
            raise ExpressionTypeError("a factor must be series-valued")
        if node.depth > self.max_depth:
            raise ExpressionLimitError(f"depth {node.depth} exceeds {self.max_depth}")
        if node.size > self.max_size:
            raise ExpressionLimitError(f"size {node.size} exceeds {self.max_size}")
        if node.lookback > self.max_lookback:
            raise ExpressionLimitError(f"lookback {node.lookback} exceeds {self.max_lookback}")
        for _, sub in node.walk():
            if isinstance(sub, Feature) and sub.name not in self.features:
                raise ExpressionTypeError(f"feature ${sub.name} is not in the grammar")
            if isinstance(sub, Call):
                if self.forbid_degenerate and is_degenerate(sub):
                    raise ExpressionTypeError(f"degenerate form {sub.canonical}")
                if (
                    self.constants_last
                    and not sub.spec.commutative
                    and isinstance(sub.args[0], Constant)
                ):
                    raise ExpressionTypeError(f"leading constant in {sub.canonical}")
                if sub.op not in self._specs:
                    raise ExpressionTypeError(f"operator {sub.op} is not in the grammar")
                if sub.window is not None and sub.window not in self.windows_for(sub.op):
                    raise ExpressionTypeError(f"window {sub.window} is not allowed for {sub.op}")

    def is_valid(self, node: Node) -> bool:
        """``True`` when :meth:`check` passes."""
        try:
            self.check(node)
        except (ExpressionTypeError, ExpressionLimitError):
            return False
        return True


# Binary operators whose result is still informative with identical operands.
NON_DEGENERATE_SELF_BINARY = frozenset({"mul"})


def is_degenerate(node: Call) -> bool:
    """Whether ``node`` is a redundant or constant-valued form at its root.

    * a non-windowed unary operator applied to its own result, such as
      ``neg(neg(x))``, ``abs(abs(x))`` or ``cs_rank(cs_rank(x))`` (identity or
      idempotent);
    * a binary operator other than ``mul`` with identical operands, such as
      ``sub(x, x)``, ``div(x, x)`` or ``ts_corr(x, x, w)`` (constant), or
      ``add(x, x)`` (a rescaling of ``x``).
    """
    spec = node.spec
    if spec.arity == 1 and not spec.takes_window:
        child = node.args[0]
        return isinstance(child, Call) and child.op == node.op
    if spec.arity == 2:
        return node.op not in NON_DEGENERATE_SELF_BINARY and node.args[0] == node.args[1]
    return False


def _sample_series(grammar: Grammar, rng: np.random.Generator, depth: int) -> Node:
    """Grow a random series-valued tree of depth at most ``depth``."""
    terminal_probability = 0.3 if depth > 1 else 1.0
    if depth <= 1 or rng.random() < terminal_probability:
        return Feature(grammar.features[rng.integers(len(grammar.features))])
    spec = grammar.spec(grammar.operators[rng.integers(len(grammar.operators))])
    args: list[Node] = []
    # With constants_last the constant always goes in the final slot; otherwise
    # either slot is equally likely.
    any_slot = int(rng.integers(spec.arity))
    slot_choice = spec.arity - 1 if grammar.constants_last and not spec.commutative else any_slot
    constant_slot = (
        slot_choice
        if spec.allows_constant and grammar.constants and rng.random() < CONSTANT_PROBABILITY
        else -1
    )
    for slot in range(spec.arity):
        if slot == constant_slot:
            args.append(Constant(grammar.constants[rng.integers(len(grammar.constants))]))
        else:
            args.append(_sample_series(grammar, rng, depth - 1))
    window = None
    if spec.takes_window:
        options = grammar.windows_for(spec.name)
        window = int(options[rng.integers(len(options))])
    return Call(spec.name, tuple(args), window)


def sample_expression(
    grammar: Grammar,
    rng: np.random.Generator,
    *,
    max_depth: int | None = None,
    require_operator: bool = True,
) -> Node:
    """Sample a random valid expression.

    Trees are grown top-down with a 30% chance of stopping at each internal
    level and rejected if they violate the grammar's size or lookback limits.

    Args:
        grammar: Vocabulary and limits.
        rng: Source of randomness; equal generator states give equal samples.
        max_depth: Depth bound; defaults to ``grammar.max_depth``.
        require_operator: Reject bare features (which are trivial factors).

    Raises:
        ExpressionLimitError: No valid tree was found within the attempt limit.
    """
    depth = min(max_depth or grammar.max_depth, grammar.max_depth)
    for _ in range(MAX_SAMPLING_ATTEMPTS):
        node = _sample_series(grammar, rng, depth)
        if require_operator and not isinstance(node, Call):
            continue
        if grammar.is_valid(node):
            return node
    raise ExpressionLimitError("could not sample a valid expression; loosen the grammar limits")
