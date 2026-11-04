"""Token vocabulary and grammar-constrained Reverse Polish Notation builder.

The RL agent writes an expression as a sequence of tokens in Reverse Polish
Notation (RPN), following the formulation of AlphaGen (Yu et al., 2023):
operands are pushed onto a stack and operators pop their operands. RPN needs
no parentheses, so every prefix of a valid program is a stack of complete
sub-expressions, which makes validity checkable after every token.

Formal specification
--------------------
*Vocabulary* ``V`` = features + constants + windows + operators + {END}, built
from a :class:`~alphalab.expressions.grammar.Grammar`. An action is an index
into ``V``.

*State* ``s = (tokens, stack)`` where ``tokens`` is the sequence emitted so far
(length ``<= max_tokens``) and ``stack`` holds expression nodes (series or
constant) and pending integer windows.

*Transitions* (``top`` is the last stack item):

=============  =============================================  ===========================
token          precondition                                   effect
=============  =============================================  ===========================
feature f      top is absent or a series node                 push ``$f``
constant c     top is a series node                           push ``c``
window w       top is a series node                           push ``w``
unary op       top is a series node                           pop 1, push ``op(x)``
binary op      top two are nodes, at least one a series       pop 2, push ``op(a, b)``
windowed op    top is ``w``; ``arity`` series nodes below     pop ``arity + 1``, push
END            stack is exactly one series node               terminal
=============  =============================================  ===========================

A pending constant must be consumed by a constant-accepting binary operator
and a pending window by a windowed operator, so constants and windows never
accumulate. Beyond these typing rules the **action mask** also requires that

* the operator result satisfies the grammar's depth, size and lookback limits;
* the sequence can still be completed within ``max_tokens`` (a counting bound
  on the reductions still needed);
* the grammar size and depth limits can still be met by the cheapest
  completion (lower bounds on final size and depth);
* no degenerate forms as defined by
  :func:`~alphalab.expressions.grammar.is_degenerate` (for example
  ``neg(neg(x))`` or ``sub(x, x)``), the same rule every search method obeys.

The bounds are necessary rather than sufficient conditions (for example, two
identical operands can only be merged by ``mul``), so a rare state can have no
legal action (a *dead end*); the environment then truncates the episode with
the invalid reward. Every expression that reaches END is grammar-valid.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np

from alphalab.errors import ExpressionTypeError
from alphalab.expressions.grammar import Grammar, is_degenerate
from alphalab.expressions.nodes import Call, Constant, Feature, Node, ValueKind
from alphalab.expressions.operators import OperatorSpec

StackItem = Node | int


class TokenKind(enum.Enum):
    """Category of a vocabulary token."""

    FEATURE = "feature"
    CONSTANT = "constant"
    WINDOW = "window"
    OPERATOR = "operator"
    END = "end"


@dataclass(frozen=True)
class Token:
    """One vocabulary entry."""

    kind: TokenKind
    value: str | float | int | None

    @property
    def label(self) -> str:
        """Human-readable form used in logs."""
        if self.kind is TokenKind.FEATURE:
            return f"${self.value}"
        if self.kind is TokenKind.END:
            return "<END>"
        if self.kind is TokenKind.WINDOW:
            return f"w{self.value}"
        return str(self.value)


class InvalidActionError(ValueError):
    """An action violated the action mask."""


@dataclass(frozen=True)
class BuilderState:
    """Immutable RPN builder state."""

    tokens: tuple[int, ...] = ()
    stack: tuple[StackItem, ...] = ()
    done: bool = False


def _is_series(item: StackItem | None) -> bool:
    return isinstance(item, Node) and item.kind is ValueKind.SERIES


class RPNBuilder:
    """Vocabulary plus the transition and mask rules for one grammar."""

    def __init__(self, grammar: Grammar, *, max_tokens: int = 20, require_operator: bool = True):
        if max_tokens < 3:
            raise ValueError("max_tokens must be at least 3")
        self.grammar = grammar
        self.max_tokens = max_tokens
        self.require_operator = require_operator
        tokens: list[Token] = [Token(TokenKind.FEATURE, f) for f in grammar.features]
        tokens += [Token(TokenKind.CONSTANT, c) for c in grammar.constants]
        tokens += [Token(TokenKind.WINDOW, w) for w in grammar.windows]
        tokens += [Token(TokenKind.OPERATOR, op) for op in grammar.operators]
        tokens.append(Token(TokenKind.END, None))
        self.tokens: tuple[Token, ...] = tuple(tokens)
        self.end_action = len(tokens) - 1
        specs = [grammar.spec(op) for op in grammar.operators]
        self._has_plain_binary = any(s.arity == 2 and not s.takes_window for s in specs)
        self._has_plain_unary = any(s.arity == 1 and not s.takes_window for s in specs)

    @property
    def n_actions(self) -> int:
        """Vocabulary size."""
        return len(self.tokens)

    def vocabulary(self) -> list[str]:
        """Token labels in action order (recorded in checkpoints)."""
        return [t.label for t in self.tokens]

    # -- counting bounds ---------------------------------------------------

    def _tokens_needed(self, stack: tuple[StackItem, ...]) -> int:
        """Lower bound on tokens (including END) needed to finish from ``stack``."""
        nodes = [s for s in stack if isinstance(s, Node)]
        windows = len(stack) - len(nodes)
        per_merge = 1 if self._has_plain_binary else 2
        if not nodes:
            return (3 if self._has_plain_unary else 4) if self.require_operator else 2
        reduce = per_merge * (len(nodes) - 1) + windows
        if reduce == 0 and self.require_operator and isinstance(nodes[0], Feature):
            reduce = 1 if self._has_plain_unary else 2
        return reduce + 1

    def _size_feasible(self, stack: tuple[StackItem, ...]) -> bool:
        nodes = [s for s in stack if isinstance(s, Node)]
        merges = max(len(nodes) - 1, 0)
        return sum(n.size for n in nodes) + merges <= self.grammar.max_size

    @staticmethod
    def _merged_depth(depths: list[int]) -> int:
        """Depth after merging a stack of node depths from the top with binary operators.

        For ``[e_0, ..., e_{k-1}]``, ``e_j`` ends ``j + 1`` levels down for
        ``j < k - 1`` and the top item ``k - 1`` levels down.
        """
        k = len(depths)
        if k == 1:
            return depths[0]
        return max(max(depths[j] + j + 1 for j in range(k - 1)), depths[-1] + k - 1)

    def _depth_feasible(self, stack: tuple[StackItem, ...]) -> bool:
        """Whether the shallowest completion of ``stack`` respects ``max_depth``.

        Constants and windows only ever occupy the top of the stack because the
        mask forces the next token to consume them; their admissibility is
        decided by a one-step lookahead instead. For node-only stacks, any
        completion with more pushes or unary operators is deeper than merging
        directly, so this is a lower bound and never rejects a completable stack.
        """
        depths = [item.depth for item in stack if isinstance(item, Node)]
        if not depths or isinstance(stack[-1], int):
            # A pending window was admitted by an explicit one-step lookahead
            # (see _consumable), which checks the stack after its operator.
            return True
        final = self._merged_depth(depths)
        bare = len(stack) == 1 and isinstance(stack[0], Feature) and self.require_operator
        final += 1 if bare else 0
        return final <= self.grammar.max_depth

    def _fits(self, state: BuilderState, new_stack: tuple[StackItem, ...]) -> bool:
        if new_stack and isinstance(new_stack[-1], int):
            # Pending window: _consumable already checked the stack after the
            # consuming operator, which is tighter than counting bounds here
            # (a binary windowed operator merges two nodes in one token).
            return True
        used = len(state.tokens) + 1
        return (
            used + self._tokens_needed(new_stack) <= self.max_tokens
            and self._size_feasible(new_stack)
            and self._depth_feasible(new_stack)
        )

    # -- transitions ---------------------------------------------------------

    def _apply_operator(self, spec: OperatorSpec, stack: tuple[StackItem, ...]) -> Node | None:
        """Return the node produced by ``spec`` on ``stack`` or ``None`` if illegal."""
        window: int | None = None
        items = list(stack)
        if spec.takes_window:
            pending = items.pop() if items else None
            if not isinstance(pending, int) or isinstance(pending, bool):
                return None
            window = pending
        elif items and isinstance(items[-1], int):
            return None
        if len(items) < spec.arity:
            return None
        operands = items[len(items) - spec.arity :]
        if not all(isinstance(o, Node) for o in operands):
            return None
        nodes = tuple(o for o in operands if isinstance(o, Node))
        try:
            node = Call(spec.name, nodes, window)
        except ExpressionTypeError:
            return None
        if self.grammar.forbid_degenerate and is_degenerate(node):
            return None
        return node if self._within_limits(node) else None

    def _within_limits(self, node: Node) -> bool:
        g = self.grammar
        if node.depth > g.max_depth or node.size > g.max_size or node.lookback > g.max_lookback:
            return False
        window = node.window if isinstance(node, Call) else None
        return window is None or window in g.windows_for(node.op)  # type: ignore[attr-defined]

    def _next_stack(self, state: BuilderState, action: int) -> tuple[StackItem, ...] | None:
        """Stack after ``action`` (ignoring length bounds), or ``None`` if ill-typed."""
        token = self.tokens[action]
        stack = state.stack
        top = stack[-1] if stack else None
        if token.kind is TokenKind.FEATURE:
            if top is not None and not _is_series(top):
                return None
            return (*stack, Feature(str(token.value)))
        if token.kind is TokenKind.CONSTANT:
            if not _is_series(top):
                return None
            constant = Constant(float(token.value))  # type: ignore[arg-type]
            pushed = (*stack, constant)
            return pushed if self._consumable(state, pushed, constant=True) else None
        if token.kind is TokenKind.WINDOW:
            if not _is_series(top):
                return None
            pushed_window = (*stack, int(token.value))  # type: ignore[arg-type]
            return pushed_window if self._consumable(state, pushed_window, constant=False) else None
        if token.kind is TokenKind.OPERATOR:
            spec = self.grammar.spec(str(token.value))
            node = self._apply_operator(spec, stack)
            if node is None:
                return None
            consumed = spec.arity + (1 if spec.takes_window else 0)
            return (*stack[: len(stack) - consumed], node)
        return None

    def _consumable(
        self, state: BuilderState, pushed: tuple[StackItem, ...], *, constant: bool
    ) -> bool:
        """One-step lookahead for a pushed constant or window.

        The push is admissible only if some operator can consume it next and
        the resulting stack still fits the length, size and depth bounds.
        """
        used = len(state.tokens) + 2
        for op in self.grammar.operators:
            spec = self.grammar.spec(op)
            if (constant and not spec.allows_constant) or (not constant and not spec.takes_window):
                continue
            node = self._apply_operator(spec, pushed)
            if node is None:
                continue
            consumed = spec.arity + (1 if spec.takes_window else 0)
            after = (*pushed[: len(pushed) - consumed], node)
            if (
                used + self._tokens_needed(after) <= self.max_tokens
                and self._size_feasible(after)
                and self._depth_feasible(after)
            ):
                return True
        return False

    def _can_end(self, state: BuilderState) -> bool:
        if len(state.stack) != 1 or not _is_series(state.stack[0]):
            return False
        node = state.stack[0]
        assert isinstance(node, Node)
        if self.require_operator and not isinstance(node, Call):
            return False
        return self.grammar.is_valid(node)

    def action_mask(self, state: BuilderState) -> np.ndarray:
        """Boolean mask of legal actions in ``state`` (all ``False`` when done)."""
        mask = np.zeros(self.n_actions, dtype=bool)
        if state.done or len(state.tokens) >= self.max_tokens:
            return mask
        for action in range(self.n_actions - 1):
            new_stack = self._next_stack(state, action)
            mask[action] = new_stack is not None and self._fits(state, new_stack)
        mask[self.end_action] = self._can_end(state)
        return mask

    def step(self, state: BuilderState, action: int) -> BuilderState:
        """Apply a legal action.

        Raises:
            InvalidActionError: ``action`` is masked out in ``state``.
        """
        if not 0 <= action < self.n_actions or not self.action_mask(state)[action]:
            raise InvalidActionError(f"action {action} is not legal in the current state")
        tokens = (*state.tokens, action)
        if action == self.end_action:
            return BuilderState(tokens, state.stack, done=True)
        new_stack = self._next_stack(state, action)
        assert new_stack is not None
        return BuilderState(tokens, new_stack)

    def expression(self, state: BuilderState) -> Node:
        """The finished expression of a terminal state."""
        if not state.done:
            raise ValueError("the expression is not finished")
        node = state.stack[0]
        assert isinstance(node, Node)
        return node

    def encode(self, node: Node) -> list[int]:
        """RPN action sequence (ending with END) that builds ``node``.

        Useful for tests and for seeding a policy with known expressions.

        Raises:
            ValueError: ``node`` puts a constant before a series in a
                non-commutative operator, which the builder cannot express.
        """
        index = {(t.kind, t.value): i for i, t in enumerate(self.tokens)}
        out: list[int] = []

        def emit(n: Node) -> None:
            if isinstance(n, Feature):
                out.append(index[(TokenKind.FEATURE, n.name)])
            elif isinstance(n, Constant):
                out.append(index[(TokenKind.CONSTANT, n.value)])
            else:
                assert isinstance(n, Call)
                args = list(n.args)
                if n.spec.commutative:
                    # Canonical order may put a constant first; RPN pushes it second.
                    args.sort(key=lambda a: isinstance(a, Constant))
                elif isinstance(args[0], Constant):
                    raise ValueError(
                        f"{n.canonical} has a leading constant; not expressible in RPN"
                    )
                for arg in args:
                    emit(arg)
                if n.window is not None:
                    out.append(index[(TokenKind.WINDOW, n.window)])
                out.append(index[(TokenKind.OPERATOR, n.op)])

        emit(node)
        out.append(self.end_action)
        return out
