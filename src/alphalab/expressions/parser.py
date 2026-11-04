"""Parser for the canonical expression syntax.

Grammar::

    expr     := feature | number | name "(" expr ("," expr)* ["," integer] ")"
    feature  := "$" identifier
    number   := optionally signed decimal or scientific literal

For operators that take a window, the final argument must be an integer
literal and is parsed as the window; every other numeric argument is a
:class:`~alphalab.expressions.nodes.Constant`. Whitespace is insignificant.
``parse(node.canonical) == node`` holds for every valid node.
"""

from __future__ import annotations

import re

from alphalab.errors import ExpressionParseError, ExpressionTypeError
from alphalab.expressions.nodes import Call, Constant, Feature, Node
from alphalab.expressions.operators import OPERATORS

_TOKEN = re.compile(
    r"\s*(?:(?P<feature>\$[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<number>[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)"
    r"|(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<punct>[(),]))"
)
_INTEGER = re.compile(r"[-+]?\d+")

MAX_EXPRESSION_CHARS = 10_000  # Guards against pathological inputs from configs or files.


def _tokenize(text: str) -> list[tuple[str, str, int]]:
    tokens: list[tuple[str, str, int]] = []
    position = 0
    while position < len(text):
        if text[position:].strip() == "":
            break
        match = _TOKEN.match(text, position)
        if match is None or match.end() == position:
            raise ExpressionParseError(
                f"unexpected character at {position}: {text[position:][:20]!r}"
            )
        kind = match.lastgroup
        assert kind is not None
        tokens.append((kind, match.group(kind), match.start(kind)))
        position = match.end()
    return tokens


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.tokens = _tokenize(text)
        self.index = 0

    def peek(self) -> tuple[str, str, int] | None:
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def take(self, expected: str | None = None) -> tuple[str, str, int]:
        token = self.peek()
        if token is None:
            raise ExpressionParseError(f"unexpected end of expression in {self.text!r}")
        if expected is not None and token[1] != expected:
            raise ExpressionParseError(f"expected {expected!r} at {token[2]}, found {token[1]!r}")
        self.index += 1
        return token

    def parse_argument(self) -> Node | int:
        kind, value, position = self.take()
        if kind == "feature":
            return Feature(value[1:])
        if kind == "number":
            if _INTEGER.fullmatch(value):
                return int(value)
            return Constant(float(value))
        if kind == "name":
            return self.parse_call(value, position)
        raise ExpressionParseError(f"unexpected {value!r} at {position}")

    def parse_call(self, name: str, position: int) -> Node:
        if name not in OPERATORS:
            raise ExpressionParseError(f"unknown operator {name!r} at {position}")
        spec = OPERATORS[name]
        self.take("(")
        raw: list[Node | int] = [self.parse_argument()]
        while (token := self.peek()) is not None and token[1] == ",":
            self.take(",")
            raw.append(self.parse_argument())
        self.take(")")
        window: int | None = None
        if spec.takes_window:
            last = raw.pop() if raw else None
            if not isinstance(last, int):
                raise ExpressionParseError(f"{name} at {position} needs an integer window last")
            window = last
        args = tuple(Constant(float(a)) if isinstance(a, int) else a for a in raw)
        try:
            return Call(name, args, window)
        except ExpressionTypeError as exc:
            raise ExpressionParseError(f"{exc} (at {position})") from exc

    def parse(self) -> Node:
        node = self.parse_argument()
        if self.peek() is not None:
            _, value, position = self.take()
            raise ExpressionParseError(f"trailing input {value!r} at {position}")
        if isinstance(node, int):
            node = Constant(float(node))
        return node


def parse(text: str) -> Node:
    """Parse ``text`` into a node.

    Raises:
        ExpressionParseError: Syntax error, unknown operator, or a type error
            in the parsed tree (reported with its position).
    """
    if len(text) > MAX_EXPRESSION_CHARS:
        raise ExpressionParseError(f"expression longer than {MAX_EXPRESSION_CHARS} characters")
    return _Parser(text).parse()
