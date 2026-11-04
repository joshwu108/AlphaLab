"""Exception hierarchy shared by every AlphaLab module.

All library errors derive from :class:`AlphaLabError` so callers can catch the
whole family without also catching unrelated ``ValueError`` instances raised by
NumPy or pandas.
"""

from __future__ import annotations


class AlphaLabError(Exception):
    """Base class for all errors raised deliberately by AlphaLab."""


class DataValidationError(AlphaLabError):
    """Input market data violates the normalized data contract."""


class ExpressionError(AlphaLabError):
    """Base class for factor-expression construction and evaluation errors."""


class ExpressionTypeError(ExpressionError):
    """An operator received operands of the wrong type or arity."""


class ExpressionParseError(ExpressionError):
    """A serialized expression could not be parsed."""


class ExpressionLimitError(ExpressionError):
    """An expression exceeds a configured depth, size or lookback limit."""


class PartitionAccessError(AlphaLabError):
    """Code attempted to read a data partition it is not permitted to read.

    Raised, for example, when a search algorithm requests scores on the locked
    test partition.
    """


class BudgetExhaustedError(AlphaLabError):
    """The evaluation budget of a search run has been fully consumed."""


class ConfigError(AlphaLabError):
    """An experiment configuration is malformed or internally inconsistent."""


class CheckpointError(AlphaLabError):
    """A checkpoint is missing, corrupt, or incompatible with the current run."""
