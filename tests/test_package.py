"""Package-level smoke tests."""

from __future__ import annotations

import alphalab
from alphalab import errors


def test_version_is_exposed() -> None:
    assert alphalab.__version__ == "0.1.0"


def test_every_library_error_derives_from_the_common_base() -> None:
    subclasses = [
        errors.DataValidationError,
        errors.ExpressionTypeError,
        errors.ExpressionParseError,
        errors.ExpressionLimitError,
        errors.PartitionAccessError,
        errors.BudgetExhaustedError,
        errors.ConfigError,
        errors.CheckpointError,
    ]
    assert all(issubclass(cls, errors.AlphaLabError) for cls in subclasses)
