"""AlphaLab: symbolic alpha-factor discovery with search and reinforcement learning.

The package is organised by research stage:

* :mod:`alphalab.data` -- market-data contract, validation, panels, labels, splits.
* :mod:`alphalab.expressions` -- typed factor-expression language and CPU evaluator.
* :mod:`alphalab.evaluation` -- information coefficients, diversity, budgets, selection.
* :mod:`alphalab.search` -- random search and genetic programming baselines.
* :mod:`alphalab.rl` -- grammar-constrained expression-building environment and policy.
* :mod:`alphalab.experiments` -- versioned configurations, runs, manifests, checkpoints.
* :mod:`alphalab.reporting` -- run summaries and cross-run comparisons.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
