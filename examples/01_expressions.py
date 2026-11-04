"""Build a synthetic panel, parse expressions, evaluate them and inspect the cache.

Run: python examples/01_expressions.py
"""

from __future__ import annotations

import numpy as np

from alphalab.data import SyntheticSpec, make_synthetic_panel
from alphalab.expressions import ExpressionCache, Grammar, evaluate, parse, sample_expression


def main() -> None:
    spec = SyntheticSpec(n_periods=120, n_instruments=10, seed=0)
    panel = make_synthetic_panel(spec)
    print(f"panel {panel.shape} fingerprint {panel.fingerprint[:12]}")

    node = parse("ts_corr($close, ts_mean($volume, 5), 10)")
    print(f"expression {node.canonical}")
    print(f"  depth={node.depth} size={node.size} lookback={node.lookback} key={node.key[:12]}")

    cache = ExpressionCache()
    values = evaluate(node, panel, cache)
    first_valid = int(np.argmax(np.isfinite(values).any(axis=1)))
    print(f"  first row with a value: {first_valid} (equals the lookback)")

    evaluate("sub(ts_mean($volume, 5), $volume)", panel, cache)  # reuses ts_mean($volume, 5)
    print(f"cache: {cache.stats}")

    rng = np.random.default_rng(1)
    grammar = Grammar(max_depth=4, max_size=9)
    for _ in range(3):
        print(f"sampled: {sample_expression(grammar, rng).canonical}")


if __name__ == "__main__":
    main()
