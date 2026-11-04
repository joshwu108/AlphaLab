"""Micro-benchmark of CPU expression evaluation with and without the sub-expression cache.

Measures wall-clock time to evaluate a fixed set of sampled expressions on a
synthetic panel. Results depend on hardware and are printed, never stored;
the repository contains no benchmark numbers.

Run: python benchmarks/bench_evaluation.py [--periods 1000 --instruments 500 --expressions 200]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from alphalab.data import SyntheticSpec, make_synthetic_panel
from alphalab.expressions import ExpressionCache, Grammar, evaluate, sample_expression


def run(periods: int, instruments: int, expressions: int, repeats: int) -> dict[str, float]:
    """Return median seconds for uncached and cached passes over the expression set."""
    panel = make_synthetic_panel(SyntheticSpec(n_periods=periods, n_instruments=instruments))
    rng = np.random.default_rng(0)
    grammar = Grammar(max_depth=5, max_size=12, max_lookback=60)
    nodes = [sample_expression(grammar, rng) for _ in range(expressions)]
    uncached, cached = [], []
    for _ in range(repeats):
        start = time.perf_counter()
        for node in nodes:
            evaluate(node, panel)
        uncached.append(time.perf_counter() - start)
        cache = ExpressionCache()
        start = time.perf_counter()
        for node in nodes:
            evaluate(node, panel, cache)
        cached.append(time.perf_counter() - start)
        hit_rate = cache.stats.hit_rate
    return {
        "uncached_s": float(np.median(uncached)),
        "cached_first_pass_s": float(np.median(cached)),
        "cache_hit_rate": hit_rate,
        "cells": float(periods * instruments),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--periods", type=int, default=1000)
    parser.add_argument("--instruments", type=int, default=500)
    parser.add_argument("--expressions", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    result = run(args.periods, args.instruments, args.expressions, args.repeats)
    per_expr = result["uncached_s"] / args.expressions * 1e3
    print(f"panel {args.periods}x{args.instruments}, {args.expressions} expressions")
    print(f"uncached: {result['uncached_s']:.3f}s ({per_expr:.2f} ms/expression)")
    print(f"cached first pass: {result['cached_first_pass_s']:.3f}s")
    print(f"cache hit rate: {result['cache_hit_rate']:.2%}")


if __name__ == "__main__":
    main()
