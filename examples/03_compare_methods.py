"""Run random search, GP and (if torch is installed) RL on the same tiny budget and compare.

All runs share data, split, grammar, scoring and budget, so the comparison
table reports "Comparability: OK". The budget is deliberately tiny; the
output demonstrates the workflow and is not a research result.

Run: python examples/03_compare_methods.py [output_dir]
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

from alphalab.experiments import config_from_dict, run_experiment
from alphalab.reporting import load_run, render_comparison

BASE = {
    "schema_version": 1,
    "seed": 0,
    "data": {"synthetic": {"n_periods": 300, "n_instruments": 25, "signal_strength": 0.4}},
    "split": {"horizon": 1, "embargo": 1},
    "grammar": {"max_depth": 4, "max_size": 9, "max_lookback": 30},
    "budget": {"max_evaluations": 60},
    "selection": {"max_factors": 3, "candidates": 15},
    "gp": {"population_size": 16},
    "rl": {"max_tokens": 19, "episodes_per_update": 8, "hidden_dim": 32},
}


def main(output: Path) -> None:
    methods = ["random", "gp"]
    if importlib.util.find_spec("torch") is not None:
        methods.append("rl")
    run_dirs = []
    for method in methods:
        config = config_from_dict({**BASE, "name": f"example-{method}", "method": method})
        outcome = run_experiment(config, output / method)
        run_dirs.append(outcome.run_dir)
        print(f"{method}: {outcome.status} -> {outcome.run_dir}")
    print(render_comparison([load_run(d) for d in run_dirs]))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(Path(sys.argv[1]))
    else:
        with tempfile.TemporaryDirectory() as tmp:
            main(Path(tmp))
