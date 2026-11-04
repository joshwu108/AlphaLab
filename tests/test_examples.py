"""Keep the examples and benchmark runnable."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = sorted((ROOT / "examples").glob("*.py"))


@pytest.mark.slow
@pytest.mark.parametrize("script", EXAMPLES, ids=[p.name for p in EXAMPLES])
def test_example_runs(script: Path, tmp_path: Path) -> None:
    args = [sys.executable, str(script)]
    if script.name.startswith("03_"):
        args.append(str(tmp_path))
    completed = subprocess.run(args, capture_output=True, text=True, timeout=300, check=False)
    assert completed.returncode == 0, completed.stderr


def test_benchmark_quick_mode() -> None:
    sys.path.insert(0, str(ROOT / "benchmarks"))
    try:
        import bench_evaluation
    finally:
        sys.path.pop(0)
    result = bench_evaluation.run(periods=60, instruments=8, expressions=5, repeats=1)
    assert result["uncached_s"] > 0 and 0 <= result["cache_hit_rate"] <= 1
