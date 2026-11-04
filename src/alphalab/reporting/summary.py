"""Run summaries and cross-run comparisons.

Reports read only the files written by :mod:`alphalab.experiments.runner`
(``manifest.json`` and ``result.json``). Comparison tables check that runs are
comparable -- same dataset fingerprint, partitions, grammar, scoring and
evaluation budget -- and list every mismatch, because a method comparison
across different budgets or data is not meaningful.

All numbers in a report are measurements of the referenced runs on their
data. Predictive metrics (IC) are not trading performance; see
:mod:`alphalab.evaluation.metrics`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alphalab.errors import AlphaLabError

COMPARABILITY_KEYS = (
    ("data", "fingerprint"),
    ("split",),
    ("config", "grammar"),
    ("config", "evaluation"),
    ("config", "budget", "max_evaluations"),
)


@dataclass(frozen=True)
class RunRecord:
    """Loaded run outputs."""

    run_dir: Path
    manifest: dict[str, Any]
    result: dict[str, Any]


def load_run(run_dir: str | Path) -> RunRecord:
    """Load ``manifest.json`` and ``result.json`` from a run directory.

    Raises:
        AlphaLabError: Missing files or a run that did not complete.
    """
    run_dir = Path(run_dir)
    manifest_path, result_path = run_dir / "manifest.json", run_dir / "result.json"
    if not manifest_path.is_file():
        raise AlphaLabError(f"{run_dir} has no manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "completed" or not result_path.is_file():
        raise AlphaLabError(f"{run_dir} is not a completed run (status {manifest.get('status')})")
    return RunRecord(run_dir, manifest, json.loads(result_path.read_text()))


def _get(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in path:
        value = value.get(key) if isinstance(value, dict) else None
    return value


def run_summary(record: RunRecord) -> dict[str, Any]:
    """Flat dictionary of the headline quantities of one run."""
    m, r = record.manifest, record.result
    search = r.get("search", {})
    selected = r.get("selected", [])
    # Orient each factor by the sign learned on train, so a strongly negative
    # (reversal-type) factor does not cancel a positive one in the average.
    valid_ric = [
        s["valid"]["direction"] * s["valid"]["rank_ic_mean"]
        for s in selected
        if s["valid"]["rank_ic_mean"] is not None
    ]
    anytime = np.asarray(r.get("anytime_best", []), dtype=np.float64)
    best = search.get("best", [])
    test = r.get("test")
    return {
        "run": record.run_dir.name,
        "name": m["config"]["name"],
        "method": m["config"]["method"],
        "seed": m["config"]["seed"],
        "budget_used": m.get("budget", {}).get("used"),
        "budget_max": m.get("budget", {}).get("max_evaluations"),
        "invalid_candidates": m.get("budget", {}).get("invalid_candidates"),
        "best_train_fitness": best[0]["fitness"] if best else None,
        "anytime_mean_best": float(anytime.mean()) if anytime.size else None,
        "n_selected": len(selected),
        "mean_selected_valid_oriented_rank_ic": float(np.mean(valid_ric)) if valid_ric else None,
        "combination_valid_rank_ic": r.get("combination", {}).get("valid_rank_ic"),
        "selected_train_redundancy": r.get("combination", {}).get("train_redundancy"),
        "test_evaluated": test is not None,
        "test_reads": _get(m, ("partition_access", "scores_by_partition", "test")),
        "stopped_reason": search.get("stopped_reason"),
    }


def comparability_issues(records: list[RunRecord]) -> list[str]:
    """Human-readable list of settings that differ between runs."""
    issues = []
    for path in COMPARABILITY_KEYS:
        values = {json.dumps(_get(r.manifest, path), sort_keys=True) for r in records}
        if len(values) > 1:
            issues.append(f"runs differ in {'.'.join(path)}")
    return issues


def comparison_frame(records: list[RunRecord]) -> pd.DataFrame:
    """One row per run with :func:`run_summary` columns."""
    return pd.DataFrame([run_summary(r) for r in records])


def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_table(frame: pd.DataFrame, *, rst: bool = False) -> str:
    """Fixed-width text table, or an RST simple table when ``rst`` is true."""
    columns = list(frame.columns)
    rows = [[_fmt(v) for v in row] for row in frame.itertuples(index=False)]
    widths = [
        max(len(c), *(len(r[i]) for r in rows)) if rows else len(c) for i, c in enumerate(columns)
    ]
    line = "  ".join(c.ljust(w) for c, w in zip(columns, widths, strict=True))
    body = ["  ".join(v.ljust(w) for v, w in zip(r, widths, strict=True)) for r in rows]
    if rst:
        rule = "  ".join("=" * w for w in widths)
        return "\n".join([rule, line, rule, *body, rule])
    return "\n".join([line, "  ".join("-" * w for w in widths), *body])


def render_run_report(record: RunRecord, *, rst: bool = False) -> str:
    """Readable report of one run: summary, selected factors and caveats."""
    summary = run_summary(record)
    title = f"Run {summary['run']} ({summary['method']})"
    lines = [title, ("=" if rst else "-") * len(title), ""]
    lines += [f"{k}: {_fmt(v)}" for k, v in summary.items()]
    selected = record.result.get("selected", [])
    if selected:
        frame = pd.DataFrame(
            [
                {
                    "expression": s["train"]["expression"],
                    "train_fitness": s["train"]["fitness"],
                    "valid_fitness": s["valid"]["fitness"],
                    "direction": s["train"]["direction"],
                    "valid_rank_ic": s["valid"]["rank_ic_mean"],
                    "valid_turnover": s["valid"]["turnover"],
                    "size": s["train"]["size"],
                }
                for s in selected
            ]
        )
        lines += ["", "Selected factors (ranked on validation):", "", render_table(frame, rst=rst)]
    lines += [
        "",
        "Caveat: IC measures cross-sectional predictive association before costs; it is "
        "not a measure of realizable trading profit.",
    ]
    if not summary["test_evaluated"]:
        lines.append("The test partition was not evaluated in this run.")
    return "\n".join(lines)


def render_comparison(records: list[RunRecord], *, fmt: str = "text") -> str:
    """Comparison table in ``text``, ``rst``, ``csv`` or ``json`` format."""
    frame = comparison_frame(records)
    issues = comparability_issues(records)
    if fmt == "csv":
        return frame.to_csv(index=False)
    if fmt == "json":
        return json.dumps(
            {"runs": json.loads(frame.to_json(orient="records")), "issues": issues}, indent=2
        )
    table = render_table(frame, rst=fmt == "rst")
    notes = ["", "Comparability: " + ("OK" if not issues else "; ".join(issues))]
    return table + "\n" + "\n".join(notes)
