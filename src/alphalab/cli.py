"""Command-line interface: ``alphalab <command> ...``.

Commands
--------
``operators``        List the expression operators.
``synth``            Write a deterministic synthetic dataset (CSV or Parquet).
``validate-data``    Validate a market-data file and print the report.
``check-config``     Validate an experiment configuration and print its hash.
``eval-expr``        Score one expression on the train and validation partitions.
``run``              Execute (or dry-run) an experiment configuration.
``report``           Summarize a completed run.
``compare``          Compare completed runs and check their comparability.

Exit status is 0 on success, 2 for invalid input (bad configuration, data or
expression) and 1 for unexpected failures.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from alphalab import __version__
from alphalab.errors import AlphaLabError
from alphalab.evaluation.export import to_json_text

EXIT_USAGE = 2


def _cmd_operators(args: argparse.Namespace) -> int:
    from alphalab.expressions.operators import OPERATORS

    for spec in OPERATORS.values():
        window = f", window>={spec.min_window}" if spec.takes_window else ""
        print(f"{spec.name:<10} {spec.category:<15} arity={spec.arity}{window}  {spec.description}")
    return 0


def _cmd_synth(args: argparse.Namespace) -> int:
    from alphalab.data.io import write_market_frame
    from alphalab.data.synthetic import SyntheticSpec, make_synthetic_frame

    spec = SyntheticSpec(
        n_periods=args.periods,
        n_instruments=args.instruments,
        seed=args.seed,
        signal_strength=args.signal,
        missing_rate=args.missing_rate,
    )
    path = write_market_frame(make_synthetic_frame(spec), args.out)
    print(f"wrote {path} ({spec.n_periods} sessions x {spec.n_instruments} instruments)")
    print(f"planted factor: {spec.planted_expression}")
    return 0


def _parse_column_map(pairs: Sequence[str]) -> dict[str, str]:
    mapping = {}
    for pair in pairs:
        source, sep, target = pair.partition("=")
        if not sep or not source or not target:
            raise AlphaLabError(f"--column-map expects VENDOR=CONTRACT, got {pair!r}")
        mapping[source] = target
    return mapping


def _cmd_validate_data(args: argparse.Namespace) -> int:
    from alphalab.data.contract import DataContract
    from alphalab.data.io import load_market_data
    from alphalab.data.panel import panel_from_frame

    contract = DataContract(duplicate_policy=args.duplicates)
    frame, report = load_market_data(args.path, contract, _parse_column_map(args.column_map))
    panel = panel_from_frame(frame, metadata={"source": "file", "path": str(args.path)})
    print(to_json_text({"report": report.__dict__, "fingerprint": panel.fingerprint}))
    return 0


def _cmd_check_config(args: argparse.Namespace) -> int:
    from alphalab.experiments.config import load_config

    config = load_config(args.config)
    if config.method == "rl":
        try:
            config.rl_settings()
        except ImportError:
            print("note: torch is not installed; RL settings were checked by name only")
    print(to_json_text({"config_hash": config.config_hash(), "config": config.to_dict()}))
    return 0


def _cmd_eval_expr(args: argparse.Namespace) -> int:
    from alphalab.evaluation.budget import EvaluationBudget
    from alphalab.experiments.config import load_config
    from alphalab.experiments.runner import build_evaluator

    config = load_config(args.config)
    evaluator, _ = build_evaluator(config)
    evaluator.budget = EvaluationBudget(1)
    evaluator.grammar = None if args.ignore_grammar else evaluator.grammar
    scores = {p: evaluator.score(args.expression, p).to_dict() for p in ("train", "valid")}
    print(to_json_text(scores))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from alphalab.experiments.config import load_config
    from alphalab.experiments.runner import run_experiment

    config = load_config(args.config)
    run_dir = (
        Path(args.output)
        if args.output
        else Path("runs") / f"{config.name}-{config.config_hash()[:8]}"
    )
    outcome = run_experiment(config, run_dir, dry_run=args.dry_run, resume=args.resume)
    print(f"{outcome.status}: {outcome.run_dir}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from alphalab.reporting.summary import load_run, render_run_report, run_summary

    record = load_run(args.run_dir)
    if args.format == "json":
        print(json.dumps(run_summary(record), indent=2))
    else:
        print(render_run_report(record, rst=args.format == "rst"))
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    from alphalab.reporting.summary import load_run, render_comparison

    print(render_comparison([load_run(d) for d in args.run_dirs], fmt=args.format))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Argument parser for all subcommands."""
    parser = argparse.ArgumentParser(
        prog="alphalab", description="Symbolic alpha-factor discovery research tools."
    )
    parser.add_argument("--version", action="version", version=f"alphalab {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("operators", help="list expression operators").set_defaults(func=_cmd_operators)

    synth = sub.add_parser("synth", help="write a deterministic synthetic dataset")
    synth.add_argument("--out", required=True, help="output .csv or .parquet path")
    synth.add_argument("--periods", type=int, default=256)
    synth.add_argument("--instruments", type=int, default=24)
    synth.add_argument("--seed", type=int, default=0)
    synth.add_argument("--signal", type=float, default=0.15, help="planted signal strength")
    synth.add_argument("--missing-rate", type=float, default=0.0)
    synth.set_defaults(func=_cmd_synth)

    validate = sub.add_parser("validate-data", help="validate a market-data file")
    validate.add_argument("path")
    validate.add_argument(
        "--column-map", nargs="*", default=[], metavar="VENDOR=CONTRACT", help="rename columns"
    )
    validate.add_argument(
        "--duplicates", choices=("error", "keep_last", "keep_first"), default="error"
    )
    validate.set_defaults(func=_cmd_validate_data)

    check = sub.add_parser("check-config", help="validate an experiment configuration")
    check.add_argument("config")
    check.set_defaults(func=_cmd_check_config)

    evaluate = sub.add_parser(
        "eval-expr", help="score an expression on train and validation (never test)"
    )
    evaluate.add_argument("expression")
    evaluate.add_argument("--config", required=True, help="configuration providing data and split")
    evaluate.add_argument(
        "--ignore-grammar", action="store_true", help="skip grammar limits for this expression"
    )
    evaluate.set_defaults(func=_cmd_eval_expr)

    run = sub.add_parser("run", help="run an experiment")
    run.add_argument("config")
    run.add_argument("--output", help="run directory (default: runs/<name>-<hash>)")
    run.add_argument("--dry-run", action="store_true", help="validate end-to-end without search")
    run.add_argument("--resume", action="store_true", help="resume from checkpoints in --output")
    run.set_defaults(func=_cmd_run)

    report = sub.add_parser("report", help="summarize a completed run")
    report.add_argument("run_dir")
    report.add_argument("--format", choices=("text", "rst", "json"), default="text")
    report.set_defaults(func=_cmd_report)

    compare = sub.add_parser("compare", help="compare completed runs")
    compare.add_argument("run_dirs", nargs="+")
    compare.add_argument("--format", choices=("text", "rst", "csv", "json"), default="text")
    compare.set_defaults(func=_cmd_compare)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit status."""
    args = build_parser().parse_args(argv)
    try:
        status: int = args.func(args)
    except (AlphaLabError, FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    return status


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
