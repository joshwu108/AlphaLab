"""Common search interfaces: history records, results and checkpoint I/O.

Every search method consumes a :class:`~alphalab.evaluation.FactorEvaluator`
whose budget is the only stopping criterion shared by all methods, records
each charged candidate in a :class:`SearchHistory`, and returns a
:class:`SearchResult`. Methods never touch validation or test data; final
selection on the validation partition happens in
:func:`select_on_validation`, identically for every method.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from alphalab.errors import CheckpointError
from alphalab.evaluation.evaluator import FactorEvaluator, FactorScore
from alphalab.evaluation.export import to_json_text
from alphalab.evaluation.selection import Candidate, select_diverse

CHECKPOINT_FORMAT = 1


@dataclass(frozen=True)
class HistoryRecord:
    """One charged evaluation."""

    index: int
    expression: str
    key: str
    fitness: float
    valid: bool
    rank_ic: float
    size: int
    source: str  # e.g. "random", "init", "mutation", "crossover", "policy"
    iteration: int


@dataclass
class SearchHistory:
    """Ordered record of charged evaluations; the basis for budget-matched comparisons."""

    records: list[HistoryRecord] = field(default_factory=list)

    def add(self, score: FactorScore, *, source: str, iteration: int) -> HistoryRecord:
        """Append a record for a newly charged train score."""
        record = HistoryRecord(
            index=len(self.records),
            expression=score.expression,
            key=score.key,
            fitness=score.fitness,
            valid=score.valid,
            rank_ic=score.rank_ic_mean,
            size=score.size,
            source=source,
            iteration=iteration,
        )
        self.records.append(record)
        return record

    def best_so_far(self) -> np.ndarray:
        """Running maximum fitness after each charged evaluation (an anytime curve)."""
        fitness = np.array([r.fitness for r in self.records], dtype=np.float64)
        return np.maximum.accumulate(fitness) if fitness.size else fitness

    def top(self, k: int) -> list[HistoryRecord]:
        """The ``k`` best valid unique records by fitness (ties broken by expression)."""
        best: dict[str, HistoryRecord] = {}
        for record in self.records:
            if record.valid and (
                record.key not in best or record.fitness > best[record.key].fitness
            ):
                best[record.key] = record
        return sorted(best.values(), key=lambda r: (-r.fitness, r.expression))[:k]

    def to_list(self) -> list[dict[str, Any]]:
        """Plain records for export."""
        return [asdict(r) for r in self.records]

    @classmethod
    def from_list(cls, rows: Iterable[dict[str, Any]]) -> SearchHistory:
        """Inverse of :meth:`to_list`."""
        return cls([HistoryRecord(**row) for row in rows])


@dataclass
class SearchResult:
    """Outcome of one search run."""

    method: str
    history: SearchHistory
    best: list[HistoryRecord]
    budget: dict[str, int]
    stopped_reason: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Summary without the full history."""
        return {
            "method": self.method,
            "budget": self.budget,
            "stopped_reason": self.stopped_reason,
            "n_records": len(self.history.records),
            "best": [asdict(r) for r in self.best],
            "extra": self.extra,
        }


class Searcher(Protocol):
    """Interface implemented by every search method."""

    method: str

    def run(self, evaluator: FactorEvaluator) -> SearchResult:
        """Search until the evaluator's budget is exhausted (or the method stops)."""
        ...


def select_on_validation(
    evaluator: FactorEvaluator,
    records: Iterable[HistoryRecord],
    *,
    max_factors: int,
    max_abs_correlation: float = 0.7,
) -> list[tuple[FactorScore, FactorScore]]:
    """Choose a diverse factor set by **validation** fitness.

    Validation scores apply the sign learned on train and are never charged.
    Redundancy is measured on train values so that the selection rule itself
    does not consume more validation information than the ranking.

    Returns:
        ``(train_score, valid_score)`` pairs in selection order.
    """
    pairs: dict[str, tuple[FactorScore, FactorScore]] = {}
    candidates = []
    for record in records:
        if record.expression in pairs:
            continue
        train = evaluator.score(record.expression, "train")
        valid = evaluator.score(record.expression, "valid")
        if not valid.valid:
            continue
        pairs[record.expression] = (train, valid)
        values = evaluator.factor_values(record.expression, "train")
        candidates.append(Candidate(record.expression, valid.fitness, values))
    kept = select_diverse(
        candidates, max_factors=max_factors, max_abs_correlation=max_abs_correlation
    )
    return [pairs[c.expression] for c in kept]


def atomic_write_json(payload: Any, path: str | Path) -> Path:
    """Write JSON via a temporary file and rename, so a crash never leaves a torn checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(to_json_text(payload) + "\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def read_checkpoint(path: str | Path, *, method: str) -> dict[str, Any]:
    """Load and sanity-check a JSON search checkpoint.

    Raises:
        CheckpointError: Missing file, invalid JSON, wrong format or method.
    """
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"checkpoint not found: {path}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise CheckpointError(f"checkpoint {path} is not valid JSON: {exc}") from exc
    if data.get("format") != CHECKPOINT_FORMAT:
        raise CheckpointError(f"unsupported checkpoint format {data.get('format')!r}")
    if data.get("method") != method:
        raise CheckpointError(f"checkpoint is for {data.get('method')!r}, not {method!r}")
    return dict(data)


def rng_state(rng: np.random.Generator) -> dict[str, Any]:
    """JSON-serializable state of a NumPy generator."""
    return dict(rng.bit_generator.state)


def rng_from_state(state: dict[str, Any]) -> np.random.Generator:
    """Generator restored from :func:`rng_state`."""
    bit_generator = getattr(np.random, state["bit_generator"])()
    bit_generator.state = state
    return np.random.Generator(bit_generator)
