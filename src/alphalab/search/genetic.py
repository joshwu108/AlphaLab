"""Genetic programming over typed expression trees.

A generational GP in the style of Koza (1992), adapted to the typed grammar:

1. **Initialization** -- ramped sampling from the grammar, deduplicated.
2. **Selection** -- tournament selection on *adjusted* fitness, i.e. train
   fitness minus an optional diversity penalty (correlation with the current
   elites), so the population is not dominated by copies of one signal.
3. **Variation** -- subtree crossover, subtree mutation, point mutation
   (feature, window, constant or same-signature operator) and hoist mutation.
   Only series-valued subtrees are exchanged, so offspring stay well-typed;
   offspring that break grammar limits are retried and finally replaced by a
   copy of the parent.
4. **Elitism** -- the best unique individuals survive unchanged.

Evaluation is memoized by canonical key; re-encountering a known expression is
free under the shared budget. A JSON checkpoint written after each generation
contains the population, generator state, history and budget, so a run can be
resumed after interruption and continue exactly where it stopped.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from alphalab.errors import (
    BudgetExhaustedError,
    CheckpointError,
    ExpressionLimitError,
    ExpressionTypeError,
)
from alphalab.evaluation.budget import EvaluationBudget
from alphalab.evaluation.evaluator import FactorEvaluator
from alphalab.evaluation.export import to_json_text
from alphalab.evaluation.selection import DiversityPenalty
from alphalab.expressions.grammar import Grammar, sample_expression
from alphalab.expressions.nodes import Call, Constant, Feature, Node, ValueKind
from alphalab.expressions.parser import parse
from alphalab.search.base import (
    CHECKPOINT_FORMAT,
    SearchHistory,
    SearchResult,
    atomic_write_json,
    read_checkpoint,
    rng_from_state,
    rng_state,
)

MAX_VARIATION_ATTEMPTS = 20


@dataclass(frozen=True)
class GPSettings:
    """Genetic-programming hyperparameters.

    Attributes:
        population_size: Individuals per generation.
        tournament_size: Contestants per tournament.
        crossover_rate: Probability an offspring is produced by crossover.
        mutation_rate: Probability an offspring (after optional crossover) is mutated.
        elitism: Best unique individuals copied unchanged to the next generation.
        diversity_weight: Weight of the correlation-with-elites penalty in
            tournament fitness; 0 disables it.
        max_generations: Optional hard limit; the budget normally stops the run.
        stall_generations: Stop after this many consecutive generations that
            charge no new candidate. Without it a converged population whose
            offspring are all known (and therefore free) would never exhaust
            the budget.
        seed: Seed for all random choices.
        top_k: Number of best records kept in the result.
    """

    population_size: int = 64
    tournament_size: int = 3
    crossover_rate: float = 0.7
    mutation_rate: float = 0.3
    elitism: int = 2
    diversity_weight: float = 0.0
    max_generations: int | None = None
    stall_generations: int = 20
    seed: int = 0
    top_k: int = 20

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if not 1 <= self.tournament_size <= self.population_size:
            raise ValueError("tournament_size must lie in [1, population_size]")
        if not (0 <= self.crossover_rate <= 1 and 0 <= self.mutation_rate <= 1):
            raise ValueError("rates must lie in [0, 1]")
        if not 0 <= self.elitism < self.population_size:
            raise ValueError("elitism must lie in [0, population_size)")
        if self.diversity_weight < 0:
            raise ValueError("diversity_weight must be non-negative")
        if self.stall_generations < 1:
            raise ValueError("stall_generations must be at least 1")


def _series_paths(node: Node) -> list[tuple[int, ...]]:
    return [path for path, sub in node.walk() if sub.kind is ValueKind.SERIES]


class Variation:
    """Typed variation operators bound to a grammar and a generator."""

    def __init__(self, grammar: Grammar, rng: np.random.Generator) -> None:
        self.grammar = grammar
        self.rng = rng

    def _choice(self, items: list[Any] | tuple[Any, ...]) -> Any:
        return items[int(self.rng.integers(len(items)))]

    def _accept(self, node: Node) -> bool:
        return isinstance(node, Call) and self.grammar.is_valid(node)

    def crossover(self, a: Node, b: Node) -> Node:
        """Replace a random series subtree of ``a`` with one from ``b``."""
        for _ in range(MAX_VARIATION_ATTEMPTS):
            path = self._choice(_series_paths(a))
            donor = b.subtree(self._choice(_series_paths(b)))
            try:
                child = a.replace(path, donor)
            except ExpressionTypeError:
                continue
            if self._accept(child) and child != a:
                return child
        return a

    def subtree_mutation(self, node: Node) -> Node:
        """Replace a random series subtree with a freshly sampled tree."""
        for _ in range(MAX_VARIATION_ATTEMPTS):
            path = self._choice(_series_paths(node))
            depth_left = max(1, self.grammar.max_depth - len(path))
            try:
                fresh = sample_expression(
                    self.grammar, self.rng, max_depth=depth_left, require_operator=False
                )
                child = node.replace(path, fresh)
            except (ExpressionTypeError, ExpressionLimitError):
                continue
            if self._accept(child) and child != node:
                return child
        return node

    def point_mutation(self, node: Node) -> Node:
        """Change one feature, window, constant or operator in place."""
        for _ in range(MAX_VARIATION_ATTEMPTS):
            path, target = self._choice(list(node.walk()))
            replacement = self._mutate_point(target)
            if replacement is None:
                continue
            try:
                child = node.replace(path, replacement)
            except ExpressionTypeError:
                continue
            if self._accept(child) and child != node:
                return child
        return node

    def _mutate_point(self, target: Node) -> Node | None:
        grammar = self.grammar
        if isinstance(target, Feature):
            return Feature(self._choice(grammar.features))
        if isinstance(target, Constant):
            return Constant(self._choice(grammar.constants)) if grammar.constants else None
        assert isinstance(target, Call)
        spec = target.spec
        if spec.takes_window and self.rng.random() < 0.5:
            return Call(target.op, target.args, self._choice(grammar.windows_for(target.op)))
        compatible = [
            name
            for name in grammar.operators
            if (other := grammar.spec(name)).arity == spec.arity
            and other.takes_window == spec.takes_window
            and (other.allows_constant or all(a.kind is ValueKind.SERIES for a in target.args))
        ]
        op = self._choice(compatible)
        window = target.window
        if window is not None and window not in grammar.windows_for(op):
            window = self._choice(grammar.windows_for(op))
        try:
            return Call(op, target.args, window)
        except ExpressionTypeError:
            return None

    def hoist_mutation(self, node: Node) -> Node:
        """Replace the tree by one of its operator subtrees (reduces bloat)."""
        calls = [sub for path, sub in node.walk() if path and isinstance(sub, Call)]
        return self._choice(calls) if calls else node

    def mutate(self, node: Node) -> Node:
        """Apply one mutation chosen uniformly among subtree, point and hoist."""
        kind = int(self.rng.integers(3))
        if kind == 0:
            return self.subtree_mutation(node)
        if kind == 1:
            return self.point_mutation(node)
        return self.hoist_mutation(node)


class GeneticProgramming:
    """Generational GP with elitism, diversity penalty and checkpointing."""

    method = "gp"

    def __init__(
        self,
        grammar: Grammar,
        settings: GPSettings | None = None,
        *,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        self.grammar = grammar
        self.settings = settings or GPSettings()
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

    # -- evaluation helpers -------------------------------------------------

    def _score(
        self, evaluator: FactorEvaluator, node: Node, history: SearchHistory, gen: int, src: str
    ) -> float:
        budget = evaluator.budget
        assert budget is not None
        before = budget.used
        score = evaluator.score(node)
        if budget.used > before:
            history.add(score, source=src, iteration=gen)
        return score.fitness

    def _adjusted(
        self, evaluator: FactorEvaluator, population: list[Node], fitness: list[float]
    ) -> list[float]:
        weight = self.settings.diversity_weight
        if weight == 0:
            return list(fitness)
        order = np.argsort(-np.asarray(fitness), kind="stable")
        elites = [population[i] for i in order[: max(1, self.settings.elitism)]]
        elite_values = {e.key: evaluator.factor_values(e, "train") for e in elites}
        penalty = DiversityPenalty(weight)
        adjusted = []
        for node, fit in zip(population, fitness, strict=True):
            refs = [v for k, v in elite_values.items() if k != node.key]
            adjusted.append(fit - penalty.against(evaluator.factor_values(node, "train"), refs))
        return adjusted

    def _tournament(self, rng: np.random.Generator, adjusted: list[float]) -> int:
        contestants = rng.choice(len(adjusted), size=self.settings.tournament_size, replace=False)
        return int(max(contestants, key=lambda i: (adjusted[i], -i)))

    # -- checkpointing -----------------------------------------------------

    def _identity(self, evaluator: FactorEvaluator) -> dict[str, Any]:
        return {
            "panel": evaluator.panel.fingerprint,
            "split": evaluator.split.to_dict(),
            # max_generations is excluded so a run stopped by a generation
            # limit can be resumed with a larger (or no) limit.
            "settings": {k: v for k, v in asdict(self.settings).items() if k != "max_generations"},
            "grammar": self.grammar.to_dict(),
        }

    def _save(
        self,
        evaluator: FactorEvaluator,
        generation: int,
        population: list[Node],
        rng: np.random.Generator,
        history: SearchHistory,
        stall: int = 0,
    ) -> None:
        if self.checkpoint_path is None or evaluator.budget is None:
            return
        atomic_write_json(
            {
                "format": CHECKPOINT_FORMAT,
                "method": self.method,
                "identity": self._identity(evaluator),
                "generation": generation,
                "stall": stall,
                "population": [n.canonical for n in population],
                "rng": rng_state(rng),
                "history": history.to_list(),
                "budget": evaluator.budget.state(),
            },
            self.checkpoint_path,
        )

    def _load(
        self, evaluator: FactorEvaluator
    ) -> tuple[int, list[Node], np.random.Generator, SearchHistory, int]:
        assert self.checkpoint_path is not None
        data = read_checkpoint(self.checkpoint_path, method=self.method)
        # Compare through a JSON round trip: tuples in the live identity become lists on disk.
        if data["identity"] != json.loads(to_json_text(self._identity(evaluator))):
            raise CheckpointError(
                "checkpoint was written for different data, split, settings or grammar"
            )
        evaluator.budget = EvaluationBudget.from_state(data["budget"])
        population = [parse(text) for text in data["population"]]
        return (
            int(data["generation"]),
            population,
            rng_from_state(data["rng"]),
            SearchHistory.from_list(data["history"]),
            int(data.get("stall", 0)),
        )

    # -- main loop -----------------------------------------------------------

    def _initial_population(self, rng: np.random.Generator) -> list[Node]:
        population: list[Node] = []
        keys: set[str] = set()
        attempts = 0
        while len(population) < self.settings.population_size:
            # Ramped depth: cycle through 2..max_depth for structural variety.
            depth = 2 + attempts % max(1, self.grammar.max_depth - 1)
            node = sample_expression(self.grammar, rng, max_depth=depth)
            attempts += 1
            if node.key not in keys or attempts > 50 * self.settings.population_size:
                keys.add(node.key)
                population.append(node)
        return population

    def _next_generation(
        self,
        evaluator: FactorEvaluator,
        population: list[Node],
        fitness: list[float],
        rng: np.random.Generator,
    ) -> list[Node]:
        settings = self.settings
        variation = Variation(self.grammar, rng)
        adjusted = self._adjusted(evaluator, population, fitness)
        order = np.argsort(-np.asarray(fitness), kind="stable")
        offspring: list[Node] = []
        seen: set[str] = set()
        for index in order:
            if len(offspring) >= settings.elitism:
                break
            if population[index].key not in seen:
                offspring.append(population[index])
                seen.add(population[index].key)
        while len(offspring) < settings.population_size:
            parent = population[self._tournament(rng, adjusted)]
            child = parent
            if rng.random() < settings.crossover_rate:
                other = population[self._tournament(rng, adjusted)]
                child = variation.crossover(parent, other)
            if rng.random() < settings.mutation_rate or child == parent:
                child = variation.mutate(child)
            offspring.append(child)
        return offspring

    def run(self, evaluator: FactorEvaluator, *, resume: bool = False) -> SearchResult:
        """Evolve until the budget (or ``max_generations``) is exhausted.

        Args:
            evaluator: Scoring gateway with a budget.
            resume: Continue from ``checkpoint_path`` instead of starting fresh.
        """
        if evaluator.budget is None:
            raise ValueError("genetic programming requires an evaluator with a budget")
        if resume:
            if self.checkpoint_path is None:
                raise CheckpointError("resume requested but no checkpoint_path is configured")
            generation, population, rng, history, stall = self._load(evaluator)
        else:
            rng = np.random.default_rng(self.settings.seed)
            history = SearchHistory()
            population = self._initial_population(rng)
            generation = 0
            stall = 0
        stopped = "budget exhausted"
        while True:
            source = "init" if generation == 0 else "offspring"
            fitness: list[float] = []
            charged_before = evaluator.budget.used
            try:
                for node in population:
                    fitness.append(self._score(evaluator, node, history, generation, source))
            except BudgetExhaustedError:
                break
            generation += 1
            stall = 0 if evaluator.budget.used > charged_before else stall + 1
            population = self._next_generation(evaluator, population, fitness, rng)
            # The checkpoint holds the *unevaluated* next generation and the
            # generator state after producing it, so resuming replays exactly
            # the computation an uninterrupted run would perform.
            self._save(evaluator, generation, population, rng, history, stall)
            if stall >= self.settings.stall_generations:
                stopped = f"stalled: no new candidate in {stall} generations"
                break
            limit = self.settings.max_generations
            if limit is not None and generation >= limit:
                stopped = "max_generations reached"
                break
            if evaluator.budget.exhausted:
                break
        assert evaluator.budget is not None
        return SearchResult(
            method=self.method,
            history=history,
            best=history.top(self.settings.top_k),
            budget=evaluator.budget.to_dict(),
            stopped_reason=stopped,
            extra={"generations": generation, "settings": asdict(self.settings)},
        )
