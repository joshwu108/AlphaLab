AlphaLab
========

AlphaLab is a research framework for discovering *formulaic alpha factors*:
short symbolic expressions over market data, such as
``div(delay($close, 5), $close)``, that predict the cross-section of future
returns. It provides a typed expression language with a causal CPU evaluator,
leakage-resistant evaluation, and three search methods behind one
budget-accounting interface:

* random search;
* genetic programming;
* reinforcement-learning-guided expression construction.

**Status.** Alpha-quality research infrastructure. Every component is
implemented and covered by CPU tests on deterministic synthetic data. **No
research experiment has been run yet.** The repository contains no performance
results, and nothing here claims that any factor or method is profitable.

.. contents::
   :local:
   :depth: 1

Research question
-----------------

Can reinforcement learning discover diverse, interpretable and economically
meaningful alpha factors more efficiently than random search and genetic
programming, when all three are given the same data, grammar, scoring rule and
evaluation budget?

AlphaLab is built so that this comparison is fair and auditable:

* every method draws from the same grammar;
* every method is charged the same way for each new candidate;
* selection happens on a validation partition, identically for all methods;
* the test partition stays locked until selection is finished.

Installation
------------

Python 3.11 or newer is required.

.. code-block:: bash

   git clone https://github.com/joshwu108/AlphaLab.git
   cd AlphaLab
   python -m venv .venv && source .venv/bin/activate
   pip install -e '.[dev,parquet]'        # core library, tests, Parquet support
   pip install -e '.[rl]'                 # optional: torch for the RL policy and trainer

The core library depends only on NumPy and pandas. ``torch`` is needed only by
``alphalab.rl.policy`` and ``alphalab.rl.trainer``. ``pyarrow`` is needed only
for Parquet files.

Quick start
-----------

Command line:

.. code-block:: bash

   alphalab operators                                   # list the expression operators
   alphalab synth --out data/synthetic.csv              # deterministic synthetic bars
   alphalab validate-data data/synthetic.csv            # contract check and fingerprint
   alphalab check-config configs/synthetic_gp.toml      # validate a configuration
   alphalab run configs/synthetic_gp.toml --dry-run --output runs/gp-dry
   alphalab run configs/synthetic_gp.toml --output runs/gp
   alphalab eval-expr 'div(delay($close, 5), $close)' --config configs/synthetic_gp.toml
   alphalab report runs/gp
   alphalab compare runs/random runs/gp runs/rl

Python:

.. code-block:: python

   from alphalab.data import SyntheticSpec, make_synthetic_panel, panel_forward_returns
   from alphalab.data import split_by_fractions
   from alphalab.evaluation import EvaluationBudget, FactorEvaluator
   from alphalab.expressions import Grammar
   from alphalab.search import GeneticProgramming, GPSettings, select_on_validation

   panel = make_synthetic_panel(SyntheticSpec(n_periods=400, n_instruments=40))
   split = split_by_fractions(panel.n_periods, 0.6, 0.2, horizon=1, embargo=1)
   evaluator = FactorEvaluator(
       panel, panel_forward_returns(panel, 1), split, budget=EvaluationBudget(200)
   )
   result = GeneticProgramming(Grammar(max_depth=4), GPSettings(seed=0)).run(evaluator)
   chosen = select_on_validation(evaluator, result.best, max_factors=5)

The ``examples/`` directory contains three runnable scripts. Each finishes in a
few seconds on a laptop CPU.

Architecture
------------

========================  =======================================================
Package                   Responsibility
========================  =======================================================
``alphalab.data``         Data contract, validation, CSV/Parquet ingestion,
                          immutable panels, universes, labels, purged splits,
                          synthetic data
``alphalab.expressions``  Typed AST, operator registry, parser, canonical
                          hashing, grammar and sampler, causal evaluator, cache
``alphalab.evaluation``   IC and rank IC, turnover, stability, redundancy,
                          partition-guarded evaluator, budgets, selection, export
``alphalab.search``       Random search, genetic programming, shared history and
                          validation-based selection
``alphalab.rl``           RPN token grammar with action masks, environment,
                          GRU actor-critic policy, policy-gradient trainer
``alphalab.experiments``  TOML configurations, runner, manifests, seeding
``alphalab.reporting``    Run reports and comparability-checked comparisons
``alphalab.cli``          ``alphalab`` command
========================  =======================================================

Data contract
-------------

Input is long-format daily bars, with one row per ``(timestamp, instrument)``:

* **Required columns:** ``timestamp``, ``instrument``, ``open``, ``high``,
  ``low``, ``close`` and ``volume``.
* **Optional columns:** ``vwap`` and ``adj_factor``.

Validation does the following:

* parses timestamps and converts timezone-aware values to naive UTC;
* rejects missing columns, non-positive prices, negative volume, infinite
  values, and OHLC inconsistencies;
* applies an explicit policy to duplicate keys (error by default);
* counts missing values but never imputes them.

Validated data is pivoted into an immutable ``MarketPanel`` of read-only
``(T, N)`` arrays. Each panel carries a SHA-256 fingerprint that keys caches,
manifests and checkpoints.

``adj_factor`` must be a *forward* cumulative factor known at each timestamp.
Back-adjusted prices embed future corporate actions in their levels, so they
are not causal. See ``alphalab/data/contract.py``.

Expression language
-------------------

Expressions are immutable typed trees. They are type-checked on construction,
so an ill-typed tree, or one with a window below an operator's minimum, cannot
be built.

==============================  ==================================================
Category                        Operators
==============================  ==================================================
Element-wise                    ``add sub mul div`` (protected), ``neg abs``
Time series (window ``w``)      ``delay delta ts_mean ts_sum ts_std ts_min ts_max
                                ts_ema ts_corr``
Cross-sectional (per session)   ``cs_rank cs_zscore``
==============================  ==================================================

Semantics:

* **Causality.** Time-series operators read only past and current rows. A
  property test perturbs future rows and checks that earlier outputs never
  change.
* **Missing data.** Rolling windows are strict: any NaN in a window produces
  NaN.
* **Non-finite values.** Division by values smaller than ``1e-12`` in
  magnitude, and every other non-finite result, become NaN.
* **Lookback.** ``ts_ema`` is truncated to ``w`` rows, so every operator has a
  finite lookback. The lookback of an expression is computed exactly.

Canonical serialization orders commutative operands, so equivalent forms share
one SHA-256 key. That key is used for deduplication, caching and budget
accounting.

Evaluation and leakage controls
-------------------------------

* **Labels.** Labels are ``horizon``-session forward returns. They are the only
  place where future data is read, and no expression can reference them.
* **Partitions.** Train, validation and test partitions are chronological.
  Training rows whose label interval reaches the next partition are purged, and
  an optional embargo removes more rows. A property test checks this.
* **Locked test set.** ``FactorEvaluator`` raises ``PartitionAccessError`` for
  test-partition requests until ``unlock_test`` is called. It logs every
  partition read, and run manifests record how often the test partition was
  read.
* **Fitness.** Fitness is the absolute train rank IC (or IC) minus an optional
  complexity penalty. The sign learned on train is applied unchanged on
  validation and test.
* **Metrics.** Pearson and rank IC, ICIR, hit rate, sign consistency across
  sub-periods, turnover of unit-gross weights, rank autocorrelation, coverage,
  factor correlation and redundancy.

**Predictive correlation is not profitability.** IC measures cross-sectional
association before transaction costs, market impact, borrow constraints and
capacity limits. A factor with a positive IC can lose money when traded.
AlphaLab does not currently simulate trading.

Search methods and budget accounting
------------------------------------

The unit of cost is one *unique* candidate, identified by canonical key, scored
on the train partition. Repeating a candidate is free. Validation and test
scoring are never charged.

* **Random search.** Samples grammar-valid expressions, deduplicates them, and
  stops when the budget is spent or the search space is exhausted.
* **Genetic programming.**

  * Ramped initialization and tournament selection on diversity-penalized
    fitness.
  * Typed subtree crossover; subtree, point and hoist mutation; elitism.
  * Atomic JSON checkpoints. A resumed run reproduces an uninterrupted one
    exactly, which is tested.
* **Reinforcement learning.** The agent writes expressions as Reverse Polish
  Notation token sequences, a formulation from AlphaGen.

  * Action masks enforce typing, the grammar's size, depth and lookback limits,
    completion within a token limit, and the shared rule against degenerate
    forms.
  * The policy is a GRU actor-critic trained by REINFORCE with a learned value
    baseline, running reward normalization and an entropy bonus.
  * Rewards are either train fitness (optionally diversity-penalized) or the
    gain of a ridge-combined factor pool.
  * Checkpoints include model, optimizer, sampler and budget state, and a
    resumed run reproduces an uninterrupted one on CPU, which is tested.

The RL trainer has run only at toy scale in CPU tests. It is implemented but
**not yet trained**.

Experiments and reproducibility
-------------------------------

An experiment is a schema-versioned TOML file (see ``configs/``):

* Unknown keys are rejected, so a typo cannot silently fall back to a default.
* The resolved configuration is hashed; the hash identifies the run and guards
  resumption.

``alphalab run`` writes a run directory containing:

* ``manifest.json``: configuration hash, dataset fingerprint, partitions, Git
  revision, library versions, budget usage, partition-access counts, and the
  traceback if the run fails;
* ``history.csv``: every charged evaluation, in order;
* ``result.json``: anytime curve, validation-selected factors and a
  train-fitted combination;
* checkpoints.

``--dry-run`` validates a configuration end-to-end at negligible cost.
``compare`` refuses to call runs comparable when they differ in data, split,
grammar, scoring or budget.

Development
-----------

.. code-block:: bash

   pytest -q                                        # full CPU suite (torch tests skip without torch)
   pytest -q --cov=alphalab --cov-report=term-missing
   ruff check src tests examples benchmarks && ruff format --check src tests examples benchmarks
   mypy

CI runs on every push. One job runs lint, type checking and the full test suite
without torch; a second job installs CPU torch and runs everything, including
the RL policy and trainer tests. See ``CONTRIBUTING.rst``.

Limitations
-----------

* **Bar frequency.** Only daily bars are supported, and each timestamp is one
  session.
* **Evaluation backend.** Evaluation is a CPU NumPy reference; no accelerated
  backend exists yet.
* **Missing data.** Strict rolling windows mean sparse data loses coverage.
  Imputation is deliberately left to the user.
* **Trading.** There is no transaction-cost, portfolio or execution
  simulation, and no risk-model neutralization (industry or size).
* **Overlapping labels.** With horizons above one session, labels overlap and
  per-session ICs are serially correlated, so the reported t-statistics
  overstate significance.
* **RL algorithm.** The RL algorithm is plain REINFORCE with a baseline; PPO is
  not implemented yet.
* **Synthetic data.** Synthetic data contains a single planted signal and is
  meant only for testing.

References and attribution
--------------------------

* S. Yu, H. Xue, X. Ao, F. Pan, J. He, D. Tu and Q. He. *Generating Synergistic
  Formulaic Alpha Collections via Reinforcement Learning.* KDD 2023.
  arXiv:2306.12964.
* AlphaGen reference implementation: https://github.com/ICT-FinD-Lab/alphagen.
* Hudson River Trading, *Intern Spotlight 2023: HRT AI Labs Summer Projects*.
* J. R. Koza. *Genetic Programming.* MIT Press, 1992.

AlphaLab takes two ideas from the AlphaGen paper: Reverse Polish Notation
generation with grammar masks, and the linear combination of a factor pool as a
reward signal. All AlphaLab code is an independent implementation; no
third-party code was copied.

AI-assisted development
-----------------------

Much of this codebase was written with an AI coding assistant (Claude Code),
under the direction and review of the repository owner. Commits are made under
the owner's identity, and this statement documents how the code was produced.

License
-------

MIT. See ``LICENSE``.
