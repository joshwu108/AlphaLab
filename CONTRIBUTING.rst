Contributing to AlphaLab
========================

Setup
-----

.. code-block:: bash

   python -m venv .venv && source .venv/bin/activate
   pip install -e '.[dev,parquet,rl]'

Checks
------

Every change must pass the same checks as CI:

.. code-block:: bash

   ruff check src tests examples benchmarks
   ruff format --check src tests examples benchmarks
   mypy
   pytest -q

Tests must be deterministic and CPU-only:

* use seeded generators;
* build data with ``alphalab.data.synthetic``;
* never download external data.

Mark a test that needs torch with ``pytest.importorskip("torch")``. Mark a
longer integration test ``@pytest.mark.slow``.

Ground rules
------------

* **Causality.** No operator, feature or preprocessing step may read rows after
  the prediction time. Add a causality property test for every new
  time-series operator.
* **Test partition.** Search, rewards, hyperparameter choices and selection
  must never read the test partition. ``FactorEvaluator`` enforces this; do not
  bypass it.
* **Fair budgets.** A new search method must charge the shared
  ``EvaluationBudget`` through ``FactorEvaluator.score`` on the train
  partition, and must respect the shared ``Grammar``, including its rule
  against degenerate forms.
* **No unverified claims.** Never commit performance numbers, and never claim
  profitability or novelty that the code and tests do not support.
* **Errors.** Library errors derive from ``alphalab.errors.AlphaLabError`` and
  should say what was wrong and how to fix it.

Adding an operator
------------------

1. Implement a kernel in ``src/alphalab/expressions/operators.py`` that maps
   ``(T, N)`` float arrays (and a window, if applicable) to a ``(T, N)`` array,
   and follows the module's NaN and causality rules.
2. Register an ``OperatorSpec`` in ``OPERATORS``, with its arity, category,
   minimum window and ``lookback_offset``.
3. Add a reference test against pandas or an explicit formula in
   ``tests/expressions/test_evaluator.py``. The causality and lookback property
   tests pick up the new operator automatically through the grammar sampler.

The operator then becomes available to the parser, the grammar, GP and the RL
vocabulary.

Adding a search method
----------------------

Implement ``run(evaluator) -> SearchResult`` (see ``alphalab.search.base``):

* record each charged evaluation in a ``SearchHistory``;
* stop when the budget is exhausted;
* add the method to ``alphalab.experiments.runner._search`` and to the
  configuration schema.

Commits
-------

Use conventional commit messages (``feat:``, ``fix:``, ``test:``, ``docs:``,
``refactor:``, ``perf:``, ``ci:``, ``chore:``) that describe one meaningful
change. Markdown files and planning documents are not tracked; public
documentation is reStructuredText.
