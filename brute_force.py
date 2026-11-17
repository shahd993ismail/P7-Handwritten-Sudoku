"""Exhaustive reference search used to verify optimality on small problems.

Enumerates every combination of at most `max_corrections` (cell, digit)
changes allowed by the problem and returns the cheapest one that passes the
goal test. Exponential - only for tests and Experiment 1's optimality check.
"""
from __future__ import annotations

import itertools
import math


def brute_force(problem) -> tuple[float, tuple | None, int]:
    """Return (best_cost, best_state, goal_tests)."""
    cells = sorted(problem.alternatives)
    best_cost, best_state, tests = math.inf, None, 0
    for k in range(0, problem.max_corrections + 1):
        for subset in itertools.combinations(cells, k):
            for digits in itertools.product(*[problem.alternatives[c] for c in subset]):
                cost = sum(c for _, c in digits)
                if cost >= best_cost:
                    continue
                state = tuple((cell, d) for cell, (d, _) in zip(subset, digits))
                tests += 1
                ok, _ = problem.goal_test(state)
                if ok:
                    best_cost, best_state = cost, state
    return best_cost, best_state, tests
