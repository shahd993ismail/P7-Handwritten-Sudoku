"""A* Search (comparison algorithm).

Frontier ordered by f(n) = g(n) + h(n). With the admissible conflict-matching
heuristic of integration/correction.py (proof in its docstring and in
REPORT.md) A* returns the same optimal cost as UCS while expanding no more
nodes in principle; it also discards states that can provably never reach a
goal (h = infinity).
"""
from __future__ import annotations

import config
from search.base import SearchResult, best_first_search


def astar(problem, max_expansions: int = config.MAX_SEARCH_EXPANSIONS) -> SearchResult:
    return best_first_search(problem, priority=lambda g, h: (g + h, g), name="A*",
                             use_heuristic=True, max_expansions=max_expansions)
