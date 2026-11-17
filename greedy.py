"""Greedy Best-First Search (comparison algorithm).

Frontier ordered by h(n) only (ties broken by g). It rushes towards states
with few remaining duplicate conflicts and is usually fast, but it is NOT
optimal: it can return a correction set that is more expensive (less likely)
than the one UCS finds.
"""
from __future__ import annotations

import config
from search.base import SearchResult, best_first_search


def greedy(problem, max_expansions: int = config.MAX_SEARCH_EXPANSIONS) -> SearchResult:
    return best_first_search(problem, priority=lambda g, h: (h, g), name="Greedy",
                             use_heuristic=True, max_expansions=max_expansions)
