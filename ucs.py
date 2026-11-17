"""Uniform Cost Search (primary algorithm).

Frontier ordered by g(n) = total correction cost so far. Because every step
cost is >= 0 and the goal test is done on expansion, the first goal popped is
a minimum-cost set of corrections (within the correction limit) - i.e. the
most likely correction under the NN probabilities.
"""
from __future__ import annotations

import config
from search.base import SearchResult, best_first_search


def ucs(problem, max_expansions: int = config.MAX_SEARCH_EXPANSIONS) -> SearchResult:
    return best_first_search(problem, priority=lambda g, h: (g,), name="UCS",
                             use_heuristic=False, max_expansions=max_expansions)
