"""Generic best-first graph search shared by UCS, Greedy and A*.

The three algorithms differ ONLY in the priority used to order the frontier:
    UCS     f(n) = g(n)                 (path cost)
    Greedy  f(n) = h(n)                 (heuristic only; ties broken by g)
    A*      f(n) = g(n) + h(n)
so they share this implementation instead of duplicating code.

A *problem* object must provide
    initial_state() -> state           hashable
    successors(state) -> iterable[(state, step_cost)]   step_cost >= 0
    goal_test(state) -> (bool, info)
    heuristic(state) -> float          (may be math.inf for dead states)
    describe(state) -> str             (for the trace)

The goal test is applied when a node is POPPED from the frontier (not when it
is generated). With non-negative step costs this makes UCS optimal, and A*
optimal whenever h is admissible (the correction state space is a tree, see
integration/correction.py, so no re-opening of closed nodes is needed).
"""
from __future__ import annotations

import heapq
import itertools
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Callable


@dataclass
class SearchStats:
    algorithm: str = ""
    nodes_generated: int = 0
    nodes_expanded: int = 0
    goal_tests: int = 0
    pruned_dead_states: int = 0
    max_frontier: int = 0
    time_s: float = 0.0
    n_corrections: int = 0
    total_cost: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class SearchResult:
    status: str                      # "found" | "not_found" | "limit_reached"
    state: tuple | None
    cost: float
    stats: SearchStats
    goal_info: object = None
    trace: list[str] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return self.status == "found"


PriorityFn = Callable[[float, float], tuple]


def best_first_search(problem, priority: PriorityFn, name: str, use_heuristic: bool,
                      max_expansions: int = 20_000, trace_limit: int = 300) -> SearchResult:
    stats = SearchStats(algorithm=name)
    trace: list[str] = []
    t0 = time.perf_counter()
    counter = itertools.count()            # FIFO tie-breaker -> deterministic results

    start = problem.initial_state()
    h0 = problem.heuristic(start) if use_heuristic else 0.0
    frontier = [(priority(0.0, h0), next(counter), 0.0, start)]
    stats.nodes_generated = 1
    closed: set = set()

    def finish(status, state=None, cost=math.inf, info=None) -> SearchResult:
        stats.time_s = time.perf_counter() - t0
        if state is not None:
            stats.n_corrections = len(state)
            stats.total_cost = cost
        return SearchResult(status, state, cost, stats, info, trace)

    while frontier:
        stats.max_frontier = max(stats.max_frontier, len(frontier))
        f, _, g, state = heapq.heappop(frontier)
        if state in closed:
            continue
        closed.add(state)

        stats.goal_tests += 1
        is_goal, info = problem.goal_test(state)
        if len(trace) < trace_limit:
            trace.append(f"pop #{stats.goal_tests:<4d} f={f[0]:7.3f} g={g:7.3f}  {problem.describe(state):40s}"
                         f" -> {'GOAL' if is_goal else getattr(info, 'reason', info)}")
        if is_goal:
            return finish("found", state, g, info)
        if stats.nodes_expanded >= max_expansions:
            return finish("limit_reached")

        stats.nodes_expanded += 1
        for child, step_cost in problem.successors(state):
            if step_cost < 0:
                raise ValueError("negative step cost - UCS/A* require non-negative costs")
            h = problem.heuristic(child) if use_heuristic else 0.0
            if math.isinf(h):               # provably cannot reach a goal
                stats.pruned_dead_states += 1
                continue
            g2 = g + step_cost
            heapq.heappush(frontier, (priority(g2, h), next(counter), g2, child))
            stats.nodes_generated += 1
    return finish("not_found")
