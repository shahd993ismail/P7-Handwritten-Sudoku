"""C. Backtracking + Forward Checking + AC-3 (BT + FC + AC-3, i.e. MAC).

AC-3 (Mackworth, 1977) makes every arc (Xi, Xj) consistent: each value x in
D(Xi) must have a *support* y in D(Xj) with x != y. For Sudoku's "not equal"
constraint, x loses its support exactly when D(Xj) == {x}.

This solver
  1. runs AC-3 on the whole puzzle before search (pre-processing), and
  2. after every assignment runs forward checking and then AC-3 again,
     starting from the arcs pointing at the cells whose domains just shrank
     (Maintaining Arc Consistency). Reductions therefore propagate through
     chains of cells - e.g. a peer reduced to a single value immediately
     removes that value from ITS peers, which FC alone never does.

`ac3()` is also used by the integration pipeline to detect conflicts in the
recognised puzzle (an empty domain = inconsistent puzzle).
"""
from __future__ import annotations

from collections import deque

from csp.forward_checking import forward_check, initial_domains, order_values, select_variable
from csp.sudoku import DIGITS, PEERS, CSPStats, SearchLimitReached


def revise(domains: list[set[int]], xi: int, xj: int, stats: CSPStats) -> bool:
    """Remove values of Xi without support in Xj. Returns True if D(Xi) changed."""
    removed = False
    for x in list(domains[xi]):
        supported = False
        for y in domains[xj]:          # look for a support y != x
            stats.constraint_checks += 1
            if y != x:
                supported = True
                break
        if not supported:
            domains[xi].discard(x)
            stats.domain_reductions += 1
            removed = True
    return removed


def ac3(domains: list[set[int]], stats: CSPStats, queue=None) -> bool:
    """Enforce arc consistency in place. Returns False if a domain becomes empty."""
    if queue is None:
        queue = deque((xi, xj) for xi in range(81) for xj in PEERS[xi])
    else:
        queue = deque(queue)
    in_queue = set(queue)
    while queue:
        xi, xj = queue.popleft()
        in_queue.discard((xi, xj))
        if revise(domains, xi, xj, stats):
            if not domains[xi]:
                return False
            for xk in PEERS[xi]:
                if xk != xj and (xk, xi) not in in_queue:
                    queue.append((xk, xi))
                    in_queue.add((xk, xi))
    return True


def ac3_check(grid) -> tuple[bool, list[set[int]], CSPStats]:
    """Run AC-3 on a (possibly wrong) recognised puzzle.

    Returns (consistent, reduced_domains, stats). Givens start with singleton
    domains, empty cells with {1..9}.
    """
    stats = CSPStats(algorithm="AC-3 check")
    domains = [{v} if v else set(DIGITS) for v in grid]
    ok = ac3(domains, stats)
    return ok, domains, stats


def solve_mac(grid, stats: CSPStats, solutions: list, count_limit: int,
              var_order: str, value_order: str, max_assignments: int) -> None:
    grid = list(grid)
    domains = initial_domains(grid, stats)       # forward checking from the givens
    if domains is None or not ac3(domains, stats):   # AC-3 pre-processing
        return
    assigned = [bool(v) for v in grid]

    def recurse(domains) -> bool:
        if all(assigned):
            solutions.append([next(iter(d)) for d in domains])
            return len(solutions) >= count_limit
        var = select_variable(domains, assigned, var_order)
        for v in order_values(domains, assigned, var, value_order):
            if stats.assignments >= max_assignments:
                raise SearchLimitReached
            stats.assignments += 1
            new = [set(d) for d in domains]
            new[var] = {v}
            assigned[var] = True
            changed: list[int] = []
            ok = forward_check(new, assigned, var, v, stats, changed)
            if ok and changed:
                # propagate from every cell whose domain shrank
                arcs = [(xk, c) for c in changed for xk in PEERS[c] if xk != var]
                ok = ac3(new, stats, arcs)
            if ok and recurse(new):
                return True
            assigned[var] = False
        stats.backtracks += 1
        return False

    recurse(domains)
