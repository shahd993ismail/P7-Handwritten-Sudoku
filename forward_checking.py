"""B. Backtracking + Forward Checking (BT + FC).

The solver keeps a domain (set of still-possible digits) for every cell.
After assigning X = v it immediately removes v from the domain of every
unassigned peer of X ("forward checking"). If any peer's domain becomes empty
(a *domain wipe-out*) the assignment is rejected at once, without descending
further - this is the pruning plain BT cannot do.

Only the direct neighbours of the assigned variable are checked; FC does not
propagate further (that is what AC-3 adds, see csp/ac3.py).
"""
from __future__ import annotations

from csp.sudoku import DIGITS, PEERS, CSPStats, SearchLimitReached


def initial_domains(grid: list[int], stats: CSPStats) -> list[set[int]] | None:
    """Givens get singleton domains; empty cells lose the digits of given peers.

    Returns None if the givens already conflict (empty domain).
    """
    domains = [{grid[i]} if grid[i] else set(DIGITS) for i in range(81)]
    for c in range(81):
        if grid[c]:
            for p in PEERS[c]:
                stats.constraint_checks += len(domains[p])
                if grid[c] in domains[p]:
                    if grid[p]:            # two givens with the same digit
                        return None
                    domains[p].discard(grid[c])
                    stats.domain_reductions += 1
                    if not domains[p]:
                        return None
    return domains


def forward_check(domains: list[set[int]], assigned: list[bool], var: int, value: int,
                  stats: CSPStats, changed: list[int] | None = None) -> bool:
    """Remove *value* from unassigned peers of *var*. False on wipe-out."""
    for p in PEERS[var]:
        if assigned[p]:
            continue
        stats.constraint_checks += len(domains[p])
        if value in domains[p]:
            domains[p].discard(value)
            stats.domain_reductions += 1
            if changed is not None:
                changed.append(p)
            if not domains[p]:
                return False
    return True


def select_variable(domains, assigned, var_order: str) -> int:
    unassigned = [i for i in range(81) if not assigned[i]]
    if var_order == "mrv":
        return min(unassigned, key=lambda i: (len(domains[i]), i))
    return unassigned[0]


def order_values(domains, assigned, var: int, value_order: str) -> list[int]:
    values = sorted(domains[var])
    if value_order == "lcv" and len(values) > 1:
        peers = [p for p in PEERS[var] if not assigned[p]]
        values.sort(key=lambda v: (sum(v in domains[p] for p in peers), v))
    return values


def solve_fc(grid, stats: CSPStats, solutions: list, count_limit: int,
             var_order: str, value_order: str, max_assignments: int) -> None:
    grid = list(grid)
    domains = initial_domains(grid, stats)
    if domains is None:
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
            if forward_check(new, assigned, var, v, stats) and recurse(new):
                return True
            assigned[var] = False
        stats.backtracks += 1
        return False

    recurse(domains)
