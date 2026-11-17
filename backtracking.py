"""A. Plain chronological Backtracking (BT).

There is NO inference: the solver keeps no domains. Whenever it needs to know
which values are legal for a cell it re-checks the binary "not equal"
constraint against every *assigned* peer (each comparison is one constraint
check). Dead ends are therefore discovered only when a variable is reached
whose every value clashes with an already-assigned peer.

Variable ordering:  "mrv"    -> Minimum Remaining Values (fewest legal values)
                    "static" -> first empty cell in row-major order
Value ordering:     "lcv"    -> Least Constraining Value (rules out the fewest
                                options of unassigned peers)
                    "natural"-> 1..9
"""
from __future__ import annotations

from csp.sudoku import DIGITS, PEERS, CSPStats, SearchLimitReached


def _legal_values(grid: list[int], var: int, stats: CSPStats) -> list[int]:
    """Values for *var* consistent with all assigned peers (counts checks)."""
    assigned = [grid[p] for p in PEERS[var] if grid[p]]
    # Each candidate value is compared with every assigned peer: 9 * |assigned| checks.
    stats.constraint_checks += len(DIGITS) * len(assigned)
    used = set(assigned)
    return [v for v in DIGITS if v not in used]


def _givens_consistent(grid: list[int], stats: CSPStats) -> bool:
    for c in range(81):
        if grid[c]:
            for p in PEERS[c]:
                if p > c and grid[p]:
                    stats.constraint_checks += 1
                    if grid[p] == grid[c]:
                        return False
    return True


def solve_bt(grid, stats: CSPStats, solutions: list, count_limit: int,
             var_order: str, value_order: str, max_assignments: int) -> None:
    grid = list(grid)
    if not _givens_consistent(grid, stats):
        return

    def recurse() -> bool:
        empty = [i for i in range(81) if grid[i] == 0]
        if not empty:
            solutions.append(list(grid))
            return len(solutions) >= count_limit

        if var_order == "mrv":
            legal = {i: _legal_values(grid, i, stats) for i in empty}
            var = min(empty, key=lambda i: (len(legal[i]), i))
            values = legal[var]
        else:
            var = empty[0]
            legal = None
            values = _legal_values(grid, var, stats)

        if value_order == "lcv" and len(values) > 1:
            # how many unassigned peers would lose this value
            if legal is None:
                legal = {i: _legal_values(grid, i, stats) for i in PEERS[var] if grid[i] == 0}
            peers = [p for p in PEERS[var] if grid[p] == 0]
            values = sorted(values, key=lambda v: (sum(v in legal[p] for p in peers), v))

        for v in values:
            if stats.assignments >= max_assignments:
                raise SearchLimitReached
            grid[var] = v
            stats.assignments += 1
            if recurse():
                return True
            grid[var] = 0
        stats.backtracks += 1
        return False

    recurse()
