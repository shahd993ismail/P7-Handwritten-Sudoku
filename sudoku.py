"""Sudoku as a Constraint Satisfaction Problem (shared definitions + dispatcher).

Variables : the 81 cells, indexed 0..80 (row-major: index = 9*row + col).
Domains   : {1..9}; a given (clue) has the singleton domain {its digit}.
Constraints: binary "not equal" constraints between every pair of cells that
             share a row, a column or a 3x3 box (the cell's 20 *peers*).

A grid is a list of 81 ints, 0 meaning an empty cell.

Three interchangeable solvers live in separate modules:
    csp/backtracking.py      -> "bt"   plain chronological backtracking
    csp/forward_checking.py  -> "fc"   backtracking + forward checking
    csp/ac3.py               -> "ac3"  backtracking + FC + AC-3 (MAC)
Use :func:`solve` to switch between them with a string.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

DIGITS = tuple(range(1, 10))

# ---------------------------------------------------------------- structure
ROWS = [[9 * r + c for c in range(9)] for r in range(9)]
COLS = [[9 * r + c for r in range(9)] for c in range(9)]
BOXES = [[9 * (br + r) + (bc + c) for r in range(3) for c in range(3)]
         for br in (0, 3, 6) for bc in (0, 3, 6)]
UNITS = ROWS + COLS + BOXES
UNIT_NAMES = [f"row {i + 1}" for i in range(9)] + [f"column {i + 1}" for i in range(9)] + \
             [f"box {i + 1}" for i in range(9)]
PEERS: list[tuple[int, ...]] = []
for _cell in range(81):
    _p = set()
    for _u in UNITS:
        if _cell in _u:
            _p.update(_u)
    _p.discard(_cell)
    PEERS.append(tuple(sorted(_p)))
assert all(len(p) == 20 for p in PEERS)

ALGORITHM_LABELS = {"bt": "BT", "fc": "BT+FC", "ac3": "BT+FC+AC-3"}
_ALIASES = {"bt": "bt", "backtracking": "bt", "bt+fc": "fc", "fc": "fc", "forward_checking": "fc",
            "bt+fc+ac-3": "ac3", "bt+fc+ac3": "ac3", "ac3": "ac3", "ac-3": "ac3", "mac": "ac3"}


def cell_name(idx: int) -> str:
    """Human-readable cell name, e.g. 0 -> 'R1C1'."""
    return f"R{idx // 9 + 1}C{idx % 9 + 1}"


def validate_grid(grid) -> list[int]:
    """Return a clean list of 81 ints in 0..9 or raise ValueError."""
    try:
        g = [int(v) for v in grid]
    except (TypeError, ValueError) as exc:
        raise ValueError("grid must contain 81 integers") from exc
    if len(g) != 81:
        raise ValueError(f"grid must contain exactly 81 cells, got {len(g)}")
    if any(v < 0 or v > 9 for v in g):
        raise ValueError("grid values must be in 0..9 (0 = empty)")
    return g


def parse_grid(text: str) -> list[int]:
    """Parse an 81-character string ('0' or '.' for blanks)."""
    chars = [ch for ch in text if ch.isdigit() or ch == "."]
    return validate_grid([0 if ch == "." else int(ch) for ch in chars])


def grid_to_str(grid, blank: str = ".") -> str:
    lines = []
    for r in range(9):
        if r in (3, 6):
            lines.append("------+-------+------")
        row = [str(grid[9 * r + c]) if grid[9 * r + c] else blank for c in range(9)]
        lines.append(" ".join(row[0:3]) + " | " + " ".join(row[3:6]) + " | " + " ".join(row[6:9]))
    return "\n".join(lines)


def find_conflicts(grid) -> list[dict]:
    """List every duplicate-digit conflict among the filled cells.

    Each conflict is a dict {unit, digit, cells}. A pair of cells may appear in
    several units (e.g. same row AND same box); each unit is reported.
    """
    conflicts = []
    for name, unit in zip(UNIT_NAMES, UNITS):
        seen: dict[int, list[int]] = {}
        for c in unit:
            if grid[c]:
                seen.setdefault(grid[c], []).append(c)
        for d, cells in seen.items():
            if len(cells) > 1:
                conflicts.append({"unit": name, "digit": d, "cells": cells})
    return conflicts


def conflict_pairs(grid) -> set[tuple[int, int]]:
    """Set of unordered cell pairs (i<j) that are peers holding the same digit."""
    pairs = set()
    for c in range(81):
        if grid[c]:
            for p in PEERS[c]:
                if p > c and grid[p] == grid[c]:
                    pairs.add((c, p))
    return pairs


def is_complete_solution(grid) -> bool:
    return all(grid) and not conflict_pairs(grid)


@dataclass
class CSPStats:
    algorithm: str = ""
    assignments: int = 0
    backtracks: int = 0
    constraint_checks: int = 0
    domain_reductions: int = 0
    time_s: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class CSPResult:
    status: str                     # "solved" | "no_solution" | "limit_reached"
    solution: list[int] | None
    n_solutions: int                # number found (capped at count_limit)
    stats: CSPStats
    solutions: list[list[int]] = field(default_factory=list)

    @property
    def solved(self) -> bool:
        return self.status == "solved"

    @property
    def message(self) -> str:
        if self.status == "solved":
            return "SOLVED"
        if self.status == "limit_reached":
            return "NO SOLUTION FOUND (search limit reached)"
        return "NO SOLUTION: the puzzle violates the Sudoku constraints"


class SearchLimitReached(Exception):
    """Raised internally when a solver exceeds its assignment budget."""


def normalise_algorithm(name: str) -> str:
    key = str(name).strip().lower().replace(" ", "")
    if key not in _ALIASES:
        raise ValueError(f"unknown CSP algorithm '{name}'. Use one of: bt, fc, ac3")
    return _ALIASES[key]


def solve(grid, algorithm: str = "ac3", count_limit: int = 1, var_order: str = "mrv",
          value_order: str = "lcv", max_assignments: int | None = None) -> CSPResult:
    """Solve *grid* with the chosen algorithm.

    count_limit = 1 -> stop at the first solution.
    count_limit = 2 -> uniqueness check (n_solutions == 1 means unique).
    """
    from config import CSP_MAX_ASSIGNMENTS
    from csp.ac3 import solve_mac
    from csp.backtracking import solve_bt
    from csp.forward_checking import solve_fc

    algo = normalise_algorithm(algorithm)
    if var_order not in ("mrv", "static"):
        raise ValueError("var_order must be 'mrv' or 'static'")
    if value_order not in ("lcv", "natural"):
        raise ValueError("value_order must be 'lcv' or 'natural'")
    if count_limit < 1:
        raise ValueError("count_limit must be >= 1")
    grid = validate_grid(grid)
    fn = {"bt": solve_bt, "fc": solve_fc, "ac3": solve_mac}[algo]
    stats = CSPStats(algorithm=ALGORITHM_LABELS[algo])
    t0 = time.perf_counter()
    solutions: list[list[int]] = []
    status = "no_solution"
    try:
        fn(grid, stats, solutions, count_limit, var_order, value_order,
           max_assignments or CSP_MAX_ASSIGNMENTS)
        status = "solved" if solutions else "no_solution"
    except SearchLimitReached:
        status = "solved" if solutions else "limit_reached"
    stats.time_s = time.perf_counter() - t0
    return CSPResult(status=status, solution=solutions[0] if solutions else None,
                     n_solutions=len(solutions), stats=stats, solutions=solutions)


def count_solutions(grid, limit: int = 2, algorithm: str = "ac3") -> int:
    """Number of solutions, capped at *limit* (use limit=2 to test uniqueness)."""
    return solve(grid, algorithm=algorithm, count_limit=limit).n_solutions
