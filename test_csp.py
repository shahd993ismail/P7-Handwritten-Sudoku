"""CSP tests: BT, BT+FC, BT+FC+AC-3 (normal, edge and impossible cases)."""
import numpy as np
import pytest

from csp.ac3 import ac3_check
from csp.sudoku import count_solutions, find_conflicts, is_complete_solution, solve, validate_grid
from data.puzzle_gen import generate_puzzle

ALGOS = ["bt", "fc", "ac3"]


@pytest.mark.parametrize("algo", ALGOS)
def test_normal_valid_sudoku_is_solved(puzzle, solution, algo):           # NORMAL 1
    res = solve(puzzle, algo)
    assert res.status == "solved" and res.solution == solution
    assert is_complete_solution(res.solution)


@pytest.mark.parametrize("algo", ALGOS)
def test_normal_one_missing_cell(solution, algo):                         # NORMAL 2
    g = list(solution)
    g[40] = 0
    res = solve(g, algo)
    assert res.solution == solution and res.stats.assignments == 1


@pytest.mark.parametrize("algo", ALGOS)
def test_normal_two_missing_cells(solution, algo):                        # NORMAL 3
    g = list(solution)
    g[0] = g[80] = 0
    assert solve(g, algo).solution == solution


@pytest.mark.parametrize("algo", ALGOS)
def test_edge_many_empty_cells(algo):                                     # EDGE 6
    g = [0] * 81                                     # completely empty grid
    res = solve(g, algo)
    assert res.solved and is_complete_solution(res.solution)
    p, sol = generate_puzzle(25, np.random.default_rng(3), unique=True)  # 56 empty cells
    assert solve(p, algo).solution == sol


@pytest.mark.parametrize("algo", ALGOS)
def test_impossible_contradictory_givens(puzzle, algo):                   # IMPOSSIBLE 9
    g = list(puzzle)
    g[2] = 5                                         # second 5 in row 1
    res = solve(g, algo)
    assert res.status == "no_solution" and res.solution is None
    assert "NO SOLUTION" in res.message


def test_impossible_without_duplicates():
    # no duplicate digits, but cell R1C9 has no legal value: row has 1..8, column has 9
    g = [0] * 81
    g[:8] = [1, 2, 3, 4, 5, 6, 7, 8]
    g[9 * 8 + 8] = 9
    assert not find_conflicts(g)
    for algo in ALGOS:
        assert solve(g, algo).status == "no_solution"
    ok, _, _ = ac3_check(g)
    assert not ok


def test_uniqueness_counting(puzzle, solution):
    assert count_solutions(puzzle, limit=2) == 1
    g = list(solution)
    for c in (0, 1, 9, 10):    # removing cells from a full solution keeps it solvable
        g[c] = 0
    assert count_solutions(g, limit=5) >= 1
    # the empty grid has many solutions; counting stops at the limit
    assert count_solutions([0] * 81, limit=2) == 2


def test_ac3_detects_duplicate_conflict(puzzle):
    g = list(puzzle)
    g[2] = 5
    ok, _, stats = ac3_check(g)
    assert not ok and stats.constraint_checks > 0
    assert find_conflicts(g)[0]["digit"] == 5


def test_algorithms_prune_differently():
    """With identical (static) ordering, FC and AC-3 must prune more than plain BT."""
    p, _ = generate_puzzle(25, np.random.default_rng(101), unique=True)
    r = {a: solve(p, a, var_order="static", value_order="natural").stats for a in ALGOS}
    assert r["bt"].assignments > r["fc"].assignments > r["ac3"].assignments
    assert r["bt"].domain_reductions == 0 and r["fc"].domain_reductions > 0
    assert all(s.constraint_checks > 0 for s in r.values())


def test_input_validation():
    with pytest.raises(ValueError):
        validate_grid([0] * 80)
    with pytest.raises(ValueError):
        validate_grid([10] + [0] * 80)
    with pytest.raises(ValueError):
        solve([0] * 81, "dfs")


def test_limit_reached_is_reported():
    p, _ = generate_puzzle(25, np.random.default_rng(101), unique=True)
    res = solve(p, "bt", var_order="static", value_order="natural", max_assignments=10)
    assert res.status == "limit_reached" and "limit" in res.message
