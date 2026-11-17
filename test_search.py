"""Search tests: UCS on a known-answer graph, and UCS / A* / Greedy vs brute force."""
import math

import numpy as np
import pytest

from integration.correction import CorrectionProblem, inject_misclassifications
from search.astar import astar
from search.brute_force import brute_force
from search.greedy import greedy
from search.ucs import ucs


class ToyGraph:
    """Small weighted graph with a known optimal path.

        S --1--> A --4--> G         cost 5
        S --2--> B --2--> G         cost 4   <- optimal
        S --1--> C --9--> G         cost 10
    h is admissible for A* but misleads Greedy (h(A)=0 makes it go via A).
    """
    edges = {"S": [("A", 1), ("B", 2), ("C", 1)], "A": [("G", 4)], "B": [("G", 2)], "C": [("G", 9)], "G": []}
    h = {"S": 3, "A": 0, "B": 2, "C": 5, "G": 0}

    def initial_state(self):
        return ("S",)

    def successors(self, state):
        for nxt, c in self.edges[state[-1]]:
            yield state + (nxt,), c

    def goal_test(self, state):
        return state[-1] == "G", "goal" if state[-1] == "G" else "not goal"

    def heuristic(self, state):
        return self.h[state[-1]]

    def describe(self, state):
        return "-".join(state)


def test_ucs_known_optimal_answer():
    res = ucs(ToyGraph())
    assert res.found and res.state == ("S", "B", "G") and res.cost == 4
    assert res.stats.nodes_expanded > 0 and res.stats.goal_tests > 0


def test_astar_known_optimal_answer():
    res = astar(ToyGraph())
    assert res.state == ("S", "B", "G") and res.cost == 4


def test_greedy_is_not_optimal_on_toy_graph():
    res = greedy(ToyGraph())
    assert res.found and res.cost == 5          # documented non-optimality


def test_negative_costs_rejected():
    class Neg(ToyGraph):
        edges = {"S": [("G", -1)], "G": []}
    with pytest.raises(ValueError):
        ucs(Neg())


def _small_problem(puzzle, solution, make_probs, seed, n):
    probs = make_probs(puzzle)
    g, p, _ = inject_misclassifications(puzzle, probs, puzzle, n, np.random.default_rng(seed))
    return CorrectionProblem(g, p)


@pytest.mark.parametrize("seed,n", [(0, 1), (1, 2), (2, 2), (3, 3), (4, 3)])
def test_search_matches_brute_force(puzzle, solution, make_probs, seed, n):
    prob = _small_problem(puzzle, solution, make_probs, seed, n)
    bf_cost, bf_state, _ = brute_force(prob)
    u = ucs(prob)
    a = astar(prob)
    gr = greedy(prob)
    assert u.found and math.isclose(u.cost, bf_cost)
    assert a.found and math.isclose(a.cost, bf_cost)       # admissible heuristic -> optimal
    assert gr.found and gr.cost >= bf_cost - 1e-9          # greedy: never better than optimal
    assert prob.heuristic(()) <= bf_cost + 1e-9            # admissibility at the root


def test_costs_are_non_negative(puzzle, make_probs):
    prob = _small_problem(puzzle, None, make_probs, 5, 2)
    assert all(cost >= 0 for alts in prob.alternatives.values() for _, cost in alts)


def test_confident_cells_are_never_candidates(puzzle, make_probs):
    prob = CorrectionProblem(puzzle, make_probs(puzzle))   # everything confident, no conflicts
    assert prob.candidate_cells == []
    assert ucs(prob).state == ()                           # solved directly
