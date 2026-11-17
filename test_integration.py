"""Integration tests: NN probabilities + Search + CSP (normal, edge, impossible)."""
import json

import numpy as np
import pytest
from PIL import Image

import config
from integration.correction import inject_misclassifications
from integration.pipeline import format_report, run_pipeline, solve_recognized


def corrupt(puzzle, make_probs, n, seed):
    return inject_misclassifications(puzzle, make_probs(puzzle), puzzle, n, np.random.default_rng(seed))


def test_valid_recognition_solved_directly(puzzle, solution, make_probs):
    res = solve_recognized(puzzle, make_probs(puzzle))
    assert res.status == "solved_directly" and res.solution == solution and res.corrections == []


@pytest.mark.parametrize("search", ["ucs", "greedy", "astar"])
def test_normal_one_injected_error_corrected(puzzle, solution, make_probs, search):     # NORMAL 4
    g, p, inj = corrupt(puzzle, make_probs, 1, 7)
    res = solve_recognized(g, p, search_algorithm=search)
    assert res.solution == solution
    assert [(c["cell"], c["to"]) for c in res.corrections] == [(inj[0]["cell"], inj[0]["true"])]
    assert all("explanation" in c and c["cost"] >= 0 for c in res.corrections)


@pytest.mark.parametrize("csp", ["bt", "fc", "ac3"])
def test_normal_two_injected_errors_corrected(puzzle, solution, make_probs, csp):       # NORMAL 5
    g, p, inj = corrupt(puzzle, make_probs, 2, 3)
    res = solve_recognized(g, p, csp_algorithm=csp)
    assert res.status == "solved_after_correction" and res.solution == solution
    assert {c["cell"] for c in res.corrections} == {r["cell"] for r in inj}


def test_edge_very_low_confidence(puzzle, make_probs):                                   # EDGE 7
    g, p, _ = corrupt(puzzle, make_probs, 1, 1)
    probs = make_probs(g, p=0.3)                 # every cell is now very uncertain
    res = solve_recognized(g, probs, max_corrections=2)
    assert res.search_stats["candidate_cells"] <= config.MAX_CANDIDATE_CELLS
    assert res.status in ("solved_after_correction", "no_solution")   # must not crash
    assert res.solution is None or res.corrected is not None


def test_edge_max_correction_limit_reached(puzzle, make_probs):                          # EDGE 8
    g, p, _ = corrupt(puzzle, make_probs, 2, 3)
    res = solve_recognized(g, p, max_corrections=1)
    assert res.status == "no_solution" and config.NO_SOLUTION_MSG in res.message
    assert len(res.least_confident) > 0
    assert config.NO_SOLUTION_MSG in format_report(res)


def test_impossible_contradictory_givens(puzzle, make_probs):                            # IMPOSSIBLE 9
    g = list(puzzle)
    g[2] = 5                                  # confidently-read duplicate 5 in row 1
    res = solve_recognized(g, make_probs(g), max_corrections=0)
    assert res.status == "no_solution" and res.conflicts and not res.ac3_consistent


def test_impossible_within_correction_limit(puzzle, make_probs):                         # IMPOSSIBLE 10
    g, p, _ = corrupt(puzzle, make_probs, 4, 0)
    res = solve_recognized(g, p, max_corrections=3)
    assert res.status == "no_solution" and res.solution is None
    assert res.search_stats["status"] == "not_found"


def test_nn_off_uses_uniform_costs(puzzle, solution, make_probs):
    g, p, _ = corrupt(puzzle, make_probs, 1, 7)
    res = solve_recognized(g, None, use_nn=False)
    assert res.settings["use_nn"] is False
    assert all(c["cost"] == 1.0 for c in res.corrections)


@pytest.mark.skipif(not (config.PUZZLE_DIR / "puzzle_00.png").exists() or not config.MODEL_PATH.exists(),
                    reason="needs data and trained model")
def test_full_pipeline_from_image():
    rec = json.loads((config.PUZZLE_DIR / "puzzle_00.json").read_text())
    res = run_pipeline(Image.open(config.PUZZLE_DIR / "puzzle_00.png"))
    assert res.solution == rec["solution"]
