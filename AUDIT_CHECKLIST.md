# Final Audit Checklist

Status as delivered. "Verified" means the command was run and its output checked before delivery
(Python 3.13 + pinned packages, Linux, CPU only). Windows-specific steps (PowerShell activation, `py -3.11`)
follow standard usage but were not run on a Windows machine before delivery. Please run `python -m pytest -v`
once on your laptop.

## Required deliverables

| # | Deliverable | Status | Evidence |
|---|---|---|---|
| 1 | Source code: data, NN, Search, CSP, integration, evaluation, tests | ✅ Done | all folders of the required structure; `search/base.py`, `search/brute_force.py`, `data/puzzle_gen.py`, `config.py` added |
| 2 | `make_data.py`: download + preprocess with fixed seed, skips existing data | ✅ Verified | MD5-checked download with mirror fallback; reruns reuse existing files |
| 3 | DATA_DICTIONARY.md (fields, types, sources, real vs generated) | ✅ Done | `DATA_DICTIONARY.md` |
| 4 | Trained NN + training script + baseline comparison | ✅ Verified | `models/trained_model.pth`; CNN 99.08 % vs LogReg 92.24 % (`nn_test_metrics.csv`) |
| 5 | Search (UCS, Greedy, A*) with counters + known-answer tests | ✅ Verified | `tests/test_search.py` (toy graph with known optimum, brute-force comparisons) |
| 6 | CSP (BT, BT+FC, BT+FC+AC-3), separate and switchable, with counters | ✅ Verified | `csp.sudoku.solve(..., "bt" / "fc" / "ac3")`; different pruning shown in Exp. 2 |
| 7 | Integrated demo `run_demo.py` with plots | ✅ Verified | `results/logs/demo_log.txt`, `demo_pipeline.png`, `demo_cooperation.png` |
| 8 | README with setup, pinned dependencies, Windows commands | ✅ Done | `README.md`, `requirements.txt` |
| 9 | Tests: ≥ 5 normal, ≥ 3 edge, ≥ 2 impossible | ✅ Verified | 51 tests pass (see mapping below) |
| 10 | Experiments: 3 experiments, 5 seeds, mean ± std, CSV + PNG | ✅ Verified | `results/tables/exp*_summary.csv`, `results/figures/exp*.png` |
| 11 | REPORT.md (all required sections) | ✅ Done | `REPORT.md`; every number comes from the result tables |
| 12 | PRESENTATION.md (10-min outline + 5-min demo) | ✅ Done | includes search trace, CSP with/without FC/AC-3, NN ON/OFF, final solution |
| 13 | CONTRIBUTIONS.md template + Git commands, no fabricated history | ✅ Done | template left blank for the team |

## Functional requirements

| Requirement | Status | Where |
|---|---|---|
| Digits 1–9 only, normalised to [0,1], 50k/10k split, untouched test set | ✅ | `data/make_data.py` (sizes after removing 0s: 45,073 / 9,004 / 9,020, explained) |
| CNN: 28×28 input, 9 softmax outputs, CrossEntropyLoss, CPU/GPU, save/load | ✅ | `models/cnn_model.py`, `nn/train.py`, `nn/predict.py` |
| Accuracy, precision, recall, F1, confusion matrix, loss curves, calibration (ECE) | ✅ | `nn/evaluate.py` |
| Cell probabilities from digit probabilities; blank-cell brightness rule | ✅ | `nn/predict.py` (documented) |
| Generated handwritten Sudoku images from MNIST | ✅ | `data/puzzle_gen.py`, `data/puzzles/*.png` |
| MRV, LCV, FC domain pruning, AC-3, counters, NO SOLUTION, solution counting | ✅ | `csp/` |
| Correction cost = −log probability difference, non-negative | ✅ | `integration/correction.py` |
| Correction limit (default 3), top-k alternatives, confident cells only changed if in conflict | ✅ | `config.py`, `CorrectionProblem` |
| Admissible A* heuristic with proof and stated limitation | ✅ | `integration/correction.py` docstring, REPORT §7 |
| Nodes generated / expanded, goal tests, time, #corrections, total cost | ✅ | `search/base.py::SearchStats` |
| UCS optimality checked against brute force | ✅ | tests + `exp1_optimality.csv` (198/198) |
| Pipeline steps 1–8; explanation for every correction; NO SOLUTION message + least confident cells | ✅ | `integration/pipeline.py` |
| Streamlit: upload, generated puzzle, confidences, conflicts, correction, algorithm selectors, statistics | ✅ Verified headless (`streamlit.testing`) | `app.py` |
| NN ON/OFF comparison with a defined baseline (uniform cost) | ✅ | `use_nn=False`; demo, app, Exp. 3 |
| Relative paths only, input validation, graceful errors | ✅ | `config.py`; `validate_grid`, image checks |

## Test mapping

| Category | Required case | Test |
|---|---|---|
| Normal 1 | valid Sudoku solved | `test_csp.py::test_normal_valid_sudoku_is_solved` (×3 algorithms) |
| Normal 2 | one missing cell | `test_csp.py::test_normal_one_missing_cell` |
| Normal 3 | two missing cells | `test_csp.py::test_normal_two_missing_cells` |
| Normal 4 | one injected misclassification corrected | `test_integration.py::test_normal_one_injected_error_corrected` (×3 searches) |
| Normal 5 | two injected misclassifications corrected | `test_integration.py::test_normal_two_injected_errors_corrected` (×3 CSPs) |
| Edge 6 | many empty cells | `test_csp.py::test_edge_many_empty_cells` (empty grid + 56 blanks) |
| Edge 7 | very low-confidence predictions | `test_integration.py::test_edge_very_low_confidence` |
| Edge 8 | maximum correction limit reached | `test_integration.py::test_edge_max_correction_limit_reached` |
| Impossible 9 | contradictory fixed givens | `test_csp.py::test_impossible_contradictory_givens`, `test_integration.py::test_impossible_contradictory_givens` |
| Impossible 10 | unsolvable within the correction limit | `test_integration.py::test_impossible_within_correction_limit` |

## Known limitations (also in REPORT §11)

* Images are generated mosaics; real photographs are not evaluated.
* Injected errors keep the true digit as the runner-up (documented); natural CNN errors are reported separately.
* 4 of 25 static-order BT runs at 25 givens hit the 200,000-assignment cap (reported as lower bounds).
