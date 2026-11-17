# Data Dictionary – P7 Handwritten Sudoku Reader and Solver

This project uses **one real dataset (MNIST)** and **several kinds of generated data**.
The project does **not** read real photographs of handwritten Sudoku puzzles: every Sudoku image is a
generated mosaic of real MNIST digits.

| Category | Real or generated | Where it comes from |
|---|---|---|
| MNIST digit images and labels | **Real** | MNIST (LeCun, Cortes & Burges), http://yann.lecun.com/exdb/mnist/ |
| Train / validation / test split indices | Derived (deterministic, seed 42) | `data/make_data.py` |
| Sudoku puzzles and solutions | **Generated** | `data/puzzle_gen.py` |
| Sudoku mosaic images | **Generated** (built from real digit images) | `data/puzzle_gen.py` |
| Injected misclassifications | **Generated** (simulated NN confusions) | `integration/correction.py::inject_misclassifications` |
| NN probabilities | Model output (computed) | `nn/predict.py` |
| Experiment results | Measured by running the code | `eval/experiments.py` |

---

## 1. Raw MNIST files – `data/raw/` (REAL)

Downloaded by `python -m data.make_data`. Each file's MD5 checksum is verified. If the official server refuses
the download, byte-identical mirrors are tried (same checksum).

| File | Content | Type | Shape |
|---|---|---|---|
| `train-images-idx3-ubyte.gz` | official training images | uint8 pixels 0–255 | 60,000 × 28 × 28 |
| `train-labels-idx1-ubyte.gz` | official training labels | uint8 digit 0–9 | 60,000 |
| `t10k-images-idx3-ubyte.gz` | official test images | uint8 pixels 0–255 | 10,000 × 28 × 28 |
| `t10k-labels-idx1-ubyte.gz` | official test labels | uint8 digit 0–9 | 10,000 |

## 2. Processed dataset – `data/processed/mnist_1to9.npz` (REAL, filtered)

Steps (seed = 42, fully reproducible):
1. Shuffle the 60,000 official training indices with `numpy.random.default_rng(42).permutation`.
2. The first 50,000 form **train**; the remaining 10,000 form **validation**.
3. Remove digit **0** from train, validation and the official test set. The test set is otherwise untouched.

Because the zeros are removed *after* the 50k/10k split, the final sizes are smaller than 50,000 / 10,000:

| Split | Images (digits 1–9) |
|---|---|
| train | 45,073 |
| val | 9,004 |
| test | 9,020 |

| Field | Type | Description |
|---|---|---|
| `x_train`, `x_val`, `x_test` | uint8 array (N, 28, 28) | grayscale pixels, white ink on black. `load_processed()` converts them to **float32 in [0, 1]** (÷255). |
| `y_train`, `y_val`, `y_test` | int64 array (N,) | true digit **1–9**. The CNN uses class index = digit − 1. |
| `idx_train`, `idx_val`, `idx_test` | int64 array (N,) | position of the image in the original MNIST file (traceability). |

## 3. Generated Sudoku puzzles – `data/puzzles/` (GENERATED)

Created by `python -m data.make_data` (6 sample puzzles); the experiments and the Streamlit app generate more
on the fly with the same code.

How a puzzle is generated (`data/puzzle_gen.py`):
1. `random_solution(rng)`: randomised backtracking fills an empty grid → a complete valid Sudoku.
2. `generate_puzzle(n_givens, rng, unique=True)`: cells are removed in random order; a removal is kept only
   if the CSP solver still finds **exactly one** solution. Repeats until `n_givens` clues remain.
3. `sample_digit_images`: for every clue a random MNIST **test** image of that digit is chosen (the CNN never saw
   these images during training).
4. `render_mosaic`: the 28×28 digits are pasted into a 296×296 sheet (4 px margin, 32 px cells, 2 px padding),
   thin and thick grid lines are drawn, and the image is inverted to dark ink on white paper.

`puzzle_XX.json` fields:

| Field | Type | Description |
|---|---|---|
| `name` | string | file stem, e.g. `puzzle_00` |
| `seed` | int | RNG seed used to create the puzzle |
| `n_givens` | int | number of clues (non-empty cells) |
| `puzzle` | list of 81 ints | row-major grid, 0 = empty cell |
| `solution` | list of 81 ints | the unique solution (ground truth) |
| `mnist_test_index` | list of 81 ints | index into the processed **test** split of the image used for each cell, −1 for empty cells |

`puzzle_XX.png`: 296×296 grayscale image (generated). `index.json`: list of `{name, n_givens}`.

## 4. Recognition output (computed by the CNN) – `nn/predict.py::cell_probabilities`

| Field | Type | Description |
|---|---|---|
| `grid` | list of 81 ints | NN reading: arg-max digit, 0 for blank cells |
| `probs` | float array (81, 9) | Softmax P(digit = d) for d = 1..9; all-zero row for blank cells |
| `conf` | list of floats / None | max probability of the cell; None for blank cells |
| `blank` | list of bool | brightness rule: mean ink of the central 20×20 region < 0.04 |

## 5. Injected misclassifications (GENERATED) – `inject_misclassifications`

Used to create controlled test cases. For each of *n* randomly chosen clue cells that the NN currently reads
correctly:

| Field | Type | Description |
|---|---|---|
| `cell`, `cell_name` | int, string | cell index 0–80 and name `R{row}C{col}` |
| `true` | int | the real digit |
| `injected` | int | wrong digit drawn uniformly from the other 8 digits; becomes the NN's arg-max |
| `p_injected` | float | p(wrong) ~ U(0.45, 0.75) |
| `p_true` | float | p(true) = (1 − p(wrong)) · U(0.5, 0.95), so the true digit is the runner-up |

The remaining probability mass is spread over the other 7 digits with a Dirichlet(1) draw. This models the
example in the brief (prediction 1 at 52 %, true 7 at 45 %). **Limitation:** real confusions are not always
low-confidence nor always have the true digit as runner-up; the CNN's own (natural) errors, which are not
controlled, are counted separately in every experiment (`natural_errors`).

## 6. Result tables – `results/tables/` (MEASURED)

| File | Rows | Main columns |
|---|---|---|
| `nn_test_metrics.csv` | CNN, Logistic Regression | accuracy, precision_macro, recall_macro, f1_macro, ece, mean_confidence |
| `nn_classification_report.csv` | digits 1–9 + averages | precision, recall, f1-score, support |
| `nn_calibration_bins.csv` | 15 confidence bins | bin_low, bin_high, count, accuracy, confidence |
| `blank_detection.csv` | 3 input types | n, flagged_blank, min/mean central ink |
| `exp1_search_raw.csv` | one row per (seed, puzzle, n_injected, algorithm) | status, recovered, nodes_generated, nodes_expanded, goal_tests, pruned, time_s, n_corrections, total_cost, natural_errors |
| `exp1_optimality.csv` | small problems (≤ 8 candidate cells, ≤ 3 injections) | brute_force_cost, UCS/A*/Greedy cost, *_optimal flags |
| `exp2_csp_raw.csv` | one row per (seed, puzzle, n_givens, ordering, algorithm) | status, correct, assignments, backtracks, constraint_checks, domain_reductions, time_s |
| `exp3_correction_raw.csv` | one row per (seed, puzzle, n_injected, method) | solvable, recovered, n_corrections, correction_cost, goal_tests |
| `*_summary.csv` | one row per condition | `<metric>_mean`, `<metric>_std` over the 5 per-seed means, `n_runs` |

Definitions: **recovered** = final solution equals the ground-truth solution; **solvable** = the method produced
a puzzle with exactly one solution; **correction cost** = Σ ln(p(predicted)/p(new digit)) using the NN's
probabilities.
