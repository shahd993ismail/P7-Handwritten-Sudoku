"""Central configuration for the P7 Handwritten Sudoku Reader and Solver.

All paths are relative to the project root so the project runs unchanged on
Windows, Linux and macOS. Change hyper-parameters here instead of inside the
individual modules.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PUZZLE_DIR = DATA_DIR / "puzzles"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "trained_model.pth"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"
LOG_DIR = RESULTS_DIR / "logs"
PROCESSED_FILE = PROCESSED_DIR / "mnist_1to9.npz"

# ---------------------------------------------------------------- data
SEED = 42
N_TRAIN_SPLIT = 50_000      # taken from the 60k official training set (before removing 0s)
N_VAL_SPLIT = 10_000        # the remaining 10k of the official training set
NUM_CLASSES = 9             # digits 1..9 -> class indices 0..8

# ---------------------------------------------------------------- CNN
EPOCHS = 6
BATCH_SIZE = 128
LEARNING_RATE = 1e-3

# ---------------------------------------------------------------- Sudoku image layout
CELL_PX = 32                # each cell is 32x32 pixels (MNIST 28x28 + 2px padding)
GRID_LINE_THIN = 1
GRID_LINE_THICK = 3
BLANK_INK_THRESHOLD = 0.04  # mean brightness in [0,1] of the cell centre below which the cell is blank

# ---------------------------------------------------------------- Search / correction
MAX_CORRECTIONS = 3         # maximum number of digit changes Search may propose
TOP_K_ALTERNATIVES = 2      # how many alternative digits per cell (NN top-k after the argmax)
CONFIDENCE_THRESHOLD = 0.90 # cells below this NN confidence are "uncertain" and may be changed
MAX_CANDIDATE_CELLS = 12    # cap on how many cells Search may touch (lowest-confidence first)
MIN_ALT_PROB = 1e-6         # alternatives below this probability are ignored
MAX_SEARCH_EXPANSIONS = 20_000
REQUIRE_UNIQUE = True       # goal = corrected puzzle with exactly one solution

# ---------------------------------------------------------------- CSP
CSP_MAX_ASSIGNMENTS = 200_000   # safety limit per solve (reported as "limit reached")

# ---------------------------------------------------------------- experiments
EXPERIMENT_SEEDS = [0, 1, 2, 3, 4]

NO_SOLUTION_MSG = "NO SOLUTION FOUND WITHIN THE ALLOWED CORRECTION LIMIT."


def ensure_dirs() -> None:
    """Create every output directory if it does not exist yet."""
    for d in (RAW_DIR, PROCESSED_DIR, PUZZLE_DIR, MODEL_DIR, FIG_DIR, TABLE_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
