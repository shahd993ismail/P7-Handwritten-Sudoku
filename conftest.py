"""Shared fixtures. Run all tests from the project root with:  python -m pytest -v"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from csp.sudoku import parse_grid  # noqa: E402

# A classic puzzle with a unique solution (well-known example).
PUZZLE = parse_grid("530070000600195000098000060800060003400803001700020006060000280000419005000080079")
SOLUTION = parse_grid("534678912672195348198342567859761423426853791713924856961537284287419635345286179")


@pytest.fixture
def puzzle():
    return list(PUZZLE)


@pytest.fixture
def solution():
    return list(SOLUTION)


def confident_probs(grid, p: float = 0.999) -> np.ndarray:
    """Synthetic NN output: the read digit gets probability p, the rest is spread evenly."""
    probs = np.zeros((81, 9))
    for i, v in enumerate(grid):
        if v:
            probs[i] = (1 - p) / 8
            probs[i][v - 1] = p
    return probs


@pytest.fixture
def make_probs():
    return confident_probs
