"""Generated data: Sudoku puzzles and handwritten-looking mosaic images.

IMPORTANT: everything produced here is GENERATED data. Only the digit images
themselves are real (taken from the MNIST *test* split, so the CNN has never
seen them during training).

Layout of a mosaic image (grayscale uint8, white paper, dark ink):
    MARGIN px border, then 9x9 cells of CELL_PX (=32) px each.
    A 28x28 MNIST digit is pasted in the centre of each given cell
    (2 px padding). Thin lines separate cells, thick lines separate boxes.
"""
from __future__ import annotations

import numpy as np

import config
from csp.sudoku import PEERS, count_solutions

MARGIN = 4
GRID_PX = 9 * config.CELL_PX
IMAGE_PX = GRID_PX + 2 * MARGIN       # 296 x 296


# ---------------------------------------------------------------- Sudoku generation
def random_solution(rng: np.random.Generator) -> list[int]:
    """A uniformly shuffled complete Sudoku grid (randomised backtracking)."""
    grid = [0] * 81

    def fill(pos: int) -> bool:
        if pos == 81:
            return True
        used = {grid[p] for p in PEERS[pos]}
        for v in rng.permutation(9) + 1:
            if int(v) not in used:
                grid[pos] = int(v)
                if fill(pos + 1):
                    return True
        grid[pos] = 0
        return False

    fill(0)
    return grid


def generate_puzzle(n_givens: int, rng: np.random.Generator, unique: bool = True,
                    max_restarts: int = 20) -> tuple[list[int], list[int]]:
    """Return (puzzle, solution) with exactly *n_givens* clues.

    unique=True : cells are removed one at a time in random order and a removal
                  is kept only if the puzzle still has exactly one solution.
                  If the target cannot be reached, a new solution is tried.
    unique=False: random cells are removed (the puzzle is solvable because it
                  is derived from a full solution, but may have several).
    """
    if not 17 <= n_givens <= 81:
        raise ValueError("n_givens must be in 17..81")
    best = None
    for _ in range(max_restarts):
        solution = random_solution(rng)
        puzzle = list(solution)
        order = rng.permutation(81)
        if not unique:
            for c in order[: 81 - n_givens]:
                puzzle[int(c)] = 0
            return puzzle, solution
        filled = 81
        for c in order:
            if filled == n_givens:
                break
            c = int(c)
            keep = puzzle[c]
            puzzle[c] = 0
            if count_solutions(puzzle, limit=2) == 1:
                filled -= 1
            else:
                puzzle[c] = keep
        if filled == n_givens:
            return puzzle, solution
        best = (puzzle, solution)
    raise RuntimeError(f"could not generate a unique puzzle with {n_givens} givens "
                       f"(closest had {sum(1 for v in best[0] if v)})")


# ---------------------------------------------------------------- image mosaics
def sample_digit_images(puzzle: list[int], x_digits: np.ndarray, y_digits: np.ndarray,
                        rng: np.random.Generator) -> tuple[list[np.ndarray | None], list[int]]:
    """For every given cell pick a random MNIST image of the same digit."""
    by_digit = {d: np.flatnonzero(y_digits == d) for d in range(1, 10)}
    imgs, src = [], []
    for v in puzzle:
        if v:
            idx = int(rng.choice(by_digit[v]))
            imgs.append(x_digits[idx])
            src.append(idx)
        else:
            imgs.append(None)
            src.append(-1)
    return imgs, src


def render_mosaic(cell_images: list[np.ndarray | None], paper_noise: float = 0.0,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """Paste 28x28 digits (float [0,1], white ink on black) into a Sudoku sheet.

    Returns a uint8 image with white paper and dark ink, like a scanned sheet.
    """
    if len(cell_images) != 81:
        raise ValueError("need 81 cell images (None for blank cells)")
    ink = np.zeros((IMAGE_PX, IMAGE_PX), dtype=np.float32)
    pad = (config.CELL_PX - 28) // 2
    for i, img in enumerate(cell_images):
        if img is None:
            continue
        img = np.asarray(img, dtype=np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        r, c = divmod(i, 9)
        y0 = MARGIN + r * config.CELL_PX + pad
        x0 = MARGIN + c * config.CELL_PX + pad
        ink[y0:y0 + 28, x0:x0 + 28] = np.maximum(ink[y0:y0 + 28, x0:x0 + 28], img)
    # grid lines
    for k in range(10):
        pos = MARGIN + k * config.CELL_PX
        w = config.GRID_LINE_THICK if k % 3 == 0 else config.GRID_LINE_THIN
        lo, hi = pos - w // 2, pos - w // 2 + w
        ink[max(lo, 0):hi, MARGIN - 1:IMAGE_PX - MARGIN + 2] = 1.0
        ink[MARGIN - 1:IMAGE_PX - MARGIN + 2, max(lo, 0):hi] = 1.0
    if paper_noise > 0:
        rng = rng or np.random.default_rng(0)
        ink = np.clip(ink + rng.normal(0, paper_noise, ink.shape), 0, 1)
    return (255 * (1.0 - ink)).astype(np.uint8)
