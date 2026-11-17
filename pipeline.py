"""End-to-end pipeline:  image -> CNN -> AC-3 check -> Search (+CSP goal test) -> solution.

STEP 1  input a (generated) handwritten Sudoku image
STEP 2  cut it into 81 cells, detect blanks, CNN gives P(digit) per cell
STEP 3  CSP checks the recognised grid: duplicate digits + AC-3 consistency
STEP 4  if it is valid and solvable it is solved directly (the empty
        correction set is the first node UCS/A*/Greedy goal-tests)
STEP 5  otherwise Search proposes corrections in order of cost
STEP 6  every popped candidate is goal-tested by the CSP solver
STEP 7  the first goal popped is returned (UCS: the cheapest one)
STEP 8  everything needed for display is returned in a PipelineResult

Nothing is ever changed silently: every correction carries an explanation,
and failures return config.NO_SOLUTION_MSG plus the least-confident cells.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np
from PIL import Image

import config
from csp.ac3 import ac3_check
from csp.sudoku import ALGORITHM_LABELS, cell_name, find_conflicts, normalise_algorithm, solve, validate_grid
from data.puzzle_gen import IMAGE_PX, MARGIN
from integration.correction import CorrectionProblem
from search.astar import astar
from search.greedy import greedy
from search.ucs import ucs

SEARCH_ALGORITHMS = {"ucs": ucs, "greedy": greedy, "astar": astar}
SEARCH_LABELS = {"ucs": "UCS", "greedy": "Greedy", "astar": "A*"}
_SEARCH_ALIASES = {"ucs": "ucs", "uniform": "ucs", "greedy": "greedy", "gbfs": "greedy",
                   "astar": "astar", "a*": "astar", "a-star": "astar"}


def normalise_search(name: str) -> str:
    key = str(name).strip().lower()
    if key not in _SEARCH_ALIASES:
        raise ValueError(f"unknown search algorithm '{name}'. Use ucs, greedy or astar")
    return _SEARCH_ALIASES[key]


# ================================================================ image -> cells
def _to_ink_array(image) -> np.ndarray:
    """Any PIL image / array -> float32 grayscale in [0,1] with INK = 1."""
    if isinstance(image, Image.Image):
        image = np.array(image.convert("L"))
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
    if arr.ndim != 2 or min(arr.shape) < 50:
        raise ValueError(f"expected a grayscale/RGB image of at least 50x50 pixels, got shape {arr.shape}")
    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr /= 255.0
    if arr.mean() > 0.5:                 # white paper, dark ink -> invert
        arr = 1.0 - arr
    return arr


def _find_grid_square(ink: np.ndarray) -> np.ndarray:
    """Locate the outer grid border (largest 4-corner contour) and warp it to a square."""
    u8 = (ink * 255).astype(np.uint8)
    _, th = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    size = 9 * config.CELL_PX
    if contours:
        c = max(contours, key=cv2.contourArea)
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.25 * ink.size:
            pts = approx.reshape(4, 2).astype(np.float32)
            s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
            src = np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], np.float32)
            dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], np.float32)
            return cv2.warpPerspective(ink, cv2.getPerspectiveTransform(src, dst), (size, size))
    return cv2.resize(ink, (size, size), interpolation=cv2.INTER_AREA)   # assume the grid fills the image


def _mnist_normalise(cell: np.ndarray) -> np.ndarray:
    """Re-centre a free-form cell like MNIST: fit the digit into 20x20, centre of mass at the centre."""
    out = np.zeros((28, 28), np.float32)
    ys, xs = np.nonzero(cell > 0.25)
    if len(ys) == 0:
        return out
    crop = cell[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    scale = 20.0 / max(h, w)
    crop = cv2.resize(crop, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    h, w = crop.shape
    y0, x0 = (28 - h) // 2, (28 - w) // 2
    out[y0:y0 + h, x0:x0 + w] = crop
    total = out.sum()
    if total > 0:
        cy, cx = np.array(np.nonzero(out)).astype(float) @ out[np.nonzero(out)] / total
        out = np.roll(out, (int(round(14 - cy)), int(round(14 - cx))), axis=(0, 1))
    return np.clip(out, 0, 1)


def extract_cells(image) -> tuple[np.ndarray, str]:
    """Return (81,28,28) cell images in [0,1] (ink = 1) and the method used.

    * Images produced by data/puzzle_gen.py (296x296) are cut with the exact
      known geometry - each crop is the original 28x28 MNIST digit.
    * Any other image: the outer grid is located with OpenCV, warped to a
      square, cut into 81 cells, grid-line borders are removed and each digit
      is re-centred MNIST-style. This works for clean, printed-style grids;
      real photographs are outside the scope of the project.
    """
    ink = _to_ink_array(image)
    cells = np.zeros((81, 28, 28), np.float32)
    pad = (config.CELL_PX - 28) // 2
    if ink.shape == (IMAGE_PX, IMAGE_PX):
        for i in range(81):
            r, c = divmod(i, 9)
            y, x = MARGIN + r * config.CELL_PX + pad, MARGIN + c * config.CELL_PX + pad
            cells[i] = ink[y:y + 28, x:x + 28]
        return cells, "exact generated layout"
    sq = _find_grid_square(ink)
    k = config.CELL_PX
    for i in range(81):
        r, c = divmod(i, 9)
        crop = sq[r * k:(r + 1) * k, c * k:(c + 1) * k].copy()
        b = max(3, k // 8)                         # wipe grid-line remnants
        crop[:b, :], crop[-b:, :], crop[:, :b], crop[:, -b:] = 0, 0, 0, 0
        cells[i] = _mnist_normalise(cv2.resize(crop, (28, 28), interpolation=cv2.INTER_AREA)) \
            if crop.max() > 0.25 else 0.0
    return cells, "grid detection + MNIST-style re-centring"


# ================================================================ result object
@dataclass
class PipelineResult:
    status: str                         # "solved_directly" | "solved_after_correction" | "no_solution"
    message: str
    recognized: list[int]
    probs: np.ndarray
    confidence: list
    corrected: list[int] | None
    solution: list[int] | None
    corrections: list[dict]
    total_cost: float
    conflicts: list[dict]
    ac3_consistent: bool
    ac3_stats: dict
    search_stats: dict
    csp_stats: dict
    search_trace: list[str]
    candidate_cells: list[dict]
    least_confident: list[dict]
    settings: dict
    confidence_threshold: float = config.CONFIDENCE_THRESHOLD
    extraction: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def solved(self) -> bool:
        return self.solution is not None

    def summary_dict(self) -> dict:
        d = asdict(self)
        d.pop("probs")
        return d


def least_confident_cells(grid, probs, n: int = 6) -> list[dict]:
    rows = []
    for i in range(81):
        if grid[i] and probs is not None and np.asarray(probs[i]).sum() > 0:
            p = np.asarray(probs[i])
            order = np.argsort(-p)[:3]
            rows.append({"cell": i, "cell_name": cell_name(i), "predicted": grid[i], "confidence": float(p.max()),
                         "alternatives": ", ".join(f"{k + 1} ({p[k]:.3f})" for k in order[1:])})
    return sorted(rows, key=lambda r: r["confidence"])[:n]


# ================================================================ main entry points
def solve_recognized(grid, probs=None, conf=None, *, search_algorithm: str = "ucs", csp_algorithm: str = "ac3",
                     max_corrections: int = config.MAX_CORRECTIONS, use_nn: bool = True,
                     require_unique: bool = config.REQUIRE_UNIQUE,
                     confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
                     top_k: int = config.TOP_K_ALTERNATIVES,
                     max_expansions: int = config.MAX_SEARCH_EXPANSIONS) -> PipelineResult:
    """Steps 3-8 on an already recognised grid (+ NN probabilities)."""
    grid = validate_grid(grid)
    s_key, c_key = normalise_search(search_algorithm), normalise_algorithm(csp_algorithm)
    settings = {"search": SEARCH_LABELS[s_key], "csp": ALGORITHM_LABELS[c_key], "max_corrections": max_corrections,
                "use_nn": use_nn, "require_unique": require_unique, "confidence_threshold": confidence_threshold,
                "top_k": top_k}
    probs_arr = None if probs is None else np.asarray(probs, dtype=np.float64)
    if probs_arr is not None and conf is None:
        conf = [float(probs_arr[i].max()) if grid[i] else None for i in range(81)]
    conf = conf if conf is not None else [None] * 81

    # STEP 3: conflict detection (duplicates + AC-3)
    conflicts = find_conflicts(grid)
    ac3_ok, _, ac3_stats = ac3_check(grid)

    # STEPS 4-7: search over corrections; the CSP is the goal test
    problem = CorrectionProblem(grid, probs_arr, conf, use_nn=use_nn, max_corrections=max_corrections,
                                top_k=top_k, confidence_threshold=confidence_threshold,
                                require_unique=require_unique, csp_algorithm=c_key)
    sres = SEARCH_ALGORITHMS[s_key](problem, max_expansions=max_expansions)
    search_stats = sres.stats.as_dict()
    search_stats.update({"status": sres.status, "csp_goal_tests": problem.csp_calls,
                         "csp_time_in_search_s": problem.csp_time,
                         "candidate_cells": len(problem.candidate_cells),
                         "branching_factor_max": sum(len(a) for a in problem.alternatives.values())})
    candidates = [{"cell_name": cell_name(c), "predicted": grid[c],
                   "confidence": None if conf[c] is None else round(conf[c], 4),
                   "alternatives (digit:cost)": ", ".join(f"{d}:{cost:.3f}" for d, cost in problem.alternatives[c]),
                   "reason": problem.reasons[c]} for c in problem.candidate_cells]
    lc = least_confident_cells(grid, probs_arr)

    if sres.found:
        corrections = problem.explain(sres.state)
        corrected = problem.apply(sres.state)
        # final, clean CSP run on the corrected puzzle with the chosen algorithm (for statistics)
        final = solve(corrected, c_key, count_limit=1)
        status = "solved_directly" if not corrections else "solved_after_correction"
        msg = ("Recognised puzzle is valid - solved directly without corrections." if not corrections else
               f"Solved after {len(corrections)} correction(s) with total cost {sres.cost:.3f}.")
        return PipelineResult(status, msg, grid, probs_arr, conf, corrected, final.solution, corrections,
                              float(sres.cost), conflicts, ac3_ok, ac3_stats.as_dict(), search_stats,
                              final.stats.as_dict(), sres.trace, candidates, lc, settings, confidence_threshold)
    reason = {"not_found": "every correction set within the limit was tried",
              "limit_reached": f"search stopped after {max_expansions} expansions"}[sres.status]
    return PipelineResult("no_solution", f"{config.NO_SOLUTION_MSG} ({reason})", grid, probs_arr, conf, None, None,
                          [], math.inf, conflicts, ac3_ok, ac3_stats.as_dict(), search_stats, {}, sres.trace,
                          candidates, lc, settings, confidence_threshold)


def recognise_image(image, model=None) -> dict:
    """Steps 1-2: image -> cells -> CNN probabilities."""
    from nn.predict import cell_probabilities
    cells, method = extract_cells(image)
    rec = cell_probabilities(cells, model)
    rec["cells"] = cells
    rec["extraction"] = method
    return rec


def run_pipeline(image, model=None, **kwargs) -> PipelineResult:
    """Full pipeline from an image."""
    rec = recognise_image(image, model)
    res = solve_recognized(rec["grid"], rec["probs"], rec["conf"], **kwargs)
    res.extraction = rec["extraction"]
    return res


def format_report(res: PipelineResult, truth: list[int] | None = None) -> str:
    """Plain-text report used by run_demo.py and the logs."""
    from csp.sudoku import grid_to_str
    L = ["=" * 72, f"SETTINGS: {res.settings}", "-" * 72, "RECOGNISED BY THE NN (raw prediction):",
         grid_to_str(res.recognized), "",
         f"Duplicate-digit conflicts: {len(res.conflicts)}"]
    for c in res.conflicts:
        L.append(f"   digit {c['digit']} repeated in {c['unit']} at " + ", ".join(cell_name(x) for x in c["cells"]))
    L.append(f"AC-3 consistency check: {'consistent' if res.ac3_consistent else 'INCONSISTENT (empty domain)'}"
             f"  [checks={res.ac3_stats['constraint_checks']}, reductions={res.ac3_stats['domain_reductions']}]")
    L.append(f"Candidate cells Search may change: {len(res.candidate_cells)}")
    for c in res.candidate_cells:
        L.append(f"   {c['cell_name']}: read {c['predicted']} conf={c['confidence']}  alts {c['alternatives (digit:cost)']}"
                 f"  ({c['reason']})")
    L += ["", "SEARCH TRACE (first lines):"] + ["   " + t for t in res.search_trace[:15]]
    if len(res.search_trace) > 15:
        L.append(f"   ... ({len(res.search_trace)} lines recorded)")
    s = res.search_stats
    L.append(f"SEARCH STATS: {s['algorithm']} status={s['status']} generated={s['nodes_generated']} "
             f"expanded={s['nodes_expanded']} goal_tests={s['goal_tests']} pruned={s['pruned_dead_states']} "
             f"time={s['time_s']:.4f}s csp_calls={s['csp_goal_tests']}")
    L.append("")
    if res.solved:
        L.append("CORRECTION LOG:" if res.corrections else "CORRECTION LOG: no correction needed")
        for c in res.corrections:
            L.append("   " + c["explanation"])
        L += [f"Total correction cost: {res.total_cost:.4f}", "", "CORRECTED PUZZLE:", grid_to_str(res.corrected), "",
              "FINAL SOLUTION:", grid_to_str(res.solution)]
        cs = res.csp_stats
        L.append(f"CSP STATS ({cs['algorithm']}): assignments={cs['assignments']} backtracks={cs['backtracks']} "
                 f"constraint_checks={cs['constraint_checks']} domain_reductions={cs['domain_reductions']} "
                 f"time={cs['time_s']:.4f}s")
        if truth is not None:
            L.append("MATCHES GROUND TRUTH: " + ("YES" if res.solution == list(truth) else "NO"))
    else:
        L += [res.message, "Least confident cells and their alternatives:"]
        for r in res.least_confident:
            L.append(f"   {r['cell_name']}: read {r['predicted']} (conf {r['confidence']:.3f}); alternatives {r['alternatives']}")
    L.append("=" * 72)
    return "\n".join(L)
