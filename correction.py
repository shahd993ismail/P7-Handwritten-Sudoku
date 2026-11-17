"""The digit-correction search problem that links the NN, Search and the CSP.

STATE      A set of corrections {(cell, new_digit), ...} applied to the NN's
           predicted grid, stored as a tuple sorted by cell index. The empty
           tuple is the initial state (= the raw NN prediction).
ACTION     Change ONE more candidate cell to one of its top-k alternative
           digits (k = config.TOP_K_ALTERNATIVES). Blank cells are never
           touched and each cell is corrected at most once.
           Canonical ordering: a new correction must be on a cell with a
           larger index than every cell already corrected. Every correction
           SET is therefore reached by exactly one path, so the state space
           is a tree (no duplicates, no permutations of the same set).
COST       NN ON : cost(cell -> d) = -ln p(d) - ( -ln p(pred) ) = ln(p(pred)/p(d))
                   >= 0 because pred is the arg-max. The total cost of a state
                   equals  -ln P(corrected grid) + ln P(predicted grid)  under
                   the NN's independent per-cell distributions, so the
                   minimum-cost goal is the MOST LIKELY solvable reading.
           NN OFF: uniform cost 1 per correction (baseline, no probabilities).
CANDIDATES (which cells may change - limits the branching factor)
           NN ON : cells whose confidence < config.CONFIDENCE_THRESHOLD, plus
                   cells that take part in a duplicate-digit conflict (the
                   explicit justification for touching a confident cell);
                   at most config.MAX_CANDIDATE_CELLS, lowest confidence first.
                   Confident cells that are not in conflict are NEVER changed.
           NN OFF: cells in duplicate conflicts; if there are none, every
                   given (the baseline has no confidence information), with
                   all 8 other digits as alternatives.
GOAL       The corrected puzzle has no duplicate digits, passes AC-3, and the
           CSP solver finds a solution (exactly one solution when
           require_unique=True, which is the default because the generated
           puzzles are unique by construction).
DEPTH      At most config.MAX_CORRECTIONS corrections.

HEURISTIC (A* and Greedy) - admissible lower bound from duplicate conflicts
           Let P be the set of peer pairs (a, b) holding the same digit in the
           current grid. Any goal must change at least one cell of every pair
           (a correction never blanks a cell). Choose a matching M of P (pairs
           sharing no cell); each pair in M needs a DIFFERENT changed cell, and
           changing cell c costs at least m(c) = its cheapest alternative.
               h(n) = sum over (a,b) in M of min(m(a), m(b))
           is therefore a lower bound on the remaining cost (admissible), for
           any matching. If a pair contains no cell that may still change
           (not a candidate, or index below the last correction because of the
           canonical ordering) or |M| exceeds the remaining correction budget,
           no goal is reachable and h = infinity (the state is pruned).
           When the grid has no duplicates but is still unsolvable, h = 0:
           the heuristic is weak there and A* behaves like UCS.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import config
from csp.ac3 import ac3_check
from csp.sudoku import cell_name, conflict_pairs, solve


@dataclass
class GoalInfo:
    reason: str
    grid: list[int] | None = None
    solution: list[int] | None = None
    csp_stats: dict | None = None


class CorrectionProblem:
    def __init__(self, grid, probs=None, conf=None, *, use_nn: bool = True,
                 max_corrections: int = config.MAX_CORRECTIONS,
                 top_k: int = config.TOP_K_ALTERNATIVES,
                 confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
                 max_candidate_cells: int = config.MAX_CANDIDATE_CELLS,
                 require_unique: bool = config.REQUIRE_UNIQUE,
                 csp_algorithm: str = "ac3"):
        if len(grid) != 81:
            raise ValueError("grid must have 81 cells")
        if max_corrections < 0:
            raise ValueError("max_corrections must be >= 0")
        if use_nn and (probs is None or np.asarray(probs).shape != (81, 9)):
            raise ValueError("NN-guided correction needs an (81,9) probability matrix")
        self.grid = [int(v) for v in grid]
        self.probs = None if probs is None else np.asarray(probs, dtype=np.float64)
        self.conf = conf if conf is not None else (
            [float(self.probs[i].max()) if self.grid[i] else None for i in range(81)]
            if self.probs is not None else [None] * 81)
        self.use_nn = use_nn
        self.max_corrections = max_corrections
        self.require_unique = require_unique
        self.csp_algorithm = csp_algorithm
        self.confidence_threshold = confidence_threshold
        self.csp_calls = 0
        self.csp_time = 0.0
        self.alternatives: dict[int, list[tuple[int, float]]] = {}
        self.reasons: dict[int, str] = {}
        self._build_candidates(top_k, confidence_threshold, max_candidate_cells)
        self.min_alt_cost = {c: min(cost for _, cost in alts) for c, alts in self.alternatives.items()}

    # ------------------------------------------------------------ candidates
    def _build_candidates(self, top_k, threshold, max_cells) -> None:
        givens = [i for i in range(81) if self.grid[i]]
        in_conflict = {c for pair in conflict_pairs(self.grid) for c in pair}
        if self.use_nn:
            chosen = [i for i in givens if self.conf[i] is not None and self.conf[i] < threshold]
            chosen = sorted(set(chosen) | in_conflict, key=lambda i: (self.conf[i] or 0.0, i))[:max_cells]
            for c in chosen:
                p = self.probs[c]
                pred = self.grid[c]
                p_pred = max(float(p[pred - 1]), 1e-12)
                order = [int(k) + 1 for k in np.argsort(-p, kind="stable") if int(k) + 1 != pred]
                alts = []
                for d in order[:top_k]:
                    p_d = float(p[d - 1])
                    if p_d < config.MIN_ALT_PROB:
                        continue
                    alts.append((d, max(0.0, math.log(p_pred) - math.log(p_d))))
                if alts:
                    self.alternatives[c] = alts
                    low = self.conf[c] < threshold
                    self.reasons[c] = (f"low NN confidence ({self.conf[c]:.2f} < {threshold:.2f})" if low
                                       else "confident cell, but involved in a duplicate-digit conflict")
        else:
            chosen = sorted(in_conflict) if in_conflict else givens
            for c in chosen:
                self.alternatives[c] = [(d, 1.0) for d in range(1, 10) if d != self.grid[c]]
                self.reasons[c] = ("involved in a duplicate-digit conflict" if in_conflict
                                   else "no conflicts visible - every given is a candidate (NN OFF)")
        self.candidate_cells = sorted(self.alternatives)

    # ------------------------------------------------------------ search interface
    def initial_state(self) -> tuple:
        return ()

    def apply(self, state) -> list[int]:
        g = list(self.grid)
        for cell, d in state:
            g[cell] = d
        return g

    def successors(self, state):
        if len(state) >= self.max_corrections:
            return
        last = state[-1][0] if state else -1
        for cell in self.candidate_cells:
            if cell <= last:
                continue
            for d, cost in self.alternatives[cell]:
                yield state + ((cell, d),), cost

    def goal_test(self, state) -> tuple[bool, GoalInfo]:
        grid = self.apply(state)
        if conflict_pairs(grid):
            return False, GoalInfo("duplicate digits")
        ok, _, _ = ac3_check(grid)
        if not ok:
            return False, GoalInfo("AC-3 wipe-out")
        self.csp_calls += 1
        res = solve(grid, self.csp_algorithm, count_limit=2 if self.require_unique else 1)
        self.csp_time += res.stats.time_s
        if not res.solved:
            return False, GoalInfo("CSP: no solution" if res.status == "no_solution" else "CSP: limit")
        if self.require_unique and res.n_solutions > 1:
            return False, GoalInfo("CSP: multiple solutions")
        return True, GoalInfo("solved", grid, res.solution, res.stats.as_dict())

    def heuristic(self, state) -> float:
        grid = self.apply(state)
        pairs = conflict_pairs(grid)
        if not pairs:
            return 0.0
        last = state[-1][0] if state else -1

        def lb(c):  # cheapest possible change of cell c, inf if c may not change any more
            return self.min_alt_cost[c] if (c in self.min_alt_cost and c > last) else math.inf

        weighted = []
        for a, b in pairs:
            w = min(lb(a), lb(b))
            if math.isinf(w):
                return math.inf               # an unfixable conflict -> dead state
            weighted.append((w, a, b))
        used, h, size = set(), 0.0, 0
        for w, a, b in sorted(weighted, reverse=True):   # greedy maximal matching
            if a in used or b in used:
                continue
            used.update((a, b))
            h += w
            size += 1
        if size > self.max_corrections - len(state):
            return math.inf
        return h

    def describe(self, state) -> str:
        if not state:
            return "{} (NN prediction)"
        return "{" + ", ".join(f"{cell_name(c)}:{self.grid[c]}->{d}" for c, d in state) + "}"

    def explain(self, state) -> list[dict]:
        """One explanation record per correction (never modify silently)."""
        out = []
        for cell, d in state:
            cost = dict(self.alternatives[cell])[d]
            rec = {"cell": cell, "cell_name": cell_name(cell), "from": self.grid[cell], "to": d,
                   "cost": cost, "reason": self.reasons[cell]}
            if self.use_nn:
                rec["p_from"] = float(self.probs[cell][self.grid[cell] - 1])
                rec["p_to"] = float(self.probs[cell][d - 1])
                rec["explanation"] = (f"{cell_name(cell)}: NN read {self.grid[cell]} (p={rec['p_from']:.3f}); "
                                      f"changed to {d} (p={rec['p_to']:.3f}); cost = ln({rec['p_from']:.3f}/"
                                      f"{rec['p_to']:.3f}) = {cost:.3f}; reason: {self.reasons[cell]}")
            else:
                rec["explanation"] = (f"{cell_name(cell)}: changed {self.grid[cell]} -> {d}; uniform cost 1 "
                                      f"(NN OFF); reason: {self.reasons[cell]}")
            out.append(rec)
        return out


# ---------------------------------------------------------------- error injection
def inject_misclassifications(grid, probs, true_grid, n: int, rng: np.random.Generator):
    """Simulate n NN confusions on given cells that the NN currently reads correctly.

    For each chosen cell a wrong digit w != true digit t is drawn uniformly and
    the probability row is replaced by a *documented* confusion profile:
        p(w) ~ U(0.45, 0.75)                     (wrong digit becomes the arg-max)
        p(t) = (1 - p(w)) * U(0.5, 0.95)         (true digit is the runner-up)
        the remaining mass is spread over the other 7 digits (Dirichlet(1)).
    This mirrors the example in the brief (pred 1 at 52 %, true 7 at 45 %).
    Returns (new_grid, new_probs, records).
    """
    grid = list(grid)
    probs = np.array(probs, dtype=np.float64, copy=True)
    eligible = [i for i in range(81) if grid[i] and grid[i] == true_grid[i]]
    if n > len(eligible):
        raise ValueError(f"cannot inject {n} errors into {len(eligible)} correctly-read cells")
    cells = sorted(int(c) for c in rng.choice(eligible, size=n, replace=False)) if n else []
    records = []
    for c in cells:
        t = true_grid[c]
        w = int(rng.choice([d for d in range(1, 10) if d != t]))
        p_w = rng.uniform(0.45, 0.75)
        p_t = (1 - p_w) * rng.uniform(0.5, 0.95)
        rest = (1 - p_w - p_t) * rng.dirichlet(np.ones(7))
        row = np.zeros(9)
        row[w - 1], row[t - 1] = p_w, p_t
        row[[d - 1 for d in range(1, 10) if d not in (w, t)]] = rest
        probs[c] = row
        grid[c] = w
        records.append({"cell": c, "cell_name": cell_name(c), "true": t, "injected": w,
                        "p_injected": float(p_w), "p_true": float(p_t)})
    return grid, probs, records
