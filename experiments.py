"""The three required experiments (5 seeds, mean ± std).

Run from the project root:
    python -m eval.experiments              # all three (~5-10 minutes on a laptop CPU)
    python -m eval.experiments --exp 2      # only Experiment 2
    python -m eval.experiments --puzzles 2  # quicker, fewer puzzles per seed

Puzzles: unique Sudokus generated with a seeded RNG, rendered as mosaics of
MNIST *test* digits and read by the trained CNN (so the CNN's own, natural
errors are included and counted separately from the injected ones).

Aggregation: for every seed the metric is averaged over that seed's puzzles;
the tables report the mean and the standard deviation of these 5 per-seed
values.

Outputs (results/tables, results/figures):
  exp1_search_raw.csv, exp1_search_summary.csv, exp1_optimality.csv, exp1_search.png
  exp2_csp_raw.csv,    exp2_csp_summary.csv,    exp2_csp.png (+ exp2_csp_mrv.png)
  exp3_correction_raw.csv, exp3_correction_summary.csv, exp3_correction.png
"""
from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image

import config
from csp.sudoku import ALGORITHM_LABELS, solve
from data.make_data import load_processed
from data.puzzle_gen import generate_puzzle, render_mosaic, sample_digit_images
from eval.plots import plot_correction_experiment, plot_csp_experiment, plot_search_experiment
from integration.correction import CorrectionProblem, inject_misclassifications
from integration.pipeline import SEARCH_LABELS, recognise_image, solve_recognized
from search.brute_force import brute_force

INJECTION_LEVELS = [0, 1, 2, 3, 4]


def summarise(raw: pd.DataFrame, cond: list[str], metrics: list[str]) -> pd.DataFrame:
    per_seed = raw.groupby(["seed"] + cond, as_index=False)[metrics].mean()
    agg = per_seed.groupby(cond)[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    n = raw.groupby(cond).size().rename("n_runs")
    return agg.join(n).reset_index()


def make_read_puzzles(seed: int, n_puzzles: int, data: dict, n_givens: int = 30) -> list[dict]:
    """Generate puzzles, render them with MNIST test digits and read them with the CNN."""
    rng = np.random.default_rng(1000 + seed)
    out = []
    for k in range(n_puzzles):
        puzzle, solution = generate_puzzle(n_givens, rng, unique=True)
        imgs, _ = sample_digit_images(puzzle, data["x_test"], data["y_test"], rng)
        rec = recognise_image(Image.fromarray(render_mosaic(imgs)))
        natural = sum(a != b for a, b in zip(rec["grid"], puzzle))
        out.append({"k": k, "puzzle": puzzle, "solution": solution, "grid": rec["grid"],
                    "probs": rec["probs"], "natural_errors": natural})
    return out


# ================================================================ Experiment 1
def experiment_search(seeds, n_puzzles, data) -> pd.DataFrame:
    rows, opt_rows = [], []
    for seed in seeds:
        for P in make_read_puzzles(seed, n_puzzles, data):
            for n_inj in INJECTION_LEVELS:
                rng = np.random.default_rng([seed, P["k"], n_inj])
                g, p, _ = inject_misclassifications(P["grid"], P["probs"], P["puzzle"], n_inj, rng)
                costs = {}
                for algo in ("ucs", "greedy", "astar"):
                    r = solve_recognized(g, p, search_algorithm=algo)
                    s = r.search_stats
                    costs[algo] = r.total_cost
                    rows.append({"seed": seed, "puzzle": P["k"], "n_injected": n_inj,
                                 "natural_errors": P["natural_errors"], "algorithm": SEARCH_LABELS[algo],
                                 "status": r.status, "found": r.solved,
                                 "recovered": r.solution == P["solution"],
                                 "nodes_generated": s["nodes_generated"], "nodes_expanded": s["nodes_expanded"],
                                 "goal_tests": s["goal_tests"], "pruned": s["pruned_dead_states"],
                                 "time_s": s["time_s"], "n_corrections": len(r.corrections) if r.solved else np.nan,
                                 "total_cost": r.total_cost if r.solved else np.nan})
                prob = CorrectionProblem(g, p)
                if n_inj <= 3 and len(prob.candidate_cells) <= 8:       # brute force only on small problems
                    t0 = time.perf_counter()
                    bf, _, tests = brute_force(prob)
                    opt_rows.append({"seed": seed, "puzzle": P["k"], "n_injected": n_inj,
                                     "candidate_cells": len(prob.candidate_cells), "brute_force_cost": bf,
                                     "brute_force_goal_tests": tests, "brute_force_time_s": time.perf_counter() - t0,
                                     **{f"{SEARCH_LABELS[a]}_cost": c for a, c in costs.items()},
                                     "UCS_optimal": _same(costs["ucs"], bf), "A*_optimal": _same(costs["astar"], bf),
                                     "Greedy_optimal": _same(costs["greedy"], bf)})
        print(f"[exp1] seed {seed} done")
    raw = pd.DataFrame(rows)
    raw.to_csv(config.TABLE_DIR / "exp1_search_raw.csv", index=False)
    opt = pd.DataFrame(opt_rows)
    opt.to_csv(config.TABLE_DIR / "exp1_optimality.csv", index=False)
    summ = summarise(raw.assign(found=raw["found"].astype(float), recovered=raw["recovered"].astype(float)),
                     ["n_injected", "algorithm"],
                     ["nodes_generated", "nodes_expanded", "goal_tests", "time_s", "found", "recovered",
                      "n_corrections", "total_cost"])
    summ.to_csv(config.TABLE_DIR / "exp1_search_summary.csv", index=False)
    plot_search_experiment(raw.assign(recovered=raw["recovered"].astype(float)), config.FIG_DIR / "exp1_search.png")
    if len(opt):
        print("[exp1] optimality vs brute force on", len(opt), "small problems:",
              {k: f"{opt[k].mean():.0%}" for k in ("UCS_optimal", "A*_optimal", "Greedy_optimal")})
    return summ


def _same(a, b) -> bool:
    return (math.isinf(a) and math.isinf(b)) or (not math.isinf(a) and not math.isinf(b) and abs(a - b) < 1e-9)


# ================================================================ Experiment 2
def experiment_csp(seeds, n_puzzles) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(2000 + seed)
        for n_givens in (25, 30, 35):
            for k in range(n_puzzles):
                puzzle, solution = generate_puzzle(n_givens, rng, unique=True)
                for order, (vo, va) in (("static", ("static", "natural")), ("MRV+LCV", ("mrv", "lcv"))):
                    for algo in ("bt", "fc", "ac3"):
                        r = solve(puzzle, algo, var_order=vo, value_order=va)
                        rows.append({"seed": seed, "puzzle": k, "n_givens": n_givens, "ordering": order,
                                     "algorithm": ALGORITHM_LABELS[algo], "status": r.status,
                                     "correct": r.solution == solution, **{
                                         key: v for key, v in r.stats.as_dict().items() if key != "algorithm"}})
        print(f"[exp2] seed {seed} done")
    raw = pd.DataFrame(rows)
    raw.to_csv(config.TABLE_DIR / "exp2_csp_raw.csv", index=False)
    summ = summarise(raw.assign(correct=raw["correct"].astype(float)), ["ordering", "n_givens", "algorithm"],
                     ["time_s", "assignments", "backtracks", "constraint_checks", "domain_reductions", "correct"])
    summ.to_csv(config.TABLE_DIR / "exp2_csp_summary.csv", index=False)
    plot_csp_experiment(raw[raw["ordering"] == "static"], config.FIG_DIR / "exp2_csp.png", "- static ordering ")
    plot_csp_experiment(raw[raw["ordering"] == "MRV+LCV"], config.FIG_DIR / "exp2_csp_mrv.png", "- MRV+LCV ")
    return summ


# ================================================================ Experiment 3
def random_correction(grid, probs, max_corr: int, rng, budget: int = 200):
    """Baseline: try up to `budget` random correction sets; keep the first solvable one."""
    problem = CorrectionProblem(grid, probs, max_corrections=max_corr)   # reuse its goal test
    givens = [i for i in range(81) if grid[i]]
    for attempt in range(1, budget + 1):
        k = int(rng.integers(1, max_corr + 1))
        cells = sorted(int(c) for c in rng.choice(givens, size=k, replace=False))
        state = tuple((c, int(rng.choice([d for d in range(1, 10) if d != grid[c]]))) for c in cells)
        ok, info = problem.goal_test(state)
        if ok:
            cost = sum(math.log(max(probs[c][grid[c] - 1], 1e-12)) - math.log(max(probs[c][d - 1], 1e-12))
                       for c, d in state)
            return info.solution, len(state), cost, attempt
    return None, np.nan, np.nan, budget


def experiment_correction(seeds, n_puzzles, data) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        for P in make_read_puzzles(seed, n_puzzles, data):
            for n_inj in INJECTION_LEVELS:
                rng = np.random.default_rng([seed, P["k"], n_inj, 3])
                g, p, _ = inject_misclassifications(P["grid"], P["probs"], P["puzzle"], n_inj, rng)
                base = {"seed": seed, "puzzle": P["k"], "n_injected": n_inj, "natural_errors": P["natural_errors"]}
                # 1. NN-guided correction (UCS)
                r = solve_recognized(g, p, search_algorithm="ucs")
                rows.append({**base, "method": "NN-guided", "solvable": r.solved, "recovered": r.solution == P["solution"],
                             "n_corrections": len(r.corrections) if r.solved else np.nan,
                             "correction_cost": r.total_cost if r.solved else np.nan,
                             "goal_tests": r.search_stats["goal_tests"]})
                # 1b. same search without NN probabilities (uniform costs)
                r = solve_recognized(g, None, use_nn=False, max_expansions=3000)
                rows.append({**base, "method": "NN OFF (uniform)", "solvable": r.solved,
                             "recovered": r.solution == P["solution"],
                             "n_corrections": len(r.corrections) if r.solved else np.nan,
                             "correction_cost": np.nan, "goal_tests": r.search_stats["goal_tests"]})
                # 2. no correction: solve the raw prediction as-is
                c = solve(g, "ac3", count_limit=2)
                ok = c.solved and c.n_solutions == 1
                rows.append({**base, "method": "No correction", "solvable": ok,
                             "recovered": ok and c.solution == P["solution"], "n_corrections": 0,
                             "correction_cost": 0.0, "goal_tests": 1})
                # 3. random correction
                sol, nc, cost, tries = random_correction(g, p, config.MAX_CORRECTIONS, rng)
                if c.solved and c.n_solutions == 1:          # already solvable: no change needed
                    sol, nc, cost, tries = c.solution, 0, 0.0, 1
                rows.append({**base, "method": "Random", "solvable": sol is not None,
                             "recovered": sol == P["solution"], "n_corrections": nc,
                             "correction_cost": cost, "goal_tests": tries})
        print(f"[exp3] seed {seed} done")
    raw = pd.DataFrame(rows)
    raw.to_csv(config.TABLE_DIR / "exp3_correction_raw.csv", index=False)
    f = raw.assign(solvable=raw["solvable"].astype(float), recovered=raw["recovered"].astype(float))
    summ = summarise(f, ["n_injected", "method"], ["recovered", "solvable", "n_corrections", "correction_cost",
                                                   "goal_tests"])
    summ.to_csv(config.TABLE_DIR / "exp3_correction_summary.csv", index=False)
    plot_correction_experiment(f, config.FIG_DIR / "exp3_correction.png")
    return summ


def replot() -> None:
    """Redraw all experiment figures from results/tables/*_raw.csv (no re-running)."""
    t = config.TABLE_DIR
    r1 = pd.read_csv(t / "exp1_search_raw.csv")
    plot_search_experiment(r1.assign(recovered=r1["recovered"].astype(float)), config.FIG_DIR / "exp1_search.png")
    r2 = pd.read_csv(t / "exp2_csp_raw.csv")
    plot_csp_experiment(r2[r2["ordering"] == "static"], config.FIG_DIR / "exp2_csp.png", "- static ordering ")
    plot_csp_experiment(r2[r2["ordering"] == "MRV+LCV"], config.FIG_DIR / "exp2_csp_mrv.png", "- MRV+LCV ")
    r3 = pd.read_csv(t / "exp3_correction_raw.csv")
    plot_correction_experiment(r3.assign(solvable=r3["solvable"].astype(float), recovered=r3["recovered"].astype(float)),
                               config.FIG_DIR / "exp3_correction.png")
    print(f"[experiments] figures redrawn in {config.FIG_DIR}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P7 experiments")
    ap.add_argument("--exp", choices=["1", "2", "3", "all"], default="all")
    ap.add_argument("--puzzles", type=int, default=10, help="puzzles per seed (and per level for Exp. 2: //2)")
    ap.add_argument("--seeds", type=int, nargs="+", default=config.EXPERIMENT_SEEDS)
    ap.add_argument("--replot", action="store_true", help="only redraw the figures from the saved raw CSVs")
    a = ap.parse_args(argv)
    if a.replot:
        replot()
        return 0
    if len(a.seeds) < 5:
        print("[warn] fewer than 5 seeds - the project brief requires at least 5")
    config.ensure_dirs()
    pd.set_option("display.width", 200, "display.max_columns", 30)
    t0 = time.perf_counter()
    data = load_processed() if a.exp in ("1", "3", "all") else None
    if a.exp in ("1", "all"):
        print(experiment_search(a.seeds, a.puzzles, data).round(4).to_string(index=False))
    if a.exp in ("2", "all"):
        print(experiment_csp(a.seeds, max(1, a.puzzles // 2)).round(4).to_string(index=False))
    if a.exp in ("3", "all"):
        print(experiment_correction(a.seeds, a.puzzles, data).round(4).to_string(index=False))
    print(f"[experiments] finished in {time.perf_counter() - t0:.1f}s; tables in {config.TABLE_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
