"""Fully automatic demonstration of NN + Search + CSP cooperation (no input needed).

Run from the project root:
    python run_demo.py                       # default: puzzle_00, 2 injected errors
    python run_demo.py --puzzle puzzle_03 --errors 3 --seed 7

Steps
  1. load a generated handwritten Sudoku (creates one if data/puzzles is empty)
  2. run the CNN on the image (cell probabilities)
  3. inject documented misclassifications (simulated NN confusions)
  4. detect conflicts (duplicates + AC-3)
  5. run Search (UCS, plus Greedy and A* for comparison)
  6. run the CSP (BT, BT+FC, BT+FC+AC-3) on the corrected puzzle
  7. print the correction log
  8. print / save the final solution
  9. save plots showing how the NN, Search and the CSP cooperate
Outputs: results/figures/demo_*.png and results/logs/demo_log.txt
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
from PIL import Image

import config
from csp.sudoku import ALGORITHM_LABELS, grid_to_str, solve
from eval.plots import plot_cooperation, plot_pipeline
from integration.correction import inject_misclassifications
from integration.pipeline import format_report, recognise_image, solve_recognized


def load_puzzle(name: str) -> tuple[dict, Image.Image]:
    js, png = config.PUZZLE_DIR / f"{name}.json", config.PUZZLE_DIR / f"{name}.png"
    if not js.exists() or not png.exists():
        print(f"[demo] {name} not found - generating sample puzzles first (python -m data.make_data)")
        from data.make_data import main as make_data
        make_data()
    if not js.exists():
        raise FileNotFoundError(f"puzzle {name} does not exist in {config.PUZZLE_DIR}")
    return json.loads(js.read_text()), Image.open(png)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P7 automatic demonstration")
    ap.add_argument("--puzzle", default="puzzle_00")
    ap.add_argument("--errors", type=int, default=2, help="number of injected misclassifications (0-6)")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--max-corrections", type=int, default=config.MAX_CORRECTIONS)
    a = ap.parse_args(argv)
    if not 0 <= a.errors <= 6:
        ap.error("--errors must be between 0 and 6")
    config.ensure_dirs()
    log: list[str] = []

    def say(text: str = "") -> None:
        print(text)
        log.append(text)

    # 1-2 ---------------------------------------------------------------
    record, image = load_puzzle(a.puzzle)
    truth_puzzle, truth_solution = record["puzzle"], record["solution"]
    rec = recognise_image(image)
    natural_errors = [i for i in range(81) if rec["grid"][i] != truth_puzzle[i]]
    say(f"STEP 1-2  {a.puzzle}: {record['n_givens']} givens, extraction = {rec['extraction']}")
    say(f"          CNN read {81 - sum(rec['blank'])} digits; natural CNN errors vs ground truth: {len(natural_errors)}")

    # 3 -----------------------------------------------------------------
    rng = np.random.default_rng(a.seed)
    grid, probs, injected = inject_misclassifications(rec["grid"], rec["probs"], truth_puzzle, a.errors, rng)
    say(f"STEP 3    injected {len(injected)} documented misclassification(s) (seed {a.seed}):")
    for r in injected:
        say(f"          {r['cell_name']}: true {r['true']} -> NN now says {r['injected']} "
            f"(p={r['p_injected']:.2f}), true digit kept as runner-up (p={r['p_true']:.2f})")

    # 4-8 (main run: UCS + BT+FC+AC-3, NN ON) ----------------------------
    main_res = solve_recognized(grid, probs, search_algorithm="ucs", csp_algorithm="ac3",
                                max_corrections=a.max_corrections)
    say("STEP 4-8  main run: UCS + BT+FC+AC-3, NN ON")
    say(format_report(main_res, truth_solution))
    plot_pipeline(main_res, np.array(image.convert("L")), config.FIG_DIR / "demo_pipeline.png")

    # 5 (comparison) -----------------------------------------------------
    rows = []
    for algo in ("ucs", "greedy", "astar"):
        for use_nn in (True, False):
            r = solve_recognized(grid, probs, search_algorithm=algo, use_nn=use_nn,
                                 max_corrections=a.max_corrections, max_expansions=5000)
            s = r.search_stats
            rows.append({"search": s["algorithm"], "NN": "ON" if use_nn else "OFF (uniform cost)",
                         "status": r.status, "recovered": r.solution == truth_solution,
                         "corrections": len(r.corrections), "cost": r.total_cost,
                         "generated": s["nodes_generated"], "expanded": s["nodes_expanded"],
                         "goal_tests": s["goal_tests"], "pruned": s["pruned_dead_states"],
                         "time_s": round(s["time_s"], 4)})
    search_df = pd.DataFrame(rows)
    say("\nSEARCH COMPARISON (same corrupted puzzle):")
    say(search_df.to_string(index=False))

    # 6 (CSP comparison on the corrected puzzle) -------------------------
    target = main_res.corrected if main_res.solved else truth_puzzle
    csp_rows = []
    for order in (("static", "natural"), ("mrv", "lcv")):
        for algo in ("bt", "fc", "ac3"):
            r = solve(target, algo, var_order=order[0], value_order=order[1])
            st = r.stats.as_dict()
            st.update({"ordering": "static" if order[0] == "static" else "MRV+LCV", "status": r.status})
            csp_rows.append(st)
    csp_df = pd.DataFrame(csp_rows)[["ordering", "algorithm", "status", "assignments", "backtracks",
                                     "constraint_checks", "domain_reductions", "time_s"]]
    say("\nCSP COMPARISON on the corrected puzzle (pruning without vs with FC / AC-3):")
    say(csp_df.to_string(index=False))

    # 9 -------------------------------------------------------------------
    ucs_rows = search_df[search_df["search"] == "UCS"]
    plot_cooperation({
        "nn_on_off": [{"mode": f"NN {r['NN'].split()[0]}", "nodes_expanded": r["expanded"], "recovered": r["recovered"]}
                      for _, r in ucs_rows.iterrows()],
        "csp": csp_df[csp_df["ordering"] == "static"].to_dict("records")},
        config.FIG_DIR / "demo_cooperation.png")
    search_df.to_csv(config.TABLE_DIR / "demo_search_comparison.csv", index=False)
    csp_df.to_csv(config.TABLE_DIR / "demo_csp_comparison.csv", index=False)
    if main_res.solution == truth_solution:
        verdict = "SOLUTION MATCHES THE GROUND TRUTH"
    elif main_res.solved:
        verdict = main_res.message + " BUT IT DOES NOT MATCH THE GROUND TRUTH (solvable != correct)"
    else:
        verdict = main_res.message
    say("\nFINAL: " + verdict)
    say(f"Figures: {config.FIG_DIR / 'demo_pipeline.png'}\n         {config.FIG_DIR / 'demo_cooperation.png'}")
    (config.LOG_DIR / "demo_log.txt").write_text("\n".join(log), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
