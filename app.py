"""Streamlit demonstration app for P7 - Handwritten Sudoku Reader and Solver.

Launch from the project root:
    streamlit run app.py
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import config
from csp.sudoku import ALGORITHM_LABELS, cell_name, conflict_pairs, solve
from data.make_data import load_processed
from data.puzzle_gen import generate_puzzle, render_mosaic, sample_digit_images
from eval.plots import COL_CONFLICT_BG, COL_CORRECTED_BG, COL_LOWCONF_BG, draw_grid
from integration.correction import inject_misclassifications
from integration.pipeline import recognise_image, solve_recognized
from nn.predict import load_model

st.set_page_config(page_title="P7 Handwritten Sudoku Reader & Solver", layout="wide")


@st.cache_resource
def get_model():
    return load_model()


@st.cache_resource
def get_test_digits():
    d = load_processed()
    return d["x_test"], d["y_test"]


def grid_figure(grid, title, **kw):
    fig, ax = plt.subplots(figsize=(4.2, 4.4))
    draw_grid(ax, grid, title, **kw)
    fig.tight_layout()
    return fig


def list_puzzles() -> list[str]:
    idx = config.PUZZLE_DIR / "index.json"
    return [r["name"] for r in json.loads(idx.read_text())] if idx.exists() else []


# ---------------------------------------------------------------- sidebar
st.sidebar.title("Settings")
source = st.sidebar.radio("Input", ["Generated test puzzle", "New random puzzle", "Upload image"])
truth_puzzle = truth_solution = None
image = None
try:
    model = get_model()
except FileNotFoundError as e:
    st.error(f"{e}")
    st.stop()

if source == "Generated test puzzle":
    names = list_puzzles()
    if not names:
        st.error("No sample puzzles found. Run:  python -m data.make_data")
        st.stop()
    name = st.sidebar.selectbox("Puzzle", names)
    rec_json = json.loads((config.PUZZLE_DIR / f"{name}.json").read_text())
    image = Image.open(config.PUZZLE_DIR / f"{name}.png")
    truth_puzzle, truth_solution = rec_json["puzzle"], rec_json["solution"]
elif source == "New random puzzle":
    seed = st.sidebar.number_input("Puzzle seed", 0, 10_000, 123)
    givens = st.sidebar.slider("Givens", 25, 40, 30)
    try:
        x_test, y_test = get_test_digits()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    rng = np.random.default_rng(int(seed))
    with st.spinner("Generating a unique puzzle..."):
        truth_puzzle, truth_solution = generate_puzzle(int(givens), rng, unique=True)
        imgs, _ = sample_digit_images(truth_puzzle, x_test, y_test, rng)
        image = Image.fromarray(render_mosaic(imgs))
else:
    up = st.sidebar.file_uploader("Sudoku image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    if up is None:
        st.info("Upload a clean, cropped image of a Sudoku grid (the generated puzzles in data/puzzles work best).")
        st.stop()
    image = Image.open(up)

n_errors = 0
if truth_puzzle is not None:
    n_errors = st.sidebar.slider("Inject misclassifications", 0, 5, 2)
    inj_seed = st.sidebar.number_input("Injection seed", 0, 10_000, 2)
st.sidebar.markdown("---")
search_algo = st.sidebar.selectbox("Search algorithm", ["UCS", "Greedy", "A*"])
csp_algo = st.sidebar.selectbox("CSP algorithm", ["BT+FC+AC-3", "BT+FC", "BT"])
use_nn = st.sidebar.toggle("Use NN probabilities (NN ON)", value=True,
                           help="OFF = uniform cost 1 per correction, no confidence information")
max_corr = st.sidebar.slider("Max corrections", 0, 5, config.MAX_CORRECTIONS)
threshold = st.sidebar.slider("Confidence threshold", 0.5, 0.999, config.CONFIDENCE_THRESHOLD)
require_unique = st.sidebar.checkbox("Goal requires a unique solution", value=config.REQUIRE_UNIQUE)
run = st.sidebar.button("Run correction & solve", type="primary")

# ---------------------------------------------------------------- recognition
st.title("P7 – Handwritten Sudoku Reader and Solver")
st.caption("CNN (PyTorch) → conflict check (AC-3) → Search (UCS / Greedy / A*) → CSP (BT / FC / AC-3). "
           "Puzzles are generated mosaics of real MNIST test digits.")
try:
    rec = recognise_image(image, model)
except ValueError as e:
    st.error(f"Could not read the image: {e}")
    st.stop()
grid, probs = rec["grid"], rec["probs"]
injected = []
if n_errors:
    try:
        grid, probs, injected = inject_misclassifications(grid, probs, truth_puzzle, n_errors,
                                                          np.random.default_rng(int(inj_seed)))
    except ValueError as e:
        st.warning(str(e))
conf = [float(probs[i].max()) if grid[i] else None for i in range(81)]

c1, c2 = st.columns(2)
with c1:
    st.subheader("1. Original handwritten Sudoku")
    st.image(image, use_container_width=True)
    st.caption(f"Cell extraction: {rec['extraction']}")
with c2:
    st.subheader("2. Recognised Sudoku (NN prediction)")
    bad = {c for p in conflict_pairs(grid) for c in p}
    bg = {i: COL_LOWCONF_BG for i in range(81) if conf[i] is not None and conf[i] < threshold}
    bg.update({i: COL_CONFLICT_BG for i in bad})
    st.pyplot(grid_figure(grid, "red = conflict, yellow = low confidence", cell_bg=bg,
                          subtext={i: f"{conf[i]:.2f}" for i in range(81) if conf[i] is not None}))
if injected:
    st.warning("Injected (simulated) misclassifications: " + "; ".join(
        f"{r['cell_name']}: true {r['true']} → read {r['injected']} (p={r['p_injected']:.2f})" for r in injected))

with st.expander("3. Confidence / probabilities of every recognised digit", expanded=False):
    rows = []
    for i in range(81):
        if grid[i]:
            order = np.argsort(-probs[i])
            rows.append({"cell": cell_name(i), "predicted": grid[i], "confidence": round(conf[i], 4),
                         "2nd": f"{order[1] + 1} ({probs[i][order[1]]:.4f})",
                         "3rd": f"{order[2] + 1} ({probs[i][order[2]]:.4f})"})
    st.dataframe(pd.DataFrame(rows).sort_values("confidence"), use_container_width=True, hide_index=True)

if not run:
    st.info("Choose the algorithms in the sidebar and press **Run correction & solve**.")
    st.stop()

# ---------------------------------------------------------------- pipeline
with st.spinner("Searching for the most likely solvable correction..."):
    res = solve_recognized(grid, probs if use_nn else None, conf, search_algorithm=search_algo,
                           csp_algorithm=csp_algo, max_corrections=max_corr, use_nn=use_nn,
                           require_unique=require_unique, confidence_threshold=threshold)

st.subheader("4. Detected conflicts")
st.write(f"AC-3 check: **{'consistent' if res.ac3_consistent else 'INCONSISTENT'}** "
         f"({res.ac3_stats['constraint_checks']} checks, {res.ac3_stats['domain_reductions']} domain reductions)")
if res.conflicts:
    st.dataframe(pd.DataFrame([{"unit": c["unit"], "digit": c["digit"],
                                "cells": ", ".join(cell_name(x) for x in c["cells"])} for c in res.conflicts]),
                 hide_index=True)
else:
    st.write("No duplicate digits.")
st.markdown("**Cells Search is allowed to change**")
st.dataframe(pd.DataFrame(res.candidate_cells), hide_index=True, use_container_width=True)

st.subheader("5–6. Proposed corrections and costs")
if res.solved:
    if res.corrections:
        st.dataframe(pd.DataFrame(res.corrections)[["cell_name", "from", "to", "cost", "explanation"]],
                     hide_index=True, use_container_width=True)
        st.write(f"**Total correction cost: {res.total_cost:.4f}**")
    else:
        st.success("No correction needed - the recognised puzzle is valid and solvable.")
else:
    st.error(res.message)
    st.markdown("Least confident cells and their alternatives:")
    st.dataframe(pd.DataFrame(res.least_confident), hide_index=True)

if res.solved:
    d1, d2 = st.columns(2)
    changed = {c["cell"]: COL_CORRECTED_BG for c in res.corrections}
    with d1:
        st.pyplot(grid_figure(res.corrected, "Corrected puzzle (green = changed)", cell_bg=changed,
                              subtext={c["cell"]: f"{c['from']}->{c['to']}" for c in res.corrections}))
    with d2:
        st.subheader("7. Final solved Sudoku")
        st.pyplot(grid_figure(res.solution, "blue = filled by the CSP",
                              givens=[i for i in range(81) if res.corrected[i]], cell_bg=changed))
        if truth_solution is not None:
            (st.success if res.solution == truth_solution else st.error)(
                "Matches the ground-truth solution" if res.solution == truth_solution
                else "Does NOT match the ground-truth solution")

s1, s2 = st.columns(2)
with s1:
    st.subheader("8. Search statistics")
    st.json({k: (round(v, 5) if isinstance(v, float) else v) for k, v in res.search_stats.items()})
with s2:
    st.subheader("9. CSP statistics")
    st.json(res.csp_stats or {"status": "not run - no solvable correction"})
with st.expander("Search trace"):
    st.code("\n".join(res.search_trace) or "(empty)")

if res.solved:
    with st.expander("Compare the three CSP algorithms on the corrected puzzle"):
        order = st.radio("Variable / value ordering", ["static", "MRV+LCV"], horizontal=True)
        vo, va = ("static", "natural") if order == "static" else ("mrv", "lcv")
        rows = []
        for a in ("bt", "fc", "ac3"):
            r = solve(res.corrected, a, var_order=vo, value_order=va)
            rows.append({"algorithm": ALGORITHM_LABELS[a], "status": r.status, **{
                k: v for k, v in r.stats.as_dict().items() if k != "algorithm"}})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
