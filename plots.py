"""All Matplotlib / Seaborn figures of the project (NN, pipeline and experiments)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")          # headless backend: works on Windows, Linux and in Streamlit
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = {"UCS": "#2a6fdb", "Greedy": "#e8871e", "A*": "#2b9348",
           "BT": "#c9184a", "BT+FC": "#e8871e", "BT+FC+AC-3": "#2b9348",
           "NN-guided": "#2a6fdb", "No correction": "#8d99ae", "Random": "#e8871e", "NN OFF (uniform)": "#c9184a"}

# colours used in Sudoku drawings
COL_GIVEN = "#1d1d1d"
COL_SOLVED = "#2a6fdb"
COL_CORRECTED_BG = "#b7e4c7"
COL_CONFLICT_BG = "#ffadad"
COL_LOWCONF_BG = "#ffe8a3"


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- NN figures
def plot_loss_curves(hist: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(hist["epoch"], hist["train_loss"], "o-", label="train")
    axes[0].plot(hist["epoch"], hist["val_loss"], "s-", label="validation")
    axes[0].set(xlabel="epoch", ylabel="cross-entropy loss", title="CNN loss")
    axes[1].plot(hist["epoch"], hist["train_acc"], "o-", label="train")
    axes[1].plot(hist["epoch"], hist["val_acc"], "s-", label="validation")
    axes[1].set(xlabel="epoch", ylabel="accuracy", title="CNN accuracy")
    for a in axes:
        a.legend()
    return _save(fig, path)


def plot_confusion(cm: np.ndarray, path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=range(1, 10), yticklabels=range(1, 10), annot_kws={"size": 8})
    ax.set(xlabel="predicted digit", ylabel="true digit", title=title)
    return _save(fig, path)


def plot_reliability(bins: pd.DataFrame, ece: float, path: Path) -> Path:
    b = bins.dropna()
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    ax.plot(b["confidence"], b["accuracy"], "o-", label="CNN")
    ax.set(xlabel="mean confidence", ylabel="accuracy", xlim=(0, 1.02), ylim=(0, 1.02),
           title=f"Reliability diagram (ECE = {ece:.4f})")
    ax.legend()
    return _save(fig, path)


# ---------------------------------------------------------------- Sudoku drawings
def draw_grid(ax, grid, title: str = "", givens=None, cell_bg: dict | None = None,
              subtext: dict | None = None) -> None:
    """Draw a 9x9 grid. `givens` = cells drawn black (others blue)."""
    cell_bg = cell_bg or {}
    subtext = subtext or {}
    givens = set(range(81)) if givens is None else set(givens)
    ax.set_xlim(0, 9)
    ax.set_ylim(9, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    for i in range(81):
        r, c = divmod(i, 9)
        if i in cell_bg:
            ax.add_patch(plt.Rectangle((c, r), 1, 1, color=cell_bg[i], zorder=0))
        if grid[i]:
            ax.text(c + 0.5, r + 0.47, str(grid[i]), ha="center", va="center", fontsize=13,
                    color=COL_GIVEN if i in givens else COL_SOLVED,
                    fontweight="bold" if i in givens else "normal")
        if i in subtext:
            ax.text(c + 0.5, r + 0.88, subtext[i], ha="center", va="center", fontsize=6, color="#444")
    for k in range(10):
        lw = 2.2 if k % 3 == 0 else 0.6
        ax.plot([0, 9], [k, k], color="black", lw=lw)
        ax.plot([k, k], [0, 9], color="black", lw=lw)
    ax.set_title(title, fontsize=11)


def plot_pipeline(result, image: np.ndarray | None, path: Path) -> Path:
    """4-panel figure: image | NN prediction | corrected puzzle | solution."""
    from csp.sudoku import conflict_pairs

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    if image is not None:
        axes[0].imshow(image, cmap="gray", vmin=0, vmax=255)
    axes[0].axis("off")
    axes[0].set_title("1. Input image (generated)")

    pred = result.recognized
    bad = {c for pair in conflict_pairs(pred) for c in pair}
    bg = {}
    sub = {}
    for i in range(81):
        if pred[i]:
            conf = result.confidence[i]
            sub[i] = "" if conf is None else f"{conf:.2f}"
            if conf is not None and conf < result.confidence_threshold:
                bg[i] = COL_LOWCONF_BG
        if i in bad:
            bg[i] = COL_CONFLICT_BG
    draw_grid(axes[1], pred, "2. NN prediction\n(red = conflict, yellow = low confidence)", cell_bg=bg, subtext=sub)

    corrected = result.corrected or pred
    changed = {c["cell"]: COL_CORRECTED_BG for c in result.corrections}
    sub2 = {c["cell"]: f"{c['from']}->{c['to']}" for c in result.corrections}
    draw_grid(axes[2], corrected, f"3. Corrected puzzle ({len(result.corrections)} change(s), "
                                  f"cost {result.total_cost:.2f})", cell_bg=changed, subtext=sub2)
    if result.solution:
        givens = [i for i in range(81) if corrected[i]]
        draw_grid(axes[3], result.solution, "4. Final solution (blue = filled by CSP)", givens=givens, cell_bg=changed)
    else:
        axes[3].axis("off")
        axes[3].text(0.5, 0.5, "NO SOLUTION FOUND\nWITHIN THE ALLOWED\nCORRECTION LIMIT",
                     ha="center", va="center", fontsize=14, color="#c9184a")
    return _save(fig, path)


# ---------------------------------------------------------------- experiment figures
HUE_ORDERS = {"algorithm": None, "method": ["NN-guided", "NN OFF (uniform)", "Random", "No correction"]}
SEARCH_ORDER = ["UCS", "Greedy", "A*"]
CSP_ORDER = ["BT", "BT+FC", "BT+FC+AC-3"]


def _bar_with_err(df, x, y, hue, ax, ylabel, log=False, hue_order=None, legend=True):
    """Bar = mean of the per-seed means, error bar = std of the per-seed means (same as the tables)."""
    per_seed = df.groupby(["seed", x, hue], as_index=False)[y].mean()
    sns.barplot(data=per_seed, x=x, y=y, hue=hue, hue_order=hue_order, ax=ax, errorbar="sd",
                palette=PALETTE, capsize=0.08)
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    if not legend and ax.get_legend() is not None:
        ax.get_legend().remove()


def plot_search_experiment(raw: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    kw = dict(hue_order=SEARCH_ORDER)
    _bar_with_err(raw, "n_injected", "nodes_expanded", "algorithm", axes[0, 0], "nodes expanded (log)", log=True, **kw)
    _bar_with_err(raw, "n_injected", "nodes_generated", "algorithm", axes[0, 1], "nodes generated (log)",
                  log=True, legend=False, **kw)
    _bar_with_err(raw, "n_injected", "time_s", "algorithm", axes[1, 0], "time (s)", legend=False, **kw)
    _bar_with_err(raw, "n_injected", "recovered", "algorithm", axes[1, 1], "exact recovery rate", legend=False, **kw)
    axes[1, 1].set_ylim(0, 1.05)
    for a in axes.flat:
        a.set_xlabel("injected misclassifications")
    fig.suptitle("Experiment 1 - Search comparison (bar = mean of 5 per-seed means, error bar = std)")
    return _save(fig, path)


def plot_csp_experiment(raw: pd.DataFrame, path: Path, title_extra: str = "") -> Path:
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.3))
    for i, (ax, col, lab) in enumerate(zip(axes, ["time_s", "assignments", "backtracks", "constraint_checks"],
                                           ["time (s)", "assignments", "backtracks", "constraint checks"])):
        _bar_with_err(raw, "n_givens", col, "algorithm", ax, lab, log=True, hue_order=CSP_ORDER, legend=i == 0)
        ax.set_xlabel("number of givens")
    fig.suptitle(f"Experiment 2 - CSP comparison {title_extra}(log scale; mean ± std of 5 per-seed means)")
    return _save(fig, path)


def plot_correction_experiment(raw: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    order = HUE_ORDERS["method"]
    _bar_with_err(raw, "n_injected", "recovered", "method", axes[0], "exact-solution recovery rate",
                  hue_order=order, legend=False)
    _bar_with_err(raw, "n_injected", "solvable", "method", axes[1], "solvability rate", hue_order=order)
    axes[1].legend(title="method", loc="upper left", bbox_to_anchor=(1.01, 1))
    for a in axes:
        a.set_ylim(0, 1.05)
        a.set_xlabel("injected misclassifications")
    fig.suptitle("Experiment 3 - Correction method comparison (mean ± std of 5 per-seed means)")
    return _save(fig, path)


def plot_cooperation(stats: dict, path: Path) -> Path:
    """Bar chart for run_demo: NN ON vs OFF and CSP algorithm pruning."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    nn_df = pd.DataFrame(stats["nn_on_off"])
    sns.barplot(data=nn_df, x="mode", y="nodes_expanded", ax=axes[0], color="#2a6fdb")
    axes[0].set(title="Search effort: NN ON vs NN OFF", ylabel="nodes expanded", xlabel="")
    for i, r in nn_df.iterrows():
        axes[0].text(i, r["nodes_expanded"], "recovered" if r["recovered"] else "NOT recovered",
                     ha="center", va="bottom", fontsize=9)
    csp_df = pd.DataFrame(stats["csp"])
    sns.barplot(data=csp_df, x="algorithm", y="assignments", ax=axes[1], palette=PALETTE, hue="algorithm")
    axes[1].set(title="CSP effort, static ordering", ylabel="assignments (log)", yscale="log", xlabel="")
    sns.barplot(data=csp_df, x="algorithm", y="domain_reductions", ax=axes[2], palette=PALETTE, hue="algorithm")
    axes[2].set(title="CSP pruning, static ordering (values removed)", ylabel="domain reductions", xlabel="")
    return _save(fig, path)
