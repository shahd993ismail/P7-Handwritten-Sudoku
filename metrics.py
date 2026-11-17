"""Metric helpers shared by NN evaluation and the experiments."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def classification_metrics(y_true, y_pred) -> dict:
    """Accuracy and macro-averaged precision / recall / F1."""
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "precision_macro": p, "recall_macro": r, "f1_macro": f}


def confusion(y_true, y_pred, labels=range(1, 10)) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(labels))


def expected_calibration_error(probs: np.ndarray, y_true_idx: np.ndarray, n_bins: int = 15) -> tuple[float, pd.DataFrame]:
    """ECE (Guo et al., 2017): weighted |accuracy - confidence| over confidence bins."""
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == y_true_idx).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            rows.append({"bin_low": lo, "bin_high": hi, "count": 0, "accuracy": np.nan, "confidence": np.nan})
            continue
        acc, cf = correct[m].mean(), conf[m].mean()
        ece += m.mean() * abs(acc - cf)
        rows.append({"bin_low": lo, "bin_high": hi, "count": int(m.sum()), "accuracy": acc, "confidence": cf})
    return float(ece), pd.DataFrame(rows)


def mean_std_table(df: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    """Aggregate per-seed results into mean and standard deviation columns."""
    agg = df.groupby(group_cols)[value_cols].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    return agg.reset_index()
