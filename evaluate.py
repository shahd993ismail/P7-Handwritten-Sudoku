"""Evaluate the trained CNN on the untouched MNIST test split (digits 1-9).

Run from the project root:
    python -m nn.evaluate

Outputs
    results/tables/nn_test_metrics.csv          accuracy / precision / recall / F1 / ECE (CNN vs baseline)
    results/tables/nn_classification_report.csv per-digit precision / recall / F1
    results/tables/nn_calibration_bins.csv      reliability-diagram data
    results/tables/blank_detection.csv          brightness-rule check on digits and empty cells
    results/figures/nn_confusion_matrix.png
    results/figures/baseline_confusion_matrix.png
    results/figures/nn_reliability.png

Baseline: multinomial Logistic Regression (scikit-learn) on the raw 784
pixels, trained on the same 1-9 training split.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import config
from data.make_data import load_processed
from eval.metrics import classification_metrics, confusion, expected_calibration_error
from eval.plots import plot_confusion, plot_reliability
from nn.predict import central_ink, load_model, predict_proba


def evaluate_cnn(data: dict) -> tuple[dict, np.ndarray]:
    model = load_model()
    probs = predict_proba(model, data["x_test"])
    y_pred = probs.argmax(1) + 1
    m = classification_metrics(data["y_test"], y_pred)
    ece, bins = expected_calibration_error(probs, data["y_test"] - 1)
    m["ece"] = ece
    m["mean_confidence"] = float(probs.max(1).mean())
    plot_confusion(confusion(data["y_test"], y_pred), config.FIG_DIR / "nn_confusion_matrix.png",
                   f"CNN confusion matrix (test acc = {m['accuracy']:.4f})")
    plot_reliability(bins, ece, config.FIG_DIR / "nn_reliability.png")
    bins.to_csv(config.TABLE_DIR / "nn_calibration_bins.csv", index=False)
    rep = classification_report(data["y_test"], y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.to_csv(config.TABLE_DIR / "nn_classification_report.csv")
    return m, probs


def evaluate_baseline(data: dict, seed: int = config.SEED) -> dict:
    t0 = time.perf_counter()
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(data["x_train"].reshape(len(data["x_train"]), -1), data["y_train"])
    fit_s = time.perf_counter() - t0
    probs = clf.predict_proba(data["x_test"].reshape(len(data["x_test"]), -1))
    y_pred = clf.classes_[probs.argmax(1)]
    m = classification_metrics(data["y_test"], y_pred)
    ece, _ = expected_calibration_error(probs, data["y_test"] - 1)
    m["ece"] = ece
    m["mean_confidence"] = float(probs.max(1).mean())
    m["fit_seconds"] = fit_s
    plot_confusion(confusion(data["y_test"], y_pred), config.FIG_DIR / "baseline_confusion_matrix.png",
                   f"Logistic Regression confusion matrix (test acc = {m['accuracy']:.4f})")
    return m


def evaluate_blank_rule(data: dict) -> pd.DataFrame:
    """How often does the brightness rule misjudge real digits / empty cells?"""
    ink = np.array([central_ink(x) for x in data["x_test"]])
    empty_cells = np.zeros((1000, 28, 28), np.float32)
    rng = np.random.default_rng(config.SEED)
    noisy_empty = np.clip(rng.normal(0, 0.03, (1000, 28, 28)), 0, 1).astype(np.float32)
    rows = [
        {"input": "MNIST test digits (1-9)", "n": len(ink),
         "flagged_blank": int((ink < config.BLANK_INK_THRESHOLD).sum()),
         "min_central_ink": float(ink.min()), "mean_central_ink": float(ink.mean())},
        {"input": "empty cells", "n": 1000,
         "flagged_blank": int(sum(central_ink(x) < config.BLANK_INK_THRESHOLD for x in empty_cells)),
         "min_central_ink": 0.0, "mean_central_ink": 0.0},
        {"input": "empty cells + paper noise (std 0.03)", "n": 1000,
         "flagged_blank": int(sum(central_ink(x) < config.BLANK_INK_THRESHOLD for x in noisy_empty)),
         "min_central_ink": float(min(central_ink(x) for x in noisy_empty)),
         "mean_central_ink": float(np.mean([central_ink(x) for x in noisy_empty]))},
    ]
    return pd.DataFrame(rows)


def main() -> int:
    config.ensure_dirs()
    data = load_processed()
    print(f"[eval] test set: {len(data['y_test'])} images (digits 1-9, official MNIST test split)")
    cnn, _ = evaluate_cnn(data)
    print("[eval] CNN      :", {k: round(float(v), 4) for k, v in cnn.items()})
    print("[eval] training Logistic Regression baseline (may take ~1 minute)...")
    base = evaluate_baseline(data)
    print("[eval] LogReg   :", {k: round(float(v), 4) for k, v in base.items()})
    table = pd.DataFrame([{"model": "CNN (PyTorch)", **cnn},
                          {"model": "Logistic Regression (baseline)", **base}])
    table.to_csv(config.TABLE_DIR / "nn_test_metrics.csv", index=False)
    blank = evaluate_blank_rule(data)
    blank.to_csv(config.TABLE_DIR / "blank_detection.csv", index=False)
    print(table.round(4).to_string(index=False))
    print(blank.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
