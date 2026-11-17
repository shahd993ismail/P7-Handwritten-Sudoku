"""Train the digit CNN on MNIST 1-9 and save models/trained_model.pth.

Run from the project root:
    python -m nn.train                 # default 6 epochs
    python -m nn.train --epochs 6 --cpu

The checkpoint with the lowest validation loss is kept. Training history is
written to results/logs/train_history.csv and the loss/accuracy curves to
results/figures/nn_loss_curves.png.
"""
from __future__ import annotations

import argparse
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import config
from data.make_data import load_processed
from models.cnn_model import DigitCNN, get_device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def make_loader(x: np.ndarray, y_digit: np.ndarray, batch: int, shuffle: bool, seed: int) -> DataLoader:
    xt = torch.from_numpy(x).float().unsqueeze(1)
    yt = torch.from_numpy(y_digit - 1).long()          # digit 1..9 -> class 0..8
    g = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(xt, yt), batch_size=batch, shuffle=shuffle, generator=g)


def run_epoch(model, loader, loss_fn, device, optimizer=None) -> tuple[float, float]:
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            n += len(yb)
    return total_loss / n, correct / n


def train(epochs: int = config.EPOCHS, batch_size: int = config.BATCH_SIZE, lr: float = config.LEARNING_RATE,
          seed: int = config.SEED, use_gpu: bool = True) -> dict:
    config.ensure_dirs()
    set_seed(seed)
    device = get_device(use_gpu)
    print(f"[train] device = {device}")
    data = load_processed()
    train_loader = make_loader(data["x_train"], data["y_train"], batch_size, True, seed)
    val_loader = make_loader(data["x_val"], data["y_val"], 512, False, seed)

    model = DigitCNN(num_classes=config.NUM_CLASSES).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history, best_val = [], float("inf")
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc = run_epoch(model, train_loader, loss_fn, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, loss_fn, device)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc, "seconds": time.perf_counter() - t0})
        print(f"[train] epoch {epoch}/{epochs}  train_loss={tr_loss:.4f} acc={tr_acc:.4f}  "
              f"val_loss={va_loss:.4f} acc={va_acc:.4f}  ({history[-1]['seconds']:.1f}s)")
        if va_loss < best_val:
            best_val = va_loss
            torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                        "num_classes": config.NUM_CLASSES, "seed": seed, "epoch": epoch,
                        "val_loss": va_loss, "val_acc": va_acc}, config.MODEL_PATH)
    hist = pd.DataFrame(history)
    hist.to_csv(config.LOG_DIR / "train_history.csv", index=False)
    from eval.plots import plot_loss_curves
    plot_loss_curves(hist, config.FIG_DIR / "nn_loss_curves.png")
    print(f"[train] best model (val_loss={best_val:.4f}) saved to {config.MODEL_PATH}")
    return {"history": hist, "best_val_loss": best_val}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train the P7 digit CNN")
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--cpu", action="store_true", help="force CPU even if a GPU is available")
    a = ap.parse_args(argv)
    train(a.epochs, a.batch_size, a.lr, a.seed, use_gpu=not a.cpu)
    return 0


if __name__ == "__main__":
    sys.exit(main())
