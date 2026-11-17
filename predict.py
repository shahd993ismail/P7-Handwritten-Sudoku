"""Inference: load the trained CNN and turn cell images into cell probabilities.

From digit probabilities to *cell* probabilities (documented method)
---------------------------------------------------------------------
For every one of the 81 cell images:
1. Blank detection (simple brightness rule): the cell is BLANK if the mean
   ink intensity of its central 20x20 region (pixels in [0,1], ink = 1) is
   below config.BLANK_INK_THRESHOLD (0.04). MNIST digits have a mean central
   ink of about 0.10-0.25, empty paper is ~0.
2. Blank cell  -> value 0, probability row = all zeros, confidence = None.
   Non-blank   -> the CNN's Softmax output p(d | image) for d = 1..9 is used
                  directly as the cell's distribution P(cell = d). The
                  predicted digit is argmax_d p and its confidence is max_d p.
Because the CNN was trained only on digits 1..9 the 9 probabilities of a
non-blank cell always sum to 1.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import config
from models.cnn_model import DigitCNN, get_device

_MODEL_CACHE: dict[str, DigitCNN] = {}


def load_model(path: Path = config.MODEL_PATH, device: torch.device | None = None) -> DigitCNN:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"trained model {path} not found. Run:  python -m nn.train")
    key = str(path.resolve())
    if key not in _MODEL_CACHE:
        device = device or get_device()
        ckpt = torch.load(path, map_location=device, weights_only=True)
        model = DigitCNN(num_classes=int(ckpt.get("num_classes", 9)))
        model.load_state_dict(ckpt["state_dict"])
        model.to(device).eval()
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def predict_proba(model: DigitCNN, images: np.ndarray, batch: int = 1024) -> np.ndarray:
    """(N,28,28) float images in [0,1] -> (N,9) Softmax probabilities."""
    images = np.asarray(images, dtype=np.float32)
    if images.ndim != 3 or images.shape[1:] != (28, 28):
        raise ValueError(f"expected images of shape (N,28,28), got {images.shape}")
    if images.max(initial=0) > 1.0:
        images = images / 255.0
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(images), batch):
        xb = torch.from_numpy(images[i:i + batch]).to(device)
        out.append(model.predict_proba(xb).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 9), np.float32)


def central_ink(img: np.ndarray) -> float:
    """Mean intensity of the central 20x20 region of a 28x28 cell (ink = 1)."""
    img = np.asarray(img, dtype=np.float32)
    if img.max(initial=0) > 1.0:
        img = img / 255.0
    return float(img[4:24, 4:24].mean())


def is_blank(img: np.ndarray, threshold: float = config.BLANK_INK_THRESHOLD) -> bool:
    return central_ink(img) < threshold


def cell_probabilities(cell_images: np.ndarray, model: DigitCNN | None = None) -> dict:
    """81 cell images (81,28,28) -> recognised grid + probability matrix.

    Returns dict with
        grid  : list[int] (0 = blank)            - the NN's raw prediction
        probs : np.ndarray (81,9), zeros for blank cells
        conf  : list[float|None] max probability (None for blank)
        blank : list[bool]
    """
    cell_images = np.asarray(cell_images, dtype=np.float32)
    if cell_images.shape != (81, 28, 28):
        raise ValueError(f"expected (81,28,28) cell images, got {cell_images.shape}")
    model = model or load_model()
    blank = [is_blank(c) for c in cell_images]
    probs = np.zeros((81, 9), dtype=np.float64)
    idx = [i for i in range(81) if not blank[i]]
    if idx:
        probs[idx] = predict_proba(model, cell_images[idx])
    grid = [0 if blank[i] else int(np.argmax(probs[i])) + 1 for i in range(81)]
    conf = [None if blank[i] else float(probs[i].max()) for i in range(81)]
    return {"grid": grid, "probs": probs, "conf": conf, "blank": blank}
