"""CNN that classifies 28x28 grayscale handwritten digits into 9 classes (1..9).

Class index k corresponds to digit k+1.

Architecture (≈ 225k parameters, trains in a few minutes on a laptop CPU):
    Input 1x28x28
    Conv 3x3, 32 ch, BatchNorm, ReLU  -> MaxPool 2  (32x14x14)
    Conv 3x3, 64 ch, BatchNorm, ReLU  -> MaxPool 2  (64x7x7)
    Flatten -> Dropout 0.3 -> Linear 3136->64 -> ReLU -> Dropout 0.3 -> Linear 64->9 (logits)

forward() returns *logits* (needed by nn.CrossEntropyLoss, which applies
log-softmax internally). predict_proba() applies Softmax to give the
probability distribution used by Search.
"""
from __future__ import annotations

import torch
from torch import nn


class DigitCNN(nn.Module):
    def __init__(self, num_classes: int = 9, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(64 * 7 * 7, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:                 # (N,28,28) -> (N,1,28,28)
            x = x.unsqueeze(1)
        return self.classifier(self.features(x))

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.softmax(self.forward(x), dim=1)


def get_device(prefer_gpu: bool = True) -> torch.device:
    """CUDA if available (and wanted), otherwise CPU."""
    return torch.device("cuda" if prefer_gpu and torch.cuda.is_available() else "cpu")
