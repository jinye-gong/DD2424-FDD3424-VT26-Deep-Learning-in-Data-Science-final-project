"""
VGG-style baseline for CIFAR-10 (32x32).

Team members add new architectures in models/ and register them in registry.py.
"""

from __future__ import annotations

import torch.nn as nn


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class VGGSmall(nn.Module):
    """Compact VGG-style network with BN and dropout — unified baseline."""

    def __init__(self, num_classes: int = 10, width: int = 64, dropout: float = 0.5) -> None:
        super().__init__()
        w = width

        self.features = nn.Sequential(
            ConvBNReLU(3, w),
            ConvBNReLU(w, w),
            nn.MaxPool2d(2),
            ConvBNReLU(w, w * 2),
            ConvBNReLU(w * 2, w * 2),
            nn.MaxPool2d(2),
            ConvBNReLU(w * 2, w * 4),
            ConvBNReLU(w * 4, w * 4),
            nn.MaxPool2d(2),
        )
        # After 3x MaxPool2d: spatial 4x4, channels w*4
        feat_dim = w * 4 * 4 * 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, w * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(w * 4, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
