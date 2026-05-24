"""
Depthwise-separable ConvNet template for CIFAR-10.

Extend this file for scaling studies (width / depth / kernel size).
Match FLOPs against vgg_baseline via the `width` argument and block counts.
"""

from __future__ import annotations

import torch.nn as nn


class DepthwiseSeparableBNReLU(nn.Module):
    """Depthwise conv + pointwise conv + BN + ReLU."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        padding: int | None = None,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                in_ch,
                kernel_size,
                padding=padding,
                groups=in_ch,
                bias=False,
            ),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DepthwiseSmall(nn.Module):
    """
    VGG-shaped skeleton with depthwise-separable blocks.

    Template defaults mirror vgg_baseline stage layout (3 pools -> 4x4).
    """

    def __init__(
        self,
        num_classes: int = 10,
        width: int = 64,
        dropout: float = 0.5,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        w = width
        pad = kernel_size // 2
        block = lambda inc, outc: DepthwiseSeparableBNReLU(inc, outc, kernel_size, pad)

        self.features = nn.Sequential(
            block(3, w),
            block(w, w),
            nn.MaxPool2d(2),
            block(w, w * 2),
            block(w * 2, w * 2),
            nn.MaxPool2d(2),
            block(w * 2, w * 4),
            block(w * 4, w * 4),
            nn.MaxPool2d(2),
        )
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
