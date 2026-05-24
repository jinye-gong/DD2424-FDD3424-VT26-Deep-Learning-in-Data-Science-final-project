"""
ConvNeXt-style block template for CIFAR-10 (32x32).

Large depthwise kernel + LayerNorm + inverted bottleneck (GELU).
Tune `kernel_size` and `width` for scaling-law experiments.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    """Channel-last LayerNorm for NCHW tensors (ConvNeXt-style)."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt block: DW conv -> LN -> 1x1 expand -> GELU -> 1x1 project + residual.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        expansion: int = 4,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2
        hidden = dim * expansion

        self.dwconv = nn.Conv2d(dim, dim, kernel_size, padding=pad, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, hidden, 1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden, dim, 1)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return shortcut + self.drop_path(x)


class DropPath(nn.Module):
    """Stochastic depth (optional; default off in template)."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep) / keep
        return x * mask


class ConvNeXtSmall(nn.Module):
    """
    Compact ConvNeXt-style network for CIFAR-10.

    Stem -> 3 stages (downsample between) -> GAP -> linear classifier.
    Adjust `depths`, `width`, and `kernel_size` to match target FLOPs budgets.
    """

    def __init__(
        self,
        num_classes: int = 10,
        width: int = 64,
        depths: tuple[int, ...] = (2, 2, 2),
        kernel_size: int = 7,
        expansion: int = 4,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        w = width

        self.stem = nn.Sequential(
            nn.Conv2d(3, w, 3, padding=1, bias=False),
            LayerNorm2d(w),
        )

        stages: list[nn.Module] = []
        channels = w
        for i, n_blocks in enumerate(depths):
            blocks = [
                ConvNeXtBlock(channels, kernel_size=kernel_size, expansion=expansion)
                for _ in range(n_blocks)
            ]
            stage: list[nn.Module] = list(blocks)
            if i < len(depths) - 1:
                out_ch = channels * 2
                stage.append(nn.Sequential(
                    LayerNorm2d(channels),
                    nn.Conv2d(channels, out_ch, 2, stride=2),
                ))
                channels = out_ch
            stages.append(nn.Sequential(*stage))

        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, num_classes),
        )
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        return self.head(x)
