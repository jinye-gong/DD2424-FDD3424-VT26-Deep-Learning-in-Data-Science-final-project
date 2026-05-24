from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from thop import profile


def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def count_flops(
    model: nn.Module,
    input_size: tuple[int, ...] = (1, 3, 32, 32),
    device: str | torch.device = "cpu",
) -> dict[str, int]:
    """
    Unified FLOPs counting via thop.

    Returns MACs and FLOPs where FLOPs = 2 * MACs (group convention).
    Always call this function for Pareto / scaling plots — do not hand-roll.
    """
    model = model.to(device)
    was_training = model.training
    model.eval()

    dummy = torch.randn(*input_size, device=device)
    macs, params_thop = profile(model, inputs=(dummy,), verbose=False)

    if was_training:
        model.train()

    macs_i = int(macs)
    return {
        "macs": macs_i,
        "flops": 2 * macs_i,
        "params_thop": int(params_thop),
        "params": count_params(model),
    }


def format_compute_stats(stats: dict[str, Any]) -> str:
    return (
        f"params={stats['params']:,} | "
        f"MACs={stats['macs']:,} | "
        f"FLOPs={stats['flops']:,}"
    )
