from __future__ import annotations

from typing import Callable

import torch.nn as nn

from models.convnext_style import ConvNeXtSmall
from models.depthwise import DepthwiseSmall
from models.vgg_baseline import VGGSmall

MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "vgg_baseline": VGGSmall,
    "depthwise_small": DepthwiseSmall,
    "convnext_small": ConvNeXtSmall,
}


def build_model(name: str, num_classes: int = 10, **kwargs) -> nn.Module:
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return MODEL_REGISTRY[name](num_classes=num_classes, **kwargs)


def list_models() -> list[str]:
    return sorted(MODEL_REGISTRY.keys())
