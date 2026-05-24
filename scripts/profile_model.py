#!/usr/bin/env python3
"""Profile params / MACs / FLOPs using the unified metrics module."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import build_model, list_models  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.metrics import count_flops, format_compute_stats  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--model", type=str, required=True, choices=list_models())
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    cfg = load_config(args.config)
    model = build_model(args.model)
    stats = count_flops(model, tuple(cfg.flops_input_size), device=args.device)
    print(f"Model: {args.model}")
    print(format_compute_stats(stats))
    print("Convention: FLOPs = 2 * MACs (thop profile)")


if __name__ == "__main__":
    main()
