#!/usr/bin/env python3
"""Validate all experiments in configs/experiments.yaml can be built and profiled."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import build_model  # noqa: E402
from scripts.run_experiments import build_run_tag  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.metrics import count_flops  # noqa: E402


def parse_depths(s: str) -> tuple[int, ...]:
    parts = [int(x.strip()) for x in s.split(",")]
    return tuple(parts)


def main() -> int:
    cfg_path = ROOT / "configs" / "experiments.yaml"
    train_cfg = load_config(ROOT / "configs" / "default.yaml")
    inp = tuple(train_cfg.flops_input_size)

    with cfg_path.open(encoding="utf-8") as f:
        exps = yaml.safe_load(f)["experiments"]

    errors: list[str] = []
    print(f"Validating {len(exps)} experiments...\n")

    for i, exp in enumerate(exps):
        run_tag = build_run_tag(exp, seed=42, epochs=200)
        kwargs: dict = {}
        if "width" in exp:
            kwargs["width"] = exp["width"]
        if "kernel_size" in exp:
            kwargs["kernel_size"] = exp["kernel_size"]
        if "depths" in exp:
            kwargs["depths"] = parse_depths(exp["depths"])

        try:
            model = build_model(exp["model"], num_classes=10, **kwargs)
            stats = count_flops(model, inp)
            print(
                f"[{i:2d}] OK  {run_tag}\n"
                f"      params={stats['params']:,}  flops={stats['flops']:,}"
            )
        except Exception as e:
            errors.append(f"[{i}] {run_tag}: {e}")
            print(f"[{i:2d}] FAIL {run_tag}: {e}")

    print()
    if errors:
        print(f"FAILED: {len(errors)} experiment(s)")
        return 1
    print("All experiments passed build + FLOPs profile.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
