#!/usr/bin/env python3
"""Run experiment queue from configs/experiments.yaml sequentially."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def build_run_tag(exp: dict, seed: int, epochs: int) -> str:
    parts = [exp["model"]]
    if "width" in exp:
        parts.append(f"w{exp['width']}")
    if "kernel_size" in exp:
        parts.append(f"k{exp['kernel_size']}")
    if "depths" in exp:
        d = exp["depths"].replace(",", "-")
        parts.append(f"d{d}")
    parts.extend([f"seed{seed}", f"ep{epochs}"])
    return "_".join(parts)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=str(ROOT / "configs" / "experiments.yaml"))
    p.add_argument("--train-config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--start", type=int, default=0, help="Start index in experiment list")
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not args.dry_run:
        pid_path = ROOT / "logs" / "queue.pid"
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.write_text(str(os.getpid()), encoding="utf-8")

    exps = data["experiments"][args.start :]
    ckpt_dir = ROOT / "checkpoints"
    train_script = ROOT / "scripts" / "train.py"

    for i, exp in enumerate(exps, start=args.start):
        run_tag = build_run_tag(exp, args.seed, args.epochs)
        ckpt = ckpt_dir / f"{run_tag}.pt"
        note = exp.get("note", "")

        legacy_ckpts = {
            "convnext_small_w64_seed42_ep200": ckpt_dir / "convnext_small_seed42_ep200.pt",
            "vgg_baseline_w64_seed42_ep200": ckpt_dir / "vgg_baseline_seed42_ep200.pt",
        }
        legacy_path = legacy_ckpts.get(run_tag)
        already_done = ckpt.exists() or (
            legacy_path is not None and legacy_path.exists()
        )
        if args.skip_existing and already_done:
            print(f"[{i+1}/{len(data['experiments'])}] SKIP (exists): {run_tag}  {note}")
            continue

        cmd = [
            sys.executable,
            str(train_script),
            "--config",
            args.train_config,
            "--model",
            exp["model"],
            "--epochs",
            str(args.epochs),
            "--seed",
            str(args.seed),
            "--run-tag",
            run_tag,
        ]
        if "width" in exp:
            cmd.extend(["--width", str(exp["width"])])
        if "kernel_size" in exp:
            cmd.extend(["--kernel-size", str(exp["kernel_size"])])
        if "depths" in exp:
            cmd.extend(["--depths", exp["depths"]])
        if "budget" in exp:
            cmd.extend(["--budget", exp["budget"]])

        total = len(data["experiments"])
        print(f"\n[{i+1}/{total}] RUN: {run_tag}  ({note})", flush=True)
        print(" ".join(cmd), flush=True)

        if args.dry_run:
            continue

        result = subprocess.run(cmd, cwd=ROOT, check=False)
        if result.returncode != 0:
            print(f"FAILED: {run_tag} exit code {result.returncode}", flush=True)
            raise SystemExit(result.returncode)

    if not args.dry_run:
        pid_path = ROOT / "logs" / "queue.pid"
        pid_path.unlink(missing_ok=True)
        print("\nAll experiments finished.", flush=True)


if __name__ == "__main__":
    main()
