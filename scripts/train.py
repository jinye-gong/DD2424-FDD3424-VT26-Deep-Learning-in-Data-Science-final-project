#!/usr/bin/env python3
"""Unified training entry point for Group 148."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import build_model, list_models  # noqa: E402
from train.pipeline import (  # noqa: E402
    build_dataloaders,
    build_optimizer,
    build_scheduler,
)
from train.trainer import Trainer  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.logging import RESULT_FIELDS, append_result_row, utc_timestamp  # noqa: E402
from utils.metrics import count_flops, format_compute_stats  # noqa: E402
from utils.seed import set_seed  # noqa: E402


def parse_depths(s: str | None) -> tuple[int, ...] | None:
    if s is None:
        return None
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError("depths must be three comma-separated integers, e.g. 2,2,2")
    return tuple(parts)


def build_run_tag(
    model: str,
    width: int | None,
    kernel_size: int | None,
    depths: tuple[int, ...] | None,
    seed: int,
    epochs: int,
) -> str:
    parts = [model]
    if width is not None:
        parts.append(f"w{width}")
    if kernel_size is not None:
        parts.append(f"k{kernel_size}")
    if depths is not None:
        parts.append(f"d{'-'.join(str(x) for x in depths)}")
    parts.append(f"seed{seed}")
    parts.append(f"ep{epochs}")
    return "_".join(parts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a model with unified DD2424 pipeline")
    p.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--model", type=str, default="vgg_baseline", choices=list_models())
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--kernel-size", type=int, default=None)
    p.add_argument("--depths", type=str, default=None, help="ConvNeXt only, e.g. 2,2,2")
    p.add_argument("--budget", type=str, default=None, help="low|med|high|scaling (for logging)")
    p.add_argument("--run-tag", type=str, default=None, help="Override checkpoint / log name")
    p.add_argument("--epochs", type=int, default=None, help="Override config epochs")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None, help="cuda or cpu")
    p.add_argument("--eval-only", action="store_true", help="Skip training, only profile + test")
    p.add_argument("--checkpoint", type=str, default=None, help="Load weights for eval-only")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.seed is not None:
        cfg.seed = args.seed

    depths = parse_depths(args.depths)
    model_kwargs: dict = {}
    if args.width is not None:
        model_kwargs["width"] = args.width
    if args.kernel_size is not None:
        model_kwargs["kernel_size"] = args.kernel_size
    if depths is not None:
        model_kwargs["depths"] = depths

    set_seed(cfg.seed)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"Device: {device} | PyTorch {torch.__version__}")

    model = build_model(args.model, num_classes=10, **model_kwargs)
    input_size = tuple(cfg.flops_input_size)
    compute = count_flops(model, input_size=input_size, device=device_str)
    print(f"Model: {args.model} | kwargs={model_kwargs} | {format_compute_stats(compute)}")

    run_tag = args.run_tag or build_run_tag(
        args.model, args.width, args.kernel_size, depths, cfg.seed, cfg.epochs
    )
    ckpt_path = Path(cfg.checkpoint_dir) / f"{run_tag}.pt"
    epoch_log = Path(cfg.log_dir) / "epochs" / f"{run_tag}.log"
    epoch_log.parent.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, split_info = build_dataloaders(cfg)
    print(
        f"Split: train={split_info['n_train']} val={split_info['n_val']} "
        f"(seed={split_info['seed']}, indices={split_info['indices_path']})"
    )
    print(f"Run tag: {run_tag}")

    trainer = Trainer(model, cfg, device, checkpoint_path=ckpt_path, epoch_log_path=epoch_log)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    trainer.setup_optim(optimizer, scheduler)

    if args.checkpoint:
        trainer.load_checkpoint(Path(args.checkpoint))

    if not args.eval_only:
        history = trainer.fit(train_loader, val_loader)
        print(f"Best val acc: {history['best_val_acc']:.4f}")
        trainer.load_checkpoint(ckpt_path)
    elif not args.checkpoint:
        print("Warning: eval-only without --checkpoint uses random weights for test.")

    test_loss, test_acc = trainer.evaluate(test_loader)
    print(f"Test loss={test_loss:.4f} test_acc={test_acc:.4f}")

    depths_str = ",".join(str(x) for x in depths) if depths else ""
    results_path = Path(cfg.results_csv)
    append_result_row(
        results_path,
        {
            "timestamp": utc_timestamp(),
            "run_tag": run_tag,
            "model": args.model,
            "seed": cfg.seed,
            "epochs": cfg.epochs,
            "width": args.width if args.width is not None else "",
            "kernel_size": args.kernel_size if args.kernel_size is not None else "",
            "depths": depths_str,
            "budget": args.budget or "",
            "params": compute["params"],
            "macs": compute["macs"],
            "flops": compute["flops"],
            "best_val_acc": round(trainer.best_val_acc, 6),
            "test_acc": round(test_acc, 6),
            "checkpoint": str(ckpt_path),
            "config": str(args.config),
        },
        RESULT_FIELDS,
    )
    print(f"Results appended to {results_path}")


if __name__ == "__main__":
    main()
