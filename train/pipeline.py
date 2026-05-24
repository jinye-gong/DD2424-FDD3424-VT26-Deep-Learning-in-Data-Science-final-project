from __future__ import annotations

from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torchvision import transforms

from data.splits import get_cifar10_splits
from utils.config import Config


def build_transforms(cfg: Config, train: bool) -> transforms.Compose:
    mean = cfg.normalize.mean
    std = cfg.normalize.std
    norm = transforms.Normalize(mean, std)

    if train:
        ops: list = []
        if cfg.augment.random_crop:
            ops.append(transforms.RandomCrop(32, padding=cfg.augment.crop_padding))
        if cfg.augment.random_horizontal_flip:
            ops.append(transforms.RandomHorizontalFlip())
        ops.extend([transforms.ToTensor(), norm])
        return transforms.Compose(ops)

    return transforms.Compose([transforms.ToTensor(), norm])


def build_dataloaders(cfg: Config) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    t_train = build_transforms(cfg, train=True)
    t_eval = build_transforms(cfg, train=False)

    train_set, val_set, test_set, split_info = get_cifar10_splits(
        data_root=cfg.data_root,
        transform_train=t_train,
        transform_eval=t_eval,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
    )

    loader_kwargs = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": cfg.pin_memory,
    }

    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, split_info


def build_optimizer(model: torch.nn.Module, cfg: Config) -> AdamW:
    if cfg.optimizer.lower() != "adamw":
        raise ValueError(f"Only adamw is supported in unified pipeline, got {cfg.optimizer}")
    return AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=tuple(cfg.betas),
    )


def build_scheduler(optimizer: AdamW, cfg: Config) -> torch.optim.lr_scheduler._LRScheduler:
    if cfg.scheduler.lower() != "cosine":
        raise ValueError(f"Only cosine schedule is supported, got {cfg.scheduler}")

    warmup_epochs = int(cfg.warmup_epochs)
    total_epochs = int(cfg.epochs)

    if warmup_epochs > 0:
        warmup = LinearLR(
            optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_epochs - warmup_epochs),
            eta_min=cfg.min_lr,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )

    return CosineAnnealingLR(
        optimizer,
        T_max=total_epochs,
        eta_min=cfg.min_lr,
    )
