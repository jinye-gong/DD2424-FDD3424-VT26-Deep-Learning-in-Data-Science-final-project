from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset
from torchvision.datasets import CIFAR10

SPLIT_DIR = Path(__file__).resolve().parent / "artifacts"
DEFAULT_INDICES_PATH = SPLIT_DIR / "cifar10_val_indices.pt"


def _make_indices(n: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    n_val = int(n * val_ratio)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx, val_idx


def load_or_create_split_indices(
    val_ratio: float = 0.1,
    seed: int = 42,
    path: Path = DEFAULT_INDICES_PATH,
) -> dict[str, list[int]]:
    """Load persisted train/val indices or create and save them once."""
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    if path.exists():
        data = torch.load(path, weights_only=False)
        return {"train_idx": data["train_idx"], "val_idx": data["val_idx"], "seed": data["seed"]}

    train_idx, val_idx = _make_indices(50_000, val_ratio, seed)
    payload = {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "seed": seed,
        "val_ratio": val_ratio,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
    }
    torch.save(payload, path)
    return {"train_idx": train_idx, "val_idx": val_idx, "seed": seed}


def get_cifar10_splits(
    data_root: str | Path,
    transform_train,
    transform_eval,
    val_ratio: float = 0.1,
    seed: int = 42,
    download: bool = True,
    indices_path: Path = DEFAULT_INDICES_PATH,
) -> tuple[Subset, Subset, CIFAR10, dict]:
    """
    Returns (train_subset, val_subset, test_set, split_info).

    train/val share the same underlying CIFAR-10 train split with different transforms.
    """
    indices = load_or_create_split_indices(val_ratio, seed, indices_path)

    base_train = CIFAR10(
        root=str(data_root),
        train=True,
        download=download,
        transform=transform_train,
    )
    base_eval = CIFAR10(
        root=str(data_root),
        train=True,
        download=False,
        transform=transform_eval,
    )
    test_set = CIFAR10(
        root=str(data_root),
        train=False,
        download=download,
        transform=transform_eval,
    )

    train_set = Subset(base_train, indices["train_idx"])
    val_set = Subset(base_eval, indices["val_idx"])

    info = {
        "seed": seed,
        "val_ratio": val_ratio,
        "n_train": len(indices["train_idx"]),
        "n_val": len(indices["val_idx"]),
        "indices_path": str(indices_path),
    }
    return train_set, val_set, test_set, info


def subset_targets(dataset: Dataset) -> list[int]:
    """Helper for sanity checks on class balance."""
    if isinstance(dataset, Subset):
        base: CIFAR10 = dataset.dataset  # type: ignore[assignment]
        return [base.targets[i] for i in dataset.indices]  # type: ignore[index]
    raise TypeError("Expected torch.utils.data.Subset")
