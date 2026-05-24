from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def ensure_csv_header(path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def _header_matches(path: Path, fieldnames: list[str]) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open(encoding="utf-8") as f:
        first = f.readline().strip()
    return first == ",".join(fieldnames)


def append_result_row(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not _header_matches(path, fieldnames):
        backup = path.with_suffix(".csv.bak")
        path.rename(backup)
        print(f"Warning: results CSV header mismatch; backed up to {backup}")
    ensure_csv_header(path, fieldnames)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


RESULT_FIELDS = [
    "timestamp",
    "run_tag",
    "model",
    "seed",
    "epochs",
    "width",
    "kernel_size",
    "depths",
    "budget",
    "params",
    "macs",
    "flops",
    "best_val_acc",
    "test_acc",
    "checkpoint",
    "config",
]
