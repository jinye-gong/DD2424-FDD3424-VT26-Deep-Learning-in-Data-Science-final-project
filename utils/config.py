from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config:
    """Nested dict wrapper with attribute access."""

    def __init__(self, data: dict[str, Any]) -> None:
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                out[key] = value.to_dict()
            else:
                out[key] = value
        return out


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(raw)
