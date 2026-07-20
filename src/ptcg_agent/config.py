"""Validated runtime configuration."""

import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    max_hours: float
    seed: int
    batch_size: int
    mixed_precision: bool

    def validate(self) -> "RuntimeConfig":
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        if not 0 < self.max_hours <= 8:
            raise ValueError("max_hours must be greater than 0 and at most 8")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.mixed_precision and self.device != "cuda":
            raise ValueError("mixed_precision requires device='cuda'")
        return self


def load_config(path: Path, max_hours: float | None = None) -> RuntimeConfig:
    with path.open("rb") as handle:
        raw: dict[str, Any] = tomllib.load(handle)
    required = {"device", "max_hours", "seed", "batch_size", "mixed_precision"}
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"missing config keys: {', '.join(sorted(missing))}")
    config = RuntimeConfig(
        device=str(raw["device"]),
        max_hours=float(raw["max_hours"]),
        seed=int(raw["seed"]),
        batch_size=int(raw["batch_size"]),
        mixed_precision=bool(raw["mixed_precision"]),
    )
    if max_hours is not None:
        config = replace(config, max_hours=max_hours)
    return config.validate()
