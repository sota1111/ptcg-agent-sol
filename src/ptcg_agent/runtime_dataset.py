"""Loader for deterministic public-observation runtime replay datasets."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuntimeTrainingSample:
    features: tuple[float, ...]
    action: int
    legal_actions: tuple[int, ...]
    value: float
    split: str


def load_runtime_dataset(path: Path, split: str | None = None) -> Iterator[RuntimeTrainingSample]:
    """Stream validated samples without requiring a GPU or engine runtime."""
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            raw: dict[str, Any] = json.loads(line)
            if raw.get("schemaVersion") != "1.0.0":
                raise ValueError(f"line {number}: unsupported dataset schema")
            sample_split = raw.get("split")
            if sample_split not in {"train", "validation", "holdout"}:
                raise ValueError(f"line {number}: invalid split")
            if split is not None and sample_split != split:
                continue
            features = tuple(float(value) for value in raw.get("features", ()))
            legal = tuple(int(action) for action in raw.get("legalActions", ()))
            action = int(raw["action"])
            if len(features) != 7 or action not in legal:
                raise ValueError(f"line {number}: invalid feature or action payload")
            yield RuntimeTrainingSample(
                features=features,
                action=action,
                legal_actions=legal,
                value=float(raw["value"]),
                split=sample_split,
            )
