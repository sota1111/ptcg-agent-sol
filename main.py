"""Kaggle PTCG submission entry point for Sol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent


def read_deck_csv() -> list[int]:
    deck = [int(line) for line in (_ROOT / "deck.csv").read_text().splitlines() if line]
    if len(deck) != 60:
        raise ValueError(f"deck.csv must contain exactly 60 cards (found {len(deck)})")
    return deck


def _stable_option_score(observation: dict[str, Any], index: int, option: Any) -> bytes:
    payload = {"observation": observation, "index": index, "option": option}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).digest()


def agent(obs_dict: dict[str, Any]) -> list[int]:
    """Return the initial 60-card deck or legal indices for a selection prompt."""
    select = (obs_dict or {}).get("select")
    if select is None:
        return read_deck_csv()
    if not isinstance(select, dict):
        raise ValueError("select must be an object or null")
    options = select.get("option") or []
    if not isinstance(options, list):
        raise ValueError("select.option must be a list")
    minimum = int(select.get("minCount") or 0)
    maximum_raw = select.get("maxCount")
    maximum = minimum if maximum_raw is None else int(str(maximum_raw))
    if minimum < 0 or maximum < minimum or maximum > len(options):
        raise ValueError("invalid selection bounds")
    ranked = sorted(
        range(len(options)),
        key=lambda index: _stable_option_score(obs_dict, index, options[index]),
        reverse=True,
    )
    return sorted(ranked[:minimum])


__all__ = ["agent", "read_deck_csv"]
