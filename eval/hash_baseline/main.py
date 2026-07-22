"""Frozen pre-SOT-1838 hash policy used only as the A/B control."""

import hashlib
import json
from pathlib import Path


def read_deck_csv() -> list[int]:
    return [int(value) for value in (Path(__file__).parent / "deck.csv").read_text().splitlines()]


def agent(obs_dict: dict) -> list[int]:
    select = (obs_dict or {}).get("select")
    if select is None:
        return read_deck_csv()
    options = select.get("option") or []
    minimum = int(select.get("minCount") or 0)
    ranked = sorted(
        range(len(options)),
        key=lambda index: hashlib.sha256(
            json.dumps(
                {"observation": obs_dict, "index": index, "option": options[index]},
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode()
        ).digest(),
        reverse=True,
    )
    return sorted(ranked[:minimum])
