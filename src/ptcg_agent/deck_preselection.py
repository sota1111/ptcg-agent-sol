"""Deterministic, CPU-only deck candidate preselection."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CURRENT_REGULATION = "sv-current-2026-07"
DECK_SIZE = 60
COPY_LIMIT = 4


@dataclass(frozen=True)
class Card:
    card_id: int
    name: str
    expansion: str
    kind: str
    stage: str
    previous_stage: str
    card_type: str
    effect: str

    @property
    def is_basic_energy(self) -> bool:
        return self.kind == "Basic Energy"


def load_cards(path: Path) -> dict[int, Card]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        return {
            int(row["Card ID"]): Card(
                card_id=int(row["Card ID"]),
                name=row["Card Name"],
                expansion=row["Expansion"],
                kind=row["Stage (Pokémon)/Type (Energy and Trainer)"],
                stage=row["Stage (Pokémon)/Type (Energy and Trainer)"],
                previous_stage=row["Previous stage"],
                card_type=row["Type"],
                effect=row["Effect Explanation"],
            )
            for row in rows
            if row["Card ID"].isdigit() and row["Expansion"]
        }


def load_deck(path: Path) -> tuple[int, ...]:
    return tuple(
        int(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )


def deck_hash(deck: tuple[int, ...]) -> str:
    payload = ",".join(str(card) for card in sorted(deck)).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_deck(deck: tuple[int, ...], cards: dict[int, Card]) -> list[str]:
    errors: list[str] = []
    if len(deck) != DECK_SIZE:
        errors.append(f"deck has {len(deck)} cards, expected {DECK_SIZE}")
    unknown = sorted(set(deck) - cards.keys())
    if unknown:
        errors.append(f"unknown or out-of-regulation card ids: {unknown}")
    for card_id, copies in Counter(deck).items():
        card = cards.get(card_id)
        if card is not None and not card.is_basic_energy and copies > COPY_LIMIT:
            errors.append(f"{card.name} has {copies} copies, maximum is {COPY_LIMIT}")
    if not any(cards[card_id].stage == "Basic Pokémon" for card_id in deck if card_id in cards):
        errors.append("deck has no Basic Pokémon")
    return errors


def _compatibility(card: Card, deck_cards: list[Card]) -> float:
    score = 0.0
    names = {item.name for item in deck_cards}
    types = Counter(item.card_type for item in deck_cards if item.card_type not in {"", "n/a"})
    if card.previous_stage in names:
        score += 3.0
    if card.stage == "Basic Pokémon":
        score += 0.8
    if card.card_type in types:
        score += min(types[card.card_type], 8) * 0.12
    text = card.effect.lower()
    for token in ("draw", "search your deck", "energy", "basic pokémon", "damage"):
        if token in text:
            score += 0.25
    return score


def _replay_metrics(paths: list[Path]) -> dict[str, float]:
    completed = wins = decisions = action_total = 0
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("status") == "completed":
                completed += 1
                wins += int(row.get("winner") == 0)
                decisions += int(row.get("decisions", 0))
            elif isinstance(row.get("action"), list):
                action_total += len(row["action"])
    return {
        "games": float(completed),
        "baselineWinRate": wins / completed if completed else 0.0,
        "meanDecisions": decisions / completed if completed else 0.0,
        "actionDensity": action_total / decisions if decisions else 0.0,
    }


def _score(
    deck: tuple[int, ...], cards: dict[int, Card], replay: dict[str, float]
) -> dict[str, Any]:
    selected = [cards[card_id] for card_id in deck]
    counts = Counter(deck)
    pokemon = sum(card.kind.endswith("Pokémon") for card in selected)
    trainers = sum(
        card.kind in {"Item", "Supporter", "Stadium", "Pokémon Tool"} for card in selected
    )
    energy = sum("Energy" in card.kind for card in selected)
    consistency = 1.0 - abs(pokemon - 14) / 60 - abs(trainers - 30) / 60 - abs(energy - 16) / 60
    synergy = (
        sum(
            _compatibility(card, selected) * count
            for card, count in ((cards[k], v) for k, v in counts.items())
        )
        / 60
    )
    tempo = min(1.0, trainers / 30) * 0.55 + min(1.0, pokemon / 14) * 0.45
    hard = consistency * 0.35 + synergy * 0.25 + tempo * 0.25 + replay["baselineWinRate"] * 0.15
    return {
        "consistency": round(consistency, 6),
        "synergy": round(synergy, 6),
        "tempo": round(tempo, 6),
        "hardOpponents": {
            "fable": round(hard, 6),
            "hash_baseline": round(hard * 0.8 + replay["actionDensity"] * 0.2, 6),
        },
        "aggregate": round(hard, 6),
    }


def _mutate(
    baseline: tuple[int, ...], cards: dict[int, Card], rng: random.Random
) -> tuple[int, ...]:
    deck = list(baseline)
    selected = [cards[card_id] for card_id in deck]
    pool = sorted(cards.values(), key=lambda card: (-_compatibility(card, selected), card.card_id))[
        :250
    ]
    for _ in range(rng.randint(1, 3)):
        remove_index = rng.randrange(len(deck))
        counts = Counter(deck)
        choices = [
            card.card_id
            for card in pool
            if card.is_basic_energy or counts[card.card_id] < COPY_LIMIT
        ]
        deck[remove_index] = rng.choice(choices)
    return tuple(sorted(deck))


def preselect(
    *,
    cards_path: Path,
    baseline_path: Path,
    replay_paths: list[Path],
    output_path: Path,
    checkpoint_path: Path,
    seed: int,
    budget: int,
    top_k: int,
    resume: bool,
) -> dict[str, Any]:
    if budget < 1 or top_k < 1:
        raise ValueError("budget and top-k must be positive")
    cards = load_cards(cards_path)
    baseline = tuple(sorted(load_deck(baseline_path)))
    errors = validate_deck(baseline, cards)
    if errors:
        raise ValueError("baseline is illegal: " + "; ".join(errors))
    rng = random.Random(seed)
    candidates: dict[str, tuple[int, ...]] = {deck_hash(baseline): baseline}
    completed = 0
    if resume and checkpoint_path.exists():
        state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if state["seed"] != seed or state["budget"] != budget:
            raise ValueError("checkpoint seed/budget does not match")
        rng.setstate(_to_tuple(state["rngState"]))
        candidates.update({item["hash"]: tuple(item["cards"]) for item in state["candidates"]})
        completed = int(state["completed"])
    for index in range(completed, budget):
        candidate = _mutate(baseline, cards, rng)
        if not validate_deck(candidate, cards):
            candidates[deck_hash(candidate)] = candidate
        if (index + 1) % 100 == 0 or index + 1 == budget:
            _write_json(
                checkpoint_path,
                {
                    "schemaVersion": "1.0.0",
                    "seed": seed,
                    "budget": budget,
                    "completed": index + 1,
                    "rngState": rng.getstate(),
                    "candidates": [
                        {"hash": key, "cards": list(value)}
                        for key, value in sorted(candidates.items())
                    ],
                },
            )
    replay = _replay_metrics(replay_paths)
    ranked: list[dict[str, Any]] = []
    baseline_hash = deck_hash(baseline)
    for candidate_hash, candidate in candidates.items():
        ranked.append(
            {
                "hash": candidate_hash,
                "cards": list(candidate),
                "isBaseline": candidate_hash == baseline_hash,
                "scores": _score(candidate, cards, replay),
                "archetype": _archetype(candidate, cards),
            }
        )
    ranked.sort(key=lambda item: (-item["scores"]["aggregate"], item["hash"]))
    shortlist = _diverse_shortlist(ranked, top_k, baseline_hash)
    objective_paths = {
        "consistency": ("scores", "consistency"),
        "synergy": ("scores", "synergy"),
        "tempo": ("scores", "tempo"),
        "fable": ("scores", "hardOpponents", "fable"),
        "hash_baseline": ("scores", "hardOpponents", "hash_baseline"),
    }
    ablation: dict[str, str] = {}
    for name, path in objective_paths.items():
        winner = max(ranked, key=lambda item: _nested_score(item, path))
        ablation[name] = str(winner["hash"])
    archetypes = {item["archetype"] for item in shortlist}
    pairwise = [
        _jaccard_distance(left["cards"], right["cards"])
        for index, left in enumerate(shortlist)
        for right in shortlist[index + 1 :]
    ]
    result = {
        "schemaVersion": "1.0.0",
        "regulation": CURRENT_REGULATION,
        "generatedAt": int(time.time()),
        "seed": seed,
        "budget": budget,
        "completed": budget,
        "baselineHash": baseline_hash,
        "replayMetrics": replay,
        "candidateCount": len(ranked),
        "shortlist": shortlist,
        "diversity": {
            "archetypeCount": len(archetypes),
            "meanPairwiseJaccardDistance": round(sum(pairwise) / len(pairwise), 6)
            if pairwise
            else 0.0,
        },
        "ablation": {
            "description": "best candidate when each surrogate objective is optimized alone",
            "topHashByObjective": ablation,
        },
    }
    result["manifestHash"] = hashlib.sha256(
        json.dumps({k: v for k, v in result.items() if k != "generatedAt"}, sort_keys=True).encode()
    ).hexdigest()
    _write_json(output_path, result)
    return result


def _archetype(deck: tuple[int, ...], cards: dict[int, Card]) -> str:
    types = Counter(
        cards[card_id].card_type
        for card_id in deck
        if cards[card_id].kind.endswith("Pokémon")
        if cards[card_id].card_type not in {"", "n/a"}
    )
    primary_type = types.most_common(1)[0][0] if types else "colorless"
    pokemon = sorted(
        {cards[card_id].name for card_id in deck if cards[card_id].kind.endswith("Pokémon")}
    )
    return f"{primary_type}:{'/'.join(pokemon)}"


def _diverse_shortlist(
    ranked: list[dict[str, Any]], top_k: int, baseline_hash: str
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    baseline = next(item for item in ranked if item["hash"] == baseline_hash)
    selected.append(baseline)
    seen.add(baseline_hash)
    for prefer_new in (True, False):
        archetypes = {item["archetype"] for item in selected}
        for item in ranked:
            if len(selected) >= top_k:
                break
            if item["hash"] in seen or (prefer_new and item["archetype"] in archetypes):
                continue
            selected.append(item)
            seen.add(item["hash"])
            archetypes.add(item["archetype"])
    return selected


def _to_tuple(value: Any) -> Any:
    return tuple(_to_tuple(item) for item in value) if isinstance(value, list) else value


def _nested_score(item: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = item
    for key in path:
        value = value[key]
    return float(value)


def _jaccard_distance(left: list[int], right: list[int]) -> float:
    left_counts, right_counts = Counter(left), Counter(right)
    intersection = sum((left_counts & right_counts).values())
    union = sum((left_counts | right_counts).values())
    return 1.0 - intersection / union


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
