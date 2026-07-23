import json
from pathlib import Path

import pytest

from ptcg_agent.deck_preselection import (
    deck_hash,
    load_cards,
    load_deck,
    preselect,
    validate_deck,
)


@pytest.fixture
def deck_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    cards = tmp_path / "cards.csv"
    cards.write_text(
        "Card ID,Card Name,Expansion,Collection No.,"
        "Stage (Pokémon)/Type (Energy and Trainer),Rule,Category,Previous stage,"
        "HP,Type,Weakness,Resistance (Type),Retreat,Move Name,Cost,Damage,"
        "Effect Explanation\n"
        "1,Basic Water Energy,SVE,1,Basic Energy,n/a,n/a,n/a,n/a,{W},,,n/a,,n/a,n/a,\n"
        "2,Sprout,TEST,1,Basic Pokémon,n/a,n/a,n/a,60,{G},,,1,Tackle,{C},10,\n"
        "3,Research,TEST,2,Supporter,n/a,n/a,n/a,n/a,n/a,,,n/a,,n/a,n/a,Draw cards\n"
        "4,Ball,TEST,3,Item,n/a,n/a,n/a,n/a,n/a,,,n/a,,n/a,n/a,Search your deck\n"
        "5,Fish,TEST,4,Basic Pokémon,n/a,n/a,n/a,60,{W},,,1,Splash,{W},10,\n"
        "6,Bird,TEST,5,Basic Pokémon,n/a,n/a,n/a,60,{C},,,1,Peck,{C},10,\n"
        "7,Flame,TEST,6,Basic Pokémon,n/a,n/a,n/a,60,{R},,,1,Heat,{R},10,\n",
        encoding="utf-8",
    )
    deck = tmp_path / "deck.csv"
    deck.write_text("\n".join(["1"] * 48 + ["2"] * 4 + ["3"] * 4 + ["4"] * 4) + "\n")
    replay = tmp_path / "replay.jsonl"
    replay.write_text(
        '{"action":[1],"player":0,"step":0}\n{"decisions":1,"status":"completed","winner":0}\n',
        encoding="utf-8",
    )
    return cards, deck, replay


def test_current_deck_is_legal_and_hash_is_order_independent(
    deck_inputs: tuple[Path, Path, Path],
) -> None:
    cards_path, deck_path, _ = deck_inputs
    cards = load_cards(cards_path)
    deck = load_deck(deck_path)
    assert validate_deck(deck, cards) == []
    assert deck_hash(deck) == deck_hash(tuple(reversed(deck)))


def test_preselection_is_deterministic_legal_and_includes_baseline(
    tmp_path: Path, deck_inputs: tuple[Path, Path, Path]
) -> None:
    cards_path, deck_path, replay_path = deck_inputs
    args = {
        "cards_path": cards_path,
        "baseline_path": deck_path,
        "replay_paths": [replay_path],
        "seed": 19,
        "budget": 120,
        "top_k": 8,
        "resume": False,
    }
    first = preselect(
        **args,
        output_path=tmp_path / "first.json",
        checkpoint_path=tmp_path / "first-checkpoint.json",
    )
    second = preselect(
        **args,
        output_path=tmp_path / "second.json",
        checkpoint_path=tmp_path / "second-checkpoint.json",
    )
    assert first["manifestHash"] == second["manifestHash"]
    assert [item["hash"] for item in first["shortlist"]] == [
        item["hash"] for item in second["shortlist"]
    ]
    assert first["shortlist"][0]["isBaseline"]
    cards = load_cards(cards_path)
    assert all(not validate_deck(tuple(item["cards"]), cards) for item in first["shortlist"])
    assert all(item["scores"]["hardOpponents"] for item in first["shortlist"])
    assert set(first["ablation"]["topHashByObjective"]) == {
        "consistency",
        "synergy",
        "tempo",
        "fable",
        "hash_baseline",
    }
    assert first["diversity"]["meanPairwiseJaccardDistance"] > 0


def test_resume_reaches_same_shortlist(
    tmp_path: Path, deck_inputs: tuple[Path, Path, Path]
) -> None:
    cards_path, deck_path, _ = deck_inputs
    checkpoint = tmp_path / "checkpoint.json"
    common = {
        "cards_path": cards_path,
        "baseline_path": deck_path,
        "replay_paths": [],
        "seed": 7,
        "budget": 101,
        "top_k": 5,
    }
    expected = preselect(
        **common,
        output_path=tmp_path / "expected.json",
        checkpoint_path=tmp_path / "expected-checkpoint.json",
        resume=False,
    )
    state = json.loads((tmp_path / "expected-checkpoint.json").read_text())
    checkpoint.write_text(json.dumps(state))
    resumed = preselect(
        **common,
        output_path=tmp_path / "resumed.json",
        checkpoint_path=checkpoint,
        resume=True,
    )
    assert resumed["manifestHash"] == expected["manifestHash"]
