import json
from pathlib import Path

from ptcg_agent.deck_preselection import (
    deck_hash,
    load_cards,
    load_deck,
    preselect,
    validate_deck,
)


def test_current_deck_is_legal_and_hash_is_order_independent() -> None:
    cards = load_cards(Path("data/EN_Card_Data.csv"))
    deck = load_deck(Path("deck.csv"))
    assert validate_deck(deck, cards) == []
    assert deck_hash(deck) == deck_hash(tuple(reversed(deck)))


def test_preselection_is_deterministic_legal_and_includes_baseline(tmp_path: Path) -> None:
    args = {
        "cards_path": Path("data/EN_Card_Data.csv"),
        "baseline_path": Path("deck.csv"),
        "replay_paths": [Path("artifacts/sol-vs-fable-sot-1792.jsonl")],
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
    cards = load_cards(Path("data/EN_Card_Data.csv"))
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


def test_resume_reaches_same_shortlist(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    common = {
        "cards_path": Path("data/EN_Card_Data.csv"),
        "baseline_path": Path("deck.csv"),
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
