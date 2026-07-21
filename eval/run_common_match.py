"""Run one real-engine match between two Kaggle-compatible submissions."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

Agent = Callable[[dict[str, Any]], list[int]]


def load_agent(repo: Path, name: str) -> Agent:
    spec = importlib.util.spec_from_file_location(name, repo / "main.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load agent from {repo}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(repo))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(repo))
    loaded = getattr(module, "agent", None)
    if not callable(loaded):
        raise RuntimeError(f"{repo}/main.py does not expose agent")
    return cast(Agent, loaded)


def load_deck(repo: Path) -> list[int]:
    deck = [int(value) for value in (repo / "deck.csv").read_text().splitlines() if value]
    if len(deck) != 60:
        raise ValueError(f"{repo}/deck.csv contains {len(deck)} cards, expected 60")
    return deck


def run(sol_repo: Path, opponent_repo: Path, log_path: Path, max_steps: int) -> dict[str, Any]:
    sys.path.insert(0, str(sol_repo))
    game: Any = importlib.import_module("cg.game")

    agents = [
        load_agent(sol_repo, "sol_submission"),
        load_agent(opponent_repo, "opponent_submission"),
    ]
    observation, start = game.battle_start(load_deck(sol_repo), load_deck(opponent_repo))
    if observation is None:
        raise RuntimeError(
            f"BattleStart failed: errorPlayer={start.errorPlayer} errorType={start.errorType}"
        )
    events: list[dict[str, Any]] = []
    try:
        for step in range(max_steps):
            current = observation.get("current") or {}
            result = current.get("result", -1)
            if result != -1:
                summary = {"status": "completed", "winner": result, "decisions": step}
                events.append(summary)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text("\n".join(json.dumps(e, sort_keys=True) for e in events) + "\n")
                return summary
            player = int(current.get("player", step % 2))
            action = agents[player](observation)
            events.append({"step": step, "player": player, "action": action})
            observation = game.battle_select(action)
        raise RuntimeError(f"match did not finish within {max_steps} decisions")
    finally:
        game.battle_finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opponent", type=Path, required=True)
    parser.add_argument("--log", type=Path, default=Path("artifacts/common-match.jsonl"))
    parser.add_argument("--max-steps", type=int, default=100000)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    print(json.dumps(run(repo, args.opponent.resolve(), args.log, args.max_steps), sort_keys=True))


if __name__ == "__main__":
    main()
