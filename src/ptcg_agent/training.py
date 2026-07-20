"""Reproducible self-play training with atomic, resumable checkpoints."""

import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ptcg_agent.policy import LegalPolicy

CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class TrainingResult:
    episodes: int
    wins: int
    losses: int
    draws: int
    interrupted: bool


def _episode(policy: LegalPolicy, rng: random.Random, epsilon: float, alpha: float) -> int:
    """Small deterministic battle proxy: players remove 1-3 tokens; taking the last wins."""
    remaining = rng.randint(8, 16)
    history: list[tuple[dict[str, int], int, int]] = []
    player = 1
    while remaining:
        observation = {"remaining": remaining, "player": player}
        legal = list(range(1, min(3, remaining) + 1))
        action = int(policy.choose(observation, legal, rng=rng, epsilon=epsilon))
        history.append((observation, action, player))
        remaining -= action
        if remaining == 0:
            winner = player
            break
        player *= -1
    for observation, action, actor in history:
        policy.update(observation, action, 1.0 if actor == winner else -1.0, alpha)
    return winner


def save_checkpoint(
    path: Path,
    policy: LegalPolicy,
    seed: int,
    result: TrainingResult,
    rng_state: tuple[Any, ...] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CHECKPOINT_VERSION,
        "seed": seed,
        "result": asdict(result),
        "values": policy.values,
        "rng_state": rng_state,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_checkpoint(path: Path, seed: int) -> tuple[LegalPolicy, TrainingResult]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported checkpoint version")
    if payload.get("seed") != seed:
        raise ValueError("checkpoint seed does not match configured seed")
    result = TrainingResult(**payload["result"])
    return LegalPolicy(payload["values"]), result


def train(
    *,
    seed: int,
    episodes: int,
    checkpoint: Path,
    max_seconds: float,
    checkpoint_every: int = 100,
    epsilon: float = 0.15,
    alpha: float = 0.2,
    resume: bool = False,
) -> tuple[LegalPolicy, TrainingResult]:
    if episodes < 1 or max_seconds <= 0 or checkpoint_every < 1:
        raise ValueError("episodes, max_seconds, and checkpoint_every must be positive")
    if resume and checkpoint.exists():
        policy, previous = load_checkpoint(checkpoint, seed)
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    else:
        policy = LegalPolicy()
        previous = TrainingResult(0, 0, 0, 0, False)
        saved = {}
    rng = random.Random(seed)
    if saved.get("rng_state") is not None:
        rng.setstate(_as_tuple(saved["rng_state"]))
    started = time.monotonic()
    wins, losses, draws = previous.wins, previous.losses, previous.draws
    completed = previous.episodes
    interrupted = False
    while completed < episodes:
        if time.monotonic() - started >= max_seconds:
            interrupted = True
            break
        winner = _episode(policy, rng, epsilon, alpha)
        wins += int(winner == 1)
        losses += int(winner == -1)
        completed += 1
        if completed % checkpoint_every == 0:
            save_checkpoint(
                checkpoint,
                policy,
                seed,
                TrainingResult(completed, wins, losses, draws, False),
                rng.getstate(),
            )
    result = TrainingResult(completed, wins, losses, draws, interrupted)
    save_checkpoint(checkpoint, policy, seed, result, rng.getstate())
    return policy, result


def _as_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_as_tuple(item) for item in value)
    return value


def evaluate(policy: LegalPolicy, seed: int, episodes: int) -> dict[str, int]:
    rng = random.Random(seed)
    learned_wins = 0
    baseline_wins = 0
    for _ in range(episodes):
        remaining = rng.randint(8, 16)
        player = 1
        while remaining:
            observation = {"remaining": remaining, "player": player}
            legal = list(range(1, min(3, remaining) + 1))
            action = policy.choose(observation, legal) if player == 1 else legal[0]
            remaining -= int(action)
            if remaining == 0:
                learned_wins += int(player == 1)
                baseline_wins += int(player == -1)
                break
            player *= -1
    return {"episodes": episodes, "learned_wins": learned_wins, "baseline_wins": baseline_wins}
