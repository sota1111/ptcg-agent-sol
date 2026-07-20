"""Legal-action baseline and deterministic tabular policy."""

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from typing import Any


def state_key(observation: Mapping[str, Any]) -> str:
    encoded = json.dumps(observation, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


def action_key(action: Any) -> str:
    return json.dumps(action, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


class LegalPolicy:
    def __init__(self, values: Mapping[str, Mapping[str, float]] | None = None) -> None:
        self.values = {state: dict(actions) for state, actions in (values or {}).items()}

    def choose(
        self,
        observation: Mapping[str, Any],
        legal_actions: Sequence[Any],
        *,
        rng: random.Random | None = None,
        epsilon: float = 0.0,
    ) -> Any:
        if not legal_actions:
            raise ValueError("legal_actions must not be empty")
        random_source = rng or random.Random(0)
        if epsilon > 0 and random_source.random() < epsilon:
            return legal_actions[random_source.randrange(len(legal_actions))]
        scores = self.values.get(state_key(observation), {})
        return max(
            legal_actions,
            key=lambda action: (scores.get(action_key(action), 0.0), -legal_actions.index(action)),
        )

    def update(
        self, observation: Mapping[str, Any], action: Any, reward: float, alpha: float
    ) -> None:
        actions = self.values.setdefault(state_key(observation), {})
        key = action_key(action)
        actions[key] = actions.get(key, 0.0) + alpha * (reward - actions.get(key, 0.0))
