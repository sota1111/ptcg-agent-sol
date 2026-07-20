"""Competition protocol adapter with strict legal-action normalization."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CompetitionRequest:
    request_id: object | None
    observation: dict[str, Any]
    legal_actions: tuple[Any, ...]


def adapt_request(payload: dict[str, Any]) -> CompetitionRequest:
    """Normalize supported competition request shapes without inventing legal moves."""
    observation = payload.get("observation", {})
    if not isinstance(observation, dict):
        raise ValueError("observation must be a JSON object")

    explicit = payload.get("legal_actions")
    if explicit is not None:
        if not isinstance(explicit, list) or not explicit:
            raise ValueError("legal_actions must be a non-empty list")
        legal = tuple(explicit)
    else:
        actions = payload.get("actions", observation.get("actions"))
        mask = payload.get("action_mask", observation.get("action_mask"))
        if not isinstance(actions, list) or not isinstance(mask, list):
            raise ValueError("request requires legal_actions or actions with action_mask")
        if len(actions) != len(mask) or not actions:
            raise ValueError("actions and action_mask must have equal non-zero length")
        if any(not isinstance(value, (bool, int)) for value in mask):
            raise ValueError("action_mask entries must be booleans or integers")
        legal = tuple(
            action for action, allowed in zip(actions, mask, strict=True) if bool(allowed)
        )
        if not legal:
            raise ValueError("action_mask contains no legal action")

    return CompetitionRequest(payload.get("request_id"), observation, legal)
