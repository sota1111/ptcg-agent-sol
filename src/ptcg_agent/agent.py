"""Deterministic baseline agent and JSONL protocol."""

from typing import Any


def choose_action(request: dict[str, Any]) -> dict[str, Any]:
    actions = request.get("legal_actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError("legal_actions must be a non-empty list")
    response: dict[str, Any] = {"action": actions[0]}
    if "request_id" in request:
        response["request_id"] = request["request_id"]
    return response
