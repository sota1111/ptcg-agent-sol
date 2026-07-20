"""Deterministic baseline agent and JSONL protocol."""

from typing import Any

from ptcg_agent.adapter import adapt_request
from ptcg_agent.policy import LegalPolicy


def choose_action(request: dict[str, Any], policy: LegalPolicy | None = None) -> dict[str, Any]:
    adapted = adapt_request(request)
    action = (policy or LegalPolicy()).choose(adapted.observation, adapted.legal_actions)
    response: dict[str, Any] = {"action": action}
    if adapted.request_id is not None:
        response["request_id"] = adapted.request_id
    return response
