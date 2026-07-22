import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agents.observation import adapt  # noqa: E402
from agents.rule_policy import COUNT_MODE, RulePolicy  # noqa: E402
from eval.battle_vs import promotion_decision, wilson_ci  # noqa: E402
from tests.support import observation, select, synthetic_card_index  # noqa: E402

DECK = list(range(1, 61))
SPEC = importlib.util.spec_from_file_location("sol_semantic_main", REPO / "main.py")
assert SPEC is not None and SPEC.loader is not None
submission = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(submission)


class StubAgent:
    def __init__(self, action=(0,), raises=False):
        self.action = list(action)
        self.raises = raises
        self.calls = 0
        self.fallback_count = 0
        self.decision_count = 0
        self.config = SimpleNamespace(time_budget_s=None)
        self.budget_violations = 0
        self.planner_fallbacks = 0
        self.degraded_count = 0

    def act(self, _obs):
        self.calls += 1
        if self.raises:
            raise RuntimeError("stub failure")
        return list(self.action)


def make_agent():
    return submission.SubmissionAgent(seed=1, deck=DECK, card_index=synthetic_card_index())


def test_all_engine_contexts_have_explicit_rule_semantics() -> None:
    assert set(COUNT_MODE) == set(range(49))


def test_rule_policy_interprets_action_type_not_serialized_hash() -> None:
    view = adapt(
        observation(
            select(
                [{"type": 14}, {"type": 10}],
                context=0,
                min_count=1,
                max_count=1,
            )
        )
    )
    assert RulePolicy(synthetic_card_index()).choose(view) == [1]


def test_time_governor_hands_search_to_greedy_before_600_seconds() -> None:
    agent = make_agent()
    search = StubAgent()
    agent._mcts = search
    agent.think_time_s = 510.0
    action = agent.act(observation(select([{"type": 14}], min_count=1, max_count=1)))
    assert search.calls == 0
    assert agent.greedy_handoffs == 1
    assert action == [0]


def test_layered_fallback_remains_legal() -> None:
    agent = make_agent()
    agent._mcts = StubAgent(raises=True)
    agent._greedy = StubAgent(raises=True)
    agent._rule = StubAgent(raises=True)
    obs = observation(select([{"type": 0}, {"type": 0}], min_count=1, max_count=2))
    action = agent.act(obs)
    assert 1 <= len(action) <= 2
    assert len(action) == len(set(action))
    assert all(0 <= index < 2 for index in action)


def test_promotion_gate_requires_20_seeds_60_percent_and_strict_ci() -> None:
    lo, hi = wilson_ci(30, 40)
    report = {
        "seeds": 20,
        "winrate_semantic_excl_draws": 0.75,
        "wilson95_excl_draws": [lo, hi],
        "faults_semantic": 0,
        "unfinished": 0,
        "max_think_s": {"semantic": 100.0},
    }
    assert promotion_decision(report) == {"promote": True, "reasons": []}
    report["seeds"] = 19
    assert not promotion_decision(report)["promote"]
