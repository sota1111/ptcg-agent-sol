import json
import random
from pathlib import Path

import pytest

from ptcg_agent.adapter import adapt_request
from ptcg_agent.agent import choose_action
from ptcg_agent.cli import main
from ptcg_agent.config import load_config
from ptcg_agent.policy import LegalPolicy
from ptcg_agent.training import evaluate, load_checkpoint, train


@pytest.mark.parametrize("seed", range(100))
def test_policy_never_emits_an_illegal_action(seed: int) -> None:
    rng = random.Random(seed)
    actions = list(range(rng.randint(1, 20)))
    legal = rng.sample(actions, rng.randint(1, len(actions)))
    chosen = LegalPolicy().choose({"seed": seed}, legal, rng=rng, epsilon=1.0)
    assert chosen in legal


def test_action_mask_adapter_and_protocol() -> None:
    request = {
        "request_id": "masked",
        "observation": {"turn": 3, "actions": ["draw", "pass", "attack"], "action_mask": [0, 1, 1]},
    }
    adapted = adapt_request(request)
    assert adapted.legal_actions == ("pass", "attack")
    assert choose_action(request) == {"request_id": "masked", "action": "pass"}


def test_empty_mask_is_rejected() -> None:
    with pytest.raises(ValueError, match="no legal action"):
        adapt_request({"actions": ["pass"], "action_mask": [0]})


def test_fixed_seed_training_and_evaluation_are_reproducible(tmp_path: Path) -> None:
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    policy_a, result_a = train(seed=42, episodes=250, checkpoint=path_a, max_seconds=10)
    policy_b, result_b = train(seed=42, episodes=250, checkpoint=path_b, max_seconds=10)
    assert result_a == result_b
    assert policy_a.values == policy_b.values
    assert evaluate(policy_a, 42, 100) == evaluate(policy_b, 42, 100)
    assert path_a.read_text() == path_b.read_text()


def test_checkpoint_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    resumed_path = tmp_path / "resumed.json"
    full_path = tmp_path / "full.json"
    train(seed=7, episodes=100, checkpoint=resumed_path, max_seconds=10)
    resumed_policy, resumed_result = train(
        seed=7, episodes=300, checkpoint=resumed_path, max_seconds=10, resume=True
    )
    full_policy, full_result = train(seed=7, episodes=300, checkpoint=full_path, max_seconds=10)
    assert resumed_result == full_result
    assert resumed_policy.values == full_policy.values


def test_checkpoint_is_versioned_and_loadable(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.json"
    policy, result = train(seed=9, episodes=5, checkpoint=checkpoint, max_seconds=10)
    payload = json.loads(checkpoint.read_text())
    loaded_policy, loaded_result = load_checkpoint(checkpoint, 9)
    assert payload["version"] == 1
    assert loaded_result == result
    assert loaded_policy.values == policy.values


def test_learned_policy_regression_against_baseline(tmp_path: Path) -> None:
    policy, _ = train(
        seed=1802, episodes=5_000, checkpoint=tmp_path / "policy.json", max_seconds=10
    )
    result = evaluate(policy, seed=1802, episodes=500)
    assert result["learned_wins"] >= result["baseline_wins"]


def test_trained_checkpoint_drives_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import io
    import sys

    checkpoint = tmp_path / "policy.json"
    train(seed=1799, episodes=5_000, checkpoint=checkpoint, max_seconds=10)
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            '{"request_id":"learned","observation":{"remaining":3,"player":1},"legal_actions":[1,2,3]}\n'
        ),
    )
    assert main(["run", "--checkpoint", str(checkpoint)]) == 0
    response = json.loads(capsys.readouterr().out)
    assert response["action"] in [1, 2, 3]


def test_gpu_config_uses_same_training_backend(tmp_path: Path) -> None:
    config = load_config(Path("configs/gpu-3080ti.toml"))
    policy, result = train(
        seed=config.seed, episodes=10, checkpoint=tmp_path / "gpu-policy.json", max_seconds=10
    )
    assert result.episodes == 10
    assert policy.values
