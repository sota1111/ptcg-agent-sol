import io
import json
import sys
from pathlib import Path

import pytest

from ptcg_agent.cli import main
from ptcg_agent.config import load_config

CPU_CONFIG = Path("configs/cpu.toml")


def test_cpu_config_is_bounded() -> None:
    config = load_config(CPU_CONFIG)
    assert config.device == "cpu"
    assert config.max_hours == 8


def test_budget_above_eight_hours_is_rejected() -> None:
    with pytest.raises(ValueError, match="at most 8"):
        load_config(CPU_CONFIG, max_hours=8.01)


def test_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["smoke", "--config", str(CPU_CONFIG)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ok"


def test_jsonl_protocol(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"request_id":"1","legal_actions":["pass"]}\n'))
    assert main(["run", "--config", str(CPU_CONFIG)]) == 0
    assert json.loads(capsys.readouterr().out) == {"request_id": "1", "action": "pass"}


def test_gpu_preset_is_valid_without_requiring_gpu() -> None:
    config = load_config(Path("configs/gpu-3080ti.toml"))
    assert config.device == "cuda"
    assert config.mixed_precision is True
    assert config.max_hours == 8
