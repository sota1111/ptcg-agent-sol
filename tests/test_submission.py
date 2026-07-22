from __future__ import annotations

import importlib.util
import subprocess
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def load_submission():
    spec = importlib.util.spec_from_file_location("sol_main", REPO / "main.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_initial_call_returns_exactly_60_cards() -> None:
    submission = load_submission()
    assert submission.agent({"select": None}) == submission.read_deck_csv()
    assert len(submission.read_deck_csv()) == 60


def test_kaggle_exec_without_file_loads_deck(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO)
    namespace: dict[str, object] = {}
    exec(compile((REPO / "main.py").read_text(), "main.py", "exec"), namespace)
    agent = namespace["agent"]
    assert callable(agent)
    assert len(agent({"select": None})) == 60


@pytest.mark.parametrize("minimum,maximum", [(0, 0), (1, 1), (2, 3)])
def test_decision_satisfies_selection_contract(minimum: int, maximum: int) -> None:
    action = load_submission().agent(
        {
            "current": {"player": 0},
            "select": {
                "option": [{"id": i} for i in range(4)],
                "minCount": minimum,
                "maxCount": maximum,
            },
        }
    )
    assert minimum <= len(action) <= maximum
    assert len(action) == len(set(action))
    assert all(0 <= index < 4 for index in action)


def test_invalid_bounds_fail_closed() -> None:
    with pytest.raises(ValueError, match="invalid selection bounds"):
        load_submission().agent({"select": {"option": [], "minCount": 1, "maxCount": 1}})


def test_submission_archive_layout_when_engine_is_installed() -> None:
    if not (REPO / "cg" / "libcg.so").is_file():
        pytest.skip("competition runtime not installed")
    subprocess.run(["bash", "scripts/build_submission.sh"], cwd=REPO, check=True)
    archive = REPO / "submission.tar.gz"
    with tarfile.open(archive, "r:gz") as bundle:
        names = bundle.getnames()
    assert "main.py" in names
    assert "deck.csv" in names
    assert "agents/planner.py" in names
    assert "cg/libcg.so" in names
    assert not any(name.startswith(("tests/", "eval/", ".git/")) for name in names)
