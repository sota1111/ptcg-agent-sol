import json
from pathlib import Path

from ptcg_agent.runtime_dataset import load_runtime_dataset


def test_runtime_dataset_smoke_load(tmp_path: Path) -> None:
    path = tmp_path / "samples.jsonl"
    path.write_text(
        json.dumps(
            {
                "schemaVersion": "1.0.0",
                "split": "train",
                "features": [1, 7, 50, 6, 7, 50, 6],
                "legalActions": [1, 2],
                "action": 1,
                "value": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    samples = list(load_runtime_dataset(path, "train"))
    assert samples[0].features == (1.0, 7.0, 50.0, 6.0, 7.0, 50.0, 6.0)
    assert samples[0].action == 1
