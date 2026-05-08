import json
from pathlib import Path

import pandas as pd


def test_run_modernmolbert_eval_wrapper_uses_shared_runner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    metadata = {
        "name": "tiny_wrapped",
        "task_type": "classification",
        "tasks": ["label"],
    }

    (dataset_dir / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    train = pd.DataFrame(
        {
            "smiles_canonical": ["CCO", "CCN", "CCC", "CCCl"],
            "selfies": ["[C][C][O]", "[C][C][N]", "[C][C][C]", "[C][C][Cl]"],
            "label": [0, 0, 1, 1],
        }
    )
    valid = pd.DataFrame(
        {
            "smiles_canonical": ["CO", "CN"],
            "selfies": ["[C][O]", "[C][N]"],
            "label": [0, 1],
        }
    )
    test = pd.DataFrame(
        {
            "smiles_canonical": ["CCO", "CCC"],
            "selfies": ["[C][C][O]", "[C][C][C]"],
            "label": [0, 1],
        }
    )

    train.to_parquet(dataset_dir / "train.parquet")
    valid.to_parquet(dataset_dir / "valid.parquet")
    test.to_parquet(dataset_dir / "test.parquet")

    # This test is easiest if run_modernmolbert_eval accepts a featurizer_config
    # override. If not, prefer testing run_frozen_benchmark.py and keep this as
    # a manual smoke command.
