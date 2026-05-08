import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_run_frozen_benchmark_with_modernmolbert_config_smoke(tmp_path: Path) -> None:
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    config_json = tmp_path / "modernmolbert_selfies.json"
    output_dir = tmp_path / "results"
    cache_dir = tmp_path / "cache"

    pd.DataFrame(
        {
            "smiles": ["CCO", "CCN", "CCC", "CCCl", "CCBr", "CO"],
            "label": [0, 0, 1, 1, 1, 0],
        }
    ).to_csv(train_csv, index=False)

    pd.DataFrame(
        {
            "smiles": ["CCO", "CCC", "CCBr", "CO"],
            "label": [0, 1, 1, 0],
        }
    ).to_csv(test_csv, index=False)

    config_json.write_text(
        json.dumps(
            {
                "type": "dummy",
                "name": "modernmolbert_selfies_smoke",
                "n_features": 8,
            }
        ),
        encoding="utf-8",
    )

    cmd = [
        sys.executable,
        "-m",
        "modernmolbert.eval.cli.run_frozen_benchmark",
        "--name",
        "modernmolbert_smoke",
        "--task_type",
        "classification",
        "--task_names",
        "label",
        "--train_csv",
        str(train_csv),
        "--test_csv",
        str(test_csv),
        "--featurizer_config",
        str(config_json),
        "--output_dir",
        str(output_dir),
        "--cache_dir",
        str(cache_dir),
        "--batch_size",
        "2",
    ]

    subprocess.run(cmd, check=True)

    assert (output_dir / "results.json").exists()
    assert (output_dir / "results.csv").exists()

    results = pd.read_csv(output_dir / "results.csv")
    assert len(results) == 1
    assert results.loc[0, "dataset"] == "modernmolbert_smoke"
    assert results.loc[0, "task"] == "label"
