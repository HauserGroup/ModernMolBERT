from pathlib import Path

import pandas as pd

from modernmolbert.eval.contributed_datasets import load_tdc_caco2_wang


def test_load_tdc_caco2_wang(tmp_path: Path) -> None:
    root = tmp_path / "tdc_caco2_wang"
    root.mkdir()

    pd.DataFrame(
        {
            "Drug": ["CCO", "CCN", "CCC"],
            "Y": [-5.1, -4.7, None],
        }
    ).to_csv(root / "train.csv", index=False)

    pd.DataFrame(
        {
            "Drug": ["CCCl", "CCBr"],
            "Y": [-5.0, -5.8],
        }
    ).to_csv(root / "test.csv", index=False)

    dataset = load_tdc_caco2_wang(root=root)

    dataset.check()
    assert dataset.name == "tdc_caco2_wang"
    assert len(dataset.train) == 2
