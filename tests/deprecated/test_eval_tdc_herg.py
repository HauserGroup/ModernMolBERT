from pathlib import Path
import pandas as pd
from modernmolbert.eval.contributed_datasets import load_tdc_herg_blockers


def test_load_tdc_herg_blockers(tmp_path: Path) -> None:
    root = tmp_path / "tdc_herg_blockers"
    root.mkdir()

    pd.DataFrame({"Drug": ["CCO", "CCN", "CCC"], "Y": [0, 1, None]}).to_csv(
        root / "train.csv", index=False
    )

    pd.DataFrame({"Drug": ["CCCl", "CCBr"], "Y": [0, 1]}).to_csv(root / "test.csv", index=False)

    dataset = load_tdc_herg_blockers(root=root)

    dataset.check()
    assert dataset.name == "tdc_herg_blockers"
    assert len(dataset.train) == 2
