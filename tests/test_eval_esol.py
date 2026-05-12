from pathlib import Path
import pandas as pd
from modernmolbert.eval.contributed_datasets import load_esol

def test_load_esol(tmp_path: Path) -> None:
    root = tmp_path / "esol"
    root.mkdir()

    pd.DataFrame({"smiles": ["CCO", "CCN", "CCC"], "target": [1.2, -0.5, None]}).to_csv(root / "train.csv", index=False)
    pd.DataFrame({"smiles": ["CCCl", "CCBr"], "target": [0.1, 0.2]}).to_csv(root / "test.csv", index=False)

    dataset = load_esol(root=root)
    dataset.check()
    
    assert dataset.name == "esol"
    assert len(dataset.train) == 2
