from pathlib import Path
import pandas as pd
from modernmolbert.eval.contributed_datasets import load_clintox

def test_load_clintox(tmp_path: Path) -> None:
    root = tmp_path / "clintox"
    root.mkdir()

    pd.DataFrame({"smiles": ["CCO", "CCN", "CCC"], "FDA_APPROVED": [0, 1, None]}).to_csv(root / "train.csv", index=False)
    pd.DataFrame({"smiles": ["CCCl", "CCBr"], "FDA_APPROVED": [0, 1]}).to_csv(root / "test.csv", index=False)

    dataset = load_clintox(root=root)
    dataset.check()
    
    assert dataset.name == "clintox"
    assert len(dataset.train) == 2
