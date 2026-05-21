from pathlib import Path

import pandas as pd

from modernmolbert.eval.datasets import EvalDataset
from modernmolbert.eval.featurizers.dummy import DummyFeaturizer
from modernmolbert.eval.runner import FrozenBenchmarkRunner


def test_runner_writes_results(tmp_path: Path) -> None:
    ds = EvalDataset(
        name="toy",
        task_type="classification",
        task_names=["label"],
        train=pd.DataFrame({"smiles": ["CCO", "CCN", "CCC", "CCCC"], "label": [0, 0, 1, 1]}),
        valid=None,
        test=pd.DataFrame({"smiles": ["CO", "CCBr"], "label": [0, 1]}),
    )

    runner = FrozenBenchmarkRunner(cache_dir=tmp_path / "cache")

    result = runner.run(
        dataset=ds,
        featurizer=DummyFeaturizer(name="dummy_8", n_features=8),
        output_dir=tmp_path / "out",
    )

    assert len(result.task_results) == 1
    assert (tmp_path / "out" / "results.csv").exists()
    assert (tmp_path / "out" / "results.json").exists()
