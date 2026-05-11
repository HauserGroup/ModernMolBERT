from pathlib import Path


def test_retired_eval_clis_are_absent() -> None:
    assert not Path("src/modernmolbert/eval/cli/run_frozen_benchmark.py").exists()
    assert not Path("src/modernmolbert/eval/cli/run_modernmolbert_eval.py").exists()
