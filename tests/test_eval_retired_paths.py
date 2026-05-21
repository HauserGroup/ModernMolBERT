from pathlib import Path

_EVAL = Path("src/modernmolbert/eval")


def test_retired_eval_clis_are_absent() -> None:
    assert not (_EVAL / "cli/run_frozen_benchmark.py").exists()
    assert not (_EVAL / "cli/run_modernmolbert_eval.py").exists()


def test_deprecated_eval_infrastructure_not_at_original_paths() -> None:
    """Guard against accidentally restoring deprecated modules to their old locations."""
    retired = [
        "cache.py",
        "cli/__init__.py",
        "contributed_datasets.py",
        "dabest_plots.py",
        "dataset_registry.py",
        "datasets.py",
        "downstream.py",
        "featurizers/dummy.py",
        "featurizers/hf_smiles.py",
        "featurizers/rdkit_ecfp.py",
        "io.py",
        "metrics.py",
        "moleculenet.py",
        "registry.py",
        "reporting.py",
        "runner.py",
        "suite.py",
        "task_eval.py",
    ]
    for rel in retired:
        assert not (_EVAL / rel).exists(), f"Deprecated module restored at original path: {rel}"
