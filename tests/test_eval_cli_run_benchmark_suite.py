from pathlib import Path

from modernmolbert.eval.suite import load_suite_config


def test_pilot_core_suite_config_parses() -> None:
    suite = load_suite_config("configs/eval_suites/pilot_core.yaml")
    assert suite.name == "pilot_core"
    assert len(suite.datasets) == 3


def test_core_moleculenet_suite_config_parses() -> None:
    suite = load_suite_config("configs/eval_suites/core_moleculenet.yaml")
    assert suite.name == "core_moleculenet"
    assert len(suite.datasets) == 8


def test_suite_config_files_exist() -> None:
    assert Path("configs/eval_suites/pilot_core.yaml").exists()
    assert Path("configs/eval_suites/core_moleculenet.yaml").exists()
    assert Path("configs/eval_suites/README.md").exists()
