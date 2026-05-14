import os
import sys
from importlib import import_module
from pathlib import Path

import joblib
import pytest


RUN_DOWNLOADS_ENV = "MODERNMOLBERT_RUN_BENCHMARK_DOWNLOADS"


def require_download_dependency(module_name: str, package_hint: str | None = None) -> None:
    try:
        import_module(module_name)
    except ImportError:
        package = package_hint or module_name
        pytest.fail(
            f"Missing dependency '{module_name}' for real benchmark downloads. "
            f"Install it with `uv sync --group eval-prep` or add `{package}`.",
            pytrace=False,
        )


def write_download_config(config_dir: Path) -> None:
    (config_dir / "embedding").mkdir(parents=True)
    (config_dir / "embedding" / "default.yaml").write_text(
        "\n".join(
            [
                "raw_directory: data/raw",
                "embedded_directory: data/embedded",
                "data_directory: data/downloaded",
                "prepared_directory: data/prepared",
                "predictions_directory: data/predictions",
                "clock_directory: data/clock",
                "svd_directory: data/svd",
                "max_invalid_embeddings: 50",
            ]
        )
    )
    (config_dir / "downloader.yaml").write_text(
        "\n".join(
            [
                "cache: true",
                "datasets:",
                "  - tdc_ames",
                "  - moleculenet_bbbp",
            ]
        )
    )
    (config_dir / "datasets.yaml").write_text(
        "\n".join(
            [
                "datasets:",
                "  tdc_ames:",
                "    name: AMES",
                "    metric: roc_auc",
                "    task: classification",
                "    source:",
                "      name: TDC",
                "      benchmark: admet",
                "      group: TOX",
                "  moleculenet_bbbp:",
                "    name: ogbg-molbbbp",
                "    metric: roc_auc",
                "    task: classification",
                "    source:",
                "      name: OGB",
            ]
        )
    )


@pytest.mark.smoke
@pytest.mark.skipif(
    os.getenv(RUN_DOWNLOADS_ENV) != "1",
    reason=f"Set {RUN_DOWNLOADS_ENV}=1 to download real benchmark datasets.",
)
def test_download_real_tdc_and_moleculenet_datasets(monkeypatch, tmp_path) -> None:
    require_download_dependency("tdc", "pytdc")
    require_download_dependency("ogb")
    require_download_dependency("rdkit")

    from modernmolbert.eval.benchmarking_molecular_models import download

    config_dir = tmp_path / "config"
    write_download_config(config_dir)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download.py",
            "--config-dir",
            str(config_dir),
            "--datasets",
            "tdc_ames",
            "moleculenet_bbbp",
        ],
    )

    download.main()

    prepared_dir = tmp_path / "data/prepared"
    expected = [
        prepared_dir / "AMES.joblib",
        prepared_dir / "ogbg-molbbbp.joblib",
    ]
    for path in expected:
        assert path.exists()
        dataset = joblib.load(path)
        assert len(dataset.data) > 0
        assert {"train", "valid", "test"} <= set(dataset.splits)
