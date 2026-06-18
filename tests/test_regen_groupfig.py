from pathlib import Path

import pandas as pd
import pytest

from modernmolbert.visualize.regen_groupfig import (
    GROUP_ORDER,
    MODELS,
    default_csv_path,
    generate_group_distribution_figure,
    group_means,
    load_group_distribution_data,
    plot_group_distribution,
    validate_group_distribution_data,
)


def test_default_source_data_is_complete_and_corrected() -> None:
    assert default_csv_path().match("*/paper/source_data/Fig_task_group_distributions.csv")

    df = load_group_distribution_data()

    assert len(df) == 25 * len(MODELS)
    assert set(df["task_group"]) == set(GROUP_ORDER)
    assert set(df["model"]) == set(MODELS)

    corrected = df.set_index(["task_group", "task", "model"])["roc_auc_x100"]
    assert corrected.loc[("MoleculeNet", "MUV", "MMB-small")] == 74.0
    assert corrected.loc[("MoleculeNet", "MUV", "MMB-base")] == 72.1
    assert corrected.loc[("MoleculeNet", "Tox21", "MMB-base")] == 74.2


def test_group_means_uses_canonical_ordering() -> None:
    means = group_means(load_group_distribution_data())

    assert list(means.index) == GROUP_ORDER
    assert list(means.columns) == MODELS
    assert means.loc["MoleculeNet", "MMB-base"] == pytest.approx(73.7, abs=0.05)


def test_plot_group_distribution_creates_pdf(tmp_path: Path) -> None:
    out = tmp_path / "Fig_task_group_distributions.pdf"
    means = plot_group_distribution(load_group_distribution_data(), out)

    assert out.exists()
    assert out.stat().st_size > 0
    assert isinstance(means, pd.DataFrame)


def test_generate_group_distribution_figure_accepts_external_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "source.csv"
    out = tmp_path / "figure.pdf"
    load_group_distribution_data().to_csv(csv_path, index=False)

    means = generate_group_distribution_figure(
        csv_path=csv_path,
        output_path=out,
        verbose=False,
    )

    assert out.exists()
    assert means.loc["TDC-ADME", "MMB-small"] > 79


def test_validate_rejects_incomplete_coverage() -> None:
    df = load_group_distribution_data().iloc[:-1].copy()

    with pytest.raises(ValueError, match="Unexpected task coverage"):
        validate_group_distribution_data(df)
