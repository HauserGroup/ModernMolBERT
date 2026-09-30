import pytest

from modernmolbert.visualize.regen_groupfig import (
    GROUP_ORDER,
    MODELS,
    default_csv_path,
    load_group_distribution_data,
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


def test_validate_rejects_incomplete_coverage() -> None:
    df = load_group_distribution_data().iloc[:-1].copy()

    with pytest.raises(ValueError, match="Unexpected task coverage"):
        validate_group_distribution_data(df)
