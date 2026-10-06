"""Internal uncertainty reports task/family intervals separately from seed variation."""

import pandas as pd
import pytest

from aggregate_revision_seeds import MODELS, SEEDS, aggregate
from compute_revision_contrast_intervals import compute_intervals, plot_intervals


def test_constant_seed_contrasts_have_distinct_declared_units(tmp_path):
    tasks = [f"task_{i:02d}" for i in range(25)]
    values = [0.60, 0.55, 0.50, 0.45, 0.65]
    matrices = {
        seed: pd.DataFrame(
            {model: value for model, value in zip(MODELS, values, strict=True)},
            index=tasks,
        )
        for seed in SEEDS
    }
    aggregate_outputs = aggregate(matrices)
    families = {task: task for task in tasks}
    for indices, name in ((range(5), "CYP"), (range(5, 8), "substrate"), (range(8, 10), "hERG")):
        for index in indices:
            families[tasks[index]] = name
    frame = compute_intervals(
        aggregate_outputs["mean_common_task_matrix.csv"],
        aggregate_outputs["overall_contrast_summary.csv"],
        families,
        n_boot=100,
        seed=42,
    )
    assert set(frame["n_families"]) == {18}
    assert set(frame["n_seeds"]) == {5}
    row = frame.set_index("contrast").loc["APE_minus_BPE_SELFIES"]
    assert row["mean_delta_roc_auc"] == pytest.approx(0.10)
    assert row["task_ci_low"] == pytest.approx(0.10)
    assert row["family_ci_high"] == pytest.approx(0.10)
    assert row["sd_across_seed_task_means"] == pytest.approx(0)
    figure = tmp_path / "intervals.pdf"
    plot_intervals(frame, figure)
    assert figure.is_file()
