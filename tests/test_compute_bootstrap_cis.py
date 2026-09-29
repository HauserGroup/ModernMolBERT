"""Tests for scripts/paper/compute_bootstrap_cis.py.

All tests are self-contained (no large files, no network). They use small
in-memory fixtures.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from compute_bootstrap_cis import (
    FAMILIES_PATH,
    build_cis,
    cluster_bootstrap,
    comparison_row,
    emit_ci_forest_plot,
    emit_latex,
    load_task_families,
    paired_bootstrap,
    read_task_families,
    run_comparisons,
)

# ── fixtures ─────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(0)


def _matrix_3task() -> pd.DataFrame:
    """Tiny 3-task results matrix for fast deterministic tests."""
    return pd.DataFrame(
        {
            "group": ["TDC-ADME", "TDC-Tox", "MoleculeNet"],
            "MMB-base": [0.80, 0.70, 0.60],
            "SELFormer": [0.70, 0.65, 0.55],
            "ECFP4": [0.85, 0.68, 0.62],
        },
        index=["task_A", "task_B", "task_C"],
    )


def _matrix_with_missing() -> pd.DataFrame:
    """Matrix where MMB-base is missing one task (NaN)."""
    df = _matrix_3task().copy()
    df.loc["task_C", "MMB-base"] = float("nan")
    return df


# ── paired_bootstrap ─────────────────────────────────────────────────────────


def test_paired_bootstrap_mean_diff_correct():
    a = np.array([0.80, 0.70, 0.60])
    b = np.array([0.70, 0.65, 0.55])
    mean_diff, _, _ = paired_bootstrap(a, b, n_boot=1000, rng=np.random.default_rng(0))
    expected = float((a - b).mean())
    assert abs(mean_diff - expected) < 1e-12


def test_paired_bootstrap_ci_ordered():
    a = np.array([0.80, 0.70, 0.60])
    b = np.array([0.70, 0.65, 0.55])
    _, ci_low, ci_high = paired_bootstrap(a, b, n_boot=2000, rng=np.random.default_rng(1))
    assert ci_low <= ci_high


def test_paired_bootstrap_ci_contains_mean():
    """The 95 % CI should contain the observed mean difference."""
    a = np.array([0.80, 0.70, 0.60])
    b = np.array([0.70, 0.65, 0.55])
    mean_diff, ci_low, ci_high = paired_bootstrap(a, b, n_boot=5000, rng=np.random.default_rng(2))
    assert ci_low <= mean_diff <= ci_high


def test_paired_bootstrap_positive_ci_when_a_always_greater():
    """When a > b on every task, the CI lower bound should be positive."""
    rng = np.random.default_rng(3)
    a = np.array([0.9, 0.85, 0.8, 0.75, 0.7])
    b = a - 0.10  # constant gap of 0.10
    _, ci_low, _ = paired_bootstrap(a, b, n_boot=5000, rng=rng)
    assert ci_low > 0.0


def test_paired_bootstrap_ci_straddles_zero_when_mixed():
    """When a beats b on some tasks and loses on others, CI should straddle zero."""
    rng = np.random.default_rng(4)
    a = np.array([0.9, 0.5, 0.9, 0.5, 0.9, 0.5] * 4)
    b = np.array([0.5, 0.9, 0.5, 0.9, 0.5, 0.9] * 4)
    _, ci_low, ci_high = paired_bootstrap(a, b, n_boot=5000, rng=rng)
    assert ci_low < 0.0 < ci_high


def test_paired_bootstrap_reproducible_with_same_seed():
    a = np.array([0.8, 0.7, 0.6])
    b = np.array([0.7, 0.65, 0.55])
    _, lo1, hi1 = paired_bootstrap(a, b, n_boot=1000, rng=np.random.default_rng(99))
    _, lo2, hi2 = paired_bootstrap(a, b, n_boot=1000, rng=np.random.default_rng(99))
    assert lo1 == lo2
    assert hi1 == hi2


def test_paired_bootstrap_raises_on_length_mismatch():
    with pytest.raises(ValueError, match="equal length"):
        paired_bootstrap(np.array([0.8, 0.7]), np.array([0.7]), n_boot=100)


def test_paired_bootstrap_raises_on_empty():
    with pytest.raises(ValueError, match="empty"):
        paired_bootstrap(np.array([]), np.array([]), n_boot=100)


# ── comparison_row ────────────────────────────────────────────────────────────


def test_comparison_row_win_tie_loss():
    matrix = _matrix_3task()
    # MMB-base [0.80, 0.70, 0.60] vs SELFormer [0.70, 0.65, 0.55]
    # diffs: [+0.10, +0.05, +0.05] → 3 wins, 0 ties, 0 losses
    row = comparison_row(matrix, "MMB-base", "SELFormer", n_boot=500, rng=np.random.default_rng(0))
    assert row["wins"] == 3
    assert row["ties"] == 0
    assert row["losses"] == 0
    assert row["n_tasks"] == 3


def test_comparison_row_drops_missing_before_counting():
    matrix = _matrix_with_missing()
    # task_C has NaN for MMB-base → only 2 tasks matched
    row = comparison_row(matrix, "MMB-base", "SELFormer", n_boot=500, rng=np.random.default_rng(0))
    assert row["n_tasks"] == 2


def test_comparison_row_mean_delta_unit_is_x100():
    matrix = _matrix_3task()
    row = comparison_row(matrix, "MMB-base", "SELFormer", n_boot=500, rng=np.random.default_rng(0))
    # mean([0.10, 0.05, 0.05]) = 0.0667, ×100 ≈ 6.67
    assert abs(row["mean_delta_roc_auc"] - 6.67) < 0.01


def test_comparison_row_loss_when_b_greater():
    matrix = _matrix_3task()
    # MMB-base [0.80, 0.70, 0.60] vs ECFP4 [0.85, 0.68, 0.62]
    # diffs: [-0.05, +0.02, -0.02] → 1 win, 0 ties, 2 losses
    row = comparison_row(matrix, "MMB-base", "ECFP4", n_boot=500, rng=np.random.default_rng(0))
    assert row["wins"] == 1
    assert row["losses"] == 2


# ── run_comparisons ───────────────────────────────────────────────────────────


def test_run_comparisons_returns_one_row_per_comparison():
    matrix = _matrix_3task()
    comps = [("MMB-base", "SELFormer"), ("MMB-base", "ECFP4")]
    df = run_comparisons(matrix, comps, n_boot=200, seed=0)
    assert len(df) == 2
    assert list(df["model_a"]) == ["MMB-base", "MMB-base"]
    assert list(df["model_b"]) == ["SELFormer", "ECFP4"]


def test_run_comparisons_consistent_with_seed():
    matrix = _matrix_3task()
    comps = [("MMB-base", "SELFormer")]
    df1 = run_comparisons(matrix, comps, n_boot=500, seed=7)
    df2 = run_comparisons(matrix, comps, n_boot=500, seed=7)
    assert df1["ci_low_95"].iloc[0] == df2["ci_low_95"].iloc[0]


# ── emit_latex ────────────────────────────────────────────────────────────────


def _make_ci_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_a": "MMB-base",
                "model_b": "SELFormer",
                "n_tasks": 24,
                "wins": 21,
                "ties": 0,
                "losses": 3,
                "mean_delta_roc_auc": 5.08,
                "ci_low_95": 3.5,
                "ci_high_95": 6.7,
            },
            {
                "model_a": "MMB-base",
                "model_b": "ECFP4",
                "n_tasks": 24,
                "wins": 7,
                "ties": 0,
                "losses": 17,
                "mean_delta_roc_auc": -1.72,
                "ci_low_95": -3.1,
                "ci_high_95": -0.3,
            },
        ]
    )


def test_emit_latex_creates_file(tmp_path: Path):
    out = tmp_path / "table.tex"
    emit_latex(_make_ci_df(), out)
    assert out.exists() and out.stat().st_size > 0


def test_emit_latex_contains_table_environment(tmp_path: Path):
    out = tmp_path / "table.tex"
    emit_latex(_make_ci_df(), out)
    content = out.read_text()
    assert r"\begin{table}" in content
    assert r"\end{table}" in content
    assert r"\label{tab:bootstrap-cis}" in content


def test_emit_latex_model_names_present(tmp_path: Path):
    out = tmp_path / "table.tex"
    emit_latex(_make_ci_df(), out)
    content = out.read_text()
    assert "SELFormer" in content
    assert "ECFP4" in content
    assert "MMB-base" in content


def test_emit_latex_ci_values_present(tmp_path: Path):
    out = tmp_path / "table.tex"
    emit_latex(_make_ci_df(), out)
    content = out.read_text()
    assert "5.08" in content
    assert "3.5" in content


# ── emit_ci_forest_plot ──────────────────────────────────────────────────────


def test_emit_ci_forest_plot_creates_pdf_and_png(tmp_path: Path):
    emit_ci_forest_plot(_make_ci_df(), tmp_path)
    pdf = tmp_path / "bootstrap_ci_forest.pdf"
    png = tmp_path / "bootstrap_ci_forest.png"
    assert pdf.exists() and pdf.stat().st_size > 0
    assert png.exists() and png.stat().st_size > 0


# ── integration: build_cis ────────────────────────────────────────────────────


def test_build_cis_end_to_end(tmp_path: Path):
    matrix = _matrix_3task()
    matrix_path = tmp_path / "matrix.csv"
    matrix.to_csv(matrix_path)
    out_dir = tmp_path / "out"
    comps = [("MMB-base", "SELFormer"), ("MMB-base", "ECFP4")]
    df = build_cis(
        matrix_path,
        out_dir,
        n_boot=200,
        seed=0,
        comparisons=comps,
        figure_dir=out_dir / "figures",
    )
    assert (out_dir / "bootstrap_cis.csv").exists()
    assert (out_dir / "table_bootstrap.tex").exists()
    assert (out_dir / "figures" / "bootstrap_ci_forest.pdf").exists()
    assert (out_dir / "figures" / "bootstrap_ci_forest.png").exists()
    assert len(df) == 2
    # SELFormer comparison: all tasks won, CI should be positive
    sel_row = df[df["model_b"] == "SELFormer"].iloc[0]
    assert sel_row["wins"] == 3
    assert sel_row["ci_low_95"] > 0


# ── task families ────────────────────────────────────────────────────────────


def test_cluster_bootstrap_with_singletons_matches_task_bootstrap():
    diffs = np.array([0.10, -0.02, 0.05, 0.03, -0.01, 0.07])
    groups = np.array([f"t{i}" for i in range(len(diffs))])
    _, lo_task, hi_task = paired_bootstrap(
        diffs, np.zeros_like(diffs), n_boot=4000, rng=np.random.default_rng(5)
    )
    est, lo_fam, hi_fam = cluster_bootstrap(
        diffs, groups, n_boot=4000, rng=np.random.default_rng(5)
    )
    assert est == pytest.approx(diffs.mean())
    assert lo_fam == pytest.approx(lo_task, abs=1e-12)
    assert hi_fam == pytest.approx(hi_task, abs=1e-12)


def test_family_weighting_counts_each_family_once():
    diffs = np.array([1.0, 1.0, 1.0, 5.0])
    groups = np.array(["F", "F", "F", "single"])
    task, _, _ = cluster_bootstrap(diffs, groups, n_boot=100, rng=np.random.default_rng(0))
    family, _, _ = cluster_bootstrap(
        diffs, groups, n_boot=100, rng=np.random.default_rng(0), weight="family"
    )
    assert task == pytest.approx(2.0)
    assert family == pytest.approx(3.0)


def test_duplicated_evidence_widens_the_family_interval():
    """Five identical tasks look like five confirmations to the task bootstrap."""
    diffs = np.array([3.0] * 5 + [-1.0, 0.5, -0.5, 1.0, 0.0])
    groups = np.array(["F"] * 5 + [f"s{i}" for i in range(5)])
    _, lo_task, hi_task = paired_bootstrap(
        diffs, np.zeros_like(diffs), n_boot=5000, rng=np.random.default_rng(1)
    )
    _, lo_fam, hi_fam = cluster_bootstrap(diffs, groups, n_boot=5000, rng=np.random.default_rng(1))
    assert hi_fam - lo_fam > hi_task - lo_task


def test_cluster_bootstrap_rejects_bad_input():
    with pytest.raises(ValueError, match="equal length"):
        cluster_bootstrap(np.array([0.1, 0.2]), np.array(["a"]))
    with pytest.raises(ValueError, match="weight"):
        cluster_bootstrap(np.array([0.1]), np.array(["a"]), weight="dataset")


def test_unlisted_tasks_are_their_own_family(tmp_path: Path):
    path = tmp_path / "families.yaml"
    path.write_text("families:\n  pair: [task_A, task_B]\n")
    assert load_task_families(path, ["task_A", "task_B", "task_C"]) == {
        "task_A": "pair",
        "task_B": "pair",
        "task_C": "task_C",
    }
    path.write_text("families:\n  one: [task_A]\n  two: [task_A]\n")
    with pytest.raises(ValueError, match="two families"):
        read_task_families(path)


def test_shipped_families_name_configured_datasets():
    config = FAMILIES_PATH.with_name("datasets.yaml")
    names = {entry["name"] for entry in yaml.safe_load(config.read_text())["datasets"].values()}
    families = read_task_families(FAMILIES_PATH)
    assert set(families) <= names
    assert len(set(families.values())) == 3


def test_comparison_row_reports_family_summary():
    matrix = _matrix_3task()
    families = {"task_A": "F", "task_B": "F", "task_C": "task_C"}
    # MMB-base − ECFP4 diffs: [-0.05, +0.02, -0.02]; family means: F -0.015, task_C -0.02
    row = comparison_row(
        matrix, "MMB-base", "ECFP4", n_boot=500, rng=np.random.default_rng(0), families=families
    )
    assert row["n_families"] == 2
    assert (row["family_wins"], row["family_losses"]) == (0, 2)
    assert row["family_weighted_delta"] == pytest.approx(-1.75)
    assert row["mean_delta_roc_auc"] == pytest.approx(-1.67)


def test_family_rng_leaves_task_intervals_unchanged():
    matrix = _matrix_3task()
    comps = [("MMB-base", "SELFormer"), ("MMB-base", "ECFP4")]
    families = {"task_A": "F", "task_B": "F", "task_C": "task_C"}
    plain = run_comparisons(matrix, comps, n_boot=300, seed=3)
    grouped = run_comparisons(matrix, comps, n_boot=300, seed=3, families=families)
    pd.testing.assert_frame_equal(grouped[plain.columns], plain)


def test_emit_latex_adds_family_columns(tmp_path: Path):
    df = _make_ci_df().assign(
        n_families=18,
        family_wins=[16, 6],
        family_ties=0,
        family_losses=[2, 12],
        family_ci_low_95=[3.4, -3.3],
        family_ci_high_95=[6.8, -0.2],
        family_weighted_delta=[5.2, -1.9],
    )
    out = tmp_path / "table.tex"
    emit_latex(df, out, caption_note="Archived scores.")
    content = out.read_text()
    assert "Task families" in content
    assert r"16/0/2 & $[+3.4,\;+6.8]$ & +5.20" in content
    assert r"\model{}-base minus each baseline" in content
    assert "Archived scores." in content


def test_build_cis_accepts_named_reference(tmp_path: Path):
    matrix = _matrix_3task().rename(columns={"MMB-base": "MMB-small (clean)"})
    matrix_path = tmp_path / "matrix.csv"
    matrix.to_csv(matrix_path)
    df = build_cis(
        matrix_path,
        tmp_path / "out",
        n_boot=100,
        seed=0,
        comparisons=[("MMB-small (clean)", "ECFP4")],
        make_figures=False,
        families_path=None,
    )
    assert df["model_a"].tolist() == ["MMB-small (clean)"]
