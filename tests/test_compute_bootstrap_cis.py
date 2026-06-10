"""Tests for scripts/paper/compute_bootstrap_cis.py.

All tests are self-contained (no large files, no network). They use small
in-memory fixtures.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compute_bootstrap_cis import (
    build_cis,
    comparison_row,
    emit_ci_forest_plot,
    emit_latex,
    paired_bootstrap,
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
