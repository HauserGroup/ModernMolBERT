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
    cluster_bootstrap,
    comparison_row,
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


# ── run_comparisons ───────────────────────────────────────────────────────────


def test_comparison_intervals_do_not_depend_on_list_order():
    comparisons = [("MMB-base", "SELFormer"), ("MMB-base", "ECFP4")]
    forward = run_comparisons(_matrix_3task(), comparisons, n_boot=200, seed=42)
    backward = run_comparisons(_matrix_3task(), comparisons[::-1], n_boot=200, seed=42)
    pd.testing.assert_frame_equal(
        forward.sort_values("model_b").reset_index(drop=True),
        backward.sort_values("model_b").reset_index(drop=True),
    )


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


def test_emit_latex_ci_values_present(tmp_path: Path):
    out = tmp_path / "table.tex"
    emit_latex(_make_ci_df(), out)
    content = out.read_text()
    assert "5.08" in content
    assert "3.5" in content


# ── emit_ci_forest_plot ──────────────────────────────────────────────────────


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
