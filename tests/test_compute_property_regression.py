"""Tests for scripts/compute_property_regression.py.

All tests use small in-memory fixtures — no large parquet files, no network.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from compute_property_regression import (
    build_regression,
    emit_latex,
    fit_ridge_r2,
    load_embeddings_and_descriptors,
    run_regression,
)

# ── fixtures ──────────────────────────────────────────────────────────────────

N = 200  # enough for a reliable Ridge split
DIM = 16  # tiny embedding dimension
SEED = 42

RNG = np.random.default_rng(SEED)


def _make_embeddings(n: int = N, dim: int = DIM) -> np.ndarray:
    return RNG.standard_normal((n, dim)).astype(np.float32)


def _make_train_parquet(tmp_path: Path, n: int = N) -> Path:
    """Write a minimal train.parquet with two descriptor cols and chembl_id."""
    ids = [f"CHEMBL{i}" for i in range(n)]
    rng2 = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "chembl_id": ids,
            "alogp": rng2.standard_normal(n),
            "heavy_atoms": rng2.integers(5, 50, size=n).astype(float),
        }
    )
    path = tmp_path / "train.parquet"
    df.to_parquet(path, index=False)
    return path


def _make_meta_parquet(tmp_path: Path, n: int = N) -> Path:
    ids = [f"CHEMBL{i}" for i in range(n)]
    df = pd.DataFrame(
        {
            "chembl_id": ids,
            "embedding_row": list(range(n)),
        }
    )
    path = tmp_path / "metadata.parquet"
    df.to_parquet(path, index=False)
    return path


def _make_embed_npy(tmp_path: Path, n: int = N, dim: int = DIM) -> Path:
    X = _make_embeddings(n, dim)
    path = tmp_path / "embeddings.npy"
    np.save(path, X)
    return path


# ── fit_ridge_r2 ──────────────────────────────────────────────────────────────


def test_fit_ridge_r2_returns_float():
    X = _make_embeddings()
    y = RNG.standard_normal(N)
    r2 = fit_ridge_r2(X, y, test_size=0.2, alpha=1.0, random_state=SEED)
    assert isinstance(r2, float)


def test_fit_ridge_r2_perfect_linear_signal():
    """When y is exactly a linear function of X, R² should be close to 1."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 8)).astype(np.float32)
    coef = rng.standard_normal(8)
    y = X @ coef  # perfect linear signal
    r2 = fit_ridge_r2(X, y, test_size=0.2, alpha=1e-6, random_state=0)
    assert r2 > 0.98, f"Expected R²>0.98 for perfect linear signal, got {r2:.4f}"


def test_fit_ridge_r2_pure_noise_near_zero():
    """When y is pure noise independent of X, R² should be close to zero."""
    rng = np.random.default_rng(5)
    X = rng.standard_normal((400, 16)).astype(np.float32)
    y = rng.standard_normal(400)
    r2 = fit_ridge_r2(X, y, test_size=0.2, alpha=1.0, random_state=5)
    assert abs(r2) < 0.15, f"Expected |R²|<0.15 for pure noise, got {r2:.4f}"


def test_fit_ridge_r2_reproducible():
    X = _make_embeddings()
    y = RNG.standard_normal(N)
    r1 = fit_ridge_r2(X, y, random_state=7)
    r2 = fit_ridge_r2(X, y, random_state=7)
    assert r1 == r2


# ── load_embeddings_and_descriptors ──────────────────────────────────────────


def test_load_returns_aligned_arrays(tmp_path: Path):
    embed_path = _make_embed_npy(tmp_path)
    meta_path = _make_meta_parquet(tmp_path)
    train_path = _make_train_parquet(tmp_path)
    X, desc = load_embeddings_and_descriptors(
        embed_path, meta_path, train_path, ["alogp", "heavy_atoms"]
    )
    assert X.shape[0] == len(desc)
    assert X.shape[1] == DIM


def test_load_drops_rows_with_missing_descriptors(tmp_path: Path):
    """Rows where a descriptor is NaN should be dropped."""
    embed_path = _make_embed_npy(tmp_path)
    # metadata for N molecules
    meta_df = pd.DataFrame(
        {
            "chembl_id": [f"CHEMBL{i}" for i in range(N)],
            "embedding_row": list(range(N)),
        }
    )
    meta_path = tmp_path / "metadata.parquet"
    meta_df.to_parquet(meta_path, index=False)

    # train parquet with first 10 rows having NaN in alogp
    rng = np.random.default_rng(2)
    train_df = pd.DataFrame(
        {
            "chembl_id": [f"CHEMBL{i}" for i in range(N)],
            "alogp": [float("nan")] * 10 + list(rng.standard_normal(N - 10)),
            "heavy_atoms": rng.integers(5, 50, N).astype(float),
        }
    )
    train_path = tmp_path / "train.parquet"
    train_df.to_parquet(train_path, index=False)

    X, desc = load_embeddings_and_descriptors(
        embed_path, meta_path, train_path, ["alogp", "heavy_atoms"]
    )
    assert len(desc) == N - 10
    assert not bool(desc["alogp"].isna().any())


def test_load_inner_join_excludes_unmatched(tmp_path: Path):
    """Molecules in meta but absent from train.parquet should be excluded."""
    embed_path = _make_embed_npy(tmp_path, n=N)
    # meta has N molecules
    meta_df = pd.DataFrame(
        {
            "chembl_id": [f"CHEMBL{i}" for i in range(N)],
            "embedding_row": list(range(N)),
        }
    )
    meta_path = tmp_path / "metadata.parquet"
    meta_df.to_parquet(meta_path, index=False)

    # train has only the first N//2 molecules
    rng = np.random.default_rng(3)
    half = N // 2
    train_df = pd.DataFrame(
        {
            "chembl_id": [f"CHEMBL{i}" for i in range(half)],
            "alogp": rng.standard_normal(half),
            "heavy_atoms": rng.integers(5, 50, half).astype(float),
        }
    )
    train_path = tmp_path / "train.parquet"
    train_df.to_parquet(train_path, index=False)

    X, desc = load_embeddings_and_descriptors(
        embed_path, meta_path, train_path, ["alogp", "heavy_atoms"]
    )
    assert len(desc) == half


# ── run_regression ────────────────────────────────────────────────────────────


def test_run_regression_returns_one_row_per_descriptor(tmp_path: Path):
    embed_path = _make_embed_npy(tmp_path)
    meta_path = _make_meta_parquet(tmp_path)
    train_path = _make_train_parquet(tmp_path)
    X, desc_df = load_embeddings_and_descriptors(
        embed_path, meta_path, train_path, ["alogp", "heavy_atoms"]
    )
    descs = [("alogp", "AlogP"), ("heavy_atoms", "Heavy atoms")]
    df = run_regression(X, desc_df, descs, test_size=0.2, alpha=1.0, random_state=SEED)
    assert len(df) == 2
    assert set(df["descriptor_col"]) == {"alogp", "heavy_atoms"}


def test_run_regression_sorted_descending(tmp_path: Path):
    embed_path = _make_embed_npy(tmp_path)
    meta_path = _make_meta_parquet(tmp_path)
    train_path = _make_train_parquet(tmp_path)
    X, desc_df = load_embeddings_and_descriptors(
        embed_path, meta_path, train_path, ["alogp", "heavy_atoms"]
    )
    descs = [("alogp", "AlogP"), ("heavy_atoms", "Heavy atoms")]
    df = run_regression(X, desc_df, descs, test_size=0.2, alpha=1.0, random_state=SEED)
    r2_vals = df["r2"].tolist()
    assert r2_vals == sorted(r2_vals, reverse=True)


def test_run_regression_skips_missing_column(tmp_path: Path):
    """A descriptor not present in desc_df should be silently skipped."""
    embed_path = _make_embed_npy(tmp_path)
    meta_path = _make_meta_parquet(tmp_path)
    train_path = _make_train_parquet(tmp_path)
    X, desc_df = load_embeddings_and_descriptors(
        embed_path, meta_path, train_path, ["alogp", "heavy_atoms"]
    )
    descs = [("alogp", "AlogP"), ("nonexistent_col", "Ghost descriptor")]
    df = run_regression(X, desc_df, descs, test_size=0.2, alpha=1.0, random_state=SEED)
    assert len(df) == 1
    assert df.iloc[0]["descriptor_col"] == "alogp"


# ── emit_latex ────────────────────────────────────────────────────────────────


def _r2_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"descriptor_col": "alogp", "label": "AlogP (lipophilicity)", "r2": 0.712},
            {"descriptor_col": "heavy_atoms", "label": "Heavy atom count", "r2": 0.503},
        ]
    )


def test_emit_latex_creates_file(tmp_path: Path):
    out = tmp_path / "table.tex"
    emit_latex(_r2_df(), out)
    assert out.exists() and out.stat().st_size > 0


def test_emit_latex_contains_table_environment(tmp_path: Path):
    out = tmp_path / "table.tex"
    emit_latex(_r2_df(), out)
    content = out.read_text()
    assert r"\begin{table}" in content
    assert r"\label{tab:property-r2}" in content


def test_emit_latex_r2_values_present(tmp_path: Path):
    out = tmp_path / "table.tex"
    emit_latex(_r2_df(), out)
    content = out.read_text()
    assert "0.712" in content
    assert "0.503" in content


def test_emit_latex_descriptor_labels_present(tmp_path: Path):
    out = tmp_path / "table.tex"
    emit_latex(_r2_df(), out)
    content = out.read_text()
    assert "AlogP" in content
    assert "Heavy atom count" in content


# ── integration: build_regression ────────────────────────────────────────────


def test_build_regression_end_to_end(tmp_path: Path):
    embed_path = _make_embed_npy(tmp_path)
    meta_path = _make_meta_parquet(tmp_path)
    train_path = _make_train_parquet(tmp_path)
    out_dir = tmp_path / "out"
    descs = [("alogp", "AlogP"), ("heavy_atoms", "Heavy atoms")]
    df = build_regression(
        embed_path=embed_path,
        meta_path=meta_path,
        train_parquet=train_path,
        out_dir=out_dir,
        descriptors=descs,
        test_size=0.2,
        alpha=1.0,
        random_state=SEED,
    )
    assert (out_dir / "embedding_property_r2.csv").exists()
    assert (out_dir / "table_property_r2.tex").exists()
    assert len(df) == 2
    assert "r2" in df.columns
