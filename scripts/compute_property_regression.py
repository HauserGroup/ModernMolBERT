#!/usr/bin/env python3
"""
compute_property_regression.py
===============================

Quantify how well frozen ModernMolBERT-small (span masking) CLS embeddings
predict nine ChEMBL physicochemical descriptors via Ridge regression.

Overview
--------
The PaCMAP figure (Fig 5) shows qualitatively that the frozen embedding space
organises molecules along gradients in lipophilicity, drug-likeness, polarity,
etc.  This script makes that claim quantitative by fitting a Ridge regressor
from embeddings → descriptor and reporting test R².

Models are evaluated in the frozen-embedder regime: no fine-tuning, no gradient
updates.  The embeddings used here are from **ModernMolBERT-small (span masking)**
(``outputs/visualize/best_span_100k/``), which has the same architecture as the
released small model (34 M parameters, 512-dim hidden) but uses span masking
during pre-training.  The two models produce qualitatively identical embeddings
on this analysis; the footnote in the table caption states which checkpoint was
used.

Method
------
1. Load 100 000 frozen CLS embeddings (512-dim, float32).
2. Join metadata with ``data/pretrain/chembl36_selfies/train.parquet`` to
   obtain all nine descriptor columns.
3. Drop rows with missing descriptor values.
4. Random 80 / 20 train-test split (``random_state=42``).
5. Fit ``Ridge(alpha=1.0)`` per descriptor; evaluate on test split; record R².
6. Emit CSV and LaTeX table.

Inputs
------
- ``outputs/visualize/best_span_100k/embeddings.npy``    (100 000 × 512)
- ``outputs/visualize/best_span_100k/metadata.parquet``  (chembl_id, embedding_row, …)
- ``data/pretrain/chembl36_selfies/train.parquet``        (descriptor columns)

Outputs
-------
- ``outputs/eval/paper/embedding_property_r2.csv``
- ``outputs/eval/paper/table_property_r2.tex``

Usage
-----
    python scripts/compute_property_regression.py
    python scripts/compute_property_regression.py --test_size 0.3 --alpha 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]

EMBED_PATH = ROOT / "outputs/visualize/best_span_100k/embeddings.npy"
META_PATH = ROOT / "outputs/visualize/best_span_100k/metadata.parquet"
TRAIN_PARQUET = ROOT / "data/pretrain/chembl36_selfies/train.parquet"
OUT_DIR = ROOT / "outputs/eval/paper"

DESCRIPTORS: list[tuple[str, str]] = [
    ("alogp", "AlogP (lipophilicity)"),
    ("qed_weighted", "QED (drug-likeness)"),
    ("psa", "PSA (polar surface area)"),
    ("hbd", "H-bond donors"),
    ("hba", "H-bond acceptors"),
    ("mw_freebase", "Molecular weight"),
    ("rtb", "Rotatable bonds"),
    ("aromatic_rings", "Aromatic rings"),
    ("heavy_atoms", "Heavy atom count"),
]

# ── data loading ─────────────────────────────────────────────────────────────


def load_embeddings_and_descriptors(
    embed_path: Path,
    meta_path: Path,
    train_parquet: Path,
    descriptor_cols: list[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return (X, desc_df) where X[i] is the CLS embedding for row i of desc_df.

    Rows with any missing descriptor value are dropped; the returned arrays are
    aligned (same row order).
    """
    # Load embeddings (100k × 512)
    X_full = np.load(embed_path)  # float32, mmap OK

    # Load metadata — embedding_row is the direct index into X_full
    meta = pd.read_parquet(meta_path, columns=["chembl_id", "embedding_row"])

    # Join with training parquet to get all descriptor columns
    # desc_needed = [c for c in descriptor_cols if c != "alogp"]  unused for now
    train_cols = ["chembl_id"] + descriptor_cols
    train_df = pd.read_parquet(train_parquet, columns=train_cols)

    merged = meta.merge(train_df, on="chembl_id", how="inner")

    # Drop rows with any missing descriptor
    merged = merged.dropna(subset=descriptor_cols).reset_index(drop=True)

    X = X_full[merged["embedding_row"].to_numpy(dtype=np.int64)]
    desc_df = merged[descriptor_cols].copy()
    return X.astype(np.float32), desc_df


# ── regression ───────────────────────────────────────────────────────────────


def fit_ridge_r2(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    alpha: float = 1.0,
    random_state: int = 42,
) -> float:
    """Fit Ridge(alpha) with a train-test split and return test R²."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = Ridge(alpha=alpha)
    model.fit(X_tr, y_tr)
    return float(model.score(X_te, y_te))


def run_regression(
    X: np.ndarray,
    desc_df: pd.DataFrame,
    descriptors: list[tuple[str, str]],
    test_size: float = 0.2,
    alpha: float = 1.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run Ridge regression for each descriptor; return a summary DataFrame."""
    rows = []
    for col, label in descriptors:
        if col not in desc_df.columns:
            continue
        y = desc_df[col].to_numpy(dtype=np.float64)
        r2 = fit_ridge_r2(X, y, test_size=test_size, alpha=alpha, random_state=random_state)
        rows.append({"descriptor_col": col, "label": label, "r2": round(r2, 3)})
    return pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)


# ── LaTeX emission ────────────────────────────────────────────────────────────


def emit_latex(df: pd.DataFrame, out_path: Path) -> None:
    """Write a small tabular LaTeX fragment for the R² table."""
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering\small",
        r"  \begin{tabular}{l r}",
        r"    \toprule",
        r"    \textbf{Descriptor} & \textbf{Test $R^2$} \\",
        r"    \midrule",
    ]
    for _, row in df.iterrows():
        lines.append(f"    {row['label']} & {row['r2']:.3f} \\\\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{%",
        r"    Ridge regression test $R^2$ from frozen \model{}-small (span masking)",
        r"    CLS embeddings to nine ChEMBL physicochemical descriptors, evaluated on",
        r"    a held-out 20\,\% split of 100\,000 ChEMBL~36 molecules.",
        r"    $\alpha = 1.0$; random state 42.",
        r"    A higher $R^2$ indicates that the embedding space encodes the corresponding",
        r"    property more linearly.  The qualitative PaCMAP gradients of",
        r"    \cref{fig:embedding-space} correspond to the properties with the highest $R^2$.",
        r"  }%",
        r"  \label{tab:property-r2}",
        r"\end{table}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── main ──────────────────────────────────────────────────────────────────────


def build_regression(
    embed_path: Path = EMBED_PATH,
    meta_path: Path = META_PATH,
    train_parquet: Path = TRAIN_PARQUET,
    out_dir: Path = OUT_DIR,
    descriptors: list[tuple[str, str]] = DESCRIPTORS,
    test_size: float = 0.2,
    alpha: float = 1.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Full pipeline: load → regress → save CSV and LaTeX."""
    out_dir.mkdir(parents=True, exist_ok=True)

    descriptor_cols = [col for col, _ in descriptors]
    X, desc_df = load_embeddings_and_descriptors(
        embed_path, meta_path, train_parquet, descriptor_cols
    )
    print(f"Loaded {len(desc_df):,} molecules, embedding shape {X.shape}")

    df = run_regression(
        X, desc_df, descriptors, test_size=test_size, alpha=alpha, random_state=random_state
    )

    df.to_csv(out_dir / "embedding_property_r2.csv", index=False)
    emit_latex(df, out_dir / "table_property_r2.tex")

    print(df.to_string(index=False))
    print(f"\nWrote outputs to {out_dir}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1].strip())
    parser.add_argument("--embed", type=Path, default=EMBED_PATH)
    parser.add_argument("--meta", type=Path, default=META_PATH)
    parser.add_argument("--train_parquet", type=Path, default=TRAIN_PARQUET)
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_regression(
        args.embed,
        args.meta,
        args.train_parquet,
        args.out_dir,
        test_size=args.test_size,
        alpha=args.alpha,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()
