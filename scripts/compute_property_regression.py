#!/usr/bin/env python3
"""
compute_property_regression.py
===============================

Quantify how well mean-pooled ModernMolBERT-small (span masking) token
embeddings predict nine ChEMBL physicochemical descriptors via Ridge regression.

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
1. Load 100 000 mean-pooled token embeddings (512-dim, float32).
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

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]

EMBED_PATH = ROOT / "outputs/visualize/best_span_100k/embeddings.npy"
META_PATH = ROOT / "outputs/visualize/best_span_100k/metadata.parquet"
TRAIN_PARQUET = ROOT / "data/pretrain/chembl36_selfies/train.parquet"
OUT_DIR = ROOT / "outputs/eval/paper"
FIGURE_DIR = OUT_DIR / "figures"

BREWER_BLUES = ["#EFF3FF", "#BDD7E7", "#6BAED6", "#3182BD", "#08519C"]
TEXT_COLOR = "#2B2B2B"
GRID_COLOR = "#D9D9D9"

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
    """Return (X, desc_df) where X[i] is the embedding for row i of desc_df.

    Rows with any missing descriptor value are dropped; the returned arrays are
    aligned (same row order).
    """
    # Load embeddings (100k × 512)
    X_full = np.load(embed_path)  # float32, mmap OK

    # Load metadata — embedding_row is the direct index into X_full
    meta = pd.read_parquet(meta_path, columns=["chembl_id", "embedding_row"])

    # Join with training parquet to get all descriptor columns
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
        r"    mean-pooled token embeddings to nine ChEMBL physicochemical descriptors, evaluated on",
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


def _apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "semibold",
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 9,
            "axes.edgecolor": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def emit_property_r2_plot(df: pd.DataFrame, out_dir: Path) -> None:
    """Write a horizontal bar plot of descriptor prediction R2 values."""
    _apply_paper_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_df = df.sort_values("r2", ascending=True).reset_index(drop=True)
    y = np.arange(len(plot_df))

    fig_height = max(3.2, 0.38 * len(plot_df) + 1.0)
    fig, ax = plt.subplots(figsize=(6.4, fig_height), constrained_layout=True)
    cmap = LinearSegmentedColormap.from_list("brewer_blues", BREWER_BLUES)
    r2_span = float(plot_df["r2"].max() - plot_df["r2"].min()) if len(plot_df) else 0.0
    if len(plot_df) > 1 and r2_span > 0:
        norm = (plot_df["r2"] - plot_df["r2"].min()) / (plot_df["r2"].max() - plot_df["r2"].min())
    else:
        norm = pd.Series([0.7] * len(plot_df), index=plot_df.index)
    colors = [cmap(0.35 + 0.55 * value) for value in norm]
    for i in y:
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#F7F7F7", zorder=0)
    bars = ax.barh(y, plot_df["r2"], color=colors, edgecolor="white", linewidth=0.7, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"].tolist())
    ax.set_xlabel(r"Test $R^2$")
    ax.set_title("Linear predictability of ChEMBL descriptors from frozen embeddings")
    ax.grid(axis="x", color=GRID_COLOR, lw=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axvline(0, color="#525252", lw=0.8)

    max_r2 = max(float(plot_df["r2"].max()), 0.0)
    min_r2 = min(float(plot_df["r2"].min()), 0.0)
    label_pad = max(0.02, (max_r2 - min_r2) * 0.03)
    for bar, value in zip(bars, plot_df["r2"], strict=False):
        value = float(value)
        if value >= 0:
            x = value + label_pad
            ha = "left"
        else:
            x = value - label_pad
            ha = "right"
        ax.text(
            x,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            ha=ha,
            fontsize=8,
            color=TEXT_COLOR,
        )

    ax.set_xlim(min(min_r2 - 0.08, 0.0), max_r2 + 0.12)
    fig.savefig(out_dir / "property_r2_bars.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "property_r2_bars.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


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
    make_figures: bool = True,
    figure_dir: Path = FIGURE_DIR,
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
    if make_figures:
        emit_property_r2_plot(df, figure_dir)

    print(df.to_string(index=False))
    print(f"\nWrote outputs to {out_dir}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Ridge regression R² from frozen embeddings to ChEMBL descriptors."
    )
    parser.add_argument("--embed", type=Path, default=EMBED_PATH)
    parser.add_argument("--meta", type=Path, default=META_PATH)
    parser.add_argument("--train_parquet", type=Path, default=TRAIN_PARQUET)
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--figure_dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--no_figures", action="store_true")
    args = parser.parse_args()
    build_regression(
        args.embed,
        args.meta,
        args.train_parquet,
        args.out_dir,
        test_size=args.test_size,
        alpha=args.alpha,
        random_state=args.seed,
        make_figures=not args.no_figures,
        figure_dir=args.figure_dir,
    )


if __name__ == "__main__":
    main()
