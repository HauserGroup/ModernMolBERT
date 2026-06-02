#!/usr/bin/env python3
"""
compute_bootstrap_cis.py
========================

Compute paired bootstrap 95 % confidence intervals on mean Δ ROC-AUC for the
four key pairwise comparisons between ModernMolBERT and baseline embedders.

Overview
--------
The benchmark evaluates mean-pooled frozen token embeddings on 25 molecular
property-prediction tasks.  For ModernMolBERT, pooling is over non-special
SELFIES tokens.  For each pair (MMB-base vs baseline), a Wilcoxon signed-rank
test already appears in the main paper.  This supplementary analysis adds
bootstrap confidence intervals to quantify the uncertainty on the *mean*
difference without assuming a parametric distribution.

Method
------
Paired bootstrap (B = 10 000 iterations by default):

1. For each iteration, resample the *n* matched task rows **with replacement**.
2. Compute mean Δ ROC-AUC on the resample.
3. Report the 2.5th and 97.5th percentiles as the 95 % CI.

The comparison is run on the set of tasks where **both** models have a result
(``results_matrix_25task.csv`` may have missing cells for some MMB variants).
Win / tie / loss counts are computed on the full matched set (not resampled).

Inputs
------
- ``outputs/eval/paper/results_matrix_25task.csv``  (25 tasks × 8 models)

Outputs
-------
- ``outputs/eval/paper/bootstrap_cis.csv``
- ``outputs/eval/paper/table_bootstrap.tex``

Usage
-----
    python scripts/compute_bootstrap_cis.py
    python scripts/compute_bootstrap_cis.py --n_boot 5000 --seed 99
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "outputs/eval/paper/results_matrix_25task.csv"
OUT_DIR = ROOT / "outputs/eval/paper"
FIGURE_DIR = OUT_DIR / "figures"

BREWER_DARK2 = {
    "mmb": "#1B9E77",
    "baseline": "#D95F02",
    "overlap": "#7570B3",
}
TEXT_COLOR = "#2B2B2B"
GRID_COLOR = "#D9D9D9"

# Comparisons: (model_a, model_b) — CI is for mean(a − b).
COMPARISONS: list[tuple[str, str]] = [
    ("MMB-base", "SELFormer"),
    ("MMB-base", "ChemBERTa-2"),
    ("MMB-base", "ECFP4"),
    ("MMB-base", "MoLFormer"),
]

# ── core bootstrap ──────────────────────────────────────────────────────────


def paired_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Return (mean_diff, ci_low, ci_high) for mean(a − b) via paired bootstrap.

    Parameters
    ----------
    a, b:
        1-D arrays of equal length; each entry is a per-task ROC-AUC.
    n_boot:
        Number of bootstrap resamples.
    alpha:
        Two-sided significance level (0.05 → 95 % CI).
    rng:
        NumPy random Generator for reproducibility.
    """
    if len(a) != len(b):
        raise ValueError(f"a and b must have equal length, got {len(a)} vs {len(b)}")
    if len(a) == 0:
        raise ValueError("Cannot bootstrap empty arrays")
    if rng is None:
        rng = np.random.default_rng()

    diffs = a - b
    mean_diff = float(diffs.mean())

    n = len(diffs)
    boot_means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = diffs[idx].mean()

    ci_low = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return mean_diff, ci_low, ci_high


# ── per-comparison summary ──────────────────────────────────────────────────


def comparison_row(
    matrix: pd.DataFrame,
    model_a: str,
    model_b: str,
    n_boot: int,
    rng: np.random.Generator,
) -> dict:
    """Compute summary statistics for one (model_a vs model_b) pair."""
    matched = matrix[[model_a, model_b]].dropna()
    a = matched[model_a].to_numpy(dtype=np.float64)
    b = matched[model_b].to_numpy(dtype=np.float64)
    diffs = a - b

    mean_diff, ci_low, ci_high = paired_bootstrap(a, b, n_boot=n_boot, rng=rng)

    return {
        "model_a": model_a,
        "model_b": model_b,
        "n_tasks": len(matched),
        "wins": int((diffs > 0).sum()),
        "ties": int((diffs == 0).sum()),
        "losses": int((diffs < 0).sum()),
        "mean_delta_roc_auc": round(mean_diff * 100, 2),
        "ci_low_95": round(ci_low * 100, 2),
        "ci_high_95": round(ci_high * 100, 2),
    }


def run_comparisons(
    matrix: pd.DataFrame,
    comparisons: list[tuple[str, str]],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    """Run all comparisons and return a summary DataFrame."""
    rng = np.random.default_rng(seed)
    rows = [comparison_row(matrix, a, b, n_boot, rng) for a, b in comparisons]
    return pd.DataFrame(rows)


# ── LaTeX emission ──────────────────────────────────────────────────────────


def emit_latex(df: pd.DataFrame, out_path: Path) -> None:
    """Write a tabular LaTeX fragment for the bootstrap CI table."""
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering\small",
        r"  \begin{tabular}{l l r r r r r r}",
        r"    \toprule",
        r"    \textbf{Model A} & \textbf{Model B} & \textbf{$n$} "
        r"& \textbf{Wins} & \textbf{Ties} & \textbf{Losses} "
        r"& \textbf{Mean $\Delta$} & \textbf{95\,\% CI} \\",
        r"    \midrule",
    ]
    for _, row in df.iterrows():
        ci_str = f"[{row['ci_low_95']:+.1f},\\;{row['ci_high_95']:+.1f}]"
        lines.append(
            f"    {row['model_a']} & {row['model_b']} & {row['n_tasks']} "
            f"& {row['wins']} & {row['ties']} & {row['losses']} "
            f"& {row['mean_delta_roc_auc']:+.2f} & ${ci_str}$ \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{%",
        r"    Paired bootstrap confidence intervals (95\,\%, $B=10{,}000$ resamples) on",
        r"    mean $\Delta$ ROC-AUC ($\times100$) between \model{}-base and each baseline,",
        r"    computed over the tasks where both models have a result.",
        r"    \emph{Wins}/\emph{Ties}/\emph{Losses} count tasks where \model{}-base is",
        r"    above, equal to, or below the baseline.",
        r"    A positive mean $\Delta$ and CI entirely above zero indicates a consistent",
        r"    advantage for \model{}-base.",
        r"  }%",
        r"  \label{tab:bootstrap-cis}",
        r"\end{table}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ci_color(row: pd.Series) -> str:
    if row["ci_low_95"] > 0:
        return BREWER_DARK2["mmb"]
    if row["ci_high_95"] < 0:
        return BREWER_DARK2["baseline"]
    return BREWER_DARK2["overlap"]


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
            "legend.fontsize": 8,
            "axes.edgecolor": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def emit_ci_forest_plot(df: pd.DataFrame, out_dir: Path) -> None:
    """Write a horizontal forest plot of mean delta ROC-AUC with 95% CIs."""
    _apply_paper_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_df = df.copy().iloc[::-1].reset_index(drop=True)
    y = np.arange(len(plot_df))

    x_min = min(float(plot_df["ci_low_95"].min()), 0.0) - 1.0
    x_max = max(float(plot_df["ci_high_95"].max()), 0.0) + 3.0
    annotation_x = x_max - 0.25

    fig, ax = plt.subplots(figsize=(6.7, 3.1), constrained_layout=True)
    for i in y:
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#F7F7F7", zorder=0)
    ax.axvline(0, color="#525252", lw=1.0, ls=(0, (4, 3)), zorder=1)

    ax.set_xlim(x_min, x_max)
    for i, (_, row) in enumerate(plot_df.iterrows()):
        yi = float(i)
        mean = float(row["mean_delta_roc_auc"])
        lo = float(row["ci_low_95"])
        hi = float(row["ci_high_95"])
        color = _ci_color(row)
        ax.errorbar(
            mean,
            yi,
            xerr=[[mean - lo], [hi - mean]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=2.0,
            capsize=3.5,
            capthick=1.2,
            markersize=5.5,
            markeredgecolor="white",
            markeredgewidth=0.7,
            zorder=3,
        )
        ax.text(
            annotation_x,
            yi,
            f"{int(row['wins'])} − {int(row['losses'])}",
            va="center",
            ha="right",
            fontsize=8,
            color=TEXT_COLOR,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["model_b"].tolist())
    ax.set_xlabel(r"Mean $\Delta$ ROC-AUC (MMB-base - baseline, x100)")
    ax.set_ylabel("Baseline embedder")
    ax.set_title("Paired bootstrap confidence intervals")
    ax.grid(axis="x", color=GRID_COLOR, lw=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        annotation_x,
        len(plot_df) - 0.18,
        "wins-losses / n",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#525252",
    )
    handles = [
        Line2D([0], [0], marker="o", color=BREWER_DARK2["mmb"], lw=2, label="MMB-base advantage"),
        Line2D(
            [0],
            [0],
            marker="o",
            color=BREWER_DARK2["baseline"],
            lw=2,
            label="Baseline advantage",
        ),
        Line2D([0], [0], marker="o", color=BREWER_DARK2["overlap"], lw=2, label="CI crosses 0"),
    ]
    ax.legend(
        handles=handles, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.34)
    )

    fig.savefig(out_dir / "bootstrap_ci_forest.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "bootstrap_ci_forest.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ── main ────────────────────────────────────────────────────────────────────


def build_cis(
    matrix_path: Path = MATRIX_PATH,
    out_dir: Path = OUT_DIR,
    n_boot: int = 10_000,
    seed: int = 42,
    comparisons: list[tuple[str, str]] = COMPARISONS,
    make_figures: bool = True,
    figure_dir: Path = FIGURE_DIR,
) -> pd.DataFrame:
    """Full pipeline: load matrix → bootstrap → save CSV and LaTeX."""
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix = pd.read_csv(matrix_path, index_col=0)

    df = run_comparisons(matrix, comparisons, n_boot=n_boot, seed=seed)

    df.to_csv(out_dir / "bootstrap_cis.csv", index=False)
    emit_latex(df, out_dir / "table_bootstrap.tex")
    if make_figures:
        emit_ci_forest_plot(df, figure_dir)

    print(df.to_string(index=False))
    print(f"\nWrote outputs to {out_dir}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute paired-bootstrap confidence intervals for benchmark ROC-AUC deltas.",
    )
    parser.add_argument("--n_boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--matrix", type=Path, default=MATRIX_PATH)
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--figure_dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--no_figures", action="store_true")
    args = parser.parse_args()
    build_cis(
        args.matrix,
        args.out_dir,
        args.n_boot,
        args.seed,
        make_figures=not args.no_figures,
        figure_dir=args.figure_dir,
    )


if __name__ == "__main__":
    main()
