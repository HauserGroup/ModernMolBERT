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

Related tasks are not independent evidence: the five CYP Veith datasets, for
example, share an assay source and test molecules. ``task_families.yaml``
groups such tasks; the family bootstrap resamples whole families instead of
tasks. It keeps the task-weighted mean as its estimate, so only the interval
changes. A second estimate weights each family once, and win / tie / loss
counts are also reported over family means. A separate random stream is used,
so the task-level intervals are unchanged by the family analysis.

Inputs
------
- ``outputs/eval/paper/results_matrix_25task.csv``  (25 tasks × 8 models)
- ``src/modernmolbert/eval/benchmarking_molecular_models/config/task_families.yaml``

Outputs
-------
- ``outputs/eval/paper/bootstrap_cis.csv``
- ``outputs/eval/paper/table_bootstrap.tex``

Usage
-----
    uv run python scripts/paper/compute_bootstrap_cis.py
    uv run python scripts/paper/compute_bootstrap_cis.py --n_boot 5000 --seed 99
"""

import argparse
import hashlib
from collections.abc import Iterable
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "outputs/eval/paper/results_matrix_25task.csv"
FAMILIES_PATH = (
    ROOT / "src/modernmolbert/eval/benchmarking_molecular_models/config/task_families.yaml"
)
OUT_DIR = ROOT / "outputs/eval/paper"
FIGURE_DIR = OUT_DIR / "figures"

BREWER_DARK2 = {
    "mmb": "#1B9E77",
    "baseline": "#D95F02",
    "overlap": "#7570B3",
}
TEXT_COLOR = "#2B2B2B"
GRID_COLOR = "#D9D9D9"

LATEX_NAMES = {"MMB-base": r"\model{}-base", "MMB-small": r"\model{}-small"}

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


def read_task_families(path: Path) -> dict[str, str]:
    """Map each listed task to its family, rejecting a task listed twice."""
    mapping: dict[str, str] = {}
    for family, members in yaml.safe_load(path.read_text())["families"].items():
        for task in members:
            if task in mapping:
                raise ValueError(f"Task {task} is listed in two families")
            mapping[task] = family
    return mapping


def load_task_families(path: Path, tasks: Iterable[str]) -> dict[str, str]:
    """Map every task to its family; tasks not listed form their own family."""
    mapping = read_task_families(path)
    return {str(task): mapping.get(str(task), str(task)) for task in tasks}


def cluster_bootstrap(
    diffs: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
    weight: str = "task",
) -> tuple[float, float, float]:
    """Return (estimate, ci_low, ci_high), resampling whole groups of tasks.

    ``weight="task"`` keeps the mean over tasks, as in ``paired_bootstrap``;
    ``weight="family"`` averages the group means, so each group counts once.
    """
    if len(diffs) != len(groups):
        raise ValueError(
            f"diffs and groups must have equal length, got {len(diffs)} vs {len(groups)}"
        )
    if len(diffs) == 0:
        raise ValueError("Cannot bootstrap empty arrays")
    if weight not in {"task", "family"}:
        raise ValueError(f"Unknown weight {weight!r}")
    if rng is None:
        rng = np.random.default_rng()

    labels = list(dict.fromkeys(groups.tolist()))
    sums = np.array([diffs[groups == label].sum() for label in labels])
    counts = np.array([(groups == label).sum() for label in labels])
    idx = rng.integers(0, len(labels), size=(n_boot, len(labels)))
    if weight == "task":
        estimate = float(diffs.mean())
        boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    else:
        means = sums / counts
        estimate = float(means.mean())
        boot = means[idx].mean(axis=1)

    ci_low = float(np.percentile(boot, 100 * alpha / 2))
    ci_high = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return estimate, ci_low, ci_high


# ── per-comparison summary ──────────────────────────────────────────────────


def comparison_row(
    matrix: pd.DataFrame,
    model_a: str,
    model_b: str,
    n_boot: int,
    rng: np.random.Generator,
    families: dict[str, str] | None = None,
    family_rng: np.random.Generator | None = None,
) -> dict:
    """Compute summary statistics for one (model_a vs model_b) pair.

    With ``families``, also resample whole task families (see module notes).
    """
    matched = matrix.loc[:, [model_a, model_b]].dropna()
    a = matched[model_a].to_numpy(dtype=np.float64)
    b = matched[model_b].to_numpy(dtype=np.float64)
    diffs = a - b

    mean_diff, ci_low, ci_high = paired_bootstrap(a, b, n_boot=n_boot, rng=rng)

    row = {
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
    if families is None:
        return row

    family_rng = rng if family_rng is None else family_rng
    groups = np.array([families[str(task)] for task in matched.index])
    _, fam_low, fam_high = cluster_bootstrap(diffs, groups, n_boot, rng=family_rng)
    weighted, weighted_low, weighted_high = cluster_bootstrap(
        diffs, groups, n_boot, rng=family_rng, weight="family"
    )
    family_diffs = np.array([diffs[groups == g].mean() for g in dict.fromkeys(groups.tolist())])
    return row | {
        "n_families": len(family_diffs),
        "family_wins": int((family_diffs > 0).sum()),
        "family_ties": int((family_diffs == 0).sum()),
        "family_losses": int((family_diffs < 0).sum()),
        "family_ci_low_95": round(fam_low * 100, 2),
        "family_ci_high_95": round(fam_high * 100, 2),
        "family_weighted_delta": round(weighted * 100, 2),
        "family_weighted_ci_low_95": round(weighted_low * 100, 2),
        "family_weighted_ci_high_95": round(weighted_high * 100, 2),
    }


def _comparison_rng(seed: int, *parts: str) -> np.random.Generator:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).digest()
    words = np.frombuffer(digest[:16], dtype="<u4").tolist()
    return np.random.default_rng([seed, *words])


def run_comparisons(
    matrix: pd.DataFrame,
    comparisons: list[tuple[str, str]],
    n_boot: int,
    seed: int,
    families: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Run all comparisons and return a summary DataFrame."""
    rows = [
        comparison_row(
            matrix,
            a,
            b,
            n_boot,
            _comparison_rng(seed, a, b, "task"),
            families=families,
            family_rng=_comparison_rng(seed, a, b, "family"),
        )
        for a, b in comparisons
    ]
    return pd.DataFrame(rows)


# ── LaTeX emission ──────────────────────────────────────────────────────────


def _ci(low: float, high: float) -> str:
    return f"$[{low:+.1f},\\;{high:+.1f}]$"


def emit_latex(df: pd.DataFrame, out_path: Path, caption_note: str = "") -> None:
    """Write the bootstrap CI table; add task-family columns when present."""
    if "family_ci_low_95" not in df.columns:
        _emit_task_latex(df, out_path, caption_note)
        return
    model_a = " and ".join(
        LATEX_NAMES.get(str(name), str(name)) for name in dict.fromkeys(df["model_a"])
    )
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering\small",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \begin{tabular}{l r c r c r c c r}",
        r"    \toprule",
        r"    & \multicolumn{4}{c}{\textbf{Tasks}} & \multicolumn{4}{c}{\textbf{Task families}} \\",
        r"    \cmidrule(lr){2-5}\cmidrule(lr){6-9}",
        r"    \textbf{Baseline} & $n$ & \textbf{W/T/L} & \textbf{Mean $\Delta$} "
        r"& \textbf{95\,\% CI} & $n$ & \textbf{W/T/L} & \textbf{95\,\% CI} "
        r"& \textbf{Family $\Delta$} \\",
        r"    \midrule",
    ]
    for row in df.to_dict("records"):
        lines.append(
            f"    {row['model_b']} & {row['n_tasks']} "
            f"& {row['wins']}/{row['ties']}/{row['losses']} "
            f"& {row['mean_delta_roc_auc']:+.2f} & {_ci(row['ci_low_95'], row['ci_high_95'])} "
            f"& {row['n_families']} "
            f"& {row['family_wins']}/{row['family_ties']}/{row['family_losses']} "
            f"& {_ci(row['family_ci_low_95'], row['family_ci_high_95'])} "
            f"& {row['family_weighted_delta']:+.2f} \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{%",
        r"    Paired bootstrap 95\,\% intervals ($B=10{,}000$) on mean $\Delta$ ROC-AUC",
        f"    ($\\times100$), {model_a} minus each baseline, over the tasks where both",
        r"    models have a result. \emph{Tasks}: tasks are resampled individually, and",
        f"    W/T/L counts tasks where {model_a} is above, equal to, or below the baseline.",
        r"    \emph{Task families}: whole families of related tasks are resampled, keeping",
        r"    the same task-weighted mean; W/T/L compares family means, and",
        r"    \emph{Family $\Delta$} weights each family equally.",
    ]
    if caption_note:
        lines.append(f"    {caption_note}")
    lines += [r"  }%", r"  \label{tab:bootstrap-cis}", r"\end{table}"]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _emit_task_latex(df: pd.DataFrame, out_path: Path, caption_note: str = "") -> None:
    """Task-level table, used when no task families are supplied."""
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
    ]
    if caption_note:
        lines.append(f"    {caption_note}")
    lines += [r"  }%", r"  \label{tab:bootstrap-cis}", r"\end{table}"]
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
    reference = " / ".join(dict.fromkeys(str(m) for m in df["model_a"]))
    ax.set_xlabel(rf"Mean $\Delta$ ROC-AUC ({reference} - baseline, x100)")
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
        Line2D(
            [0], [0], marker="o", color=BREWER_DARK2["mmb"], lw=2, label=f"{reference} advantage"
        ),
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
    families_path: Path | None = FAMILIES_PATH,
    caption_note: str = "",
) -> pd.DataFrame:
    """Full pipeline: load matrix → bootstrap → save CSV and LaTeX."""
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix = pd.read_csv(matrix_path, index_col=0)
    families = (
        load_task_families(families_path, matrix.index) if families_path is not None else None
    )

    df = run_comparisons(matrix, comparisons, n_boot=n_boot, seed=seed, families=families)

    df.to_csv(out_dir / "bootstrap_cis.csv", index=False)
    emit_latex(df, out_dir / "table_bootstrap.tex", caption_note)
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
    parser.add_argument("--families", type=Path, default=FAMILIES_PATH)
    parser.add_argument(
        "--no_families", action="store_true", help="Report task-level intervals only."
    )
    parser.add_argument("--caption_note", default="", help="Sentence appended to the caption.")
    parser.add_argument(
        "--reference",
        default=None,
        help="Matrix column compared with each baseline (default: the four archived comparisons).",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=[b for _, b in COMPARISONS],
        help="Baseline matrix columns, used with --reference.",
    )
    args = parser.parse_args()
    comparisons = (
        COMPARISONS
        if args.reference is None
        else [(args.reference, baseline) for baseline in args.baselines]
    )
    build_cis(
        args.matrix,
        args.out_dir,
        args.n_boot,
        args.seed,
        comparisons=comparisons,
        make_figures=not args.no_figures,
        figure_dir=args.figure_dir,
        families_path=None if args.no_families else args.families,
        caption_note=args.caption_note,
    )


if __name__ == "__main__":
    main()
