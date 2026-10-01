#!/usr/bin/env python3
"""Task and family bootstrap intervals for five-seed internal contrasts."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from aggregate_revision_seeds import CONTRASTS, MODELS
from compute_bootstrap_cis import cluster_bootstrap, load_task_families, paired_bootstrap
from modernmolbert.utils import file_sha256, get_git_revision

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ROOT / "src/modernmolbert/eval/benchmarking_molecular_models/config/task_families.yaml"


def compute_intervals(
    matrix: pd.DataFrame,
    seed_summary: pd.DataFrame,
    families: dict[str, str],
    *,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    if len(matrix) != 25 or set(matrix.columns) != set(MODELS):
        raise ValueError("Expected 25 tasks and the five revision configurations")
    if not np.isfinite(matrix.to_numpy(dtype=float)).all():
        raise ValueError("Internal mean matrix has missing or nonfinite scores")
    if set(matrix.index) != set(families):
        raise ValueError("Task family map differs from the internal matrix")
    if set(seed_summary["contrast"]) != set(CONTRASTS) or not seed_summary["n_seeds"].eq(5).all():
        raise ValueError("Expected five-seed summaries for all prespecified contrasts")
    if n_boot < 100:
        raise ValueError("At least 100 bootstrap draws are required")
    summary = seed_summary.set_index("contrast")
    groups = np.asarray([families[str(task)] for task in matrix.index])
    rows = []
    for index, (name, terms) in enumerate(CONTRASTS.items()):
        differences = np.asarray(
            sum(weight * matrix[model] for model, weight in terms), dtype=float
        )
        estimate, task_low, task_high = paired_bootstrap(
            differences,
            np.zeros_like(differences),
            n_boot=n_boot,
            rng=np.random.default_rng([seed, index, 0]),
        )
        family_estimate, family_low, family_high = cluster_bootstrap(
            differences,
            groups,
            n_boot=n_boot,
            rng=np.random.default_rng([seed, index, 1]),
        )
        family_weighted, weighted_low, weighted_high = cluster_bootstrap(
            differences,
            groups,
            n_boot=n_boot,
            rng=np.random.default_rng([seed, index, 2]),
            weight="family",
        )
        if not np.isclose(estimate, family_estimate):
            raise ValueError("Task and family bootstrap point estimates differ")
        if not np.isclose(estimate, float(summary.loc[name, "mean"])):
            raise ValueError(f"Seed aggregate and mean matrix disagree: {name}")
        rows.append(
            {
                "contrast": name,
                "n_tasks": len(differences),
                "n_families": len(set(groups)),
                "n_seeds": 5,
                "mean_delta_roc_auc": estimate,
                "task_ci_low": task_low,
                "task_ci_high": task_high,
                "family_ci_low": family_low,
                "family_ci_high": family_high,
                "family_weighted_delta": family_weighted,
                "family_weighted_ci_low": weighted_low,
                "family_weighted_ci_high": weighted_high,
                "sd_across_seed_task_means": float(summary.loc[name, "sd_across_seeds"]),
                "min_seed_task_mean": float(summary.loc[name, "min"]),
                "max_seed_task_mean": float(summary.loc[name, "max"]),
            }
        )
    return pd.DataFrame(rows)


def plot_intervals(frame: pd.DataFrame, path: Path) -> None:
    names = [name.replace("_", " ") for name in frame["contrast"]]
    y = np.arange(len(frame))
    estimate = frame["mean_delta_roc_auc"].to_numpy(dtype=float) * 100
    task_low = frame["task_ci_low"].to_numpy(dtype=float) * 100
    task_high = frame["task_ci_high"].to_numpy(dtype=float) * 100
    family_low = frame["family_ci_low"].to_numpy(dtype=float) * 100
    family_high = frame["family_ci_high"].to_numpy(dtype=float) * 100
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.errorbar(
        estimate,
        y - 0.12,
        xerr=[estimate - task_low, task_high - estimate],
        fmt="o",
        color="#1B9E77",
        capsize=3,
        label="Task bootstrap",
    )
    ax.errorbar(
        estimate,
        y + 0.12,
        xerr=[estimate - family_low, family_high - estimate],
        fmt="s",
        color="#7570B3",
        capsize=3,
        label="Family bootstrap",
    )
    ax.axvline(0, color="0.4", linestyle="--", linewidth=1)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel("Mean ROC-AUC difference (percentage points)")
    ax.legend(loc="best", frameon=False)
    ax.grid(axis="x", color="0.87", linestyle=":")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mean-matrix", type=Path, required=True)
    parser.add_argument("--seed-contrast-summary", type=Path, required=True)
    parser.add_argument("--families", type=Path, default=FAMILIES)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    matrix = pd.read_csv(args.mean_matrix, index_col=0)
    seed_summary = pd.read_csv(args.seed_contrast_summary)
    families = load_task_families(args.families, matrix.index)
    frame = compute_intervals(
        matrix,
        seed_summary,
        families,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "internal_contrast_intervals.csv"
    figure_path = args.output_dir / "internal_contrast_intervals.pdf"
    frame.to_csv(csv_path, index=False)
    plot_intervals(frame, figure_path)
    (args.output_dir / "internal_contrast_intervals_manifest.json").write_text(
        json.dumps(
            {
                "code": get_git_revision(),
                "mean_matrix_sha256": file_sha256(args.mean_matrix),
                "seed_contrast_summary_sha256": file_sha256(args.seed_contrast_summary),
                "families_sha256": file_sha256(args.families),
                "n_boot": args.n_boot,
                "seed": args.seed,
                "interval": "95% percentile, task and fixed-family resampling kept separate",
                "interpretation": (
                    "Intervals condition on the five trained seeds' mean task scores; "
                    "between-seed SD is reported separately, not pooled with task replicates."
                ),
                "csv_sha256": file_sha256(csv_path),
                "figure_sha256": file_sha256(figure_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {csv_path} and {figure_path}")


if __name__ == "__main__":
    main()
