"""Plot ROC and precision-recall curves from score.py .npz prediction files.

Usage
-----
    python plot_predictions.py --embedder my_model
    python plot_predictions.py --predictions-dir data/predictions --output-dir data/plots
    python plot_predictions.py --embedder my_model --datasets AMES BBBP

Output
------
    data/plots/<dataset>.png          — ROC + PR curves (classification)
    data/plots/<dataset>_regression.png — true-vs-predicted scatter (regression)

Each .npz file was written by score.py under:
    data/predictions/<dataset>/<embedder>/<head>.npz
with arrays:
    y_true  — ground-truth labels
    y_score — positive-class probability (binary) or per-task scores (multioutput)
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, precision_recall_curve, roc_curve


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def _find_npz_files(
    predictions_dir: Path,
    embedder: str | None,
    datasets: list[str] | None,
) -> dict[str, dict[str, dict[str, Path]]]:
    """Return nested mapping: dataset → embedder → head → Path."""
    result: dict[str, dict[str, dict[str, Path]]] = {}
    for npz in sorted(predictions_dir.rglob("*.npz")):
        parts = npz.relative_to(predictions_dir).parts
        if len(parts) != 3:
            continue
        ds, emb, head_file = parts
        head = head_file[:-4]
        if embedder and emb != embedder:
            continue
        if datasets and ds not in datasets:
            continue
        result.setdefault(ds, {}).setdefault(emb, {})[head] = npz
    return result


def _load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path)
    return d["y_true"], d["y_score"]


# ---------------------------------------------------------------------------
# Per-axis helpers
# ---------------------------------------------------------------------------


def _plot_roc(ax, y_true: np.ndarray, y_score: np.ndarray, label: str, color=None) -> float:
    mask = np.isfinite(y_true)
    fpr, tpr, _ = roc_curve(y_true[mask], y_score[mask])
    auroc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{label}  AUROC={auroc:.3f}", color=color, lw=1.5)
    return auroc


def _plot_pr(ax, y_true: np.ndarray, y_score: np.ndarray, label: str, color=None) -> float:
    mask = np.isfinite(y_true)
    precision, recall, _ = precision_recall_curve(y_true[mask], y_score[mask])
    auprc = auc(recall, precision)
    ax.plot(recall, precision, label=f"{label}  AUPRC={auprc:.3f}", color=color, lw=1.5)
    return auprc


def _has_enough_data(y_true: np.ndarray) -> bool:
    mask = np.isfinite(y_true)
    return mask.sum() >= 2 and len(np.unique(y_true[mask])) >= 2


# ---------------------------------------------------------------------------
# Per-dataset figure builders
# ---------------------------------------------------------------------------


def _clf_figure(
    dataset: str,
    entries: dict[str, dict[str, Path]],
    output_dir: Path,
) -> Path:
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ci = 0

    for emb in sorted(entries):
        for head in sorted(entries[emb]):
            y_true, y_score = _load(entries[emb][head])
            color = colors[ci % len(colors)]

            if y_score.ndim == 2:
                # Multioutput: one line per task column.
                n_tasks = y_score.shape[1]
                for t in range(n_tasks):
                    col_true = y_true[:, t] if y_true.ndim == 2 else y_true
                    col_score = y_score[:, t]
                    if not _has_enough_data(col_true):
                        continue
                    tag = f"{emb}/{head}/task{t}"
                    c = colors[ci % len(colors)]
                    _plot_roc(ax_roc, col_true, col_score, tag, color=c)
                    _plot_pr(ax_pr, col_true, col_score, tag, color=c)
                    ci += 1
            else:
                if not _has_enough_data(y_true):
                    continue
                tag = f"{emb}/{head}"
                _plot_roc(ax_roc, y_true, y_score, tag, color=color)
                _plot_pr(ax_pr, y_true, y_score, tag, color=color)
                ci += 1

    ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
    ax_roc.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"ROC — {dataset}",
    )
    ax_roc.legend(fontsize=7, loc="lower right")

    ax_pr.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Recall",
        ylabel="Precision",
        title=f"Precision-Recall — {dataset}",
    )
    ax_pr.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    out = output_dir / f"{dataset}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _reg_figure(
    dataset: str,
    entries: dict[str, dict[str, Path]],
    output_dir: Path,
) -> Path:
    n_panels = sum(len(heads) for heads in entries.values())
    fig, axes = plt.subplots(1, max(n_panels, 1), figsize=(5 * max(n_panels, 1), 5), squeeze=False)
    axes = axes[0]
    idx = 0

    for emb in sorted(entries):
        for head in sorted(entries[emb]):
            y_true, y_score = _load(entries[emb][head])
            ax = axes[idx]
            mask = np.isfinite(y_true) & np.isfinite(y_score)
            lo = min(float(y_true[mask].min()), float(y_score[mask].min()))
            hi = max(float(y_true[mask].max()), float(y_score[mask].max()))
            ax.scatter(y_true[mask], y_score[mask], alpha=0.4, s=12)
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
            ax.set(xlabel="True", ylabel="Predicted", title=f"{emb}/{head}")
            idx += 1

    fig.suptitle(f"Regression — {dataset}")
    fig.tight_layout()
    out = output_dir / f"{dataset}_regression.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def make_plots(
    predictions_dir: Path,
    output_dir: Path,
    embedder: str | None = None,
    datasets: list[str] | None = None,
) -> list[Path]:
    """Generate one figure per dataset found under predictions_dir.

    Returns list of written paths.
    """
    tree = _find_npz_files(predictions_dir, embedder, datasets)
    if not tree:
        print(f"No .npz files found under {predictions_dir}")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for dataset, entries in sorted(tree.items()):
        first_path = next(p for heads in entries.values() for p in heads.values())
        y_true, _ = _load(first_path)
        flat = y_true.ravel()
        is_clf = set(flat[np.isfinite(flat)].tolist()).issubset({0.0, 1.0})

        if is_clf:
            out = _clf_figure(dataset, entries, output_dir)
        else:
            out = _reg_figure(dataset, entries, output_dir)

        print(f"  {out}")
        saved.append(out)

    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ROC / PR curves from score.py .npz prediction files."
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("data/predictions"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/plots"),
    )
    parser.add_argument(
        "--embedder",
        default=None,
        help="Restrict to one embedder (default: all).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Restrict to specific dataset names (default: all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Reading predictions from: {args.predictions_dir}")
    saved = make_plots(
        predictions_dir=args.predictions_dir,
        output_dir=args.output_dir,
        embedder=args.embedder,
        datasets=args.datasets,
    )
    print(f"\n{len(saved)} plot(s) written to {args.output_dir}")


if __name__ == "__main__":
    main()
