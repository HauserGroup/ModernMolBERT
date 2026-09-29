#!/usr/bin/env python3
# ruff: noqa: E402
"""
make_paper_figures.py

Generate paper figures from the 25-task results matrix. No new computation.

Outputs (PDF) into the manuscript figures directory:
  Fig_2.pdf            internal comparison (paired scatter: size, span masking;
                       plus hetero-span masking with --include-hetero-span)
  Fig_baselines.pdf    best model vs four baselines (paired scatter, 4 panels)
  Fig_groupbars.pdf    per-task-group mean ROC-AUC grouped bar chart
  Fig_task_group_distributions.pdf
                       per-task distributions from packaged source data

Main-analysis exclusions:
- ogbg-moltoxcast  (26th MoleculeNet set; no Praski baseline)

The hetero-span ablation (MMB-small-hetero) is excluded by default; pass
--include-hetero-span (and build the matrix with the same flag) to add it back.

For a newly trained model, pass the matrix written by
``build_paper_results.py --task-matrix`` with ``--matrix`` and name the model
with ``--reference``. Fig_2 is skipped unless the released small, base and
span columns are present. The task-group distribution figure is then drawn
from a source CSV written from the same matrix into ``--source-data-dir``.
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser(description="Generate paper figures from the results matrix.")
parser.add_argument(
    "--include-hetero-span",
    action="store_true",
    help="Add the MMB-small-hetero (hetero_span masking) panel to Fig_2.",
)
ROOT = Path(__file__).resolve().parents[2]
parser.add_argument(
    "--matrix", type=Path, default=ROOT / "outputs/eval/paper/results_matrix_25task.csv"
)
parser.add_argument("--figure-dir", type=Path, default=ROOT / "paper/figures")
parser.add_argument(
    "--reference", default="MMB-base", help="ModernMolBERT column compared with the baselines."
)
parser.add_argument(
    "--source-data-dir",
    type=Path,
    default=None,
    help="Write the task-group source CSV from the matrix here and plot from it "
    "(default: plot the bundled paper/source_data CSV).",
)
ARGS = parser.parse_args()

SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from modernmolbert.visualize.regen_groupfig import generate_group_distribution_figure

MATRIX = ARGS.matrix
FIGDIR = ARGS.figure_dir
FIGDIR.mkdir(parents=True, exist_ok=True)
BASELINES = ["ECFP4", "ChemBERTa-2", "SELFormer", "MoLFormer"]
EXCLUDED_DATASETS = {"ogbg-moltoxcast"}

GROUP_COLORS = {
    "TDC-ADME": "#4C72B0",
    "TDC-Tox": "#DD8452",
    "TDC-HTS": "#55A868",
    "MoleculeNet": "#C44E52",
}
GROUP_ORDER = ["TDC-ADME", "TDC-Tox", "TDC-HTS", "MoleculeNet"]

# short, readable task labels for outlier annotation
SHORT = {
    "Bioavailability_Ma": "Bioav",
    "HIA_Hou": "HIA",
    "Pgp_Broccatelli": "Pgp",
    "PAMPA_NCATS": "PAMPA",
    "CYP1A2_Veith": "CYP1A2",
    "CYP2C19_Veith": "CYP2C19",
    "CYP2C9_Veith": "CYP2C9",
    "CYP2D6_Veith": "CYP2D6",
    "CYP3A4_Veith": "CYP3A4",
    "CYP2C9_Substrate_CarbonMangels": "CYP2C9-sub",
    "CYP2D6_Substrate_CarbonMangels": "CYP2D6-sub",
    "CYP3A4_Substrate_CarbonMangels": "CYP3A4-sub",
    "AMES": "AMES",
    "DILI": "DILI",
    "hERG": "hERG",
    "hERG_Karim": "hERG-K",
    "SARSCoV2_3CLPro_Diamond": "3CLPro",
    "SARSCoV2_Vitro_Touret": "SARS-Vitro",
    "ogbg-molbace": "BACE",
    "ogbg-molbbbp": "BBBP",
    "ogbg-molclintox": "ClinTox",
    "ogbg-molhiv": "HIV",
    "ogbg-molmuv": "MUV",
    "ogbg-molsider": "SIDER",
    "ogbg-moltox21": "Tox21",
}

df = pd.read_csv(MATRIX, index_col=0)
df = df.loc[~df.index.isin(EXCLUDED_DATASETS)].copy()
if ARGS.include_hetero_span and "MMB-small-hetero" not in df.columns:
    raise ValueError(
        f"{MATRIX} has no MMB-small-hetero column; rerun "
        "build_paper_results.py with --include-hetero-span first."
    )
if absent := [m for m in [*BASELINES, ARGS.reference] if m not in df.columns]:
    raise ValueError(f"{MATRIX} lacks model columns {absent}")
MMB_MODELS = [c for c in df.columns if c.startswith("MMB-") and c != "MMB-small-hetero"]


def paired_panel(ax, xcol, ycol, gap=0.05, lim=(0.45, 1.0)):
    """Scatter ycol vs xcol, identity line, color by group, label big gaps."""
    sub = df[[xcol, ycol, "group"]].dropna()
    for g in GROUP_ORDER:
        s = sub[sub["group"] == g]
        ax.scatter(
            s[xcol],
            s[ycol],
            s=46,
            c=GROUP_COLORS[g],
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
            label=g,
        )
    ax.plot(lim, lim, ls="--", c="0.4", lw=1, zorder=1)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal")
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    # annotate tasks where the gap exceeds threshold
    for t, r in sub.iterrows():
        if abs(r[ycol] - r[xcol]) > gap:
            ax.annotate(
                SHORT.get(t, t),  # type: ignore
                (r[xcol], r[ycol]),
                fontsize=6.5,
                xytext=(3, 3),
                textcoords="offset points",
                color="0.25",
            )
    # win count annotation
    d = sub[ycol] - sub[xcol]
    ax.text(
        0.04,
        0.96,
        f"{ycol} > {xcol}: {int((d > 0).sum())}/{len(d)}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", lw=0.5),
    )


def group_legend(fig):
    handles = [
        Line2D([0], [0], marker="o", ls="", mfc=GROUP_COLORS[g], mec="white", ms=8, label=g)
        for g in GROUP_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=9,
    )


# ---------- Figure 2: internal comparison ----------
panels = [
    ("MMB-small", "MMB-base", "(a) size"),
    ("MMB-small", "MMB-small-span", "(b) span masking"),
]
if ARGS.include_hetero_span:
    panels.append(("MMB-small", "MMB-small-hetero", "(c) hetero-span masking"))
written = []
if all({x, y} <= set(df.columns) for x, y, _ in panels):
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.3))
    for ax, (x, y, title) in zip(axes, panels, strict=False):
        paired_panel(ax, x, y)
        ax.set_title(title, fontsize=10)
    group_legend(fig)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIGDIR / "Fig_2.pdf", bbox_inches="tight")
    plt.close(fig)
    written.append("Fig_2.pdf")
else:
    print("Skipping Fig_2: the internal comparison needs the released small, base and span columns")

# ---------- Figure baselines: reference model vs 4 baselines ----------
BEST = ARGS.reference
fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.3))
for ax, base in zip(axes, ["ECFP4", "ChemBERTa-2", "SELFormer", "MoLFormer"], strict=False):
    paired_panel(ax, base, BEST)
    ax.set_title(f"{BEST} vs {base}", fontsize=10)
group_legend(fig)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(FIGDIR / "Fig_baselines.pdf", bbox_inches="tight")
plt.close(fig)
written.append("Fig_baselines.pdf")

# ---------- Bar chart: per-group mean ROC-AUC ----------
bar_models = [*BASELINES, *MMB_MODELS]
bar_colors = ["#7f7f7f", "#bcbd22", "#17becf", "#9467bd", "#1f77b4", "#d62728", "#8c564b"]
means = {m: [df.loc[df["group"] == g, m].mean() for g in GROUP_ORDER] for m in bar_models}
x = np.arange(len(GROUP_ORDER))
w = 0.13
fig, ax = plt.subplots(figsize=(9, 4.5))
for i, m in enumerate(bar_models):
    ax.bar(
        x + (i - (len(bar_models) - 1) / 2) * w,
        means[m],
        w,
        label=m,
        color=bar_colors[i],
        edgecolor="white",
        linewidth=0.4,
    )
ax.set_xticks(x)
ax.set_xticklabels(GROUP_ORDER)
ax.set_ylabel("Mean ROC-AUC")
ax.set_ylim(0.55, 0.90)
ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16))
ax.grid(axis="y", ls=":", c="0.85")
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(FIGDIR / "Fig_groupbars.pdf", bbox_inches="tight")
plt.close(fig)
written.append("Fig_groupbars.pdf")

group_csv = None
if ARGS.source_data_dir is not None:
    ARGS.source_data_dir.mkdir(parents=True, exist_ok=True)
    group_csv = ARGS.source_data_dir / "Fig_task_group_distributions.csv"
    long = df[["group", *bar_models]].melt(
        id_vars="group", var_name="model", value_name="roc_auc", ignore_index=False
    )
    pd.DataFrame(
        {
            "task_group": long["group"],
            "task": [SHORT.get(str(t), str(t)) for t in long.index],
            "model": long["model"],
            "roc_auc_x100": (long["roc_auc"] * 100).round(1),
        }
    ).to_csv(group_csv, index=False)
generate_group_distribution_figure(
    csv_path=group_csv,
    output_path=FIGDIR / "Fig_task_group_distributions.pdf",
    verbose=False,
    models=bar_models if group_csv is not None else None,
)
written.append("Fig_task_group_distributions.pdf")

print("Wrote", ", ".join(written), "to", FIGDIR)
