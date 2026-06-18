#!/usr/bin/env python3
# ruff: noqa: E402
"""
make_paper_figures.py

Generate paper figures from the 25-task results matrix. No new computation.

Outputs (PDF) into the manuscript figures directory:
  Fig_2.pdf            four-model internal comparison (paired scatter, 3 panels)
  Fig_baselines.pdf    best model vs four baselines (paired scatter, 4 panels)
  Fig_groupbars.pdf    per-task-group mean ROC-AUC grouped bar chart
  Fig_task_group_distributions.pdf
                       per-task distributions from packaged source data

Main-analysis exclusions:
- ogbg-moltoxcast  (26th MoleculeNet set; no Praski baseline)
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from modernmolbert.visualize.regen_groupfig import generate_group_distribution_figure

MATRIX = ROOT / "outputs/eval/paper/results_matrix_25task.csv"
FIGDIR = ROOT / "paper/figures"
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
    "ogbg-molsider": "SIDER",
}

df = pd.read_csv(MATRIX, index_col=0)
df = df.loc[~df.index.isin(EXCLUDED_DATASETS)].copy()


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


# ---------- Figure 2: four-model internal ----------
fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
panels = [
    ("MMB-small", "MMB-base", "(a) size"),
    ("MMB-small", "MMB-small-span", "(b) span masking"),
    ("MMB-small", "MMB-small-hetero", "(c) hetero-span masking"),
]
for ax, (x, y, title) in zip(axes, panels, strict=False):
    paired_panel(ax, x, y)
    ax.set_title(title, fontsize=10)
group_legend(fig)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(FIGDIR / "Fig_2.pdf", bbox_inches="tight")
plt.close(fig)

# ---------- Figure baselines: best (MMB-base) vs 4 baselines ----------
BEST = "MMB-base"
fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.3))
for ax, base in zip(axes, ["ECFP4", "ChemBERTa-2", "SELFormer", "MoLFormer"], strict=False):
    paired_panel(ax, base, BEST)
    ax.set_title(f"{BEST} vs {base}", fontsize=10)
group_legend(fig)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(FIGDIR / "Fig_baselines.pdf", bbox_inches="tight")
plt.close(fig)

# ---------- Bar chart: per-group mean ROC-AUC ----------
bar_models = ["ECFP4", "ChemBERTa-2", "SELFormer", "MoLFormer", "MMB-small", "MMB-base"]
bar_colors = ["#7f7f7f", "#bcbd22", "#17becf", "#9467bd", "#1f77b4", "#d62728"]
means = {m: [df.loc[df["group"] == g, m].mean() for g in GROUP_ORDER] for m in bar_models}
x = np.arange(len(GROUP_ORDER))
w = 0.13
fig, ax = plt.subplots(figsize=(9, 4.5))
for i, m in enumerate(bar_models):
    ax.bar(
        x + (i - 2.5) * w,
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

generate_group_distribution_figure(
    output_path=FIGDIR / "Fig_task_group_distributions.pdf",
    verbose=False,
)

print(
    "Wrote Fig_2.pdf, Fig_baselines.pdf, Fig_groupbars.pdf, Fig_task_group_distributions.pdf to",
    FIGDIR,
)
