#!/usr/bin/env python3
"""
build_paper_results.py

Derive all paper-facing benchmark numbers from the already-built
`outputs/eval/best_metric_by_dataset_embedder.csv` (one row per
dataset x embedder, best downstream head already selected upstream).

Produces:
  outputs/eval/paper/results_matrix_25task.csv   (tasks x models, ROC-AUC)
  outputs/eval/paper/group_means.csv             (model x task-group means + overall)
  outputs/eval/paper/table2.tex                  (main benchmark LaTeX table)
  outputs/eval/paper/stats.txt                   (Wilcoxon tests + prose counts)

No model runs, no new benchmarking. Pure wrangling of existing eval output.
ogbg-moltoxcast is excluded: it has ModernMolBERT results but no Praski
baselines, so it is outside the official 25-task benchmark.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs/eval/best_metric_by_dataset_embedder.csv"
OUT = ROOT / "outputs/eval/paper"
OUT.mkdir(parents=True, exist_ok=True)

# ---- Task -> group map (18 TDC + 7 MoleculeNet = 25; ToxCast excluded) ----
TDC_ADME = [
    "Bioavailability_Ma",
    "HIA_Hou",
    "Pgp_Broccatelli",
    "PAMPA_NCATS",
    "CYP1A2_Veith",
    "CYP2C19_Veith",
    "CYP2C9_Veith",
    "CYP2D6_Veith",
    "CYP3A4_Veith",
    "CYP2C9_Substrate_CarbonMangels",
    "CYP2D6_Substrate_CarbonMangels",
    "CYP3A4_Substrate_CarbonMangels",
]
TDC_TOX = ["AMES", "DILI", "hERG", "hERG_Karim"]
TDC_HTS = ["SARSCoV2_3CLPro_Diamond", "SARSCoV2_Vitro_Touret"]
MOLNET = [
    "ogbg-molbace",
    "ogbg-molbbbp",
    "ogbg-molclintox",
    "ogbg-molhiv",
    "ogbg-molmuv",
    "ogbg-molsider",
    "ogbg-moltox21",
]
GROUPS = {
    **{t: "TDC-ADME" for t in TDC_ADME},
    **{t: "TDC-Tox" for t in TDC_TOX},
    **{t: "TDC-HTS" for t in TDC_HTS},
    **{t: "MoleculeNet" for t in MOLNET},
}
TASKS_25 = TDC_ADME + TDC_TOX + TDC_HTS + MOLNET
assert len(TASKS_25) == 25

# ---- Model name map: paper label -> embedder key in source CSV ----
MODELS = {
    "ECFP4": "ECFP",
    "ChemBERTa-2": "ChemBERTa-77M-MLM",
    "SELFormer": "SELFormer",
    "MoLFormer": "MoLFormer-XL-both-10pct",
    "MMB-small": "modernmolbert_best_standard",
    "MMB-base": "modernmolbert_best_base",
    "MMB-small-span": "modernmolbert_best_span",
    "MMB-small-hetero": "modernmolbert_best_hetero_span",
}

df = pd.read_csv(SRC)
df = df[df["test_metric_name"] == "roc_auc"]

# pivot: rows tasks, cols embedder
pivot = df.pivot_table(index="dataset", columns="embedder", values="test_metric", aggfunc="first")
matrix = pd.DataFrame(index=TASKS_25)
for label, key in MODELS.items():
    matrix[label] = pivot[key].reindex(TASKS_25) if key in pivot.columns else np.nan
matrix.insert(0, "group", [GROUPS[t] for t in TASKS_25])
matrix.to_csv(OUT / "results_matrix_25task.csv")

# ---- Missing cells ----
missing = {m: matrix.index[matrix[m].isna()].tolist() for m in MODELS if matrix[m].isna().any()}

# ---- Group means + overall (per model, over available tasks) ----
group_order = ["TDC-ADME", "TDC-Tox", "TDC-HTS", "MoleculeNet"]
rows = []
for label in MODELS:
    rec = {"model": label}
    for g in group_order:
        sub = matrix.loc[matrix["group"] == g, label]
        rec[g] = sub.mean()
        rec[g + "_n"] = sub.notna().sum()
    rec["Overall"] = matrix[label].mean()
    rec["Overall_n"] = matrix[label].notna().sum()
    rows.append(rec)
gm = pd.DataFrame(rows).set_index("model")
gm.to_csv(OUT / "group_means.csv")

# ---- LaTeX Table 2 (×100, 1 decimal; bold best per column) ----
table_models = ["ECFP4", "ChemBERTa-2", "SELFormer", "MoLFormer", "MMB-small", "MMB-base"]
disp_name = {
    "ECFP4": "ECFP4",
    "ChemBERTa-2": "ChemBERTa-2 (MLM)",
    "SELFormer": "SELFormer",
    "MoLFormer": "MoLFormer",
    "MMB-small": r"\textbf{\model{}-small}",
    "MMB-base": r"\textbf{\model{}-base}",
}
cols = group_order + ["Overall"]
best = {c: gm.loc[table_models, c].max() for c in cols}


def fmt(label, c):
    v = gm.loc[label, c]
    if pd.isna(v):
        return "--"
    s = f"{v * 100:.1f}"
    if abs(v - best[c]) < 1e-9:
        s = r"\textbf{" + s + "}"
    return s


lines = [
    r"\begin{table}[htbp]",
    r"  \centering",
    r"  \small",
    r"  \begin{tabularx}{\linewidth}{l r r r r r}",
    r"    \toprule",
    r"    \textbf{Model} & \textbf{TDC-ADME} & \textbf{TDC-Tox} & "
    r"\textbf{TDC-HTS} & \textbf{MoleculeNet} & \textbf{Overall} \\",
    r"    \midrule",
]
for label in ["ECFP4", "ChemBERTa-2", "SELFormer", "MoLFormer"]:
    lines.append(
        "    " + disp_name[label] + " & " + " & ".join(fmt(label, c) for c in cols) + r" \\"
    )
lines.append(r"    \midrule")
for label in ["MMB-small", "MMB-base"]:
    lines.append(
        "    " + disp_name[label] + " & " + " & ".join(fmt(label, c) for c in cols) + r" \\"
    )
lines += [
    r"    \bottomrule",
    r"  \end{tabularx}",
    r"  \caption{%",
    r"    Mean ROC-AUC ($\times100$) on the 25-task benchmark of "
    r"\citet{praskiBenchmarkingPretrainedMolecular2025}, broken down by task",
    r"    group. Each entry averages per-task ROC-AUC using the best "
    r"cross-validated downstream head (ridge / random forest / $k$NN) per task.",
    r"    \emph{Overall} is the unweighted mean across all 25 tasks. "
    r"\textbf{Bold} marks the best value per column.",
    r"  }%",
    r"  \label{tab:main-results}",
    r"\end{table}",
]
(OUT / "table2.tex").write_text("\n".join(lines) + "\n")


# ---- Stats: Wilcoxon best-MMB vs ECFP4 and vs SELFormer; ECFP4 counts ----
def headline():
    # headline = best overall among the two released models
    return (
        "MMB-base"
        if gm.loc["MMB-base", "Overall"] >= gm.loc["MMB-small", "Overall"]
        else "MMB-small"
    )


out = []
hl = headline()
out.append(f"Headline released model (higher overall): {hl}\n")
out.append("Overall mean ROC-AUC (x100), n tasks:\n")
for label in MODELS:
    out.append(
        f"  {label:18s} {gm.loc[label, 'Overall'] * 100:5.1f}  (n={int(gm.loc[label, 'Overall_n'])})\n"
    )
out.append("\nGroup means (x100):\n")
out.append(gm[[*group_order, "Overall"]].mul(100).round(1).to_string() + "\n")


# paired comparisons on common tasks
def paired(a, b):
    s = matrix[[a, b]].dropna()
    return s[a].values, s[b].values, s.index.tolist()


for comp in ["ECFP4", "SELFormer"]:
    a, b, idx = paired(hl, comp)
    diff = a - b
    nz = diff[diff != 0]
    stat, p = wilcoxon(a, b) if len(nz) else (np.nan, np.nan)
    wins = int((diff > 0).sum())
    out.append(
        f"\n{hl} vs {comp} (n={len(idx)} common tasks): "
        f"{hl} wins {wins}, ties {(diff == 0).sum()}, losses {(diff < 0).sum()}; "
        f"Wilcoxon W={stat}, p={p:.4g}; mean diff={diff.mean() * 100:.2f}\n"
    )
    big = [i for i, d in zip(idx, diff, strict=False) if d > 0.02]
    out.append(f"  tasks where {hl} exceeds {comp} by >0.02: {len(big)} -> {big}\n")

# vs ECFP4 detailed win count for prose (best released model)
a, b, idx = paired(hl, "ECFP4")
diff = a - b
out.append(
    f"\nProse (ECFP4): on {int((diff > 0).sum())} of {len(idx)} tasks "
    f"{hl} exceeds ECFP4; margin>0.02 on {int((diff > 0.02).sum())}.\n"
)

# four-model internal (small variants) on common tasks
out.append("\nFour-model internal (mean ROC-AUC over common tasks):\n")
for pair in [
    ("MMB-small", "MMB-base"),
    ("MMB-small", "MMB-small-span"),
    ("MMB-small", "MMB-small-hetero"),
]:
    a, b, idx = paired(*pair)
    out.append(
        f"  {pair[0]} vs {pair[1]} (n={len(idx)}): "
        f"{a.mean() * 100:.1f} vs {b.mean() * 100:.1f}  "
        f"(diff {(a.mean() - b.mean()) * 100:+.2f})\n"
    )

out.append("\nMissing cells (model -> tasks with no result):\n")
for m, ts in missing.items():
    out.append(f"  {m}: {ts}\n")

# best-head distribution for released models
out.append("\nBest downstream head distribution (released models):\n")
for key in ["modernmolbert_best_standard", "modernmolbert_best_base"]:
    sub = df[(df["embedder"] == key) & (df["dataset"].isin(TASKS_25))]
    out.append(f"  {key}: {sub['model'].value_counts().to_dict()}\n")

(OUT / "stats.txt").write_text("".join(out))
print("".join(out))
print(f"\nWrote outputs to {OUT}")
