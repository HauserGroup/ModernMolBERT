#!/usr/bin/env python3
"""
build_paper_results.py

Derive paper-facing benchmark numbers from the native and matched score matrices.

Two inputs are supported:
- ``--task-matrix``: ``task_matrix.csv`` from ``build_common_row_benchmark.py``
  (CV-selected heads; baselines from the imported table, columns already
  labelled). For the five-model revision also pass ``--common-task-matrix``
  so internal scores use matched test rows; external table baselines remain
  descriptive scores on their own unverified molecule sets.
- default: the archived ``outputs/eval/best_metric_by_dataset_embedder.csv``
  (one row per dataset x embedder) with the released checkpoints' labels.

Produces:
    outputs/eval/paper/results_matrix_25task.csv   (tasks x models, ROC-AUC; name kept for compatibility)
  outputs/eval/paper/group_means.csv             (model x task-group means + overall)
  outputs/eval/paper/table2.tex                  (main benchmark LaTeX table)
  outputs/eval/paper/stats.txt                   (Wilcoxon tests + prose counts)

No model runs, no new benchmarking. Pure wrangling of existing eval output.

Main-analysis dataset exclusions:
- ogbg-moltoxcast  (26th MoleculeNet set; no Praski baseline)

The hetero-span ablation (MMB-small-hetero) is exploratory and supplementary-only:
it is written to the results matrix (for the appendix per-task table) but left out
of group means, Table 2 and all stats unless --include-hetero-span is passed.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

parser = argparse.ArgumentParser(description="Build paper-facing benchmark tables and stats.")
parser.add_argument(
    "--include-hetero-span",
    action="store_true",
    help="Also include MMB-small-hetero (hetero_span masking) in group means and stats.",
)
parser.add_argument(
    "--task-matrix",
    type=Path,
    default=None,
    help="task_matrix.csv from build_common_row_benchmark.py (columns are model labels).",
)
parser.add_argument(
    "--common-task-matrix",
    type=Path,
    default=None,
    help="Matched internal scores from common_task_matrix.csv; requires --task-matrix.",
)
parser.add_argument(
    "--seed-aggregate-manifest",
    type=Path,
    default=None,
    help="Verify that --common-task-matrix is the recorded five-seed mean matrix.",
)
parser.add_argument(
    "--reference",
    default=None,
    help="Headline ModernMolBERT label for the stats (required with --task-matrix).",
)
parser.add_argument("--out-dir", type=Path, default=None, help="Output directory.")
ARGS = parser.parse_args()
if ARGS.task_matrix is not None and ARGS.reference is None:
    parser.error("--reference is required with --task-matrix")
if ARGS.common_task_matrix is not None and ARGS.task_matrix is None:
    parser.error("--common-task-matrix requires --task-matrix")
if ARGS.seed_aggregate_manifest is not None and ARGS.common_task_matrix is None:
    parser.error("--seed-aggregate-manifest requires --common-task-matrix")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "outputs/eval/best_metric_by_dataset_embedder.csv"
OUT = ARGS.out_dir or ROOT / "outputs/eval/paper"
OUT.mkdir(parents=True, exist_ok=True)

# ---- Task -> group map for manuscript main analysis ----
# Canonical benchmark = 25 datasets (18 TDC + 7 MoleculeNet). Only ogbg-moltoxcast
# (a 26th MoleculeNet set) is excluded: no Praski baseline exists for it.
EXCLUDED_DATASETS = {"ogbg-moltoxcast"}

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
TASKS_MAIN = TDC_ADME + TDC_TOX + TDC_HTS + MOLNET
BASELINES = ["ECFP4", "ChemBERTa-2", "SELFormer", "MoLFormer"]
N_TASKS_MAIN = len(TASKS_MAIN)
REVISION_INTERNAL = [
    "MMB-small-APE-SELFIES",
    "MMB-small-APE-SMILES",
    "MMB-small-BPE-SELFIES",
    "MMB-small-BPE-SMILES",
    "MMB-base-APE-SELFIES",
]
aggregate_provenance = None
if ARGS.seed_aggregate_manifest is not None:
    aggregate_provenance = json.loads(ARGS.seed_aggregate_manifest.read_text(encoding="utf-8"))
    expected_hash = aggregate_provenance.get("output_sha256", {}).get("mean_common_task_matrix.csv")
    if (
        ARGS.common_task_matrix.name != "mean_common_task_matrix.csv"
        or aggregate_provenance.get("seeds") != [42, 43, 44, 45, 46]
        or aggregate_provenance.get("models") != REVISION_INTERNAL
        or aggregate_provenance.get("task_count") != 25
        or expected_hash != file_sha256(ARGS.common_task_matrix)
    ):
        raise ValueError("Common matrix does not match the verified five-seed aggregate")

# ---- Model name map: paper label -> embedder key in source CSV ----
MODELS = {
    "ECFP4": "ECFP",
    "ChemBERTa-2": "ChemBERTa-77M-MLM",
    "SELFormer": "SELFormer",
    "MoLFormer": "MoLFormer-XL-both-10pct",
    "MMB-small": "modernmolbert_best_standard",
    "MMB-base": "modernmolbert_best_base",
    "MMB-small-span": "modernmolbert_best_span",
}
# Supplementary-only models: kept in the results matrix for the appendix per-task
# table, but not used for any aggregate or claim unless explicitly requested.
SUPPLEMENTARY_MODELS = {"MMB-small-hetero": "modernmolbert_best_hetero_span"}
if ARGS.include_hetero_span:
    MODELS.update(SUPPLEMENTARY_MODELS)

df: pd.DataFrame | None = None
if ARGS.task_matrix is not None:
    source = pd.read_csv(ARGS.task_matrix, index_col=0)
    source = source.loc[~source.index.isin(list(EXCLUDED_DATASETS))]
    if absent := [m for m in [*BASELINES, ARGS.reference] if m not in source.columns]:
        raise ValueError(f"{ARGS.task_matrix} lacks model columns {absent}; check --matrix-labels")
    if extra := sorted(set(source.index) - set(TASKS_MAIN)):
        raise ValueError(f"{ARGS.task_matrix} has datasets outside the benchmark: {extra}")
    ordered = [*BASELINES, *[c for c in source.columns if c not in BASELINES]]
    internal = [c for c in ordered if c not in BASELINES]
    if len(internal) > 1 and ARGS.common_task_matrix is None:
        raise ValueError("Multiple internal models require --common-task-matrix")
    if ARGS.common_task_matrix is not None:
        common = pd.read_csv(ARGS.common_task_matrix, index_col=0)
        if set(common.index) != set(source.index) or set(common.columns) != set(internal):
            raise ValueError("Common matrix must contain the same tasks and all internal models")
        source = source.copy()
        source.loc[:, internal] = common.loc[source.index, internal].to_numpy(dtype=float)
    MODELS = {label: label for label in ordered}
    SUPPLEMENTARY_MODELS = {}
    matrix = pd.DataFrame(index=TASKS_MAIN)
    for label in ordered:
        matrix[label] = pd.Series(source[label], dtype=float).reindex(TASKS_MAIN)
else:
    results = pd.read_csv(SRC)
    if (
        "selection_metric" not in results
        or not results["selection_metric"].eq("cv_metric").to_numpy().all()
    ):
        raise ValueError(
            "Regenerate source results using CV head selection before building paper tables"
        )
    results = results.loc[
        results["test_metric_name"].eq("roc_auc")
        & ~results["dataset"].isin(list(EXCLUDED_DATASETS))
    ].copy()
    df = results

    # pivot: rows tasks, cols embedder
    pivot = results.pivot(index="dataset", columns="embedder", values="test_metric")
    matrix = pd.DataFrame(index=TASKS_MAIN)
    for label, key in {**MODELS, **SUPPLEMENTARY_MODELS}.items():
        matrix[label] = pivot[key].reindex(TASKS_MAIN) if key in pivot.columns else np.nan
matrix.insert(0, "group", [GROUPS[t] for t in TASKS_MAIN])
matrix.to_csv(OUT / "results_matrix_25task.csv")
if ARGS.task_matrix is not None:
    (OUT / "results_matrix_provenance.json").write_text(
        json.dumps(
            {
                "external_baselines": {
                    "columns": BASELINES,
                    "source": str(ARGS.task_matrix),
                    "population": "table-only baseline test rows; molecule identities unverified",
                },
                "internal_models": {
                    "columns": [m for m in MODELS if m not in BASELINES],
                    "source": str(ARGS.common_task_matrix or ARGS.task_matrix),
                    "population": (
                        "five-seed mean of five-model common test-row scores"
                        if aggregate_provenance is not None
                        else "five-model common test rows"
                        if ARGS.common_task_matrix is not None
                        else "native verified test rows"
                    ),
                    "aggregate_manifest": (
                        str(ARGS.seed_aggregate_manifest)
                        if ARGS.seed_aggregate_manifest is not None
                        else None
                    ),
                    "aggregate_manifest_sha256": (
                        file_sha256(ARGS.seed_aggregate_manifest)
                        if ARGS.seed_aggregate_manifest is not None
                        else None
                    ),
                },
            },
            indent=2,
        )
        + "\n"
    )

# ---- Missing cells ----
missing = {
    m: matrix.index[matrix[m].isna()].tolist() for m in MODELS if matrix[m].isna().to_numpy().any()
}

# ---- Group means + overall (per model, over available tasks) ----
group_order = ["TDC-ADME", "TDC-Tox", "TDC-HTS", "MoleculeNet"]
rows = []
for label in MODELS:
    rec: dict[str, Any] = {"model": label}
    for g in group_order:
        sub = pd.Series(matrix.loc[matrix["group"] == g, label], dtype=float)
        rec[g] = sub.mean()
        rec[g + "_n"] = sub.count()
    overall = pd.Series(matrix[label], dtype=float)
    rec["Overall"] = overall.mean()
    rec["Overall_n"] = overall.count()
    rows.append(rec)
gm = pd.DataFrame(rows).set_index("model")
gm.to_csv(OUT / "group_means.csv")

# ---- LaTeX Table 2 (×100, 1 decimal; bold best per column) ----
mmb_models = (
    [m for m in MODELS if m not in BASELINES]
    if ARGS.task_matrix is not None
    else ["MMB-small", "MMB-base"]
)
table_models = [*BASELINES, *mmb_models]


def display_name(label: str) -> str:
    if label == "ChemBERTa-2":
        return "ChemBERTa-2 (MLM)"
    if label.startswith("MMB-"):
        return r"\textbf{MMB-" + label.removeprefix("MMB-") + "}"
    return label


cols = group_order + ["Overall"]
best = {c: gm.loc[table_models, c].max() for c in cols}


def scalar_float(value: object) -> float:
    return float(np.asarray(pd.to_numeric([value], errors="raise"), dtype=np.float64)[0])


def scalar_int(value: object) -> int:
    return int(np.asarray(pd.to_numeric([value], errors="raise"), dtype=np.int64)[0])


def fmt(label, c):
    v = scalar_float(gm.loc[label, c])
    if pd.isna(v):
        return "--"
    n = scalar_int(gm.loc[label, c + "_n"])
    s = rf"{v * 100:.1f}\,({n})"
    if ARGS.common_task_matrix is None and abs(v - best[c]) < 1e-9:
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
for label in BASELINES:
    lines.append(
        "    " + display_name(label) + " & " + " & ".join(fmt(label, c) for c in cols) + r" \\"
    )
lines.append(r"    \midrule")
for label in mmb_models:
    lines.append(
        "    " + display_name(label) + " & " + " & ".join(fmt(label, c) for c in cols) + r" \\"
    )
lines += [
    r"    \bottomrule",
    r"  \end{tabularx}",
    r"  \caption{%",
    rf"    Mean ROC-AUC ($\times100$) on the {N_TASKS_MAIN}-task benchmark of "
    r"\citet{praskiBenchmarkingPretrainedMolecular2025}, broken down by task",
    r"    group. MMB = ModernMolBERT. Each entry averages per-task ROC-AUC. "
    + (
        r"Internal downstream heads (logistic regression, random forest or $k$NN) "
        r"are selected by training-side cross-validation. The imported baseline "
        r"table selected among head families using test scores."
        if ARGS.common_task_matrix is not None
        else r"Downstream heads are selected by the archived analysis protocol."
    ),
    r"    \emph{Overall} is the unweighted mean across available tasks; "
    r"parentheses give the number of scored tasks for each cell. "
    + (
        r"Internal models use five-model common test rows."
        + (
            r" Scores are means across five training seeds."
            if aggregate_provenance is not None
            else ""
        )
        + r" External table baselines use their own unverified test molecules. "
        r"Cross-group differences are descriptive."
        if ARGS.common_task_matrix is not None
        else r"\textbf{Bold} marks the best value per column."
    ),
    r"  }%",
    r"  \label{tab:main-results}",
    r"\end{table}",
]
(OUT / "table2.tex").write_text("\n".join(lines) + "\n")


# ---- Stats: Wilcoxon best-MMB vs ECFP4 and vs SELFormer; ECFP4 counts ----
def headline():
    # headline = best overall among the two released models
    base_overall = scalar_float(gm.loc["MMB-base", "Overall"])
    small_overall = scalar_float(gm.loc["MMB-small", "Overall"])
    return "MMB-base" if base_overall >= small_overall else "MMB-small"


out = []
hl = ARGS.reference or headline()
out.append(f"Headline model: {hl}\n")
out.append("Overall mean ROC-AUC (x100), n tasks:\n")
for label in MODELS:
    overall = scalar_float(gm.loc[label, "Overall"])
    overall_n = scalar_int(gm.loc[label, "Overall_n"])
    out.append(f"  {label:18s} {overall * 100:5.1f}  (n={overall_n})\n")
out.append("\nGroup means (x100):\n")
out.append(gm[[*group_order, "Overall"]].mul(100).round(1).to_string() + "\n")


# paired comparisons on common tasks
def paired(a, b):
    s = matrix[[a, b]].dropna()
    return (
        np.asarray(s[a], dtype=np.float64),
        np.asarray(s[b], dtype=np.float64),
        s.index.tolist(),
    )


for comp in ["ECFP4", "SELFormer"]:
    a, b, idx = paired(hl, comp)
    diff = a - b
    nz = diff[diff != 0]
    stat, p = wilcoxon(a, b) if len(nz) and ARGS.common_task_matrix is None else (np.nan, np.nan)
    wins = int((diff > 0).sum())
    out.append(
        f"\n{hl} vs {comp} (n={len(idx)} common tasks): "
        f"{hl} wins {wins}, ties {(diff == 0).sum()}, losses {(diff < 0).sum()}; "
        + (
            f"Wilcoxon W={stat}, p={p:.4g}; "
            if ARGS.common_task_matrix is None
            else "descriptive task-level comparison, unmatched test molecules; "
        )
        + f"mean diff={diff.mean() * 100:.2f}\n"
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

# internal comparisons (size and masking variants) on common tasks
internal_pairs = (
    [
        ("MMB-small-APE-SELFIES", "MMB-small-BPE-SELFIES"),
        ("MMB-small-APE-SMILES", "MMB-small-BPE-SMILES"),
        ("MMB-small-APE-SELFIES", "MMB-small-APE-SMILES"),
        ("MMB-small-BPE-SELFIES", "MMB-small-BPE-SMILES"),
        ("MMB-small-APE-SELFIES", "MMB-base-APE-SELFIES"),
    ]
    if set(REVISION_INTERNAL) <= set(matrix.columns)
    else [("MMB-small", "MMB-base"), ("MMB-small", "MMB-small-span")]
)
if ARGS.include_hetero_span and ARGS.common_task_matrix is None:
    internal_pairs.append(("MMB-small", "MMB-small-hetero"))
out.append("\nInternal comparisons (mean ROC-AUC over common tasks):\n")
for pair in internal_pairs:
    if not set(pair) <= set(matrix.columns):
        continue
    a, b, idx = paired(*pair)
    out.append(
        f"  {pair[0]} vs {pair[1]} (n={len(idx)}): "
        f"{a.mean() * 100:.1f} vs {b.mean() * 100:.1f}  "
        f"(diff {(a.mean() - b.mean()) * 100:+.2f})\n"
    )

out.append("\nMissing cells (model -> tasks with no result):\n")
for m, ts in missing.items():
    out.append(f"  {m}: {ts}\n")

# best-head distribution for released models (task-matrix runs: see selected_heads.csv)
if df is not None:
    out.append("\nBest downstream head distribution (released models):\n")
    for key in ["modernmolbert_best_standard", "modernmolbert_best_base"]:
        sub = df[(df["embedder"] == key) & (df["dataset"].isin(TASKS_MAIN))]
        out.append(f"  {key}: {pd.Series(sub['model']).value_counts().to_dict()}\n")

(OUT / "stats.txt").write_text("".join(out))
print("".join(out))
print(f"\nWrote outputs to {OUT}")
