#!/usr/bin/env python3
"""Per-task full ROC-AUC table (Appendix C / S3) from the main-analysis matrix.

Main-analysis exclusions:
- ogbg-moltox21
- ogbg-molmuv
- ogbg-moltoxcast

The exploratory hetero-span variant (MMB-small-hetero) is kept here as a
supplementary column, flagged as exploratory and not used for any claim; pass
--exclude-hetero-span to drop it.
"""

import argparse
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser(description="Write the per-task appendix ROC-AUC table.")
parser.add_argument(
    "--exclude-hetero-span",
    action="store_true",
    help="Drop the exploratory MMB-small-hetero (hetero_span masking) column.",
)
ARGS = parser.parse_args()

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "outputs/eval/paper/results_matrix_25task.csv"
OUT = ROOT / "outputs/eval/paper/table_pertask.tex"
EXCLUDED_DATASETS = {"ogbg-moltox21", "ogbg-molmuv", "ogbg-moltoxcast"}

PRETTY = {
    "Bioavailability_Ma": "Bioavailability",
    "HIA_Hou": "HIA",
    "Pgp_Broccatelli": "Pgp",
    "PAMPA_NCATS": "PAMPA",
    "CYP1A2_Veith": "CYP1A2",
    "CYP2C19_Veith": "CYP2C19",
    "CYP2C9_Veith": "CYP2C9",
    "CYP2D6_Veith": "CYP2D6",
    "CYP3A4_Veith": "CYP3A4",
    "CYP2C9_Substrate_CarbonMangels": "CYP2C9 (substrate)",
    "CYP2D6_Substrate_CarbonMangels": "CYP2D6 (substrate)",
    "CYP3A4_Substrate_CarbonMangels": "CYP3A4 (substrate)",
    "AMES": "AMES",
    "DILI": "DILI",
    "hERG": "hERG",
    "hERG_Karim": "hERG (Karim)",
    "SARSCoV2_3CLPro_Diamond": "SARS-CoV-2 3CLPro",
    "SARSCoV2_Vitro_Touret": "SARS-CoV-2 (Vitro)",
    "ogbg-molbace": "BACE",
    "ogbg-molbbbp": "BBBP",
    "ogbg-molclintox": "ClinTox",
    "ogbg-molhiv": "HIV",
    "ogbg-molsider": "SIDER",
}
GROUP_LABEL = {
    "TDC-ADME": "TDC -- ADME",
    "TDC-Tox": "TDC -- Toxicity",
    "TDC-HTS": "TDC -- HTS",
    "MoleculeNet": "MoleculeNet",
}
COLS = [
    "ECFP4",
    "ChemBERTa-2",
    "SELFormer",
    "MoLFormer",
    "MMB-small",
    "MMB-base",
    "MMB-small-span",
]
HEAD = ["ECFP4", "ChBa-2", "SELF.", "MoLF.", "MMB-s", "MMB-b", "MMB-sp"]
LEGEND = (
    r"\emph{MMB-s} = \model{}-small (standard), "
    r"\emph{MMB-b} = \model{}-base, \emph{MMB-sp} = small span masking"
)
if not ARGS.exclude_hetero_span:
    COLS.append("MMB-small-hetero")
    HEAD.append("MMB-h")
    LEGEND += (
        r", \emph{MMB-h} = exploratory small hetero-span masking variant "
        r"(reported in preprint v1; single seed, not used for any claim)"
    )

df = pd.read_csv(MATRIX, index_col=0)
df = df.loc[~df.index.isin(EXCLUDED_DATASETS)].copy()
missing_cols = [c for c in COLS if c not in df.columns]
if missing_cols:
    raise ValueError(
        f"{MATRIX} is missing columns {missing_cols}; rebuild it with the current "
        "build_paper_results.py (or pass --exclude-hetero-span)."
    )


def cell(v):
    return "--" if pd.isna(v) else f"{v * 100:.1f}"


lines = [
    r"{\footnotesize\setlength{\tabcolsep}{3.5pt}",
    r"\begin{longtable}{l " + "r " * len(COLS) + "}",
    r"  \caption{Per-task test ROC-AUC ($\times100$) for all models on the "
    r"25-task benchmark. " + LEGEND + r". ``--'' marks evaluations not "
    r"yet run.}\label{tab:pertask}\\",
    r"  \toprule",
    r"  \textbf{Task} & " + " & ".join(r"\textbf{" + h + "}" for h in HEAD) + r" \\",
    r"  \midrule",
    r"  \endfirsthead",
    r"  \toprule",
    r"  \textbf{Task} & " + " & ".join(r"\textbf{" + h + "}" for h in HEAD) + r" \\",
    r"  \midrule",
    r"  \endhead",
]
for g in ["TDC-ADME", "TDC-Tox", "TDC-HTS", "MoleculeNet"]:
    lines.append(
        r"  \multicolumn{" + str(len(COLS) + 1) + r"}{l}{\textit{" + GROUP_LABEL[g] + r"}} \\"
    )
    for t in df.index[df["group"] == g]:
        row = " & ".join(cell(df.loc[t, c]) for c in COLS)
        lines.append("  " + PRETTY.get(t, t) + " & " + row + r" \\")  # type: ignore
    lines.append(r"  \addlinespace")
lines += [r"  \bottomrule", r"\end{longtable}", r"}"]
OUT.write_text("\n".join(lines) + "\n")
print("Wrote", OUT)
