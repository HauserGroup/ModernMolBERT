#!/usr/bin/env python3
"""Check which head-selection rule reproduces the published per-task table.

For every cell of the preprint's per-task ROC-AUC table, this collects the
archived candidate heads for that dataset x model and asks whether the printed
value (x100, one decimal) equals the head with the highest *test* ROC-AUC, the
head with the highest training-side *CV* ROC-AUC, both, or neither.

Candidate identities follow the historical pipeline: ModernMolBERT rows take
their embedder from the ``praski_best_*`` result directory, as the preprint-era
``normalize_own_result`` did, and baseline rows come from the imported Praski
CSV. This is a provenance audit of archived files; it does not produce
corrected scores.

Usage:
    uv run python scripts/paper/audit_published_head_selection.py \\
        --table paper/tables/pertask_table.tex \\
        --output outputs/audit/published_head_selection_audit.csv
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from build_benchmark_results_frames import infer_embedder_from_result_path

TABLE_COLUMNS = {
    "ECFP4": "ECFP",
    "ChBa-2": "ChemBERTa-77M-MLM",
    "SELF.": "SELFormer",
    "MoLF.": "MoLFormer-XL-both-10pct",
    "MMB-s": "modernmolbert_best_standard",
    "MMB-b": "modernmolbert_best_base",
    "MMB-sp": "modernmolbert_best_span",
    "MMB-h": "modernmolbert_best_hetero_span",
}
TABLE_TASKS = {
    "Bioavailability": "Bioavailability_Ma",
    "HIA": "HIA_Hou",
    "Pgp": "Pgp_Broccatelli",
    "PAMPA": "PAMPA_NCATS",
    "CYP1A2": "CYP1A2_Veith",
    "CYP2C19": "CYP2C19_Veith",
    "CYP2C9": "CYP2C9_Veith",
    "CYP2D6": "CYP2D6_Veith",
    "CYP3A4": "CYP3A4_Veith",
    "CYP2C9 (substrate)": "CYP2C9_Substrate_CarbonMangels",
    "CYP2D6 (substrate)": "CYP2D6_Substrate_CarbonMangels",
    "CYP3A4 (substrate)": "CYP3A4_Substrate_CarbonMangels",
    "AMES": "AMES",
    "DILI": "DILI",
    "hERG": "hERG",
    "hERG (Karim)": "hERG_Karim",
    "SARS-CoV-2 3CLPro": "SARSCoV2_3CLPro_Diamond",
    "SARS-CoV-2 (Vitro)": "SARSCoV2_Vitro_Touret",
    "BACE": "ogbg-molbace",
    "BBBP": "ogbg-molbbbp",
    "ClinTox": "ogbg-molclintox",
    "HIV": "ogbg-molhiv",
    "MUV": "ogbg-molmuv",
    "SIDER": "ogbg-molsider",
    "Tox21": "ogbg-moltox21",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check which head-selection rule reproduces the published per-task table."
    )
    parser.add_argument("--table", type=Path, default=Path("paper/tables/pertask_table.tex"))
    parser.add_argument(
        "--praski-csv",
        type=Path,
        default=Path("data/Praski_benchmarking_results/arxiv_preprint_2025_08.csv"),
    )
    parser.add_argument("--results-root", type=Path, default=Path("outputs/eval"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def parse_pertask_table(path: Path) -> pd.DataFrame:
    """Return one row per printed cell: dataset, table column, printed value."""
    header: list[str] | None = None
    cells = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if header is None and line.startswith(r"\textbf{Task}"):
            header = re.findall(r"\\textbf\{([^}]*)\}", line)[1:]
            continue
        if header is None or "&" not in line or line.startswith(r"\textbf{Task}"):
            continue
        parts = [part.strip() for part in line.removesuffix(r"\\").split("&")]
        task, values = parts[0], parts[1:]
        if task not in TABLE_TASKS or len(values) != len(header):
            continue
        for column, value in zip(header, values, strict=True):
            cells.append({"dataset": TABLE_TASKS[task], "table_column": column, "published": value})
    if header is None:
        raise ValueError(f"No per-task header found in {path}")
    return pd.DataFrame(cells)


def load_candidates(praski_csv: Path, results_root: Path) -> pd.DataFrame:
    frames = [pd.read_csv(praski_csv).assign(result_source=str(praski_csv))]
    for path in sorted(results_root.glob("praski_best_*/results.csv")):
        frame = pd.read_csv(path)
        frame["recorded_embedder"] = frame["embedder"]
        frame["embedder"] = infer_embedder_from_result_path(path)
        frames.append(frame.assign(result_source=str(path)))
    candidates = pd.concat(frames, ignore_index=True)
    candidates = candidates.loc[candidates["test_metric_name"].eq("roc_auc")]
    for column in ("test_metric", "cv_metric"):
        candidates[column] = pd.to_numeric(candidates[column], errors="coerce")
    return candidates.loc[np.isfinite(candidates["test_metric"])]


def _printed(value: float) -> str:
    return f"{value * 100:.1f}"


def audit_cell(published: str, cands: pd.DataFrame) -> dict[str, object]:
    record: dict[str, object] = {"n_candidates": len(cands)}
    if published == "--" or cands.empty:
        return record | {"rule": "missing"}
    test = cands["test_metric"].to_numpy(dtype=float)
    cv = cands["cv_metric"].to_numpy(dtype=float)
    test_max = float(test.max())
    has_cv = np.isfinite(cv).any()
    cv_rows = cands.loc[cv == np.nanmax(cv)] if has_cv else cands.iloc[0:0]
    cv_values = sorted({_printed(v) for v in cv_rows["test_metric"]})
    matches_test = _printed(test_max) == published
    matches_cv = published in cv_values
    if matches_test and matches_cv:
        rule = "both"
    elif matches_test:
        rule = "test_max_only"
    elif matches_cv:
        rule = "cv_selected_only"
    else:
        rule = "neither"
    return record | {
        "test_max_printed": _printed(test_max),
        "test_max_head": ";".join(sorted(set(cands.loc[test == test_max, "model"]))),
        "cv_selected_printed": ";".join(cv_values),
        "cv_selected_head": ";".join(sorted(set(cv_rows["model"]))),
        "n_cv_selected_runs": len(cv_rows),
        "any_candidate_matches": published in {_printed(v) for v in test},
        "rule": rule,
    }


def audit(table: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cell in table.to_dict("records"):
        embedder = TABLE_COLUMNS[str(cell["table_column"])]
        cands = candidates.loc[
            candidates["dataset"].eq(cell["dataset"]) & candidates["embedder"].eq(embedder)
        ]
        rows.append(cell | {"embedder": embedder} | audit_cell(str(cell["published"]), cands))
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = audit(
        parse_pertask_table(args.table), load_candidates(args.praski_csv, args.results_root)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Audited {len(result)} published cells; wrote {args.output}")
    print(result["rule"].value_counts().to_string())
    print()
    print(pd.crosstab(result["table_column"], result["rule"]).to_string())


if __name__ == "__main__":
    main()
