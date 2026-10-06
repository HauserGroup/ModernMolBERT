#!/usr/bin/env python3
"""Build the supplementary coverage tables from verified five-seed source data."""

import argparse
import csv
from pathlib import Path


SPLIT_LABELS = {
    "TDC ADMET group train_val/test split": "TDC ADMET",
    "Murcko scaffold split (create_tdc_scaffold_split); valid merged into train": "TDC scaffold",
    "OGB scaffold split": "OGB scaffold",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def latex(value: str) -> str:
    """Escape source labels without changing their spelling."""
    return "".join(
        {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_\allowbreak{}",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }.get(char, char)
        for char in value
    )


def validate(datasets: list[dict[str, str]], endpoints: list[dict[str, str]]) -> None:
    names = [row["dataset"] for row in datasets]
    if len(names) != 25 or len(set(names)) != 25:
        raise ValueError("Expected exactly 25 unique benchmark datasets")
    if len(endpoints) != 79:
        raise ValueError("Expected exactly 79 benchmark endpoints")
    endpoint_names = [(row["dataset"], row["endpoint"]) for row in endpoints]
    if len(set(endpoint_names)) != len(endpoint_names):
        raise ValueError("Duplicate dataset/endpoint pair")
    if {row["dataset"] for row in endpoints} != set(names):
        raise ValueError("Endpoint datasets do not match dataset summary")
    for row in datasets:
        if row["split_rule"] not in SPLIT_LABELS:
            raise ValueError(f"Unknown split rule for {row['dataset']}")
        for split in ("train", "valid", "test"):
            raw = int(row[f"raw_{split}"])
            retained = int(row[f"common_{split}"])
            excluded = int(row[f"excluded_{split}"])
            if raw != retained + excluded:
                raise ValueError(f"Split accounting mismatch for {row['dataset']} {split}")
        matching = [item for item in endpoints if item["dataset"] == row["dataset"]]
        if len(matching) != int(row["n_test_endpoints"]):
            raise ValueError(f"Endpoint count mismatch for {row['dataset']}")
        scored = sum(item["roc_auc_defined"].lower() == "true" for item in matching)
        if scored != int(row["n_scored_test_endpoints"]):
            raise ValueError(f"Scored endpoint count mismatch for {row['dataset']}")
        if sum(int(item["common_test_missing"]) for item in matching) != (
            int(row["common_test"]) * len(matching) - int(row["common_test_observed_cells"])
        ):
            raise ValueError(f"Missing-label count mismatch for {row['dataset']}")


def split_triplet(row: dict[str, str], prefix: str) -> str:
    return "/".join(row[f"{prefix}_{split}"] for split in ("train", "valid", "test"))


def render_dataset_table(rows: list[dict[str, str]]) -> str:
    lines = [
        r"\begin{landscape}",
        r"{\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{longtable}{@{}p{5.0cm}p{2.9cm}rrrp{1.6cm}@{}}",
        r"\caption{Prepared and shared-coverage benchmark cohorts. Raw and retained "
        r"counts are train/validation/test row triplets; excluded rows are their "
        r"difference. TDC ADMET is the TDC group train\_val/test split; TDC "
        r"scaffold is the Murcko scaffold split with validation merged into "
        r"training; OGB scaffold is the supplied OGB split. The shared cohort "
        r"is the intersection across all 25 trained encoders. Unassigned prepared "
        r"rows (22 for hERG) are omitted from all three splits. The final column "
        r"is the number of endpoints with both observed test classes divided "
        r"by the total endpoints. Failed embeddings cause the excluded-row "
        r"counts; endpoint missing labels do not remove a row.}\label{tab:revision-datasets}\\",
        r"\toprule",
        r"Dataset & Split rule & Raw T/V/Test & Retained T/V/Test & Excluded T/V/Test & Scored/total \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{6}{l}{\emph{Continued from previous page}}\\",
        r"\toprule",
        r"Dataset & Split rule & Raw T/V/Test & Retained T/V/Test & Excluded T/V/Test & Scored/total \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for row in rows:
        lines.append(
            f"{latex(row['dataset'])} & {SPLIT_LABELS[row['split_rule']]} & "
            f"{split_triplet(row, 'raw')} & {split_triplet(row, 'common')} & "
            f"{split_triplet(row, 'excluded')} & "
            f"{row['n_scored_test_endpoints']}/{row['n_test_endpoints']} " + r"\\"
        )
    lines += [r"\end{longtable}", r"}", r"\end{landscape}", ""]
    return "\n".join(lines)


def render_endpoint_table(rows: list[dict[str, str]]) -> str:
    lines = [
        r"\begin{landscape}",
        r"{\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{longtable}{@{}p{5.0cm}p{8.0cm}rrrrl@{}}",
        r"\caption{Endpoint-level common test coverage. Rows are named as in the "
        r"prepared benchmark files. Retained is the common test-row count; "
        r"observed, missing and positive are label-cell counts on those rows. "
        r"ROC-AUC is defined only where observed labels contain both classes. "
        r"The production head fit treats missing labels as negative to match "
        r"the imported benchmark protocol; these columns retain the original "
        r"missingness so its extent is visible.}\label{tab:revision-endpoints}\\",
        r"\toprule",
        r"Dataset & Endpoint & Retained & Observed & Missing & Positive & ROC-AUC \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{7}{l}{\emph{Continued from previous page}}\\",
        r"\toprule",
        r"Dataset & Endpoint & Retained & Observed & Missing & Positive & ROC-AUC \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for row in rows:
        status = "yes" if row["roc_auc_defined"].lower() == "true" else "no"
        lines.append(
            f"{latex(row['dataset'])} & {latex(row['endpoint'])} & "
            f"{row['common_test_rows']} & {row['common_test_observed']} & "
            f"{row['common_test_missing']} & {row['common_test_positive']} & {status} " + r"\\"
        )
    lines += [r"\end{longtable}", r"}", r"\end{landscape}", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-coverage", type=Path, required=True)
    parser.add_argument("--endpoint-coverage", type=Path, required=True)
    parser.add_argument("--dataset-tex", type=Path, required=True)
    parser.add_argument("--endpoint-tex", type=Path, required=True)
    args = parser.parse_args()
    datasets = read_rows(args.dataset_coverage)
    endpoints = read_rows(args.endpoint_coverage)
    validate(datasets, endpoints)
    for path, content in (
        (args.dataset_tex, render_dataset_table(datasets)),
        (args.endpoint_tex, render_endpoint_table(endpoints)),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    print(f"Wrote {len(datasets)} datasets and {len(endpoints)} endpoints")


if __name__ == "__main__":
    main()
