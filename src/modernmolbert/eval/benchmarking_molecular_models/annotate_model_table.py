"""
CLI wrapper around model_annotations.annotate_csv_or_tsv.

Usage:
    uv run python -m modernmolbert.eval.benchmarking_molecular_models.annotate_model_table \\
        --input data/Praski_benchmarking_results/arxiv_preprint_2025_08.csv \\
        --output outputs/eval/annotated_arxiv_preprint_2025_08.csv

Delimiter is auto-detected from file extension (.tsv = tab, otherwise comma).
Model column is auto-detected (looks for 'embedder' or 'Model'); override with --model-col.
"""

import argparse
from pathlib import Path

from modernmolbert.eval.benchmarking_molecular_models.model_annotations import annotate_csv_or_tsv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate a benchmark result CSV/TSV with model family and class metadata."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input CSV or TSV path.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV or TSV path.")
    parser.add_argument(
        "--model-col",
        type=str,
        default=None,
        help="Column containing model/embedder names. Auto-detected if not specified.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotate_csv_or_tsv(
        input_path=args.input,
        output_path=args.output,
        model_col=args.model_col,
    )
    print(f"Annotated: {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
