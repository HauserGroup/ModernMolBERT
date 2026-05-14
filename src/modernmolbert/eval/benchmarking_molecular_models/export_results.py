from __future__ import annotations

import argparse
from pathlib import Path

from modernmolbert.eval.benchmarking_molecular_models.praski_export import (
    export_classification_reports,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export benchmark SQLite results to the public CSV schema.",
    )
    parser.add_argument("--database", type=Path, default=Path("data/meta.db"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/classificationreport.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = export_classification_reports(
        database_path=args.database,
        output_csv=args.output_csv,
    )
    print(f"Wrote {len(frame)} rows to {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
