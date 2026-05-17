import argparse
from pathlib import Path
import sys

from modernmolbert.eval.reporting import create_benchmark_deliverables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create static benchmark tables and figures from a full sweep directory.",
    )
    parser.add_argument(
        "--sweep_dir",
        type=Path,
        required=True,
        help="Sweep directory containing results.csv and optional sweep artifacts.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <sweep_dir>/deliverables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.sweep_dir / "deliverables"

    try:
        manifest = create_benchmark_deliverables(
            sweep_dir=args.sweep_dir,
            output_dir=output_dir,
        )
        print("Done.", flush=True)
        print(f"Deliverables: {output_dir}", flush=True)
        print(f"Tables: {len(manifest['tables'])}", flush=True)
        print(f"Figures: {len(manifest['figures'])}", flush=True)
        if manifest["warnings"]:
            print("Warnings:", flush=True)
            for warning in manifest["warnings"]:
                print(f"- {warning}", flush=True)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
