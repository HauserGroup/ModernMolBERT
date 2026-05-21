import argparse
from pathlib import Path
import sys

from modernmolbert.eval.reporting import write_standard_plots, write_summary_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize and plot benchmark suite results.",
    )

    parser.add_argument(
        "--results_csv",
        type=Path,
        required=True,
        help="Path to benchmark results.csv.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where report tables and plots will be written.",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Only write summary tables, not plots.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)

        tables_dir = args.output_dir / "tables"
        plots_dir = args.output_dir / "plots"

        table_outputs = write_summary_tables(
            results_path=args.results_csv,
            output_dir=tables_dir,
        )

        print(f"Wrote {len(table_outputs)} summary table(s) to {tables_dir}", flush=True)

        if not args.no_plots:
            plot_outputs = write_standard_plots(
                results_path=args.results_csv,
                output_dir=plots_dir,
            )
            print(f"Wrote {len(plot_outputs)} plot(s) to {plots_dir}", flush=True)

        print("Done.", flush=True)

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
