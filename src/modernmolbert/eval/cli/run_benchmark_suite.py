import argparse
from pathlib import Path
import shutil
import sys

from modernmolbert.eval.suite import load_suite_config, run_benchmark_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a config-driven frozen-representation benchmark suite.",
    )

    parser.add_argument(
        "--suite",
        type=Path,
        required=True,
        help="Path to a benchmark suite config file, usually YAML.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where suite outputs will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=None,
        help=(
            "Optional feature cache directory. Defaults to <output_dir>/cache. "
            "Use this to share cached features across suite runs."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output_dir before running if it already exists.",
    )
    parser.add_argument(
        "--write_single_run_outputs",
        action="store_true",
        help=(
            "Also write per-run outputs under <output_dir>/runs/. "
            "Useful for debugging, but can create many files."
        ),
    )

    return parser.parse_args()


def validate_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    """Validate or prepare the output directory."""

    if output_dir.exists() and overwrite:
        if output_dir.resolve() == Path.cwd().resolve():
            raise ValueError("Refusing to overwrite the current working directory.")

        if output_dir.resolve() == output_dir.anchor:
            raise ValueError(f"Refusing to overwrite filesystem root: {output_dir}")

        shutil.rmtree(output_dir)
        return

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory already exists and is not empty: {output_dir}\n"
            "Pass --overwrite to remove it before running, or choose a new output_dir."
        )


def main() -> None:
    args = parse_args()

    try:
        validate_output_dir(args.output_dir, overwrite=args.overwrite)

        suite = load_suite_config(args.suite)

        print(f"Running benchmark suite: {suite.name}", flush=True)
        print(f"Suite config: {args.suite}", flush=True)
        print(f"Output directory: {args.output_dir}", flush=True)
        if args.cache_dir is not None:
            print(f"Feature cache directory: {args.cache_dir}", flush=True)
        else:
            print(f"Feature cache directory: {args.output_dir / 'cache'}", flush=True)

        results = run_benchmark_suite(
            suite=suite,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            write_single_run_outputs=args.write_single_run_outputs,
        )

        print("Done.", flush=True)
        print(f"Result rows: {len(results)}", flush=True)
        print(f"Results CSV: {args.output_dir / 'results.csv'}", flush=True)
        print(f"Manifest: {args.output_dir / 'manifest.json'}", flush=True)

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
