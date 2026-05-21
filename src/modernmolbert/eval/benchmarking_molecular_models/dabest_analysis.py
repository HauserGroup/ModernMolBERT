"""Wrangle benchmark results for dabest repeated-measures paired analysis.

Each dataset is the paired subject (id_col), each embedder is a condition (x),
and the per-head test scores are averaged across scoring heads (rf/ridge/knn).

Dabest repeated-measures paired usage
--------------------------------------

    import dabest
    from modernmolbert.eval.benchmarking_molecular_models.dabest_analysis import wrangle

    df = wrangle("data/benchmark/arxiv_preprint_2025_08.csv")

    analysis = dabest.load(
        df,
        idx=("molbert", "unimolv2"),     # (control, treatment) or tuple of many
        x="embedder",
        y="test_metric",
        id_col="dataset",
        paired="baseline",               # or "sequential"
    )
    analysis.mean_diff.plot()

Run as a script to save the wrangled CSV:

    uv run python -m modernmolbert.eval.benchmarking_molecular_models.dabest_analysis \\
      --input  data/benchmark/arxiv_preprint_2025_08.csv \\
      --output data/benchmark/arxiv_preprint_2025_08_dabest.csv

See: https://acclab.github.io/dabestr/articles/tutorial_repeated_measures.html

# Notes
Works. 1425 rows = 25 datasets × 57 embedders, one score per pair (mean of heads).

Usage:


uv run python src/modernmolbert/eval/benchmarking_molecular_models/dabest_analysis.py \
  --input data/benchmark/arxiv_preprint_2025_08.csv \
  --output data/benchmark/arxiv_preprint_2025_08_dabest.csv
Or in a notebook:


import dabest
from modernmolbert.eval.benchmarking_molecular_models.dabest_analysis import wrangle

df = wrangle("data/benchmark/arxiv_preprint_2025_08.csv")

# Compare one or more embedders against a baseline
analysis = dabest.load(
    df,
    idx=("molbert", "unimolv2", "ChemBERTa-77M-MLM"),
    x="embedder",
    y="test_metric",
    id_col="dataset",
    paired="baseline",
)
analysis.mean_diff.plot()
wrangle() also accepts embedders=[...] and datasets=[...] to restrict the comparison, and agg="max" to use best-head instead of mean-head.
"""

import argparse
from pathlib import Path

import pandas as pd


def wrangle(
    csv_path: str | Path,
    *,
    agg: str = "mean",
    embedders: list[str] | None = None,
    datasets: list[str] | None = None,
    require_complete: bool = True,
) -> pd.DataFrame:
    """Load and wrangle benchmark results into dabest long format.

    Parameters
    ----------
    csv_path:
        Path to the benchmark results CSV (standard schema with columns
        dataset, embedder, model, test_metric).
    agg:
        How to aggregate test_metric across scoring heads per (dataset, embedder).
        'mean' (default) or 'max'.
    embedders:
        Optional subset of embedder names to keep. None = all.
    datasets:
        Optional subset of dataset names to keep. None = all.
    require_complete:
        If True (default), drop datasets that are missing any of the selected
        embedders. Dabest paired analysis requires every id to appear in every
        group.

    Returns
    -------
    DataFrame with columns: dataset, embedder, test_metric
    """
    df = pd.read_csv(csv_path)

    if embedders is not None:
        df = df[df["embedder"].isin(embedders)]

    if datasets is not None:
        df = df[df["dataset"].isin(datasets)]

    # Aggregate across scoring heads (rf/ridge/knn) per dataset/embedder pair.
    long = df.groupby(["dataset", "embedder"], sort=False)["test_metric"].agg(agg).reset_index()

    if require_complete:
        all_embedders = long["embedder"].unique()
        counts = long.groupby("dataset")["embedder"].nunique()
        complete = counts[counts == len(all_embedders)].index
        n_dropped = long["dataset"].nunique() - len(complete)
        if n_dropped > 0:
            print(
                f"[wrangle] dropped {n_dropped} dataset(s) with incomplete embedder coverage",
                flush=True,
            )
        long = long[long["dataset"].isin(complete)]

    return long.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wrangle benchmark results CSV into dabest long format."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/benchmark/arxiv_preprint_2025_08.csv"),
        help="Input benchmark results CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <input stem>_dabest.csv.",
    )
    parser.add_argument(
        "--agg",
        choices=["mean", "max"],
        default="mean",
        help="Aggregation across scoring heads (default: mean).",
    )
    parser.add_argument(
        "--embedders",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Restrict to these embedders. Default: all.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Restrict to these datasets. Default: all.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    long = wrangle(
        args.input,
        agg=args.agg,
        embedders=args.embedders,
        datasets=args.datasets,
    )

    output = args.output or args.input.with_name(args.input.stem + "_dabest.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(output, index=False)

    n_datasets = long["dataset"].nunique()
    n_embedders = long["embedder"].nunique()

    print(f"Wrangled: {len(long)} rows, {n_datasets} datasets, {n_embedders} embedders")
    print(f"Output:   {output}")
    print()
    print("Dabest usage:")
    print("    import dabest")
    print("    import pandas as pd")
    print()
    print(f'    df = pd.read_csv("{output}")')
    print("    analysis = dabest.load(")
    print("        df,")
    embedder_list = long["embedder"].unique().tolist()
    idx_example = tuple(embedder_list[:2]) if len(embedder_list) >= 2 else tuple(embedder_list)
    print(f"        idx={idx_example},")
    print('        x="embedder",')
    print('        y="test_metric",')
    print('        id_col="dataset",')
    print('        paired="baseline",')
    print("    )")
    print("    analysis.mean_diff.plot()")


if __name__ == "__main__":
    main()
