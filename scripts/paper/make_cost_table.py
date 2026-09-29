#!/usr/bin/env python3
"""Write the feature-extraction cost table from ``measure_embedding_cost.py`` output.

Reads ``summary.csv`` and ``manifest.json`` from one measurement run and writes
a LaTeX table: encoder parameters, weight-file size, feature dimension,
median throughput on CPU and MPS, and peak resident memory of the CPU run.
The caption takes the machine, software, molecule count and repeats from the
manifest, so the table cannot drift from the measurement it reports.

Usage:
    uv run python scripts/paper/make_cost_table.py \\
        --input-dir outputs/audit/embedding_cost \\
        --output paper/tables/extraction_cost_table.tex
"""

import argparse
import json
from pathlib import Path

import pandas as pd

LABELS = {
    "ECFP4-binary": "ECFP4, binary",
    "ECFP4-count": "ECFP4, count",
    "ModernMolBERT-small": r"\model{}-small",
    "ModernMolBERT-base": r"\model{}-base",
    "ChemBERTa-2": "ChemBERTa-2 (77M-MLM)",
    "MoLFormer": "MoLFormer-XL",
    "SELFormer": "SELFormer",
}
CPU_BATCH = 32
MPS_BATCHES = (32, 128)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--exclude", nargs="*", default=[], help="Configuration names to leave out."
    )
    return parser.parse_args(argv)


def rate(summary: pd.DataFrame, name: str, device: str, batch_size: int | None) -> str:
    rows = summary.loc[(summary["name"] == name) & (summary["device"] == device)]
    if batch_size is not None:
        rows = rows.loc[rows["batch_size"] == batch_size]
    if rows.empty:
        return "---"
    return f"{rows['median_molecules_per_second'].iloc[0]:,.0f}"


def table_rows(summary: pd.DataFrame, manifest: dict, exclude: list[str]) -> list[str]:
    lines = []
    for name in dict.fromkeys(summary["name"]):
        if name in exclude:
            continue
        rows = summary.loc[summary["name"] == name]
        first = rows.iloc[0]
        fingerprint = first["kind"] == "fingerprint"
        cpu_batch = None if fingerprint else CPU_BATCH
        cpu = rows.loc[rows["device"] == "cpu"]
        if not fingerprint:
            cpu = cpu.loc[cpu["batch_size"] == CPU_BATCH]
        peak_gb = cpu["max_peak_rss_bytes"].iloc[0] / 1e9 if not cpu.empty else None
        record = manifest["checkpoints"].get(name) or manifest.get("transformers", {}).get(name, {})
        weights = record.get("weights_bytes")
        cells = [
            LABELS.get(name, name),
            "---" if fingerprint else f"{first['n_parameters'] / 1e6:.1f}",
            "---" if weights is None else f"{weights / 1e6:.1f}",
            f"{int(first['feature_dim']):,}",
            rate(summary, name, "cpu", cpu_batch),
            *("---" if fingerprint else rate(summary, name, "mps", batch) for batch in MPS_BATCHES),
            "---" if peak_gb is None else f"{peak_gb:.2f}",
        ]
        lines.append("    " + " & ".join(cells) + r" \\")
    return lines


def caption(
    summary: pd.DataFrame, manifest: dict, measurements: pd.DataFrame, exclude: list[str]
) -> str:
    hardware = manifest["hardware"]
    software = manifest["software"]
    molecules = manifest["molecules"]
    threads = "/".join(str(int(n)) for n in sorted(measurements["torch_threads"].dropna().unique()))
    load = measurements["load_average_before"]
    encoders = summary.loc[(summary["kind"] != "fingerprint") & ~summary["name"].isin(exclude)]
    valid = sorted(int(n) for n in encoders["n_valid"].unique())
    embedded = (
        "--".join(f"{n:,}" for n in (valid[0], valid[-1])) if len(valid) > 1 else f"{valid[0]:,}"
    )
    return (
        "Feature-extraction cost on one workstation "
        f"({hardware['processor']}, {hardware['physical_cores']} CPU cores, "
        f"{hardware['memory_bytes'] / 2**30:.0f}\\,GB; PyTorch {software['torch']}) "
        f"for {molecules['n_sampled']:,} distinct benchmark test molecules. "
        f"Throughput is the median of {manifest['repeats']} interleaved repeats after "
        "one untimed warm-up pass, and counts every input molecule. It includes SMILES "
        "parsing for ECFP4 and, for \\model{}, SMILES-to-\\selfies{} conversion, "
        "tokenisation, the encoder forward pass and mean pooling, with inputs capped at "
        f"{manifest['max_seq_length']} tokens; \\model{{}} embedded {embedded} of the "
        "molecules and its featuriser rejected the rest as lossy or unsupported inputs. "
        f"CPU runs used batch size {CPU_BATCH} and {threads or 'default'} PyTorch threads; "
        "the MPS columns give batch sizes 32 and 128 on the Apple GPU. \\emph{Params}: "
        "encoder parameters used for embedding, excluding the MLM head; \\emph{Weights}: "
        "size of the model weight file; \\emph{Peak RAM}: peak resident memory of a fresh "
        "CPU process. ECFP4 uses RDKit Morgan fingerprints (radius "
        f"{manifest['ecfp']['radius']}, {manifest['ecfp']['bits']:,} bits) in one process. "
        "Other applications were running (one-minute load average "
        f"{load.min():.0f}--{load.max():.0f}); absolute rates depend on the machine and "
        "its load."
    )


def render(
    summary: pd.DataFrame, manifest: dict, measurements: pd.DataFrame, exclude: list[str]
) -> str:
    header = [
        "% Generated by scripts/paper/make_cost_table.py from measure_embedding_cost.py output.",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{@{} l r r r r r r r @{}}",
        r"    \toprule",
        r"    & & & & \multicolumn{3}{c}{\textbf{Molecules per second}} & \\",
        r"    \cmidrule(lr){5-7}",
        r"    \textbf{Representation} & \textbf{Params (M)} & \textbf{Weights (MB)}"
        r" & \textbf{Dim.} & \textbf{CPU} & \textbf{MPS, 32} & \textbf{MPS, 128}"
        r" & \textbf{Peak RAM (GB)} \\",
        r"    \midrule",
    ]
    footer = [
        r"    \bottomrule",
        r"  \end{tabular}",
        f"  \\caption{{{caption(summary, manifest, measurements, exclude)}}}",
        r"  \label{tab:extraction-cost}",
        r"\end{table}",
    ]
    return "\n".join([*header, *table_rows(summary, manifest, exclude), *footer]) + "\n"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = pd.read_csv(args.input_dir / "summary.csv")
    manifest = json.loads((args.input_dir / "manifest.json").read_text())
    measurements = pd.read_csv(args.input_dir / "measurements.csv")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(summary, manifest, measurements, args.exclude))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
