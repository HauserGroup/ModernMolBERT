#!/usr/bin/env python3
"""
make_ape_token_table.py
=======================

Generate a supplementary table of the most frequent APE merged tokens in the
100k ChEMBL SELFIES sample, suitable for Appendix B (Tokenizer Details) of the
ModernMolBERT paper.

Overview
--------
APE (Atom Pair Encoding) training merges adjacent SELFIES primitive pairs whose
co-occurrence frequency exceeds a threshold (min_freq=3000 here).  The resulting
vocabulary contains both single-primitive tokens (e.g. ``[C]``, ``[O]``) and
two-primitive merged tokens (e.g. ``[C][C]``, ``[C][=O]``).

This script:

1. Loads the released APE vocabulary (631 tokens) and identifies the 256
   two-primitive merged tokens by counting ``[…]`` groups.
2. Counts substring occurrences of each merged token in 100 000 ChEMBL SELFIES
   strings (``outputs/visualize/best_span_100k/metadata.parquet``, column
   ``selfies``).  This serves as a corpus-frequency proxy — pair counts were
   not persisted during tokenizer training.
3. Emits a CSV (``outputs/eval/paper/ape_token_freq.csv``) and a LaTeX table
   (``outputs/eval/paper/table_ape_tokens.tex``) of the top-N tokens by
   frequency, with a ``chemical_fragment`` column left for manual annotation.

Inputs
------
- ``runs/chembl36_small_mask_mlm_lr_sweep/modernmolbert_best_standard/
  final_model/ape_tokenizer/vocab.json``          (631-token vocabulary)
- ``outputs/visualize/best_span_100k/metadata.parquet``  (100k SELFIES sample)

Outputs
-------
- ``outputs/eval/paper/ape_token_freq.csv``
- ``outputs/eval/paper/table_ape_tokens.tex``

Usage
-----
    uv run python scripts/make_ape_token_table.py            # default top 30
    uv run python scripts/make_ape_token_table.py --top 50
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[2]

VOCAB_PATH = (
    ROOT
    / "runs/chembl36_small_mask_mlm_lr_sweep"
    / "modernmolbert_best_standard/final_model/ape_tokenizer/vocab.json"
)
PARQUET_PATH = ROOT / "outputs/visualize/best_span_100k/metadata.parquet"
OUT_DIR = ROOT / "outputs/eval/paper"
FIGURE_DIR = OUT_DIR / "figures"

BREWER_PUBUGN = ["#F6EFF7", "#D0D1E6", "#A6BDDB", "#67A9CF", "#1C9099", "#016C59"]
TEXT_COLOR = "#2B2B2B"
GRID_COLOR = "#D9D9D9"

# ── primitive counting ──────────────────────────────────────────────────────

_BRACKET_RE = re.compile(r"\[[^\]]+\]")


def count_primitives(token: str) -> int:
    """Return the number of ``[…]`` groups in *token*."""
    return len(_BRACKET_RE.findall(token))


def load_merged_tokens(vocab_path: Path) -> dict[str, int]:
    """Return ``{token: vocab_id}`` for every two-primitive merged token."""
    vocab: dict[str, int] = json.loads(vocab_path.read_text(encoding="utf-8"))
    return {tok: idx for tok, idx in vocab.items() if count_primitives(tok) >= 2}


# ── frequency counting ──────────────────────────────────────────────────────


def count_token_frequencies(
    merged_tokens: dict[str, int],
    selfies_series: pd.Series,
) -> pd.DataFrame:
    """Count substring occurrences of each merged token across *selfies_series*.

    Joins all SELFIES strings with a space separator before counting so that a
    token cannot straddle two molecules.  Returns a DataFrame sorted by
    descending frequency with columns ``token``, ``vocab_id``, ``count``.
    """
    corpus = " ".join(selfies_series.tolist())
    rows = []
    for tok, vid in merged_tokens.items():
        rows.append({"token": tok, "vocab_id": vid, "count": corpus.count(tok)})
    df = pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)
    return df


# ── LaTeX emission ──────────────────────────────────────────────────────────

_LATEXSAFE = str.maketrans({"[": r"\texttt{[}", "]": r"]}", "_": r"\_", "&": r"\&"})


def _safe(tok: str) -> str:
    """Escape a SELFIES token for LaTeX verbatim-style rendering."""
    return r"\texttt{" + tok.replace("[", r"\textbf{[").replace("]", r"]}") + "}"


def emit_latex(df: pd.DataFrame, out_path: Path, top: int = 30) -> None:
    """Write a longtable LaTeX fragment for the top-*top* tokens."""
    top_df = df.head(top).copy()
    # Placeholder annotation column — intended for manual fill-in
    if "chemical_fragment" not in top_df.columns:
        top_df["chemical_fragment"] = ""

    lines = [
        r"{\footnotesize\setlength{\tabcolsep}{4pt}",
        r"\begin{longtable}{l r p{6.5cm}}",
        r"  \caption{%",
        r"    Top "
        + str(top)
        + r" highest-frequency APE merged tokens in 100\,000 ChEMBL~36 \selfies{} strings.",
        r"    Each token is a two-primitive merge produced during APE vocabulary training",
        r"    (min\_freq\,$=3000$, max\_merge\_pieces\,$=2$).  Frequency is the substring",
        r"    occurrence count in the 100k corpus and serves as a proxy for co-occurrence",
        r"    frequency at tokenizer training time.  The \emph{Fragment interpretation}",
        r"    column gives an approximate chemical reading; note that APE merges are",
        r"    sequence-adjacent (not necessarily graph-adjacent), so bracket concatenation",
        r"    does not guarantee a covalent bond between the two primitives.",
        r"  }\label{tab:ape-tokens}\\",
        r"  \toprule",
        r"  \textbf{Token} & \textbf{Frequency} & \textbf{Fragment interpretation} \\",
        r"  \midrule",
        r"  \endfirsthead",
        r"  \toprule",
        r"  \textbf{Token} & \textbf{Frequency} & \textbf{Fragment interpretation} \\",
        r"  \midrule",
        r"  \endhead",
    ]
    for _, row in top_df.iterrows():
        tok_tex = r"\texttt{" + row["token"] + "}"
        frag = row["chemical_fragment"] if row["chemical_fragment"] else r"\emph{(to annotate)}"
        lines.append(f"  {tok_tex} & {int(row['count']):,} & {frag} \\\\")
    lines += [
        r"  \bottomrule",
        r"\end{longtable}}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _compact_count(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}k"
    return str(value)


def _apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "semibold",
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.edgecolor": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def emit_ape_token_frequency_plot(df: pd.DataFrame, out_dir: Path, top: int = 20) -> None:
    """Write a horizontal bar plot of the most frequent APE merged tokens."""
    _apply_paper_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_df = df.head(top).sort_values("count", ascending=True).reset_index(drop=True)
    y = list(range(len(plot_df)))

    fig_height = max(4.0, 0.28 * len(plot_df) + 1.1)
    fig, ax = plt.subplots(figsize=(6.8, fig_height), constrained_layout=True)
    cmap = LinearSegmentedColormap.from_list("brewer_pubugn", BREWER_PUBUGN)
    if len(plot_df) > 1:
        rank_colors = np.linspace(0.45, 0.95, len(plot_df))
    else:
        rank_colors = np.array([0.75])
    colors = [cmap(value) for value in rank_colors]
    for i in y:
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#F7F7F7", zorder=0)
    bars = ax.barh(y, plot_df["count"], color=colors, edgecolor="white", linewidth=0.7, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["token"].tolist(), fontfamily="monospace", fontsize=8)
    ax.set_xlabel("Occurrences in 100k ChEMBL SELFIES strings")
    ax.set_title("Highest-frequency APE merged tokens")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: _compact_count(int(value))))
    ax.grid(axis="x", color=GRID_COLOR, lw=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    max_count = int(plot_df["count"].max()) if len(plot_df) else 0
    label_pad = max(max_count * 0.015, 1)
    for bar, count in zip(bars, plot_df["count"], strict=False):
        count = int(count)
        ax.text(
            count + label_pad,
            bar.get_y() + bar.get_height() / 2,
            _compact_count(count),
            va="center",
            ha="left",
            fontsize=8,
            color=TEXT_COLOR,
        )

    ax.set_xlim(0, max_count * 1.14 if max_count else 1)
    ax.text(
        0,
        -0.12,
        "APE merges are sequence-adjacent SELFIES tokens; graph adjacency is not implied.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color="#525252",
    )
    fig.savefig(out_dir / "ape_token_frequency_top20.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "ape_token_frequency_top20.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ── main ────────────────────────────────────────────────────────────────────


def build_table(
    vocab_path: Path = VOCAB_PATH,
    parquet_path: Path = PARQUET_PATH,
    out_dir: Path = OUT_DIR,
    top: int = 30,
    make_figures: bool = True,
    figure_dir: Path = FIGURE_DIR,
    figure_top: int = 20,
) -> pd.DataFrame:
    """Full pipeline: load → count → save CSV and LaTeX.  Returns the full sorted DataFrame."""
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = load_merged_tokens(vocab_path)
    selfies = pd.read_parquet(parquet_path, columns=["selfies"])["selfies"]

    df = count_token_frequencies(merged, selfies)

    df.to_csv(out_dir / "ape_token_freq.csv", index=False)
    emit_latex(df, out_dir / "table_ape_tokens.tex", top=top)
    if make_figures:
        emit_ape_token_frequency_plot(df, figure_dir, top=figure_top)

    print(f"Merged tokens: {len(merged)} / {len(merged) + (631 - 256 - 5)} vocab entries")
    print(f"Top {top} by frequency:")
    print(df.head(top).to_string(index=False))
    print(f"\nWrote outputs to {out_dir}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the APE merged-token frequency table for paper outputs.",
    )
    parser.add_argument("--top", type=int, default=30, help="Number of rows in the LaTeX table")
    parser.add_argument("--vocab", type=Path, default=VOCAB_PATH)
    parser.add_argument("--parquet", type=Path, default=PARQUET_PATH)
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--figure_dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--figure_top", type=int, default=20)
    parser.add_argument("--no_figures", action="store_true")
    args = parser.parse_args()
    build_table(
        args.vocab,
        args.parquet,
        args.out_dir,
        args.top,
        make_figures=not args.no_figures,
        figure_dir=args.figure_dir,
        figure_top=args.figure_top,
    )


if __name__ == "__main__":
    main()
