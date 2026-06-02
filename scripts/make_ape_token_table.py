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

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

VOCAB_PATH = (
    ROOT
    / "runs/chembl36_small_mask_mlm_lr_sweep"
    / "modernmolbert_best_standard/final_model/ape_tokenizer/vocab.json"
)
PARQUET_PATH = ROOT / "outputs/visualize/best_span_100k/metadata.parquet"
OUT_DIR = ROOT / "outputs/eval/paper"

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


# ── main ────────────────────────────────────────────────────────────────────


def build_table(
    vocab_path: Path = VOCAB_PATH,
    parquet_path: Path = PARQUET_PATH,
    out_dir: Path = OUT_DIR,
    top: int = 30,
) -> pd.DataFrame:
    """Full pipeline: load → count → save CSV and LaTeX.  Returns the full sorted DataFrame."""
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = load_merged_tokens(vocab_path)
    selfies = pd.read_parquet(parquet_path, columns=["selfies"])["selfies"]

    df = count_token_frequencies(merged, selfies)

    df.to_csv(out_dir / "ape_token_freq.csv", index=False)
    emit_latex(df, out_dir / "table_ape_tokens.tex", top=top)

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
    args = parser.parse_args()
    build_table(args.vocab, args.parquet, args.out_dir, args.top)


if __name__ == "__main__":
    main()
