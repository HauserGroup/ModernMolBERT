"""Tests for scripts/paper/make_ape_token_table.py.

All tests are self-contained (no large files, no network). They use small
in-memory fixtures to exercise every function in the module.
"""

import json
from pathlib import Path

import pandas as pd

from make_ape_token_table import (
    build_table,
    count_primitives,
    emit_ape_token_frequency_plot,
    count_token_frequencies,
    emit_latex,
    load_merged_tokens,
)


# ── fixtures ────────────────────────────────────────────────────────────────


def _tiny_vocab() -> dict[str, int]:
    """Minimal vocab with 5 specials, 3 single-prim tokens, 4 merged tokens."""
    return {
        "<s>": 0,
        "<pad>": 1,
        "</s>": 2,
        "<unk>": 3,
        "<mask>": 4,
        "[C]": 5,
        "[O]": 6,
        "[N]": 7,
        "[C][C]": 8,
        "[C][O]": 9,
        "[N][C]": 10,
        "[O][=C]": 11,
    }


def _tiny_selfies() -> pd.Series:
    return pd.Series(
        [
            "[C][C][O][C][C]",
            "[C][O][C][N][C]",
            "[O][=C][N][C][C]",
            "[C][C][C][C]",
        ]
    )


# ── count_primitives ────────────────────────────────────────────────────────


def test_count_primitives_single():
    assert count_primitives("[C]") == 1
    assert count_primitives("[O]") == 1
    assert count_primitives("[=Branch1]") == 1


def test_count_primitives_merged_two():
    assert count_primitives("[C][C]") == 2
    assert count_primitives("[C][=O]") == 2
    assert count_primitives("[Ring1][=Branch1]") == 2


def test_count_primitives_special_tokens_are_single():
    # Special tokens like <s> have no brackets; count_primitives returns 0.
    assert count_primitives("<s>") == 0
    assert count_primitives("<mask>") == 0


# ── load_merged_tokens ──────────────────────────────────────────────────────


def test_load_merged_tokens_returns_only_merged(tmp_path: Path):
    vocab_file = tmp_path / "vocab.json"
    vocab_file.write_text(json.dumps(_tiny_vocab()), encoding="utf-8")
    merged = load_merged_tokens(vocab_file)
    assert set(merged.keys()) == {"[C][C]", "[C][O]", "[N][C]", "[O][=C]"}


def test_load_merged_tokens_preserves_vocab_ids(tmp_path: Path):
    vocab_file = tmp_path / "vocab.json"
    vocab_file.write_text(json.dumps(_tiny_vocab()), encoding="utf-8")
    merged = load_merged_tokens(vocab_file)
    assert merged["[C][C]"] == 8
    assert merged["[O][=C]"] == 11


def test_load_merged_tokens_excludes_specials_and_singles(tmp_path: Path):
    vocab_file = tmp_path / "vocab.json"
    vocab_file.write_text(json.dumps(_tiny_vocab()), encoding="utf-8")
    merged = load_merged_tokens(vocab_file)
    for tok in ("<s>", "<pad>", "[C]", "[O]", "[N]"):
        assert tok not in merged


# ── count_token_frequencies ─────────────────────────────────────────────────


def test_count_frequencies_returns_all_merged_tokens(tmp_path: Path):
    vocab_file = tmp_path / "vocab.json"
    vocab_file.write_text(json.dumps(_tiny_vocab()), encoding="utf-8")
    merged = load_merged_tokens(vocab_file)
    df = count_token_frequencies(merged, _tiny_selfies())
    assert set(df["token"]) == set(merged.keys())


def test_count_frequencies_sorted_descending(tmp_path: Path):
    vocab_file = tmp_path / "vocab.json"
    vocab_file.write_text(json.dumps(_tiny_vocab()), encoding="utf-8")
    merged = load_merged_tokens(vocab_file)
    df = count_token_frequencies(merged, _tiny_selfies())
    counts = df["count"].tolist()
    assert counts == sorted(counts, reverse=True)


def test_count_frequencies_correct_counts(tmp_path: Path):
    """Verify [C][C] substring count against manual trace of the fixture corpus."""
    vocab_file = tmp_path / "vocab.json"
    vocab_file.write_text(json.dumps(_tiny_vocab()), encoding="utf-8")
    merged = load_merged_tokens(vocab_file)
    df = count_token_frequencies(merged, _tiny_selfies())
    cc_count = int(df.loc[df["token"] == "[C][C]", "count"].iloc[0])
    # Space-joined corpus:
    #   "[C][C][O][C][C] [C][O][C][N][C] [O][=C][N][C][C] [C][C][C][C]"
    # Hits: [C][C] at mol0 pos 0, mol0 pos 9, mol2 end, mol3 pos 0,1,2 → 5 total
    # (Python str.count does NOT skip overlapping matches, so [C][C][C][C] → 3 hits)
    assert cc_count == 5


def test_count_frequencies_no_cross_molecule_matches(tmp_path: Path):
    """A token split across the molecule boundary must not be counted."""
    vocab = {
        "<s>": 0,
        "<pad>": 1,
        "</s>": 2,
        "<unk>": 3,
        "<mask>": 4,
        "[C]": 5,
        "[N]": 6,
        "[C][N]": 7,
    }
    vocab_file = tmp_path / "vocab.json"
    vocab_file.write_text(json.dumps(vocab), encoding="utf-8")
    merged = load_merged_tokens(vocab_file)
    # Two molecules: last token of mol1 = [C], first of mol2 = [N].
    # Space-joined: "[C] [N]" — the space prevents a [C][N] match across the boundary.
    series = pd.Series(["[C]", "[N]"])
    df = count_token_frequencies(merged, series)
    cn_count = int(df.loc[df["token"] == "[C][N]", "count"].iloc[0])
    assert cn_count == 0


# ── emit_latex ───────────────────────────────────────────────────────────────


def _make_freq_df() -> pd.DataFrame:
    rows = [
        {"token": "[C][C]", "vocab_id": 8, "count": 100},
        {"token": "[C][O]", "vocab_id": 9, "count": 80},
        {"token": "[N][C]", "vocab_id": 10, "count": 60},
        {"token": "[O][=C]", "vocab_id": 11, "count": 40},
    ]
    return pd.DataFrame(rows)


def test_emit_latex_creates_file(tmp_path: Path):
    df = _make_freq_df()
    out = tmp_path / "table.tex"
    emit_latex(df, out, top=4)
    assert out.exists()
    assert out.stat().st_size > 0


def test_emit_latex_contains_longtable(tmp_path: Path):
    df = _make_freq_df()
    out = tmp_path / "table.tex"
    emit_latex(df, out, top=4)
    content = out.read_text()
    assert r"\begin{longtable}" in content
    assert r"\end{longtable}" in content
    assert r"\label{tab:ape-tokens}" in content


def test_emit_latex_top_limits_rows(tmp_path: Path):
    df = _make_freq_df()
    out = tmp_path / "table.tex"
    emit_latex(df, out, top=2)
    content = out.read_text()
    # Only the first 2 tokens should appear as table rows
    assert "[C][C]" in content
    assert "[C][O]" in content
    assert "[N][C]" not in content
    assert "[O][=C]" not in content


def test_emit_latex_frequency_values_present(tmp_path: Path):
    df = _make_freq_df()
    out = tmp_path / "table.tex"
    emit_latex(df, out, top=4)
    content = out.read_text()
    assert "100" in content
    assert "80" in content


def test_emit_latex_annotation_placeholder_when_empty(tmp_path: Path):
    df = _make_freq_df()  # no chemical_fragment column
    out = tmp_path / "table.tex"
    emit_latex(df, out, top=2)
    content = out.read_text()
    assert r"\emph{(to annotate)}" in content


def test_emit_latex_uses_provided_annotations(tmp_path: Path):
    df = _make_freq_df()
    df["chemical_fragment"] = ["C--C alkyl", "C--O ether", "N--C amine", "O=C carbonyl"]
    out = tmp_path / "table.tex"
    emit_latex(df, out, top=4)
    content = out.read_text()
    assert "C--C alkyl" in content
    assert r"\emph{(to annotate)}" not in content


# ── emit_ape_token_frequency_plot ────────────────────────────────────────────


def test_emit_ape_token_frequency_plot_creates_pdf_and_png(tmp_path: Path):
    emit_ape_token_frequency_plot(_make_freq_df(), tmp_path, top=3)
    pdf = tmp_path / "ape_token_frequency_top20.pdf"
    png = tmp_path / "ape_token_frequency_top20.png"
    assert pdf.exists() and pdf.stat().st_size > 0
    assert png.exists() and png.stat().st_size > 0


# ── integration: build_table (no real files) ────────────────────────────────


def test_build_table_end_to_end(tmp_path: Path):
    """build_table writes both outputs from in-memory fixtures."""
    vocab_file = tmp_path / "vocab.json"
    vocab_file.write_text(json.dumps(_tiny_vocab()), encoding="utf-8")

    parquet_file = tmp_path / "metadata.parquet"
    _tiny_selfies().to_frame(name="selfies").to_parquet(parquet_file, index=False)

    out_dir = tmp_path / "out"
    df = build_table(
        vocab_path=vocab_file,
        parquet_path=parquet_file,
        out_dir=out_dir,
        top=3,
        figure_dir=out_dir / "figures",
        figure_top=3,
    )

    assert (out_dir / "ape_token_freq.csv").exists()
    assert (out_dir / "table_ape_tokens.tex").exists()
    assert (out_dir / "figures" / "ape_token_frequency_top20.pdf").exists()
    assert (out_dir / "figures" / "ape_token_frequency_top20.png").exists()
    # DataFrame has all merged tokens
    assert len(df) == 4
    # Sorted descending
    assert df["count"].iloc[0] >= df["count"].iloc[-1]
