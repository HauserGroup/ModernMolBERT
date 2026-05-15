import json
import subprocess
import sys
from pathlib import Path

from modernmolbert.tokenization.filter_missing_selfies_symbols import read_symbol_counts


def test_read_symbol_counts_accepts_header_comments_and_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "symbol_counts.tsv"
    path.write_text(
        "\n".join(
            [
                "# comment",
                "",
                "symbol\tcount",
                "[C@@H1]\t18109",
                "[C@H1]\t17718",
                "[/C]\t15660",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert read_symbol_counts(path) == [
        ("[C@@H1]", 18109),
        ("[C@H1]", 17718),
        ("[/C]", 15660),
    ]


def test_read_symbol_counts_skips_malformed_rows(tmp_path: Path) -> None:
    path = tmp_path / "symbol_counts.tsv"
    path.write_text(
        "\n".join(
            [
                "symbol\tcount",
                "[C]\t100",
                "missing_count_column",
                "[O]\tnot_an_integer",
                "[N]\t50",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert read_symbol_counts(path) == [
        ("[C]", 100),
        ("[N]", 50),
    ]


def test_cli_writes_only_missing_symbols_above_min_count(tmp_path: Path) -> None:
    vocab_path = tmp_path / "vocab.json"
    counts_path = tmp_path / "symbol_counts.tsv"
    output_path = tmp_path / "missing_symbols.txt"

    vocab_path.write_text(
        json.dumps(
            {
                "<s>": 0,
                "<pad>": 1,
                "</s>": 2,
                "<unk>": 3,
                "<mask>": 4,
                "[C]": 5,
                "[O]": 6,
                "[N]": 7,
            }
        ),
        encoding="utf-8",
    )

    counts_path.write_text(
        "\n".join(
            [
                "symbol\tcount",
                "[C]\t1000",  # already in vocab: excluded
                "[C@@H1]\t100",  # missing and above threshold: included
                "[C@H1]\t100",  # missing and above threshold: included
                "[/C]\t10",  # missing and at threshold: included
                "[Au]\t9",  # missing but below threshold: excluded
                "[O]\t1000",  # already in vocab: excluded
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "modernmolbert.tokenization.filter_missing_selfies_symbols",
            "--vocab",
            str(vocab_path),
            "--symbol_counts",
            str(counts_path),
            "--output",
            str(output_path),
            "--min_count",
            "10",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Read symbols:       6" in result.stdout
    assert "Selected missing:   3" in result.stdout

    lines = output_path.read_text(encoding="utf-8").splitlines()

    assert lines[:2] == [
        "# Missing SELFIES primitive symbols selected from benchmark diagnostics.",
        "# min_count=10",
    ]

    assert lines[2:] == [
        "[C@@H1]",
        "[C@H1]",
        "[/C]",
    ]


def test_cli_sorts_by_count_descending_then_symbol(tmp_path: Path) -> None:
    vocab_path = tmp_path / "vocab.json"
    counts_path = tmp_path / "symbol_counts.tsv"
    output_path = tmp_path / "missing_symbols.txt"

    vocab_path.write_text(json.dumps({"<unk>": 0}), encoding="utf-8")

    counts_path.write_text(
        "\n".join(
            [
                "symbol\tcount",
                "[B]\t10",
                "[A]\t10",
                "[C]\t20",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "modernmolbert.tokenization.filter_missing_selfies_symbols",
            "--vocab",
            str(vocab_path),
            "--symbol_counts",
            str(counts_path),
            "--output",
            str(output_path),
            "--min_count",
            "10",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    lines = [
        line
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]

    assert lines == ["[C]", "[A]", "[B]"]


def test_cli_creates_parent_directory(tmp_path: Path) -> None:
    vocab_path = tmp_path / "vocab.json"
    counts_path = tmp_path / "symbol_counts.tsv"
    output_path = tmp_path / "nested" / "dir" / "missing_symbols.txt"

    vocab_path.write_text(json.dumps({"<unk>": 0}), encoding="utf-8")
    counts_path.write_text("[C@@H1]\t100\n", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "modernmolbert.tokenization.filter_missing_selfies_symbols",
            "--vocab",
            str(vocab_path),
            "--symbol_counts",
            str(counts_path),
            "--output",
            str(output_path),
            "--min_count",
            "10",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert output_path.exists()
    assert "[C@@H1]" in output_path.read_text(encoding="utf-8")
