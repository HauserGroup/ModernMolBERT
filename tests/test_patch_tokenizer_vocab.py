"""Tests for modernmolbert.tokenization.patch_tokenizer_vocab."""

import json
from pathlib import Path

import pytest

from modernmolbert.tokenization.patch_tokenizer_vocab import (
    load_symbols,
    sha256_of_file,
    validate_symbols,
)


# ---------------------------------------------------------------------------
# load_symbols
# ---------------------------------------------------------------------------


def test_load_symbols_returns_symbols(tmp_path: Path) -> None:
    p = tmp_path / "symbols.txt"
    p.write_text("[C@@H1]\n[C@H1]\n[/C]\n", encoding="utf-8")
    assert load_symbols(p) == ["[C@@H1]", "[C@H1]", "[/C]"]


def test_load_symbols_skips_comments_and_blanks(tmp_path: Path) -> None:
    p = tmp_path / "symbols.txt"
    p.write_text("# header\n\n[C]\n# another comment\n[O]\n", encoding="utf-8")
    assert load_symbols(p) == ["[C]", "[O]"]


def test_load_symbols_strips_whitespace(tmp_path: Path) -> None:
    p = tmp_path / "symbols.txt"
    p.write_text("  [C]  \n  [O]  \n", encoding="utf-8")
    assert load_symbols(p) == ["[C]", "[O]"]


def test_load_symbols_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "symbols.txt"
    p.write_text("", encoding="utf-8")
    assert load_symbols(p) == []


# ---------------------------------------------------------------------------
# validate_symbols
# ---------------------------------------------------------------------------


def test_validate_symbols_accepts_bracket_form() -> None:
    validate_symbols(["[C]", "[C@@H1]", "[/C]", "[=Branch1]"])


def test_validate_symbols_rejects_non_bracket_tokens() -> None:
    with pytest.raises(ValueError, match="Malformed SELFIES symbols"):
        validate_symbols(["[C]", "carbon"])


def test_validate_symbols_rejects_unclosed_bracket() -> None:
    with pytest.raises(ValueError, match="Malformed SELFIES symbols"):
        validate_symbols(["[C", "[O]"])


def test_validate_symbols_empty_list_passes() -> None:
    validate_symbols([])


# ---------------------------------------------------------------------------
# sha256_of_file
# ---------------------------------------------------------------------------


def test_sha256_of_file_consistent(tmp_path: Path) -> None:
    p = tmp_path / "data.json"
    p.write_bytes(b'{"a": 1}')
    h1 = sha256_of_file(p)
    h2 = sha256_of_file(p)
    assert h1 == h2
    assert len(h1) == 64  # hex sha256


def test_sha256_of_file_differs_with_content(tmp_path: Path) -> None:
    p1 = tmp_path / "a.json"
    p2 = tmp_path / "b.json"
    p1.write_bytes(b'{"a": 1}')
    p2.write_bytes(b'{"a": 2}')
    assert sha256_of_file(p1) != sha256_of_file(p2)


# ---------------------------------------------------------------------------
# Full patch workflow (calling main() internals via helper functions)
# ---------------------------------------------------------------------------


def _make_vocab(tmp_path: Path, tokens: dict[str, int]) -> Path:
    p = tmp_path / "vocab.json"
    p.write_text(json.dumps(tokens, indent=4), encoding="utf-8")
    return p


def _make_extra(tmp_path: Path, symbols: list[str]) -> Path:
    p = tmp_path / "extra.txt"
    p.write_text("\n".join(symbols) + "\n", encoding="utf-8")
    return p


def test_patch_adds_missing_symbols(tmp_path: Path) -> None:
    import subprocess
    import sys

    vocab_path = _make_vocab(
        tmp_path,
        {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "<mask>": 4, "[C]": 5},
    )
    extra_path = _make_extra(tmp_path, ["[O]", "[N]"])
    out_path = tmp_path / "vocab_patched.json"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "modernmolbert.tokenization.patch_tokenizer_vocab",
            "--input_file",
            str(vocab_path),
            "--extra_file",
            str(extra_path),
            "--output_file",
            str(out_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(out_path.read_text(encoding="utf-8"))
    assert "[O]" in result
    assert "[N]" in result
    assert result["[O]"] == 6
    assert result["[N]"] == 7
    assert result["[C]"] == 5


def test_patch_skips_already_present_symbols(tmp_path: Path) -> None:
    import subprocess
    import sys

    vocab_path = _make_vocab(
        tmp_path,
        {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "<mask>": 4, "[C]": 5, "[O]": 6},
    )
    extra_path = _make_extra(tmp_path, ["[C]", "[O]", "[N]"])
    out_path = tmp_path / "vocab_patched.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "modernmolbert.tokenization.patch_tokenizer_vocab",
            "--input_file",
            str(vocab_path),
            "--extra_file",
            str(extra_path),
            "--output_file",
            str(out_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    vocab = json.loads(out_path.read_text(encoding="utf-8"))
    assert vocab["[N]"] == 7
    assert "Already present   : 2" in result.stdout


def test_patch_dry_run_does_not_write(tmp_path: Path) -> None:
    import subprocess
    import sys

    vocab_path = _make_vocab(tmp_path, {"<s>": 0, "[C]": 1})
    extra_path = _make_extra(tmp_path, ["[O]"])
    out_path = tmp_path / "should_not_exist.json"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "modernmolbert.tokenization.patch_tokenizer_vocab",
            "--input_file",
            str(vocab_path),
            "--extra_file",
            str(extra_path),
            "--output_file",
            str(out_path),
            "--dry_run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert not out_path.exists()


def test_patch_no_op_when_all_present(tmp_path: Path) -> None:
    import subprocess
    import sys

    vocab_path = _make_vocab(tmp_path, {"<s>": 0, "[C]": 1, "[O]": 2})
    extra_path = _make_extra(tmp_path, ["[C]", "[O]"])
    out_path = tmp_path / "vocab_out.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "modernmolbert.tokenization.patch_tokenizer_vocab",
            "--input_file",
            str(vocab_path),
            "--extra_file",
            str(extra_path),
            "--output_file",
            str(out_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Nothing to do" in result.stdout
    assert not out_path.exists()


def test_patch_updates_companion_metadata(tmp_path: Path) -> None:
    import subprocess
    import sys

    vocab_path = _make_vocab(tmp_path, {"<s>": 0, "[C]": 1})
    extra_path = _make_extra(tmp_path, ["[O]"])
    out_path = tmp_path / "vocab_out.json"

    # Write companion metadata (same stem + "_metadata.json")
    metadata_path = out_path.with_name(out_path.stem + "_metadata.json")
    metadata_path.write_text(
        json.dumps({"vocab_size": 2, "tokenizer_sha256": "old_hash"}), encoding="utf-8"
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "modernmolbert.tokenization.patch_tokenizer_vocab",
            "--input_file",
            str(vocab_path),
            "--extra_file",
            str(extra_path),
            "--output_file",
            str(out_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert meta["vocab_size"] == 3
    assert meta["tokenizer_sha256"] != "old_hash"
    assert "patch_history" in meta
    assert len(meta["patch_history"]) == 1
    assert meta["patch_history"][0]["symbols_added"] == 1
