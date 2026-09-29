"""Tests for modernmolbert.tokenization.patch_tokenizer_vocab."""

import json
from pathlib import Path

import pytest

from modernmolbert.tokenization.patch_tokenizer_vocab import (
    load_symbols,
    validate_symbols,
)


# ---------------------------------------------------------------------------
# load_symbols
# ---------------------------------------------------------------------------


def test_load_symbols_returns_symbols(tmp_path: Path) -> None:
    p = tmp_path / "symbols.txt"
    p.write_text("[C@@H1]\n[C@H1]\n[/C]\n", encoding="utf-8")
    assert load_symbols(p) == ["[C@@H1]", "[C@H1]", "[/C]"]


# ---------------------------------------------------------------------------
# validate_symbols
# ---------------------------------------------------------------------------


def test_validate_symbols_rejects_non_bracket_tokens() -> None:
    with pytest.raises(ValueError, match="Malformed SELFIES symbols"):
        validate_symbols(["[C]", "carbon"])


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
