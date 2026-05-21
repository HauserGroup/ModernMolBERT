"""Tests for modernmolbert.data.pretokenize_chembl36 pure functions."""

import json
from pathlib import Path

import pandas as pd


def _setup_vocab(tmp_path: Path) -> Path:
    vocab = {
        "<s>": 0,
        "<pad>": 1,
        "</s>": 2,
        "<unk>": 3,
        "<mask>": 4,
        "[C]": 5,
        "[O]": 6,
        "[N]": 7,
        "[C][O]": 8,
    }
    p = tmp_path / "vocab.json"
    p.write_text(json.dumps(vocab), encoding="utf-8")
    return p


def _init(vocab_path: Path) -> None:
    from modernmolbert.data import pretokenize_chembl36 as m

    m._init_worker(str(vocab_path))


# ---------------------------------------------------------------------------
# _tokenize_one
# ---------------------------------------------------------------------------


def test_tokenize_one_known_tokens(tmp_path: Path) -> None:
    from modernmolbert.data import pretokenize_chembl36 as m

    _init(_setup_vocab(tmp_path))
    ids = m._tokenize_one("[C][O]")
    assert ids[0] == m._bos_id
    assert ids[-1] == m._eos_id
    # [C][O] is a merged piece in our vocab — should get ID 8
    assert 8 in ids


def test_tokenize_one_uses_longest_match(tmp_path: Path) -> None:
    from modernmolbert.data import pretokenize_chembl36 as m

    _init(_setup_vocab(tmp_path))
    # [C][O] matches the merged token (id=8) rather than two separate tokens
    ids = m._tokenize_one("[C][O]")
    inner = ids[1:-1]  # strip BOS/EOS
    assert inner == [8]


def test_tokenize_one_falls_back_to_unk_for_unknown(tmp_path: Path) -> None:
    from modernmolbert.data import pretokenize_chembl36 as m

    _init(_setup_vocab(tmp_path))
    ids = m._tokenize_one("[Xe]")
    assert m._unk_id in ids


def test_tokenize_one_empty_selfies_returns_bos_unk_eos(tmp_path: Path) -> None:
    from modernmolbert.data import pretokenize_chembl36 as m

    _init(_setup_vocab(tmp_path))
    ids = m._tokenize_one("")
    assert ids == [m._bos_id, m._unk_id, m._eos_id]


def test_tokenize_one_always_starts_with_bos_ends_with_eos(tmp_path: Path) -> None:
    from modernmolbert.data import pretokenize_chembl36 as m

    _init(_setup_vocab(tmp_path))
    for selfies in ["[C]", "[C][O]", "[C][N][O]"]:
        ids = m._tokenize_one(selfies)
        assert ids[0] == m._bos_id
        assert ids[-1] == m._eos_id


# ---------------------------------------------------------------------------
# _process_shard
# ---------------------------------------------------------------------------


def test_process_shard_writes_input_ids_column(tmp_path: Path) -> None:
    from modernmolbert.data import pretokenize_chembl36 as m

    vocab_path = _setup_vocab(tmp_path)
    _init(vocab_path)

    src = tmp_path / "shard.parquet"
    dst = tmp_path / "out" / "shard.parquet"

    df = pd.DataFrame({"selfies": ["[C][O]", "[C][N]", "[O]"]})
    df.to_parquet(src, index=False)

    name, n_rows = m._process_shard((src, dst))

    assert name == "shard.parquet"
    assert n_rows == 3
    assert dst.exists()

    out_df = pd.read_parquet(dst)
    assert "input_ids" in out_df.columns
    assert len(out_df) == 3
    for ids in out_df["input_ids"]:
        ids = list(ids)
        assert ids[0] == m._bos_id
        assert ids[-1] == m._eos_id


def test_process_shard_creates_parent_directory(tmp_path: Path) -> None:
    from modernmolbert.data import pretokenize_chembl36 as m

    _init(_setup_vocab(tmp_path))

    src = tmp_path / "shard.parquet"
    dst = tmp_path / "deep" / "nested" / "shard.parquet"

    pd.DataFrame({"selfies": ["[C]"]}).to_parquet(src)
    m._process_shard((src, dst))
    assert dst.exists()
