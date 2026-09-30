"""The four supported training combinations share metadata and loading rules."""

import json
from pathlib import Path

import numpy as np
import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from modernmolbert.tokenization.load import load_checkpoint_tokenizer, load_verified_tokenizer
from modernmolbert.tokenization.ape import train_ape
from modernmolbert.tokenization.bpe import train_bpe
from modernmolbert.train_tokenizer import collect_aligned_sample, main
from modernmolbert.utils import metadata_path_for_vocab, resolve_special_ids


@pytest.mark.parametrize("algorithm", ["APE", "BPE"])
@pytest.mark.parametrize(
    ("representation", "corpus"),
    [
        ("SELFIES", ["[C][O]", "[C][C]", "[O][C]"]),
        ("SMILES", ["CO", "CC", "OC"]),
    ],
)
def test_train_and_reload_each_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    algorithm: str,
    representation: str,
    corpus: list[str],
) -> None:
    monkeypatch.setattr(
        "modernmolbert.train_tokenizer.collect_corpus_for_tokenizer",
        lambda **_kwargs: corpus,
    )
    output = tmp_path / ("tokenizer.json" if algorithm == "BPE" else "vocab.json")
    main(
        [
            "--algorithm",
            algorithm,
            "--representation",
            representation,
            "--output_vocab_path",
            str(output),
            "--tokenizer_train_size",
            str(len(corpus)),
            "--max_vocab_size",
            "30",
            "--min_freq_for_merge",
            "1",
        ]
    )

    metadata = json.loads(metadata_path_for_vocab(output).read_text())
    assert metadata["algorithm"] == algorithm
    assert metadata["representation"] == representation
    tokenizer, _, _, _ = load_verified_tokenizer(output)
    assert resolve_special_ids(tokenizer) == {
        "bos_token": 0,
        "pad_token": 1,
        "eos_token": 2,
        "unk_token": 3,
        "mask_token": 4,
    }
    for molecule in corpus:
        assert "".join(tokenizer.tokenize(molecule)) == molecule
        assert tokenizer.unk_token_id not in tokenizer.encode(molecule)

    if algorithm == "BPE":
        checkpoint = tmp_path / "checkpoint"
        tokenizer.save_pretrained(checkpoint)
        reloaded, checkpoint_representation = load_checkpoint_tokenizer(checkpoint)
        assert checkpoint_representation == representation
        assert reloaded.encode(corpus[0]) == tokenizer.encode(corpus[0])


def test_vocabulary_limit_must_hold_initial_alphabet() -> None:
    with pytest.raises(ValueError, match="primitive symbols"):
        train_ape(
            ["[C][O]"],
            "SELFIES",
            max_vocab_size=1,
            min_freq_for_merge=1,
            max_merge_pieces=2,
            log=lambda _message: None,
        )
    with pytest.raises(ValueError, match="initial characters"):
        train_bpe(["CO"], vocab_size=6, min_frequency=1)


def test_aligned_sample_uses_same_source_rows_for_both_representations(tmp_path: Path) -> None:
    parquet = tmp_path / "paired.parquet"
    indices = tmp_path / "indices.npy"
    pq.write_table(
        pa.table(
            {
                "smiles": ["C", "O", "N", "CO"],
                "selfies": ["[C]", "[O]", "[N]", "[C][O]"],
            }
        ),
        parquet,
    )
    smiles, smiles_meta = collect_aligned_sample(parquet, indices, "smiles", 3, 42)
    selfies, selfies_meta = collect_aligned_sample(parquet, indices, "selfies", 3, 42)
    pairs = {"C": "[C]", "O": "[O]", "N": "[N]", "CO": "[C][O]"}
    assert [pairs[value] for value in smiles] == selfies
    assert smiles_meta == selfies_meta

    np.save(indices, np.load(indices)[::-1], allow_pickle=False)
    with pytest.raises(ValueError, match="declared seed"):
        collect_aligned_sample(parquet, indices, "smiles", 3, 42)
