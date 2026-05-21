from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
import pytest

from modernmolbert.eval.benchmarking_molecular_models import export_benchmark_corpus as corpus


@dataclass
class _DatasetObject:
    data: pd.DataFrame


def test_get_dataset_frame_accepts_object_or_dict_and_rejects_bad_shapes(tmp_path: Path) -> None:
    frame = pd.DataFrame({"smiles": ["CCO"]})

    assert corpus.get_dataset_frame(_DatasetObject(frame), tmp_path / "a.joblib").equals(frame)
    assert corpus.get_dataset_frame({"data": frame}, tmp_path / "b.joblib").equals(frame)

    with pytest.raises(TypeError, match="expected object with .data"):
        corpus.get_dataset_frame(object(), tmp_path / "c.joblib")

    with pytest.raises(TypeError, match="pandas DataFrame"):
        corpus.get_dataset_frame({"data": ["CCO"]}, tmp_path / "d.joblib")


def test_iter_smiles_for_split_normalizes_valid_and_filters_blank_values() -> None:
    frame = pd.DataFrame(
        {
            "smiles": [" CCO ", "CCN", "", None, "CCC", "COC"],
            "split": ["val", "validation", "valid", "test", "train", "TEST"],
        }
    )

    assert corpus.normalize_split_name("val") == "valid"
    assert list(corpus.iter_smiles_for_split(frame, "valid")) == ["CCO", "CCN"]
    assert list(corpus.iter_smiles_for_split(frame, "test")) == ["COC"]
    assert list(corpus.iter_smiles_for_split(frame.drop(columns=["split"]), "train")) == [
        "CCO",
        "CCN",
        "CCC",
        "COC",
    ]

    with pytest.raises(ValueError, match="missing required 'smiles'"):
        list(corpus.iter_smiles_for_split(pd.DataFrame({"split": ["train"]}), "train"))


def test_smiles_to_selfies_returns_none_for_encoder_failures(monkeypatch) -> None:
    monkeypatch.setattr(corpus.sf, "encoder", lambda smiles: "")
    assert corpus.smiles_to_selfies("CCO") is None

    def boom(smiles: str) -> str:
        raise ValueError("bad smiles")

    monkeypatch.setattr(corpus.sf, "encoder", boom)
    assert corpus.smiles_to_selfies("not-smiles") is None


def test_export_benchmark_corpus_main_smiles_mode_counts_primitives_without_conversion(
    monkeypatch, tmp_path: Path
) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    frame = pd.DataFrame({"smiles": ["CCO", "CCO", "N#N"]})
    joblib.dump(_DatasetObject(frame), prepared / "tiny.joblib")
    output = tmp_path / "symbols.tsv"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_benchmark_corpus.py",
            "--prepared_dir",
            str(prepared),
            "--output",
            str(output),
            "--split",
            "all",
            "--mode",
            "symbol_counts",
            "--representation",
            "SMILES",
        ],
    )

    corpus.main()

    lines = output.read_text(encoding="utf-8").splitlines()
    counts = {parts[0]: int(parts[1]) for parts in (line.split("\t") for line in lines[1:])}
    # duplicate CCO is deduplicated; unique SMILES are CCO and N#N
    assert counts["C"] == 2  # CCO → C, C
    assert counts["O"] == 1  # CCO → O
    assert counts["N"] == 2  # N#N → N, N
    assert counts["#"] == 1  # N#N → #


def test_export_benchmark_corpus_selfies_mode_rejects_smiles_representation(
    monkeypatch, tmp_path: Path
) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    frame = pd.DataFrame({"smiles": ["CCO"]})
    joblib.dump(_DatasetObject(frame), prepared / "tiny.joblib")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_benchmark_corpus.py",
            "--prepared_dir",
            str(prepared),
            "--output",
            str(tmp_path / "out.txt"),
            "--mode",
            "selfies",
            "--representation",
            "SMILES",
        ],
    )

    with pytest.raises(ValueError, match="--mode selfies"):
        corpus.main()


def test_export_benchmark_corpus_main_writes_symbol_counts_from_joblib(
    monkeypatch, tmp_path: Path
) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    frame = pd.DataFrame(
        {
            "smiles": ["CCO", "CCO", "N#N", "ignored"],
            "split": ["train", "train", "valid", "test"],
        }
    )
    joblib.dump(_DatasetObject(frame), prepared / "tiny.joblib")
    output = tmp_path / "symbols.tsv"

    monkeypatch.setattr(
        corpus,
        "smiles_to_selfies",
        lambda smiles: {"CCO": "[C][C][O]", "N#N": "[N][#N]"}.get(smiles),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_benchmark_corpus.py",
            "--prepared_dir",
            str(prepared),
            "--output",
            str(output),
            "--split",
            "all",
            "--mode",
            "symbol_counts",
            "--progress_every",
            "1000",
        ],
    )

    corpus.main()

    assert output.read_text(encoding="utf-8").splitlines() == [
        "symbol\tcount",
        "[C]\t2",
        "[O]\t1",
        "[N]\t1",
        "[#N]\t1",
    ]
