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
