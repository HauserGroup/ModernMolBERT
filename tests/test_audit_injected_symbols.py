import re

import pandas as pd

from audit_injected_symbols import count_benchmark, count_pretraining, load_symbols
from modernmolbert.eval.benchmarking_molecular_models.common.types import Dataset


def test_counts_symbol_use_by_prepared_split_and_pretraining(tmp_path):
    symbols = tmp_path / "symbols.txt"
    symbols.write_text("# comment\n[C]\n")
    assert load_symbols(symbols) == ["[C]"]
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    Dataset(
        name="example",
        task="classification",
        data=pd.DataFrame({"smiles": ["C", "O", "C.O"], "label": [0, 1, 0]}),
        splits={"train": [0, 1], "test": [2]},
    ).serialize_legacy(prepared / "example.json")
    pattern = re.compile(r"\[C\]")
    rows, symbol_counts = count_benchmark(prepared, pattern)
    by_split = {row["split"]: row for row in rows}
    assert by_split["train"]["n_with_any_injected_symbol"] == 1
    assert by_split["test"]["n_with_any_injected_symbol"] == 1
    assert symbol_counts["[C]"] == 2

    corpus = tmp_path / "train.parquet"
    pd.DataFrame({"selfies": ["[C]", "[C][C]", "[O]"]}).to_parquet(corpus)
    counts = count_pretraining(corpus, pattern)
    assert counts["n_inputs"] == 3
    assert counts["n_with_any_injected_symbol"] == 2
    assert counts["[C]"] == 2
