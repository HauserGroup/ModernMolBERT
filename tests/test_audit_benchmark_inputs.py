import pandas as pd
import pytest

from audit_benchmark_inputs import audit_dataset, audit_smiles
from modernmolbert.eval.benchmarking_molecular_models.common.types import Dataset
from modernmolbert.tokenization_ape import APEPreTrainedTokenizer


def _tokenizer() -> APEPreTrainedTokenizer:
    tokenizer = APEPreTrainedTokenizer(representation="SELFIES")
    tokenizer.vocabulary = {
        "<s>": 0,
        "<pad>": 1,
        "</s>": 2,
        "<unk>": 3,
        "<mask>": 4,
        "[C]": 5,
        "[O]": 6,
        ".": 7,
    }
    tokenizer.update_reverse_vocabulary()
    return tokenizer


def test_audit_counts_component_loss_and_conversion_failures_by_split():
    dataset = Dataset(
        name="example",
        task="classification",
        data=pd.DataFrame({"smiles": ["C", "C.O", "invalid_smiles"]}),
        splits={"train": [0], "test": [1, 2]},
    )
    tokenizer = _tokenizer()
    rows = {row["split"]: row for row in audit_dataset(dataset, tokenizer, 128)}
    assert rows["train"]["n_inputs"] == 1
    assert rows["train"]["n_unknown_strict_parser"] == 0
    assert rows["test"]["n_disconnected"] == 1
    assert rows["test"]["n_unknown_strict_parser"] == 0
    assert rows["test"]["n_conversion_failure"] == 1
    assert audit_smiles("C", tokenizer, 2)["truncated"] == 1


def test_audit_rejects_ambiguous_split_membership():
    dataset = Dataset(
        name="example",
        task="classification",
        data=pd.DataFrame({"smiles": ["C"]}),
        splits={"train": [0], "test": [0]},
    )
    with pytest.raises(ValueError, match="two splits"):
        audit_dataset(dataset, _tokenizer(), 128)


def test_audit_keeps_prepared_rows_outside_scored_splits_visible():
    dataset = Dataset(
        name="example",
        task="classification",
        data=pd.DataFrame({"smiles": ["C", "C.O"]}),
        splits={"test": [0]},
    )
    rows = {row["split"]: row for row in audit_dataset(dataset, _tokenizer(), 128)}
    assert rows["test"]["n_inputs"] == 1
    assert rows["unspecified"]["n_inputs"] == 1
    assert rows["unspecified"]["n_disconnected"] == 1
