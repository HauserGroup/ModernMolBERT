import pandas as pd

from audit_split_overlap import audit_dataset
from modernmolbert.eval.benchmarking_molecular_models.common.types import Dataset


def test_split_audit_separates_exact_and_stereo_only_overlap():
    dataset = Dataset(
        name="toy",
        task="classification",
        data=pd.DataFrame(
            {
                "smiles": [
                    "C[C@H](O)F",
                    "CCO",
                    "CCN",
                    "C[C@@H](O)F",
                    "OCC",
                    "CCCC",
                    "not_a_smiles",
                ],
                "label": [0, 1, 0, 1, 0, 1, 0],
            }
        ),
        splits={"train": [0, 1], "valid": [2], "test": [3, 4, 5, 6]},
    )
    summary, rows = audit_dataset(dataset, "a" * 64)
    assert summary["n_test_rows"] == 4
    assert summary["n_test_same_raw_smiles"] == 0
    assert summary["n_test_same_isomeric_smiles"] == 1
    assert summary["n_test_same_inchikey"] == 1
    assert summary["n_test_same_nonisomeric_smiles"] == 2
    assert summary["n_test_stereo_variant_only"] == 1
    assert summary["n_invalid_test_smiles"] == 1
    assert rows[0]["test_source_row_index"] == 3
    assert rows[0]["stereo_variant_only"] is True
    assert rows[1]["same_isomeric_smiles"] is True
