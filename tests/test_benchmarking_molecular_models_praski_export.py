import pandas as pd

from modernmolbert.eval.benchmarking_molecular_models.praski_export import (
    PRASKI_COLUMNS,
    to_praski_schema,
)


def test_to_praski_schema_classification_row() -> None:
    native = pd.DataFrame(
        [
            {
                "dataset": "bbbp",
                "display_name": "MoleculeNet_BBBP",
                "task_type": "classification",
                "downstream_name": "ridge",
                "metric_name": "roc_auc",
                "cv_metric": 0.80,
                "test_metric": 0.82,
                "downstream_best_params": '{"clf__C": 1.0}',
            }
        ]
    )

    out = to_praski_schema(
        native,
        embedder_name="ModernMolBERT_SELFIES_ChEMBL36_2M",
        library_hash=0,
    )

    assert list(out.columns) == PRASKI_COLUMNS
    assert out.loc[0, "dataset"] == "MoleculeNet_BBBP"
    assert out.loc[0, "task"] == "classification"
    assert out.loc[0, "model"] == "ridge"
    assert out.loc[0, "cv_metric_name"] == "roc_auc"
    assert out.loc[0, "test_metric_name"] == "roc_auc"
    assert out.loc[0, "key"] == "MoleculeNet_BBBP_ModernMolBERT_SELFIES_ChEMBL36_2M_ridge"


def test_to_praski_schema_regression_row() -> None:
    native = pd.DataFrame(
        [
            {
                "dataset": "esol",
                "display_name": "MoleculeNet_ESOL",
                "task_type": "regression",
                "downstream_name": "rf",
                "metric_name": "rmse",
                "cv_metric": 0.71,
                "test_metric": 0.74,
                "downstream_best_params": '{"reg__n_estimators": 500}',
            }
        ]
    )

    out = to_praski_schema(
        native,
        embedder_name="ModernMolBERT_SELFIES_ChEMBL36_2M",
        library_hash=0,
    )

    assert out.loc[0, "task"] == "regression"
    assert out.loc[0, "cv_metric_name"] == "rmse"
    assert out.loc[0, "test_metric_name"] == "rmse"
