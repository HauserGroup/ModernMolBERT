import pandas as pd

from modernmolbert.eval.benchmarking_molecular_models.praski_export import (
    PRASKI_COLUMNS,
    append_result_row,
    count_result_rows,
    delete_result_rows,
    read_results_csv,
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


def test_csv_result_store_appends_counts_and_deletes_rows(tmp_path) -> None:
    output_csv = tmp_path / "benchmark_results.csv"
    row = {
        "dataset": "bbbp",
        "task": "classification",
        "embedder": "modernmolbert",
        "model": "rf",
        "hyperparams": '{"clf__n_estimators": 500}',
        "library_hash": "abc123",
        "pooling": "mean",
        "pooling_special_tokens_excluded": True,
        "embedding_model_dir": "runs/model/final_model",
        "embedding_tokenizer_path": "runs/model/final_model",
        "embedding_max_seq_length": 256,
        "cv_metric_name": "roc_auc",
        "cv_metric": 0.75,
        "test_metric_name": "roc_auc",
        "test_metric": 0.80,
    }

    frame = append_result_row(output_csv, row)

    assert list(frame.columns) == PRASKI_COLUMNS
    assert output_csv.exists()
    assert frame.loc[0, "id"] == 1
    assert frame.loc[0, "key"] == "bbbp_modernmolbert_rf"
    assert frame.loc[0, "pooling"] == "mean"
    assert bool(frame.loc[0, "pooling_special_tokens_excluded"]) is True
    assert frame.loc[0, "embedding_max_seq_length"] == 256
    assert (
        count_result_rows(
            output_csv,
            dataset="bbbp",
            embedder="modernmolbert",
            cv_metric_name="roc_auc",
            model="rf",
        )
        == 1
    )

    append_result_row(output_csv, {**row, "model": "ridge"})
    assert len(read_results_csv(output_csv)) == 2

    out = delete_result_rows(
        output_csv,
        dataset="bbbp",
        embedder="modernmolbert",
        cv_metric_name="roc_auc",
        model="rf",
    )
    assert len(out) == 1
    assert out.loc[out.index[0], "model"] == "ridge"
