import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from modernmolbert.eval.datasets import load_moleculenet
from modernmolbert.eval.embeddings import embed_smiles
from modernmolbert.eval.sklearn_baselines import compute_metrics, fit_predict_sklearn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["bbbp", "hiv", "tox21", "esol", "freesolv", "lipo"],
        required=True,
    )
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--splitter", default="scaffold")
    parser.add_argument(
        "--sklearn_model", choices=["ridge_or_logreg", "rf"], default="ridge_or_logreg"
    )
    parser.add_argument("--max_train", type=int, default=None)
    parser.add_argument("--max_eval", type=int, default=None)
    return parser.parse_args()


def _limit(df: pd.DataFrame, n: int | None) -> pd.DataFrame:
    if n is None or len(df) <= n:
        return df
    return df.sample(n=n, random_state=13).reset_index(drop=True)


def _valid_label_mask(df: pd.DataFrame, task: str) -> np.ndarray:
    y = df[task].to_numpy()
    weight_col = f"{task}__weight"
    if weight_col in df:
        w = df[weight_col].to_numpy()
        return np.isfinite(y) & (w != 0)
    return np.isfinite(y)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = load_moleculenet(args.dataset, splitter=args.splitter)

    train_df = _limit(ds.train, args.max_train)
    test_df = _limit(ds.test, args.max_eval)

    X_train_all, train_valid_smiles = embed_smiles(
        train_df[ds.smiles_column].tolist(),
        model_dir=args.model_dir,
        tokenizer_path=args.tokenizer_path,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        device=args.device,
    )

    X_test_all, test_valid_smiles = embed_smiles(
        test_df[ds.smiles_column].tolist(),
        model_dir=args.model_dir,
        tokenizer_path=args.tokenizer_path,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        device=args.device,
    )

    results = []

    for task in ds.label_columns:
        train_label_mask = _valid_label_mask(train_df, task)
        test_label_mask = _valid_label_mask(test_df, task)

        # Combine label-valid and embedding-valid masks.
        train_keep = train_label_mask & train_valid_smiles
        test_keep = test_label_mask & test_valid_smiles

        # X arrays only contain valid-smiles rows, so map masks down.
        X_train = X_train_all[train_label_mask[train_valid_smiles]]
        y_train = train_df.loc[train_keep, task].to_numpy()

        X_test = X_test_all[test_label_mask[test_valid_smiles]]
        y_test = test_df.loc[test_keep, task].to_numpy()

        if len(y_train) == 0 or len(y_test) == 0:
            continue

        if ds.task_type == "classification" and len(np.unique(y_train)) < 2:
            continue

        y_pred, y_score = fit_predict_sklearn(
            X_train,
            y_train,
            X_test,
            task_type=ds.task_type,
            model_name=args.sklearn_model,
        )

        y_score = np.asarray(y_score)

        metrics = compute_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_score=y_score,
            task_type=ds.task_type,
        )

        result = {
            "dataset": ds.name,
            "task": task,
            "task_type": ds.task_type,
            "sklearn_model": args.sklearn_model,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "metrics": metrics,
        }

        results.append(result)
        print(json.dumps(result, indent=2))

    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
