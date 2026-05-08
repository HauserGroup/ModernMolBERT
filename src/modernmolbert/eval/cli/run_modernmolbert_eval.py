import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from modernmolbert.eval.datasets import load_prepared_moleculenet_dataset
from modernmolbert.eval.embeddings import embed_smiles
from modernmolbert.eval.sklearn_baselines import compute_metrics, fit_predict_sklearn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "[legacy] Evaluate a ModernMolBERT checkpoint on a locally prepared "
            "MoleculeNet dataset. Prefer run_frozen_benchmark.py for new runs."
        )
    )
    parser.add_argument(
        "--dataset_dir",
        required=True,
        help=(
            "Prepared dataset directory containing metadata.json and parquet splits, "
            "for example data/eval/moleculenet_sanitized/esol"
        ),
    )
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--eval_split",
        default="test",
        help="Prepared parquet split to evaluate on, usually test.",
    )
    parser.add_argument(
        "--smiles_column",
        default="smiles_canonical",
        help="Column to embed. For SMILES-based embedding use smiles_canonical.",
    )
    parser.add_argument(
        "--selfies_column",
        default="selfies",
        help="SELFIES column name, retained in EvalDataset metadata.",
    )
    parser.add_argument(
        "--merge_train_valid",
        action="store_true",
        help="Concatenate train and valid splits before fitting the downstream model.",
    )
    parser.add_argument(
        "--sklearn_model",
        choices=["ridge_or_logreg", "rf"],
        default="ridge_or_logreg",
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

    if weight_col in df.columns:
        w = df[weight_col].to_numpy()
        return np.isfinite(y) & (w != 0)

    return np.isfinite(y)


def _subset_features_for_kept_rows(
    X_valid_smiles: np.ndarray,
    embedding_valid_mask: np.ndarray,
    keep_mask: np.ndarray,
) -> np.ndarray:
    """Map a full-row keep mask onto an embedding matrix with invalid rows removed.

    embed_smiles returns:
      - X_valid_smiles: features only for rows where embedding_valid_mask is True
      - embedding_valid_mask: Boolean mask over the original dataframe rows

    keep_mask is also over the original dataframe rows. This function returns
    features for rows where both masks are True.
    """

    if len(embedding_valid_mask) != len(keep_mask):
        raise ValueError(
            "Mask length mismatch: "
            f"embedding_valid_mask={len(embedding_valid_mask)}, keep_mask={len(keep_mask)}"
        )

    return X_valid_smiles[keep_mask[embedding_valid_mask]]


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = load_prepared_moleculenet_dataset(
        dataset_dir=Path(args.dataset_dir),
        eval_split=args.eval_split,
        smiles_column=args.smiles_column,
        selfies_column=args.selfies_column,
        merge_train_valid=args.merge_train_valid,
    )

    train_df = _limit(ds.train, args.max_train)
    eval_df = _limit(ds.test, args.max_eval)

    X_train_all, train_valid_smiles = embed_smiles(
        train_df[ds.smiles_column].tolist(),
        model_dir=args.model_dir,
        tokenizer_path=args.tokenizer_path,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        device=args.device,
    )

    X_eval_all, eval_valid_smiles = embed_smiles(
        eval_df[ds.smiles_column].tolist(),
        model_dir=args.model_dir,
        tokenizer_path=args.tokenizer_path,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        device=args.device,
    )

    train_valid_smiles = np.asarray(train_valid_smiles, dtype=bool)
    eval_valid_smiles = np.asarray(eval_valid_smiles, dtype=bool)

    results = []

    for task in ds.label_columns:
        train_label_mask = _valid_label_mask(train_df, task)
        eval_label_mask = _valid_label_mask(eval_df, task)

        train_keep = train_label_mask & train_valid_smiles
        eval_keep = eval_label_mask & eval_valid_smiles

        if int(train_keep.sum()) == 0 or int(eval_keep.sum()) == 0:
            continue

        X_train = _subset_features_for_kept_rows(
            X_valid_smiles=X_train_all,
            embedding_valid_mask=train_valid_smiles,
            keep_mask=train_keep,
        )
        y_train = train_df.loc[train_keep, task].to_numpy()

        X_eval = _subset_features_for_kept_rows(
            X_valid_smiles=X_eval_all,
            embedding_valid_mask=eval_valid_smiles,
            keep_mask=eval_keep,
        )
        y_eval = eval_df.loc[eval_keep, task].to_numpy()

        if ds.task_type == "classification" and len(np.unique(y_train)) < 2:
            continue

        y_pred, y_score = fit_predict_sklearn(
            X_train,
            y_train,
            X_eval,
            task_type=ds.task_type,
            model_name=args.sklearn_model,
        )

        metrics = compute_metrics(
            y_true=y_eval,
            y_pred=y_pred,
            y_score=np.asarray(y_score),
            task_type=ds.task_type,
        )

        result = {
            "dataset": ds.name,
            "dataset_dir": str(args.dataset_dir),
            "eval_split": args.eval_split,
            "task": task,
            "task_type": ds.task_type,
            "sklearn_model": args.sklearn_model,
            "n_train": int(len(y_train)),
            "n_eval": int(len(y_eval)),
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
