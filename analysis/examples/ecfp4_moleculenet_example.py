# %%
"""
ECFP4 baseline example on prepared MoleculeNet data.

This percent-format notebook demonstrates the core frozen-featurizer workflow:

1. Load one locally prepared MoleculeNet dataset.
2. Extract canonical SMILES and labels.
3. Featurize molecules with ECFP4.
4. Train a simple sklearn classifier or regressor.
5. Evaluate on the held-out test split.

Run from the repository root, for example:

    uv run python examples/ecfp4_moleculenet_example.py

Or open this file as a percent-format notebook in VS Code / Jupyter-compatible editors.
"""

# %%

import json

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from modernmolbert.eval.featurizers.rdkit_ecfp import ECFP4Featurizer
from modernmolbert.paths import find_project_root, data_path, outputs_path


PROJECT_ROOT = find_project_root()

print(PROJECT_ROOT.stem)


# %%
# Choose one prepared MoleculeNet dataset.
#
# Classification examples:
#   bbbp, bace, clintox, tox21, sider
#
# Regression examples:
#   esol, freesolv, lipophilicity

DATASET_NAME = "bbbp"

DATASET_DIR = data_path("eval", "moleculenet_sanitized", DATASET_NAME)

TRAIN_PATH = DATASET_DIR / "train.parquet"
VALID_PATH = DATASET_DIR / "valid.parquet"
TEST_PATH = DATASET_DIR / "test.parquet"
METADATA_PATH = DATASET_DIR / "metadata.json"

assert TRAIN_PATH.exists(), f"Missing {TRAIN_PATH}. Run prepare_moleculenet first."
assert VALID_PATH.exists(), f"Missing {VALID_PATH}. Run prepare_moleculenet first."
assert TEST_PATH.exists(), f"Missing {TEST_PATH}. Run prepare_moleculenet first."
assert METADATA_PATH.exists(), f"Missing {METADATA_PATH}."


# %%
# Load metadata and inspect the dataset contract.

metadata = json.loads(METADATA_PATH.read_text())

metadata_summary = {
    "name": metadata["name"],
    "task_type": metadata["task_type"],
    "preferred_metric": metadata["preferred_metric"],
    "tasks": metadata["tasks"],
    "split": metadata["split"],
    "row_counts": metadata["row_counts"],
}

print(metadata_summary)


# %%
# Load train/valid/test splits.
#
# For this small example, we train on train + valid and evaluate on test.
# For hyperparameter tuning, train on train, select on valid, and report test only once.

train_df = pd.read_parquet(TRAIN_PATH)
valid_df = pd.read_parquet(VALID_PATH)
test_df = pd.read_parquet(TEST_PATH)

train_valid_df = pd.concat([train_df, valid_df], ignore_index=True)

print("train:", train_df.shape)
print("valid:", valid_df.shape)
print("test: ", test_df.shape)
print("train+valid:", train_valid_df.shape)

train_valid_df.head()


# %%
# Pick a task.
#
# Some MoleculeNet datasets are multitask, e.g. tox21 and sider.
# This example uses the first task by default.

TASK_NAME = metadata["tasks"][0]
TASK_TYPE = metadata["task_type"]

print("Task:", TASK_NAME)
print("Task type:", TASK_TYPE)


# %%
# Helper: extract SMILES and labels, dropping rows with missing labels.
#
# Prepared datasets should already contain valid canonical SMILES because invalid
# molecules were removed before splitting unless keep_invalid=True was used.

SMILES_COLUMN = "smiles_canonical"


def get_smiles_and_labels(df: pd.DataFrame, task_name: str):
    if SMILES_COLUMN not in df.columns:
        raise ValueError(f"Missing required SMILES column: {SMILES_COLUMN}")

    if task_name not in df.columns:
        raise ValueError(f"Missing task column: {task_name}")

    keep = df[SMILES_COLUMN].notna() & df[task_name].notna()
    kept = df.loc[keep].copy()

    smiles = kept[SMILES_COLUMN].astype(str).tolist()
    y = kept[task_name].to_numpy(dtype=float)

    return smiles, y, kept


train_smiles, y_train, train_kept = get_smiles_and_labels(train_valid_df, TASK_NAME)
test_smiles, y_test, test_kept = get_smiles_and_labels(test_df, TASK_NAME)

print("Train rows with labels:", len(train_smiles))
print("Test rows with labels: ", len(test_smiles))
print("Train label shape:", y_train.shape)
print("Test label shape: ", y_test.shape)


# %%
# Featurize molecules with ECFP4.
#
# ECFP4 here means Morgan fingerprints with radius=2.
# The featurizer returns a FeatureBatch:
#
#   X:          features for valid molecules only
#   valid_mask: boolean mask over the original input SMILES list
#
# For canonical prepared datasets, all SMILES should usually be valid, but we
# still respect the valid_mask.

featurizer = ECFP4Featurizer(n_bits=2048, radius=2)

train_features = featurizer.featurize_smiles(train_smiles)
test_features = featurizer.featurize_smiles(test_smiles)

train_features.check(n_inputs=len(train_smiles))
test_features.check(n_inputs=len(test_smiles))

X_train = train_features.X
X_test = test_features.X

y_train_valid = y_train[train_features.valid_mask]
y_test_valid = y_test[test_features.valid_mask]

print("X_train:", X_train.shape)
print("X_test: ", X_test.shape)
print("Invalid train SMILES:", int((~train_features.valid_mask).sum()))
print("Invalid test SMILES: ", int((~test_features.valid_mask).sum()))


# %%
# Train a simple downstream model.
#
# For classification:
#   - LogisticRegression is a strong simple baseline for ECFP features.
#   - RandomForestClassifier is also common for fingerprints.
#
# For regression:
#   - Ridge regression is a simple linear baseline.
#   - RandomForestRegressor is a simple nonlinear baseline.
#
# Set MODEL_KIND to one of:
#   classification: "logistic", "random_forest"
#   regression:     "ridge", "random_forest"

MODEL_KIND = "logistic" if TASK_TYPE == "classification" else "ridge"

if TASK_TYPE == "classification":
    # Convert labels to integer classes for sklearn classifiers.
    y_train_model = y_train_valid.astype(int)
    y_test_model = y_test_valid.astype(int)

    if MODEL_KIND == "logistic":
        model = make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                solver="liblinear",
                random_state=13,
            ),
        )
    elif MODEL_KIND == "random_forest":
        model = RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=13,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported classification MODEL_KIND: {MODEL_KIND}")

else:
    y_train_model = y_train_valid
    y_test_model = y_test_valid

    if MODEL_KIND == "ridge":
        model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=1.0),
        )
    elif MODEL_KIND == "random_forest":
        model = RandomForestRegressor(
            n_estimators=500,
            random_state=13,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported regression MODEL_KIND: {MODEL_KIND}")

print(model)


# %%
# Fit the model.

model.fit(X_train, y_train_model)


# %%
# Evaluate.

if TASK_TYPE == "classification":
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test_model, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test_model, y_pred),
    }

    # ROC-AUC and average precision require both classes to be present.
    if len(np.unique(y_test_model)) == 2:
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]  # type: ignore
        else:
            y_score = model.decision_function(X_test)  # type: ignore

        metrics["roc_auc"] = roc_auc_score(y_test_model, y_score)
        metrics["average_precision"] = average_precision_score(y_test_model, y_score)
    else:
        metrics["roc_auc"] = np.nan
        metrics["average_precision"] = np.nan

else:
    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test_model, y_pred, squared=False)

    metrics = {
        "rmse": rmse,
        "mae": mean_absolute_error(y_test_model, y_pred),
        "r2": r2_score(y_test_model, y_pred),
    }

print(metrics)


# %%
# Present metrics as a small table.

metrics_df = pd.DataFrame(
    [
        {
            "dataset": DATASET_NAME,
            "task": TASK_NAME,
            "task_type": TASK_TYPE,
            "model_kind": MODEL_KIND,
            **metrics,
        }
    ]
)

print(metrics_df)


# %%
# Optional: save the example result.

OUTPUT_DIR = outputs_path("examples", "ecfp4_moleculenet")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

out_path = OUTPUT_DIR / f"{DATASET_NAME}_{TASK_NAME}_ecfp4_{MODEL_KIND}.csv"
metrics_df.round(4).to_csv(out_path, index=False)

print(f"Wrote {out_path}")

# %%
