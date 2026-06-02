# %%
"""
Educational manual example.

This notebook demonstrates ECFP4 featurization and a simple sklearn model on one

prepared MoleculeNet dataset. It is not the canonical benchmark path.

For benchmark results, use the suite runner:

    uv run python -m modernmolbert.eval.cli.run_benchmark_suite \

      --suite configs/eval_suites/pilot_core.yaml \

      --output_dir outputs/eval/pilot_core


Frozen ModernMolBERT embedding baseline on prepared MoleculeNet data.

This percent-format notebook demonstrates the frozen-featurizer workflow:

1. Load one locally prepared MoleculeNet dataset.
2. Extract canonical SMILES and labels.
3. Convert SMILES to SELFIES.
4. Tokenize with the model's SELFIES tokenizer.
5. Extract frozen ModernMolBERT embeddings.
6. Train a simple sklearn classifier or regressor.
7. Evaluate on the held-out test split.

Run from the repository root, for example:

    uv run python examples/modernmolbert_moleculenet_example.py

Or open this file as a percent-format notebook in VS Code / Jupyter-compatible editors.
"""

# %%

import json
from pathlib import Path

import numpy as np
import pandas as pd
import selfies as sf
import torch

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
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from modernmolbert.common.paths import find_project_root, data_path, outputs_path


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
# Choose the trained ModernMolBERT checkpoint.
#
# Point this at the final_model directory from your completed run.

MODEL_DIR = Path("../runs/pubchem10m_mps_base_pilot_256/final_model")

# If you are already inside the run directory, use:
# MODEL_DIR = Path("final_model")

TOKENIZER_PATH = MODEL_DIR / "ape_tokenizer"
TOKENIZER_METADATA_PATH = MODEL_DIR / "tokenizer_metadata.json"

assert MODEL_DIR.exists(), f"Missing model directory: {MODEL_DIR}"
assert (MODEL_DIR / "config.json").exists(), f"Missing config.json in {MODEL_DIR}"
assert (MODEL_DIR / "model.safetensors").exists(), f"Missing model.safetensors in {MODEL_DIR}"
assert TOKENIZER_PATH.exists(), f"Missing tokenizer directory: {TOKENIZER_PATH}"
assert TOKENIZER_METADATA_PATH.exists(), f"Missing tokenizer metadata: {TOKENIZER_METADATA_PATH}"


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
# For this simple frozen-feature baseline, we train on train + valid and
# evaluate on test. For hyperparameter tuning, train on train, select on valid,
# and report test only once.

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
# Convert SMILES to SELFIES.
#
# The checkpoint expects SELFIES input, not SMILES. Invalid conversions are
# dropped and the label array is filtered to match.


def smiles_list_to_selfies(smiles_list: list[str]):
    selfies_list: list[str | None] = []

    for smi in smiles_list:
        try:
            selfies_str = sf.encoder(smi)
        except Exception:
            selfies_str = None

        if selfies_str is None or not str(selfies_str).strip():
            selfies_list.append(None)
        else:
            selfies_list.append(str(selfies_str))

    return selfies_list


def keep_valid_selfies(selfies_list: list[str | None], y: np.ndarray):
    valid_mask = np.array([s is not None for s in selfies_list], dtype=bool)
    valid_selfies = [s for s in selfies_list if s is not None]
    valid_y = y[valid_mask]
    return valid_selfies, valid_y, valid_mask


train_selfies_raw = smiles_list_to_selfies(train_smiles)
test_selfies_raw = smiles_list_to_selfies(test_smiles)

train_selfies, y_train_selfies, train_selfies_mask = keep_valid_selfies(train_selfies_raw, y_train)
test_selfies, y_test_selfies, test_selfies_mask = keep_valid_selfies(test_selfies_raw, y_test)

print("Train SELFIES valid:", len(train_selfies), "/", len(train_smiles))
print("Test SELFIES valid: ", len(test_selfies), "/", len(test_smiles))

print("Example SMILES:", train_smiles[0])
print("Example SELFIES:", train_selfies[0])


# %%
# Load frozen ModernMolBERT and its APE tokenizer.

device = (
    "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH), trust_remote_code=True)

model = AutoModel.from_pretrained(str(MODEL_DIR))
model.to(device)
model.eval()

print("Model loaded.")
print("Hidden size:", model.config.hidden_size)
print("Vocab size:", model.config.vocab_size)


# %%
# Tokenization helpers.
#
# This batching helper pads manually to the longest sequence in each batch.

MAX_SEQ_LENGTH = 256
BATCH_SIZE = 32


def tokenize_selfies_batch(selfies_batch: list[str]):
    encoded = [
        tokenizer(
            s,
            add_special_tokens=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors=None,
        )
        for s in selfies_batch
    ]

    max_len = max(len(x["input_ids"]) for x in encoded)

    input_ids = []
    attention_mask = []

    for item in encoded:
        ids = item["input_ids"]
        mask = item["attention_mask"]

        pad_len = max_len - len(ids)
        ids = ids + [tokenizer.pad_token_id] * pad_len
        mask = mask + [0] * pad_len

        input_ids.append(ids)
        attention_mask.append(mask)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor):
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom


@torch.no_grad()
def embed_selfies(selfies_list: list[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
    vectors = []

    for start in tqdm(range(0, len(selfies_list), batch_size), desc="Embedding"):
        batch_selfies = selfies_list[start : start + batch_size]

        batch = tokenize_selfies_batch(batch_selfies)
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)

        pooled = mean_pool(outputs.last_hidden_state, batch["attention_mask"])
        vectors.append(pooled.detach().cpu().numpy())

    return np.concatenate(vectors, axis=0)


# %%
# Extract frozen embeddings.

X_train = embed_selfies(train_selfies)
X_test = embed_selfies(test_selfies)

y_train_valid = y_train_selfies
y_test_valid = y_test_selfies

print("X_train:", X_train.shape)
print("X_test: ", X_test.shape)
print("y_train:", y_train_valid.shape)
print("y_test: ", y_test_valid.shape)


# %%
# Train a simple downstream model.
#
# For classification:
#   - LogisticRegression is a strong simple baseline for frozen embeddings.
#   - RandomForestClassifier is also available but often less natural for dense embeddings.
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
    y_train_model = y_train_valid.astype(int)
    y_test_model = y_test_valid.astype(int)

    if MODEL_KIND == "logistic":
        downstream_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                solver="liblinear",
                random_state=13,
            ),
        )
    elif MODEL_KIND == "random_forest":
        downstream_model = RandomForestClassifier(
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
        downstream_model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=1.0),
        )
    elif MODEL_KIND == "random_forest":
        downstream_model = RandomForestRegressor(
            n_estimators=500,
            random_state=13,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported regression MODEL_KIND: {MODEL_KIND}")

print(downstream_model)


# %%
# Fit the downstream model.

downstream_model.fit(X_train, y_train_model)


# %%
# Evaluate.

if TASK_TYPE == "classification":
    y_pred = downstream_model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test_model, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test_model, y_pred),
    }

    if len(np.unique(y_test_model)) == 2:
        if hasattr(downstream_model, "predict_proba"):
            y_score = downstream_model.predict_proba(X_test)[:, 1]  # type: ignore
        else:
            y_score = downstream_model.decision_function(X_test)  # type: ignore

        metrics["roc_auc"] = roc_auc_score(y_test_model, y_score)
        metrics["average_precision"] = average_precision_score(y_test_model, y_score)
    else:
        metrics["roc_auc"] = np.nan
        metrics["average_precision"] = np.nan

else:
    y_pred = downstream_model.predict(X_test)

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
            "checkpoint": str(MODEL_DIR),
            "embedding": "modernmolbert_mean_pool",
            "model_kind": MODEL_KIND,
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            **metrics,
        }
    ]
)

print(metrics_df)


# %%
# Save the example result.

OUTPUT_DIR = outputs_path("examples", "modernmolbert_moleculenet")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

safe_task_name = TASK_NAME.replace("/", "_").replace(" ", "_")
out_path = OUTPUT_DIR / f"{DATASET_NAME}_{safe_task_name}_modernmolbert_{MODEL_KIND}.csv"

metrics_df.round(4).to_csv(out_path, index=False)

print(f"Wrote {out_path}")

# %%
