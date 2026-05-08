#!/usr/bin/env bash
set -euo pipefail

DATASET_NAME="mikemayuare/PubChem10M_SMILES_SELFIES"
SELFIES_COLUMN="SELFIES"

TOKENIZER_PATH="tokenizer/pubchem10m_selfies_ape_tokenizer_1m.json"
TOKENIZER_METADATA_PATH="tokenizer/pubchem10m_selfies_ape_tokenizer_1m.metadata.json"

mkdir -p tokenizer

echo "[1/2] Training APE SELFIES tokenizer..."
uv run python -m modernmolbert.train_ape_tokenizer \
  --dataset_name "${DATASET_NAME}" \
  --selfies_column "${SELFIES_COLUMN}" \
  --output_vocab_path "${TOKENIZER_PATH}" \
  --tokenizer_train_size 1000000 \
  --max_vocab_size 5000 \
  --min_freq_for_merge 500 \
  --seed 13 \
  --shuffle_buffer_size 100000

echo "[2/2] Validating tokenizer..."
uv run python -m modernmolbert.validate_tokenizer \
  --dataset_name "${DATASET_NAME}" \
  --selfies_column "${SELFIES_COLUMN}" \
  --split train \
  --representation SELFIES \
  --tokenizer_vocab_path "${TOKENIZER_PATH}" \
  --tokenizer_metadata_path "${TOKENIZER_METADATA_PATH}" \
  --n 10000 \
  --max_seq_length 512

echo "Done."
echo "Tokenizer: ${TOKENIZER_PATH}"
echo "Metadata:  ${TOKENIZER_METADATA_PATH}"
