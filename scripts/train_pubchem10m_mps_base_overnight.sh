#!/usr/bin/env bash
set -euo pipefail

DATASET_NAME="mikemayuare/PubChem10M_SMILES_SELFIES"
SELFIES_COLUMN="SELFIES"

TOKENIZER_PATH="tokenizer/pubchem10m_selfies_ape_tokenizer_1m.json"
TOKENIZER_METADATA_PATH="tokenizer/pubchem10m_selfies_ape_tokenizer_1m.metadata.json"

OUTPUT_DIR="runs/pubchem10m_mps_base_pilot_256"

mkdir -p runs

if [[ ! -f "${TOKENIZER_PATH}" ]]; then
  echo "Missing tokenizer: ${TOKENIZER_PATH}"
  echo "Run scripts/train_pubchem10m_tokenizer.sh first."
  exit 1
fi

if [[ ! -f "${TOKENIZER_METADATA_PATH}" ]]; then
  echo "Missing tokenizer metadata: ${TOKENIZER_METADATA_PATH}"
  echo "Run scripts/train_pubchem10m_tokenizer.sh first."
  exit 1
fi

if [[ -d "${OUTPUT_DIR}/final_model" ]]; then
  echo "Run already completed: ${OUTPUT_DIR}/final_model exists."
  echo "Choose a new OUTPUT_DIR or remove the directory to re-run."
  exit 1
elif [[ -d "${OUTPUT_DIR}" && -n "$(ls -A "${OUTPUT_DIR}")" ]]; then
  echo "Incomplete run detected. Cleaning up: ${OUTPUT_DIR}"
  rm -rf "${OUTPUT_DIR}"
fi

echo "Starting PubChem10M SELFIES ModernBERT-base MPS pilot run..."
echo "Output directory: ${OUTPUT_DIR}"

uv run python -m modernmolbert.train_selfies_ape_modernbert \
  --dataset_name "${DATASET_NAME}" \
  --selfies_column "${SELFIES_COLUMN}" \
  --train_split train \
  --output_dir "${OUTPUT_DIR}" \
  --device_backend mps \
  --model_size base \
  --tokenizer_vocab_path "${TOKENIZER_PATH}" \
  --tokenizer_metadata_path "${TOKENIZER_METADATA_PATH}" \
  --max_seq_length 256 \
  --max_steps 6000 \
  --eval_size 512 \
  --max_eval_batches 64 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --mlm_probability 0.15 \
  --learning_rate 1e-4 \
  --warmup_steps 500 \
  --logging_steps 25 \
  --eval_steps 1000 \
  --save_steps 1000 \
  --save_total_limit 3 \
  --num_workers 0 \
  --compute_masked_accuracy \
  --report_to tensorboard

echo "Done."
echo "Final model: ${OUTPUT_DIR}/final_model"
echo "TensorBoard: uv run tensorboard --logdir ${OUTPUT_DIR}"
