#!/usr/bin/env bash
set -euo pipefail

# Run one ChEMBL36 SELFIES ModernBERT-small CUDA training job.
#
# Usage:
#   scripts/train_chembl36_cuda_small_single.sh [mlm_probability] [learning_rate] [run_name]
#
# Examples:
#   scripts/train_chembl36_cuda_small_single.sh
#   scripts/train_chembl36_cuda_small_single.sh 0.35 1e-4
#   CUDA_VISIBLE_DEVICES=1 scripts/train_chembl36_cuda_small_single.sh 0.35 3e-4 gpu1_small_mlm_0p35_lr_3e-4

# Local prepared ChEMBL36 SELFIES dataset.
DATASET_NAME="data/pretrain/chembl36_selfies"
SELFIES_COLUMN="selfies"
TRAIN_SPLIT="train"
VALIDATION_SPLIT="valid"
MASKING_STRATEGY="standard"

# Expected tokenizer artifacts for ChEMBL36.
# These names assume the tokenizer was trained on up to 2M ChEMBL36 SELFIES.
TOKENIZER_PATH="tokenizer/chembl36_selfies_2m_benchmark_covered_ape_tokenizer.json"
TOKENIZER_METADATA_PATH="tokenizer/chembl36_selfies_2m_benchmark_covered_ape_tokenizer.metadata.json"

# Small ModernBERT-style molecular model.
MODEL_SIZE="small"
MAX_SEQ_LENGTH=256

# Training setup.
MAX_STEPS=500
EVAL_SIZE=1024
MAX_EVAL_BATCHES=128

PER_DEVICE_TRAIN_BATCH_SIZE=8
PER_DEVICE_EVAL_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=8

WARMUP_STEPS=1500
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0

SAVE_STEPS=250
EVAL_STEPS=250
LOGGING_STEPS=100
SAVE_TOTAL_LIMIT=5

NUM_WORKERS=4
SEED=13

# Better default than the old BERT-style 0.15 for molecular MLM.
MLM_PROBABILITY="${1:-0.35}"
LEARNING_RATE="${2:-1e-4}"

RUN_ROOT="runs/chembl36_cuda_small_lr_mlm_sweep"

if [[ $# -ge 3 ]]; then
  RUN_NAME="$3"
else
  RUN_NAME="small_mlm_${MLM_PROBABILITY}_lr_${LEARNING_RATE}"
  RUN_NAME="${RUN_NAME//./p}"
fi

OUTPUT_DIR="${RUN_ROOT}/${RUN_NAME}"

mkdir -p "${RUN_ROOT}"

if [[ ! -d "${DATASET_NAME}" ]]; then
  echo "Missing local dataset directory: ${DATASET_NAME}"
  echo "Run prepare_chembl36_selfies first."
  exit 1
fi

if [[ ! -f "${TOKENIZER_PATH}" ]]; then
  echo "Missing tokenizer: ${TOKENIZER_PATH}"
  echo "Run the ChEMBL36 tokenizer training command first."
  exit 1
fi

if [[ ! -f "${TOKENIZER_METADATA_PATH}" ]]; then
  echo "Missing tokenizer metadata: ${TOKENIZER_METADATA_PATH}"
  echo "Run the ChEMBL36 tokenizer training command first."
  exit 1
fi

if [[ -d "${OUTPUT_DIR}/final_model" ]]; then
  echo "Run already completed: ${OUTPUT_DIR}/final_model exists."
  echo "Choose a different run_name or remove the directory to re-run."
  exit 1
elif [[ -d "${OUTPUT_DIR}" && -n "$(ls -A "${OUTPUT_DIR}")" ]]; then
  echo "Incomplete run detected. Cleaning up: ${OUTPUT_DIR}"
  rm -rf "${OUTPUT_DIR}"
fi

echo "============================================================"
echo "Starting single CUDA run: ${RUN_NAME}"
echo "Output directory: ${OUTPUT_DIR}"
echo "model_size=${MODEL_SIZE}"
echo "max_seq_length=${MAX_SEQ_LENGTH}"
echo "mlm_probability=${MLM_PROBABILITY}"
echo "learning_rate=${LEARNING_RATE}"
echo "effective_batch_size=$((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "============================================================"

uv run accelerate launch -m modernmolbert.train_selfies_ape_modernbert \
  --dataset_name "${DATASET_NAME}" \
  --selfies_column "${SELFIES_COLUMN}" \
  --train_split "${TRAIN_SPLIT}" \
  --use_validation_split \
  --validation_split "${VALIDATION_SPLIT}" \
  --output_dir "${OUTPUT_DIR}" \
  --device_backend cuda \
  --model_size "${MODEL_SIZE}" \
  --tokenizer_vocab_path "${TOKENIZER_PATH}" \
  --tokenizer_metadata_path "${TOKENIZER_METADATA_PATH}" \
  --max_seq_length "${MAX_SEQ_LENGTH}" \
  --max_steps "${MAX_STEPS}" \
  --eval_size "${EVAL_SIZE}" \
  --max_eval_batches "${MAX_EVAL_BATCHES}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --mlm_probability "${MLM_PROBABILITY}" \
  --masking_strategy "${MASKING_STRATEGY}" \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --max_grad_norm "${MAX_GRAD_NORM}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --logging_steps "${LOGGING_STEPS}" \
  --eval_steps "${EVAL_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --num_workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --no-bf16 \
  --fp16 \
  --compute_masked_accuracy \
  --report_to tensorboard

echo "Finished run: ${RUN_NAME}"
echo "Final model: ${OUTPUT_DIR}/final_model"
echo "TensorBoard: uv run tensorboard --logdir ${OUTPUT_DIR}"
