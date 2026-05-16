#!/usr/bin/env bash
set -euo pipefail

# Local prepared ChEMBL36 SELFIES dataset.
DATASET_NAME="data/pretrain/chembl36_selfies"
SELFIES_COLUMN="selfies"
TRAIN_SPLIT="train"
VALIDATION_SPLIT="valid"

# Expected tokenizer artifacts for ChEMBL36.
TOKENIZER_PATH="tokenizer/chembl36_selfies_500k_ape_tokenizer.json"
TOKENIZER_METADATA_PATH="tokenizer/chembl36_selfies_500k_ape_tokenizer.metadata.json"

# One realistic backbone setup; we sweep only MLM probability + LR.
MODEL_SIZE="base"
MAX_SEQ_LENGTH=256
MAX_STEPS=30000
EVAL_SIZE=4096
MAX_EVAL_BATCHES=256
PER_DEVICE_TRAIN_BATCH_SIZE=8
PER_DEVICE_EVAL_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=8
WARMUP_STEPS=1500
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0
SAVE_STEPS=2000
EVAL_STEPS=2000
LOGGING_STEPS=100
SAVE_TOTAL_LIMIT=5
NUM_WORKERS=4
SEED=13

# Sweep grid.
MLM_PROBS=(0.15 0.20)
LEARNING_RATES=(5e-5 1e-4 2e-4)

RUN_ROOT="runs/chembl36_cuda_base_lr_mlm_sweep"
mkdir -p "${RUN_ROOT}"

if [[ ! -d "${DATASET_NAME}" ]]; then
  echo "Missing local dataset directory: ${DATASET_NAME}"
  echo "Run prepare_chembl36_selfies first."
  exit 1
fi

if [[ ! -f "${TOKENIZER_PATH}" ]]; then
  echo "Missing tokenizer: ${TOKENIZER_PATH}"
  exit 1
fi

if [[ ! -f "${TOKENIZER_METADATA_PATH}" ]]; then
  echo "Missing tokenizer metadata: ${TOKENIZER_METADATA_PATH}"
  exit 1
fi

for mlm_probability in "${MLM_PROBS[@]}"; do
  for learning_rate in "${LEARNING_RATES[@]}"; do
    run_name="mlm_${mlm_probability}_lr_${learning_rate}"
    run_name="${run_name//./p}"
    output_dir="${RUN_ROOT}/${run_name}"

    if [[ -d "${output_dir}" && -n "$(ls -A "${output_dir}")" ]]; then
      echo "Skipping existing non-empty output dir: ${output_dir}"
      continue
    fi

    echo "============================================================"
    echo "Starting CUDA run: ${run_name}"
    echo "Output directory: ${output_dir}"
    echo "mlm_probability=${mlm_probability}, learning_rate=${learning_rate}"

uv run accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --dynamo_backend inductor \
  --mixed_precision fp16 \
  -m modernmolbert.train_selfies_ape_modernbert \
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

    echo "Finished run: ${run_name}"
    echo "Best/final artifacts under: ${output_dir}"
  done
done

echo "All sweep runs completed."
echo "TensorBoard root: ${RUN_ROOT}"
echo "Launch with: uv run tensorboard --logdir ${RUN_ROOT}"
