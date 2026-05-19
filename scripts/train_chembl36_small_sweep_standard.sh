#!/usr/bin/env bash
# train_chembl36_small_sweep_standard.sh
#
# Hyperparameter sweep for ModernMolBERT-small on ChEMBL36 SELFIES.
# Axes: masking_strategy (1: standard) × mlm_probability (3) × learning_rate (3) = 9 runs.
#
# Runs sequentially on a single GPU (CUDA device 0).
# Each run's stdout+stderr is written to <output_dir>/train.log.
# Already-populated output directories are skipped (safe to re-run after partial failure).
#
# Requirements: uv

set -euo pipefail

# ─── Dataset ──────────────────────────────────────────────────────────────────
DATASET_NAME="data/pretrain/chembl36_selfies_tokenized"
SELFIES_COLUMN="selfies"
TRAIN_SPLIT="train"
VALIDATION_SPLIT="valid"

# ─── Tokenizer ────────────────────────────────────────────────────────────────
TOKENIZER_PATH="tokenizer/chembl36_selfies_2m_min2000.json"
TOKENIZER_METADATA_PATH="tokenizer/chembl36_selfies_2m_min2000.metadata.json"

# ─── Fixed training hyperparameters ───────────────────────────────────────────
MODEL_SIZE="small"
MAX_SEQ_LENGTH=128
MAX_STEPS=30000

# EVAL_SIZE controls how many validation examples are sampled.
# At eval batch_size=256 this is 16 batches — MAX_EVAL_BATCHES is set to match.
EVAL_SIZE=4096
MAX_EVAL_BATCHES=16

PER_DEVICE_TRAIN_BATCH_SIZE=256
PER_DEVICE_EVAL_BATCH_SIZE=256
GRADIENT_ACCUMULATION_STEPS=1

WARMUP_STEPS=1500
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0
SAVE_STEPS=5000
EVAL_STEPS=5000
LOGGING_STEPS=100
SAVE_TOTAL_LIMIT=2

NUM_WORKERS=4
SEED=42

# ─── Sweep grid: 1 × 3 × 3 = 9 runs ──────────────────────────────────────────
MASKING_STRATEGIES=(standard)

MLM_PROBS=(0.15 0.20 0.25)

# Shifted up from (5e-5 1e-4 2e-4): 5e-5 converges too slowly from random init
# in 30K steps. Upper bound raised to 4e-4 which is well within stable range for
# this model size and batch.
LEARNING_RATES=(1e-4 2e-4 4e-4)

# ─── Output root ──────────────────────────────────────────────────────────────
RUN_ROOT="runs/chembl36_small_mask_mlm_lr_sweep"
mkdir -p "${RUN_ROOT}"

# ─── Preflight checks ─────────────────────────────────────────────────────────
if [[ ! -d "${DATASET_NAME}" ]]; then
  echo "ERROR: Missing local dataset directory: ${DATASET_NAME}"
  echo "       Run prepare_chembl36_selfies first."
  exit 1
fi

if [[ ! -f "${TOKENIZER_PATH}" ]]; then
  echo "ERROR: Missing tokenizer: ${TOKENIZER_PATH}"
  exit 1
fi

if [[ ! -f "${TOKENIZER_METADATA_PATH}" ]]; then
  echo "ERROR: Missing tokenizer metadata: ${TOKENIZER_METADATA_PATH}"
  exit 1
fi

# ─── Build pending run list, skipping completed runs ──────────────────────────
declare -a PENDING=()
TOTAL=0
SKIPPED=0

for masking_strategy in "${MASKING_STRATEGIES[@]}"; do
  for mlm_probability in "${MLM_PROBS[@]}"; do
    for learning_rate in "${LEARNING_RATES[@]}"; do
      (( TOTAL++ )) || true

      run_name="mask_${masking_strategy}__mlm_${mlm_probability}__lr_${learning_rate}"
      run_name="${run_name//./p}"
      output_dir="${RUN_ROOT}/${run_name}"

      if [[ -d "${output_dir}" && -n "$(ls -A "${output_dir}" 2>/dev/null)" ]]; then
        echo "SKIP  already populated: ${output_dir}"
        (( SKIPPED++ )) || true
        continue
      fi

      PENDING+=("${masking_strategy}|${mlm_probability}|${learning_rate}|${output_dir}|${run_name}")
    done
  done
done

echo "──────────────────────────────────────────────────────────────"
echo "Total grid: ${TOTAL}  Skipped: ${SKIPPED}  To run: ${#PENDING[@]}"
echo "──────────────────────────────────────────────────────────────"

if [[ "${#PENDING[@]}" -eq 0 ]]; then
  echo "Nothing to do. All runs already present."
  echo "TensorBoard: uv run tensorboard --logdir ${RUN_ROOT}"
  exit 0
fi

# ─── Dispatch loop ────────────────────────────────────────────────────────────
for run_spec in "${PENDING[@]}"; do
  IFS='|' read -r masking_strategy mlm_probability learning_rate output_dir run_name \
    <<< "${run_spec}"

  mkdir -p "${output_dir}"
  log_file="${output_dir}/train.log"

  echo "────────────────────────────────────────────────────────────"
  echo "LAUNCH  ${run_name}"
  echo "  mask=${masking_strategy}  mlm=${mlm_probability}  lr=${learning_rate}"
  echo "  output → ${output_dir}"
  echo "  log    → ${log_file}"

  uv run accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --dynamo_backend no \
    --mixed_precision bf16 \
    -m modernmolbert.train_selfies_ape_modernbert \
    --dataset_name            "${DATASET_NAME}" \
    --selfies_column          "${SELFIES_COLUMN}" \
    --train_split             "${TRAIN_SPLIT}" \
    --use_validation_split \
    --validation_split        "${VALIDATION_SPLIT}" \
    --output_dir              "${output_dir}" \
    --device_backend          cuda \
    --model_size              "${MODEL_SIZE}" \
    --tokenizer_vocab_path    "${TOKENIZER_PATH}" \
    --tokenizer_metadata_path "${TOKENIZER_METADATA_PATH}" \
    --max_seq_length          "${MAX_SEQ_LENGTH}" \
    --max_steps               "${MAX_STEPS}" \
    --eval_size               "${EVAL_SIZE}" \
    --max_eval_batches        "${MAX_EVAL_BATCHES}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --per_device_eval_batch_size  "${PER_DEVICE_EVAL_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --mlm_probability         "${mlm_probability}" \
    --masking_strategy        "${masking_strategy}" \
    --learning_rate           "${learning_rate}" \
    --weight_decay            "${WEIGHT_DECAY}" \
    --max_grad_norm           "${MAX_GRAD_NORM}" \
    --warmup_steps            "${WARMUP_STEPS}" \
    --logging_steps           "${LOGGING_STEPS}" \
    --eval_steps              "${EVAL_STEPS}" \
    --save_steps              "${SAVE_STEPS}" \
    --save_total_limit        "${SAVE_TOTAL_LIMIT}" \
    --num_workers             "${NUM_WORKERS}" \
    --seed                    "${SEED}" \
    --compute_masked_accuracy \
    --report_to tensorboard \
    2>&1 | tee "${log_file}"

  echo "  Done: ${run_name}"
done

echo "════════════════════════════════════════════════════════════"
echo "Sweep complete."
echo "Results: ${RUN_ROOT}"
echo "TensorBoard: uv run tensorboard --logdir ${RUN_ROOT}"
echo "════════════════════════════════════════════════════════════"
