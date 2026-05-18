#!/usr/bin/env bash

mkdir -p logs/tokenizer

# uv run python -m modernmolbert.train_ape_tokenizer \
#   --output_vocab_path tokenizer/chembl36_selfies_2m_ape_max4_min500.json \
#   --dataset_name data/pretrain/chembl36_selfies \
#   --selfies_column selfies \
#   --representation SELFIES \
#   --tokenizer_train_size 2000000 \
#   --max_vocab_size 5000 \
#   --min_freq_for_merge 500 \
#   --max_merge_pieces 4 \
#   --seed 42 \
#   --show_progress \
#   > logs/tokenizer/chembl36_2mq_max4.log 2>&1 &

uv run python -m modernmolbert.train_ape_tokenizer \
  --output_vocab_path tokenizer/chembl36_selfies_2m_ape_max8_min500.json \
  --dataset_name data/pretrain/chembl36_selfies \
  --selfies_column selfies \
  --representation SELFIES \
  --tokenizer_train_size 2000000 \
  --max_vocab_size 5000 \
  --min_freq_for_merge 500 \
  --max_merge_pieces 8 \
  --seed 42 \
  --show_progress \
  > logs/tokenizer/chembl36_2m_max8.log 2>&1 &

wait
