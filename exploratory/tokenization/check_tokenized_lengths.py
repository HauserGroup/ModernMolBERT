#!/usr/bin/env python3
"""Check SELFIES tokenized length distribution and truncation rate.

Example:
  uv run python scripts/check_tokenized_lengths.py \
    --dataset_name data/pretrain/chembl36_selfies \
    --tokenizer_vocab_path tokenizer/chembl36_selfies_2m_ape_tokenizer.json \
    --tokenizer_metadata_path tokenizer/chembl36_selfies_2m_ape_tokenizer.metadata.json \
    --selfies_column selfies \
    --max_seq_length 256 \
    --sample_size 100000
"""

import argparse
import statistics

from pathlib import Path

import numpy as np

from modernmolbert.tokenization_ape import APEPreTrainedTokenizer
from modernmolbert.utils import (
    SELFIES_REPRESENTATION,
    assert_metadata_representation,
    encode_sequence,
    get_streaming_dataset,
    infer_selfies_column,
    load_tokenizer_metadata,
    metadata_path_for_vocab,
    normalize_sequence,
    resolve_special_ids,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--selfies_column", default=None)
    parser.add_argument("--tokenizer_vocab_path", required=True)
    parser.add_argument("--tokenizer_metadata_path", default=None)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--sample_size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--shuffle_buffer_size", type=int, default=100_000)
    return parser.parse_args()


def percentile(values, q):
    return float(np.percentile(np.asarray(values), q))


def main():
    args = parse_args()
    args.selfies_column = infer_selfies_column(args.dataset_name, args.selfies_column)

    metadata_path = (
        args.tokenizer_metadata_path
        if args.tokenizer_metadata_path is not None
        else str(metadata_path_for_vocab(args.tokenizer_vocab_path))
    )
    metadata = load_tokenizer_metadata(Path(metadata_path))
    assert_metadata_representation(metadata, expected_representation=SELFIES_REPRESENTATION)

    tokenizer = APEPreTrainedTokenizer(representation=SELFIES_REPRESENTATION)
    tokenizer.load_vocabulary_file(args.tokenizer_vocab_path)
    special_ids = resolve_special_ids(tokenizer)

    ds = get_streaming_dataset(
        args.dataset_name,
        split=args.split,
        seed=args.seed,
        buffer_size=args.shuffle_buffer_size,
    )

    raw_lengths = []
    capped_lengths = []
    truncation_count = 0
    unk_count = 0
    eligible_count = 0
    n = 0

    for row in ds:
        seq = normalize_sequence(row, args.selfies_column)
        if seq is None:
            continue

        full = encode_sequence(tokenizer, seq, max_seq_length=None)["input_ids"]
        capped = encode_sequence(tokenizer, seq, max_seq_length=args.max_seq_length)["input_ids"]

        raw_len = len(full)
        capped_len = len(capped)

        raw_lengths.append(raw_len)
        capped_lengths.append(capped_len)

        if raw_len > args.max_seq_length:
            truncation_count += 1

        for token_id in capped:
            if token_id in special_ids.values():
                continue
            eligible_count += 1
            if token_id == special_ids["unk_token"]:
                unk_count += 1

        n += 1
        if n >= args.sample_size:
            break

    if n == 0:
        raise RuntimeError("No valid sequences found.")

    truncation_rate = truncation_count / n
    unk_rate = unk_count / max(1, eligible_count)

    print("Tokenized length check")
    print("======================")
    print(f"dataset:             {args.dataset_name}")
    print(f"split:               {args.split}")
    print(f"selfies_column:      {args.selfies_column}")
    print(f"sample_size:         {n}")
    print(f"max_seq_length:      {args.max_seq_length}")
    print()
    print("Raw tokenized length, before truncation")
    print("--------------------------------------")
    print(f"mean:                {statistics.mean(raw_lengths):.2f}")
    print(f"median:              {statistics.median(raw_lengths):.2f}")
    print(f"p90:                 {percentile(raw_lengths, 90):.2f}")
    print(f"p95:                 {percentile(raw_lengths, 95):.2f}")
    print(f"p99:                 {percentile(raw_lengths, 99):.2f}")
    print(f"p99.5:               {percentile(raw_lengths, 99.5):.2f}")
    print(f"p99.9:               {percentile(raw_lengths, 99.9):.2f}")
    print(f"max:                 {max(raw_lengths)}")
    print()
    print("Truncation / tokenizer quality")
    print("------------------------------")
    print(f"truncated_count:     {truncation_count}")
    print(f"truncation_rate:     {truncation_rate:.6f}")
    print(f"unk_rate:            {unk_rate:.6f}")
    print()
    print("Simple interpretation")
    print("---------------------")
    if truncation_rate <= 0.001:
        print("Excellent: <=0.1% truncated.")
    elif truncation_rate <= 0.005:
        print("Good: <=0.5% truncated.")
    elif truncation_rate <= 0.01:
        print("Probably acceptable: <=1% truncated.")
    elif truncation_rate <= 0.05:
        print("Borderline: <=5% truncated. Consider max_length=512 or filtering long molecules.")
    else:
        print("High truncation: consider max_length=512 or filtering long molecules.")


if __name__ == "__main__":
    main()
