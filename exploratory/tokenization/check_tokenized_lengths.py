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
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from pathlib import Path

import joblib
import numpy as np
import selfies as sf

from modernmolbert.eval.benchmarking_molecular_models.src.common.config import (
    expand_dataset_selection,
    load_dataset_config,
)
from modernmolbert.tokenization_ape import APEPreTrainedTokenizer, pre_tokenize_molecule
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


@dataclass
class LengthStats:
    name: str
    n_inputs: int
    n_selfies_valid: int
    selfies_failures: int
    raw_lengths: list[int]
    capped_lengths: list[int]
    truncation_count: int
    unk_count: int
    eligible_count: int
    unknown_selfies_symbols: Counter[str]

    @property
    def truncation_rate(self) -> float:
        return self.truncation_count / max(1, self.n_selfies_valid)

    @property
    def selfies_failure_rate(self) -> float:
        return self.selfies_failures / max(1, self.n_inputs)

    @property
    def unk_rate(self) -> float:
        return self.unk_count / max(1, self.eligible_count)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--selfies_column", default=None)
    parser.add_argument("--tokenizer_vocab_path", required=True)
    parser.add_argument("--tokenizer_metadata_path", default=None)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--sample_size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--shuffle_buffer_size", type=int, default=100_000)
    parser.add_argument(
        "--benchmark_config_dir",
        type=Path,
        default=None,
        help="Benchmark config directory. Enables prepared benchmark analysis.",
    )
    parser.add_argument(
        "--prepared_dir",
        type=Path,
        default=Path("data/prepared"),
        help="Prepared benchmark dataset directory for --benchmark_config_dir mode.",
    )
    parser.add_argument(
        "--benchmark_datasets",
        nargs="+",
        default=["all"],
        help="Dataset config names/globs for --benchmark_config_dir mode.",
    )
    return parser.parse_args()


def percentile(values, q):
    return float(np.percentile(np.asarray(values), q))


def resolve_metadata_path(tokenizer_vocab_path: str, tokenizer_metadata_path: str | None) -> Path:
    if tokenizer_metadata_path is not None:
        return Path(tokenizer_metadata_path)

    vocab_path = Path(tokenizer_vocab_path)
    candidates = [
        metadata_path_for_vocab(vocab_path),
        vocab_path.parent / "tokenizer_metadata.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_tokenizer(tokenizer_vocab_path: str, tokenizer_metadata_path: str | None):
    metadata_path = resolve_metadata_path(tokenizer_vocab_path, tokenizer_metadata_path)
    metadata = load_tokenizer_metadata(metadata_path)
    assert_metadata_representation(metadata, expected_representation=SELFIES_REPRESENTATION)

    tokenizer = APEPreTrainedTokenizer(representation=SELFIES_REPRESENTATION)
    tokenizer.load_vocabulary_file(tokenizer_vocab_path)
    special_ids = resolve_special_ids(tokenizer)
    return tokenizer, special_ids, metadata_path


def compute_length_stats(
    name: str,
    sequences: Iterable[str | None],
    *,
    tokenizer: APEPreTrainedTokenizer,
    special_ids: dict[str, int],
    max_seq_length: int,
    sample_size: int | None = None,
    input_representation: str,
) -> LengthStats:
    raw_lengths = []
    capped_lengths = []
    truncation_count = 0
    unk_count = 0
    eligible_count = 0
    unknown_selfies_symbols: Counter[str] = Counter()
    n_inputs = 0
    n_selfies_valid = 0
    selfies_failures = 0

    for seq in sequences:
        if sample_size is not None and n_inputs >= sample_size:
            break
        n_inputs += 1

        if seq is None:
            selfies_failures += 1
            continue

        text = str(seq).strip()
        if input_representation == "SMILES":
            try:
                encoded_selfies = sf.encoder(text)
            except Exception:
                selfies_failures += 1
                continue
        elif input_representation == "SELFIES":
            encoded_selfies = text
        else:
            raise ValueError(f"Unknown input representation: {input_representation!r}")

        if not encoded_selfies:
            selfies_failures += 1
            continue

        for symbol in pre_tokenize_molecule(encoded_selfies, SELFIES_REPRESENTATION):
            symbol_id = tokenizer.convert_tokens_to_ids([symbol])[0]  # type: ignore
            if symbol_id == special_ids["unk_token"]:
                unknown_selfies_symbols[symbol] += 1

        full = encode_sequence(tokenizer, encoded_selfies, max_seq_length=None)["input_ids"]
        capped = encode_sequence(tokenizer, encoded_selfies, max_seq_length=max_seq_length)[
            "input_ids"
        ]

        raw_len = len(full)
        capped_len = len(capped)
        raw_lengths.append(raw_len)
        capped_lengths.append(capped_len)

        if raw_len > max_seq_length:
            truncation_count += 1

        ignored_special_ids = {
            special_ids["pad_token"],
            special_ids["bos_token"],
            special_ids["eos_token"],
            special_ids["mask_token"],
        }
        for token_id in capped:
            if token_id in ignored_special_ids:
                continue
            eligible_count += 1
            if token_id == special_ids["unk_token"]:
                unk_count += 1

        n_selfies_valid += 1

    if n_inputs == 0:
        raise RuntimeError(f"No sequences found for {name}.")
    if n_selfies_valid == 0:
        raise RuntimeError(f"No valid SELFIES sequences found for {name}.")

    return LengthStats(
        name=name,
        n_inputs=n_inputs,
        n_selfies_valid=n_selfies_valid,
        selfies_failures=selfies_failures,
        raw_lengths=raw_lengths,
        capped_lengths=capped_lengths,
        truncation_count=truncation_count,
        unk_count=unk_count,
        eligible_count=eligible_count,
        unknown_selfies_symbols=unknown_selfies_symbols,
    )


def print_stats(stats: LengthStats, *, max_seq_length: int) -> None:
    print(f"{stats.name}")
    print("-" * len(stats.name))
    print(f"inputs:              {stats.n_inputs}")
    print(f"selfies_valid:       {stats.n_selfies_valid}")
    print(f"selfies_failures:    {stats.selfies_failures}")
    print(f"selfies_failure_rate:{stats.selfies_failure_rate: .6f}")
    print(f"max_seq_length:      {max_seq_length}")
    print(f"mean_raw_len:        {statistics.mean(stats.raw_lengths):.2f}")
    print(f"median_raw_len:      {statistics.median(stats.raw_lengths):.2f}")
    print(f"p90_raw_len:         {percentile(stats.raw_lengths, 90):.2f}")
    print(f"p95_raw_len:         {percentile(stats.raw_lengths, 95):.2f}")
    print(f"p99_raw_len:         {percentile(stats.raw_lengths, 99):.2f}")
    print(f"p99.5_raw_len:       {percentile(stats.raw_lengths, 99.5):.2f}")
    print(f"p99.9_raw_len:       {percentile(stats.raw_lengths, 99.9):.2f}")
    print(f"max_raw_len:         {max(stats.raw_lengths)}")
    print(f"truncated_count:     {stats.truncation_count}")
    print(f"truncation_rate:     {stats.truncation_rate:.6f}")
    print(f"unk_count:           {stats.unk_count}")
    print(f"unk_rate:            {stats.unk_rate:.6f}")
    print()


def print_summary(all_stats: list[LengthStats], *, max_seq_length: int) -> None:
    raw_lengths = [value for stats in all_stats for value in stats.raw_lengths]
    total_inputs = sum(stats.n_inputs for stats in all_stats)
    total_valid = sum(stats.n_selfies_valid for stats in all_stats)
    total_failures = sum(stats.selfies_failures for stats in all_stats)
    total_truncated = sum(stats.truncation_count for stats in all_stats)
    total_unknown = sum(stats.unk_count for stats in all_stats)
    total_eligible = sum(stats.eligible_count for stats in all_stats)
    unknown_symbols: Counter[str] = Counter()
    for stats in all_stats:
        unknown_symbols.update(stats.unknown_selfies_symbols)

    print("Aggregate")
    print("=========")
    print(f"datasets:            {len(all_stats)}")
    print(f"inputs:              {total_inputs}")
    print(f"selfies_valid:       {total_valid}")
    print(f"selfies_failures:    {total_failures}")
    print(f"selfies_failure_rate:{total_failures / max(1, total_inputs): .6f}")
    print(f"max_seq_length:      {max_seq_length}")
    print(f"mean_raw_len:        {statistics.mean(raw_lengths):.2f}")
    print(f"median_raw_len:      {statistics.median(raw_lengths):.2f}")
    print(f"p95_raw_len:         {percentile(raw_lengths, 95):.2f}")
    print(f"p99_raw_len:         {percentile(raw_lengths, 99):.2f}")
    print(f"p99.9_raw_len:       {percentile(raw_lengths, 99.9):.2f}")
    print(f"max_raw_len:         {max(raw_lengths)}")
    print(f"truncated_count:     {total_truncated}")
    print(f"truncation_rate:     {total_truncated / max(1, total_valid):.6f}")
    print(f"unk_count:           {total_unknown}")
    print(f"unk_rate:            {total_unknown / max(1, total_eligible):.6f}")
    print()

    print("Worst datasets")
    print("==============")
    for stats in sorted(
        all_stats,
        key=lambda item: (item.truncation_rate, item.selfies_failure_rate, max(item.raw_lengths)),
        reverse=True,
    )[:10]:
        print(
            f"{stats.name:32s} trunc={stats.truncation_rate:.6f} "
            f"selfies_fail={stats.selfies_failure_rate:.6f} "
            f"p99={percentile(stats.raw_lengths, 99):.1f} max={max(stats.raw_lengths)}"
        )
    print()

    print("Highest unknown-token rates")
    print("===========================")
    for stats in sorted(all_stats, key=lambda item: item.unk_rate, reverse=True)[:10]:
        print(f"{stats.name:32s} unk_rate={stats.unk_rate:.6f} unk_count={stats.unk_count}")
    print()

    print("Top unknown SELFIES symbols")
    print("===========================")
    for symbol, count in unknown_symbols.most_common(20):
        print(f"{symbol:16s} {count}")


def run_benchmark_analysis(args, tokenizer, special_ids) -> None:
    dataset_names = expand_dataset_selection(args.benchmark_config_dir, args.benchmark_datasets)
    all_stats: list[LengthStats] = []

    print("Prepared benchmark tokenized length check")
    print("=========================================")
    print(f"config_dir:          {args.benchmark_config_dir}")
    print(f"prepared_dir:        {args.prepared_dir}")
    print()

    for dataset_config_name in dataset_names:
        dataset_config = load_dataset_config(args.benchmark_config_dir, dataset_config_name)
        path = args.prepared_dir / f"{dataset_config.name}.joblib"
        dataset = joblib.load(path)
        stats = compute_length_stats(
            dataset_config.name,
            dataset.data["smiles"].astype(str).tolist(),
            tokenizer=tokenizer,
            special_ids=special_ids,
            max_seq_length=args.max_seq_length,
            sample_size=None,
            input_representation="SMILES",
        )
        all_stats.append(stats)
        print_stats(stats, max_seq_length=args.max_seq_length)

    print_summary(all_stats, max_seq_length=args.max_seq_length)


def run_streaming_analysis(args, tokenizer, special_ids) -> None:
    if args.dataset_name is None:
        raise ValueError("--dataset_name is required unless --benchmark_config_dir is provided.")

    args.selfies_column = infer_selfies_column(args.dataset_name, args.selfies_column)
    ds = get_streaming_dataset(
        args.dataset_name,
        split=args.split,
        seed=args.seed,
        buffer_size=args.shuffle_buffer_size,
    )

    sequences: list[str] = []
    for row in ds:
        seq = normalize_sequence(row, args.selfies_column)
        if seq is None:
            continue
        sequences.append(seq)
        if len(sequences) >= args.sample_size:
            break

    if not sequences:
        raise RuntimeError("No valid sequences found.")

    stats = compute_length_stats(
        args.dataset_name,
        sequences,
        tokenizer=tokenizer,
        special_ids=special_ids,
        max_seq_length=args.max_seq_length,
        input_representation="SELFIES",
    )

    print("Tokenized length check")
    print("======================")
    print(f"dataset:             {args.dataset_name}")
    print(f"split:               {args.split}")
    print(f"selfies_column:      {args.selfies_column}")
    print_stats(stats, max_seq_length=args.max_seq_length)
    print()
    print("Simple interpretation")
    print("---------------------")
    if stats.truncation_rate <= 0.001:
        print("Excellent: <=0.1% truncated.")
    elif stats.truncation_rate <= 0.005:
        print("Good: <=0.5% truncated.")
    elif stats.truncation_rate <= 0.01:
        print("Probably acceptable: <=1% truncated.")
    elif stats.truncation_rate <= 0.05:
        print("Borderline: <=5% truncated. Consider max_length=512 or filtering long molecules.")
    else:
        print("High truncation: consider max_length=512 or filtering long molecules.")


def main():
    args = parse_args()
    tokenizer, special_ids, metadata_path = load_tokenizer(
        args.tokenizer_vocab_path,
        args.tokenizer_metadata_path,
    )
    print(f"tokenizer_vocab_path: {args.tokenizer_vocab_path}")
    print(f"tokenizer_metadata_path: {metadata_path}")
    print()

    if args.benchmark_config_dir is not None:
        run_benchmark_analysis(args, tokenizer, special_ids)
    else:
        run_streaming_analysis(args, tokenizer, special_ids)


if __name__ == "__main__":
    main()
