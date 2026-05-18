"""Hugging Face-compatible tokenizer for APE molecular vocabularies.

This file is intentionally self-contained so it can be copied into a model repo
and loaded by ``AutoTokenizer.from_pretrained(..., trust_remote_code=True)``.
"""

import json
import os
import re
from collections.abc import Mapping
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from transformers import PreTrainedTokenizer


Representation = Literal["SELFIES", "SMILES"]

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json"}
SELFIES_RE = re.compile(r"\[[^\]]+\]")
SMILES_RE = re.compile(
    r"(\[[^\]]+\]|Br?|Cl?|Si?|Se?|Li?|Na?|Mg?|Al?|Ca?|Fe?|Zn?|"
    r"N|O|S|P|F|I|K|B|C|H|"
    r"b|c|n|o|s|p|"
    r"\%\d{2}|\d|"
    r"\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|\*|\$)"
)


def _base_piece_count(token: str, representation: str) -> int:
    """Count primitive molecular pieces in a vocab token."""
    pieces = pre_tokenize_molecule(token, representation)
    return max(1, len(pieces))


def _max_vocab_piece_span(vocab: dict[str, int], representation: str) -> int:
    """Maximum number of primitive pieces covered by any non-special vocab token."""
    max_span = 1
    for token in vocab:
        if token.startswith("<") and token.endswith(">"):
            continue
        max_span = max(max_span, _base_piece_count(token, representation))
    return max_span


def _coerce_vocab(vocab: Mapping[str, Any]) -> dict[str, int]:
    if not isinstance(vocab, Mapping):
        raise ValueError("Vocabulary must be a JSON object mapping token strings to integer IDs.")
    out = {str(token): int(idx) for token, idx in vocab.items()}
    if len(set(out.values())) != len(out):
        raise ValueError("Vocabulary token IDs must be unique.")
    return out


def _token_text(token: Any) -> str:
    return str(getattr(token, "content", token))


def _normalize_representation(representation: str) -> Representation:
    normalized = representation.upper()
    if normalized not in {"SELFIES", "SMILES"}:
        raise ValueError(f"representation must be 'SELFIES' or 'SMILES', got {representation!r}")
    return normalized  # type: ignore[return-value]


def pre_tokenize_molecule(molecule: str, representation: str) -> list[str]:
    active_representation = _normalize_representation(representation)
    if active_representation == "SELFIES":
        return SELFIES_RE.findall(molecule)

    tokens: list[str] = []
    cursor = 0
    for match in SMILES_RE.finditer(molecule):
        if match.start() > cursor:
            tokens.extend(molecule[cursor : match.start()])
        tokens.append(match.group(0))
        cursor = match.end()
    if cursor < len(molecule):
        tokens.extend(molecule[cursor:])
    return [token for token in tokens if token and not token.isspace()]


def ape_tokenize(
    text: str,
    vocab: dict[str, int],
    representation: str,
    unk_token: str = "<unk>",
    max_piece_span: int | None = None,
) -> list[str]:
    pieces = pre_tokenize_molecule(text, representation)
    if not pieces:
        return [unk_token]

    if max_piece_span is None:
        max_piece_span = _max_vocab_piece_span(vocab, representation)

    n = len(pieces)
    tokens: list[str] = []
    i = 0

    while i < n:
        upper = min(n, i + max_piece_span)

        for j in range(upper, i, -1):
            candidate = "".join(pieces[i:j])
            if candidate in vocab:
                tokens.append(candidate)
                i = j
                break
        else:
            tokens.append(unk_token)
            i += 1

    return tokens


class APEPreTrainedTokenizer(PreTrainedTokenizer):
    """Hugging Face tokenizer backend for APE molecular tokenization. (Not fast)"""

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str | os.PathLike[str] | None = None,
        vocab: dict[str, Any] | None = None,
        representation: str = "SELFIES",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        mask_token: str = "<mask>",
        model_max_length: int = 256,
        **kwargs,
    ) -> None:
        if vocab is None:
            if vocab_file is None:
                vocab = {
                    bos_token: 0,
                    pad_token: 1,
                    eos_token: 2,
                    unk_token: 3,
                    mask_token: 4,
                }
            else:
                with open(vocab_file, encoding="utf-8") as f:
                    vocab = json.load(f)

        if vocab is None:
            raise ValueError("Loaded vocabulary is None.")

        self.vocab_file = str(vocab_file) if vocab_file is not None else None
        self.vocab = _coerce_vocab(vocab)
        self._require_special_tokens(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
        )
        self.ids_to_tokens = {idx: token for token, idx in self.vocab.items()}
        self.representation = _normalize_representation(representation)
        self.vocabulary_frequency: dict[str, int] = {}
        self.pair_counts: dict[tuple[str, str] | str, int] = {}
        self._max_piece_span = _max_vocab_piece_span(self.vocab, self.representation)

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            model_max_length=model_max_length,
            representation=self.representation,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def vocabulary(self) -> dict[str, int]:
        """Legacy alias for callers that previously used APETokenizer."""
        return self.vocab

    @vocabulary.setter
    def vocabulary(self, value: dict[str, int]) -> None:
        self.vocab = _coerce_vocab(value)
        self.update_reverse_vocabulary()
        self._refresh_tokenization_cache()

    @property
    def special_tokens(self) -> dict[str, int]:
        bos_token = str(self.bos_token)
        pad_token = str(self.pad_token)
        eos_token = str(self.eos_token)
        unk_token = str(self.unk_token)
        mask_token = str(self.mask_token)
        return {
            bos_token: self._convert_token_to_id(bos_token),
            pad_token: self._convert_token_to_id(pad_token),
            eos_token: self._convert_token_to_id(eos_token),
            unk_token: self._convert_token_to_id(unk_token),
            mask_token: self._convert_token_to_id(mask_token),
        }

    @special_tokens.setter
    def special_tokens(self, value: dict[str, int]) -> None:
        for token, token_id in value.items():
            self.vocab.setdefault(str(token), int(token_id))
        self.vocab = _coerce_vocab(self.vocab)
        self.update_reverse_vocabulary()
        self._refresh_tokenization_cache()

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab)

    def update_reverse_vocabulary(self) -> None:
        self.ids_to_tokens = {idx: token for token, idx in self.vocab.items()}

    def _refresh_tokenization_cache(self) -> None:
        self._max_piece_span = _max_vocab_piece_span(self.vocab, self.representation)

    def _require_special_tokens(
        self,
        *,
        bos_token: str,
        eos_token: str,
        unk_token: str,
        pad_token: str,
        mask_token: str,
    ) -> None:
        missing = [
            token_text
            for token in [bos_token, eos_token, unk_token, pad_token, mask_token]
            if (token_text := _token_text(token)) not in self.vocab
        ]
        if missing:
            raise ValueError(f"Vocabulary is missing required special tokens: {missing}")

    def pre_tokenize(self, molecule: str, representation: str | None = None) -> list[str]:
        return pre_tokenize_molecule(molecule, representation or self.representation)

    def _tokenize(self, text: str, **kwargs) -> list[str]:

        return ape_tokenize(
            text,
            vocab=self.vocab,
            representation=self.representation,
            unk_token=str(self.unk_token),
            max_piece_span=self._max_piece_span,
        )

    def encode_molecule(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: int | None = None,
        truncation: bool = True,
    ) -> list[int]:
        """Fast molecular encode path avoiding generic Hugging Face tokenizer overhead."""

        tokens = self._tokenize(text)

        ids = [self._convert_token_to_id(token) for token in tokens]

        if add_special_tokens:
            ids = self.build_inputs_with_special_tokens(ids)

        if max_length is not None and truncation:
            ids = ids[:max_length]

        return ids

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab[str(self.unk_token)])

    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens.get(int(index), str(self.unk_token))

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens)

    def _required_special_token_id(
        self,
        token_value: int | list[int] | str | list[str] | None,
        token_name: str,
    ) -> int:
        if token_value is None:
            raise ValueError(f"{token_name} must be set.")
        if isinstance(token_value, int):
            return token_value
        if isinstance(token_value, str):
            return self._convert_token_to_id(token_value)
        if len(token_value) == 1:
            only_value = token_value[0]
            if isinstance(only_value, int):
                return only_value
            if isinstance(only_value, str):
                return self._convert_token_to_id(only_value)
        raise ValueError(f"{token_name} must resolve to a single token id.")

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
    ) -> list[int]:
        bos_id = self._required_special_token_id(self.bos_token, "bos_token")
        eos_id = self._required_special_token_id(self.eos_token, "eos_token")
        if token_ids_1 is None:
            return [bos_id, *token_ids_0, eos_id]
        return [bos_id, *token_ids_0, eos_id, *token_ids_1, eos_id]

    def create_token_type_ids_from_sequences(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
    ) -> list[int]:
        return [0] * len(self.build_inputs_with_special_tokens(token_ids_0, token_ids_1))

    def pad(
        self,
        encoded_inputs: Any,
        padding: Any = True,
        max_length: int | None = None,
        pad_to_multiple_of: int | None = None,
        padding_side: str | None = None,
        return_attention_mask: bool | None = None,
        return_tensors: Any = None,
        verbose: bool = True,
    ):
        padding_enabled = padding not in (False, "do_not_pad")
        if (
            padding_enabled
            and isinstance(encoded_inputs, list)
            and any("labels" in item for item in encoded_inputs)
        ):
            target_length = max(
                len(item.get("input_ids", item.get("labels", []))) for item in encoded_inputs
            )
            if padding == "max_length" and max_length is not None:
                target_length = max_length

            if pad_to_multiple_of and target_length % pad_to_multiple_of:
                target_length = ((target_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

            padded_inputs = []
            for item in encoded_inputs:
                item = dict(item)
                labels = list(item.get("labels", []))
                pad_len = max(0, target_length - len(labels))
                if pad_len:
                    label_padding = [-100] * pad_len
                    if self.padding_side == "left":
                        labels = label_padding + labels
                    else:
                        labels = labels + label_padding
                    item["labels"] = labels
                padded_inputs.append(item)
            encoded_inputs = padded_inputs

        return super().pad(
            encoded_inputs,
            padding=padding,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            padding_side=padding_side,
            return_attention_mask=return_attention_mask,
            return_tensors=return_tensors,
            verbose=verbose,
        )

    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: str | None = None,
    ) -> tuple[str, ...]:
        if not os.path.isdir(save_directory):
            raise ValueError(f"Vocabulary path ({save_directory}) should be a directory.")

        vocab_file = Path(save_directory) / (
            f"{filename_prefix}-vocab.json" if filename_prefix else "vocab.json"
        )
        with vocab_file.open("w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=4)
        return (str(vocab_file),)

    def add_tokens_to_vocabulary(self, tokens: list[str]) -> int:
        """Add tokens to the tokenizer vocabulary if they are not already present.

        This is intended for forcing coverage of rare valid molecular primitive
        symbols, especially SELFIES bracket tokens, after APE merge training.
        """

        if not tokens:
            return 0

        next_id = max(self.vocab.values(), default=-1) + 1
        added = 0

        for token in tokens:
            token = str(token).strip()
            if not token:
                continue
            if token in self.vocab:
                continue

            self.vocab[token] = next_id
            next_id += 1
            added += 1

        if added:
            self.update_reverse_vocabulary()
            self._refresh_tokenization_cache()

        return added

    def save_pretrained(self, save_directory: str | os.PathLike[str], *args, **kwargs):
        saved_files = super().save_pretrained(save_directory, *args, **kwargs)
        save_path = Path(save_directory)

        special_tokens_map = {
            "bos_token": str(self.bos_token),
            "eos_token": str(self.eos_token),
            "unk_token": str(self.unk_token),
            "pad_token": str(self.pad_token),
            "mask_token": str(self.mask_token),
        }
        with (save_path / "special_tokens_map.json").open("w", encoding="utf-8") as f:
            json.dump(special_tokens_map, f, ensure_ascii=False, indent=2)

        tokenizer_config_path = save_path / "tokenizer_config.json"
        if tokenizer_config_path.exists():
            with tokenizer_config_path.open(encoding="utf-8") as f:
                tokenizer_config = json.load(f)
        else:
            tokenizer_config = {}
        tokenizer_config.pop("tokenizer_class", None)
        tokenizer_config.update(
            {
                "representation": self.representation,
                "model_max_length": self.model_max_length,
                "auto_map": {
                    "AutoTokenizer": [
                        "tokenization_ape.APEPreTrainedTokenizer",
                        None,
                    ],
                },
            }
        )
        with tokenizer_config_path.open("w", encoding="utf-8") as f:
            json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)

        return saved_files

    def save_vocabulary_file(self, file_path: str | os.PathLike[str]) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        freq_path = path.with_name(f"{path.stem}_freq.json")

        with path.open("w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=4)
        with freq_path.open("w", encoding="utf-8") as f:
            json.dump(self.vocabulary_frequency, f, ensure_ascii=False, indent=4)

    def load_vocabulary_file(
        self,
        file_path: str | os.PathLike[str],
        representation: str | None = None,
    ) -> None:
        if representation is not None:
            self.representation = _normalize_representation(representation)
        with open(file_path, encoding="utf-8") as f:
            vocab = json.load(f)
        self.vocab = _coerce_vocab(vocab)
        self._require_special_tokens(
            bos_token=str(self.bos_token),
            eos_token=str(self.eos_token),
            unk_token=str(self.unk_token),
            pad_token=str(self.pad_token),
            mask_token=str(self.mask_token),
        )
        self.ids_to_tokens = {idx: token for token, idx in self.vocab.items()}
        self._refresh_tokenization_cache()

    def train(
        self,
        corpus,
        type: str = "selfies",
        representation: str | None = None,
        max_vocab_size: int = 5000,
        min_freq_for_merge: int = 2000,
        max_merge_pieces: int | None = 8,
        save_checkpoint: bool = False,
        checkpoint_path: str = "checkpoint",
        checkpoint_interval: int = 500,
    ) -> None:
        self.representation = _normalize_representation(representation or type)
        text_padding = " " * 80

        print(f"Pretokenizing {self.representation}", end="\r")
        tokenized_corpus = []
        vocabulary_frequency: defaultdict[str, int] = defaultdict(int)

        for sentence in corpus:
            tokens = self.pre_tokenize(str(sentence))
            if not tokens:
                continue
            tokenized_corpus.append(tokens)
            for token in tokens:
                vocabulary_frequency[token] += 1
        print(
            f"Pretokenization complete, found {len(vocabulary_frequency)} tokens",
            end="\r",
        )

        if not tokenized_corpus:
            raise ValueError("Cannot train APE tokenizer on an empty corpus.")

        pre_tokens_counts = len(vocabulary_frequency)
        merged_counter = len(vocabulary_frequency) + 1
        checkpoint_increment = checkpoint_interval
        batch = checkpoint_interval + pre_tokens_counts
        piece_count_cache: dict[str, int] = {}

        def merged_piece_count(token: str) -> int:
            count = piece_count_cache.get(token)
            if count is None:
                count = _base_piece_count(token, self.representation)
                piece_count_cache[token] = count
            return count

        def get_most_common_pair(tokenized):
            pair_counts: defaultdict[tuple[str, str], int] = defaultdict(int)
            for tokens in tokenized:
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])

                    if max_merge_pieces is not None:
                        merged_candidate = "".join(pair)
                        if merged_piece_count(merged_candidate) > max_merge_pieces:
                            continue

                    pair_counts[pair] += 1

            merged_pair_counts: dict[tuple[str, str] | str, int] = {
                pair: count for pair, count in pair_counts.items()
            }
            self.pair_counts = merged_pair_counts
            if not pair_counts:
                return ("", ""), 0
            return max(pair_counts.items(), key=lambda x: x[1], default=(("", ""), 0))

        while True:
            if save_checkpoint and len(vocabulary_frequency) == batch:
                self.vocabulary_frequency = dict(vocabulary_frequency)
                self.vocab = {
                    **{
                        str(self.bos_token): 0,
                        str(self.pad_token): 1,
                        str(self.eos_token): 2,
                        str(self.unk_token): 3,
                        str(self.mask_token): 4,
                    },
                    **{
                        word: idx
                        for idx, word in enumerate(
                            vocabulary_frequency.keys(),
                            start=5,
                        )
                    },
                }
                self.ids_to_tokens = {idx: token for token, idx in self.vocab.items()}
                self._refresh_tokenization_cache()
                checkpoint_dir = Path(checkpoint_path)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                self.save_vocabulary_file(checkpoint_dir / f"checkpoint_{batch}.json")
                self.save_pretrained(str(checkpoint_dir / f"checkpoint_{batch}"))
                print(f"Checkpoint saved at {checkpoint_dir}/checkpoint_{batch}.json")
                batch += checkpoint_increment

            if len(vocabulary_frequency) >= max_vocab_size:
                print("\rMax vocabulary achieved", text_padding)
                break

            if all(len(tokens) < 2 for tokens in tokenized_corpus):
                print("\rNo more mergeable pairs", text_padding)
                break

            most_common_pair, freq = get_most_common_pair(tokenized_corpus)
            if freq < min_freq_for_merge:
                print("\rNot enough frequency found", text_padding)
                break

            if not most_common_pair[0] or not most_common_pair[1]:
                print("\rNo valid merge pair found", text_padding)
                break

            left_token, right_token = most_common_pair
            merged_word = left_token + right_token
            if merged_word not in vocabulary_frequency:
                print(
                    f"New merge found: {merged_word} {merged_counter}/{max_vocab_size} "
                    f"{round(merged_counter / max_vocab_size * 100, 2)}%"
                )
                merged_counter += 1
            vocabulary_frequency[merged_word] = vocabulary_frequency.get(merged_word, 0) + freq

            new_tokenized_corpus = []
            for tokens in tokenized_corpus:
                new_tokens = []
                append_token = new_tokens.append
                i = 0
                token_count = len(tokens)
                while i < token_count:
                    if (
                        i < token_count - 1
                        and tokens[i] == left_token
                        and tokens[i + 1] == right_token
                    ):
                        append_token(merged_word)
                        i += 2
                    else:
                        append_token(tokens[i])
                        i += 1

                new_tokenized_corpus.append(new_tokens)

            tokenized_corpus = new_tokenized_corpus

        self.vocabulary_frequency = dict(vocabulary_frequency)
        self.vocab = {
            str(self.bos_token): 0,
            str(self.pad_token): 1,
            str(self.eos_token): 2,
            str(self.unk_token): 3,
            str(self.mask_token): 4,
            **{word: idx for idx, word in enumerate(vocabulary_frequency.keys(), start=5)},
        }

        self.ids_to_tokens = {idx: token for token, idx in self.vocab.items()}
        self._refresh_tokenization_cache()

        checkpoint_dir = Path(checkpoint_path)

    def train_from_iterator(self, iterator, *args, **kwargs) -> None:
        raise NotImplementedError("train_from_iterator is not implemented for APE")


APEPreTrainedTokenizer.register_for_auto_class("AutoTokenizer")
