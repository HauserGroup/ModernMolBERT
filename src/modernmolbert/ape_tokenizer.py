"""from https://github.com/mikemayuare/apetokenizer"""

from collections import defaultdict
import json
import os
import re
from typing import Any
from pathlib import Path


class APETokenizer:
    def __init__(
        self,
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        mask_token="<mask>",
    ):
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.mask_token = mask_token
        self.vocabulary_frequency = defaultdict(int)
        self.pair_counts = defaultdict(int)
        self.special_tokens = {
            self.bos_token: 0,
            self.pad_token: 1,
            self.eos_token: 2,
            self.unk_token: 3,
            self.mask_token: 4,
        }
        self.vocabulary = dict(self.special_tokens)
        self.update_reverse_vocabulary()

    @property
    def bos_token_id(self):
        return self.special_tokens[self.bos_token]

    @property
    def eos_token_id(self):
        return self.special_tokens[self.eos_token]

    @property
    def pad_token_id(self):
        return self.special_tokens[self.pad_token]

    @property
    def mask_token_id(self):
        return self.special_tokens[self.mask_token]

    @property
    def unk_token_id(self):
        return self.special_tokens[self.unk_token]

    def __call__(
        self,
        text,
        padding=False,
        max_length=None,
        return_tensors=None,
        add_special_tokens=False,
        truncation: bool | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Tokenize and prepare the input text.

        :param text: str, the text to tokenize and encode.
        :param add_special_tokens: bool, whether to add special tokens (like <s> and </s>).
        :param max_length: int, the maximum length of the token sequence.
        :param return_tensors: str, the type of tensors to return ('pt' for PyTorch, 'tf' for TensorFlow).
        :return: A dictionary with tokenized and encoded information.
        """
        # Encode the text using the encode method
        encoded_inputs = self.encode(
            text,
            padding=padding,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
        )

        # Create a dictionary to hold the output
        outputs: dict[str, Any] = {"input_ids": encoded_inputs}

        # Calculate the attention mask (1 for tokens, 0 for padding)
        attention_mask = [
            1 if token_id != self.vocabulary[self.pad_token] else 0 for token_id in encoded_inputs
        ]
        outputs["attention_mask"] = attention_mask

        # Truncate the sequences to max_length if necessary
        if max_length is not None:
            outputs["input_ids"] = outputs["input_ids"][:max_length]
            outputs["attention_mask"] = outputs["attention_mask"][:max_length]

        # Convert outputs to tensors if return_tensors is specified
        if return_tensors == "pt":  # For PyTorch
            import torch

            outputs["input_ids"] = torch.tensor(outputs["input_ids"])
            outputs["attention_mask"] = torch.tensor(outputs["attention_mask"])

        return outputs

    def __len__(self):
        """
        Return the number of tokens in the tokenizer's vocabulary.
        """
        return len(self.vocabulary)

    def pre_tokenize(self, molecule):
        """Pretokenize SELFIES strings into bracketed tokens.

        SELFIES tokens are bracketed, e.g. [C], [=Branch1], [Ring1].
        """
        words = re.findall(r"\[[^\]]+\]", molecule)
        return words

    def train(
        self,
        corpus,
        type="selfies",
        max_vocab_size: int = 5000,
        min_freq_for_merge: int = 2000,
        save_checkpoint: bool = False,
        checkpoint_path: str = "checkpoint",
        checkpoint_interval=500,
    ):
        self.max_vocab_size = max_vocab_size
        self.min_freq_for_merge = min_freq_for_merge
        # self.max_token_length = max_token_length

        text_padding = " " * 80

        # Preprocessing: tokenize each molecule separately to preserve boundaries.
        print("Pretokenizing", end="\r")
        tokenized_corpus = [
            tokens for sentence in corpus if (tokens := self.pre_tokenize(sentence))
        ]
        vocabulary_frequency = defaultdict(int)
        for tokens in tokenized_corpus:
            for token in tokens:
                vocabulary_frequency[token] += 1
        print(
            f"Pretokenization complete, found {len(vocabulary_frequency)} tokens",
            end="\r",
        )

        # to add the pretokens to the vocabulary numbering
        pre_tokens_counts = len(vocabulary_frequency)

        # Recompute pair counts from scratch every merge iteration.
        # Persisting counts across iterations can bias stale pairs and prevent
        # convergence.
        def get_most_common_pair(tokenized_corpus):
            pair_counts = defaultdict(int)
            for tokens in tokenized_corpus:
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    pair_counts[pair] += 1

            self.pair_counts = pair_counts

            if not pair_counts:
                return ("", ""), 0

            # Minimize lookups by using max function directly
            most_common_pair, freq = max(
                pair_counts.items(), key=lambda x: x[1], default=(("", ""), 0)
            )
            return most_common_pair, freq

        merged_counter = len(vocabulary_frequency) + 1
        checkpoint_increment = checkpoint_interval
        batch = checkpoint_interval + pre_tokens_counts

        while True:
            if save_checkpoint and len(vocabulary_frequency) == batch:
                self.vocabulary_frequency = dict(vocabulary_frequency)
                self.vocabulary = {
                    **self.special_tokens,
                    **{
                        word: idx
                        for idx, word in enumerate(
                            vocabulary_frequency.keys(),
                            start=len(self.special_tokens),
                        )
                    },
                }

                if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_path)

                self.save_vocabulary(f"{checkpoint_path}/checkpoint_{batch}.json")
                print(f"Checkpoint saved at {checkpoint_path}/checkpoint_{batch}.json")
                self.save_pretrained(f"{checkpoint_path}/checkpoint_{batch}")
                batch += checkpoint_increment

            if len(vocabulary_frequency) >= self.max_vocab_size:
                print("\rMax vocabulary achieved", text_padding)
                break

            if all(len(tokens) < 2 for tokens in tokenized_corpus):
                print("\rNo more mergeable pairs", text_padding)
                break

            most_common_pair, freq = get_most_common_pair(tokenized_corpus)
            if freq < self.min_freq_for_merge:
                print("\rNot enough frequency found", text_padding)
                break

            if not most_common_pair[0] or not most_common_pair[1]:
                print("\rNo valid merge pair found", text_padding)
                break

            merged_word = "".join(most_common_pair)
            if merged_word not in vocabulary_frequency:
                print(
                    f"New merge found: {merged_word} {merged_counter}/{max_vocab_size} {round(merged_counter / max_vocab_size * 100, 2)}%"
                )
                merged_counter += 1
            merged_word_freq = vocabulary_frequency.get(merged_word, 0)
            vocabulary_frequency[merged_word] = merged_word_freq + freq

            # Apply merges inside each molecule only.
            new_tokenized_corpus = []
            for tokens in tokenized_corpus:
                new_tokens = []
                skip_next = False
                for i in range(len(tokens)):
                    if skip_next:
                        skip_next = False
                        continue

                    if (
                        i < len(tokens) - 1
                        and tokens[i] == most_common_pair[0]
                        and tokens[i + 1] == most_common_pair[1]
                    ):
                        new_tokens.append(merged_word)
                        skip_next = True
                    else:
                        new_tokens.append(tokens[i])

                new_tokenized_corpus.append(new_tokens)

            tokenized_corpus = new_tokenized_corpus

        # Convert vocabulary_frequency to a regular dictionary for final output
        self.vocabulary_frequency = dict(vocabulary_frequency)
        self.vocabulary = {
            **self.special_tokens,
            **{word: idx for idx, word in enumerate(vocabulary_frequency.keys(), start=5)},
        }
        print("\nTraining complete.")

        return None

    def pad(
        self,
        batch,
        padding=False,
        return_tensors=None,
        pad_to_multiple_of=None,
        **kwargs,
    ):
        # Determine the maximum length in this batch for padding
        max_length = max(len(seq["input_ids"]) for seq in batch)

        if pad_to_multiple_of:
            # Ensure max_length is a multiple of pad_to_multiple_of
            max_length = ((max_length - 1) // pad_to_multiple_of + 1) * pad_to_multiple_of

        padded_sequences = []
        attention_masks = []
        labels = []  # Prepare to collect labels
        for seq in batch:
            # Extract the input_ids from the current sequence (assuming it's a dictionary)
            input_ids = seq["input_ids"]
            padding_length = max_length - len(input_ids)

            # Create the padded sequence and attention mask
            padded_seq = input_ids + [self.pad_token_id] * padding_length
            attention_mask = [1] * len(input_ids) + [0] * padding_length

            padded_sequences.append(padded_seq)
            attention_masks.append(attention_mask)

            # Handle labels if they are present in the batch
            if "labels" in seq:
                padded_labels = seq["labels"] + ([-100] * padding_length)
                labels.append(padded_labels)

        # Convert to tensors or the appropriate format
        if return_tensors == "pt":
            import torch

            padded_sequences = torch.tensor(padded_sequences)
            attention_masks = torch.tensor(attention_masks)
            output = {"input_ids": padded_sequences, "attention_mask": attention_masks}
            if labels:
                output["labels"] = torch.tensor(labels)
            return output
        else:
            # Return as lists if tensors are not requested
            output = {"input_ids": padded_sequences, "attention_mask": attention_masks}
            if labels:
                output["labels"] = labels
            return output

    def get_special_tokens_mask(self, token_ids, already_has_special_tokens=False):
        """
        Retrieves a mask array indicating which tokens are special tokens.

        :param token_ids: List[int], the tokenized code of the text.
        :param already_has_special_tokens: bool, whether the token_ids already contain special tokens.
        :return: List[int], a list of the same length as token_ids, where 1 indicates a special token.
        """
        if already_has_special_tokens:
            special_ids = set(self.special_tokens.values())
            return [1 if token_id in special_ids else 0 for token_id in token_ids]
        return [0] * len(token_ids)

    def train_from_iterator(self, iterator):
        raise NotImplementedError("train_from_iterator is not implemented for APETokenizer")

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):  # Single token
            return self.vocabulary.get(tokens, self.vocabulary[self.unk_token])
        else:  # List of tokens
            return [self.vocabulary.get(token, self.vocabulary[self.unk_token]) for token in tokens]

    def update_reverse_vocabulary(self):
        """Updates the reverse vocabulary based on the current state of the vocabulary."""
        # Create a reverse mapping from IDs to tokens
        self.reverse_vocabulary = {v: k for k, v in self.vocabulary.items()}

    def convert_ids_to_tokens(self, token_ids):
        """
        Converts a sequence of token IDs back to a list of string tokens.

        :param token_ids: List[int], a list of token IDs.
        :return: List[str], a list of string tokens corresponding to the token IDs.
        """
        # Map each token ID to its corresponding string token
        return [self.reverse_vocabulary.get(token_id, self.unk_token) for token_id in token_ids]

    def encode(self, text, padding=False, max_length=None, add_special_tokens=False):
        """
        Encode SELFIES text into vocabulary IDs.

        This operates on SELFIES pre-tokens, not raw characters. For example:
            "[C][C][O]" -> ["[C]", "[C]", "[O]"]

        Greedy APE matching is then done over spans of SELFIES tokens, so merged
        vocabulary entries like "[C][C]" or "[C][C][O]" can still be used.
        """
        encoded_tokens = []

        if add_special_tokens:
            encoded_tokens.append(self.vocabulary[self.bos_token])

        pieces = self.pre_tokenize(text)

        if not pieces:
            encoded_tokens.append(self.vocabulary[self.unk_token])
        else:
            i = 0
            while i < len(pieces):
                # Longest-match over SELFIES-token spans, not raw string characters.
                for j in range(len(pieces), i, -1):
                    possible_match = "".join(pieces[i:j])
                    if possible_match in self.vocabulary:
                        encoded_tokens.append(self.vocabulary[possible_match])
                        i = j
                        break

                else:
                    encoded_tokens.append(self.vocabulary[self.unk_token])
                    i += 1

        if add_special_tokens:
            encoded_tokens.append(self.vocabulary[self.eos_token])

        if max_length is not None:
            encoded_tokens = encoded_tokens[:max_length]

        if padding:
            if max_length is None:
                raise ValueError("max_length must be specified if padding is True or 'max_length'")
            pad_token = self.vocabulary[self.pad_token]
            while len(encoded_tokens) < max_length:
                encoded_tokens.append(pad_token)

        return encoded_tokens

    def save_vocabulary(self, file_path):
        path = Path(file_path)
        freq_path = path.with_name(f"{path.stem}_freq.json")

        with path.open("w", encoding="utf-8") as f:
            json.dump(
                self.vocabulary,
                f,
                ensure_ascii=False,
                indent=4,
            )
        with freq_path.open("w", encoding="utf-8") as f:
            json.dump(
                self.vocabulary_frequency,
                f,
                ensure_ascii=False,
                indent=4,
            )

    def load_vocabulary(self, file_path):
        with open(file_path, encoding="utf_8") as f:
            self.vocabulary = json.load(f)

        self.update_reverse_vocabulary()
        # with open(f"{file_path.rstrip('.json')}_freq.json", "r", encoding="utf_8") as f:
        #     self.vocabulary_frequency = json.load(f)

    def save_pretrained(self, save_directory):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save vocabulary
        vocab_file = os.path.join(save_directory, "vocab.json")
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocabulary, f, ensure_ascii=False, indent=4)

        # Save special tokens
        special_tokens_file = os.path.join(save_directory, "special_tokens.json")
        with open(special_tokens_file, "w", encoding="utf-8") as f:
            json.dump(self.special_tokens, f, ensure_ascii=False, indent=4)

        # Save training state
        # Prepare the data to be JSON serializable
        vocabulary_frequency_serializable = {
            str(k): v for k, v in self.vocabulary_frequency.items()
        }
        pair_counts_serializable = {str(k): v for k, v in self.pair_counts.items()}

        training_state = {
            "vocabulary_frequency": vocabulary_frequency_serializable,
            "pair_counts": pair_counts_serializable,
        }

        training_state_file = os.path.join(save_directory, "training_state.json")
        with open(training_state_file, "w", encoding="utf-8") as f:
            json.dump(training_state, f, ensure_ascii=False, indent=4)

        print(f"Tokenizer and training state saved in {save_directory}")

    @classmethod
    def from_pretrained(cls, pretrained_directory):
        vocab_file = os.path.join(pretrained_directory, "vocab.json")
        special_tokens_file = os.path.join(pretrained_directory, "special_tokens.json")
        training_state_file = os.path.join(pretrained_directory, "training_state.json")

        # Load vocabulary
        if os.path.isfile(vocab_file):
            with open(vocab_file, encoding="utf-8") as f:
                vocabulary = json.load(f)
        else:
            raise FileNotFoundError(f"Vocabulary file {vocab_file} not found.")

        # Load special tokens
        if os.path.isfile(special_tokens_file):
            with open(special_tokens_file, encoding="utf-8") as f:
                special_tokens = json.load(f)
        else:
            raise FileNotFoundError(f"Special tokens file {special_tokens_file} not found.")

        # Initialize the tokenizer
        tokenizer = cls()
        tokenizer.vocabulary = vocabulary
        tokenizer.special_tokens = special_tokens
        tokenizer.update_reverse_vocabulary()

        # Load training state if it exists
        if os.path.isfile(training_state_file):
            with open(training_state_file, encoding="utf-8") as f:
                training_state = json.load(f)
            tokenizer.vocabulary_frequency = defaultdict(
                int, training_state["vocabulary_frequency"]
            )
            tokenizer.pair_counts = defaultdict(int, training_state["pair_counts"])

        return tokenizer
