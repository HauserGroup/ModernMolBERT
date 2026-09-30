"""Character-level BPE tokenizers for molecular strings.

Byte-pair encoding (BPE) starts from single characters and repeatedly merges the
most frequent adjacent pair. Unlike APE it ignores symbol boundaries, so a
learned token can end inside a SELFIES bracket symbol or a SMILES bracket atom.
Each molecule is one word, so merges never cross molecules. Training and
tokenization use Hugging Face ``tokenizers``; the saved file is a standard
``tokenizer.json`` that loads without remote code.
"""

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from modernmolbert.utils import EXPECTED_SPECIAL_IDS, SPECIAL_TOKENS

if TYPE_CHECKING:
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast


def special_tokens_in_id_order() -> list[str]:
    return [
        SPECIAL_TOKENS[name]
        for name in sorted(EXPECTED_SPECIAL_IDS, key=lambda name: EXPECTED_SPECIAL_IDS[name])
    ]


def train_bpe(
    corpus: Sequence[str],
    *,
    vocab_size: int,
    min_frequency: int,
    alphabet: Iterable[str] = (),
) -> "Tokenizer":
    """Learn a character-level BPE tokenizer.

    ``vocab_size`` counts the five special tokens, the characters and the merges.
    Merging also stops when the most frequent pair occurs fewer than
    ``min_frequency`` times. Characters in ``alphabet`` enter the vocabulary even
    if the corpus lacks them; any other unseen character becomes ``<unk>``.
    """
    from tokenizers import Tokenizer, decoders, models, processors, trainers

    specials = special_tokens_in_id_order()
    characters = set(alphabet)
    for molecule in corpus:
        characters.update(molecule)
    minimum_size = len(specials) + len(characters)
    if vocab_size < minimum_size:
        raise ValueError(
            f"BPE vocabulary limit {vocab_size} is smaller than the "
            f"{minimum_size} special tokens and initial characters"
        )
    tokenizer = Tokenizer(models.BPE(unk_token=SPECIAL_TOKENS["unk_token"]))
    tokenizer.decoder = decoders.Fuse()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=specials,
        initial_alphabet=sorted(characters),
        show_progress=False,
    )
    tokenizer.train_from_iterator(corpus, trainer=trainer, length=len(corpus))
    bos = SPECIAL_TOKENS["bos_token"]
    eos = SPECIAL_TOKENS["eos_token"]
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{bos} $A {eos}",
        pair=f"{bos} $A {eos} $B {eos}",
        special_tokens=[
            (bos, EXPECTED_SPECIAL_IDS["bos_token"]),
            (eos, EXPECTED_SPECIAL_IDS["eos_token"]),
        ],
    )
    return tokenizer


def load_bpe_tokenizer(
    path: str | Path,
    *,
    representation: str,
    model_max_length: int | None = None,
) -> "PreTrainedTokenizerFast":
    """Wrap a saved BPE ``tokenizer.json`` as a Transformers tokenizer."""
    from transformers import PreTrainedTokenizerFast

    options = {} if model_max_length is None else {"model_max_length": model_max_length}
    return PreTrainedTokenizerFast(
        tokenizer_file=str(path),
        representation=representation,
        **SPECIAL_TOKENS,
        **options,
    )
