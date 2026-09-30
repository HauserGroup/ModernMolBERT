"""Train APE vocabularies with incremental pair counts.

Adjacent pair candidates are counted within each molecule, including overlaps.
The most frequent eligible pair wins, with the earliest corpus occurrence
breaking ties. Replacements are non-overlapping and proceed left to right.
Returned frequencies count tokens that actually remain after those replacements.
"""

from array import array
from collections import defaultdict
from collections.abc import Callable, Iterable

import numpy as np

from modernmolbert.tokenization_ape import pre_tokenize_molecule

MAX_PAIR_TABLE_BYTES = 512 * 1024 * 1024


def train_ape(
    corpus: Iterable[str],
    representation: str,
    *,
    max_vocab_size: int,
    min_freq_for_merge: int,
    max_merge_pieces: int | None,
    log: Callable[[str], None] = print,
) -> dict[str, int]:
    """Return vocabulary tokens in learned order with their final corpus counts."""
    # default_factory hands each new piece the next id, so ids follow first appearance.
    new_ids: defaultdict[str, int] = defaultdict()
    new_ids.default_factory = new_ids.__len__
    lookup = new_ids.__getitem__
    flat = array("i")
    sequence_ends = array("q")
    n_malformed = 0

    log(f"Pretokenizing {representation}...")
    for molecule in corpus:
        try:
            pieces = pre_tokenize_molecule(str(molecule), representation)
        except ValueError:
            n_malformed += 1
            continue
        if pieces:
            flat.extend(map(lookup, pieces))
            sequence_ends.append(len(flat) - 1)
    if n_malformed:
        log(f"Skipped {n_malformed} malformed sequences")
    if not new_ids:
        raise ValueError("Cannot train APE tokenizer on an empty corpus.")
    if len(new_ids) > max_vocab_size:
        raise ValueError(
            f"APE vocabulary limit {max_vocab_size} is smaller than the "
            f"{len(new_ids)} primitive symbols in the corpus"
        )
    log(f"Pretokenization complete, found {len(new_ids)} tokens")

    token_ids = dict(new_ids)
    strings = list(token_ids)
    n = len(flat)
    # One spare slot so the neighbour of the corpus's last token is a valid index.
    ids = np.zeros(n + 1, dtype=np.int32)
    ids[:n] = np.frombuffer(flat, dtype=np.int32)
    del flat
    frequency = dict(
        zip(strings, np.bincount(ids[:n], minlength=len(strings)).tolist(), strict=True)
    )
    ends = np.zeros(n + 1, dtype=bool)
    ends[np.frombuffer(sequence_ends, dtype=np.int64)] = True
    ends[n] = True
    n_molecules = len(sequence_ends)
    n_live = n

    # Linked list over positions: a merge unlinks the right token instead of shifting the array.
    nxt = np.arange(1, n + 2, dtype=np.int32)
    prv = np.arange(-1, n, dtype=np.int32)

    # A pair's code is left * base + right; base bounds every id the loop can create.
    base = max(max_vocab_size, len(strings))
    pair_table_bytes = base * base * (np.dtype(np.intp).itemsize + 2)
    if pair_table_bytes > MAX_PAIR_TABLE_BYTES:
        raise ValueError(
            "APE vocabulary limit would require oversized dense pair tables: "
            f"at least {pair_table_bytes / 1024**2:.0f} MiB for {base} tokens. "
            "Reduce --max_vocab_size."
        )
    code_dtype = np.int32 if base * base < np.iinfo(np.int32).max else np.int64
    no_pair = base * base
    codes = ids[:-1].astype(code_dtype) * base + ids[1:]
    codes = np.append(codes, no_pair).astype(code_dtype)
    codes[ends] = no_pair
    counts = np.bincount(codes, minlength=no_pair + 1)[:no_pair]
    allowed = _AllowedPairs(strings, representation, max_merge_pieces, base)
    threshold = max(min_freq_for_merge, 1)

    while True:
        if len(frequency) >= max_vocab_size:
            log("Max vocabulary achieved")
            break
        if n_live == n_molecules:
            log("No more mergeable pairs")
            break

        n_tokens = len(strings)
        eligible = allowed.mask(counts.reshape(base, base)[:n_tokens, :n_tokens], threshold)
        count = int(eligible.max())
        if count < min_freq_for_merge:
            log("Not enough frequency found")
            break
        if count == 0:
            log("No valid merge pair found")
            break
        rows, cols = np.nonzero(eligible == count)
        tied = rows * base + cols
        # The earliest occurrence in corpus order breaks ties.
        code = int(tied[0]) if len(tied) == 1 else int(codes[np.isin(codes, tied).argmax()])
        left, right = divmod(code, base)

        merged = strings[left] + strings[right]
        merged_id = token_ids.get(merged)
        if merged_id is None:
            merged_id = token_ids[merged] = len(strings)
            strings.append(merged)
            log(
                f"New merge found: {merged} {len(frequency) + 1}/{max_vocab_size} "
                f"{round((len(frequency) + 1) / max_vocab_size * 100, 2)}%"
            )
        starts = np.flatnonzero(codes == code)
        if left == right:
            starts = _drop_overlaps(starts, prv)
        # Overlapping candidates are counted when choosing a pair, but only
        # non-overlapping replacements contribute to post-merge frequencies.
        applied = len(starts)
        frequency[merged] = frequency.get(merged, 0) + applied
        frequency[strings[left]] = max(0, frequency[strings[left]] - applied)
        frequency[strings[right]] = max(0, frequency[strings[right]] - applied)
        rights = nxt[starts]
        before = prv[starts]

        # Retire the pairs around each merge site, apply the merges, then count the new pairs.
        # Sites are disjoint, but one site's right token can be the next site's left neighbour.
        shared = np.r_[False, before[1:] == rights[:-1]]
        touched = np.concatenate([before[(before >= 0) & ~shared], starts, rights])
        _add(counts, codes[touched], -1, no_pair)

        after = nxt[rights]
        ids[starts] = merged_id
        ends[starts] = ends[rights]
        nxt[starts] = after
        prv[after] = starts
        codes[rights] = no_pair
        n_live -= len(starts)

        before = prv[starts]
        shared = np.r_[False, before[1:] == starts[:-1]]
        touched = np.concatenate([before[(before >= 0) & ~shared], starts])
        new_codes = ids[touched].astype(code_dtype) * base + ids[nxt[touched]]
        new_codes[ends[touched]] = no_pair
        codes[touched] = new_codes
        _add(counts, new_codes, 1, no_pair)

    return frequency


def _add(counts: np.ndarray, codes: np.ndarray, sign: int, no_pair: int) -> None:
    pairs, n = np.unique(codes[codes != no_pair], return_counts=True)
    counts[pairs] += sign * n


def _drop_overlaps(starts: np.ndarray, prv: np.ndarray) -> np.ndarray:
    """Keep the pair starts a left-to-right scan merges when a pair repeats one token."""
    if len(starts) < 2:
        return starts
    run_begins = np.r_[True, prv[starts[1:]] != starts[:-1]]
    index = np.arange(len(starts))
    run_first = np.maximum.accumulate(np.where(run_begins, index, 0))
    return starts[(index - run_first) % 2 == 0]


class _AllowedPairs:
    """Which token pairs may merge under the ``max_merge_pieces`` cap, checked lazily.

    Only pairs frequent enough to be merged are ever checked, and each only once.
    """

    def __init__(self, strings: list[str], representation: str, max_pieces: int | None, base: int):
        self.strings = strings
        self.representation = representation
        self.max_pieces = max_pieces
        self.base = base
        self.checked = np.zeros((base, base), dtype=bool)
        self.blocked = np.zeros((base, base), dtype=bool)

    def mask(self, counts: np.ndarray, threshold: int) -> np.ndarray:
        """Return the square block ``counts`` with the pairs over the cap set to zero."""
        if self.max_pieces is None:
            return counts
        n = len(counts)
        checked = self.checked[:n, :n]
        blocked = self.blocked[:n, :n]
        for left, right in zip(*np.nonzero((counts >= threshold) & ~checked), strict=True):
            merged = self.strings[left] + self.strings[right]
            blocked[left, right] = (
                len(pre_tokenize_molecule(merged, self.representation)) > self.max_pieces
            )
            checked[left, right] = True
        return np.where(blocked, 0, counts)
