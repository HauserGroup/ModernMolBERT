# MLM masking strategies

## Overview

Three masking strategies are available via `--masking_strategy`:

| Strategy | Description |
|---|---|
| `standard` | Independent Bernoulli per token (original BERT) |
| `span` | Budget-based contiguous APE-token span masking |
| `hetero_span` | Span masking with span-start positions weighted toward heteroatom-containing tokens |

## Motivation: MLM-FG

The `span` and `hetero_span` strategies take inspiration from MLM-FG (Peng et al., 2025,
*npj Artificial Intelligence*), which improves molecular MLM pre-training by masking
entire functional groups rather than random individual tokens. The core idea: forcing the
model to reconstruct chemically meaningful substructures from context produces better
representations than masking isolated atoms.

MLM-FG operates on SMILES and uses RDKit SMARTS patterns to identify functional groups,
then masks the entire token cluster belonging to one group at a time. Direct translation
to SELFIES is not possible because SELFIES does not expose a SMARTS-queryable graph.

## What the span strategies capture

**Multi-token unit masking.** Both span strategies mask contiguous windows of APE tokens,
requiring the model to reconstruct a stretch of sequence rather than a single position.
This is the key structural parallel with MLM-FG.

**APE tokenization partially substitutes for functional group identification.** The APE
tokenizer merges frequent consecutive SELFIES symbols into single tokens (e.g. `[C][=O]`
may become one token). Masking a few APE tokens starting at a heteroatom position is
therefore closer to functional-group-level masking than character-level span masking on
SMILES would be.

**SELFIES branch locality.** In SELFIES, branch contents appear sequentially inline.
A contiguous span can cover an entire branch (e.g. a ketone `=O`), whereas in SMILES
the equivalent branch requires tracking parentheses and is rarely contiguous. This makes
span masking semantically more coherent in SELFIES than in the MLM-FG setting.

**`hetero_span` start-position bias.** Tokens containing heteroatom brackets (`[=O]`,
`[N]`, `[Cl]`, etc.) are more likely to be part of functional groups than carbon-chain
tokens. The heteroatom regex uses `re.search`, so APE-merged tokens like `[C][=O]` also
receive the elevated weight — correct behavior.

## What the span strategies do not capture

**No functional group identification.** MLM-FG identifies whether a token belongs to a
named functional group (via SMARTS). `hetero_span` identifies whether a token *contains*
a heteroatom. These are different questions. A carboxylic acid `[C][=O][O][H]` starting
from the carbon receives weight 1.0 for its start; the functional character comes from
connectivity, not any individual atom's identity.

**Contiguity does not align with ring topology.** SELFIES ring-closure tokens
(`[Ring1_N]`) are placed at specific points in the derivation tree and may be far from
the ring-opening atom in the token sequence. A span cannot capture a ring system as a
semantic unit; SMARTS-based FG detection in MLM-FG handles this correctly.

**Fixed probability budget vs. MLM-FG's adaptive group count.** MLM-FG masks exactly
one functional group for molecules with fewer than 10 groups, and 10% of groups
otherwise. This keeps enough structure intact for reconstruction to be tractable. The
fixed `mlm_probability` budget may over-mask small molecules relative to this heuristic.

**`Branch`/`Ring` tokens not preferentially targeted.** SELFIES structural tokens
(`[Branch1_3]`, `[Ring2_1]`) carry essential graph topology but contain no heteroatom in
their string representation and receive weight 1.0 in `hetero_span`.

## Implementation notes

Span lengths are drawn from `Geometric(span_p_geom)`, shifted so the minimum length is
1, and clamped to `span_max_length`. With defaults `p=0.4`, `max=6`, the realized mean
is approximately 2.4 APE tokens per span.

The masking loop runs for at most `budget × 5` draws. If many draws land on
already-covered positions the budget fraction may not be fully reached; there is no
warning in this case. The minimum-1-token fallback in `_sample_batch_span_mask` only
guarantees at least one masked position, not that `mlm_probability` was met.

The heteroatom weight lookup indexes `_token_start_weights` by token ID, not position,
so the weight table is built once at collator construction and is correct across batches.

## Reference

Peng, T. et al. Pre-trained molecular language models with random functional group
masking. *npj Artificial Intelligence* **1**, 28 (2025).
https://doi.org/10.1038/s44387-025-00029-3
