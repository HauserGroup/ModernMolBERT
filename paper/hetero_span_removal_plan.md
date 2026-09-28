# Plan: remove hetero-span masking from the next manuscript edition

**Decision (2026-09-28):** the heteroatom-biased span-masking ablation
(`hetero_span`, labelled *MMB-small-hetero* / *MMB-h*) is dropped from the next
edition of the manuscript. The code keeps it as an opt-in option (see
[Code status](#code-status)); nothing in the default paper pipeline produces
hetero-span output any more.

Line numbers refer to `paper/main.tex` as of commit `f3ff4ac`. They will drift
as the text is edited, so each item also names its section or label.

After removal the internal comparison has **three** ModernMolBERT checkpoints
(small-standard, small-span, base) instead of four, and the masking ablation
becomes a single contrast: **standard vs span**.

---

## 1. Text changes in `main.tex`

### 1.1 Introduction: masking motivation (~l. 309–317)

- The paragraph motivates chemically informed masking with the
  functional-group masking of `pengPretrainedMolecularLanguage2025`. Hetero-span
  masking was the paper's parser-free proxy for that idea, so once it is gone
  the paper no longer tests chemically informed masking.
- **Reword:** keep BERT vs SpanBERT as the motivation and frame the ablation as
  *token vs contiguous-span masking*. Either delete the Peng et al. sentence or
  move it to Future Work as an untested direction.
- Rewrite "Whether such structure-aware masking benefits a frozen SELFIES
  embedder is an open question we examine in an ablation" to refer to span
  masking. As written, it promises an experiment that is no longer reported.

### 1.2 Methods › Masking strategies (`sec:pretraining-procedure`, ~l. 607–623)

- "We implement three masking strategies" → **two**.
- Delete the *Heteroatom-biased span* description (~l. 614–620).
- "All three apply the standard BERT corruption rule" → "Both apply…".
- "\cref{sec:ablations} compares all three" → "compares both".
- Optional: add a footnote that the code also implements a heteroatom-biased
  variant, which is not evaluated here. This keeps the paper consistent with the
  repository and the `collator.py` docs.

### 1.3 Results › Ablation Studies (`sec:ablations`, ~l. 888–923)

This section contains the conclusions that change most.

- ~l. 902–904: delete "and heteroatom-biased span masking consistently weakest".
- ~l. 906–910: delete "and heteroatom-biased span masking by $+0.5$". The
  remaining numbers (size $+0.5$, span $-0.3$) are unaffected.
- ~l. 910–913 (**conclusion to reword**): the sentence "Notably, the
  validation-MLM ranking does not carry through to downstream performance:
  heteroatom-biased span masking is weakest on the validation objective yet
  matches MMB-base downstream, while span masking is strongest on validation MLM
  yet marginally trails standard masking" used hetero-span as its **strongest
  example**. Only the span-vs-standard evidence remains:
  - Suggested rewrite: *"Span masking is strongest on the validation MLM
    objective yet marginally trails standard masking downstream ($-0.3$), so the
    validation ranking does not carry through to frozen-embedding performance."*
  - Present it as one observed mismatch, not a general pattern: with a single
    contrast, "the validation ranking does not transfer" is weaker evidence than
    before.
- ~l. 914–915 and 917–919: "no reliable downstream effect of masking strategy"
  and "largely insensitive to masking strategy" still hold, but now rest on one
  contrast (standard vs span). Say *"span masking"* rather than *"masking
  strategy"* in general.
- ~l. 893 and 923: both reference `fig:four-model`; update them when that label
  is renamed (see 2.1).

### 1.4 Results figure captions

- **Fig_3 caption** (`fig:masking-sweep`, ~l. 937–939): "line colour denotes
  masking strategy (span, hetero-span, standard)" → "(span, standard)".
  Regenerate the figure (see 2.2).
- **Fig5_task_group_distributions caption** (`fig:group-bars`, ~l. 861–862):
  "Masking-strategy ablation variants are omitted here and reported in
  \cref{tab:pertask}" → "The span-masking ablation variant is omitted here…"
  (singular).

### 1.5 Conclusion (`sec:conclusion`, ~l. 1113–1116)

- "Internal ablations further show that downstream embedding quality is largely
  insensitive to masking strategy and to the small-to-base size increase" →
  suggest *"…largely insensitive to span versus token masking and to the
  small-to-base size increase"*.

### 1.6 Appendix C › Additional Benchmark Results (`app:additional-results`, ~l. 1297–1302)

- "including the span- and heteroatom-span masking variants of MMB-small" →
  "including the span-masking variant of MMB-small".
- "All four MMB checkpoints and the four baselines" → "All three MMB
  checkpoints…".

### 1.7 Appendix C › Fig_2 caption (`fig:four-model`, ~l. 1311–1329)

- "Internal comparison of the four MMB checkpoints" → **three**.
- Delete panel **(c)** (heteroatom-biased span masking).
- "All three panels are evaluated on the full 25 datasets" → "Both panels…".
- Rename the label `fig:four-model` (e.g. `fig:internal-comparison`) and update
  its references at ~l. 893 and ~l. 923.

### 1.8 Appendix E › Supplementary_1 caption (`fig:masking-heatmap`, ~l. 1359–1386)

- "columns compare masking strategies: span masking, hetero-span masking, and
  standard token masking" → "span masking and standard token masking".
- Delete "Hetero-span masking is consistently worse for both accuracy and loss."
- Regenerate the figure without the hetero-span column (see 2.3).

### 1.9 Appendix A › `tab:hparams` (~l. 1237–1249)

- "Swept ($3\times3$ grid, 9 runs each)" does not list masking strategy as a
  swept axis, although the small model was also swept over strategy.
  - Hetero-span runs were part of the 27-run small sweep.
  - Without them the small sweep reported in the paper is 18 runs (2 strategies
    × 3 × 3).
  - Suggest adding a row "Masking strategy: {standard, span} (small only)" and
    correcting the run count.

### 1.10 Places checked that need no change

- Abstract: no masking-strategy claims.
- Main results section, `tab:main-results`, `fig:bootstrap-ci`, `tab:bootstrap-cis`,
  `fig:baseline-paired`: use only released models and baselines.
- Discussion › Advances and Limitations: no hetero-span claims. The Limitations
  and Future Work sections could optionally list chemically informed
  (functional-group) masking as future work, replacing the dropped motivation
  from 1.1.

---

## 2. Figures

| Figure file | Label | Change | Generator |
|---|---|---|---|
| `figures/Fig_2.pdf` | `fig:four-model` | Drop panel (c); 2 panels | `scripts/paper/make_paper_figures.py`: now 2 panels by default. Rerun after `build_paper_results.py`. |
| `figures/Fig_3.pdf` | `fig:masking-sweep` | Remove hetero-span lines from the small-model panels (a, c) | Generator not identified in the repo. `R/FigX.R` already filters to `standard`/`span` but writes `figures/FigX_sweep_all.*`, not `Fig_3`. **Find or recreate the Fig_3 script.** |
| `figures/Supplementary_1.pdf` | `fig:masking-heatmap` | Remove the hetero-span column from the small-model heatmaps (e, g) | Generator not identified in the repo. **Find or recreate the script.** Source data: `results/sweep_results*.csv` (contain `hetero_span` rows; filter them out). |
| `figures/Fig5_task_group_distributions.pdf` | `fig:group-bars` | None (caption wording only, 1.4) | `modernmolbert.visualize.regen_groupfig` |
| `figures/Fig_baselines.pdf`, `bootstrap_ci_forest.pdf` | | None | |

## 3. Tables

| Table file | Change | Generator |
|---|---|---|
| `tables/pertask_table.tex` | Drop the `MMB-h` column and its legend entry | `scripts/paper/make_appendix_table.py` (column now omitted by default). **Caveat:** the committed `.tex` includes MUV and Tox21, but the script's `EXCLUDED_DATASETS` drops `ogbg-moltox21`/`ogbg-molmuv`. Reconcile before regenerating, or the table will silently lose two rows. |
| `tables/main_results_table.tex`, `table_bootstrap.tex` | None | |

## 4. Regeneration order (default = no hetero-span)

```bash
uv run python scripts/paper/build_paper_results.py
```

```bash
uv run python scripts/paper/make_paper_figures.py
```

```bash
uv run python scripts/paper/make_appendix_table.py
```

Then copy `outputs/eval/paper/table_pertask.tex` → `paper/tables/pertask_table.tex`
(after resolving the MUV/Tox21 caveat), and regenerate Fig_3 and
Supplementary_1 once their scripts are located.

To reproduce the previous four-checkpoint outputs, pass `--include-hetero-span`
to all three scripts.

## 5. Outside the manuscript

- **Hugging Face:** `HauserGroup/ModernMolBERT-small-hetero-span` was published
  as an ablation checkpoint. Decide whether to keep it (marked as not in the
  paper), make it private, or leave it undocumented. Its model card is now
  written only with `python -m modernmolbert.model_cards --include-hetero-span`.
- **Parameter mismatch found while auditing:**
  - `upload_model.MASKING_DEFAULTS["hetero_span"]` uses `mlm_probability=0.20`.
  - `model_cards.HETERO_SPAN_VARIANT` records `mlm=0.15`.
  - One of them is wrong; check the run config before any re-upload.
- `paper/source_data/`: no hetero-span rows; no change needed.
- `plans/ModernMolBERT-critical-review.md` (external review): its controlled
  ablation plan (E2) should not reintroduce hetero-span.

## Code status

hetero_span is off by default everywhere; the `--include-hetero-span` flags and
`INCLUDE_HETERO_SPAN` switch opt back in.

| Location | Default now | Opt-in |
|---|---|---|
| `train_selfies_ape_modernbert.py`, `collator.py`, `upload_model.py` | `standard` (unchanged; hetero_span was already a non-default choice) | `--masking_strategy hetero_span` |
| `scripts/sweeps/run_sweep.py` | sweeps `standard span` | `--masking standard span hetero_span` |
| `scripts/paper/build_paper_results.py` | no `MMB-small-hetero` column or stats | `--include-hetero-span` |
| `scripts/paper/make_paper_figures.py` | Fig_2 has 2 panels | `--include-hetero-span` |
| `scripts/paper/make_appendix_table.py` | no `MMB-h` column | `--include-hetero-span` |
| `src/modernmolbert/model_cards.py` | hetero-span card not written | `--include-hetero-span` |
| `analysis/validation/rerun_missing_embeddings.py` | no default target; `--embedder`/`--model-dir` required (default was hetero_span) | pass them explicitly |
| `analysis/check_missing_benchmarks_and_rerun.ipynb` | hetero_span not audited | `INCLUDE_HETERO_SPAN = True` |
| `scripts/maintenance/patch_model_max_length.py` | unchanged: one-off provenance script; skips missing paths | n/a |
| `analysis/sweep/collect_sweep_results.py`, `R/collect_sweep_results.R` | unchanged: parse whatever runs exist | n/a |
| `scripts/paper/build_benchmark_results_frames.py` | unchanged: wrangles all `praski_best_*` results and does not choose models; the paper scripts above do | n/a |
