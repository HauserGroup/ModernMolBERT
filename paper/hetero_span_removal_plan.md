# Plan: move hetero-span masking to the supplement

**Decision (2026-09-28):** the heteroatom-biased span-masking variant
(`hetero_span`, labelled *MMB-small-hetero* / *MMB-h*) leaves the main text and
all claims of the revised manuscript. It stays in the supplement as a
clearly labelled exploratory result. It is **not** removed silently.

**Why demote rather than delete:**

- Preprint v1 (ChemRxiv, 19 June 2026, `paper/published_manuscript.pdf`) reports
  hetero-span in Methods, Ablations, Figs 7, 9 and 10, and Table S3. The
  checkpoint `HauserGroup/ModernMolBERT-small-hetero-span` is public. Readers can
  compare versions.
- It was the best small variant downstream (77.9 vs 77.4 for standard masking in
  v1) and the main example for "validation ranking does not carry to downstream".
  Dropping it without comment would invite a selective-reporting question.
- Still, it should not support claims:
  - it is single-seed and its +0.5 delta sits within the dataset-to-dataset scatter;
  - cross-strategy validation MLM losses are not comparable (critical review item 9);
  - the review's minimum route is to shorten the masking section anyway.

Line numbers refer to `paper/main.tex` as of commit `f3ff4ac`. They will drift,
so each item also names its section or label.

The main-text internal comparison becomes **three** checkpoints (small-standard,
small-span, base). The main-text masking ablation is a single contrast,
**standard vs span**. Hetero-span appears only in the supplementary per-task
table and the sweep heatmap, each with a disclosure.

> **Also required:** all benchmark numbers must be regenerated with CV-based head
> selection first (see the TODO at the top of `revision_notes.md`). The v1
> numbers quoted here (77.9, +0.5, −0.3) will change.

---

## 1. Text changes in `main.tex`

### 1.1 Introduction: masking motivation (~l. 309–317)

- The paragraph motivates chemically informed masking with the functional-group
  masking of `pengPretrainedMolecularLanguage2025`. Hetero-span was the paper's
  only test of that idea, and it is no longer used for claims.
- **Reword:** keep BERT vs SpanBERT as the motivation and frame the ablation as
  *token vs contiguous-span masking*. Move the Peng et al. sentence to Future
  Work, or keep it with a pointer to the supplementary exploratory variant.
- Rewrite "Whether such structure-aware masking benefits a frozen SELFIES
  embedder is an open question we examine in an ablation" to refer to span
  masking.

### 1.2 Methods › Masking strategies (`sec:pretraining-procedure`, ~l. 607–623)

- "We implement three masking strategies" → **two** evaluated strategies.
- Replace the *Heteroatom-biased span* description (~l. 614–620) with a
  footnote. Suggested text: *"The collator also implements a heteroatom-biased
  span variant, reported in the preprint of this work; it is included in
  \cref{tab:pertask} as an exploratory result only."*
- "All three apply the standard BERT corruption rule" → "Both apply…".
- "\cref{sec:ablations} compares all three" → "compares both".

### 1.3 Results › Ablation Studies (`sec:ablations`, ~l. 888–923)

This section contains the conclusions that change most.

- ~l. 902–904: delete "and heteroatom-biased span masking consistently weakest".
- ~l. 906–910: delete "and heteroatom-biased span masking by $+0.5$".
- ~l. 910–913 (**conclusion to reword**): "Notably, the validation-MLM ranking
  does not carry through to downstream performance…" used hetero-span as its
  strongest example. Critical review item 9 also notes that cross-strategy
  validation losses are not comparable.
  - Either drop the validation-vs-downstream argument or state it descriptively
    for span only. Suggested text: *"Span masking attains the highest validation
    MLM accuracy yet marginally trails standard masking downstream ($-0.3$);
    because each strategy is validated on its own corruption, we do not read the
    validation ranking as a quality ranking."*
- ~l. 914–919: "no reliable downstream effect of masking strategy" and "largely
  insensitive to masking strategy" → say *"span versus token masking"*.
- ~l. 893 and 923: both reference `fig:four-model`; update them when that label
  is renamed (1.7).

### 1.4 Results figure captions

- **Fig_3 caption** (`fig:masking-sweep`, main text, ~l. 937–939): "line colour
  denotes masking strategy (span, hetero-span, standard)" → "(span, standard)".
  Regenerate the figure without hetero-span lines (see 2).
- **Fig5_task_group_distributions caption** (`fig:group-bars`, ~l. 861–862): "Masking-strategy
  ablation variants are omitted here and reported in \cref{tab:pertask}" is
  still correct (span plus the exploratory hetero-span variant). No change.

### 1.5 Conclusion (`sec:conclusion`, ~l. 1113–1116)

- "…largely insensitive to masking strategy and to the small-to-base size
  increase" → *"…largely insensitive to span versus token masking and to the
  small-to-base size increase"*.

### 1.6 Appendix C › Additional Benchmark Results (`app:additional-results`, ~l. 1297–1302)

- Replace "including the span- and heteroatom-span masking variants of
  MMB-small. All four MMB checkpoints and the four baselines are evaluated on the
  full set of 25 datasets" with a disclosure. Suggested text:

  > *\Cref{tab:pertask} gives the full per-task test ROC-AUC for the three
  > \model{} checkpoints analysed in the main text and the four baselines. For
  > completeness it also lists an exploratory heteroatom-biased span-masking
  > variant (MMB-h), reported in the preprint version of this work. MMB-h was
  > trained with a single seed, is not used for any claim, and is omitted from
  > all aggregate comparisons.*

### 1.7 Appendix C › Fig_2 caption (`fig:four-model`, ~l. 1311–1329)

- "Internal comparison of the four MMB checkpoints" → **three**.
- Delete panel **(c)**. The figure is now 2 panels; the exploratory variant
  appears only in `tab:pertask`.
- "All three panels are evaluated on the full 25 datasets" → "Both panels…".
- Rename the label `fig:four-model` (e.g. `fig:internal-comparison`) and update
  its references at ~l. 893 and ~l. 923.

### 1.8 Appendix E › Supplementary_1 caption (`fig:masking-heatmap`, ~l. 1359–1386)

- The heatmap may **keep** the hetero-span column: it is supplementary and
  documents the full sweep, and no regeneration is needed.
- Caption:
  - mark the column as exploratory, e.g. "…span masking, an exploratory
    hetero-span variant (not used for any claim), and standard token masking";
  - delete "Hetero-span masking is consistently worse for both accuracy and
    loss." Per critical review item 9, cross-strategy losses are not comparable,
    so this comparative claim should go for all strategies.
- If you prefer a strictly two-strategy supplement, regenerate without the
  column instead (the generator is not in the repo; see 2).

### 1.9 Appendix A › `tab:hparams` (~l. 1237–1249)

- "Swept ($3\times3$ grid, 9 runs each)" does not list masking strategy, although
  the small model was also swept over strategy (27 runs including hetero-span).
- Suggest adding a row "Masking strategy (small only): {standard, span,
  hetero-span$^{\ddagger}$}", with the footnote "$^{\ddagger}$exploratory;
  supplementary only".

### 1.10 Places checked that need no change

- Abstract: no masking-strategy claims.
- `tab:main-results`, `fig:bootstrap-ci`, `tab:bootstrap-cis`,
  `fig:baseline-paired`: released models and baselines only.
- Discussion › Advances and Limitations: no hetero-span claims. Optionally list
  chemically informed (functional-group) masking under Future Work.

---

## 2. Figures

| Figure file | Label | Change | Generator |
|---|---|---|---|
| `figures/Fig_2.pdf` | `fig:four-model` | Drop panel (c); 2 panels | `scripts/paper/make_paper_figures.py`: 2 panels by default |
| `figures/Fig_3.pdf` (main text) | `fig:masking-sweep` | Remove hetero-span lines from the small-model panels (a, c) | Not found in the repo. `R/FigX.R` filters to `standard`/`span` but writes `figures/FigX_sweep_all.*`. **Find or recreate.** Source data: `results/sweep_results*.csv` |
| `figures/Supplementary_1.pdf` | `fig:masking-heatmap` | None if the column is kept (caption only, 1.8) | Not found in the repo |
| `figures/Fig5_task_group_distributions.pdf` | `fig:group-bars` | None | `modernmolbert.visualize.regen_groupfig` |
| `figures/Fig_baselines.pdf`, `bootstrap_ci_forest.pdf` | | None | |

## 3. Tables

| Table file | Change | Generator |
|---|---|---|
| `tables/pertask_table.tex` | Keep the `MMB-h` column. The legend now marks it "exploratory … (reported in preprint v1; single seed, not used for any claim)" | `scripts/paper/make_appendix_table.py` (column included by default; `--exclude-hetero-span` drops it). **Caveat:** the committed `.tex` includes MUV and Tox21, but the script's `EXCLUDED_DATASETS` drops them. Reconcile before regenerating. |
| `tables/main_results_table.tex`, `table_bootstrap.tex` | None (hetero-span never included) | |

## 4. Regeneration order (defaults implement this plan)

After the CV head-selection re-run of `build_benchmark_results_frames.py`:

```bash
uv run python scripts/paper/build_paper_results.py
```

```bash
uv run python scripts/paper/make_paper_figures.py
```

```bash
uv run python scripts/paper/make_appendix_table.py
```

Then copy `outputs/eval/paper/table_pertask.tex` to
`paper/tables/pertask_table.tex` (after the MUV/Tox21 caveat) and regenerate
Fig_3.

## 5. Outside the manuscript

- **Hugging Face:** keep `HauserGroup/ModernMolBERT-small-hetero-span` public,
  because v1 cites it. Its generated card now opens with an "Exploratory
  ablation" note: reported in v1, supplementary-only in the revision, not used
  for claims.
  - Regenerate the card with
    `python -m modernmolbert.model_cards --include-hetero-span`.
  - Push the updated README to the Hub repo (manual step, not done).
- **Parameter mismatch found while auditing:**
  - `upload_model.MASKING_DEFAULTS["hetero_span"]` uses `mlm_probability=0.20`.
  - `model_cards.HETERO_SPAN_VARIANT` records `mlm=0.15`.
  - Check the run config before regenerating or uploading the card.
- `paper/source_data/`: no hetero-span rows; no change needed.
- `plans/ModernMolBERT-critical-review.md`: its controlled ablation plan (E2)
  should not reintroduce hetero-span.

## Code status

| Location | Default now | Override |
|---|---|---|
| `train_selfies_ape_modernbert.py`, `collator.py`, `upload_model.py` | `standard` masking | `--masking_strategy hetero_span` |
| `scripts/sweeps/run_sweep.py` | sweeps `standard span` | `--masking standard span hetero_span` |
| `scripts/paper/build_paper_results.py` | `MMB-small-hetero` written to `results_matrix_25task.csv` (for the appendix) but excluded from group means, Table 2 and all stats | `--include-hetero-span` adds it to means and stats |
| `scripts/paper/make_paper_figures.py` | Fig_2 has 2 panels | `--include-hetero-span` adds panel (c) |
| `scripts/paper/make_appendix_table.py` | `MMB-h` column included, labelled exploratory | `--exclude-hetero-span` drops it |
| `src/modernmolbert/model_cards.py` | hetero-span card not written; when written, carries the exploratory note | `--include-hetero-span` |
| `analysis/validation/rerun_missing_embeddings.py` | no default target; `--embedder`/`--model-dir` required | pass them explicitly |
| `analysis/check_missing_benchmarks_and_rerun.ipynb` | hetero_span not audited | `INCLUDE_HETERO_SPAN = True` |
| `scripts/maintenance/patch_model_max_length.py`, `analysis/sweep/collect_sweep_results.py`, `R/collect_sweep_results.R`, `scripts/paper/build_benchmark_results_frames.py` | unchanged: provenance scripts, or process whatever runs exist | n/a |
