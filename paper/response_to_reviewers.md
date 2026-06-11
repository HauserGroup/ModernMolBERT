# Response to reviewer comments — ModernMolBERT

Prepared 11 June 2026. Covers all comments in `comments_SA.md` and `comments_ASH.md/.csv`.
All edits were made directly in `main.tex`, `references.bib`, and `tables/`, and the
manuscript was re-compiled cleanly (`pdflatex → bibtex → pdflatex ×2`, 0 errors,
0 undefined references/citations).

Legend: **Done** = implemented in source · **Done (note)** = implemented with a
decision worth your attention · **Flagged** = needs your input (figure regeneration,
new experiments, or a number only you have).

---

## 1. SA comments (factual checks)

| # | Comment | Outcome |
|---|---------|---------|
| SA-1 | Consistent British English | **Done (note).** Standardised to Cambridge British (`-ise`/`-isation`) throughout, matching the document's existing majority style (optimisation, visualisation, generalisation). Converted: *revolutionised, canonicalised, canonicalisation, sanitised, sanitisation, initialised, normalisation,* and — for full consistency — *tokeniser/tokenisation/tokenised* (incl. table headers, captions, and the related table file). CRediT taxonomy terms (Conceptualization, Methodology, …) were **left in their official spelling**, as required by the taxonomy. If you would rather keep the technical "tokenizer" spelling, it is a single find-and-replace to revert. |
| SA-2 | Check MolFormer-XL size — is it 46.8M? | **Done.** Verified: MoLFormer-XL is **46.8M** parameters (IBM model card / Ross et al. 2022). The manuscript said 44.4M in two places — corrected in the text (§Model Architecture) and in the appendix embedders table. Also reordered the size list so it stays ascending (Chemformer 45M < MoLFormer-XL 46.8M < Uni-Mol 47.6M). |
| SA-3 | Is the author list correct for the Gilmer "Neural message passing" reference? | **Done.** It was **wrong**. The stored authors (Schütt, Height, Zitnik, Coley, Barzilay, Jaakkola) do not match the paper. Correct authors are **Gilmer, Schoenholz, Riley, Vinyals, Dahl** (ICML 2017, PMLR v70). Fixed in `references.bib`. |
| SA-4 | Does the "SimSon (Kovacs et al., JCIM, 10.1021/acs.jcim.4c02247)" reference exist? | **Done (note).** **No — that entry was fabricated.** No SimSon paper by Kovacs/Schneider/Stiefl/Jiménez-Luna in JCIM with that DOI exists. The real paper is **Lee, Kim, Min & Han, "SimSon: simple contrastive learning of SMILES for molecular property prediction," *Bioinformatics* 41(5):btaf275, 2025, doi:10.1093/bioinformatics/btaf275.** Replaced the bib entry and updated the citation key (`kovacs…` → `lee…`). This matches the SimSon row already in the appendix embedders table (SMILES, contrastive). **Please confirm** this is the paper you intended to cite for randomised-SMILES/enumeration augmentation. |

---

## 2. ASH comments (point-by-point)

| # | Line / location | Comment | Outcome |
|---|---|---|---|
| T001 | Intro | Single-paragraph subheaders unconventional | **Done.** Removed all `\paragraph{}` subheaders in the Introduction; it now reads as flowing prose. (Methods subheaders kept — conventional there.) |
| T002 | Intro (ECFP) | Have references in order, starting from [1] | **Done.** Switched `\bibliographystyle{plainnat}` → `unsrtnat`, so citations are numbered in order of appearance ([1] = ECFP, the first cited). |
| T003 | Table 1 | More informative with parameter / dimension size? | **Done.** Added **Params** and **corpus Size (M)** columns to Table 1 (values from your vetted appendix table). Embedding dimensions are already in the appendix table; Table 1's caption now points there to avoid an over-wide table. |
| T004 | Intro "Atom Pair Encoding" | Closer to methods than intro | **Done (note).** Folded into prose as motivation; the implementation detail stays in §Methods (Atom Pair Encoding Tokeniser). Kept the conceptual motivation in the intro intentionally — happy to cut it entirely if you prefer. |
| T005 | Intro | Reference Fig 1 before Fig 2 | **Done.** Added an early `\cref{fig:overview}` (Figure 1) in the second intro paragraph, so Figure 1 is now cited before Figure 2 (the design figure). |
| T006 | Intro | You don't reference Figure 1 anywhere | **Done.** Same fix as T005 — Figure 1 is now explicitly referenced early (it was only referenced late before). |
| T007 | Intro (RoPE) | Explain *why* RoPE can solve this limitation | **Done.** Added a sentence: RoPE conditions attention on *relative* offsets, making the encoder invariant to where a traversal starts (the chemically meaningless part of string position) while preserving local ordering. |
| T008 | Intro "Contributions" | Unconventional | **Done.** Removed the "Contributions" subheader (part of the T001 fix). |
| T009 | Fig 1C | "chemically principled like atom-level" could be shown better in the APE merge panel | **Flagged.** Figure-content change — needs regeneration of `Fig1_C` (source `.pptx` is in `figures/`). |
| T010 | Methods pipeline | After MLM training, can you decode to valid molecules? | **Done.** Added a note in §Discussion (Scope): the model is used purely as an encoder, but because inputs and the MLM head operate over SELFIES, any predicted sequence decodes to a valid molecule by construction; generative use is left to future work. |
| T011 | Eval protocol | Are the downstream hyperparameters explained anywhere? | **Done.** Clarified that the three heads' CV search grids are taken directly from the Praski et al. benchmark code; added the pointer in §Evaluation Protocol. |
| T012 | §Related Work header | Why its own header? Overlaps with intro | **Done (note).** Added a lead-in framing the section as extending the intro along four axes and naming the baselines. I **retained** the section rather than dissolving it — the comment was tentative and a Related Work section is standard; full integration risks the structure and many cross-references. Say the word and I'll merge it into Intro/Discussion. |
| T013 | Fig 2 (Fig1_A/B) | Panel A stretched; both panels low-res | **Flagged.** Regenerate/export `Fig1_A`, `Fig1_B` at higher resolution and corrected aspect ratio (source `.pptx` in `figures/`). |
| T014 | §Rel. representations | Reference or own calculation for the validity stats? | **Done.** Reworded to attribute the 26.6%/0.2%/100% figures explicitly to Krenn et al. ("as reported by …"). |
| T015 | §Molecular Language Models | Better suited for discussion | **Done (note).** Retained in Related Work (model descriptions are conventional there, not in Discussion); see T012. Flagged for your call. |
| T016 | §Rel. LMs (scope) | Very true — esp. whether benchmark molecules are in the ChEMBL training set | **Done.** Added a train–benchmark overlap caveat to §Limitations: corpus and benchmarks share medicinal-chemistry sources, structures are not deduplicated across them, so absolute scores may be modestly inflated for all pretrained encoders (relative comparisons less affected). |
| T017 | §Molecular Tokenisation | Integrate into intro/discussion | **Done (note).** Retained; see T012. |
| T018 | §Methods APE | "several modifications" — specify | **Done.** Enumerated the modifications inline (molecular-aware pre-tokenisation, frequency/span-controlled vocab, greedy longest-match inference, post-training symbol injection). |
| T019 | §Pretraining data | Is it just ~10k compounds filtered that way? | **Flagged.** Needs the raw pre-filter molecule count (only you have it) to state how many were removed. I did not fabricate a number. |
| T020 | §Eval protocol | Ridge/RF/kNN — why those specifically? | **Done.** Justified: three lightweight heads spanning linear, ensemble, and instance-based biases, adopted from the Praski et al. protocol so the comparison to their baselines is exact. |
| T021 | §Benchmarks | "1 to 27 (SIDER)" — which? ref? | **Done (minor).** The dataset (SIDER, 27 tasks) is named and the per-dataset statistics are referenced (Table 4 of Praski et al.) in the same sentence; no change needed beyond confirming the pointer. |
| T022 | §Baselines | Integrate directly into Methods? | **Flagged (recommendation).** Not moved. Benchmarks + Baselines currently sit under §Experiments, which is a standard home for them. Moving both into §Methods is reasonable but empties §Experiments; recommend deciding before I relocate. |
| T023 | Fig (forest plot) | Show an example ROC to understand the aggregation | **Flagged.** New supplementary panel (an example per-dataset ROC curve) — needs to be generated from your eval outputs. |
| T024 | Results | Should continue from 1 | **Done.** Interpreted as citation numbering; fixed by the `unsrtnat` switch (T002). If it referred to something else, let me know. |
| T025 | Results | Is the per-dataset, per-baseline data shown anywhere? | **Done (answer).** Yes — `tab:pertask` (appendix) gives full per-task ROC-AUC for every model; the text already points to it. No change needed. |
| T026 | Forest plot | Annotate the positive MMB direction in the figure | **Flagged.** Figure annotation — add a "← favours baseline / favours ModernMolBERT →" label when regenerating `bootstrap_ci_forest`. |
| T027 | Results | Why the difference in number of datasets again? | **Done.** Added an inline reminder that Tox21 is omitted for ModernMolBERT-base (24 vs 25), pointing to `tab:pertask`. |
| T028 | Results (corpus) | Could that be stated in the initial table? | **Done.** MoLFormer's much larger corpus is now visible in Table 1 via the new Size (M) column ( >1,100M vs 2.4M). |
| T029 | Results | REF for "frozen embeddings transfer best to global-character tasks" | **Done.** Framed as our interpretation and supported with the broader benchmarking analyses (Yang et al.; Deng et al.). |
| T030 | Results (PaCMAP) | Not all of these are global properties | **Done.** Reworded: the descriptors span both whole-molecule properties (lipophilicity, drug-likeness, polarity, size) and more local counts (flexibility, aromaticity); dropped the over-broad "global". |
| T031 | Fig (Fig_baselines) | Very small font | **Flagged.** Increase font toward legend size when regenerating `Fig_baselines`. |
| T032 | Fig ordering | Fig_baselines underlies Fig 3 — should it come first? | **Done (note).** In the source the per-task baseline figure already precedes the group-bar figure and is referenced first; final placement depends on LaTeX float packing. If you want it locked, I can pin both with `[H]`. |
| T033 | Fig (Fig_4) | y-axis labels very small | **Flagged.** Enlarge axis labels when regenerating `Fig_4`. |
| T034 | Fig 4 caption | Spell out QED | **Done.** Now "the quantitative estimate of drug-likeness (QED)". |
| T035 | Fig (embedding space) | So what's the interpretation? | **Done.** Added an explicit qualitative interpretation: neighbouring embeddings correspond to chemically similar molecules (what makes them useful for the similarity-based heads); no quantitative claim from projected coordinates. |
| T036 | Fig_2 (4-checkpoint) | More of a supplementary figure | **Done.** Moved `fig:four-model` to the appendix (Additional Benchmark Results); the in-text reference now points there. |
| T037 | Discussion | Funny wording ("It is important to read every number…") | **Done.** Reworded to "Every number in this paper should be read as a frozen-embedding result." |
| T038 | Discussion | REF (MoLFormer 400× more molecules) | **Done.** Added the MoLFormer citation at that claim. |
| T039 | Discussion | Could you train MMB on ZINC instead? | **Done (direction) / Flagged (experiment).** Added to Future Work (matched-corpus / broader-corpus pretraining). The actual ZINC run is a new experiment beyond the edit. |
| T040 | Discussion | Wording ("advantage of our is accessibility") | **Done.** Fixed to "A practical advantage of ModernMolBERT is accessibility." |
| T041 | §Limitations | Could one train on the same corpus to compare more directly? | **Done (direction) / Flagged (experiment).** Added an explicit matched-corpus comparison to Future Work, separating data scale from model design. |
| T042 | §Limitations | What about peptides? | **Done.** Added peptides and macrocycles to the under-represented chemical spaces in §Limitations. |
| T043 | Future Work | Yes — how does it scale? | **Done.** Added a scaling direction (model size, corpus size, compute) asking whether the gap to large-scale SMILES models narrows with scale. |

---

## 3. Items needing your action (summary)

**Figures (regenerate from the `.pptx` sources in `figures/`):**
T009 (Fig1_C APE panel), T013 (Fig1_A/B resolution + panel A aspect), T023 (new example-ROC supplementary panel), T026 (annotate direction on forest plot), T031 (Fig_baselines font), T033 (Fig_4 axis labels).

**New experiments (flagged, only scoped in text):**
T039 (train on ZINC), T041 (matched-corpus comparison), T043 (scaling study).

**Numbers / decisions only you can supply:**
T019 (raw pre-filter molecule count), T022 (whether to move Benchmarks+Baselines into Methods), T012/T015/T017 (whether to dissolve Related Work into Intro/Discussion).

**Please confirm:** SA-4 — that Lee et al. (Bioinformatics 2025) is the SimSon paper you intended.

---

## 4. Notes on global decisions

- **British English:** standardised to `-ise`/`-isation` (Cambridge), including the technical term *tokeniser*. CRediT terms left in official spelling. Reversible if you prefer "tokenizer".
- **Bibliography style:** `unsrtnat` (citation-order numbering). This changes every citation number in the PDF relative to the old alphabetical `plainnat`.
- **Related Work:** retained and tightened rather than dissolved (T012/T015/T017), pending your decision.
- The minimal `siunitx` stub used to compile in this environment is **not** part of your project; your Overleaf/TeX Live build is unaffected.
