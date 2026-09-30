# Documentation

| Document | Contents |
|---|---|
| [datasets.md](datasets.md) | ChEMBL 36 SELFIES pretraining data and the benchmark datasets |
| [tokenizer.md](tokenizer.md) | APE and BPE tokenizers: training, validation, saving and loading |
| [masking_strategies.md](masking_strategies.md) | MLM masking strategies (standard, span, heteroatom span) |
| [evaluation.md](evaluation.md) | Benchmark pipeline: download, embed, score, output schema |
| [baselines.md](baselines.md) | The imported Praski et al. baselines and how they enter the comparison |
| [upload.md](upload.md) | Uploading models and tokenizers to the Hugging Face Hub |
| [tests.md](tests.md) | CI commands, readiness gate before training, smoke tests |

Five-model revision (priorities and status live only in `MASTER_REVISION_PLAN.md` in the
manuscript repository):

| Document | Contents |
|---|---|
| [revision_factorial_v1_handoff.md](revision_factorial_v1_handoff.md) | Frozen inputs: corpus, row IDs, four tokenizers, benchmark cohort |
| [revision_factorial_v1_preflight.md](revision_factorial_v1_preflight.md) | Helios environment, pilots, resume proof, benchmark smoke, remaining launch gates |
| [revision_run.md](revision_run.md) | From embeddings to paper tables and figures |
| [revision_run_record.md](revision_run_record.md) | The superseded one-model run: frozen corpus, tokenizer, length statistics |

Code review:

| Document | Contents |
|---|---|
| [code_audit_2026-09-29.md](code_audit_2026-09-29.md) | Open audit findings; the closed ones are in git history |
| [simplification_candidates_2026-09-30.md](simplification_candidates_2026-09-30.md) | What could be retired or merged, and when |
