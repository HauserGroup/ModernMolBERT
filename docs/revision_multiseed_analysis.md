# Five-seed evaluation handoff

The manuscript repository's `MASTER_REVISION_PLAN.md` is the status tracker. This file gives the executable order after the 25 full encoders are accepted. Keep the Helios training worktree at commit `2260bb3` until its 20-run queue finishes. Seed-42 scoring completed at the original scoring checkout's `f47da23`; its saved identities name that commit.

**CV metadata correction before further analysis:** The completed scorer passed `cv=5` to scikit-learn, which used unshuffled folds. The seed-42 evaluation manifest incorrectly recorded shuffled folds, although the score files themselves were computed with the intended unchanged scorer. Once Helios access is restored, update the separate CPU analysis checkout to the CV-correction commit and run:

```bash
uv run --locked python scripts/correct_revision_cv_manifest.py \
  outputs/revision_factorial_v1/evaluation_manifest.json \
  --expected-sha256 6010628869ef50cc6c61821fba59e8c22d1d4e99e518d5c7adae4ac5220041e4
scripts/run_revision_multiseed_analysis.sh select-seed 42
```

The repair changes only `cv`, saves the original manifest next to it and prints the new SHA-256. The second command reselects heads against that corrected manifest and refreshes the selection evidence. Do not use the earlier seed-42 selected-head manifest in the five-seed aggregate. New seed-43–46 evaluation manifests record the corrected policy directly; all five seeds must then pass the same-policy aggregate gate.

1. Confirm `outputs/revision_factorial_multiseed_v1/queue_status.txt` says `complete 20/20 ...`, all 20 final run identities select step 30,000, and their `/data` backups match. Confirm `outputs/revision_factorial_v1/final_scoring_queue.status` says `complete 125/125 ...`. Only then move an analysis checkout to the committed revision-analysis code; do not change either active checkout mid-queue.
2. Check `nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader`. When it reports no compute process, run the four seed-specific embedding passes serially from the updated analysis checkout. Each pass checks again before every model and refuses a different training/evaluation cohort:

   ```bash
   for seed in 43 44 45 46; do
     scripts/run_revision_multiseed_embeddings.sh "$seed"
   done
   ```

3. Score each verified common cohort on CPU. This stage sets `CUDA_VISIBLE_DEVICES` empty and writes one log per model/task pair. `N_JOBS` changes CPU parallelism, not the frozen scoring identity. Re-run the same command to resume verified pairs after interruption:

   ```bash
   for seed in 43 44 45 46; do
     N_JOBS=16 scripts/run_revision_final_scoring.sh "$seed"
   done
   ```

4. Select heads and make matched internal matrices only after each seed's 125-pair queue completes. The script verifies all 125 score files, the frozen evaluation manifest and embedding hashes, imported baseline table, and split-overlap audit. It writes table-only external scores separately from five-model common-row scores:

   ```bash
   for seed in 42 43 44 45 46; do
     scripts/run_revision_multiseed_analysis.sh select-seed "$seed"
   done
   scripts/run_revision_multiseed_analysis.sh aggregate
   scripts/run_revision_multiseed_analysis.sh paper
   ```

The aggregate stage refuses a missing seed, changed common test rows, labels, split assignments, endpoint viability, CV policy, or selected matrix whose SHA-256 differs from its selection manifest. Its `mean_common_task_matrix.csv` averages the five seed scores per task/model; `overall_contrast_summary.csv` retains between-seed SD. The paper stage checks the aggregate's recorded matrix SHA-256, the current evaluation and selection manifest hashes for all five seeds, and the seed-42 imported-baseline task matrix before writing task and fixed-family bootstrap intervals on the five-seed mean. Seed variation is reported separately. It generates tables, figures, and source data under `outputs/eval/revision_factorial_multiseed_v1/paper/`. These outputs require scientific review and a deliberate manuscript replacement; they do not overwrite the archived manuscript tables.

The four imported baselines remain table-only context. Their molecule identities and checkpoint revisions cannot be made identical to the five new models from the available source table. Do not use their differences as paired common-molecule contrasts or as evidence of pretraining-seed variation.
