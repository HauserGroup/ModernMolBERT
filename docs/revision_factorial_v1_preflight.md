# Five-model factorial preflight, 30 September 2026

This record supplies evidence for `MASTER_REVISION_PLAN.md` G4–G5. Pilot models are
diagnostics and are excluded from downstream selection. The frozen tokenizer/data
contract is in `docs/revision_factorial_v1_handoff.md`.

## Helios environment and inputs

- Host: `helios.tail670d76.ts.net`, checkout
  `/home/jakob/projects/ModernMolBERT`, clean commit `2b23d86` for the five
  200-step pilots; the resume proof uses clean commit `177390d`.
- GPU: RTX 5090, 32,607 MiB, driver 595.91.07, compute capability 12.0;
  CUDA 13.0, BF16 supported. CPU: 32 logical CPUs; RAM: 96,295,034,880
  bytes. Disk free at the later proof: 3,408,390,397,952 bytes.
- Python 3.13.14, Torch 2.12.1+cu130, Transformers 5.17.0, Datasets 5.0.1,
  NumPy 2.4.4. `flash_attn` is absent; the saved model resolves to `sdpa`.
  The environment JSON at
  `outputs/audit/revision_factorial_v1/helios_environment.json` on Helios has
  SHA-256 `50c487b43648ea4997be8859c530a0a87118ac2fdd018bcb567a43fd28985136`.
  The `uv.lock` SHA-256 is
  `52f1cdc215c65ba309f4c3ba6a33acf31add980e235f42c11b356ed51a80b588`.
- The two Parquet files, four tokenizer bundles, training-order array, 4,096
  validation-row array, prepared benchmark cohort and imported baseline CSV
  were checksum-matched after staging. Each pilot's `run_identity.json` records
  both Parquet hashes, the tokenizer and metadata hashes, the config and lock
  hashes, and both row-array hashes.

## GPU pilots

All five ran for 200 optimizer steps at context 384, seed 42, standard MLM 0.15,
BF16, microbatch 32 × accumulation 8, four data workers, the frozen training
order and validation rows, learning rate 4e-4 with 1,500 warmup steps. Each
consumed 51,200 streamed molecule presentations. The validation loss is a
software diagnostic only and must not be ranked across tokenizers.

| Pilot | Train loss | Validation loss | `train_runtime` | Checkpoint size |
|---|---:|---:|---:|---:|
| Small APE–SELFIES | 4.6414 | 3.3977 | 10.95 s | 0.41 GB |
| Small APE–SMILES | 6.6802 | 6.1106 | 10.19 s | 0.41 GB |
| Small BPE–SELFIES | 7.0280 | 6.5734 | 10.36 s | 0.42 GB |
| Small BPE–SMILES | 7.0174 | 6.6107 | 10.40 s | 0.42 GB |
| Base APE–SELFIES | 4.1906 | 2.7374 | 21.62 s | 1.37 GB |

Every run reached terminal step 200, and all logged loss/gradient values were
finite. The MLM heads reloaded with the matching tokenizer vocabularies (600,
1,376, 1,690, 1,602 and 600 respectively). The frozen encoder adapter produced
finite vectors for the same five SMILES inputs in every run: `CCO`, `C.CN`,
`[13CH3][NH3+]`, `F/C=C/F`, and a 250-carbon chain. No row was rejected; small
vectors were 512-dimensional and base vectors 768-dimensional. Model head
weights are expected to be unused when loading the encoder-only adapter.

Pilot artifacts and logs are under `runs/revision_factorial_v1/preflight/` on
Helios. Source data and trained weights remain ignored and are not committed.

The GPU sampler observed 2,124 MiB peak for the four small pilots and 4,538
MiB peak for a repeated 200-step base pilot; idle memory was 18 MiB. The base
repeat on commit `177390d` reached terminal step 200 with 21.67 seconds of
`train_runtime`, 0.71 seconds of validation runtime, 51,200 molecule
presentations and 1,413,791 nonpadding tokens seen. The sampler uses
`nvidia-smi` at 100 ms for base and one second for the small grid, so the
small peak may miss shorter transients. Both leave substantial margin under
32,607 MiB, but the production run should still monitor memory.

Scaling the measured 200-step runtimes to 30,000 steps gives about 2.65 GPU
hours for all five runs together. A 30% operational allowance brings the
working GPU reservation to **3.45 hours**, excluding CPU downstream evaluation
and any rerun. This is a planning estimate, not a throughput guarantee: the
pilots cover only the first 51,200 ordered molecules, and full runs may have
different padding and I/O behavior. The five pilot directories together use
about 5.8 GB; individual dataset caches used 143–564 MB. Three retained
checkpoints per production run would occupy about 9.2 GB at pilot sizes,
plus approximately 1 GB of final weights, caches, logs, embeddings, backup
copies and temporary files. Reserving at least 50 GB for the campaign and
another 50 GB for a durable copy leaves far more than 20% of Helios's
3.4 TB free space. The CPU evaluation time must be measured after G7 scoring
is ready.

## Interrupted versus uninterrupted proof

Commit `177390d` adds opt-in `--trace_source_rows`; production runs leave it
off. A 200-step small APE–SELFIES reference and an identical run interrupted
immediately after its complete step-20 checkpoint used the same frozen row
order, four workers and effective batch 256. The interrupted process exited
on SIGTERM; its checkpoint's trainer state, optimizer, scheduler and RNG files
were present. The trace from any work after step 20 was preserved separately,
and the active trace was truncated to the saved checkpoint boundary before
resuming from `checkpoint-20`.

Both completed at step 200 with 1,600 traced microbatches and 51,200 source
row IDs. The ordered source-row IDs matched exactly, as did the logged LR at
steps 180 and 200. The final `model.safetensors`, checkpoint-200 optimizer,
scheduler and RNG files were byte-identical (SHA-256 comparison); maximum
absolute difference between corresponding model tensors was zero. The
aggregate `train_loss` in `train_results.json` differs because Trainer
reports only the resumed segment for that field; the terminal weights and
state are identical. Diagnostic paths are `resume_continuous_200/` and
`resume_interrupted_200/` under the same preflight root.

## Frozen benchmark embedding smoke

The real `embed_modernmolbert` CLI loaded each pilot final bundle and embedded
the frozen 640-row Bioavailability_Ma prepared dataset with mean pooling at
context 384. Each representation retained the same 639 source rows and rejected
only original source row 84 for tokenizer coverage, with zero over-context
rows. Row 84 is an Au-containing training molecule; no validation or test
row was removed. All five outputs record the same prepared-data SHA-256,
`c524d9881200b88e76b5fe3e2d2409c14433a95f04e0aced6437a1c68b8240f8`.
They are stored as `PREFLIGHT_*` embeddings on Helios and must not enter paper
scores. This confirms one endpoint's common-row path; G7 still requires
coverage and shared supervised rows/folds across all 25 endpoints.

The same five pilot checkpoints then embedded all **25 frozen paper tasks**,
excluding ogbg-moltoxcast because the imported Praski table has no baseline
for it. All 125 embedding jobs completed. `scripts/build_revision_coverage_manifest.py`
verified their embedding/prepared-data hashes, finite vectors and source-row
maps, and wrote
`outputs/audit/revision_factorial_v1/pilot_benchmark_coverage.json` on Helios
(SHA-256 `214626c96d35ac8b2cf67503a9e9c694c75025c152c13b2e446d9c8aa9343b69`).
Across the 241,949 source rows assigned to supervised train/validation/test
splits, 240,659 are common to all five tokenizers: **1,290 exclusions (0.53%)**.
There were zero over-context rows. The largest proportional common losses
were SIDER, 62/1,427 (4.34%), and HIV, 895/41,120 (2.18%). The original
hERG JSON has 22 source rows assigned to no supervised split; the manifest
records them but keeps the original split unchanged. MUV test endpoints 0
and 12 have no positive class even before embedding, so this was not caused
by common-row filtering. The five full-cohort embedding passes took about
6.1 minutes combined on Helios; the 25-task scoring and common-fold timing
is still unmeasured. Final-model embeddings and source-row hashes must be
rechecked after production; these pilot weights never enter paper scores.

## Remaining launch gates

- Measure CPU evaluation runtime after G7 scoring is ready, then revise the
  combined wall-clock reservation. Monitor production throughput and GPU memory
  against the pilot values.
- Finish the G7 common-row scoring, coverage and resume safeguards before
  production results can be accepted. The 25 prepared benchmark splits remain
  frozen; the migrated cache preserves the original JSON row order and split.
- `scripts/run_revision_factorial_v1.py` pins every G3 argument, the five run
  IDs, tokenizer representation/size and the frozen input hashes. Its dry run
  validates inputs and prints the command. An actual launch also requires a
  clean checkout and a commit-matched `production_gate.json` with pilot,
  capacity and G7 coverage evidence hashes; the gate is deliberately absent
  until the remaining checks pass. Keep each production run on the same clean
  commit and archive complete recovery checkpoints.
