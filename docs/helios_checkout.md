# Helios checkout

As of 6 October 2026, `/home/jakob/projects/ModernMolBERT` is the sole
ModernMolBERT checkout on Helios. It owns the ignored `data/`, `runs/`, and
`outputs/` directories, including completed revision and SMIRK results.
The former `ModernMolBERT-analysis` and `ModernMolBERT-multiseed` worktrees
were clean, pointed those directories back to the primary checkout, and were
retired after their queues completed. Their historical commits remain in Git.

Run training, embedding, scoring, and analysis from the primary checkout.
Before any GPU work, inspect `nvidia-smi` for existing compute processes.
Preserve the code commit recorded in each run's identity files; updating the
checkout does not change the provenance of completed results.
