#!/usr/bin/env python3
"""One-off fix: force ``model_max_length: 128`` in tokenizer_config.json files.

Run from the repo root. The paths below are hardcoded to the specific released
sweep run directories whose tokenizer_config.json shipped without an explicit
``model_max_length``. Kept for provenance; not part of any pipeline.
"""

import json

from pathlib import Path

PATHS = [
    "runs/chembl36_small_mask_mlm_lr_sweep/modernmolbert_best_standard/final_model/tokenizer_config.json",
    "runs/chembl36_small_mask_mlm_lr_sweep/modernmolbert_best_standard/final_model/ape_tokenizer/tokenizer_config.json",
    "runs/chembl36_small_mask_mlm_lr_sweep/modernmolbert_best_span/final_model/tokenizer_config.json",
    "runs/chembl36_small_mask_mlm_lr_sweep/modernmolbert_best_span/final_model/ape_tokenizer/tokenizer_config.json",
    "runs/chembl36_small_mask_mlm_lr_sweep/modernmolbert_best_base/final_model/tokenizer_config.json",
    "runs/chembl36_small_mask_mlm_lr_sweep/modernmolbert_best_base/final_model/ape_tokenizer/tokenizer_config.json",
    "runs/chembl36_small_mask_mlm_lr_sweep/modernmolbert_best_hetero_span/tokenizer_config.json",
    "runs/chembl36_small_mask_mlm_lr_sweep/modernmolbert_best_hetero_span/ape_tokenizer/tokenizer_config.json",
    "tmp-hf-model/tokenizer_config.json",
    "tmp-hf-model/ape_tokenizer/tokenizer_config.json",
    "tmp-hf-tokenizer/tokenizer_config.json",
]

for path_str in PATHS:
    path = Path(path_str)

    if not path.exists():
        print(f"missing: {path}")

        continue

    data = json.loads(path.read_text())

    old = data.get("model_max_length")

    data["model_max_length"] = 128

    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

    print(f"patched: {path} model_max_length {old} -> 128")
